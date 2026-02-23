import torch
import math
import json
from transformers import AutoTokenizer, AutoModel
import spacy
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence


# Detect GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load models once
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
bert_model = AutoModel.from_pretrained("bert-base-uncased").to(device)
bert_model.eval()
nlp = spacy.load("en_core_web_sm")

def preprocess_sample(data, window_size=8, alpha=0.25):
    sentence = data["sentence_normalized"]
    target_info = data["targets"][0]
    target_mention = target_info["mention"]
    target_start = target_info["from"]
    target_end = target_info["to"]

    # NER
    doc = nlp(sentence)
    aligned_entities = []
    tokenized = tokenizer(sentence, return_offsets_mapping=True, return_tensors="pt", truncation=True)
    offsets = tokenized["offset_mapping"][0].tolist()

    for ent in doc.ents:
        ent_tokens = [
            i for i, (start, end) in enumerate(offsets)
            if start >= ent.start_char and end <= ent.end_char
        ]
        aligned_entities.append({
            "mention": ent.text,
            "start_char": ent.start_char,
            "end_char": ent.end_char,
            "token_indices": ent_tokens,
            "label": ent.label_
        })

    # Fallback if needed
    if not any(target_mention.lower() in ent["mention"].lower() for ent in aligned_entities):
        fallback_tokens = [
            i for i, (start, end) in enumerate(offsets)
            if start >= target_start and end <= target_end
        ]
        if fallback_tokens:
            aligned_entities.append({
                "mention": target_mention,
                "start_char": target_start,
                "end_char": target_end,
                "token_indices": fallback_tokens,
                "label": "TARGET_FALLBACK"
            })

    # BERT hidden states
    encoded = tokenizer(sentence, return_offsets_mapping=True, return_tensors="pt", truncation=True)
    offsets = encoded.pop("offset_mapping")[0].tolist()  # <-- REMOVE offset_mapping before model input

    encoded = {k: v.to(device) for k, v in encoded.items()}  # Move to device

    with torch.no_grad():
        outputs = bert_model(**encoded)
        H = outputs.last_hidden_state[0]  # [seq_len, hidden_size]

    def get_position_weighted_vector(H, center_idx):
        seq_len = H.shape[0]
        start = max(center_idx - window_size, 0)
        end = min(center_idx + window_size + 1, seq_len)

        context_indices = list(range(start, end))
        distances = [abs(i - center_idx) for i in context_indices]
        weights = torch.tensor([torch.exp(torch.tensor(-alpha * d)) for d in distances], dtype=torch.float32, device=device)
        weights = weights / weights.sum()

        context_embeddings = H[context_indices, :]
        weighted_context = torch.sum(context_embeddings * weights.unsqueeze(1), dim=0)
        return weighted_context

    def get_context_embeddings(H, center_idx, window_size=8):
        seq_len = H.shape[0]
        start = max(center_idx - window_size, 0)
        end = min(center_idx + window_size + 1, seq_len)

        context_indices = list(range(start, end))
        context_embeddings = H[context_indices, :]  # Shape: [variable_window_size, hidden_size]
        return context_embeddings


    entity_context_vectors = []

    for entity in aligned_entities:
        if entity["token_indices"]:
            center_idx = int(sum(entity["token_indices"]) / len(entity["token_indices"]))
            # context_vector = get_position_weighted_vector(H, center_idx)
            context_vector = get_context_embeddings(H, center_idx)
            entity_context_vectors.append({
                "mention": entity["mention"],
                "vector": context_vector,
                "label": entity["label"],
                "center_idx": center_idx
            })

    # Proximity weights
    target_entity = next((e for e in entity_context_vectors if target_mention.lower() in e["mention"].lower()), None)
    if target_entity is None:
        raise ValueError(f"Target mention '{target_mention}' not found even after fallback.")

    target_center_idx = target_entity["center_idx"]

    raw_weights = []
    for entity in entity_context_vectors:
        distance = abs(entity["center_idx"] - target_center_idx)
        weight = math.exp(-alpha * distance)
        raw_weights.append(weight)

    raw_weights_tensor = torch.tensor(raw_weights, device=device)
    importance_weights = torch.softmax(raw_weights_tensor, dim=0).tolist()

    for i, entity in enumerate(entity_context_vectors):
        entity["importance_weight"] = importance_weights[i]

    return entity_context_vectors, sentence, data["targets"][0]["polarity"]

# --- Load your train-test split ---
import json
from sklearn.model_selection import train_test_split

with open("train.jsonl", "r") as f:
    all_data = [json.loads(line.strip()) for line in f]
print("################1###################")
train_data, test_data = train_test_split(all_data, test_size=0.2, random_state=42)

# --- Model ---
import torch.nn as nn
import torch.nn.functional as F

class SubsentenceSentimentHead(nn.Module):
    def __init__(self, hidden_size=768, num_classes=3):
        super().__init__()
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, context_vectors):
        context_mean = context_vectors.mean(dim=1)
        logits = self.fc(context_mean)
        return logits

def map_polarity(polarity):
    if polarity == 2.0:
        return 0  # negative
    elif polarity == 4.0:
        return 1  # neutral
    elif polarity == 6.0:
        return 2  # positive
    else:
        raise ValueError(f"Unexpected polarity: {polarity}")

# --- Training ---
import torch.optim as optim

model_subsentiment = SubsentenceSentimentHead().to(device)
optimizer = optim.Adam(model_subsentiment.parameters(), lr=2e-4)
criterion = nn.CrossEntropyLoss()

epochs = 150

for epoch in range(epochs):
    model_subsentiment.train()
    total_loss = 0
    total_samples = 0

    for data in train_data:
        try:
            entity_context_vectors, sentence, polarity = preprocess_sample(data)

            if len(entity_context_vectors) == 0:
                continue

            context_list = [e["vector"] for e in entity_context_vectors]  # [num_entities, hidden]
            padded_contexts = pad_sequence(context_list, batch_first=True)
            subsentiment_logits = model_subsentiment(padded_contexts)

            importance_weights = torch.tensor(
                [e["importance_weight"] for e in entity_context_vectors],
                dtype=torch.float32,
                device=device
            ).unsqueeze(1)

            weighted_logits = subsentiment_logits * importance_weights
            global_logits = weighted_logits.sum(dim=0).unsqueeze(0)

            global_labels = torch.tensor([map_polarity(polarity)], dtype=torch.long, device=device)

            loss = criterion(global_logits, global_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_samples += 1

        except Exception as e:
            print(f"Skipping sample due to error: {e}")

    avg_loss = total_loss / total_samples
    print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")

# --- Testing ---
model_subsentiment.eval()

correct = 0
total = 0

for sample in tqdm(test_data):
    try:
        entity_context_vectors, sentence, true_polarity = preprocess_sample(sample)

        if len(entity_context_vectors) == 0:
            continue

        context_vectors = pad_sequence([e["vector"] for e in entity_context_vectors], batch_first=True)
        subsentiment_logits = model_subsentiment(context_vectors)

        importance_weights = torch.tensor(
            [e["importance_weight"] for e in entity_context_vectors],
            dtype=torch.float32,
            device=device
        ).unsqueeze(1)

        weighted_logits = subsentiment_logits * importance_weights
        global_logits = weighted_logits.sum(dim=0).unsqueeze(0)

        predicted_label = torch.argmax(global_logits, dim=1).item()
        true_label = int(true_polarity) // 2

        if predicted_label == true_label:
            correct += 1

        total += 1

    except Exception as e:
        print(f"Skipping sample during testing due to error: {e}")

accuracy = correct / total
print(f"Test Accuracy: {accuracy:.4f}")
