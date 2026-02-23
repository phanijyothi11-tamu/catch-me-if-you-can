import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import spacy
import json
from sklearn.metrics import f1_score
import math
 
# ====== Setup ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
 
# ====== Load Pretrained Model ======
tokenizer = AutoTokenizer.from_pretrained("microsoft/infoxlm-base")
bert_model = AutoModel.from_pretrained("microsoft/infoxlm-base").to(device)
 
# Freeze initially
for param in bert_model.parameters():
    param.requires_grad = False
 
# Unfreeze last few layers
for name, param in bert_model.named_parameters():
    if "encoder.layer.9" in name or "encoder.layer.10" in name or "encoder.layer.11" in name or "pooler" in name:
        param.requires_grad = True
 
nlp = spacy.load("en_core_web_sm")
 
# ====== Preprocessing ======
def preprocess_sample(data, window_size=8, alpha=0.25):
    sentence = data["sentence_normalized"]
    target_info = data["targets"][0]
    target_mention = target_info["mention"]
    target_start = target_info["from"]
    target_end = target_info["to"]
 
    doc = nlp(sentence)
    aligned_entities = []
    tokenized = tokenizer(sentence, return_offsets_mapping=True, return_tensors="pt", truncation=True)
    offsets = tokenized.pop("offset_mapping")[0].tolist()
 
    for ent in doc.ents:
        ent_tokens = [i for i, (start, end) in enumerate(offsets) if start >= ent.start_char and end <= ent.end_char]
        aligned_entities.append({"mention": ent.text, "token_indices": ent_tokens})
 
    if not any(target_mention.lower() in e["mention"].lower() for e in aligned_entities):
        fallback_tokens = [i for i, (start, end) in enumerate(offsets) if start >= target_start and end <= target_end]
        if fallback_tokens:
            aligned_entities.append({"mention": target_mention, "token_indices": fallback_tokens})
 
    encoded = tokenizer(sentence, return_tensors="pt", truncation=True).to(device)
    with torch.no_grad():
        outputs = bert_model(**encoded, output_hidden_states=True)
    H = torch.stack(outputs.hidden_states[-4:], dim=0).mean(dim=0).squeeze(0)
 
    entity_context_vectors = []
    for entity in aligned_entities:
        if entity["token_indices"]:
            center_idx = int(sum(entity["token_indices"]) / len(entity["token_indices"]))
            start = max(center_idx - window_size, 0)
            end = min(center_idx + window_size + 1, H.shape[0])
            context_embeddings = H[start:end, :]
            entity_context_vectors.append((context_embeddings, center_idx))
 
    if not entity_context_vectors:
        return None, None
 
    target_center_idx = next((c for _, c in entity_context_vectors), entity_context_vectors[0][1])
 
    raw_weights = [math.exp(-alpha * abs(center_idx - target_center_idx)) for _, center_idx in entity_context_vectors]
    raw_weights = torch.tensor(raw_weights, device=device)
    importance_weights = torch.softmax(raw_weights, dim=0)
 
    contexts = [ctx for ctx, _ in entity_context_vectors]
 
    return contexts, importance_weights
 
# ====== Model ======
class EntityContextBiLSTM(nn.Module):
    def __init__(self, hidden_size=768, lstm_hidden_size=256, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(hidden_size, lstm_hidden_size, batch_first=True, bidirectional=True)
        self.attention = nn.Linear(lstm_hidden_size * 2, 1)
        self.fc = nn.Linear(lstm_hidden_size * 2, num_classes)
 
    def forward(self, contexts, importance_weights):
        entity_logits = []
        for context in contexts:
            lstm_out, _ = self.lstm(context.unsqueeze(0))
            attn_scores = self.attention(lstm_out).squeeze(-1)
            attn_weights = F.softmax(attn_scores, dim=1)
            pooled = torch.bmm(attn_weights.unsqueeze(1), lstm_out).squeeze(1)
            logit = self.fc(pooled)
            entity_logits.append(logit)
 
        entity_logits = torch.cat(entity_logits, dim=0)
        weighted_logits = (entity_logits * importance_weights.unsqueeze(1)).sum(dim=0, keepdim=True)
        return weighted_logits
 
# ====== Load Data ======
train_data = [json.loads(line) for line in open("train.jsonl")]
val_data = [json.loads(line) for line in open("validation.jsonl")]
test_data = [json.loads(line) for line in open("test.jsonl")]
 
# ====== Train Setup ======
model = EntityContextBiLSTM().to(device)
optimizer = optim.Adam([
    {"params": model.parameters()},
    {"params": bert_model.parameters(), "lr": 5e-6}
], lr=2e-4)
criterion = nn.CrossEntropyLoss()
 
best_val_macro_f1 = 0
patience = 5
patience_counter = 0
save_path = "best_dynamic_context_model.pt"
 
# ====== Correct Label Mapping ======
def map_polarity(polarity):
    if polarity == 2.0:
        return 0
    elif polarity == 4.0:
        return 1
    elif polarity == 6.0:
        return 2
    else:
        raise ValueError(f"Unexpected polarity: {polarity}")
 
# ====== Training Loop ======
epochs = 30
 
for epoch in range(1, epochs + 1):
    model.train()
    total_loss, correct, total = 0, 0, 0
    y_true, y_pred = [], []
 
    for sample in tqdm(train_data, desc=f"Epoch {epoch} - Training"):
        contexts, importance_weights = preprocess_sample(sample)
        if contexts is None:
            continue
 
        polarity = sample["targets"][0]["polarity"]
        label = map_polarity(polarity)
        label_tensor = torch.tensor([label], device=device)
 
        logits = model(contexts, importance_weights)
        loss = criterion(logits, label_tensor)
 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
 
        total_loss += loss.item()
        pred = torch.argmax(logits, dim=1).item()
        y_true.append(label)
        y_pred.append(pred)
 
        correct += (pred == label)
        total += 1
 
    train_accuracy = correct / total
    train_macro_f1 = f1_score(y_true, y_pred, average='macro')
    print(f"[Epoch {epoch}] Train Loss: {total_loss/total:.4f}, Accuracy: {train_accuracy:.4f}, Macro F1: {train_macro_f1:.4f}")
 
    # ====== Validation ======
    model.eval()
    val_true, val_pred = [], []
 
    with torch.no_grad():
        for sample in tqdm(val_data, desc=f"Epoch {epoch} - Validation"):
            contexts, importance_weights = preprocess_sample(sample)
            if contexts is None:
                continue
 
            polarity = sample["targets"][0]["polarity"]
            label = map_polarity(polarity)
 
            logits = model(contexts, importance_weights)
            pred = torch.argmax(logits, dim=1).item()
 
            val_true.append(label)
            val_pred.append(pred)
 
    val_macro_f1 = f1_score(val_true, val_pred, average='macro')
    val_accuracy = sum([p == t for p, t in zip(val_pred, val_true)]) / len(val_true)
    print(f"[Epoch {epoch}] Validation Accuracy: {val_accuracy:.4f}, Macro F1: {val_macro_f1:.4f}")
 
    if val_macro_f1 > best_val_macro_f1:
        best_val_macro_f1 = val_macro_f1
        patience_counter = 0
        torch.save(model.state_dict(), save_path)
        print(f"Best model saved at epoch {epoch}.")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("Early stopping triggered.")
            break
 
# ====== Testing ======
print("Loading best model for testing...")
model.load_state_dict(torch.load(save_path))
model.eval()
 
test_true, test_pred = [], []
 
with torch.no_grad():
    for sample in tqdm(test_data, desc="Testing"):
        contexts, importance_weights = preprocess_sample(sample)
        if contexts is None:
            continue
 
        polarity = sample["targets"][0]["polarity"]
        label = map_polarity(polarity)
 
        logits = model(contexts, importance_weights)
        pred = torch.argmax(logits, dim=1).item()
 
        test_true.append(label)
        test_pred.append(pred)
 
test_accuracy = sum([p == t for p, t in zip(test_pred, test_true)]) / len(test_true)
test_macro_f1 = f1_score(test_true, test_pred, average='macro')
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Macro F1 Score: {test_macro_f1:.4f}")