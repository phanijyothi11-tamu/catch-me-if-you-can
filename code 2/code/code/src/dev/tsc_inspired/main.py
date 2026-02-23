import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm
import json
from sklearn.metrics import f1_score  # <<== Added

# ====== Setup ======
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# ====== Load Pretrained Model ======
tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
bert_model = AutoModel.from_pretrained("xlm-roberta-base").to(device)

# Freeze all layers first
for param in bert_model.parameters():
    param.requires_grad = False

# Unfreeze only last 2 layers + pooler
for name, param in bert_model.named_parameters():
    if "encoder.layer.10" in name or "encoder.layer.11" in name or "pooler" in name:
        param.requires_grad = True

# ====== Dataset ======
def load_data(filename):
    with open(filename, "r") as f:
        data = [json.loads(line.strip()) for line in f]
    return data

train_data = load_data("train.jsonl")
val_data = load_data("validation.jsonl")
test_data = load_data("test.jsonl")
print(f"Loaded {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples.")

# ====== Simple Preprocessing ======
def preprocess_sample(sample):
    sentence = sample["sentence_normalized"]
    target = sample["targets"][0]["mention"]

    input_text = sentence + " " + tokenizer.sep_token + " " + target
    encoded = tokenizer(input_text, return_tensors="pt", truncation=True, padding="max_length", max_length=128)
    encoded = {k: v.squeeze(0).to(device) for k, v in encoded.items()}  # Remove batch dimension

    polarity = sample["targets"][0]["polarity"]
    if polarity == 2.0:
        label = 0  # negative
    elif polarity == 4.0:
        label = 1  # neutral
    elif polarity == 6.0:
        label = 2  # positive
    else:
        raise ValueError(f"Unknown polarity {polarity}")

    return encoded, label

# ====== Model ======
class SPCModel(nn.Module):
    def __init__(self, bert_model, hidden_size=768, num_classes=3):
        super().__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        logits = self.classifier(cls_embedding)
        return logits

model = SPCModel(bert_model).to(device)
print(model)

# ====== Optimizer, Loss ======
optimizer = optim.Adam([
    {'params': model.classifier.parameters(), 'lr': 2e-4},
    {'params': model.bert.parameters(), 'lr': 2e-5}
])
criterion = nn.CrossEntropyLoss()

# ====== Training ======
epochs = 20
best_val_accuracy = 0
patience = 3
patience_counter = 0
save_path = "best_spc_model.pt"

for epoch in range(1, epochs + 1):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    train_true = []
    train_pred = []

    for sample in tqdm(train_data, desc=f"Epoch {epoch} - Training"):
        try:
            inputs, label = preprocess_sample(sample)

            logits = model(
                input_ids=inputs["input_ids"].unsqueeze(0),
                attention_mask=inputs["attention_mask"].unsqueeze(0)
            )

            loss = criterion(logits, torch.tensor([label], dtype=torch.long, device=device))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            pred = torch.argmax(logits, dim=1).item()
            if pred == label:
                correct += 1
            total += 1

            train_true.append(label)
            train_pred.append(pred)

        except Exception as e:
            print(f"Skipping sample due to error: {e}")

    train_loss = total_loss / total
    train_accuracy = correct / total
    train_macro_f1 = f1_score(train_true, train_pred, average='macro')
    print(f"[Epoch {epoch}] Train Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, Macro F1: {train_macro_f1:.4f}")

    # ====== Validation ======
    model.eval()
    val_correct = 0
    val_total = 0
    val_true = []
    val_pred = []

    with torch.no_grad():
        for sample in tqdm(val_data, desc=f"Epoch {epoch} - Validation"):
            try:
                inputs, label = preprocess_sample(sample)

                logits = model(
                    input_ids=inputs["input_ids"].unsqueeze(0),
                    attention_mask=inputs["attention_mask"].unsqueeze(0)
                )

                pred = torch.argmax(logits, dim=1).item()
                if pred == label:
                    val_correct += 1
                val_total += 1

                val_true.append(label)
                val_pred.append(pred)

            except Exception as e:
                print(f"Skipping validation sample due to error: {e}")

    val_accuracy = val_correct / val_total
    val_macro_f1 = f1_score(val_true, val_pred, average='macro')
    print(f"[Epoch {epoch}] Validation Accuracy: {val_accuracy:.4f}, Macro F1: {val_macro_f1:.4f}")

    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        patience_counter = 0
        torch.save(model.state_dict(), save_path)
        print(f"Best model saved at epoch {epoch}.")
    else:
        patience_counter += 1
        print(f"No improvement for {patience_counter} epochs.")

        if patience_counter >= patience:
            print("Early stopping triggered.")
            break

# ====== Testing ======
print("Loading best model for testing...")
model.load_state_dict(torch.load(save_path))
model.eval()

test_correct = 0
test_total = 0
test_true = []
test_pred = []

with torch.no_grad():
    for sample in tqdm(test_data, desc="Testing"):
        try:
            inputs, label = preprocess_sample(sample)

            logits = model(
                input_ids=inputs["input_ids"].unsqueeze(0),
                attention_mask=inputs["attention_mask"].unsqueeze(0)
            )

            pred = torch.argmax(logits, dim=1).item()
            if pred == label:
                test_correct += 1
            test_total += 1

            test_true.append(label)
            test_pred.append(pred)

        except Exception as e:
            print(f"Skipping test sample due to error: {e}")

test_accuracy = test_correct / test_total
test_macro_f1 = f1_score(test_true, test_pred, average='macro')
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test Macro F1 Score: {test_macro_f1:.4f}")
