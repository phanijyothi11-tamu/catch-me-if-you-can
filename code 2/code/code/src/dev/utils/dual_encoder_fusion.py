# dual_encoder_fusion_train.py

import os
import json
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, accuracy_score
from tscbench.data.load.absa import AbsaDataset, AbsaModelProcessor, AbsaDataCollator

# Fix seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
set_seed()

# Paths
DATA_DIR = "data/MAD_TSC/original"
LANGUAGES = ["de", "en", "es", "fr", "it", "nl", "pt", "ro"]
SAVE_DIR = "experiments/dual_fusion_run_outputs"
os.makedirs(SAVE_DIR, exist_ok=True)

# Model Names
MONO_MODELS = {
    "en": "roberta-base",
    "es": "bertin-project/bertin-roberta-base-spanish",
    "de": "bert-base-german-cased",
    "it": "dbmdz/bert-base-italian-uncased",
    "fr": "camembert-base",
    "pt": "neuralmind/bert-base-portuguese-cased",
    "nl": "pdelobelle/robbert-v2-dutch-base",
    "ro": "dumitrescustefan/bert-base-romanian-cased-v1"
}
MULTI_MODEL = "xlm-roberta-base"

# Dual Encoder Fusion Model
class DualEncoderFusion(nn.Module):
    def __init__(self, mono_model, multi_model, hidden_dim=768, num_labels=3):
        super().__init__()
        self.mono_model = mono_model
        self.multi_model = multi_model
        self.fusion_layer = nn.Linear(hidden_dim * 2, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_labels)

    def forward(self, mono_input, multi_input, mono_mask, multi_mask):
        mono_out = self.mono_model(mono_input, attention_mask=mono_mask).last_hidden_state[:, 0, :]
        multi_out = self.multi_model(multi_input, attention_mask=multi_mask).last_hidden_state[:, 0, :]
        fused = torch.cat([mono_out, multi_out], dim=-1)
        fused = self.fusion_layer(fused)
        logits = self.classifier(fused)
        return logits

# Training Loop
def train_one_language(lang, epochs=15):
    print(f"\n====== Training for {lang.upper()} ======")
    save_path = os.path.join(SAVE_DIR, f"{lang}_fusion")
    os.makedirs(save_path, exist_ok=True)

    mono_tokenizer = AutoTokenizer.from_pretrained(MONO_MODELS[lang])
    multi_tokenizer = AutoTokenizer.from_pretrained(MULTI_MODEL)

    mono_model = AutoModel.from_pretrained(MONO_MODELS[lang])
    multi_model = AutoModel.from_pretrained(MULTI_MODEL)

    model = DualEncoderFusion(mono_model, multi_model)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    processor = AbsaModelProcessor(
        prompt_template=f"{mono_tokenizer.sep_token} <entity>",
        tokenizer=mono_tokenizer,
        replace_by_main_mention=False,
        replace_by_special_token=None,
        sentiment_mapping={2: 0, 4: 1, 6: 2},
    )

    # Load train/validation
    train_examples = load_jsonl(os.path.join(DATA_DIR, lang, "train.jsonl"))
    val_examples = load_jsonl(os.path.join(DATA_DIR, lang, "validation.jsonl"))

    train_dataset = prepare_dataset(train_examples, processor)
    val_dataset = prepare_dataset(val_examples, processor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Loss (with class weights)
    class_weights = compute_class_weights(train_dataset)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))

    optimizer = optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.7)

    best_f1 = -1

    for epoch in range(1, epochs + 1):
        model.train()
        for batch in tqdm(train_loader, desc=f"Epoch {epoch} - Training"):
            optimizer.zero_grad()
            outputs = model(batch['mono_input'].to(device), batch['multi_input'].to(device),
                             batch['mono_mask'].to(device), batch['multi_mask'].to(device))
            loss = criterion(outputs, batch['labels'].to(device))
            loss.backward()
            optimizer.step()

        scheduler.step()
        f1, acc = evaluate(model, val_loader, device)

        print(f"[Epoch {epoch}] Validation Macro-F1: {f1:.4f} | Accuracy: {acc:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), os.path.join(save_path, "best_model.pt"))
            print(f"\t[Best model updated]")

# Utilities
def load_jsonl(path):
    examples = []
    with open(path, "r") as f:
        for line in f:
            examples.append(json.loads(line.strip()))
    return examples

def prepare_dataset(examples, processor):
    all_tokens, all_mentions_tokens, all_mentions_pos, all_sentiments, all_params = [], [], [], [], []
    for ex in examples:
        sent = ex["sentence_normalized"]
        target = ex["targets"][0]
        tokens, mention_tokens, mentions_positions, label, param = processor.process_entry(
            sent, [(target["from"], target["to"])], target["mention"], target["polarity"], [target["mention"]]
        )
        all_tokens.append(tokens[0])
        all_mentions_tokens.append(mention_tokens)
        all_mentions_pos.append(mentions_positions)
        all_sentiments.append(label)
        all_params.append(param if param is not None else -1)
    return AbsaDataset(all_tokens, all_mentions_tokens, all_mentions_pos, all_sentiments, all_params)

def collate_fn(batch):
    mono_input = torch.stack([torch.tensor(b[0]) for b in batch])
    multi_input = torch.stack([torch.tensor(b[1]) for b in batch])
    x_select = torch.stack([torch.tensor(b[2]) for b in batch])
    classifying_locations = torch.stack([torch.tensor(b[3]) for b in batch])
    counts = torch.tensor([b[4] for b in batch])
    sentiments = torch.tensor([b[5] for b in batch])
    params = torch.tensor([b[6] for b in batch])
    return mono_input, multi_input, x_select, classifying_locations, counts, sentiments, params


def compute_class_weights(dataset):
    labels = []
    for batch in torch.utils.data.DataLoader(dataset, batch_size=32, collate_fn=lambda x: x):
        for item in batch:
            labels.append(item[-2])  # sentiments is second last element
    counts = np.bincount(labels, minlength=3)
    weights = 1.0 / (counts + 1e-6)
    weights = torch.FloatTensor(weights)
    return weights

def evaluate(model, loader, device):
    model.eval()
    preds, labels = [], []
    with torch.no_grad():
        for batch in loader:
            outputs = model(batch['mono_input'].to(device), batch['multi_input'].to(device),
                             batch['mono_mask'].to(device), batch['multi_mask'].to(device))
            pred_labels = outputs.argmax(dim=1)
            preds.extend(pred_labels.cpu().numpy())
            labels.extend(batch['labels'].cpu().numpy())
    f1 = f1_score(labels, preds, average="macro")
    acc = accuracy_score(labels, preds)
    return f1, acc

# Main
if __name__ == "__main__":
    for lang in LANGUAGES:
        train_one_language(lang)
