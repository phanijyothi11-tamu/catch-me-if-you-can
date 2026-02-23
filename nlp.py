import os
import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torch.optim import AdamW


from torch import nn
from torch.utils.data import DataLoader
from transformers import RobertaTokenizer, RobertaModel, get_scheduler
from sklearn.metrics import classification_report, f1_score
import random
import numpy as np
from tqdm import tqdm
import json

# Hyperparameters
SEEDS = [42, 302, 668, 745, 343]
LR = 2e-5
BATCH_SIZE = 32
WEIGHT_DECAY = 0.01
WARMUP_FRAC = 0.06
MAX_EPOCHS = 40
ADAM_EPS = 1e-6
BETA1 = 0.9
BETA2 = 0.98
EARLY_PATIENCE = 5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Gradient Reversal Layer
class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return -grad_output

# DANN with SPC-style input
class DANN(nn.Module):
    def __init__(self):
        super(DANN, self).__init__()
        self.roberta = RobertaModel.from_pretrained('roberta-base')
        self.sentiment_fc = nn.Linear(self.roberta.config.hidden_size, 3)
        self.domain_fc = nn.Linear(self.roberta.config.hidden_size, 3)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.roberta(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)
        pooled_output = outputs.pooler_output
        sentiment_output = self.sentiment_fc(pooled_output)
        reversed_features = GradientReversalLayer.apply(pooled_output)
        domain_output = self.domain_fc(reversed_features)
        return sentiment_output, domain_output

# Load JSONL data
def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]

# Paths
MAD = r"C:\Users\kurad\OneDrive\Desktop\NLP PROJECT\MAD_TSC\MAD_TSC\original\en"
NEWS = r"C:\Users\kurad\OneDrive\Desktop\NLP PROJECT\NewsMTSC-dataset"

mad_train = load_jsonl(f"{MAD}/train.jsonl")
mad_val = load_jsonl(f"{MAD}/validation.jsonl")
news_train = load_jsonl(f"{NEWS}/train.jsonl")
news_mt = load_jsonl(f"{NEWS}/devtest_mt.jsonl")
news_rw = load_jsonl(f"{NEWS}/devtest_rw.jsonl")

# Dataset
class TextDataset(torch.utils.data.Dataset):
    def __init__(self, data, tokenizer, max_length=512):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        sentence = item['sentence_normalized']

        if not item['targets'] or 'term' not in item['targets'][0]:
            target = "[UNK]"
            sentiment = 1
        else:
            target = item['targets'][0]['term']
            sentiment = item['targets'][0]['polarity']

        domain = 0 if 'mad' in item['primary_gid'].lower() else 1

        encoding = self.tokenizer(
            sentence,
            target,
            padding='max_length',
            truncation='longest_first',
            max_length=self.max_length,
            return_tensors='pt'
        )

        out = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'y_sent': torch.tensor(sentiment),
            'y_dom': torch.tensor(domain)
        }

        # Include token_type_ids if they exist
        if 'token_type_ids' in encoding:
            out['token_type_ids'] = encoding['token_type_ids'].squeeze(0)
        else:
            # Dummy tensor of zeros if token_type_ids are missing
            out['token_type_ids'] = torch.zeros_like(out['attention_mask'])

        return out



# DataLoaders
train_dataset = TextDataset(mad_train, tokenizer)
val_dataset = TextDataset(mad_val, tokenizer)
test_dataset = TextDataset(news_mt + news_rw, tokenizer)

train_dl = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_dl = DataLoader(test_dataset, batch_size=BATCH_SIZE)

# Loss function
ce = nn.CrossEntropyLoss()

# Epoch run function
def run_epoch(model, dl, optim=None, sched=None):
    train = optim is not None
    model.train() if train else model.eval()
    total_loss, preds, gold = 0, [], []

    for b in tqdm(dl, leave=False):
        ids = b["input_ids"].to(DEVICE)
        mask = b["attention_mask"].to(DEVICE)
        types = b["token_type_ids"].to(DEVICE)
        ys = b["y_sent"].to(DEVICE)
        yd = b["y_dom"].to(DEVICE)

        if train: optim.zero_grad()
        out_s, out_d = model(ids, mask, types)
        loss = ce(out_s, ys) + ce(out_d, yd)
        if train:
            loss.backward()
            optim.step()
            sched.step()

        total_loss += loss.item()
        preds += out_s.argmax(1).detach().cpu().tolist()
        gold += ys.detach().cpu().tolist()

    macro_f1 = f1_score(gold, preds, average='macro')
    per_class_report = classification_report(gold, preds, output_dict=True, target_names=["neg", "neu", "pos"])
    per_class_f1 = {cls: per_class_report[cls]["f1-score"] for cls in ["neg", "neu", "pos"]}

    return total_loss / len(dl), macro_f1, per_class_f1

# Training Loop
for seed in SEEDS:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    model = DANN().to(DEVICE)
    optim = AdamW(
        model.parameters(), lr=LR, eps=ADAM_EPS,
        betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY
    )
    total_steps = MAX_EPOCHS * len(train_dl)
    warm_steps = int(WARMUP_FRAC * total_steps)
    sched = torch.optim.lr_scheduler.SequentialLR(
        optim,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(optim, start_factor=0.1, total_iters=warm_steps),
            torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=total_steps - warm_steps)
        ],
        milestones=[warm_steps]
    )

    best_f1 = 0
    bad_epochs = 0

    for ep in range(1, MAX_EPOCHS + 1):
        train_loss, train_f1, _ = run_epoch(model, train_dl, optim, sched)
        val_loss, val_f1, val_per_class_f1 = run_epoch(model, val_dl)

        print(f"Epoch {ep}/{MAX_EPOCHS} - "
              f"Train Loss: {train_loss:.3f} - Train Macro-F1: {train_f1:.3f} - "
              f"Val Loss: {val_loss:.3f} - Val Macro-F1: {val_f1:.3f}")
        print(f"Per-Class F1 (Validation): {val_per_class_f1}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            bad_epochs = 0
            torch.save(model.state_dict(), f"best_seed{seed}.pt")
        else:
            bad_epochs += 1
            if bad_epochs == EARLY_PATIENCE:
                break

    print(f"Best validation Macro-F1 for seed {seed}: {best_f1:.3f}")

    model.load_state_dict(torch.load(f"best_seed{seed}.pt"))
    test_loss, test_f1, test_per_class_f1 = run_epoch(model, test_dl)
    print(f"Test Loss: {test_loss:.3f} - Test Macro-F1: {test_f1:.3f}")
    print(f"Per-Class F1 (Test): {test_per_class_f1}")
