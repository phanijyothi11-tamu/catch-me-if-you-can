import os
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torch.optim import AdamW
from tqdm import tqdm
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics import f1_score, classification_report
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------- 2. Hyper-parameters --------------------------------
LR              = 2e-5
BATCH_SIZE      = 32
VAL_BATCH       = 64                           # double train batch
WEIGHT_DECAY    = 0.01
WARMUP_FRAC     = 0.06
MAX_EPOCHS      = 40
ADAM_EPS        = 1e-6
BETA1, BETA2    = 0.9, 0.98
SEEDS           = [42, 302, 668, 745, 343]     # run each seed
EARLY_PATIENCE  = 3
MAX_LEN         = 128                          # RoBERTa max tokens (same as BERT)

# ------------------- 3. Paths -------------------------------------------
MAD = "/scratch/user/phanijyothi11/phani11/MAD_TSC/MAD_TSC/original/en"
NEWS = "/scratch/user/phanijyothi11/phani11/NewsMTSC-dataset"


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(l) for l in f]

# Load NewsMTSC and MAD-TSC datasets
news_mt   = load_jsonl(f"{NEWS}/devtest_mt.jsonl")
news_rw   = load_jsonl(f"{NEWS}/devtest_rw.jsonl")
mad_test  = load_jsonl(f"{MAD}/test.jsonl")   # Load MAD test dataset

# ------------------- 4. Dataset & loaders ------------------------------
tok = RobertaTokenizer.from_pretrained("roberta-base")  # Changed to RobertaTokenizer
label_map = {2:0, 4:1, 6:2}                    # neg, neu, pos

# Define Dataset for MAD-TSC (test) and NewsMTSC (train and validation)
class SentData(Dataset):
    def __init__(self, records, dom_id):
        self.items = []
        for r in records:
            for t in r["targets"]:
                self.items.append(
                    (r["sentence_normalized"], t["polarity"], dom_id)
                )
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        sent, pol, dom = self.items[i]
        enc = tok(
            sent, max_length=MAX_LEN, truncation=True,
            padding="max_length", return_tensors="pt"
        )
        return {
            "input_ids":      enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "y_sent": torch.tensor(label_map[pol]),
            "y_dom" : torch.tensor(dom),
        }

# Train on both NewsMTSC `mt` and `rw`, validate on `rw`, and test on MAD-TSC `test.jsonl`
mt_ds    = SentData(news_mt, 1)
rw_ds    = SentData(news_rw, 1)
mad_ds   = SentData(mad_test, 0)

# Concat `mt_ds` and `rw_ds` for training on both subsets
train_ds = ConcatDataset([mt_ds, rw_ds])

# DataLoaders
train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl   = DataLoader(rw_ds, batch_size=VAL_BATCH)  # Validate on RW
test_dl  = DataLoader(mad_ds, batch_size=VAL_BATCH)  # Test on MAD

# ------------------- 5. DANN model -------------------------------------
class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lamb): ctx.lamb = lamb; return x.view_as(x)
    @staticmethod
    def backward(ctx, g): return -ctx.lamb * g, None

class DANN(nn.Module):
    def __init__(self, lamb=1.0, n_sent=3, n_dom=2):
        super().__init__()
        self.roberta = RobertaModel.from_pretrained("roberta-base")  # Changed to RobertaModel
        h = self.roberta.config.hidden_size
        self.sent_head = nn.Linear(h, n_sent)
        self.dom_head  = nn.Sequential(
            nn.Linear(h,128), nn.ReLU(), nn.Linear(128, n_dom)
        )
        self.lamb = lamb

    def forward(self, ids, mask):
        pooled = self.roberta(ids, attention_mask=mask).pooler_output  # Using RoBERTa pooler
        sent   = self.sent_head(pooled)
        dom    = self.dom_head(GRL.apply(pooled, self.lamb))
        return sent, dom

# ------------------- 6. Train / eval helpers ---------------------------
ce = nn.CrossEntropyLoss()

def reset_state(model):
    for m in model.modules():
        if hasattr(m, "reset_parameters"): m.reset_parameters()

def run_epoch(model, dl, optim=None, sched=None):
    train = optim is not None
    model.train() if train else model.eval()
    total, preds, gold = 0, [], []
    for b in tqdm(dl, leave=False):
        ids  = b["input_ids"].to(device)
        mask = b["attention_mask"].to(device)
        ys   = b["y_sent"].to(device)
        yd   = b["y_dom"].to(device)

        if train: optim.zero_grad()
        out_s, out_d = model(ids, mask)
        loss = ce(out_s, ys) + ce(out_d, yd)
        if train:
            loss.backward()
            optim.step()
            sched.step()
        total += loss.item()
        preds += out_s.argmax(1).detach().cpu().tolist()
        gold  += ys.detach().cpu().tolist()
    macro = f1_score(gold, preds, average="macro")
    per   = classification_report(
        gold, preds, output_dict=True,
        target_names=["neg","neu","pos"]
    )
    return total/len(dl), macro, per

# ------------------- 7. Multi-seed training ----------------------------
best_overall = 0
for seed in SEEDS:
    random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

    model = DANN().to(device)
    optim = AdamW(
        model.parameters(), lr=LR, eps=ADAM_EPS,
        betas=(BETA1, BETA2), weight_decay=WEIGHT_DECAY
    )

    total_steps  = MAX_EPOCHS * len(train_dl)
    warm_steps   = int(WARMUP_FRAC * total_steps)
    sched = torch.optim.lr_scheduler.SequentialLR(
      optim,
      schedulers=[ 
          torch.optim.lr_scheduler.LinearLR(
              optim, start_factor=0.1, total_iters=warm_steps  # Adjusted start_factor
          ),
          torch.optim.lr_scheduler.CosineAnnealingLR(
              optim, T_max=total_steps - warm_steps
          )
      ],
      milestones=[warm_steps]
  )

    best, bad = 0, 0
    for ep in range(1, MAX_EPOCHS+1):
        run_epoch(model, train_dl, optim, sched)            # train
        _, val_f1, _ = run_epoch(model, val_dl)             # evaluate
        print(f"[seed {seed}] Epoch {ep}/{MAX_EPOCHS}  val Macro-F1 = {val_f1:.3f}")
        if val_f1 > best:
            best = val_f1; bad = 0
            torch.save(model.state_dict(), f"best_seedtdnewstomad{seed}.pt")
        else:
            bad += 1
            if bad == EARLY_PATIENCE: break
    best_overall = max(best_overall, best)
    print(f"? seed {seed} best Macro-F1 = {best:.3f}\n")

print(f"==========================")
print(f"Best validation Macro-F1 across seeds: {best_overall:.3f}")

# ------------------- 8. Load top models & final tests -------------------
# Loop through all seeds' models for evaluation
for seed in SEEDS:
    model.load_state_dict(torch.load(f"best_seedtdnewstomad{seed}.pt"))
    for name, dl in [("MAD-TSC", test_dl)]:
        _, mac, per = run_epoch(model, dl)
        print(f"\n{name} (Seed {seed})  Macro-F1 = {mac:.3f}")
        for cls in ["neg", "neu", "pos"]:
            print(f"  {cls:3}-F1  {per[cls]['f1-score']:.3f}")
