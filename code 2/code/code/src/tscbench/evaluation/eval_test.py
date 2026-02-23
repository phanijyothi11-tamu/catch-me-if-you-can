# eval_test.py

import torch
import os
import sys
import json
from tqdm import tqdm

sys.path.append("./src")

from tscbench.finetuning.plightning.plfinetuneabsa import PlFineTuneAbsaModel
from tscbench.finetuning.absa.constants import MODE_MASK
from tscbench.data.load.absa import AbsaModelProcessor, AbsaDataset, AbsaDataCollator
from sklearn.metrics import f1_score, accuracy_score

# =============================
# ==== SETUP VARIABLES ========
# =============================

# Path to your trained model checkpoint
ckpt_path = "experiments/run_outputs/it_bert_italian/validation_loss_sg/unruffled_jones#validation_loss_sg=0.7699.ckpt"

# Directory where the final model artifacts are
model_dir = "experiments/run_outputs/it_bert_italian/"

# Name of the run folder (like 'en_roberta_spc_run1_exp')
run_name = "it_bert_italian_run1_exp"

# Path to validation file
validation_file = "data/MAD_TSC/original/it/test.jsonl"

# =============================
# ==== LOAD MODEL ============
# =============================

print("[INFO] Loading model from checkpoint...")
pl_model = PlFineTuneAbsaModel.load_from_checkpoint(ckpt_path)

device = "cuda" if torch.cuda.is_available() else "cpu"
pl_model = pl_model.to(device)
pl_model = pl_model.eval()

# IMPORTANT: DO NOT call pl_model.setup() here

# =============================
# ==== LOAD VALIDATION DATA ===
# =============================

print("[INFO] Loading validation dataset manually...")

# Get the processor from the loaded model
processor = pl_model.model.processor
processor.set_return_tensors(True)

# Load examples
examples = []
with open(validation_file, "r") as f:
    for line in f:
        examples.append(json.loads(line.strip()))

# Tokenize manually
all_tokens = []
all_mentions_tokens = []
all_mentions_pos = []
all_sentiments = []
all_params = []

for example in tqdm(examples, desc="Tokenizing examples"):
    sentence = example["sentence_normalized"]
    mention_info = example["targets"][0]
    mentions_pos = [(mention_info["from"], mention_info["to"])]
    main_mention = mention_info["mention"]
    sentiment = mention_info["polarity"]
    all_mentions = [mention_info["mention"]]

    tokens, mention_tokens, mentions_positions, label, param = processor.process_entry(
        sentence,
        mentions_pos,
        main_mention,
        sentiment,
        all_mentions,
    )

    all_tokens.append(tokens[0])  # [0] because tokens is a list (per template)
    all_mentions_tokens.append(mention_tokens)
    all_mentions_pos.append(mentions_positions)
    all_sentiments.append(label)
    if param is not None:
        all_params.append(param)
    else:
        all_params.append(-1)

validation_dataset = AbsaDataset(
    all_tokens,
    all_mentions_tokens,
    all_mentions_pos,
    all_sentiments,
    all_params,
)

# Create data collator
data_collator = AbsaDataCollator(
    mode_mask=MODE_MASK[pl_model.model_config["absa_model"]],
    tokenizer_mask_id=pl_model.tokenizer.mask_token_id,
    tokenizer_padding_id=pl_model.tokenizer.pad_token_id,
    force_gpu=True if torch.cuda.is_available() else False,
)

val_loader = torch.utils.data.DataLoader(
    validation_dataset,
    batch_size=pl_model.optimizer_config["optimizer"]["batch_size_dataloader"] * 2,
    shuffle=False,
    num_workers=0,
    collate_fn=data_collator,
    drop_last=False,
)

# =============================
# ==== RUN EVALUATION =========
# =============================

print("[INFO] Starting evaluation...")

all_preds = []
all_labels = []

with torch.no_grad():
    for batch_tokens, attention_mask, x_select, classifying_locations, counts, sentiments, params in tqdm(val_loader):
        out = pl_model.forward(
            batch_tokens=batch_tokens,
            attention_mask=attention_mask,
            classifying_locations=classifying_locations,
            counts=counts,
            x_select=x_select,
        )
        preds = out.detach().cpu().numpy()
        labels = sentiments.cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels)

# =============================
# ==== COMPUTE METRICS ========
# =============================

print("[INFO] Computing metrics...")

preds_tensor = torch.FloatTensor(all_preds)
labels_tensor = torch.LongTensor(all_labels)

preds_labels = torch.argmax(preds_tensor, dim=-1)
true_labels = labels_tensor

f1_macro = f1_score(true_labels.numpy(), preds_labels.numpy(), average="macro")
acc = accuracy_score(true_labels.numpy(), preds_labels.numpy())

print(f"[RESULT] Validation Macro F1 Score: {f1_macro:.4f}")
print(f"[RESULT] Validation Accuracy: {acc:.4f}")

# Save results
results = {
    "validation_f1_macro": f1_macro,
    "validation_accuracy": acc,
}

save_path = os.path.join(model_dir, run_name, "validation_eval_results.json")
os.makedirs(os.path.dirname(save_path), exist_ok=True)

with open(save_path, "w") as f:
    json.dump(results, f, indent=4)

print(f"[INFO] Results saved at {save_path}")
