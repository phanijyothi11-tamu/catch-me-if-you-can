# eval_test.py

import torch
import os
import sys
import json
from tqdm import tqdm
import pandas as pd

sys.path.append("./src")

from tscbench.finetuning.plightning.plfinetuneabsa import PlFineTuneAbsaModel
from tscbench.finetuning.absa.constants import MODE_MASK
from tscbench.data.load.absa import AbsaModelProcessor, AbsaDataset, AbsaDataCollator
from sklearn.metrics import f1_score, accuracy_score

# =============================
# ==== SETUP VARIABLES ========
# =============================

# Define the 8 experiments
experiments = [
    {
        "model_dir": "experiments/run_outputs/ML_original_de_de_spc",
        "run_name": "ML_spc_original_de_de_exp",
        "test_file": "data/MAD_TSC/original/de/test.jsonl",
        "ckpt_path": "experiments/run_outputs/ML_original_de_de_spc/validation_loss_sg/amazing_easley#validation_loss_sg=0.7538.ckpt",
    },
    {
        "model_dir": "experiments/run_outputs/ML_original_en_en_spc",
        "run_name": "ML_spc_original_en_en_exp",
        "test_file": "data/MAD_TSC/original/en/test.jsonl",
        "ckpt_path": "experiments/run_outputs/ML_original_en_en_spc/validation_loss_sg/fervent_hugle#validation_loss_sg=0.6749.ckpt",
    },
    {
        "model_dir": "experiments/run_outputs/ML_original_es_es_spc",
        "run_name": "ML_spc_original_es_es_exp",
        "test_file": "data/MAD_TSC/original/es/test.jsonl",
        "ckpt_path": "experiments/run_outputs/ML_original_es_es_spc/validation_loss_sg/fervent_hugle#validation_loss_sg=0.6935.ckpt",
    },
    {
        "model_dir": "experiments/run_outputs/ML_original_fr_fr_spc",
        "run_name": "ML_spc_original_fr_fr_exp",
        "test_file": "data/MAD_TSC/original/fr/test.jsonl",
        "ckpt_path": "experiments/run_outputs/ML_original_fr_fr_spc/validation_loss_sg/amazing_fermat#validation_loss_sg=0.7159.ckpt",
    },
    {
        "model_dir": "experiments/run_outputs/ML_original_it_it_spc",
        "run_name": "ML_spc_original_it_it_exp",
        "test_file": "data/MAD_TSC/original/it/test.jsonl",
        "ckpt_path": "experiments/run_outputs/ML_original_it_it_spc/validation_loss_sg/optimistic_greider#validation_loss_sg=0.7615.ckpt",
    },
    {
        "model_dir": "experiments/run_outputs/ML_original_nl_nl_spc",
        "run_name": "ML_spc_original_nl_nl_exp",
        "test_file": "data/MAD_TSC/original/nl/test.jsonl",
        "ckpt_path": "experiments/run_outputs/ML_original_nl_nl_spc/validation_loss_sg/fervent_hugle#validation_loss_sg=0.7264.ckpt",
    },
    {
        "model_dir": "experiments/run_outputs/ML_original_pt_pt_spc",
        "run_name": "ML_spc_original_pt_pt_exp",
        "test_file": "data/MAD_TSC/original/pt/test.jsonl",
        "ckpt_path": "experiments/run_outputs/ML_original_pt_pt_spc/validation_loss_sg/amazing_fermat#validation_loss_sg=0.7080.ckpt",
    },
    {
        "model_dir": "experiments/run_outputs/ML_original_ro_ro_spc",
        "run_name": "ML_spc_original_ro_ro_exp",
        "test_file": "data/MAD_TSC/original/ro/test.jsonl",
        "ckpt_path": "experiments/run_outputs/ML_original_ro_ro_spc/validation_loss_sg/friendly_meitner#validation_loss_sg=0.7328.ckpt",
    },
]

# Directory to save final collated results
final_results_dir = "experiments/ML_eval_collated_results"
os.makedirs(final_results_dir, exist_ok=True)

# To store all results
all_experiment_results = []

# =============================
# ==== LOAD MODEL ============
# =============================

for exp in experiments:
    ckpt_path = exp["ckpt_path"]
    model_dir = exp["model_dir"]
    run_name = exp["run_name"]
    test_file = exp["test_file"]

    print("=" * 80)
    print(f"[INFO] Evaluating: {run_name}")
    print("=" * 80)

    # ==== LOAD MODEL ======
    print("[INFO] Loading model from checkpoint...")
    pl_model = PlFineTuneAbsaModel.load_from_checkpoint(ckpt_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pl_model = pl_model.to(device)
    pl_model = pl_model.eval()

    # ==== LOAD TEST DATA ====
    print("[INFO] Loading test dataset manually...")
    processor = pl_model.model.processor
    processor.set_return_tensors(True)

    examples = []
    with open(test_file, "r") as f:
        for line in f:
            examples.append(json.loads(line.strip()))

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

        all_tokens.append(tokens[0])
        all_mentions_tokens.append(mention_tokens)
        all_mentions_pos.append(mentions_positions)
        all_sentiments.append(label)
        all_params.append(param if param is not None else -1)

    validation_dataset = AbsaDataset(
        all_tokens,
        all_mentions_tokens,
        all_mentions_pos,
        all_sentiments,
        all_params,
    )

    data_collator = AbsaDataCollator(
        mode_mask=MODE_MASK[pl_model.model_config["absa_model"]],
        tokenizer_mask_id=pl_model.tokenizer.mask_token_id,
        tokenizer_padding_id=pl_model.tokenizer.pad_token_id,
        tokenizer = pl_model.tokenizer,
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

    # ==== RUN EVALUATION =====
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

    # ==== COMPUTE METRICS =====
    print("[INFO] Computing metrics...")
    preds_tensor = torch.FloatTensor(all_preds)
    labels_tensor = torch.LongTensor(all_labels)

    preds_labels = torch.argmax(preds_tensor, dim=-1)
    true_labels = labels_tensor

    f1_macro = f1_score(true_labels.numpy(), preds_labels.numpy(), average="macro")
    acc = accuracy_score(true_labels.numpy(), preds_labels.numpy())

    print(f"[RESULT] Test Macro F1 Score: {f1_macro:.4f}")
    print(f"[RESULT] Test Accuracy: {acc:.4f}")

    results = {
        "run_name": run_name,
        "test_f1_macro": f1_macro,
        "test_accuracy": acc,
    }
    all_experiment_results.append(results)

    # Save individual results
    save_path = os.path.join(final_results_dir, f"{run_name}_test_eval_results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"[INFO] Results saved at {save_path}")

# ==== After all experiments: Pretty Table ====

df = pd.DataFrame(all_experiment_results)
print("\n=== Final Evaluation Summary ===\n")
print(df.to_string(index=False))

# Save collated results
collated_path = os.path.join(final_results_dir, "collated_test_eval_results.csv")
df.to_csv(collated_path, index=False)
print(f"\n[INFO] Collated results saved at {collated_path}")

