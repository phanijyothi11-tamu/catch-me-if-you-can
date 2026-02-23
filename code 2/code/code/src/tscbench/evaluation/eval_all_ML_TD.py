# eval_test.py (Extended with plotting and organization)

import torch
import os
import sys
import json
import shutil
from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix

sys.path.append("./src")

from tscbench.finetuning.plightning.plfinetuneabsa import PlFineTuneAbsaModel
from tscbench.finetuning.absa.constants import MODE_MASK
from tscbench.data.load.absa import AbsaModelProcessor, AbsaDataset, AbsaDataCollator

# =============================
# ==== SETUP VARIABLES ========
# =============================

# Define the 8 experiments
experiments = [
    {
        "model_dir": "experiments/run_outputs/TD_ML_german_bert_run1/german_bert_run1_exp",
        "run_name": "TD_ML_german_bert_run1_exp",
        "test_file": "data/MAD_TSC/original/de/test.jsonl",
        "ckpt_path": "experiments/run_outputs/TD_ML_german_bert_run1/german_bert_run1_exp/validation_loss_sg/relaxed_boyd#validation_loss_sg=0.7784.ckpt",
    },
    {
        "model_dir": "experiments/run_outputs/TD_ML_en_run1/en_run1_exp",
        "run_name": "TD_ML_en_run1_exp",
        "test_file": "data/MAD_TSC/original/en/test.jsonl",
        "ckpt_path": "experiments/run_outputs/TD_ML_en_run1/en_run1_exp/validation_loss_sg/bold_clarke#validation_loss_sg=0.6807.ckpt",
    },
    {
        "model_dir": "experiments/run_outputs/spanish_camembert_run1/spanish_camembert_run1_exp",
        "run_name": "spanish_camembert_run1_exp",
        "test_file": "data/MAD_TSC/original/es/test.jsonl",
        "ckpt_path": "experiments/run_outputs/spanish_camembert_run1/spanish_camembert_run1_exp/validation_loss_sg/unruffled_jones#validation_loss_sg=0.6887.ckpt",
    },
    {
        "model_dir": "experiments/run_outputs/TD_ML_fr_run1/french_camembert_run1_exp",
        "run_name": "TD_ML_fr_run1_exp",
        "test_file": "data/MAD_TSC/original/fr/test.jsonl",
        "ckpt_path": "experiments/run_outputs/TD_ML_fr_run1/french_camembert_run1_exp/validation_loss_sg/inspiring_leavitt#validation_loss_sg=0.6926.ckpt",
    },
    {
        "model_dir": "experiments/run_outputs/TD_ML_italian_bert_run1/italian_bert_run1_exp",
        "run_name": "TD_ML_italian_bert_run1_exp",
        "test_file": "data/MAD_TSC/original/it/test.jsonl",
        "ckpt_path": "experiments/run_outputs/TD_ML_italian_bert_run1/italian_bert_run1_exp/validation_loss_sg/unruffled_jones#validation_loss_sg=0.7234.ckpt",
    },
    {
        "model_dir": "experiments/run_outputs/TD_ML_nl_run1/dutch_bert_run1_exp",
        "run_name": "TD_ML_nl_run1_exp",
        "test_file": "data/MAD_TSC/original/nl/test.jsonl",
        "ckpt_path": "experiments/run_outputs/TD_ML_nl_run1/dutch_bert_run1_exp/validation_loss_sg/inspiring_leavitt#validation_loss_sg=0.7401.ckpt",
    },
    {
        "model_dir": "experiments/run_outputs/TD_ML_portuguese_bert_run1/portuguese_bert_run1_exp",
        "run_name": "TD_ML_portuguese_bert_run1_exp",
        "test_file": "data/MAD_TSC/original/pt/test.jsonl",
        "ckpt_path": "experiments/run_outputs/TD_ML_portuguese_bert_run1/portuguese_bert_run1_exp/validation_loss_sg/determined_mcclintock#validation_loss_sg=0.7069.ckpt",
    },
    {
        "model_dir": "experiments/run_outputs/TD_ML_roman_bert_run1/roman_bert_run1_exp",
        "run_name": "TD_ML_roman_bert_run1_exp",
        "test_file": "data/MAD_TSC/original/ro/test.jsonl",
        "ckpt_path": "experiments/run_outputs/TD_ML_roman_bert_run1/roman_bert_run1_exp/validation_loss_sg/heuristic_hermann#validation_loss_sg=0.7047.ckpt",
    },
]


final_results_dir = "experiments/ML_TD_eval_collated_results"
plots_dir = os.path.join(final_results_dir, "plots")
os.makedirs(final_results_dir, exist_ok=True)
os.makedirs(plots_dir, exist_ok=True)

# Results storage
all_experiment_results = []
true_vs_pred = {}  # Save true and predicted labels for confusion matrices

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

    # Load model
    pl_model = PlFineTuneAbsaModel.load_from_checkpoint(ckpt_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pl_model = pl_model.to(device).eval()

    # Load dataset
    processor = pl_model.model.processor
    processor.set_return_tensors(True)

    examples = []
    with open(test_file, "r") as f:
        for line in f:
            examples.append(json.loads(line.strip()))

    all_tokens, all_mentions_tokens, all_mentions_pos, all_sentiments, all_params = [], [], [], [], []

    for example in tqdm(examples, desc="Tokenizing examples"):
        sentence = example["sentence_normalized"]
        mention_info = example["targets"][0]
        mentions_pos = [(mention_info["from"], mention_info["to"])]
        main_mention = mention_info["mention"]
        sentiment = mention_info["polarity"]
        all_mentions = [mention_info["mention"]]

        tokens, mention_tokens, mentions_positions, label, param = processor.process_entry(
            sentence, mentions_pos, main_mention, sentiment, all_mentions,
        )

        all_tokens.append(tokens[0])
        all_mentions_tokens.append(mention_tokens)
        all_mentions_pos.append(mentions_positions)
        all_sentiments.append(label)
        all_params.append(param if param is not None else -1)

    test_dataset = AbsaDataset(all_tokens, all_mentions_tokens, all_mentions_pos, all_sentiments, all_params)

    data_collator = AbsaDataCollator(
        mode_mask=MODE_MASK[pl_model.model_config["absa_model"]],
        tokenizer_mask_id=pl_model.tokenizer.mask_token_id,
        tokenizer_padding_id=pl_model.tokenizer.pad_token_id,
        tokenizer=pl_model.tokenizer,
        force_gpu=torch.cuda.is_available()
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=pl_model.optimizer_config["optimizer"]["batch_size_dataloader"] * 2,
        shuffle=False,
        num_workers=0,
        collate_fn=data_collator,
        drop_last=False,
    )

    # Run evaluation
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch_tokens, attention_mask, x_select, classifying_locations, counts, sentiments, params in tqdm(test_loader):
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

    preds_tensor = torch.FloatTensor(all_preds)
    labels_tensor = torch.LongTensor(all_labels)

    preds_labels = torch.argmax(preds_tensor, dim=-1)
    true_labels = labels_tensor

    f1_macro = f1_score(true_labels.numpy(), preds_labels.numpy(), average="macro")
    acc = accuracy_score(true_labels.numpy(), preds_labels.numpy())

    print(f"[RESULT] Test Macro F1 Score: {f1_macro:.4f}")
    print(f"[RESULT] Test Accuracy: {acc:.4f}")

    lang = run_name.split("_")[3]  # Assuming run_name like ML_spc_original_en_en_exp

    results = {
        "language": lang,
        "run_name": run_name,
        "test_f1_macro": f1_macro,
        "test_accuracy": acc,
    }
    all_experiment_results.append(results)
    true_vs_pred[lang] = (true_labels.numpy(), preds_labels.numpy())

    save_path = os.path.join(final_results_dir, f"{run_name}_test_eval_results.json")
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

# Collate

results_df = pd.DataFrame(all_experiment_results)
results_df.to_csv(os.path.join(final_results_dir, "collated_test_eval_results.csv"), index=False)
print("\n[INFO] Collated results saved.")

# =============================
# ==== PLOTTING SECTION =======
# =============================

# Bar Plot: F1 Macro per Language
plt.figure(figsize=(10, 6))
plt.bar(results_df["language"], results_df["test_f1_macro"])
plt.xlabel("Language")
plt.ylabel("Test Macro F1 Score")
plt.title("Macro F1 Score per Language")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "bar_f1_per_language.png"))
plt.close()

# Bar Plot: Accuracy per Language
plt.figure(figsize=(10, 6))
plt.bar(results_df["language"], results_df["test_accuracy"])
plt.xlabel("Language")
plt.ylabel("Test Accuracy")
plt.title("Accuracy per Language")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "bar_accuracy_per_language.png"))
plt.close()

# Scatter Plot: F1 vs Accuracy
plt.figure(figsize=(8, 6))
plt.scatter(results_df["test_accuracy"], results_df["test_f1_macro"])
for i, lang in enumerate(results_df["language"]):
    plt.text(results_df["test_accuracy"][i], results_df["test_f1_macro"][i], lang)
plt.xlabel("Test Accuracy")
plt.ylabel("Macro F1 Score")
plt.title("F1 Score vs Accuracy")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "scatter_f1_vs_accuracy.png"))
plt.close()

# Error Rate Plot
error_rate = 1 - results_df["test_accuracy"]
plt.figure(figsize=(10, 6))
plt.bar(results_df["language"], error_rate)
plt.xlabel("Language")
plt.ylabel("Error Rate (1 - Accuracy)")
plt.title("Error Rate per Language")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "bar_error_rate_per_language.png"))
plt.close()

# Heatmap of F1 and Accuracy
heatmap_data = results_df.set_index("language")[["test_f1_macro", "test_accuracy"]]
plt.figure(figsize=(8, 6))
sns.heatmap(heatmap_data, annot=True, cmap="Blues", fmt=".2f")
plt.title("Heatmap of Language Performance (F1 and Accuracy)")
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "heatmap_language_performance.png"))
plt.close()

# Confusion Matrices for English and French
for lang in ["en", "fr"]:
    if lang in true_vs_pred:
        y_true, y_pred = true_vs_pred[lang]
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title(f"Confusion Matrix for {lang.upper()}")
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"confusion_matrix_{lang}.png"))
        plt.close()

# Grouping: Language Families (Bonus)
language_family = {
    "en": "Germanic", "de": "Germanic", "nl": "Germanic",
    "fr": "Romance", "es": "Romance", "it": "Romance", "pt": "Romance", "ro": "Romance"
}

results_df["family"] = results_df["language"].map(language_family)
family_avg = results_df.groupby("family")[["test_f1_macro", "test_accuracy"]].mean()

plt.figure(figsize=(8, 6))
family_avg.plot(kind="bar")
plt.title("Average Performance by Language Family")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(plots_dir, "family_performance_comparison.png"))
plt.close()

print("\n[INFO] All plots saved!")


# ==== After all experiments: Pretty Table ====

df = pd.DataFrame(all_experiment_results)
print("\n=== Final Evaluation Summary ===\n")
print(df.to_string(index=False))