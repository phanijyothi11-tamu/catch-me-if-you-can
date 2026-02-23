# eval_test_monolingual.py

import torch
import os
import sys
import json
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, ConfusionMatrixDisplay

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
        "model_dir": "experiments/run_outputs/german_bert_run1",
        "run_name": "german_bert_run1_exp",
        "validation_file": "data/MAD_TSC/original/de/test.jsonl",
        "ckpt_path": "experiments/run_outputs/german_bert_run1/validation_loss_sg/optimistic_greider#validation_loss_sg=0.8192.ckpt",
    },
    {
        "model_dir": "experiments/run_outputs/english_roberta_run1",
        "run_name": "english_roberta_run1_exp",
        "validation_file": "data/MAD_TSC/original/en/test.jsonl",
        "ckpt_path": "experiments/run_outputs/english_roberta_run1/validation_loss_sg/inspiring_leavitt#validation_loss_sg=0.5957.ckpt",
    },
    {
        "model_dir": "experiments/run_outputs/spanish_camembert_run1",
        "run_name": "spanish_camembert_run1_exp",
        "validation_file": "data/MAD_TSC/original/es/test.jsonl",
        "ckpt_path": "experiments/run_outputs/spanish_camembert_run1/validation_loss_sg/unruffled_jones#validation_loss_sg=0.8980.ckpt",
    },
    {
        "model_dir": "experiments/run_outputs/french_camembert_run1",
        "run_name": "french_camembert_run1_exp",
        "validation_file": "data/MAD_TSC/original/fr/test.jsonl",
        "ckpt_path": "experiments/run_outputs/french_camembert_run1/validation_loss_sg/heuristic_hermann#validation_loss_sg=0.6705.ckpt",
    },
    {
        "model_dir": "experiments/run_outputs/italian_bert_run1",
        "run_name": "italian_bert_run1_exp",
        "validation_file": "data/MAD_TSC/original/it/test.jsonl",
        "ckpt_path": "experiments/run_outputs/italian_bert_run1/validation_loss_sg/distracted_knuth#validation_loss_sg=0.7870.ckpt",
    },
    {
        "model_dir": "experiments/run_outputs/dutch_bert_run1",
        "run_name": "dutch_bert_run1_exp",
        "validation_file": "data/MAD_TSC/original/nl/test.jsonl",
        "ckpt_path": "experiments/run_outputs/dutch_bert_run1/validation_loss_sg/objective_lichterman#validation_loss_sg=0.8288.ckpt",
    },
    {
        "model_dir": "experiments/run_outputs/portuguese_bert_run1",
        "run_name": "portuguese_bert_run1_exp",
        "validation_file": "data/MAD_TSC/original/pt/test.jsonl",
        "ckpt_path": "experiments/run_outputs/portuguese_bert_run1/validation_loss_sg/affectionate_bouman#validation_loss_sg=0.6885.ckpt",
    },
    {
        "model_dir": "experiments/run_outputs/roman_bert_run1",
        "run_name": "roman_bert_run1_exp",
        "validation_file": "data/MAD_TSC/original/ro/test.jsonl",
        "ckpt_path": "experiments/run_outputs/roman_bert_run1/validation_loss_sg/laughing_brattain#validation_loss_sg=0.7664.ckpt",
    },
]


final_results_dir = "experiments/TD_eval_collated_results"
os.makedirs(final_results_dir, exist_ok=True)
plots_dir = os.path.join(final_results_dir, "plots_monolingual")
os.makedirs(plots_dir, exist_ok=True)

all_experiment_results = []

# =============================
# ==== EVALUATION ============
# =============================

for exp in experiments:
    ckpt_path = exp["ckpt_path"]
    run_name = exp["run_name"]
    validation_file = exp["validation_file"]

    pl_model = PlFineTuneAbsaModel.load_from_checkpoint(ckpt_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pl_model = pl_model.to(device).eval()

    processor = pl_model.model.processor
    processor.set_return_tensors(True)

    examples = []
    with open(validation_file, "r") as f:
        for line in f:
            examples.append(json.loads(line.strip()))

    all_tokens, all_mentions_tokens, all_mentions_pos, all_sentiments, all_params = [], [], [], [], []

    for example in tqdm(examples, desc=f"Processing {run_name}"):
        sentence = example["sentence_normalized"]
        mention_info = example["targets"][0]
        mentions_pos = [(mention_info["from"], mention_info["to"])]
        tokens, mention_tokens, mentions_positions, label, param = processor.process_entry(
            sentence, mentions_pos, mention_info["mention"], mention_info["polarity"], [mention_info["mention"]]
        )
        all_tokens.append(tokens[0])
        all_mentions_tokens.append(mention_tokens)
        all_mentions_pos.append(mentions_positions)
        all_sentiments.append(label)
        all_params.append(param if param is not None else -1)

    dataset = AbsaDataset(all_tokens, all_mentions_tokens, all_mentions_pos, all_sentiments, all_params)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=False, collate_fn=AbsaDataCollator(
        mode_mask=MODE_MASK[pl_model.model_config["absa_model"]],
        tokenizer_mask_id=pl_model.tokenizer.mask_token_id,
        tokenizer_padding_id=pl_model.tokenizer.pad_token_id,
        tokenizer=pl_model.tokenizer,
        force_gpu=True if torch.cuda.is_available() else False
    ))

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch_tokens, attention_mask, x_select, classifying_locations, counts, sentiments, params in dataloader:
            out = pl_model.forward(batch_tokens=batch_tokens, attention_mask=attention_mask, classifying_locations=classifying_locations, counts=counts, x_select=x_select)
            preds = out.detach().cpu().numpy()
            labels = sentiments.cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels)

    preds_tensor = torch.FloatTensor(all_preds)
    labels_tensor = torch.LongTensor(all_labels)
    preds_labels = torch.argmax(preds_tensor, dim=-1)

    f1_macro = f1_score(labels_tensor.numpy(), preds_labels.numpy(), average="macro")
    acc = accuracy_score(labels_tensor.numpy(), preds_labels.numpy())

    result = {"run_name": run_name, "validation_f1_macro": f1_macro, "validation_accuracy": acc}
    all_experiment_results.append(result)

    save_path = os.path.join(final_results_dir, f"{run_name}_validation_eval_results.json")
    with open(save_path, "w") as f:
        json.dump(result, f, indent=4)

# =============================
# ==== PLOTTING ================
# =============================

df = pd.DataFrame(all_experiment_results)
df.to_csv(os.path.join(final_results_dir, "collated_validation_eval_results.csv"), index=False)

# Extract language
languages = [name.split("_")[0] for name in df.run_name]
df["language"] = languages

# 1. Bar plot: F1 per language
plt.figure()
sns.barplot(x="language", y="validation_f1_macro", data=df)
plt.title("Macro F1 Score per Language (Monolingual Models)")
plt.savefig(os.path.join(plots_dir, "bar_f1_per_language_monolingual.png"))

# 2. Bar plot: Accuracy per language
plt.figure()
sns.barplot(x="language", y="validation_accuracy", data=df)
plt.title("Accuracy per Language (Monolingual Models)")
plt.savefig(os.path.join(plots_dir, "bar_accuracy_per_language_monolingual.png"))

# 3. Scatter: F1 vs Accuracy
plt.figure()
sns.scatterplot(x="validation_accuracy", y="validation_f1_macro", hue="language", data=df)
plt.title("F1 vs Accuracy (Monolingual Models)")
plt.savefig(os.path.join(plots_dir, "scatter_f1_vs_accuracy_monolingual.png"))

# 4. Error Rate plot
plt.figure()
df["error_rate"] = 1 - df["validation_accuracy"]
sns.barplot(x="language", y="error_rate", data=df)
plt.title("Error Rate per Language (Monolingual Models)")
plt.savefig(os.path.join(plots_dir, "bar_error_rate_per_language_monolingual.png"))

# 5. Heatmap
heat_df = df.set_index("language")[["validation_f1_macro", "validation_accuracy"]]
plt.figure()
sns.heatmap(heat_df, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Heatmap of Language Performance (Monolingual Models)")
plt.savefig(os.path.join(plots_dir, "heatmap_language_performance_monolingual.png"))

# 6. Confusion Matrices for EN, FR
# for lang_code, run_name_match in [("en", "en_roberta_spc"), ("fr", "fr_camembert_french")]:
#     match = [e for e in experiments if run_name_match in e["run_name"]][0]
#     pl_model = PlFineTuneAbsaModel.load_from_checkpoint(match["ckpt_path"])
#     pl_model = pl_model.to(device).eval()
#     processor = pl_model.model.processor
#     processor.set_return_tensors(True)
#     examples = []
#     with open(match["validation_file"], "r") as f:
#         for line in f:
#             examples.append(json.loads(line.strip()))
#     labels, preds = [], []
#     for example in tqdm(examples, desc=f"Confusion {lang_code}"):
#         sent = example["sentence_normalized"]
#         mention_info = example["targets"][0]
#         tokens, mention_tokens, mentions_positions, label, param = processor.process_entry(
#             sent, [(mention_info["from"], mention_info["to"])], mention_info["mention"], mention_info["polarity"], [mention_info["mention"]]
#         )
#         out = pl_model.forward(
#             batch_tokens=torch.stack([tokens[0]]).to(device),
#             attention_mask=(torch.stack([tokens[0]]) != pl_model.tokenizer.pad_token_id).to(device),
#             classifying_locations=torch.stack([mention_tokens[0][0]] if isinstance(mention_tokens[0], (tuple, list)) else [mention_tokens[0]]).to(device),
#             counts=torch.tensor([1]).to(device),
#             x_select=None
#         )
#         pred = torch.argmax(out, dim=-1).item()
#         preds.append(pred)
#         labels.append(label)

#     cm = confusion_matrix(labels, preds)
#     disp = ConfusionMatrixDisplay(confusion_matrix=cm)
#     disp.plot(cmap="Blues")
#     plt.title(f"Confusion Matrix ({lang_code.upper()}) - Monolingual")
#     plt.savefig(os.path.join(plots_dir, f"confusion_matrix_{lang_code}_monolingual.png"))

# 7. Language Grouping
families = {"Romance": ["es", "fr", "it", "pt", "ro"], "Germanic": ["en", "de", "nl"]}
family_scores = {}
for fam, langs in families.items():
    subset = df[df.language.isin(langs)]
    family_scores[fam] = {"F1": subset.validation_f1_macro.mean(), "Accuracy": subset.validation_accuracy.mean()}
family_df = pd.DataFrame(family_scores).T

plt.figure()
family_df.plot(kind="bar")
plt.title("Language Family Comparison (Monolingual Models)")
plt.xticks(rotation=0)
plt.savefig(os.path.join(plots_dir, "family_performance_comparison_monolingual.png"))

print("\n[INFO] Monolingual plots generated and saved.")


# ==== After all experiments: Pretty Table ====

df = pd.DataFrame(all_experiment_results)
print("\n=== Final Evaluation Summary ===\n")
print(df.to_string(index=False))