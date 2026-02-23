# eval_translations_spc.py

import torch
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.metrics import f1_score, accuracy_score

sys.path.append("./src")

from tscbench.finetuning.plightning.plfinetuneabsa import PlFineTuneAbsaModel
from tscbench.finetuning.absa.constants import MODE_MASK
from tscbench.data.load.absa import AbsaModelProcessor, AbsaDataset, AbsaDataCollator

# =============================
# === CONFIGURATION ==========
# =============================

ckpt_path = "experiments/run_outputs/en_roberta_spc/validation_loss_sg/laughing_brattain#validation_loss_sg=0.5701.ckpt"
model_dir = "experiments/run_outputs/en_roberta_spc/"
run_name = "en_roberta_spc_run1_exp"

translated_sources = ["de", "fr", "es", "it", "pt", "nl", "ro"]
translation_types = ["deepl", "m2m12B"]

output_dir = os.path.join(model_dir, run_name)
metrics_dir = os.path.join(output_dir, "metrics")
noise_dir = os.path.join(output_dir, "translation_noise_analysis")
confidence_dir = os.path.join(output_dir, "confidence_vs_correctness")
length_correctness_dir = os.path.join(noise_dir, "length_vs_correctness")

os.makedirs(metrics_dir, exist_ok=True)
os.makedirs(noise_dir, exist_ok=True)
os.makedirs(confidence_dir, exist_ok=True)
os.makedirs(length_correctness_dir, exist_ok=True)

# =============================
# === LOAD MODEL =============
# =============================

print("[INFO] Loading model from checkpoint...")
pl_model = PlFineTuneAbsaModel.load_from_checkpoint(ckpt_path)
device = "cuda" if torch.cuda.is_available() else "cpu"
pl_model = pl_model.to(device)
pl_model = pl_model.eval()

processor = pl_model.model.processor
processor.set_return_tensors(True)

results_table = {}
noise_analysis = {}
confidence_analysis = {}
length_vs_correctness = {}

# =============================
# === EVALUATION LOOP ========
# =============================

def evaluate_file(path, lang, trans_type):
    examples = []
    with open(path, "r") as f:
        for line in f:
            examples.append(json.loads(line.strip()))

    all_tokens, all_mentions_tokens, all_mentions_pos = [], [], []
    all_sentiments, all_params = [], []
    sentence_lengths = []
    prediction_correctness = []

    for example in tqdm(examples, desc=f"Tokenizing examples from {path}"):
        sentence = example["sentence_normalized"]
        mention_info = example["targets"][0]
        mentions_pos = [(mention_info["from"], mention_info["to"])]
        main_mention = mention_info["mention"]
        sentiment = mention_info["polarity"]
        all_mentions = [mention_info["mention"]]

        tokens, mention_tokens, mentions_positions, label, param = processor.process_entry(
            sentence, mentions_pos, main_mention, sentiment, all_mentions
        )

        all_tokens.append(tokens[0])
        all_mentions_tokens.append(mention_tokens)
        all_mentions_pos.append(mentions_positions)
        all_sentiments.append(label)
        all_params.append(param if param is not None else -1)
        sentence_lengths.append(len(sentence.split()))

    dataset = AbsaDataset(
        all_tokens, all_mentions_tokens, all_mentions_pos, all_sentiments, all_params
    )

    collator = AbsaDataCollator(
        mode_mask=MODE_MASK[pl_model.model_config["absa_model"]],
        tokenizer_mask_id=pl_model.tokenizer.mask_token_id,
        tokenizer_padding_id=pl_model.tokenizer.pad_token_id,
        force_gpu=(device == "cuda"),
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=pl_model.optimizer_config["optimizer"]["batch_size_dataloader"] * 2,
        shuffle=False,
        num_workers=0,
        collate_fn=collator,
        drop_last=False,
    )

    all_preds, all_labels, all_confidences = [], [], []
    with torch.no_grad():
        for batch in dataloader:
            batch_tokens, attention_mask, x_select, classifying_locations, counts, sentiments, params = batch
            out = pl_model.forward(
                batch_tokens=batch_tokens,
                attention_mask=attention_mask,
                classifying_locations=classifying_locations,
                counts=counts,
                x_select=x_select,
            )
            softmax_out = torch.softmax(out, dim=-1)
            confidences, preds = torch.max(softmax_out, dim=-1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(sentiments.cpu().numpy())
            all_confidences.extend(confidences.cpu().numpy())

    labels_np = np.array(all_labels)
    preds_np = np.array(all_preds)

    correctness_array = (labels_np == preds_np).astype(int)

    key = f"{lang}|{trans_type}"

    noise_analysis[key] = {
        "avg_sentence_length": np.mean(sentence_lengths),
        "std_sentence_length": np.std(sentence_lengths),
        "wrong_prediction_rate": 1 - (labels_np == preds_np).mean()
    }

    confidence_analysis[key] = {
        "confidences": all_confidences,
        "correctness": correctness_array.tolist()
    }

    length_vs_correctness[key] = {
        "sentence_lengths": sentence_lengths,
        "correctness": correctness_array.tolist()
    }

    f1_macro = f1_score(labels_np, preds_np, average="macro")
    f1_pn = f1_score(labels_np, preds_np, average=None)[1]
    acc = accuracy_score(labels_np, preds_np)
    recall_macro = (labels_np == preds_np).mean()

    return f1_macro, f1_pn, acc, recall_macro

# =============================
# === MAIN LOOP ===============
# =============================

for trans_type in translation_types:
    results_table[trans_type] = {}
    print(f"\n==============================")
    print(f"Evaluating on translation: {trans_type.upper()}")
    print(f"==============================")
    for lang in translated_sources:
        test_path = f"data/MAD_TSC/{trans_type}/from_{lang}_to_en/test.jsonl"
        if not os.path.exists(test_path):
            print(f"[WARNING] Missing file: {test_path}")
            continue

        print(f"\n[INFO] Running evaluation on: {lang} → en [{trans_type}]")
        f1_macro, f1_pn, acc, recall_macro = evaluate_file(test_path, lang, trans_type)
        results_table[trans_type][lang] = {
            "f1_macro": f1_macro,
            "f1_pn": f1_pn,
            "accuracy": acc,
            "recall_macro": recall_macro
        }
        print(f"[RESULT] F1_macro: {f1_macro:.4f}, F1_pn: {f1_pn:.4f}, Acc: {acc:.4f}, Rec: {recall_macro:.4f}")

# =============================
# === SAVE EVERYTHING ========
# =============================

metrics_save_path = os.path.join(metrics_dir, "cross_lingual_eval_results_full_metrics.json")
with open(metrics_save_path, "w") as f:
    json.dump(results_table, f, indent=4)

with open(os.path.join(noise_dir, "translation_noise_stats.json"), "w") as f:
    json.dump(noise_analysis, f, indent=4)

import numbers

def convert_np(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    raise TypeError

# When saving confidence analysis:
with open(os.path.join(confidence_dir, "confidence_vs_correctness.json"), "w") as f:
    json.dump(confidence_analysis, f, indent=4, default=convert_np)

with open(os.path.join(length_correctness_dir, "length_vs_correctness.json"), "w") as f:
    json.dump(length_vs_correctness, f, indent=4)

print(f"\n[INFO] All metrics and stats saved ✅")

# =============================
# === PLOTTING ===============
# =============================

for stat in ["avg_sentence_length", "wrong_prediction_rate"]:
    plt.figure(figsize=(10,6))
    for trans_type in translation_types:
        values = [noise_analysis[f"{lang}|{trans_type}"][stat] for lang in translated_sources]
        plt.plot(translated_sources, values, marker="o", label=f"{trans_type.upper()}")
    plt.title(f"{stat.replace('_', ' ').title()} across translations")
    plt.xlabel("Source Language")
    plt.ylabel(stat.replace('_', ' ').title())
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(noise_dir, f"{stat}_comparison.png"))
    plt.close()

# === Sentence Length vs Wrong Rate Plot ===

bucket_size = 5
max_length = 80
buckets = np.arange(0, max_length+bucket_size, bucket_size)

for trans_type in translation_types:
    plt.figure(figsize=(10,6))
    for lang in translated_sources:
        key = f"{lang}|{trans_type}"
        if key not in length_vs_correctness:
            continue

        lengths = np.array(length_vs_correctness[key]["sentence_lengths"])
        correctness = np.array(length_vs_correctness[key]["correctness"])

        wrong_rates = []
        for i in range(len(buckets)-1):
            idx = (lengths >= buckets[i]) & (lengths < buckets[i+1])
            if idx.sum() == 0:
                wrong_rates.append(np.nan)
            else:
                wrong_rates.append(1 - correctness[idx].mean())

        plt.plot(buckets[:-1], wrong_rates, marker="o", label=f"{lang}")

    plt.title(f"Wrong Prediction Rate vs Sentence Length [{trans_type.upper()}]")
    plt.xlabel("Sentence Length Bucket")
    plt.ylabel("Wrong Prediction Rate")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(length_correctness_dir, f"wrong_rate_vs_length_{trans_type}.png"))
    plt.close()

print("\n==============================")
print(" ALL EVALUATIONS + ANALYSIS + LENGTH CORRECTNESS DONE ✅")
print("==============================")
