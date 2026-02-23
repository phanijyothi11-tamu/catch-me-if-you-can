# check_features_alignment.py

import json
from transformers import AutoTokenizer
from src.tscbench.data.load.absa import AbsaModelProcessor
from src.tscbench.finetuning.plightning.plfinetuneabsa import PlFineTuneAbsaModel

import torch

# === CONFIG ===
checkpoint_path = "experiments/run_outputs/en_roberta_spc/validation_loss_sg/laughing_brattain#validation_loss_sg=0.5701.ckpt"
tokenizer_path = "roberta-base"  # same one used in training
example_file = "data/MAD_TSC/original/en/train.jsonl"  # pick one entry from your training set

# === MANUAL PROCESSOR (training tokenizer logic) ===
print("[INFO] Loading manual processor...")
processor = AbsaModelProcessor(
    tokenizer_path=tokenizer_path,
    prompt_template="",  # put your prompt template here if you used it
    replace_by_main_mention=False,
    replace_by_special_token=None,
    return_tensors=True,
)

# === LOADED MODEL (what checkpoint expects) ===
print("[INFO] Loading trained PlFineTuneAbsaModel from checkpoint...")
pl_model = PlFineTuneAbsaModel.load_from_checkpoint(checkpoint_path)

# === Load an example ===
with open(example_file, "r") as f:
    example = json.loads(f.readline())

sentence = example["sentence_normalized"]
mention_info = example["targets"][0]
mentions_pos = [(mention_info["from"], mention_info["to"])]
main_mention = mention_info["mention"]
sentiment = mention_info["polarity"]
all_mentions = [mention_info["mention"]]

# === MANUAL tokenization ===
tokens_manual, _, _, _, _ = processor.process_entry(
    sentence,
    mentions_pos,
    main_mention,
    sentiment,
    all_mentions,
)
tokens_manual = tokens_manual[0].squeeze(0)  # [0] because you get a list per template
print(f"[INFO] Manual tokens shape: {tokens_manual.shape}")

# === Model tokenization (pl_model.processor) ===
processor_model = pl_model.model.processor
processor_model.set_return_tensors(True)

tokens_model, _, _, _, _ = processor_model.process_entry(
    sentence,
    mentions_pos,
    main_mention,
    sentiment,
    all_mentions,
)
tokens_model = tokens_model[0].squeeze(0)
print(f"[INFO] Model tokens shape: {tokens_model.shape}")

# === Comparison ===
if torch.equal(tokens_manual, tokens_model):
    print("[SUCCESS] Manual processor and model processor are generating IDENTICAL input tokens ✅")
else:
    diff = (tokens_manual != tokens_model).sum()
    print(f"[WARNING] There are {diff.item()} mismatched tokens ⚠️")
    print("Manual:", tokens_manual)
    print("Model :", tokens_model)
