import pandas as pd
import time
import torch
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
import json

# ========== Step 1: Load Datasets ==========
print("Loading English datasets...")
df_validation = pd.read_json(r'./../../MAD_TSC/data/MAD_TSC/original/en/validation.jsonl', lines=True, encoding='utf-8-sig')
df_test = pd.read_json(r'./../../MAD_TSC/data/MAD_TSC/original/en/test.jsonl', lines=True, encoding='utf-8-sig')
df_train = pd.read_json(r'./../../MAD_TSC/data/MAD_TSC/original/en/train.jsonl', lines=True, encoding='utf-8-sig')

# Record sizes to split later
n_val = len(df_validation)
n_test = len(df_test)
n_train = len(df_train)

# Concatenate all for batch processing
df = pd.concat([df_validation, df_test, df_train], ignore_index=True)
df_sentences = df['sentence_normalized'].tolist()  # Extract list of sentences

print(f"Loaded {len(df_sentences)} sentences.\n")

# ========== Step 2: Load Model and Tokenizer ==========
print("Loading model and tokenizer...")
model_name = "facebook/m2m100-12B-last-ckpt"  # Lighter version (12B model is much bigger)
model = M2M100ForConditionalGeneration.from_pretrained(model_name,  device_map="auto", torch_dtype=torch.float16)
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
tokenizer.src_lang = "en"
print("Model and tokenizer loaded.\n")

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#model.to(device)
print(f"Using device: {device}\n")

# ========== Step 3: Translate Sentences in Batches ==========
print("Starting sentence translation...")
batch_size = 4
hindi_translated_sentences = []

for start_idx in range(0, len(df_sentences), batch_size):
    end_idx = min(start_idx + batch_size, len(df_sentences))
    batch = df_sentences[start_idx:end_idx]
    
    # Tokenize and move to device
    inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(device)

    # Translate
    start_time = time.time()
    generated_tokens = model.generate(
        **inputs,
        forced_bos_token_id=tokenizer.get_lang_id("hi"),
        max_length=512
    )
    elapsed = time.time() - start_time

    # Decode and store
    translations = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
    hindi_translated_sentences.extend(translations)

    # Logging
    print(f"Translated {end_idx}/{len(df_sentences)} sentences — Batch took {elapsed:.2f}s "
          f"(Avg {elapsed/len(batch):.2f}s per sentence)")

# Update 'sentence_normalized' column
df['sentence_normalized'] = hindi_translated_sentences
print("\nSentence translation complete.\n")

# ========== Step 4: Translate Mentions ==========
print("Starting mention translation inside targets...")
new_targets_list = []

for idx, record in df.iterrows():
    targets = record['targets']
    new_targets = []
    
    for target in targets:
        original_mention = target['mention']

        # Translate mention
        tokenizer.src_lang = "en"
        encoded = tokenizer(original_mention, return_tensors="pt").to(device)
        generated_tokens = model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id("hi"),
            max_length=50
        )
        translated_mention = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]

        # Update mention (keep 'from' and 'to' unchanged)
        target['mention'] = translated_mention
        new_targets.append(target)

    new_targets_list.append(new_targets)

# Update targets column
df['targets'] = new_targets_list
print("Mention translation complete.\n")

# ========== Step 5: Split Back into Train / Validation / Test ==========
print("Splitting translated data back into train/validation/test splits...")

df_validation_hindi = df.iloc[:n_val]
df_test_hindi = df.iloc[n_val:n_val + n_test]
df_train_hindi = df.iloc[n_val + n_test:]

# Save separately
df_validation_hindi.to_json('hindi_validation.jsonl', orient='records', lines=True, force_ascii=False)
df_test_hindi.to_json('hindi_test.jsonl', orient='records', lines=True, force_ascii=False)
df_train_hindi.to_json('hindi_train.jsonl', orient='records', lines=True, force_ascii=False)

print("\nTranslation complete. Hindi datasets saved as:")
print("- hindi_validation.jsonl")
print("- hindi_test.jsonl")
print("- hindi_train.jsonl")
