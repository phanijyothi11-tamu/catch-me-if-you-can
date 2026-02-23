from datasets import load_dataset

# # Load CNN/DailyMail
dataset = load_dataset("cnn_dailymail", "3.0.0", split="train")

# print(dataset[0])  # Check what fields are available

import spacy
import json
import random
from tqdm import tqdm
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer

# # Load spaCy English NER
nlp = spacy.load("en_core_web_sm")

# # Load M2M100 model for translation (if you want later)
# model_name = "facebook/m2m100_418M"
# tokenizer = M2M100Tokenizer.from_pretrained(model_name)
# model = M2M100ForConditionalGeneration.from_pretrained(model_name)

# def translate_to_dutch(text, src_lang="en", tgt_lang="nl"):
#     tokenizer.src_lang = src_lang
#     encoded = tokenizer(text, return_tensors="pt")
#     generated_tokens = model.generate(**encoded, forced_bos_token_id=tokenizer.get_lang_id(tgt_lang))
#     return tokenizer.decode(generated_tokens[0], skip_special_tokens=True)

# # Prepare extraction
# extracted_samples = []
# gid_counter = 30000

dataset_small = dataset.select(range(1000))

# for item in tqdm(dataset_small):  # Limit to 1000 articles
#     text = item["article"]

#     doc = nlp(text)

#     for sent in doc.sents:
#         ent_doc = nlp(sent.text)
#         entities = [ent for ent in ent_doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "LOC", "PRODUCT"]]

#         if entities:
#             entity = random.choice(entities)

#             try:
#                 dutch_sentence = translate_to_dutch(sent.text)
#             except Exception as e:
#                 print(f"Translation failed, skipping sentence: {sent.text}")
#                 continue

#             if entity.text not in dutch_sentence:
#                 continue

#             start_idx = dutch_sentence.index(entity.text)
#             end_idx = start_idx + len(entity.text)

#             primary_gid = f"{gid_counter}_{random.randint(0,99)}_{start_idx}_{end_idx}"

#             sample = {
#                 "primary_gid": primary_gid,
#                 "sentence_normalized": dutch_sentence,
#                 "targets": [
#                     {
#                         "Input.gid": primary_gid,
#                         "from": start_idx,
#                         "to": end_idx,
#                         "mention": entity.text,
#                         "polarity": 3.0
#                     }
#                 ]
#             }
#             extracted_samples.append(sample)
#             gid_counter += 1
#             break  # 1 sentence per article

# print(f"Extracted {len(extracted_samples)} samples.")

# # Save to .jsonl
# with open("news_tsc_nonpolitical_nl.jsonl", "w", encoding="utf-8") as f:
#     for item in extracted_samples:
#         f.write(json.dumps(item, ensure_ascii=False) + "\n")

# print("✅ All done!")


# First build JSONL file from English sentences
# Without translation step now

extracted_samples = []
gid_counter = 30000

for item in tqdm(dataset_small):
    text = item["article"]
    doc = nlp(text)

    # Try to find meaningful entity
    found = False
    for sent in doc.sents:
        if len(sent.text.strip()) < 30:  # Ignore very short sentences (likely metadata)
            continue
        
        ent_doc = nlp(sent.text)
        entities = [ent for ent in ent_doc.ents if ent.label_ in ["PERSON", "ORG", "PRODUCT"]]

        if entities:
            entity = random.choice(entities)

            primary_gid = f"{gid_counter}_{random.randint(0,99)}_{entity.start_char}_{entity.end_char}"

            sample = {
                "primary_gid": primary_gid,
                "sentence_normalized": sent.text,
                "targets": [
                    {
                        "Input.gid": primary_gid,
                        "from": entity.start_char,
                        "to": entity.end_char,
                        "mention": entity.text,
                        "polarity": 3.0  # Default neutral
                    }
                ]
            }

            extracted_samples.append(sample)
            gid_counter += 1
            found = True
            break  # Move to next article after finding a good sample

    if not found:
        continue  # Skip articles without good sentences

print(f"Extracted {len(extracted_samples)} samples.")

# Save English version
with open("news_tsc_nonpolitical_en.jsonl", "w", encoding="utf-8") as f:
    for item in extracted_samples:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

print("✅ Saved better English TSC data.")
