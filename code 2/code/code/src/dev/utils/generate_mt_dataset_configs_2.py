import os
import json

# Languages to generate for
langs = ["es", "de", "it", "fr", "pt", "ro", "nl"]
methods = {
    "m2m12B": "m2m12B",
    "deepl": "deepl"
}

# Output directory
output_dir = "configs/datasets"
os.makedirs(output_dir, exist_ok=True)

for lang in langs:
    for method_key, subfolder in methods.items():
        # === from_{lang}_to_en ===
        config_from_lang_to_en = {
            "name_dataset": f"from_{lang}_to_en_{method_key}",
            "format": "newsmtsc",
            "folder_dataset": f"from_{lang}_to_en",
            "filenames": {
                "test": "test.jsonl"
            },
            "splitting_strategy": {
                "train": [0, 0, 0],
                "test": [1, 0, 0],
                "validation": [0, 0, 0]
            }
        }

        file_name_from_lang_to_en = f"{output_dir}/MAD_TSC_{method_key}_from_{lang}_to_en.json"
        with open(file_name_from_lang_to_en, "w") as f:
            json.dump(config_from_lang_to_en, f, indent=4)
        print(f"✅ Created: {file_name_from_lang_to_en}")

        # === from_en_to_{lang} ===
        config_from_en_to_lang = {
            "name_dataset": f"from_en_to_{lang}_{method_key}",
            "format": "newsmtsc",
            "folder_dataset": f"from_en_to_{lang}",
            "filenames": {
                "train": "train.jsonl",
                "validation": "validation.jsonl",
                "test": "test.jsonl"
            },
            "splitting_strategy": {
                "train": [1, 0, 0],
                "validation": [0, 1, 0],
                "test": [0, 0, 1]
            }
        }

        file_name_from_en_to_lang = f"{output_dir}/MAD_TSC_{method_key}_from_en_to_{lang}.json"
        with open(file_name_from_en_to_lang, "w") as f:
            json.dump(config_from_en_to_lang, f, indent=4)
        print(f"✅ Created: {file_name_from_en_to_lang}")