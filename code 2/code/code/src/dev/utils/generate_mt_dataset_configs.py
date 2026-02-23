import os
import json

# Languages to generate for
langs = ["es", "de", "it", "fr", "pt", "ro"]
methods = {
    "m2m": "m2m12B",
    "deepl": "deepl"
}

# Output directory
output_dir = "configs/datasets"
os.makedirs(output_dir, exist_ok=True)

for lang in langs:
    for method, subfolder in methods.items():
        config = {
            "name_dataset": f"test_from_{lang}_to_en_{method}",
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

        file_name = f"{output_dir}/test_from_{lang}_to_en_{method}.json"
        with open(file_name, "w") as f:
            json.dump(config, f, indent=4)
        print(f"✅ Created: {file_name}")
