# compare_mono_vs_multi.py

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Paths
mono_csv = "experiments/eval_collated_results/collated_validation_eval_results.csv"
multi_csv = "experiments/ML_eval_collated_results/collated_test_eval_results.csv"
output_dir = "experiments/monovsmulti/plots"
os.makedirs(output_dir, exist_ok=True)

# Load CSVs
mono_df = pd.read_csv(mono_csv)
multi_df = pd.read_csv(multi_csv)

# Extract language names
mono_df["language"] = mono_df["run_name"].apply(lambda x: x.split("_")[0])
multi_df["language"] = multi_df["language"]

# Align and merge
merged = pd.merge(
    mono_df[["language", "validation_f1_macro", "validation_accuracy"]],
    multi_df[["language", "test_f1_macro", "test_accuracy"]],
    on="language",
    suffixes=("_mono", "_multi")
)

# Calculate Differences
merged["f1_difference"] = merged["validation_f1_macro"] - merged["test_f1_macro"]
merged["accuracy_difference"] = merged["validation_accuracy"] - merged["test_accuracy"]

# Save merged CSV
merged.to_csv("experiments/monovsmulti/monovsmulti_comparison.csv", index=False)

# === Plot 1: F1 Mono vs Multi ===
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x="test_f1_macro",
    y="validation_f1_macro",
    hue="language",
    data=merged,
    s=100
)
plt.plot([0, 1], [0, 1], 'k--', label="y = x")
plt.xlabel("Multilingual F1 Score")
plt.ylabel("Monolingual F1 Score")
plt.title("F1 Score: Monolingual vs Multilingual")
plt.legend()
plt.grid(True)
plt.xlim(0.6, 0.8)
plt.ylim(0.6, 0.8)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "scatter_f1_mono_vs_multi.png"))
plt.close()

# === Plot 2: Accuracy Mono vs Multi ===
plt.figure(figsize=(8, 6))
sns.scatterplot(
    x="test_accuracy",
    y="validation_accuracy",
    hue="language",
    data=merged,
    s=100
)
plt.plot([0, 1], [0, 1], 'k--', label="y = x")
plt.xlabel("Multilingual Accuracy")
plt.ylabel("Monolingual Accuracy")
plt.title("Accuracy: Monolingual vs Multilingual")
plt.legend()
plt.grid(True)
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "scatter_accuracy_mono_vs_multi.png"))
plt.close()

# Final message
print("\n✅ Monolingual vs Multilingual comparison plots saved in:", output_dir)
print("✅ Merged table with differences saved at: experiments/monovsmulti/monovsmulti_comparison.csv")
print("\n=== Final Comparison Table ===\n")
print(merged.to_string(index=False))
