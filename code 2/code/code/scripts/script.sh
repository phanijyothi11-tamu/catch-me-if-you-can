#!/bin/bash

# ======== SETTINGS ========
# scratch and experiment directories
BASE_DIR="/scratch/user/ps41/ws/nlp/reproduce/MAD_TSC"
SCRATCH_DATA_DIR="$BASE_DIR/scratch/data"
EXPERIMENT_DIR="$BASE_DIR/experiments"

mkdir -p $SCRATCH_DATA_DIR
mkdir -p $EXPERIMENT_DIR

# Model mappings (language -> HF model name)
declare -A MODEL_MAP
MODEL_MAP["en"]="roberta-base"
MODEL_MAP["es"]="bertin-project/bertin-roberta-base-spanish"
MODEL_MAP["de"]="bert-base-german-cased"
MODEL_MAP["it"]="dbmdz/bert-base-italian-uncased"
MODEL_MAP["fr"]="camembert-base"
MODEL_MAP["pt"]="neuralmind/bert-base-portuguese-cased"
MODEL_MAP["nl"]="pdelobelle/robbert-v2-dutch-base"
MODEL_MAP["ro"]="dumitrescustefan/bert-base-romanian-cased-v1"

# Datasets folders
FOLDERS=("m2m12B")

# ======== START JOB SUBMISSION ========

for folder in "${FOLDERS[@]}"; do
    DATASET_ROOT="$BASE_DIR/data/MAD_TSC/$folder"

    for dataset_path in "$DATASET_ROOT"/*; do
        dataset_name=$(basename "$dataset_path")  # e.g., from_en_to_de
        echo "[INFO] Found dataset: $dataset_name in $folder"

        # Infer language based on dataset name
        lang="${dataset_name##*_}"  # last part after last underscore
        model_name="${MODEL_MAP[$lang]}"

        if [ -z "$model_name" ]; then
            echo "[WARN] No model mapping found for language: $lang. Skipping..."
            continue
        fi

        # Define job and folder names
        job_name="spc_${folder}_${dataset_name}_${lang}"
        exp_subfolder="run_outputs/${folder}_${dataset_name}_${lang}_spc"

        # === Create temp SBATCH script ===
        sbatch_script="sbatch_scripts/${job_name}.sh"
        mkdir -p sbatch_scripts

        cat <<EOT > $sbatch_script
#!/bin/bash
#SBATCH --export=ALL
#SBATCH --get-user-env=L
#SBATCH --job-name=${job_name}
#SBATCH -A 132742115536
#SBATCH --time=12:00:00
#SBATCH --tasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --output=logs/job_m2m_train.o%J
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load GCCcore/12.2.0
module load Python/3.10.8
module load CUDA/12.4.1

source /scratch/user/ps41/ws/nlp/nlpvenv/bin/activate

export http_proxy="http://10.73.132.63:8080"
export https_proxy="http://10.73.132.63:8080"

export UUID=\$RANDOM
export TOKENIZERS_PARALLELISM=false
export SCRATCH_PREFIX=$SCRATCH_DATA_DIR
export DATA_SCRATCH=\$SCRATCH_PREFIX
export DATA_DIR=\$DATA_SCRATCH
export EXPERIMENT_DIR=$EXPERIMENT_DIR
export EXPERIMENT_SCRATCH=$BASE_DIR/scratch/io/\$UUID

mkdir -p \$DATA_SCRATCH
mkdir -p "\$DATA_SCRATCH/\$UUID/${dataset_name}"
mkdir -p \$EXPERIMENT_DIR
mkdir -p \$EXPERIMENT_SCRATCH

# Copy dataset
rsync -avh ./data/MAD_TSC/${folder}/${dataset_name}/ "\$DATA_SCRATCH/\$UUID/${dataset_name}/"

# Log starting time
start_time=\$(date +%s)
echo "Starting training at \$(date)..."
echo "[INFO] Training SPC model on ${folder}/${dataset_name} with model ${model_name}"

# Launch training
tscbench finetune tsc \\
-n ${job_name} \\
-m "${model_name}" \\
-t "${model_name}" \\
--tsc-model-config configs/models/spc_model_default.json \\
--dataset-config configs/datasets/MAD_TSC_${folder}_${dataset_name}.json \\
--gpu-pl-config configs/lightning/default_lightning_config.json \\
--sub-path-final-folder ${exp_subfolder} \\
--optimizer-config configs/optimizers/default_optimizer.json \\
--keep-best-models

exit_code=\$?

end_time=\$(date +%s)
duration=\$((end_time - start_time))

hours=\$((duration / 3600))
minutes=\$(((duration % 3600) / 60))
seconds=\$((duration % 60))

printf "[%s] Duration: %02d:%02d:%02d (Exit code: %d)\n\n" \\
  "\$(date)" "\$hours" "\$minutes" "\$seconds" "\$exit_code"
EOT

        chmod +x $sbatch_script
        echo "[INFO] Submitting job: $job_name"
        sbatch $sbatch_script
    done
done
