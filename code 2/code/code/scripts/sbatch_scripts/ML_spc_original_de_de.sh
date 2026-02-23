#!/bin/bash
#SBATCH --export=ALL
#SBATCH --get-user-env=L
#SBATCH --job-name=ML_spc_original_de_de
#SBATCH -A 132742115536
#SBATCH --time=12:00:00
#SBATCH --tasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16G
#SBATCH --output=logs/job_ML.o%J
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1

module load GCCcore/12.2.0
module load Python/3.10.8
module load CUDA/12.4.1

source /scratch/user/ps41/ws/nlp/nlpvenv/bin/activate

export http_proxy="http://10.73.132.63:8080"
export https_proxy="http://10.73.132.63:8080"

export UUID=$RANDOM
echo "[INFO] Job UUID: $UUID"

export TOKENIZERS_PARALLELISM=false
export SCRATCH_PREFIX=/scratch/user/ps41/ws/nlp/reproduce/MAD_TSC/scratch/data
export DATA_SCRATCH=$SCRATCH_PREFIX
export DATA_DIR=$DATA_SCRATCH
export EXPERIMENT_DIR=/scratch/user/ps41/ws/nlp/reproduce/MAD_TSC/experiments
export EXPERIMENT_SCRATCH=/scratch/user/ps41/ws/nlp/reproduce/MAD_TSC/scratch/io/$UUID

mkdir -p $DATA_SCRATCH
mkdir -p "$DATA_SCRATCH/$UUID/MAD_TSC_de"
mkdir -p $EXPERIMENT_DIR
mkdir -p $EXPERIMENT_SCRATCH

# Copy dataset
rsync -avh ./data/MAD_TSC/original/de/ "$DATA_SCRATCH/$UUID/MAD_TSC_de/"

# Log starting time
start_time=$(date +%s)
echo "Starting training at $(date)..."
echo "[INFO] Training SPC model on original/de with model xlm-roberta-base"

# Launch training
tscbench finetune tsc \
-n ML_spc_original_de_de \
-m "xlm-roberta-base" \
-t "xlm-roberta-base" \
--tsc-model-config configs/models/spc_model_default.json \
--dataset-config configs/datasets/MAD_TSC_de.json \
--gpu-pl-config configs/lightning/default_lightning_config.json \
--sub-path-final-folder run_outputs/ML_original_de_de_spc \
--optimizer-config configs/optimizers/default_optimizer.json \
--keep-best-models

exit_code=$?

end_time=$(date +%s)
duration=$((end_time - start_time))

hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

printf "[%s] Duration: %02d:%02d:%02d (Exit code: %d)\n\n" \
  "$(date)" "$hours" "$minutes" "$seconds" "$exit_code"
