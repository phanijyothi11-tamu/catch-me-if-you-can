#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=ALL
#SBATCH --get-user-env=L
 
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=MAD-TSC-HI   # Sets the job name to AbaqusJob
#SBATCH --time=12:00:00          # Sets the runtime limit to 2 hr
#SBATCH --tasks=1              # Request 1 node
#SBATCH --ntasks-per-node=1    # Requests 10 cores per node (1 node)
#SBATCH --mem=16G               # Requests 50GB of memory per node
#SBATCH --output=job_hindi.o%J  # Sends stdout and stderr to job.o[jobID]
 
#SBATCH --partition=gpu   #Request job to be put in the GPU queue
#SBATCH --gres=gpu:1      #Request 1 GPU per node can be 1 or 2
 
## Load the module
module load GCCcore/12.2.0
module load Python/3.10.8
module load CUDA/12.4.1
 
source /scratch/user/ps41/ws/nlp/nlpvenv/bin/activate 

# Set proxy
export http_proxy="http://10.73.132.63:8080"
export https_proxy="http://10.73.132.63:8080"

## Set working directory
cd /scratch/user/ps41/ws/nlp/reproduce/MAD_TSC

## Environment variables
export UUID=$RANDOM
echo "[INFO] Job UUID: $UUID"

export TOKENIZERS_PARALLELISM=false
# ✅ NEW: redirect internal code's rsync path
export SCRATCH_PREFIX=/scratch/user/ps41/ws/nlp/reproduce/MAD_TSC/scratch/data

export DATA_SCRATCH=$SCRATCH_PREFIX
export DATA_DIR=$DATA_SCRATCH
export EXPERIMENT_DIR=/scratch/user/ps41/ws/nlp/reproduce/MAD_TSC/experiments
export EXPERIMENT_SCRATCH=/scratch/user/ps41/ws/nlp/reproduce/MAD_TSC/scratch/io/$UUID

mkdir -p $DATA_SCRATCH
mkdir -p "$DATA_SCRATCH/$UUID/MAD_TSC_hi"
mkdir -p $EXPERIMENT_DIR
mkdir -p $EXPERIMENT_SCRATCH

## Copy original English dataset to scratch
# rsync -avh --progress ./data/MAD_TSC/original/en/ "/scratch/data/"$UUID"/"
rsync -avh ./data/MAD_TSC/original/hi/ "$DATA_SCRATCH/$UUID/MAD_TSC_hi/"

# === Log timing and status ===
start_time=$(date +%s)
echo "Starting test.py at $(date)..."
echo "[INFO] Training with SPC on Hindi"
cat configs/models/spc_model_default.json || echo "[WARN] Model config not found!"

## Launch fine-tuning
tscbench finetune tsc \
-n TG_hindi_spc_run1 \
-m google/muril-base-cased \
-t google/muril-base-cased \
--tsc-model-config configs/models/spc_model_default.json \
--dataset-config configs/datasets/MAD_TSC_hi.json \
--gpu-pl-config configs/lightning/default_lightning_config.json \
--sub-path-final-folder run_outputs/hi_spc \
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