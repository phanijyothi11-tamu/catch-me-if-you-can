#!/bin/bash
##ENVIRONMENT SETTINGS; CHANGE WITH CAUTION
#SBATCH --export=ALL
#SBATCH --get-user-env=L
 
##NECESSARY JOB SPECIFICATIONS
#SBATCH --job-name=MAD-TSC-Evaluate   # Sets the job name to AbaqusJob
#SBATCH -A 132742115536
#SBATCH --time=10:00:00          # Sets the runtime limit to 2 hr
#SBATCH --tasks=1              # Request 1 node
#SBATCH --ntasks-per-node=1    # Requests 10 cores per node (1 node)
#SBATCH --mem=20G               # Requests 50GB of memory per node
#SBATCH --output=logs/job_eval_28April_MT_to_lang.o%J  # Sends stdout and stderr to job.o[jobID]
 
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

# === Log timing and status ===
start_time=$(date +%s)
echo "Starting test.py at $(date)..."

python -u src/tscbench/evaluation/eval_test_on_MT_en_to_target.py

exit_code=$?

end_time=$(date +%s)
duration=$((end_time - start_time))

hours=$((duration / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

printf "[%s] Duration: %02d:%02d:%02d (Exit code: %d)\n\n" \
  "$(date)" "$hours" "$minutes" "$seconds" "$exit_code"