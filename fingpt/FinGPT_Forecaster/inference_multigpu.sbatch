#!/bin/bash
#SBATCH --job-name=Llama3_Inference_MultiGPU
#SBATCH --account=rrg-bengioy-ad_gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:h100:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

module load apptainer/1.4.5
module load cudacore/.11.8.0

FINGPT_DIR="/home/riaarora/scratch/FinGPT/fingpt/FinGPT_Forecaster"
HF_CACHE_DIR="$SLURM_TMPDIR/hf_cache"
mkdir -p "$HF_CACHE_DIR"

HF_TOKEN="${HF_TOKEN:?HF_TOKEN env var must be set}"

apptainer exec \
  --nv \
  --bind "$FINGPT_DIR":/mnt \
  --bind "$HF_CACHE_DIR":/hf_cache \
  --env HF_TOKEN="$HF_TOKEN" \
  --env HF_HOME=/hf_cache \
  --env TRANSFORMERS_CACHE=/hf_cache/hub \
  --env NUM_GPUS=4 \
  --env SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt \
  --env REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt \
  /home/riaarora/scratch/fingpt_forecaster_l3 \
  bash /mnt/run_inference_multigpu.sh \
    --base_model llama3 \
    --dataset dow30-202305-202405 \
    --run_name inference-eval-llama3-base-mask-numbers \
    --mask_numbers \
    --wandb

rm -rf "$HF_CACHE_DIR"
