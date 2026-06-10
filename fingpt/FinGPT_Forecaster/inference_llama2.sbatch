#!/bin/bash
#SBATCH --job-name=Llama2_Inference
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
module load cuda/12.1

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
  --env TRANSFORMERS_NO_ADVISORY_WARNINGS=1 \
  --env TOKENIZERS_PARALLELISM=0 \
  /home/riaarora/scratch/fingpt_forecaster_l3_v2 \
  python3 /mnt/run_inference.py \
    --peft_model /mnt/finetuned_models/dow30v3-llama2-1e-5lr-mcgjzasr_202603191536/checkpoint-342 \
    --dataset dow30-202305-202405 \
    --run_name inference-fingpt-llama2-mask-words \
    --output_dir /mnt/inference_llama2_fingpt_mask_words \
    --mask_fin_words \
    --wandb

rm -rf "$HF_CACHE_DIR"
