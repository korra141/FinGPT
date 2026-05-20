#!/bin/bash
# Inner script executed inside the Apptainer container for multi-GPU training.
# Launched via torchrun so HuggingFace Trainer uses DDP automatically.

# --- Environment ---
export NCCL_IGNORE_DISABLED_P2P=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=0
export DS_BUILD_OPS=0
export DS_BUILD_SPARSE_ATTN=0
export DEEPSPEED_DISABLE_FUSED_ADAM=1

# HF cache is bound to /hf_cache by the SLURM script
export HF_HOME=/hf_cache
export TRANSFORMERS_CACHE=/hf_cache/hub

# Number of GPUs is passed in from the SLURM script; default to 4
NUM_GPUS=${NUM_GPUS:-4}

cd /mnt

# torchrun sets LOCAL_RANK automatically for each worker process.
# train_lora.py reads LOCAL_RANK to select the correct GPU.
torchrun \
  --standalone \
  --nnodes=1 \
  --nproc_per_node="$NUM_GPUS" \
  train_lora.py \
  --run_name dow30v3-llama3-1e-5lr-multigpu \
  --base_model llama3 \
  --dataset dow30-202305-202405 \
  --max_length 4096 \
  --batch_size 1 \
  --gradient_accumulation_steps 4 \
  --learning_rate 5e-5 \
  --num_epochs 5 \
  --log_interval 10 \
  --warmup_ratio 0.03 \
  --scheduler constant \
  --evaluation_strategy steps \
  --ds_config config.json
