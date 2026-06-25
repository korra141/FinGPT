#!/bin/bash
# Inner script executed inside the Apptainer container for multi-node, multi-GPU training.
# Launched via srun from the SLURM script — torchrun coordinates ranks across nodes
# using c10d rendezvous at MASTER_ADDR:MASTER_PORT set by the SLURM script.

# --- Environment ---
export NCCL_IGNORE_DISABLED_P2P=1
export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=DETAIL
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=0
export DS_BUILD_OPS=0
export DS_BUILD_SPARSE_ATTN=0
export DEEPSPEED_DISABLE_FUSED_ADAM=1

# HF cache is bound to /hf_cache by the SLURM script
export HF_HOME=/hf_cache
export TRANSFORMERS_CACHE=/hf_cache/hub

# Injected by the SLURM script
NUM_GPUS=${NUM_GPUS:-4}
NNODES=${NNODES:-1}
MASTER_ADDR=${MASTER_ADDR:-localhost}
MASTER_PORT=${MASTER_PORT:-29500}
NODE_RANK=${NODE_RANK:-0}

cd /mnt

torchrun \
  --nnodes="$NNODES" \
  --nproc_per_node="$NUM_GPUS" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  --node_rank="$NODE_RANK" \
  train_lora_cot.py \
  --run_name dow30-llama2-cot-4bit\
  --base_model llama2 \
  --cot_dataset cot_dow_dataset \
  --dataset dow30-202305-202405 \
  --max_length 8192 \
  --load_in_4bit \
  --batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --num_epochs 10 \
  --log_interval 10 \
  --warmup_ratio 0.03 \
  --scheduler constant \
  --evaluation_strategy steps \
  --gen_eval_batch_size 8 \
  --ul_weight 0.2 \
  --resume_from_checkpoint finetuned_models/dow30-llama2-cot-4bit-l3u45tzy_202606241239 \
  --ds_config config_zero3_fp16.json
