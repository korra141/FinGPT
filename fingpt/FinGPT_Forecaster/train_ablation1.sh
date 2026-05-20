export NCCL_IGNORE_DISABLED_P2P=1
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:128"
export TRANSFORMERS_NO_ADVISORY_WARNINGS=1
export TOKENIZERS_PARALLELISM=0
export DS_BUILD_OPS=0
export DS_BUILD_SPARSE_ATTN=0
export DEEPSPEED_DISABLE_FUSED_ADAM=1
export HF_HOME=/mnt/.cache/huggingface
export TRANSFORMERS_CACHE=/mnt/.cache/huggingface/hub

# deepspeed \
# --include localhost:0 \

python3 train_lora.py \
--run_name dow30v3-llama3-1e-5lr \
--base_model llama3 \
--dataset dow30-202305-202405 \
--max_length 1024 \
--batch_size 1 \
--gradient_accumulation_steps 16 \
--learning_rate 5e-5 \
--num_epochs 5 \
--log_interval 10 \
--warmup_ratio 0.03 \
--scheduler constant \
--evaluation_strategy steps \
--ds_config config.json
