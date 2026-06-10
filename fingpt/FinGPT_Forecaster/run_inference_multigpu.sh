#!/bin/bash
# Launcher called from inside the apptainer container.
# Expects all args to be forwarded by the sbatch script.

torchrun \
  --nproc_per_node="${NUM_GPUS:-4}" \
  --master_port=29500 \
  /mnt/run_inference_llama3.py "$@"
