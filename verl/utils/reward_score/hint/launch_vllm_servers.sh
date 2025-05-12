#!/bin/bash
eval "$(conda shell.bash hook)"
conda activate verl

# Set environment variables
hf_cache_dir="/home/anikait.singh/.cache/"
export WANDB_API_KEY=a393f29dee9351c0a8c4e410e626e20733564d26
export WANDB_USERNAME=gurpreetkaur94539
export WANDB_USER_EMAIL=gurpreetkaur94539@gmail.com
export WANDB__SERVICE_WAIT=300
# export WANDB_ENTITY=cocolab
export HF_DATASETS_CACHE=$hf_cache_dir
export HF_TOKEN='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'

set -e  # Exit immediately if a command exits with a non-zero status

# Array of ports to use
PORTS=(10000 10001 10002 10003)
gpus=(4 5 6 7)

# Launch 4 servers on different ports and GPUs
for i in "${!PORTS[@]}"; do
    port=${PORTS[$i]}
    gpu=${gpus[$i]}
    
    echo "Starting server on port $port using GPU $gpu"
    CUDA_VISIBLE_DEVICES=$gpu vllm serve Qwen/Qwen3-4B \
        --dtype auto \
        --port $port &
    echo "Started server on port $port using GPU $gpu"
done

# Wait for all background processes
wait 