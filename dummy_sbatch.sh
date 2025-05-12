#!/bin/bash
#SBATCH --partition=iris-hi # Run on IRIS nodes, optionally iris-hi
#SBATCH --time=120:00:00 # Max job length is 5 days
#SBATCH --nodes=1 # Only use one node (machine)
#SBATCH --cpus-per-task=16 # Request 16 CPUs for this task
#SBATCH --mem=128G # Request 128GB of memory
#SBATCH --gres=gpu:h200:4 # Attempt to request 2 NVIDIA A40 GPUs
#SBATCH --job-name=ppo_exps # Name the job (for easier monitoring)
#SBATCH --account=iris
#SBATCH --exclude=iris1,iris2,iris3,iris4

watch -n 0.1 nvidia-smi