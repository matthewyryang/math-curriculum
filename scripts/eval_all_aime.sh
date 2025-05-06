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

model_names=(
    /home/anikait.singh/rl_behaviors_verl_stable/sft/openthoughts_100k_sft_qwen3_1.7b_lr1e5/global_step_7810
    /home/anikait.singh/rl_behaviors_verl_stable/sft/openthoughts_100k_sft_qwen3_1.7b_lr5e5/global_step_7810
    /home/anikait.singh/rl_behaviors_verl_stable/sft/openthoughts_100k_sft_qwen3_1.7b_lr1e6/global_step_7810
    /home/anikait.singh/rl_behaviors_verl_stable/sft/openthoughts_100k_sft_qwen3_1.7b_lr5e6/global_step_7810
    /home/anikait.singh/rl_behaviors_verl_stable/sft/openthoughts_100k_sft_qwen3_1.7b_lr1e7/global_step_7810
    /home/anikait.singh/rl_behaviors_verl_stable/sft/openthoughts_100k_sft_qwen3_1.7b_lr5e7/global_step_7810
)
num_model_names=${#model_names[@]}

output_names=(
    aime_2025_responses_openthoughts-sft-qwen3-1.7b-lr1e5
    aime_2025_responses_openthoughts-sft-qwen3-1.7b-lr5e5
    aime_2025_responses_openthoughts-sft-qwen3-1.7b-lr1e6
    aime_2025_responses_openthoughts-sft-qwen3-1.7b-lr5e6
    aime_2025_responses_openthoughts-sft-qwen3-1.7b-lr1e7
    aime_2025_responses_openthoughts-sft-qwen3-1.7b-lr5e7
)
num_output_names=${#output_names[@]}

gpus=(
    0
    1
    2
    3
    4
    5
)
num_gpus=${#gpus[@]}

if [ $num_model_names -ne $num_output_names ]; then
    echo "Number of model names and output names should be the same"
    exit 1
fi

if [ $num_model_names -ne $num_gpus ]; then
    echo "Number of model names and gpus should be the same"
    exit 1
fi

exp_num=0
dry_run=0
which_exp=${1:--1}
if [ $which_exp -eq -1 ]; then
    echo "Running all experiments" 
fi

for i in $(seq 0 $((num_model_names - 1))); do
    if [ $which_exp -ne -1 ] && [ $exp_num -ne $which_exp ]; then
        exp_num=$((exp_num+1))
        continue
    fi

    model_name=${model_names[$i]}
    output_name=${output_names[$i]}
    gpu=${gpus[$i]}
    export CUDA_VISIBLE_DEVICES=$gpu
    echo "Evaluating $model_name on gpu $gpu with output name $output_name"
    command="python /home/anikait.singh/verl-stable/scripts/eval_aime.py --model_name $model_name --output_name $output_name"
    echo $command
    if [ $dry_run -eq 0 ]; then
        eval $command &
    fi
    exp_num=$((exp_num+1))
done
wait