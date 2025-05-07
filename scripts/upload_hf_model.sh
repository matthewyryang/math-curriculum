eval "$(conda shell.bash hook)"
conda activate verl

# Set environment variables
hf_cache_dir="/home/anikait.singh/.cache"
export WANDB_API_KEY=a393f29dee9351c0a8c4e410e626e20733564d26
export WANDB_USERNAME=gurpreetkaur94539
export WANDB_USER_EMAIL=gurpreetkaur94539gmail.com
export WANDB__SERVICE_WAIT=300
# export WANDB_ENTITY=cocolab
export HF_DATASETS_CACHE=$hf_cache_dir
export HF_TOKEN='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'

all_local_dirs=(
    '/home/anikait.singh/rl_behaviors_verl_stable/ppo/insight-grpo-sft1e5-bsz64-maxlen2k-2epoch/global_step_60/actor'
)
num_local_dirs=${#all_local_dirs[@]}

all_target_dirs=(
    'Asap7772/insight-grpo-step60'
)
num_target_dirs=${#all_target_dirs[@]}
all_hf_model_paths=(
    'Qwen/Qwen2.5-3B'
)
num_hf_model_paths=${#all_hf_model_paths[@]}

if [ $num_local_dirs -ne $num_target_dirs ]; then
    echo "Number of local directories and target directories do not match"
    exit 1
fi

if [ $num_local_dirs -ne $num_hf_model_paths ]; then
    echo "Number of local directories and hf model paths do not match"
    exit 1
fi

exp_num=0
dry_run=false
which_exp=${1:--1}
if [ $which_exp -eq -1 ]; then
    echo "Running all experiments" 
fi

for i in $(seq 0 $((num_local_dirs - 1))); do
    LOCAL_DIR=${all_local_dirs[$i]}
    TARGET_DIR=${all_target_dirs[$i]}
    hf_model_path=${all_hf_model_paths[$i]}

    command="python /home/anikait.singh/verl-stable/scripts/model_merger.py --backend fsdp \
        --hf_model_path $hf_model_path \
        --local_dir $LOCAL_DIR \
        --hf_upload_path $TARGET_DIR"
    echo $command

    command2="python /home/anikait.singh/verl-stable/scripts/save_tokenizer.py \
        --hf_model_path $hf_model_path \
        --hf_upload_path $TARGET_DIR"
    echo $command2
    
    if [ $dry_run = true ]; then
        echo -e "Dry run. Skipping...\n\n"
    else
        eval ${command} &
        eval ${command2} &
    fi

    exp_num=$((exp_num+1))
done
wait
