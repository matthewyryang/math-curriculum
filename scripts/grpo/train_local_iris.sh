eval "$(conda shell.bash hook)"
conda activate verl

# Set environment variables
hf_cache_dir="/iris/u/asap7772/.cache"
export WANDB_API_KEY=a393f29dee9351c0a8c4e410e626e20733564d26
export WANDB_USERNAME=gurpreetkaur94539
export WANDB_USER_EMAIL=gurpreetkaur94539gmail.com
export WANDB__SERVICE_WAIT=300
# export WANDB_ENTITY=cocolab
export HF_DATASETS_CACHE=$hf_cache_dir
export HF_TOKEN='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'

models=(
    Qwen/Qwen3-1.7B
    Qwen/Qwen3-1.7B
)
num_models=${#models[@]}
names=(
    qwen3-1.7b-hintsolgen-mixtrue-d1shs0ap-easy-chatfix
    qwen3-1.7b-hintsolgen-mixtrue-d1shs0ap-easy-chatfix-zerorew
)
num_names=${#names[@]}

train_data_dirs=(
    "/iris/u/asap7772/rl_behaviors_verl_stable/data_d1shs0ap-easy-mixTrue-nochat"
    "/iris/u/asap7772/rl_behaviors_verl_stable/data_d1shs0ap-easy-mixTrue-nochat"
)
num_train_data_dirs=${#train_data_dirs[@]}

eval_data_dirs=(
    "/iris/u/asap7772/rl_behaviors_verl_stable/data_d1shs0ap-easy-mixTrue-nochat"
    "/iris/u/asap7772/rl_behaviors_verl_stable/data_d1shs0ap-easy-mixTrue-nochat"
)
num_eval_data_dirs=${#eval_data_dirs[@]}

gpus=(
    "0,1,2,3,4,5,6,7"
    "0,1,2,3,4,5,6,7"
)
num_gpus=${#gpus[@]}

project_names=(
    grpo_qwen3_hintsolgen_d1shs0ap_easy_chatfix_0510
    grpo_qwen3_hintsolgen_d1shs0ap_easy_chatfix_0510
)
num_project_names=${#project_names[@]}

commands=(
    "bash /iris/u/asap7772/verl-stable/scripts/grpo/grpo_run_dualclip_iris.sh"
    "bash /iris/u/asap7772/verl-stable/scripts/grpo/grpo_run_dualclip_iris_zerorew.sh"
)

if [ $num_models -ne $num_names ]; then
    echo "Number of models and names should be the same"
    exit 1
fi

if [ $num_models -ne $num_gpus ]; then
    echo "Number of models and gpus should be the same"
    exit 1
fi

if [ $num_models -ne $num_train_data_dirs ]; then
    echo "Number of models and data directories should be the same"
    exit 1
fi

if [ $num_models -ne $num_eval_data_dirs ]; then
    echo "Number of models and eval data directories should be the same"
    exit 1
fi

if [ $num_models -ne $num_project_names ]; then
    echo "Number of models and project names should be the same"
    exit 1
fi

exp_num=0
dry_run=false
which_exp=${1:--1}
if [ $which_exp -eq -1 ]; then
    echo "Running all experiments" 
fi

for i in $(seq 0 $((num_models-1))); do
    if [ $which_exp -ne -1 ] && [ $exp_num -ne $which_exp ]; then
        exp_num=$((exp_num+1))
        continue
    fi

    curr_train_data_dir=${train_data_dirs[$i]}
    curr_eval_data_dir=${eval_data_dirs[$i]}
    if [ ! -d $curr_train_data_dir ]; then
        echo "Data directory $curr_train_data_dir does not exist"
        exit 1
    fi
    if [ ! -d $curr_eval_data_dir ]; then
        echo "Data directory $curr_eval_data_dir does not exist"
        exit 1
    fi

    export N_GPUS=8
    export BASE_MODEL=${models[$i]}
    export TRAIN_DATA_DIR=$curr_train_data_dir
    export EVAL_DATA_DIR=$curr_eval_data_dir
    export ROLLOUT_TP_SIZE=1
    export EXPERIMENT_NAME=${names[$i]}
    # export VLLM_ATTENTION_BACKEND=XFORMERS
    export CUDA_VISIBLE_DEVICES=${gpus[$i]}
    export PROJECT_NAME=$PROJECT_NAME
    export MAX_MODEL_LEN=8192
    export MAX_PROMPT_LENGTH=1024
    export EPOCHS=30
    export PROJECT_NAME=${project_names[$i]}

    command=${commands[$i]}

    echo "Using GPU: $CUDA_VISIBLE_DEVICES"
    echo $command
    if [ $dry_run = true ]; then
        echo -e "Dry run. Skipping...\n\n"
    else
        eval $command
    fi
    
    exp_num=$((exp_num+1))
done
