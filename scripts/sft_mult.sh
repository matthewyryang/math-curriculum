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

# Shared training parameters
prompt_key="query"
response_key="completion"
micro_batch_size=8
micro_batch_size_per_gpu=1
train_batch_size=64

total_epochs=5
logger="['console','wandb']"
truncation="right"
apply_chat_template=False

model_names=(
  'Qwen/Qwen3-1.7B-Base'
  'Qwen/Qwen2.5-3B'
)
num_model_names=${#model_names[@]}

project_names=(
  'insight-sft-0510'
  'insight-sft-0510'
)
num_project_names=${#project_names[@]}

base_data_paths=(
  '/home/anikait.singh/rl_behaviors_verl_stable/insight-v2-sft'
  '/home/anikait.singh/rl_behaviors_verl_stable/insight-v2-sft'
)
num_base_data_paths=${#base_data_paths[@]}

experiment_names=(
  'insight-qwen3-1.7b-sft-0510'
  'insight-qwen2.5-3b-sft-0510'
)
num_experiment_names=${#experiment_names[@]}

max_lengths=(
  12288
  12288
)
num_max_lengths=${#max_lengths[@]}

lrs=(
  1e-6
  1e-6
)
num_lrs=${#lrs[@]}

if [ ${num_base_data_paths} -ne ${num_experiment_names} ]; then 
  echo "Number of base data paths and experiment names do not match"
  exit 1
fi

if [ ${num_base_data_paths} -ne ${num_project_names} ]; then
  echo "Number of base data paths and project names do not match"
  exit 1
fi

if [ ${num_base_data_paths} -ne ${num_max_lengths} ]; then
  echo "Number of base data paths and max lengths do not match"
  exit 1
fi

if [ ${num_base_data_paths} -ne ${num_lrs} ]; then
  echo "Number of base data paths and lrs do not match"
  exit 1
fi

exp_num=0
dry_run=false
which_exp=${1:--1}
if [ $which_exp -eq -1 ]; then
    echo "Running all experiments" 
fi

for i in $(seq 0 $((num_base_data_paths - 1))); do
  if [ $which_exp -ne -1 ] && [ $exp_num -ne $which_exp ]; then
    exp_num=$((exp_num+1))
    continue
  fi

  base_data_path=${base_data_paths[$i]}
  experiment_name=${experiment_names[$i]}
  project_name=${project_names[$i]}
  model_name=${model_names[$i]}
  max_length=${max_lengths[$i]}
  lr=${lrs[$i]}

  default_hdfs_dir="/home/anikait.singh/rl_behaviors_verl_stable/sft_hdfs/${experiment_name}"
  default_local_dir="/home/anikait.singh/rl_behaviors_verl_stable/sft/${experiment_name}"
  mkdir -p "${default_local_dir}"
  mkdir -p "${default_hdfs_dir}"

  # Iterate over each condition and launch a training job
  train_file="${base_data_path}/train.parquet"
  val_file="${base_data_path}/test.parquet"
  save_dir="${default_local_dir}"

  echo "Train file: ${train_file}"
  echo "Val file:   ${val_file}"
  echo "Experiment name: ${experiment_name}"
  echo "Model name: ${model_name}"
  echo "Max length: ${max_length}"
  echo "Project name: ${project_name}"
  echo "Default local dir: ${default_local_dir}"
  echo "Default hdfs dir: ${default_hdfs_dir}"
  echo "--------------------------------------------------"

  command="torchrun --nproc_per_node=8 -m verl.trainer.fsdp_sft_trainer \
    data.train_files=${train_file} \
    data.val_files=${val_file} \
    data.prompt_key=${prompt_key} \
    data.truncation=${truncation} \
    data.apply_chat_template=${apply_chat_template} \
    data.response_key=${response_key} \
    data.micro_batch_size=${micro_batch_size} \
    data.micro_batch_size_per_gpu=${micro_batch_size_per_gpu} \
    data.train_batch_size=${train_batch_size} \
    data.max_length=${max_length} \
    model.partial_pretrain=${model_name} \
    trainer.default_hdfs_dir=${default_hdfs_dir} \
    trainer.default_local_dir=${save_dir} \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.total_epochs=${total_epochs} \
    trainer.logger=${logger} \
    optim.lr=${lr} \
    model.enable_gradient_checkpointing=True \
    model.use_liger=True \
  "

  echo "--------------------------------------------------"
  echo "Running command: ${command}"
  echo "--------------------------------------------------"

  if [ $dry_run = true ]; then
    echo -e "Dry run. Skipping...\n\n"
  else
    eval ${command}
  fi

  exp_num=$((exp_num+1))
done
