#!/bin/bash

set -e  # Exit immediately if a command exits with a non-zero status

# Shared training parameters
prompt_key="prompt"
response_key="completion"
micro_batch_size=8
micro_batch_size_per_gpu=1
train_batch_size=64
max_length=3100
model_name="Qwen/Qwen2.5-Math-1.5B"
experiment_name="math-sft-openthoughts-maxlen3k"
default_hdfs_dir="/project/flame/asetlur/${experiment_name}"
default_local_dir="/project/flame/asetlur/${experiment_name}"
# Create the default local directory if it doesn't exist
mkdir -p "${default_local_dir}"
# base_data_path="/project/flame/asetlur/data/OpenThoughts-114k-r1-format"
base_data_path="/project/flame/asetlur/data/OpenThoughts-114k-r1-format-maxlen3k/"
project_name="math-sft"
total_epochs=3
logger="['console','wandb']"
truncation="right"
apply_chat_template=False

# Iterate over each condition and launch a training job
train_file="${base_data_path}/train.parquet"
val_file="${base_data_path}/test.parquet"
save_dir="${default_local_dir}"

echo "Train file: ${train_file}"
echo "Val file:   ${val_file}"
echo "Experiment name: ${experiment_name}"
echo ""

torchrun --nproc_per_node=8 -m verl.trainer.fsdp_sft_trainer \
  data.train_files="${train_file}" \
  data.val_files="${val_file}" \
  data.prompt_key="${prompt_key}" \
  data.truncation="${truncation}" \
  data.apply_chat_template="${apply_chat_template}" \
  data.response_key="${response_key}" \
  data.micro_batch_size="${micro_batch_size}" \
  data.micro_batch_size_per_gpu="${micro_batch_size_per_gpu}" \
  data.train_batch_size="${train_batch_size}" \
  data.max_length="${max_length}" \
  model.partial_pretrain="${model_name}" \
  trainer.default_hdfs_dir="${default_hdfs_dir}" \
  trainer.default_local_dir="${save_dir}" \
  trainer.project_name="${project_name}" \
  trainer.experiment_name="${experiment_name}" \
  trainer.total_epochs="${total_epochs}" \
  trainer.logger="${logger}" "${@:1}" > /home/asetlur/math-curriculum/logs/$experiment_name.log 2>&1

echo "--------------------------------------------------"
