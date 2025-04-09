export DATA_DISTRIBUTION="medium"
export CONTEXT_LENGTH=8192
# export BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B
export BASE_MODEL=/home/cmu/math-curriculum/checkpoints/Math/easy-4096/global_step_200/hf

export EXPERIMENT_NAME="$DATA_DISTRIBUTION-$CONTEXT_LENGTH-from-easy-4096"

# ray stop --force && ray start --head

nohup bash /home/cmu/math-curriculum/scripts/grpo/grpo_8k.sh > /home/cmu/math-curriculum/logs/$EXPERIMENT_NAME.log 2>&1 &
