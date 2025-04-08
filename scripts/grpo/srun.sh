export DATA_DISTRIBUTION="easy"
export CONTEXT_LENGTH=4096
export BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

export EXPERIMENT_NAME="$DATA_DISTRIBUTION-$CONTEXT_LENGTH"
export OUTPUT_DIR="/home/cmu/math-curriculum/checkpoints/$EXPERIMENT_NAME"

# ray stop --force && ray start --head

nohup bash /home/cmu/math-curriculum/scripts/grpo/grpo.sh > /home/cmu/math-curriculum/logs/$EXPERIMENT_NAME.log 2>&1 &
