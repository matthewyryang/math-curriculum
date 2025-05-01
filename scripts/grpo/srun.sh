export CONTEXT_LENGTH=2048
# export BASE_MODEL=Qwen/Qwen2.5-Math-1.5B
export BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

export DATA_DISTRIBUTION="train"
export EXPERIMENT_NAME="$DATA_DISTRIBUTION-$CONTEXT_LENGTH"
export EPOCHS=8

# ray stop --force && ray start --head

nohup bash /home/cmu/math-curriculum/scripts/grpo/grpo.sh > /home/cmu/math-curriculum/logs/$EXPERIMENT_NAME.log 2>&1 &
