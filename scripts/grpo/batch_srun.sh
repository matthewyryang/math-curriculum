
export CONTEXT_LENGTH=2048
export BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B


# export DATA_DISTRIBUTION="train"
# export EXPERIMENT_NAME="$DATA_DISTRIBUTION-$CONTEXT_LENGTH"
# export EPOCHS=2

# nohup bash /home/cmu/math-curriculum/scripts/grpo/grpo.sh > /home/cmu/math-curriculum/logs/$EXPERIMENT_NAME.log 2>&1


for dist in "train-3-5" "train-1-5"; do
    export DATA_DISTRIBUTION=$dist
    export EXPERIMENT_NAME="$DATA_DISTRIBUTION-$CONTEXT_LENGTH"
    export EPOCHS=6
    
    nohup bash /home/cmu/math-curriculum/scripts/grpo/grpo.sh > /home/cmu/math-curriculum/logs/$EXPERIMENT_NAME.log 2>&1
done