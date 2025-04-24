
export CONTEXT_LENGTH=2048
export BASE_MODEL=Qwen/Qwen2.5-Math-1.5B

for dist in "train-1-5" "train-3-5"; do
    export DATA_DISTRIBUTION=$dist
    export EXPERIMENT_NAME="$DATA_DISTRIBUTION-$CONTEXT_LENGTH"
    export EPOCHS=6
    
    nohup bash /home/cmu/math-curriculum/scripts/grpo/grpo.sh > /home/cmu/math-curriculum/logs/$EXPERIMENT_NAME.log 2>&1
done

