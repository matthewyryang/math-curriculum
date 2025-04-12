export CONTEXT_LENGTH=2048
export BASE_MODEL=Qwen/Qwen2.5-Math-1.5B

export DATA_DISTRIBUTION="train-3-5"
export EXPERIMENT_NAME="$DATA_DISTRIBUTION-$CONTEXT_LENGTH"
export EPOCHS=8

# ray stop --force && ray start --head

nohup bash /home/cmu/math-curriculum/scripts/grpo/grpo.sh > /home/cmu/math-curriculum/logs/$EXPERIMENT_NAME.log 2>&1 &
