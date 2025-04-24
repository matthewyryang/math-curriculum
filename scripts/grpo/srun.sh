export CUDA_VISIBLE_DEVICES=0,1,2,3
export GPUS=4

export CONTEXT_LENGTH=8192
export BASE_MODEL=Qwen/Qwen2.5-3B
export TEMPLATE="simple"
# export BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

export DATA_DISTRIBUTION="hard"
export EXPERIMENT_NAME="$DATA_DISTRIBUTION-$CONTEXT_LENGTH-$TEMPLATE"
export EPOCHS=8

# ray stop --force && ray start --head

nohup bash /home/cmu/math-curriculum/scripts/grpo/grpo_8k.sh > /home/cmu/math-curriculum/logs/$EXPERIMENT_NAME.log 2>&1 &
