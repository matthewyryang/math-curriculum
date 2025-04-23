export CUDA_VISIBLE_DEVICES=4,5,6,7
export GPUS=4

export CONTEXT_LENGTH=8192
export BASE_MODEL=Qwen/Qwen2.5-1.5B
export TEMPLATE="simple"
# export BASE_MODEL=deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B

export EXPERIMENT_NAME="$CONTEXT_LENGTH-$TEMPLATE-hkust-1.5B-batchsize-64"
export EPOCHS=20

# ray stop --force && ray start --head

nohup bash /home/cmu/math-curriculum/scripts/grpo/grpo.sh > /home/cmu/math-curriculum/logs/$EXPERIMENT_NAME.log 2>&1 &
