LOCAL_DIR=/home/cmu/math-curriculum/checkpoints/Math/easy-4096/global_step_200/actor
TARGET_DIR=d1shs0ap/medium-8192-from-easy-4096

python scripts/model_merger.py --backend fsdp \
    --hf_model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B \
    --local_dir $LOCAL_DIR \
    --hf_upload_path $TARGET_DIR

