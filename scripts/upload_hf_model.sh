# LOCAL_DIR=/project/flame/asetlur/checkpoints/math-curriculum/Math/8klen-qwen3-easy-crh0.5l0.2-ent0.002/global_step_100/actor/hf-format
LOCAL_DIR=/project/flame/asetlur/checkpoints/math-curriculum/Math/16klen-qwen3easy8k-medium2500-medium2500set2/global_step_25/actor
TARGET_DIR=d1shs0ap/easy-8k-med16k

python scripts/model_merger.py --backend fsdp \
    --hf_model_path Qwen/Qwen3-1.7B \
    --local_dir $LOCAL_DIR \
    --hf_upload_path $TARGET_DIR