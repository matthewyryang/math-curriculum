# LOCAL_DIR=/project/flame/asetlur/checkpoints/math-curriculum/Math/8klen-qwen3-easy-crh0.5l0.2-ent0.002/global_step_100/actor/hf-format
# LOCAL_DIR=/project/flame/asetlur/checkpoints/math-curriculum/Math/16klen-qwen3easy8k-medium2500-medium2500set2/global_step_25/actor


# LOCAL_DIR=/project/flame/asetlur/checkpoints/math-curriculum/Math/16klen-qwen3easy8k-medium2500-b64mb32n32-crh0.35l0.2/global_step_60/actor/hf-format
# TARGET_DIR=CMU-AIRe/e3-1.7B

LOCAL_DIR=/project/flame/asetlur/data/hmmt_and_aime2025.parquet
TARGET_DIR=CMU-AIRe/hmmt-aime-2025

python scripts/model_merger.py --backend upload_only \
    --hf_model_path Qwen/Qwen3-1.7B \
    --local_dir $LOCAL_DIR \
    --hf_upload_path $TARGET_DIR