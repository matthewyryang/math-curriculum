#!/bin/bash

export EXPERIMENT_NAME="eval32k-16klen-qwen3easy8k-medium2500-b64mb32n32-crh0.35l0.2-ckpt60-hmmtaime25"

# export MODEL_PATH="/project/flame/asetlur/checkpoints/math-curriculum/Math/16klen-qwen3easy8k-medium-b128mb64n16/global_step_60/actor/hf-format"
# export MODEL_PATH="/project/flame/asetlur/hub/models--Qwen--Qwen3-1.7B/snapshots/d3e258980a49b060055ea9038dad99d75923f7c4"
# export MODEL_PATH="/project/flame/asetlur/checkpoints/math-curriculum/Math/8klen-qwen3-easy-crh0.5l0.2-ent0.002/global_step_100/actor/hf-format"
# export MODEL_PATH='/project/flame/asetlur/hub/models--d1shs0ap--easy-8k-med16k/snapshots/d5098716ffe5f300df84e8bfb315f83a2c502009'
# export MODEL_PATH='/project/flame/asetlur/checkpoints/math-curriculum/Math/16klen-qwen3easy8k-medium2500-b128mb64n16/global_step_120/actor/hf-format'
# export MODEL_PATH="/project/flame/asetlur/checkpoints/math-curriculum/Math/8klen-qwen3base-easy-b128mb64n8-crh0.35l0.2/global_step_120/actor/hf-format"
# export MODEL_PATH="/project/flame/asetlur/checkpoints/math-curriculum/Math/8klen-qwen3base-easy-b128mb64n32-crh0.5l0.2/global_step_90/actor/hf-format"
# export MODEL_PATH="/project/flame/asetlur/checkpoints/math-curriculum/Math/8klen-qwen3base-easy-b128mb64n32-crh0.5l0.2/global_step_150/actor/hf-format"
# export MODEL_PATH="/project/flame/asetlur/checkpoints/math-curriculum/Math/16klen-qwen3easy8k-medium2500-b64mb32n32-crh0.35l0.2/global_step_60/actor/hf-format"
# export MODEL_PATH="/project/flame/asetlur/checkpoints/math-curriculum/Math/16klen-qwen3easy8koldckpt-medium5000-b64mb32n32-crh0.5l0.2/global_step_90/actor/hf-format"
# export MODEL_PATH="/project/flame/asetlur/checkpoints/math-curriculum/Math/16klen-qwen3easy8koldckpt-medium5000-b64mb32n32-crh0.5l0.2/global_step_90/actor/hf-format"
# export MODEL_PATH="/project/flame/asetlur/checkpoints/math-curriculum/Math/16klen-qwen3easy8knewckpt-medium2500-b64mb32n32-crh0.35l0.2/global_step_60/actor/hf-format"
# export MODEL_PATH="/project/flame/asetlur/checkpoints/math-curriculum/Math/16klen-qwen3easy8koldckpt-medium5000-b64mb32n32-crh0.5l0.2/global_step_60/actor/hf-format"

export MODEL_PATH="/project/flame/asetlur/checkpoints/math-curriculum/Math/16klen-qwen3easy8k-medium2500-b64mb32n32-crh0.35l0.2/global_step_60/actor/hf-format"

source /home/asetlur/miniconda3/bin/activate verl 
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/project/flame/asetlur/data/hmmt_and_aime2025.parquet \
    data.val_files=/project/flame/asetlur/data/hmmt_and_aime2025.parquet \
    data.train_batch_size=30 \
    data.max_prompt_length=1024 \
    data.max_response_length=32768 \
    data.max_extrapolation_length=32768 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.clip_ratio=0.3 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.only_train_on_positive=False \
    actor_rollout_ref.actor.remove_truncated=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=50000 \
    actor_rollout_ref.rollout.max_num_batched_tokens=50000 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.001 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=256 \
    actor_rollout_ref.rollout.val_kwargs.top_k=20 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=verl/utils/reward_score/curriculum_math/compute_score.py \
    trainer.extrapolation_val=False \
    trainer.val_only=True \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=Math \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.save_freq=100 \
    trainer.test_freq=25 \
    trainer.total_training_steps=501 \
    trainer.total_epochs=2 "${@:1}" > /home/asetlur/math-curriculum/logs/$EXPERIMENT_NAME.log 2>&1