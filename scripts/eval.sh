#!/bin/bash

export EXPERIMENT_NAME="eval16k-8klen-qwen3-easy-crh0.5l0.2-ent0.002-medium"
# export EXPERIMENT_NAME="eval32k-12klen-qwen3base-easymed-crh0.5l0.2-ent0.002-aime"

# export EXPERIMENT_NAME="eval24k-grpo16k-grpo8k-q1.5sft16k-dapo-17k"
# export MODEL_PATH="/project/flame/asetlur/math-sft-openthoughts-qwenformat-maxlen16k/global_step_2518" # SFT model
# export MODEL_PATH="/project/flame/asetlur/checkpoints/math-curriculum/Math/8klen-q1.5sft16k-cr0.3-dualclip-bs32/global_step_300/actor/hf-format" # grpo 8k model
# export MODEL_PATH="/project/flame/asetlur/checkpoints/math-curriculum/Math/16klen-q1.5sft16k-grpo8k-cr0.5-dualclip-bs32_global_step_200/hf-format"
# export MODEL_PATH="/project/flame/asetlur/hub/models--agentica-org--DeepScaleR-1.5B-Preview/snapshots/e3f524ce413a296b4d388e7560dd5c82c1c56725"
# export MODEL_PATH="/project/flame/asetlur/sft24k-on-q1.5sft16k-grpo8k-grpo16k/global_step_365"

# export MODEL_PATH="/project/flame/asetlur/hub/models--Qwen--Qwen3-1.7B/snapshots/d3e258980a49b060055ea9038dad99d75923f7c4"

# export MODEL_PATH="/project/flame/asetlur/checkpoints/math-curriculum/Math/8klen-qwen3-easymed-ent0.002-crh0.5l0.2/global_step_100/actor/hf-format"
# export MODEL_PATH="/project/flame/asetlur/checkpoints/math-curriculum/Math/8klen-qwen3--ent0.002-crh0.5l0.2/global_step_100/actor/hf-format"

# export MODEL_PATH="/project/flame/asetlur/checkpoints/math-curriculum/Math/16klen-qwen3-medhard-crh0.5/global_step_100/actor/hf-format"
export MODEL_PATH="/project/flame/asetlur/checkpoints/math-curriculum/Math/8klen-qwen3-easy-crh0.5l0.2-ent0.002/global_step_100/actor/hf-format"

# export MODEL_PATH="/project/flame/asetlur/checkpoints/math-curriculum/Math/16klen-qwen3-medhard-crh0.5-minibs64/global_step_100/actor/hf-format"

# export MODEL_PATH="/project/flame/asetlur/checkpoints/math-curriculum/Math/12klen-qwen3base-easymed-crh0.5l0.2-ent0.002/global_step_100/actor/hf-format"

# source /home/asetlur/miniconda3/bin/activate verl 
# python3 -m verl.trainer.main_ppo \
#     algorithm.adv_estimator=grpo \
#     data.train_files=/project/flame/asetlur/data/hmmt.parquet \
#     data.val_files=/project/flame/asetlur/data/hmmt.parquet \
#     data.train_batch_size=30 \
#     data.max_prompt_length=1024 \
#     data.max_response_length=32768 \
#     data.max_extrapolation_length=32768 \
#     data.filter_overlong_prompts=True \
#     actor_rollout_ref.model.path=$MODEL_PATH \
#     actor_rollout_ref.actor.optim.lr=1e-6 \
#     actor_rollout_ref.model.use_remove_padding=True \
#     actor_rollout_ref.actor.clip_ratio=0.3 \
#     actor_rollout_ref.actor.ppo_mini_batch_size=32 \
#     actor_rollout_ref.actor.ppo_micro_batch_size=32 \
#     actor_rollout_ref.actor.use_dynamic_bsz=True \
#     actor_rollout_ref.actor.only_train_on_positive=False \
#     actor_rollout_ref.actor.remove_truncated=False \
#     actor_rollout_ref.actor.ppo_max_token_len_per_gpu=50000 \
#     actor_rollout_ref.rollout.max_num_batched_tokens=50000 \
#     actor_rollout_ref.actor.use_kl_loss=True \
#     actor_rollout_ref.actor.kl_loss_coef=0.001 \
#     actor_rollout_ref.actor.kl_loss_type=low_var_kl \
#     actor_rollout_ref.actor.entropy_coeff=0.001 \
#     actor_rollout_ref.model.enable_gradient_checkpointing=True \
#     actor_rollout_ref.actor.fsdp_config.param_offload=False \
#     actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
#     actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
#     actor_rollout_ref.rollout.name=vllm \
#     actor_rollout_ref.rollout.temperature=0.6 \
#     actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
#     actor_rollout_ref.rollout.val_kwargs.do_sample=True \
#     actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
#     actor_rollout_ref.rollout.n=8 \
#     actor_rollout_ref.rollout.val_kwargs.n=8 \
#     actor_rollout_ref.ref.fsdp_config.param_offload=True \
#     actor_rollout_ref.rollout.enforce_eager=False \
#     actor_rollout_ref.rollout.free_cache_engine=False \
#     algorithm.use_kl_in_reward=False \
#     custom_reward_function.path=verl/utils/reward_score/curriculum_math/compute_score.py \
#     trainer.extrapolation_val=False \
#     trainer.val_only=True \
#     trainer.critic_warmup=0 \
#     trainer.logger=['console','wandb'] \
#     trainer.project_name=Math \
#     trainer.experiment_name=$EXPERIMENT_NAME \
#     trainer.val_before_train=True \
#     trainer.n_gpus_per_node=8 \
#     trainer.nnodes=1 \
#     trainer.save_freq=100 \
#     trainer.test_freq=25 \
#     trainer.total_training_steps=501 \
#     trainer.total_epochs=2 "${@:1}" > /home/asetlur/math-curriculum/logs/$EXPERIMENT_NAME.log 2>&1




source /home/asetlur/miniconda3/bin/activate verl 
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/project/flame/asetlur/data/medium.parquet \
    data.val_files=/project/flame/asetlur/data/medium.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=16384 \
    data.max_extrapolation_length=16384 \
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
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
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
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=25 \
    trainer.total_training_steps=501 \
    trainer.total_epochs=2 "${@:1}" > /home/asetlur/math-curriculum/logs/$EXPERIMENT_NAME.log 2>&1