#!/bin/bash

export VLLM_ATTENTION_BACKEND=XFORMERS
export EXPERIMENT_NAME="8klen-SFTon16k-qwen-format-remove-truncation"
# export MODEL_PATH="/project/flame/asetlur/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
# export MODEL_PATH="/project/flame/asetlur/hub/models--Qwen--Qwen2.5-1.5B/snapshots/8faed761d45a263340a0528343f099c05c9a4323"
# export MODEL_PATH="/project/flame/asetlur/hub/models--Qwen--Qwen2.5-Math-1.5B/snapshots/4a83ca6e4526a4f2da3aa259ec36c259f66b2ab2"
# export MODEL_PATH="/project/flame/asetlur/hub/models--sail--Qwen2.5-Math-1.5B-Oat-Zero/snapshots/79fbbe3fff006979162b8ae104ab9dfd196fe5ec"
# export MODEL_PATH="/project/flame/asetlur/math-sft-openthoughts-maxlen3k/global_step_798"
# export MODEL_PATH="/project/flame/asetlur/math-sft-openthoughts-maxlen8k/global_step_192"
# export MODEL_PATH="/project/flame/asetlur/math-sft-openthoughts-maxlen8k/global_step_1092"
# export MODEL_PATH="/project/flame/asetlur/math-sft-openthoughts-maxlen8k-on-qwenr1distill/global_step_1092"
# export MODEL_PATH="/project/flame/asetlur/checkpoints/math-curriculum/Math/grpo-8k-r1template-SFT1ksteps-openthoughts8k/global_step_200/actor/hf-format"

# export MODEL_PATH="/project/flame/asetlur/checkpoints/math-curriculum/Math/grpo-4k-r1template-SFT-openthoughts2k/global_step_300/actor/hf-format"

# export MODEL_PATH="/project/flame/asetlur/hub/models--deepseek-ai--DeepSeek-R1-Distill-Qwen-7B/snapshots/916b56a44061fd5cd7d6a8fb632557ed4f724f60"
# export MODEL_PATH="/project/flame/asetlur/math-sft-openthoughts-qwenformat-maxlen2k/global_step_630"
export MODEL_PATH="/project/flame/asetlur/math-sft-openthoughts-qwenformat-maxlen16k/global_step_2518"

# kl is typically set to 0.001

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/math-curriculum/data/train.parquet \
    data.val_files=$HOME/math-curriculum/data/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.max_extrapolation_length=16384 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size=64 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.only_train_on_positive=False \
    actor_rollout_ref.actor.remove_truncated=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=verl/utils/reward_score/curriculum_math/compute_score.py \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=Math \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=25 \
    trainer.total_epochs=2 "${@:1}" > /home/asetlur/math-curriculum/logs/$EXPERIMENT_NAME.log 2>&1