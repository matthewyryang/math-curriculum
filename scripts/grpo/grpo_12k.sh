#!/bin/bash

# export EXPERIMENT_NAME="8klen-sftqwen2.5-easymed-ent0.002-crh0.5l0.2"
export EXPERIMENT_NAME="12klen-qwen3base-easymed-crh0.5l0.2-ent0.002"

# export MODEL_PATH="/project/flame/asetlur/math-sft-openthoughts-qwenformat-maxlen16k/global_step_2518"
# export MODEL_PATH="/project/flame/asetlur/math-sft-openthoughts-qwenformat-maxlen2k/global_step_630"
# export MODEL_PATH="/project/flame/asetlur/math-sft-openthoughts-maxlen8k-on-qwenr1distill/global_step_1092"
# export MODEL_PATH="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

export MODEL_PATH="/project/flame/asetlur/hub/models--Qwen--Qwen3-1.7B/snapshots/d3e258980a49b060055ea9038dad99d75923f7c4"

source /home/asetlur/miniconda3/bin/activate verl 
python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=/project/flame/asetlur/data/easy_medium.parquet \
    data.val_files=/project/flame/asetlur/data/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=1024 \
    data.max_response_length=12288 \
    data.max_extrapolation_length=16384 \
    data.filter_overlong_prompts=True \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.5 \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size=32 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.only_train_on_positive=False \
    actor_rollout_ref.actor.remove_truncated=False \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.002 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=verl/utils/reward_score/curriculum_math/compute_score.py \
    trainer.extrapolation_val=False \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name=Math \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.val_before_train=True \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=100 \
    trainer.test_freq=10 \
    trainer.total_training_steps=501 \
    trainer.total_epochs=2 "${@:1}" > /home/asetlur/math-curriculum/logs/$EXPERIMENT_NAME.log 2>&1