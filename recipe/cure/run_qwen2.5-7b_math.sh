#!/usr/bin/env bash

set -euo pipefail
set -x
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
source "${CONDA_PREFIX_ROOT:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-cure}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
# Set WANDB_API_KEY in your environment before running.

# ================= data/model =================
DATA_HOME=${DATA_HOME:-$REPO_ROOT/data}
deepmath_path=$DATA_HOME/DeepMath-103K/train_76k8.parquet
olympiad_test_path=$DATA_HOME/OlympiadBench/test.parquet

train_files="['$deepmath_path']"
test_files="['$olympiad_test_path']"

model_path=${MODEL_PATH:-Qwen/Qwen2.5-7B-Instruct}

# wandb
project_name='cure'
experiment_name='qwen2.5-7b-instruct-deepmath-cure'
default_local_dir=${CHECKPOINT_DIR:-$REPO_ROOT/checkpoints}/$project_name/$experiment_name

# ================= algorithm =================
adv_estimator=cure

use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 6))
max_response_length=$((1024 * 4))
actor_lr=1e-6
actor_warmup_steps=20

train_batch_size=128
ppo_mini_batch_size=512
n_resp_per_prompt=4
n_resp_per_prompt_val=1

# ================= performance =================
infer_tp=1 # vllm
train_sp=1 # train
offload=False

actor_max_token_len_per_gpu=$(( (max_prompt_length + max_response_length) * 1 ))
log_prob_max_token_len_per_gpu=$(( actor_max_token_len_per_gpu * 4 ))

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=$adv_estimator \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    algorithm.kl_ctrl.kl_coef=$kl_coef \
    data.train_files="$train_files" \
    data.val_files="$test_files" \
    data.train_batch_size=$train_batch_size \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=64 \
    data.truncation='error' \
    +data.critique_batch_size=256 \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.use_kl_loss=$use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$kl_loss_coef \
    actor_rollout_ref.actor.clip_ratio_low=$clip_ratio_low \
    actor_rollout_ref.actor.clip_ratio_high=$clip_ratio_high \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.optim.lr=$actor_lr \
    actor_rollout_ref.actor.optim.lr_warmup_steps=$actor_warmup_steps \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=$ppo_mini_batch_size \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=$actor_max_token_len_per_gpu \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$train_sp \
    actor_rollout_ref.actor.fsdp_config.param_offload=$offload \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$offload \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=$log_prob_max_token_len_per_gpu \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$infer_tp \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
    actor_rollout_ref.rollout.n=$n_resp_per_prompt \
    +actor_rollout_ref.rollout.critique_kwargs.n=1 \
    +actor_rollout_ref.rollout.refine_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.n=$n_resp_per_prompt_val \
    custom_reward_function.path=${SCRIPT_DIR}/reward_score.py \
    custom_reward_function.name=reward_func \
    trainer.logger=['console','wandb'] \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=4 \
    trainer.val_before_train=False \
    trainer.log_val_generations=10 \
    trainer.nnodes=1 \
    trainer.save_freq=20 \
    trainer.default_local_dir=$default_local_dir \
    trainer.test_freq=10 \
    trainer.total_epochs=1 \
    +trainer.critique=True \
    +trainer.critique_focus_on_failed_solutions=True \
    +trainer.critique_interval_start=0 \
    +trainer.critique_interval_end=0.9 \
    +trainer.critique_train_discriminability=True \
    +trainer.critique_force_error_conclusion=True \
    +trainer.critique_format_reward=True \
    +trainer.refine=True \
    +trainer.refine_only_wrong_original=True \
    +trainer.disable_refine_training=False \
    +trainer.experience_replay=True \
    +trainer.disable_solving_training=False $@
