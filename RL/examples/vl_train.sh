#!/bin/bash
# ============================================================
# Pure VL (Vision-Language) RL training script — NO Monet/latent patch.
# Supports any standard Qwen2.5-VL (or compatible) checkpoint.
#
# Usage:
#   bash examples/vl_train.sh
# ============================================================

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
set -x

# ---------- User settings ----------
MODEL_PATH=path/to/your/Qwen2.5-VL-7B-Instruct   # HF or local checkpoint
TRAIN_DATA=path/to/your/train_dataset              # parquet dir, file, or "hf_id@split"
VAL_DATA=path/to/your/val_dataset

ROLLOUT_N=8
TEMPERATURE=1.0
GPU_UTILIZATION=0.85
KL_COEF=0.01
ORI_BSZ=64
ONLINE_ACCUM_SIZE=256
TRAIN_MAX_SAMPLES=-1
VAL_MAX_SAMPLES=-1
N_GPUS_PER_NODE=8
TENSOR_PARALLEL_SIZE=1
MAX_PROMPT_LENGTH=4096
MAX_RESPONSE_LENGTH=4096

# Wandb / API keys
export WANDB_API_KEY=your_wandb_api_key
# Only needed if using gemini API judge; leave empty to disable API judge
export GEMINI_API_KEY=your_gemini_api_key

# ---------- Infrastructure ----------
export PYTHONUNBUFFERED=1
unset LD_PRELOAD
unset NCCL_TOPO_FILE
export NCCL_IB_DISABLE=1
export NCCL_DEBUG=WARN
export RAY_WORKER_REGISTER_TIMEOUT_SECONDS=300
export VLLM_NO_USAGE_STATS=1
export VLLM_USE_V1=1           # use vLLM v1 engine (standard, no Monet hook)
export RAY_USAGE_STATS_ENABLED=0
export RAY_DISABLE_DASHBOARD=1
export RAY_DASHBOARD_ENABLED=0
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export RAY_NUM_CPUS=16
export RAY_NUM_GPUS=${N_GPUS_PER_NODE}
export USE_RAY_LOCAL=1
export RAY_ADDRESS=local
export RAY_METRICS_EXPORT_PORT=0
export RAY_LOG_TO_STDERR=0
export RAY_LOCAL_MODE=0
export RAY_task_exit_on_oom=1
export RAY_SPILL_DIR="${RAY_SPILL_DIR:-${PWD}/ray_spill}"
export RAY_TMPDIR="${RAY_TMPDIR:-${PWD}/ray_tmp}"
mkdir -p "$RAY_SPILL_DIR" "$RAY_TMPDIR"
export RAY_OBJECT_STORE_MEMORY=134217728

# ---------- Launch ----------
# Note: we call verl.trainer.vl_main (not verl.trainer.main) so that the
# Monet/latent patch is never loaded.
cd "$(dirname "$0")/.."   # cd to RL/ root

python -m verl.trainer.vl_main \
    config=examples/config_vl.yaml \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    worker.actor.model.model_path="${MODEL_PATH}" \
    trainer.experiment_name="vl_grpo_n${ROLLOUT_N}_temp${TEMPERATURE}" \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE} \
    worker.rollout.tensor_parallel_size=${TENSOR_PARALLEL_SIZE} \
    worker.actor.fsdp.torch_dtype=bf16 \
    worker.actor.optim.strategy=adamw_bf16 \
    worker.rollout.n=${ROLLOUT_N} \
    worker.rollout.temperature=${TEMPERATURE} \
    worker.rollout.gpu_memory_utilization=${GPU_UTILIZATION} \
    worker.rollout.enable_chunked_prefill=true \
    worker.rollout.sampling_strategy=greedy \
    worker.rollout.max_num_seqs=128 \
    worker.reward.reward_function=./examples/reward_function/vl_reward_function.py:compute_score \
    worker.rule_based_judge.judge_function=./examples/reward_function/vl_reward_function.py:rule_then_api_batch_judge \
    worker.rule_based_judge.api_name="gemini-2.5-pro" \
    algorithm.kl_coef=${KL_COEF} \
    data.rollout_batch_size=${ORI_BSZ} \
    data.online_accum_size=${ONLINE_ACCUM_SIZE} \
    data.dataloader_num_workers=8 \
    data.train_max_samples=${TRAIN_MAX_SAMPLES} \
    data.val_max_samples=${VAL_MAX_SAMPLES} \
    data.max_prompt_length=${MAX_PROMPT_LENGTH} \
    data.max_response_length=${MAX_RESPONSE_LENGTH} \
    data.filter_overlong_and_invalid_prompts=false
