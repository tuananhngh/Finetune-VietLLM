#!/usr/bin/env bash
# Launch SFT or DPO on however many GPUs this machine has.
#
#   bash scripts/train.sh <run-config.yaml> [accelerate-config.yaml] [extra overrides...]
#
#   bash scripts/train.sh configs/sft/qwen3-8b-qlora-bactrian.yaml
#   bash scripts/train.sh configs/sft/qwen3-8b-lora-mix.yaml configs/accelerate/deepspeed_zero3.yaml
#   bash scripts/train.sh configs/sft/qwen3-8b-qlora-bactrian.yaml "" --max_steps 50
#
# The task (SFT / DPO) is picked from the config path (configs/dpo/* -> DPO) or TASK=sft|dpo.
set -euo pipefail

CONFIG=${1:?usage: train.sh <config.yaml> [accelerate.yaml] [overrides...]}
ACCEL_CONFIG=${2:-}
shift $(( $# >= 2 ? 2 : 1 ))

if [[ -z "${TASK:-}" ]]; then
  [[ "$CONFIG" == *"/dpo/"* ]] && TASK=dpo || TASK=sft
fi
MODULE="vietllm.train_${TASK}"

NUM_GPUS=${NUM_GPUS:-$(command -v nvidia-smi >/dev/null && nvidia-smi -L | wc -l | tr -d ' ' || echo 0)}
echo ">>> task=$TASK config=$CONFIG gpus=$NUM_GPUS"

if [[ -z "${WANDB_API_KEY:-}" && -z "${WANDB_MODE:-}" ]]; then
  export WANDB_MODE=offline   # still logs locally; sync later with `wandb sync`
fi
# Run inside the project's uv environment (no implicit re-sync: extras installed by `uv sync` are kept).
RUN=()
if command -v uv >/dev/null && [[ -f pyproject.toml ]]; then
  RUN=(uv run --no-sync)
fi

export HF_HUB_ENABLE_HF_TRANSFER=${HF_HUB_ENABLE_HF_TRANSFER:-1}
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

if [[ "$NUM_GPUS" -le 1 && -z "$ACCEL_CONFIG" ]]; then
  exec "${RUN[@]}" python -m "$MODULE" --config "$CONFIG" "$@"
fi

ACCEL_CONFIG=${ACCEL_CONFIG:-configs/accelerate/multi_gpu.yaml}
exec "${RUN[@]}" accelerate launch \
  --config_file "$ACCEL_CONFIG" \
  --num_processes "$NUM_GPUS" \
  -m "$MODULE" --config "$CONFIG" "$@"
