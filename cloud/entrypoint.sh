#!/usr/bin/env bash
# Container / VM entrypoint, cloud-agnostic (RunPod, Vast.ai, Lambda, GCP, AWS, k8s, bare metal).
#
# Environment variables:
#   CONFIG          run config to train, e.g. configs/sft/qwen3-8b-qlora-bactrian.yaml
#                   (unset -> no training, the container just stays up for SSH / Jupyter / manual work)
#   ACCEL_CONFIG    accelerate config for multi-GPU (default: DDP when >1 GPU)
#   EXTRA_ARGS      extra CLI overrides, e.g. "--max_steps 100 --push_to_hub true --hub_model_id me/model"
#   HF_TOKEN        Hugging Face token (gated models, push_to_hub)
#   WANDB_API_KEY   Weights & Biases key (otherwise W&B logs offline)
#   WORKSPACE       persistent volume mount (default /workspace) – outputs + HF cache live here
#   AUTO_STOP       "true" -> stop the RunPod pod when training ends (saves money); ignored elsewhere
#   TORCH_EXTRA     torch CUDA build when installing on a stock image: cu126 (default) | cu132
#   PUBLIC_KEY      SSH public key to authorise (RunPod sets this automatically from your account)
set -uo pipefail

WORKSPACE=${WORKSPACE:-/workspace}
REPO_DIR=${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
export HF_HOME=${HF_HOME:-$WORKSPACE/.cache/huggingface}
mkdir -p "$WORKSPACE/outputs" "$HF_HOME"

# --- SSH (optional) ---------------------------------------------------------------------------
if [[ -n "${PUBLIC_KEY:-}" ]] && command -v sshd >/dev/null; then
  mkdir -p ~/.ssh /run/sshd && chmod 700 ~/.ssh
  echo "$PUBLIC_KEY" >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys
  ssh-keygen -A >/dev/null 2>&1
  /usr/sbin/sshd && echo ">>> sshd started"
fi

# --- install if running on a stock image (e.g. RunPod PyTorch template) ----------------------
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
cd "$REPO_DIR"
if ! uv run --no-sync python -c "import vietllm" 2>/dev/null; then
  echo ">>> vietllm environment missing, running uv sync (torch build: ${TORCH_EXTRA:-cu126})"
  uv sync --frozen --no-dev --extra tracking --extra "${TORCH_EXTRA:-cu126}"
fi

# Persist outputs on the volume while keeping relative paths in configs working.
if [[ ! -L outputs ]]; then
  [[ -d outputs ]] && cp -rn outputs/. "$WORKSPACE/outputs/" 2>/dev/null; rm -rf outputs
  ln -sfn "$WORKSPACE/outputs" outputs
fi

if [[ -n "${HF_TOKEN:-}" ]]; then
  export HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
fi

nvidia-smi || echo "!!! no NVIDIA GPU visible"

if [[ -z "${CONFIG:-}" ]]; then
  echo ">>> CONFIG not set: container idle. Connect via SSH / web terminal and run e.g."
  echo "    cd $REPO_DIR && bash scripts/train.sh configs/sft/qwen3-8b-qlora-bactrian.yaml"
  exec sleep infinity
fi

LOG="$WORKSPACE/outputs/train-$(date +%Y%m%d-%H%M%S).log"
echo ">>> training $CONFIG, log: $LOG"
# shellcheck disable=SC2086
bash scripts/train.sh "$CONFIG" "${ACCEL_CONFIG:-}" ${EXTRA_ARGS:-} 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
echo ">>> training finished with status $STATUS"

if [[ "${AUTO_STOP:-false}" == "true" && -n "${RUNPOD_POD_ID:-}" ]]; then
  echo ">>> AUTO_STOP: stopping pod $RUNPOD_POD_ID (volume is kept)"
  runpodctl stop pod "$RUNPOD_POD_ID" || true
fi
# Keep the container alive so you can inspect / download results.
exec sleep infinity
