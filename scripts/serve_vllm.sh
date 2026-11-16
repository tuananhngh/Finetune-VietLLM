#!/usr/bin/env bash
# OpenAI-compatible server with vLLM, run through `uvx` in its own environment
# (vLLM pins its own torch/transformers, so it is not part of the training env). Pin with VLLM_VERSION.
#
#   bash scripts/serve_vllm.sh outputs/qwen3-8b-qlora-bactrian/merged          # merged model
#   BASE=Qwen/Qwen3-8B bash scripts/serve_vllm.sh outputs/qwen3-8b-qlora-bactrian  # base + LoRA adapter
#
# Then: curl localhost:8000/v1/chat/completions -H 'Content-Type: application/json' \
#         -d '{"model":"vietllm","messages":[{"role":"user","content":"Xin chào!"}]}'
set -euo pipefail
MODEL=${1:?usage: serve_vllm.sh <merged-model-dir | adapter-dir>}
PORT=${PORT:-8000}
VLLM=(uvx --python 3.12 "vllm==${VLLM_VERSION:-0.33.0}")
command -v uvx >/dev/null || VLLM=(vllm)   # fall back to a vllm already on PATH

if [[ -f "$MODEL/adapter_config.json" ]]; then
  BASE=${BASE:-$(python -c "import json,sys;print(json.load(open('$MODEL/adapter_config.json'))['base_model_name_or_path'])")}
  RANK=$(python -c "import json;print(json.load(open('$MODEL/adapter_config.json'))['r'])")
  exec "${VLLM[@]}" serve "$BASE" --port "$PORT" --served-model-name base \
    --enable-lora --max-lora-rank "$RANK" --lora-modules "vietllm=$MODEL" "${@:2}"
fi
exec "${VLLM[@]}" serve "$MODEL" --port "$PORT" --served-model-name vietllm "${@:2}"
