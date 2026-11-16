IMAGE := env("VIETLLM_IMAGE", "vietllm:latest")
TORCH_EXTRA := "cu126"

# list recipes
default:
    @just --list

# dev install (macOS / CPU / default CUDA torch) + dev tools
install:
    uv sync --extra tracking

# GPU box: pick the torch CUDA build (just TORCH_EXTRA=cu132 install-gpu)
install-gpu:
    uv sync --no-dev --extra tracking --extra deepspeed --extra {{TORCH_EXTRA}}

# re-resolve uv.lock after editing pyproject.toml
lock:
    uv lock

# bump every dependency to the newest allowed version
upgrade:
    uv lock --upgrade

# fast unit tests (no downloads)
test:
    uv run pytest -m "not slow" -q

lint:
    uv run ruff check src tests cloud
    uv run ruff format --check src tests cloud

format:
    uv run ruff format src tests cloud
    uv run ruff check --fix src tests cloud

# tiny end-to-end SFT + merge + DPO run (downloads a few-MB model)
smoke:
    uv run python -m vietllm.train_sft --config configs/sft/smoke-test.yaml
    uv run python -m vietllm.train_dpo --config configs/dpo/smoke-test.yaml

# real LoRA fine-tune of Qwen3-0.6B on Bactrian-X vi using the Mac GPU (MPS), ~7 min
mac-test:
    uv run python -m vietllm.train_sft --config configs/sft/mac-qwen3-0.6b.yaml

# chat with the model from `just mac-test`
mac-chat:
    uv run vietllm-chat --model outputs/mac-qwen3-0.6b/merged --enable-thinking false

# launch training with a config
train config="configs/sft/qwen3-8b-qlora-bactrian.yaml":
    bash scripts/train.sh {{config}}

docker-build:
    docker build --platform linux/amd64 --build-arg TORCH_EXTRA={{TORCH_EXTRA}} -f docker/Dockerfile -t {{IMAGE}} .

docker-push: docker-build
    docker push {{IMAGE}}
