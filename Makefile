IMAGE ?= $(or $(VIETLLM_IMAGE),vietllm:latest)
CONFIG ?= configs/sft/qwen3-8b-qlora-bactrian.yaml
TORCH_EXTRA ?= cu126

.PHONY: install install-gpu lock upgrade test lint format smoke mac-test mac-chat train docker-build docker-push

install:            ## dev install (macOS / CPU / default CUDA torch) + dev tools
	uv sync --extra tracking

install-gpu:        ## GPU box: pick the torch CUDA build (TORCH_EXTRA=cu126|cu132)
	uv sync --no-dev --extra tracking --extra deepspeed --extra $(TORCH_EXTRA)

lock:               ## re-resolve uv.lock after editing pyproject.toml
	uv lock

upgrade:            ## bump every dependency to the newest allowed version
	uv lock --upgrade

test:               ## fast unit tests (no downloads)
	uv run pytest -m "not slow" -q

lint:
	uv run ruff check src tests cloud
	uv run ruff format --check src tests cloud

format:
	uv run ruff format src tests cloud
	uv run ruff check --fix src tests cloud

smoke:              ## tiny end-to-end SFT + merge + DPO run (downloads a few-MB model)
	uv run python -m vietllm.train_sft --config configs/sft/smoke-test.yaml
	uv run python -m vietllm.train_dpo --config configs/dpo/smoke-test.yaml

mac-test:           ## real LoRA fine-tune of Qwen3-0.6B on Bactrian-X vi using the Mac GPU (MPS), ~7 min
	uv run python -m vietllm.train_sft --config configs/sft/mac-qwen3-0.6b.yaml

mac-chat:           ## chat with the model from `make mac-test`
	uv run vietllm-chat --model outputs/mac-qwen3-0.6b/merged --enable-thinking false

train:
	bash scripts/train.sh $(CONFIG)

docker-build:
	docker build --platform linux/amd64 --build-arg TORCH_EXTRA=$(TORCH_EXTRA) -f docker/Dockerfile -t $(IMAGE) .

docker-push: docker-build
	docker push $(IMAGE)
