# Training image for vietllm. Works on RunPod, Lambda, Vast.ai, GCP, AWS, Azure, k8s…
#
#   docker build --platform linux/amd64 -f docker/Dockerfile -t <registry>/vietllm:latest .
#   docker build ... --build-arg TORCH_EXTRA=cu132 -t <registry>/vietllm:cu132 .   # NVIDIA driver >= 580
#   docker run --gpus all -e HF_TOKEN -e CONFIG=configs/sft/qwen3-8b-qlora-bactrian.yaml \
#       -v $PWD/workspace:/workspace <registry>/vietllm:latest
#
# No CUDA toolkit in the image: the torch wheels ship the CUDA runtime, bitsandbytes ships binaries and
# Liger uses Triton. TORCH_EXTRA picks the torch build:
#   cu126 (default) torch 2.14 + CUDA 12.6 – runs on NVIDIA driver >= 525, i.e. almost every cloud host
#   cu132           torch 2.15 + CUDA 13.2 – needs driver >= 580 (recent H100/H200/B200 hosts)
FROM ghcr.io/astral-sh/uv:python3.12-bookworm

ARG TORCH_EXTRA=cu126
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    TORCH_EXTRA=${TORCH_EXTRA} \
    PATH=/opt/venv/bin:$PATH \
    HF_HOME=/workspace/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1

RUN apt-get update && apt-get install -y --no-install-recommends \
        git git-lfs openssh-server tmux htop build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/vietllm

# 1) dependencies only (cached unless pyproject.toml / uv.lock change)
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project \
        --extra tracking --extra deepspeed --extra "${TORCH_EXTRA}"

# 2) the project itself
COPY . .
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --extra tracking --extra deepspeed --extra "${TORCH_EXTRA}"

WORKDIR /workspace
ENTRYPOINT ["bash", "/opt/vietllm/cloud/entrypoint.sh"]
