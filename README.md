# Finetune-VietLLM

Fine-tune open LLMs for **Vietnamese** — SFT and DPO with LoRA / QLoRA / DoRA — on one GPU, several GPUs, or a cloud
GPU (RunPod, or any cloud through SkyPilot / Docker).

Built on the current Hugging Face stack (TRL 1.x, transformers 5, PEFT, Accelerate):

| Feature | What it gives you |
|---|---|
| **QLoRA / LoRA / DoRA / rsLoRA** | 8B models on a single 24 GB GPU; adapters on `all-linear` layers |
| **Assistant-only loss** | loss only on the assistant's tokens, multi-turn aware (TRL training chat templates) |
| **Packing (BFD, padding-free)** | no wasted compute on padding → much faster epochs |
| **Liger kernels** | fused RMSNorm / RoPE / SwiGLU / cross-entropy: less VRAM, more throughput |
| **DPO + variants** | preference tuning after SFT (`sigmoid`, `ipo`, `apo_zero`, RPO-style `[sigmoid, sft]`, …) |
| **Dataset mixtures** | Alpaca, ShareGPT, OpenAI-messages, prompt/response and preference formats; NFC Unicode normalisation for Vietnamese diacritics, dedup, Vietnamese system prompt |
| **Multi-GPU** | DDP, FSDP-QLoRA, DeepSpeed ZeRO-3 via `accelerate` configs |
| **Preemption-safe** | `resume: auto` restarts from the newest checkpoint — spot / community pods are fine |
| **Export & serve** | merge adapter → standalone model, push to the Hub, serve with vLLM (OpenAI API) |
| **Eval** | side-by-side generations on Vietnamese prompts + `lm-eval` benchmarks (`arc_vi`, `hellaswag_vi`, `m_mmlu_vi`) |

The default recipe fine-tunes **Qwen3-8B** on the Vietnamese split of
[Bactrian-X](https://huggingface.co/datasets/MBZUAI/Bactrian-X). The original POC recipe (Vistral-7B-Chat) is kept as
`configs/sft/vistral-7b-qlora-bactrian.yaml`. Any causal LM with a chat template works: change `model_name_or_path`.

## Project layout

```
configs/
  sft/                     run configs (model + data + training in one YAML)
    qwen3-8b-qlora-bactrian.yaml   1x 24 GB GPU, QLoRA          ← start here
    qwen3-8b-lora-mix.yaml         multi-GPU bf16 LoRA, dataset mixture
    qwen3-32b-qlora-fsdp.yaml      FSDP + QLoRA for big models (advanced)
    vistral-7b-qlora-bactrian.yaml the original POC, ported
    mac-qwen3-0.6b.yaml            real fine-tune on an Apple Silicon GPU (MPS)
    smoke-test.yaml                tiny model, runs on a laptop in ~30 s
  dpo/                     preference-tuning configs
  accelerate/              single_gpu, multi_gpu (DDP), fsdp_qlora, deepspeed_zero3
src/vietllm/
  config.py   data.py   modeling.py   device.py (CUDA / MPS / CPU fallbacks)
  train_sft.py  train_dpo.py  merge.py  infer.py  evaluate.py
data/examples/             small Vietnamese sample files (SFT, Alpaca, preference)
data/eval/prompts_vi.jsonl Vietnamese prompts for qualitative comparison
scripts/train.sh           auto-detects GPUs, picks python / accelerate launch
scripts/serve_vllm.sh      OpenAI-compatible server
docker/Dockerfile          CUDA 12.8 image, used for every cloud
cloud/entrypoint.sh        cloud-agnostic container entrypoint
cloud/runpod/launch_pod.py create a RunPod pod from the CLI
cloud/skypilot/sft.yaml    run on any cloud (RunPod, Lambda, AWS, GCP, Azure, Vast, k8s…)
tests/                     unit tests + end-to-end smoke test
```

## Setup with uv

The project is managed with [uv](https://docs.astral.sh/uv/): dependencies live in `pyproject.toml`, exact versions
in `uv.lock`, and the Python version (3.12) in `.python-version`. uv creates `.venv/` and installs Python itself.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh     # once

# GPU machine: choose the torch CUDA build that matches the NVIDIA driver (`nvidia-smi`)
uv sync --extra tracking --extra cu126              # driver >= 525: torch 2.14 + CUDA 12.6 (safe default)
uv sync --extra tracking --extra cu132              # driver >= 580: torch 2.15 + CUDA 13.2

# laptop / CI (macOS or CPU), for development and the smoke test
uv sync --extra tracking

cp .env.example .env    # add HF_TOKEN / WANDB_API_KEY, then: set -a; source .env; set +a
```

| Extra / group | Adds |
|---|---|
| `--extra tracking` | Weights & Biases, TensorBoard |
| `--extra deepspeed` | DeepSpeed ZeRO (Linux, built from source against the installed torch) |
| `--extra eval` | lm-evaluation-harness (HF backend) |
| `--extra cu126` / `--extra cu132` | torch from the PyTorch CUDA 12.6 / 13.2 index (mutually exclusive) |
| `dev` group (default) | pytest, ruff, mypy; skip with `--no-dev` |

Run commands inside the environment with `uv run` (or `source .venv/bin/activate`), e.g.
`uv run vietllm-chat --model …`. `scripts/train.sh` uses `uv run` automatically.

Dependency management:

```bash
uv add <package>              # add a dependency (updates pyproject.toml + uv.lock)
uv add --optional eval <pkg>  # add to an extra
uv lock --upgrade             # bump everything to the newest allowed versions (make upgrade)
uv lock --upgrade-package trl # bump one package
```

vLLM pins its own torch / transformers versions, so it is **not** part of the training environment: serving and
the vLLM benchmark backend run it in an isolated environment with `uvx` (installed on first use).

## Quick start

Check everything works with the tiny model (no GPU needed, ~30 s). For a real run on a Mac GPU, see
[Testing on a Mac](#testing-on-a-mac-apple-silicon-gpu).

```bash
make smoke
```

Train:

```bash
bash scripts/train.sh configs/sft/qwen3-8b-qlora-bactrian.yaml
```

Override any field from the CLI (second arg is the accelerate config; `""` = auto):

```bash
bash scripts/train.sh configs/sft/qwen3-8b-qlora-bactrian.yaml "" --max_steps 200 --learning_rate 1e-4
```

Multi-GPU (all GPUs on the machine are used automatically):

```bash
bash scripts/train.sh configs/sft/qwen3-8b-lora-mix.yaml
bash scripts/train.sh configs/sft/qwen3-8b-lora-mix.yaml configs/accelerate/deepspeed_zero3.yaml
```

Output in `outputs/<run>/`: the LoRA adapter, checkpoints, metrics, and `merged/` (full model) when
`merge_after_training: true`.

## Testing on a Mac (Apple Silicon GPU)

Training and inference run on the Mac GPU through PyTorch's MPS backend: no CUDA, no cloud. `uv sync --extra tracking`
is all you need.

```bash
make mac-test     # real LoRA fine-tune: Qwen3-0.6B on 2,000 Bactrian-X vi rows, 100 steps, ~7 min on an M3 Max
make mac-chat     # chat with the result
```

`configs/sft/mac-qwen3-0.6b.yaml` uses the same model family, chat template and loss masking as the Qwen3-8B cloud
recipe, so a good run here validates the data and the setup before you pay for a GPU.

**Every config also runs on a Mac.** Without CUDA, CUDA-only options are swapped automatically and logged as
`[mps] field -> value`: QLoRA becomes plain LoRA on bf16 weights, Liger kernels and FlashAttention are turned
off, and 8-bit/paged optimizers become `adamw_torch`. For example, to try the real cloud recipe for 10 steps on
a smaller model:

```bash
bash scripts/train.sh configs/sft/qwen3-8b-qlora-bactrian.yaml "" \
    --model_name_or_path Qwen/Qwen3-1.7B --max_steps 10 --output_dir outputs/mac-try
```

Notes:

* Throughput is about 500 tokens/s for a 0.6B model on an M3 Max, so this is for testing, not full runs. Memory is
  unified: a 0.6–1.7B model is comfortable on 16 GB, and 4–8B in bf16 needs 32–64 GB+.
* MPS attention has no memory-efficient kernel, so keep `max_length` around 1–2k tokens.
* `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically, so the few ops MPS lacks run on the CPU instead of failing.
* Force a device with `VIETLLM_DEVICE=cpu` (or `mps`). Multi-GPU, DeepSpeed and FSDP need CUDA.

## After training

```bash
# chat (adapter or merged model)
uv run vietllm-chat --model outputs/qwen3-8b-qlora-bactrian/merged
uv run vietllm-chat --model outputs/qwen3-8b-qlora-bactrian --prompt "Hãy giới thiệu về Vịnh Hạ Long."

# compare base vs fine-tuned on the Vietnamese prompt set → reports/compare/generations.md
uv run vietllm-eval generate --models Qwen/Qwen3-8B outputs/qwen3-8b-qlora-bactrian/merged --output reports/compare

# benchmarks: merged model → vLLM backend via uvx; adapters → HF backend (uv sync --extra eval)
uv run vietllm-eval benchmark --model outputs/qwen3-8b-qlora-bactrian/merged --tasks arc_vi,hellaswag_vi,m_mmlu_vi

# merge manually / publish
uv run vietllm-merge --adapter outputs/qwen3-8b-qlora-bactrian --output outputs/qwen3-8b-vi --push-to-hub <you>/qwen3-8b-vi

# serve with an OpenAI-compatible API (vLLM via uvx; pin with VLLM_VERSION=...)
bash scripts/serve_vllm.sh outputs/qwen3-8b-qlora-bactrian/merged
```

Then DPO on top of the SFT model:

```bash
bash scripts/train.sh configs/dpo/qwen3-8b-dpo.yaml
```

`data/examples/preference_vi.jsonl` is only a format example: point the config at a real Vietnamese preference dataset
(`prompt` / `chosen` / `rejected` columns).

## Data

Each run config has a `datasets:` list. Rows are converted to chat messages, a Vietnamese system prompt is added where
missing, text is NFC-normalised, exact duplicates are removed, and a held-out eval split is made.

```yaml
datasets:
  - path: some-org/vietnamese-instructions   # Hub dataset (subset with `name:`)
    format: alpaca                   # instruction / input / output
  - path: json                       # raw files on the Hub (here Bactrian-X vi, whose loading script
    data_files: hf://datasets/MBZUAI/Bactrian-X/data/vi.json.gz   # datasets>=4 refuses to run)
    format: alpaca
  - path: data/my_chats.jsonl        # local file (.jsonl / .json / .csv / .parquet)
    format: messages                 # {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]}
  - path: some-org/vietnamese-sharegpt
    format: sharegpt                 # {"conversations": [{"from": "human", "value": ...}, ...]}
    max_samples: 50000               # cap / weight this source
    columns: {question: prompt}      # optional renames
    system_prompt: "..."             # optional per-dataset override
```

Formats: `alpaca`, `sharegpt`, `messages`, `prompt_response`, `preference` (DPO), or `auto` (detected from the
columns).

`conversation_format` picks how SFT computes the loss:

* `messages` + `assistant_only_loss: true` (default): every assistant turn of multi-turn chats is trained. Needs a chat
  template TRL can patch for training (Qwen 2.5/3/3.5+, Llama 3.x, Gemma, Phi, Mistral, DeepSeek, GLM, …). With Qwen3
  the template adds an empty `<think></think>` block to the answer, so the model learns to reply in non-thinking mode.
* `prompt_completion`: only the last assistant turn is trained. Works with **any** chat template (used for Vistral).

## Cloud GPUs

Everything runs from the same Docker image, so RunPod is just one target.

### 1. Build and push the image

```bash
make docker-push IMAGE=ghcr.io/<you>/vietllm:latest                       # CUDA 12.6, any recent host
make docker-push IMAGE=ghcr.io/<you>/vietllm:cu132 TORCH_EXTRA=cu132      # CUDA 13.2, driver >= 580
```

The image is built from `uv.lock` with `uv sync --frozen`, so it gets exactly the versions you tested locally.

```bash
```

### 2a. RunPod — from the CLI

```bash
export RUNPOD_API_KEY=... HF_TOKEN=... WANDB_API_KEY=...
uv run cloud/runpod/launch_pod.py --list-gpus      # inline script deps: uv installs `runpod` on the fly
uv run cloud/runpod/launch_pod.py --image ghcr.io/<you>/vietllm:latest \
    --gpu "NVIDIA GeForce RTX 4090" \
    --config configs/sft/qwen3-8b-qlora-bactrian.yaml --auto-stop
```

* `--auto-stop` stops the pod when training ends (the volume with your outputs is kept).
* `--network-volume-id <id> --data-center-id <dc>` keeps checkpoints and the HF model cache across pods; with
  `resume: auto` a new pod picks up where the last one stopped.
* Multi-GPU: `--gpu "NVIDIA H100 80GB HBM3" --gpu-count 4 --config configs/sft/qwen3-8b-lora-mix.yaml`.

### 2b. RunPod — from the web console

Create a pod with **Container image** = your image, **Volume mount path** = `/workspace`, expose TCP 22, and set
environment variables `CONFIG`, `HF_TOKEN`, `WANDB_API_KEY` (optionally `EXTRA_ARGS`, `AUTO_STOP=true`).
Leave `CONFIG` empty to get an idle pod you can SSH into and run `bash scripts/train.sh …` by hand.

Using RunPod's stock PyTorch template instead of a custom image also works: open the web terminal, then

```bash
cd /workspace && git clone <this repo> && cd Finetune-VietLLM
CONFIG=configs/sft/qwen3-8b-qlora-bactrian.yaml bash cloud/entrypoint.sh
```

(the entrypoint installs uv and runs `uv sync` if needed; set `TORCH_EXTRA=cu132` on driver >= 580 hosts).

### 3. Any other cloud

* **SkyPilot** (RunPod, Lambda, Vast, AWS, GCP, Azure, Nebius, Kubernetes…), cheapest matching GPU is picked:
  ```bash
  uv tool install "skypilot[runpod,aws,gcp]" && sky check      # once; list the clouds you use
  sky launch -c vietllm cloud/skypilot/sft.yaml --env HF_TOKEN --env WANDB_API_KEY
  sky logs vietllm && sky down vietllm
  ```
* **Plain Docker** on any GPU VM (Lambda, Vast.ai, GCE, EC2…):
  ```bash
  docker run --gpus all --shm-size 16g -e HF_TOKEN -e WANDB_API_KEY \
      -e CONFIG=configs/sft/qwen3-8b-qlora-bactrian.yaml \
      -v /data/vietllm:/workspace ghcr.io/<you>/vietllm:latest
  ```

`cloud/entrypoint.sh` env vars: `CONFIG`, `ACCEL_CONFIG`, `EXTRA_ARGS`, `TORCH_EXTRA`, `HF_TOKEN`, `WANDB_API_KEY`, `WORKSPACE`
(default `/workspace`), `AUTO_STOP`, `PUBLIC_KEY`. Outputs go to `$WORKSPACE/outputs`, the HF cache to
`$WORKSPACE/.cache/huggingface`, and a log to `$WORKSPACE/outputs/train-<timestamp>.log`.

### Which GPU?

| Recipe | VRAM | Example GPUs |
|---|---|---|
| 7–8B QLoRA, 2k context | ~16–22 GB | RTX 4090, L4, A5000, A10 |
| 7–8B bf16 LoRA, 4k context | ~40–60 GB | A100 80GB, H100, L40S (lower batch) |
| 32B QLoRA | ~40 GB (single) / 2 GPUs with FSDP | A100 80GB, H100, 2x A6000 |
| 70B QLoRA | 2–4x 80 GB with FSDP | H100 / A100 80GB |

Out of memory? Lower `per_device_train_batch_size` (raise `gradient_accumulation_steps` to compensate) or
`max_length`.

## Tracking

`report_to: wandb` in the configs. Without `WANDB_API_KEY`, `scripts/train.sh` sets `WANDB_MODE=offline` (sync later
with `wandb sync`). Use `report_to: tensorboard` or `none` if you prefer.

## Development

```bash
uv sync --extra tracking   # dev group is installed by default
make test     # unit tests: data formatting + every shipped config parses against the installed TRL
make lint
uv run pytest -m slow   # end-to-end: tiny SFT → merge → generate, tiny DPO (downloads a few MB)
```

What has been verified: unit tests, every config parsing against TRL 1.18 / transformers 5.23 / PEFT 0.21, and the
end-to-end smoke runs on CPU/MPS, and a real Qwen3-0.6B LoRA run on Bactrian-X on an M3 Max GPU. The CUDA-only paths (bitsandbytes QLoRA, Liger kernels, multi-GPU accelerate
configs, the Docker image and the cloud launchers) need a real GPU machine to exercise, so do a short run
(`--max_steps 20`) the first time you use one.
