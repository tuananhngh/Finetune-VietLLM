# Finetune-VietLLM

Fine-tuning Vietnamese Large Language Models using QLoRA. This project fine-tunes [Vistral-7B-Chat](https://huggingface.co/Viet-Mistral/Vistral-7B-Chat) on the Vietnamese subset of the [Bactrian-X](https://huggingface.co/datasets/MBZUAI/Bactrian-X) dataset.

## Project Structure

```
Finetune-VietLLM/
├── configs/
│   └── vistral_bactrian.yaml   # Training configuration (Hydra)
├── scripts/
│   ├── train.py                # Training script
│   └── inference.py            # Inference script
├── requirements.txt
└── README.md
```

## Prerequisites

- Python 3.10+
- CUDA-compatible GPU (tested with CUDA 11.8)
- ~16 GB VRAM minimum (4-bit quantization)

## Setup

```bash
pip install -r requirements.txt
```

## Configuration

All training parameters are defined in `configs/vistral_bactrian.yaml` and managed via [Hydra](https://hydra.cc/).

Key configuration sections:

| Section         | Description                                      |
|-----------------|--------------------------------------------------|
| `data_args`     | Dataset path, cache directory, sample size, split ratio |
| `training_args` | Batch size, learning rate, max steps, checkpointing    |
| `token_args`    | Tokenizer settings (max length, padding, special tokens)|
| `lora_args`     | LoRA rank, alpha, dropout                              |

You can override any parameter from the command line:

```bash
python scripts/train.py training_args.max_steps=200 training_args.learning_rate=1e-4
```

## Training

Run fine-tuning with default configuration:

```bash
python scripts/train.py
```

The training script will:
1. Load Vistral-7B-Chat with 4-bit quantization (QLoRA)
2. Apply LoRA adapters to attention and MLP layers
3. Train on the Vietnamese Bactrian-X dataset
4. Save checkpoints to the output directory

Checkpoints are saved every 25 steps by default. Training logs are written to `./logs/`.

## Inference

Single prompt:

```bash
python scripts/inference.py --prompt "Hãy tính 30 triệu đô la ra tiền việt."
```

Interactive chat mode:

```bash
python scripts/inference.py --interactive
```

Options:

| Flag                   | Default                              | Description              |
|------------------------|--------------------------------------|--------------------------|
| `--base-model`         | `Viet-Mistral/Vistral-7B-Chat`      | HuggingFace model ID     |
| `--checkpoint`         | `./vistral_finetune-bactrian-vi/checkpoint-100` | LoRA checkpoint path |
| `--max-new-tokens`     | `256`                                | Max tokens to generate   |
| `--temperature`        | `0.5`                                | Sampling temperature     |
| `--top-k`              | `25`                                 | Top-k sampling           |
| `--top-p`              | `0.5`                                | Nucleus sampling         |
| `--repetition-penalty` | `1.15`                               | Repetition penalty       |

## Technical Details

- **Base model**: Viet-Mistral/Vistral-7B-Chat
- **Quantization**: 4-bit NF4 with double quantization (BitsAndBytes)
- **Fine-tuning method**: LoRA (rank=32, alpha=64)
- **Target modules**: q_proj, k_proj, v_proj, out_proj, gate_proj, up_proj, down_proj, lm_head
- **Optimizer**: paged_adamw_8bit
- **Multi-GPU**: Supported via FSDP (Accelerate)
