"""Model / tokenizer helpers shared by the SFT and DPO entry points."""

from __future__ import annotations

import logging
import os
from typing import Any

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from transformers.trainer_utils import get_last_checkpoint
from trl import ModelConfig, get_peft_config, get_quantization_config

logger = logging.getLogger(__name__)


def setup_logging(is_main_process: bool = True) -> None:
    logging.basicConfig(
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        level=logging.INFO if is_main_process else logging.WARNING,
    )
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def resolve_dtype(model_args: ModelConfig) -> torch.dtype | str:
    dtype = model_args.dtype
    if dtype in (None, "auto"):
        return "auto"
    if isinstance(dtype, str):
        dtype = getattr(torch, dtype)
    # bf16 needs Ampere+ (A100, A6000, L4, L40S, RTX 30xx/40xx/50xx, H100…). Fall back on older GPUs.
    if dtype == torch.bfloat16 and torch.cuda.is_available() and not torch.cuda.is_bf16_supported():
        logger.warning("GPU does not support bfloat16, falling back to float16")
        dtype = torch.float16
    return dtype


def model_init_kwargs(model_args: ModelConfig, use_cache: bool = False) -> dict[str, Any]:
    """kwargs forwarded by TRL to ``AutoModelForCausalLM.from_pretrained``."""
    kwargs: dict[str, Any] = {
        "revision": model_args.model_revision,
        "dtype": resolve_dtype(model_args),
        "use_cache": use_cache,
    }
    if model_args.attn_implementation:
        kwargs["attn_implementation"] = model_args.attn_implementation
    return kwargs


def quantization_and_peft(model_args: ModelConfig):
    """Return ``(BitsAndBytesConfig | None, LoraConfig | None)`` built from the ModelConfig."""
    if model_args.load_in_4bit or model_args.load_in_8bit:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "load_in_4bit / load_in_8bit (QLoRA) need a CUDA GPU with bitsandbytes installed"
            )
        if isinstance(model_args.dtype, str) and model_args.dtype not in ("auto", None):
            # bitsandbytes compute dtype must be a torch dtype
            model_args.dtype = getattr(torch, model_args.dtype)
    quant = get_quantization_config(model_args)
    peft = get_peft_config(model_args)
    if quant is not None and peft is None:
        raise ValueError("A quantized base model can only be fine-tuned with an adapter: set use_peft: true")
    return quant, peft


def load_tokenizer(model_args: ModelConfig, trust_remote_code: bool = False) -> PreTrainedTokenizerBase:
    tok = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=trust_remote_code,
    )
    if tok.pad_token is None:
        # Avoid padding with EOS: EOS labels could get masked and the model would never learn to stop.
        for candidate in (
            "<|finetune_right_pad_id|>",
            "<|fim_pad|>",
            "<pad>",
            "[PAD]",
            "<unk>",
        ):
            if candidate in tok.get_vocab():
                tok.pad_token = candidate
                break
        else:
            tok.pad_token = tok.eos_token
    return tok


def find_resume_checkpoint(resume: str, output_dir: str) -> str | None:
    if resume == "none":
        return None
    if resume == "auto":
        if os.path.isdir(output_dir):
            ckpt = get_last_checkpoint(output_dir)
            if ckpt:
                logger.info("Resuming from %s", ckpt)
            return ckpt
        return None
    return resume


def log_trainable_parameters(model) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(
        "Trainable params: %s / %s (%.3f%%)",
        f"{trainable:,}",
        f"{total:,}",
        100 * trainable / total,
    )
