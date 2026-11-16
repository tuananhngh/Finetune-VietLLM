"""Device detection and automatic fallbacks so CUDA recipes also run on Apple Silicon (MPS) or CPU.

The GPU configs use CUDA-only features. Without CUDA they are swapped for portable equivalents, so any
config can be tried on a Mac before going to a cloud GPU:

======================================  ===============================================
CUDA feature                            Fallback on MPS / CPU
======================================  ===============================================
QLoRA (``load_in_4bit/8bit``)           plain LoRA on bf16 (MPS) / fp32 (CPU) weights
Liger kernels (Triton)                  disabled
8-bit / paged optimizers (bitsandbytes) ``adamw_torch``
FlashAttention / hub kernels            ``sdpa``
``torch_compile``                       disabled
pinned host memory                      disabled
bf16 mixed precision                    kept on MPS (macOS 14+), disabled on CPU
======================================  ===============================================

Force a device with ``VIETLLM_DEVICE=cpu`` (or ``mps`` / ``cuda``).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import torch

logger = logging.getLogger(__name__)


def detect_device() -> str:
    forced = os.environ.get("VIETLLM_DEVICE")
    if forced:
        return forced.lower()
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def mps_supports_bf16() -> bool:
    return torch.backends.mps.is_available() and torch.backends.mps.is_macos_or_newer(14, 0)


def inference_device_and_dtype() -> tuple[str, torch.dtype]:
    device = detect_device()
    if device == "cuda":
        return device, torch.bfloat16
    if device == "mps":
        return device, torch.bfloat16 if mps_supports_bf16() else torch.float16
    return "cpu", torch.float32


def _is_cuda_only_optim(optim: Any) -> bool:
    name = str(getattr(optim, "value", optim)).lower()
    return "8bit" in name or "paged" in name or "bnb" in name


def fallback_overrides(values: dict[str, Any], device: str | None = None) -> dict[str, str]:
    """Return ``{field: new_value}`` CLI overrides needed to run ``values`` (parsed config) on this device."""
    device = device or detect_device()
    if device == "cuda":
        return {}

    out: dict[str, str] = {}

    def set_if(field: str, condition: bool, value: str, reason: str) -> None:
        if field in values and condition:
            out[field] = value
            logger.warning("[%s] %s -> %s (%s)", device, field, value, reason)

    set_if("load_in_4bit", bool(values.get("load_in_4bit")), "false", "QLoRA needs CUDA; using plain LoRA")
    set_if("load_in_8bit", bool(values.get("load_in_8bit")), "false", "bitsandbytes needs CUDA")
    set_if(
        "use_liger_kernel", bool(values.get("use_liger_kernel")), "false", "Liger kernels need CUDA/Triton"
    )
    set_if("optim", _is_cuda_only_optim(values.get("optim", "")), "adamw_torch", "bitsandbytes optimizer")
    attn = str(values.get("attn_implementation") or "")
    set_if(
        "attn_implementation", "flash" in attn or attn.startswith("kernels"), "sdpa", "CUDA-only attention"
    )
    set_if("torch_compile", bool(values.get("torch_compile")), "false", "not reliable off CUDA")
    set_if("dataloader_pin_memory", bool(values.get("dataloader_pin_memory")), "false", "no pinned memory")
    set_if("fp16", bool(values.get("fp16")) and device == "cpu", "false", "no fp16 mixed precision on CPU")

    bf16_ok = device == "mps" and mps_supports_bf16()
    set_if("bf16", bool(values.get("bf16")) and not bf16_ok, "false", "bf16 not supported here")
    dtype = str(values.get("dtype") or "")
    if device == "cpu":
        set_if("dtype", dtype in {"bfloat16", "float16"}, "float32", "half precision is slow on CPU")
    elif not bf16_ok:
        set_if("dtype", dtype == "bfloat16", "float16", "bf16 needs macOS 14+")
    return out
