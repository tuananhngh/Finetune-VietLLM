from pathlib import Path

import pytest
from trl import ModelConfig, SFTConfig

from vietllm.config import DataArguments, RunArguments, parse_args
from vietllm.device import fallback_overrides

ROOT = Path(__file__).resolve().parents[1]
TYPES = (ModelConfig, SFTConfig, DataArguments, RunArguments)
CUDA_VALUES = {
    "load_in_4bit": True,
    "use_liger_kernel": True,
    "optim": "paged_adamw_8bit",
    "attn_implementation": "flash_attention_2",
    "bf16": True,
    "dtype": "bfloat16",
    "dataloader_pin_memory": True,
}


def test_no_fallback_on_cuda():
    assert fallback_overrides(CUDA_VALUES, device="cuda") == {}


def test_mps_fallbacks(monkeypatch):
    monkeypatch.setattr("vietllm.device.mps_supports_bf16", lambda: True)
    out = fallback_overrides(CUDA_VALUES, device="mps")
    assert out == {
        "load_in_4bit": "false",
        "use_liger_kernel": "false",
        "optim": "adamw_torch",
        "attn_implementation": "sdpa",
        "dataloader_pin_memory": "false",
    }


def test_cpu_fallbacks():
    out = fallback_overrides(CUDA_VALUES, device="cpu")
    assert out["bf16"] == "false" and out["dtype"] == "float32" and out["load_in_4bit"] == "false"


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_cuda_recipe_parses_for_mac_or_cpu(monkeypatch, device):
    if device == "mps":
        # Simulate a bf16-capable Mac, also for transformers' own check (CI VMs have no usable MPS).
        monkeypatch.setattr("vietllm.device.mps_supports_bf16", lambda: True)
        monkeypatch.setattr("transformers.training_args.is_torch_bf16_gpu_available", lambda: True)
    monkeypatch.setenv("VIETLLM_DEVICE", device)
    model, train, *_ = parse_args(TYPES, ["--config", str(ROOT / "configs/sft/qwen3-8b-qlora-bactrian.yaml")])
    assert not model.load_in_4bit and not train.use_liger_kernel
    assert "8bit" not in str(train.optim.value)


def test_fallbacks_can_be_disabled(monkeypatch):
    monkeypatch.setenv("VIETLLM_DEVICE", "cpu")
    # --bf16 false: without fallbacks, transformers itself rejects bf16 on a machine with no bf16 GPU.
    argv = ["--config", str(ROOT / "configs/sft/qwen3-8b-qlora-bactrian.yaml"), "--bf16", "false"]
    model, *_ = parse_args(TYPES, argv, device_fallbacks=False)
    assert model.load_in_4bit
