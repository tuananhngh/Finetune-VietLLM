"""End-to-end: train a tiny model for a few steps, merge it, chat with it. Needs network (Hub)."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.slow
def test_sft_merge_and_generate(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    from vietllm.train_sft import main as sft_main

    out = tmp_path / "sft"
    sft_main(["--config", "configs/sft/smoke-test.yaml", "--output_dir", str(out)])
    assert (out / "adapter_config.json").exists()
    assert (out / "merged" / "config.json").exists()

    from types import SimpleNamespace

    from vietllm.infer import generate, load_for_inference

    model, tok = load_for_inference(str(out))
    args = SimpleNamespace(
        enable_thinking=None, max_new_tokens=8, temperature=0, top_p=1.0, repetition_penalty=1.0
    )
    text = generate(model, tok, [{"role": "user", "content": "Xin chào"}], args)
    assert isinstance(text, str)


@pytest.mark.slow
def test_dpo(tmp_path, monkeypatch):
    monkeypatch.chdir(ROOT)
    from vietllm.train_dpo import main as dpo_main

    out = tmp_path / "dpo"
    dpo_main(["--config", "configs/dpo/smoke-test.yaml", "--output_dir", str(out)])
    assert (out / "adapter_config.json").exists()
    assert json.loads((out / "adapter_config.json").read_text())["r"] == 8
