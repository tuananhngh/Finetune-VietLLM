"""Every shipped YAML config must parse against the installed TRL / transformers versions."""

from pathlib import Path

import pytest
from trl import DPOConfig, ModelConfig, SFTConfig

from vietllm.config import DataArguments, RunArguments, parse_args

ROOT = Path(__file__).resolve().parents[1]
SFT_CONFIGS = sorted((ROOT / "configs/sft").glob("*.yaml"))
DPO_CONFIGS = sorted((ROOT / "configs/dpo").glob("*.yaml"))


@pytest.mark.parametrize("path", SFT_CONFIGS, ids=lambda p: p.name)
def test_sft_configs_parse(path):
    model, train, data, run, specs = parse_args(
        (ModelConfig, SFTConfig, DataArguments, RunArguments), ["--config", str(path)]
    )
    assert model.model_name_or_path and specs
    assert data.conversation_format in {"messages", "prompt_completion"}
    if data.conversation_format == "prompt_completion":
        assert not train.assistant_only_loss


@pytest.mark.parametrize("path", DPO_CONFIGS, ids=lambda p: p.name)
def test_dpo_configs_parse(path):
    model, train, data, run, specs = parse_args(
        (ModelConfig, DPOConfig, DataArguments, RunArguments), ["--config", str(path)]
    )
    assert all(s.format == "preference" for s in specs)


def test_cli_overrides_yaml():
    cfg = ROOT / "configs/sft/smoke-test.yaml"
    _, train, _, _, _ = parse_args(
        (ModelConfig, SFTConfig, DataArguments, RunArguments),
        ["--config", str(cfg), "--max_steps", "7", "--learning_rate", "3e-5"],
    )
    assert train.max_steps == 7 and train.learning_rate == 3e-5


def test_unknown_key_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("model_name_or_path: x\nnot_a_field: 1\n")
    with pytest.raises(ValueError, match="not_a_field"):
        parse_args((ModelConfig, SFTConfig, DataArguments, RunArguments), ["--config", str(bad)])
