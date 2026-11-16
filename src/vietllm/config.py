"""Configuration: one YAML file per run, overridable from the command line.

A config file is a flat mapping of fields from these dataclasses:

* ``trl.ModelConfig``            – base model, dtype, LoRA / QLoRA / DoRA settings
* ``trl.SFTConfig`` / ``DPOConfig`` – every ``transformers.TrainingArguments`` field plus TRL extras
* :class:`DataArguments`         – Vietnamese data pipeline options
* :class:`RunArguments`          – resume / merge / export behaviour

plus one special key, ``datasets``: a list of :class:`DatasetSpec` mappings (the data mixture).
Any scalar field can be overridden on the CLI, e.g. ``--learning_rate 1e-4 --max_steps 50``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SYSTEM_PROMPT = (
    "Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. "
    "Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn. "
    "Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trả lời là bạn không biết "
    "và không chia sẻ thông tin sai lệch."
)

SUPPORTED_FORMATS = ("auto", "alpaca", "sharegpt", "messages", "prompt_response", "preference")


@dataclass
class DatasetSpec:
    """One entry of the ``datasets`` mixture in a YAML config."""

    path: str
    """Hub dataset id (``MBZUAI/Bactrian-X``), a loader name (``json``) or a local file / directory."""
    name: str | None = None
    """Dataset config / subset, e.g. ``vi`` for Bactrian-X."""
    split: str = "train"
    data_files: str | list[str] | None = None
    revision: str | None = None
    format: str = "auto"
    """One of ``auto, alpaca, sharegpt, messages, prompt_response, preference``."""
    columns: dict[str, str] | None = None
    """Rename columns before formatting, ``{source_name: expected_name}``."""
    max_samples: int | None = None
    """Keep at most this many rows (after shuffling) – the simplest way to weight a mixture."""
    system_prompt: str | None = None
    """Per-dataset system prompt; overrides ``DataArguments.system_prompt``."""

    def __post_init__(self) -> None:
        if self.format not in SUPPORTED_FORMATS:
            raise ValueError(f"Unknown dataset format {self.format!r}; expected one of {SUPPORTED_FORMATS}")
        # A bare local file path is loaded with the matching builder.
        p = Path(self.path)
        if self.data_files is None and p.suffix in {".json", ".jsonl", ".csv", ".parquet"} and p.exists():
            self.data_files = str(p)
            self.path = {".jsonl": "json"}.get(p.suffix, p.suffix.lstrip("."))


@dataclass
class DataArguments:
    system_prompt: str | None = field(
        default=DEFAULT_SYSTEM_PROMPT,
        metadata={"help": "System prompt injected into conversations that have none. Set to '' to disable."},
    )
    eval_split_size: float = field(
        default=0.02, metadata={"help": "Fraction of the mixture held out for evaluation (0 disables eval)."}
    )
    max_eval_samples: int | None = field(default=500, metadata={"help": "Cap on held-out eval rows."})
    conversation_format: str = field(
        default="messages",
        metadata={
            "help": "SFT only. 'messages': full multi-turn conversations, use with assistant_only_loss=true "
            "(needs a TRL-supported chat template: Qwen, Llama 3, Gemma, Phi, Mistral…). "
            "'prompt_completion': last assistant turn is the completion, works with ANY chat template "
            "(e.g. Vistral) and is trained with completion_only_loss."
        },
    )
    deduplicate: bool = field(default=True, metadata={"help": "Drop exact duplicate conversations."})
    min_response_chars: int = field(default=2, metadata={"help": "Drop rows whose answer is shorter."})
    preprocessing_num_workers: int | None = field(default=None)


@dataclass
class RunArguments:
    resume: str = field(
        default="auto",
        metadata={
            "help": "'auto' resumes from the newest checkpoint in output_dir (safe for spot / preemptible "
            "GPUs), 'none' starts fresh, anything else is a checkpoint path."
        },
    )
    merge_after_training: bool = field(
        default=False, metadata={"help": "Merge the LoRA adapter into the base model after training."}
    )
    merged_output_dir: str | None = field(
        default=None, metadata={"help": "Where to save the merged model (default: <output_dir>/merged)."}
    )
    push_merged_to_hub: bool = field(
        default=False, metadata={"help": "Push the merged model to hub_model_id."}
    )


def split_config_file(config_path: str | os.PathLike) -> tuple[dict[str, Any], list[DatasetSpec]]:
    """Load a YAML config and separate the ``datasets`` list from the flat dataclass fields."""
    with open(config_path, encoding="utf-8") as fh:
        raw = yaml.safe_load(fh) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    for key, value in (raw.pop("env", None) or {}).items():
        os.environ.setdefault(key, str(value))
    specs = [DatasetSpec(**d) for d in raw.pop("datasets", [])]
    return raw, specs


def parse_args(
    dataclass_types: tuple[type, ...], argv: list[str] | None = None, device_fallbacks: bool = True
):
    """Parse ``--config file.yaml`` + CLI overrides into ``(*dataclass_instances, dataset_specs)``.

    With ``device_fallbacks`` (default), CUDA-only options are replaced when no CUDA GPU is present,
    see :mod:`vietllm.device`.
    """
    import sys

    from trl import TrlParser

    argv = list(sys.argv[1:] if argv is None else argv)
    config: dict[str, Any] = {}
    specs: list[DatasetSpec] = []
    if "--config" in argv:
        i = argv.index("--config")
        config, specs = split_config_file(argv[i + 1])
        del argv[i : i + 2]

    parser = TrlParser(dataclass_types)
    known = {f.name for dc in dataclass_types for f in fields(dc)}
    unknown = sorted(set(config) - known)
    if unknown:
        raise ValueError(f"Unknown config keys: {unknown}")
    parser.set_defaults(**config)
    parsed = parser.parse_args_into_dataclasses(args=argv)

    # Without CUDA (Mac / CPU), swap CUDA-only options for portable ones and parse again.
    if device_fallbacks:
        from .device import fallback_overrides

        values = {f.name: getattr(obj, f.name) for obj in parsed for f in fields(obj)}
        overrides = fallback_overrides(values)
        if overrides:
            extra = [tok for k, v in overrides.items() for tok in (f"--{k}", v)]
            parsed = parser.parse_args_into_dataclasses(args=[*argv, *extra])
    return (*parsed, specs)
