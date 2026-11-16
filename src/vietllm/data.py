"""Load heterogeneous Vietnamese datasets and normalise them to TRL's conversational formats.

Every source row becomes either

* SFT  -> ``{"messages": [{"role": ..., "content": ...}, ...]}``
          or ``{"prompt": [...], "completion": [{"role": "assistant", ...}]}``
* DPO  -> ``{"prompt": [...], "chosen": [{"role": "assistant", ...}], "rejected": [...]}``
"""

from __future__ import annotations

import hashlib
import json
import logging
import unicodedata
from collections.abc import Callable
from typing import Any

from datasets import Dataset, concatenate_datasets, load_dataset

from .config import DataArguments, DatasetSpec

logger = logging.getLogger(__name__)

Message = dict[str, str]

_SHAREGPT_ROLES = {
    "human": "user",
    "user": "user",
    "gpt": "assistant",
    "assistant": "assistant",
    "bot": "assistant",
    "chatgpt": "assistant",
    "system": "system",
}


def normalize_text(text: Any) -> str:
    """NFC-normalise Vietnamese text (mixed NFC/NFD diacritics are common in scraped data)."""
    if text is None:
        return ""
    return unicodedata.normalize("NFC", str(text)).strip()


# --------------------------------------------------------------------------------------------------
# Row formatters: raw row -> list of messages (SFT) or preference triple (DPO)
# --------------------------------------------------------------------------------------------------


def from_alpaca(row: dict[str, Any]) -> list[Message]:
    instruction = normalize_text(row.get("instruction"))
    extra = normalize_text(row.get("input"))
    user = f"{instruction}\n\n{extra}" if extra else instruction
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": normalize_text(row["output"])},
    ]


def from_sharegpt(row: dict[str, Any]) -> list[Message]:
    messages = []
    for turn in row["conversations"]:
        role = _SHAREGPT_ROLES.get(str(turn.get("from", turn.get("role", ""))).lower())
        if role is None:
            raise ValueError(f"Unknown ShareGPT role in turn: {turn}")
        messages.append({"role": role, "content": normalize_text(turn.get("value", turn.get("content")))})
    return messages


def from_messages(row: dict[str, Any]) -> list[Message]:
    return [{"role": m["role"], "content": normalize_text(m["content"])} for m in row["messages"]]


def from_prompt_response(row: dict[str, Any]) -> list[Message]:
    prompt = row.get("prompt", row.get("question"))
    response = row.get("response", row.get("answer", row.get("completion")))
    return [
        {"role": "user", "content": normalize_text(prompt)},
        {"role": "assistant", "content": normalize_text(response)},
    ]


SFT_FORMATTERS: dict[str, Callable[[dict[str, Any]], list[Message]]] = {
    "alpaca": from_alpaca,
    "sharegpt": from_sharegpt,
    "messages": from_messages,
    "prompt_response": from_prompt_response,
}


def _as_messages(value: Any, default_role: str) -> list[Message]:
    if isinstance(value, list):
        return [{"role": m["role"], "content": normalize_text(m["content"])} for m in value]
    return [{"role": default_role, "content": normalize_text(value)}]


def from_preference(row: dict[str, Any]) -> dict[str, list[Message]]:
    """``prompt`` / ``chosen`` / ``rejected`` as strings or as message lists."""
    prompt = _as_messages(row["prompt"], "user")
    chosen = _as_messages(row["chosen"], "assistant")
    rejected = _as_messages(row["rejected"], "assistant")
    # Some datasets store the full conversation in chosen/rejected: strip the shared prompt prefix.
    if len(chosen) > 1 and chosen[: len(prompt)] == prompt:
        chosen, rejected = chosen[len(prompt) :], rejected[len(prompt) :]
    return {"prompt": prompt, "chosen": chosen, "rejected": rejected}


def detect_format(column_names: list[str]) -> str:
    cols = set(column_names)
    if {"prompt", "chosen", "rejected"} <= cols:
        return "preference"
    if "messages" in cols:
        return "messages"
    if "conversations" in cols:
        return "sharegpt"
    if {"instruction", "output"} <= cols:
        return "alpaca"
    if cols & {"prompt", "question"} and cols & {"response", "answer", "completion"}:
        return "prompt_response"
    raise ValueError(f"Cannot detect dataset format from columns {sorted(cols)}; set `format` explicitly.")


# --------------------------------------------------------------------------------------------------
# Post-processing helpers
# --------------------------------------------------------------------------------------------------


def add_system_prompt(messages: list[Message], system_prompt: str | None) -> list[Message]:
    if not system_prompt or (messages and messages[0]["role"] == "system"):
        return messages
    return [{"role": "system", "content": system_prompt}, *messages]


def is_valid_conversation(messages: list[Message], min_response_chars: int) -> bool:
    non_system = [m for m in messages if m["role"] != "system"]
    if len(non_system) < 2 or non_system[-1]["role"] != "assistant":
        return False
    if any(m["role"] not in {"user", "assistant", "tool"} for m in non_system):
        return False
    return all(len(m["content"]) >= min_response_chars for m in non_system if m["role"] == "assistant")


def to_prompt_completion(messages: list[Message]) -> dict[str, list[Message]]:
    return {"prompt": messages[:-1], "completion": messages[-1:]}


def _fingerprint(example: dict[str, Any]) -> str:
    return hashlib.sha1(json.dumps(example, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


def deduplicate(dataset: Dataset) -> Dataset:
    seen: set[str] = set()
    keep = []
    for i, ex in enumerate(dataset):
        fp = _fingerprint(ex)
        if fp not in seen:
            seen.add(fp)
            keep.append(i)
    if len(keep) < len(dataset):
        logger.info("Deduplication removed %d rows", len(dataset) - len(keep))
    return dataset.select(keep)


# --------------------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------------------


def load_raw(spec: DatasetSpec, seed: int) -> Dataset:
    try:
        ds = load_dataset(
            spec.path,
            name=spec.name,
            split=spec.split,
            data_files=spec.data_files,
            revision=spec.revision,
        )
    except RuntimeError as err:
        if "Dataset scripts are no longer supported" not in str(err):
            raise
        raise RuntimeError(
            f"{spec.path} uses a loading script, which `datasets` no longer runs. Load its data files "
            f"directly instead, e.g. `path: json` + `data_files: hf://datasets/{spec.path}/<file>`, or use "
            "the Hub's parquet conversion with `revision: refs/convert/parquet`."
        ) from err
    if spec.columns:
        ds = ds.rename_columns({k: v for k, v in spec.columns.items() if k in ds.column_names})
    if spec.max_samples is not None and spec.max_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(spec.max_samples))
    return ds


def format_sft_dataset(
    ds: Dataset, spec: DatasetSpec, data_args: DataArguments, num_proc: int | None
) -> Dataset:
    fmt = spec.format if spec.format != "auto" else detect_format(ds.column_names)
    if fmt not in SFT_FORMATTERS:
        raise ValueError(f"Dataset {spec.path} has format {fmt!r}, which cannot be used for SFT")
    formatter = SFT_FORMATTERS[fmt]
    system_prompt = spec.system_prompt if spec.system_prompt is not None else data_args.system_prompt

    def _map(row: dict[str, Any]) -> dict[str, Any]:
        return {"messages": add_system_prompt(formatter(row), system_prompt)}

    ds = ds.map(_map, remove_columns=ds.column_names, num_proc=num_proc, desc=f"format {spec.path}")
    return ds.filter(
        lambda r: is_valid_conversation(r["messages"], data_args.min_response_chars),
        num_proc=num_proc,
        desc="filter",
    )


def format_preference_dataset(
    ds: Dataset, spec: DatasetSpec, data_args: DataArguments, num_proc: int | None
) -> Dataset:
    fmt = spec.format if spec.format != "auto" else detect_format(ds.column_names)
    if fmt != "preference":
        raise ValueError(f"Dataset {spec.path} has format {fmt!r}; DPO needs prompt/chosen/rejected columns")
    system_prompt = spec.system_prompt if spec.system_prompt is not None else data_args.system_prompt

    def _map(row: dict[str, Any]) -> dict[str, Any]:
        out = from_preference(row)
        out["prompt"] = add_system_prompt(out["prompt"], system_prompt)
        return out

    ds = ds.map(_map, remove_columns=ds.column_names, num_proc=num_proc, desc=f"format {spec.path}")
    return ds.filter(
        lambda r: bool(r["chosen"]) and bool(r["rejected"]) and r["chosen"] != r["rejected"],
        num_proc=num_proc,
    )


def build_datasets(
    specs: list[DatasetSpec], data_args: DataArguments, seed: int, task: str = "sft"
) -> tuple[Dataset, Dataset | None]:
    """Load, format, mix, deduplicate and split. ``task`` is ``"sft"`` or ``"dpo"``."""
    if not specs:
        raise ValueError("No datasets configured: add a `datasets:` list to your YAML config.")
    num_proc = data_args.preprocessing_num_workers
    parts = []
    for spec in specs:
        raw = load_raw(spec, seed)
        if task == "sft":
            part = format_sft_dataset(raw, spec, data_args, num_proc)
        else:
            part = format_preference_dataset(raw, spec, data_args, num_proc)
        logger.info("Loaded %s (%s/%s): %d rows", spec.path, spec.name, spec.split, len(part))
        parts.append(part)

    mixture = concatenate_datasets(parts).shuffle(seed=seed)
    if data_args.deduplicate:
        mixture = deduplicate(mixture)
    if task == "sft" and data_args.conversation_format == "prompt_completion":
        mixture = mixture.map(
            lambda r: to_prompt_completion(r["messages"]), remove_columns=["messages"], num_proc=num_proc
        )
    elif task == "sft" and data_args.conversation_format != "messages":
        raise ValueError("conversation_format must be 'messages' or 'prompt_completion'")

    if data_args.eval_split_size <= 0 or len(mixture) < 10:
        return mixture, None
    test_size = max(1, int(len(mixture) * data_args.eval_split_size))
    if data_args.max_eval_samples is not None:
        test_size = min(test_size, data_args.max_eval_samples)
    split = mixture.train_test_split(test_size=test_size, seed=seed)
    logger.info("Train rows: %d | eval rows: %d", len(split["train"]), len(split["test"]))
    return split["train"], split["test"]
