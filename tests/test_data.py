import json
from pathlib import Path

import pytest
from datasets import Dataset

from vietllm.config import DataArguments, DatasetSpec
from vietllm.data import (
    add_system_prompt,
    build_datasets,
    deduplicate,
    detect_format,
    from_alpaca,
    from_preference,
    from_sharegpt,
    is_valid_conversation,
    normalize_text,
    to_prompt_completion,
)

ROOT = Path(__file__).resolve().parents[1]


def test_normalize_text_nfc():
    nfd = "Việt Nam"  # decomposed "Việt"
    assert normalize_text(nfd) == "Việt Nam"


def test_alpaca_with_and_without_input():
    msgs = from_alpaca({"instruction": "Dịch", "input": "hello", "output": "xin chào"})
    assert msgs == [
        {"role": "user", "content": "Dịch\n\nhello"},
        {"role": "assistant", "content": "xin chào"},
    ]
    msgs = from_alpaca({"instruction": "Chào", "input": "", "output": "Xin chào"})
    assert msgs[0]["content"] == "Chào"


def test_sharegpt_roles():
    row = {
        "conversations": [
            {"from": "system", "value": "S"},
            {"from": "human", "value": "Q"},
            {"from": "gpt", "value": "A"},
        ]
    }
    assert [m["role"] for m in from_sharegpt(row)] == ["system", "user", "assistant"]
    with pytest.raises(ValueError):
        from_sharegpt({"conversations": [{"from": "alien", "value": "x"}]})


def test_preference_strips_shared_prompt():
    prompt = [{"role": "user", "content": "Q"}]
    row = {
        "prompt": prompt,
        "chosen": prompt + [{"role": "assistant", "content": "good"}],
        "rejected": prompt + [{"role": "assistant", "content": "bad"}],
    }
    out = from_preference(row)
    assert out["chosen"] == [{"role": "assistant", "content": "good"}]
    assert out["rejected"] == [{"role": "assistant", "content": "bad"}]
    out = from_preference({"prompt": "Q", "chosen": "good", "rejected": "bad"})
    assert out["prompt"] == prompt


@pytest.mark.parametrize(
    "cols,fmt",
    [
        (["instruction", "input", "output", "id"], "alpaca"),
        (["conversations"], "sharegpt"),
        (["messages"], "messages"),
        (["prompt", "chosen", "rejected"], "preference"),
        (["question", "answer"], "prompt_response"),
    ],
)
def test_detect_format(cols, fmt):
    assert detect_format(cols) == fmt


def test_detect_format_unknown():
    with pytest.raises(ValueError):
        detect_format(["foo"])


def test_system_prompt_and_validation():
    msgs = [{"role": "user", "content": "Q"}, {"role": "assistant", "content": "Answer"}]
    with_sys = add_system_prompt(msgs, "SYS")
    assert with_sys[0] == {"role": "system", "content": "SYS"}
    assert add_system_prompt(with_sys, "OTHER") == with_sys
    assert add_system_prompt(msgs, "") == msgs
    assert is_valid_conversation(with_sys, 2)
    assert not is_valid_conversation(msgs[:1], 2)  # no assistant
    assert not is_valid_conversation([msgs[0], {"role": "assistant", "content": "x"}], 2)  # too short


def test_prompt_completion_split():
    msgs = [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "A"},
    ]
    out = to_prompt_completion(msgs)
    assert out["prompt"] == msgs[:2] and out["completion"] == msgs[2:]


def test_deduplicate():
    ds = Dataset.from_list([{"a": 1}, {"a": 1}, {"a": 2}])
    assert len(deduplicate(ds)) == 2


def test_local_jsonl_spec_resolves_to_json_builder():
    spec = DatasetSpec(path=str(ROOT / "data/examples/sft_vi.jsonl"))
    assert spec.path == "json" and spec.data_files.endswith("sft_vi.jsonl")
    with pytest.raises(ValueError):
        DatasetSpec(path="x", format="nope")


def test_build_sft_mixture_from_examples():
    specs = [
        DatasetSpec(path=str(ROOT / "data/examples/sft_vi.jsonl"), format="messages"),
        DatasetSpec(path=str(ROOT / "data/examples/alpaca_vi.jsonl")),
    ]
    train, test = build_datasets(specs, DataArguments(eval_split_size=0.2), seed=0, task="sft")
    assert test is not None and len(train) + len(test) == 14
    row = train[0]["messages"]
    assert row[0]["role"] == "system" and row[-1]["role"] == "assistant"
    # an explicit system prompt in the data is preserved
    all_rows = [r["messages"] for r in train] + [r["messages"] for r in test]
    assert any(r[0]["content"] == "Bạn là trợ lí dịch thuật." for r in all_rows)


def test_build_sft_prompt_completion():
    specs = [DatasetSpec(path=str(ROOT / "data/examples/alpaca_vi.jsonl"))]
    args = DataArguments(eval_split_size=0, conversation_format="prompt_completion", system_prompt=None)
    train, test = build_datasets(specs, args, seed=0, task="sft")
    assert test is None
    assert set(train.column_names) == {"prompt", "completion"}
    assert train[0]["completion"][0]["role"] == "assistant"


def test_build_dpo():
    specs = [DatasetSpec(path=str(ROOT / "data/examples/preference_vi.jsonl"))]
    train, _ = build_datasets(specs, DataArguments(eval_split_size=0), seed=0, task="dpo")
    assert set(train.column_names) == {"prompt", "chosen", "rejected"}
    assert train[0]["prompt"][0]["role"] == "system"


def test_eval_prompts_file_is_valid():
    with open(ROOT / "data/eval/prompts_vi.jsonl", encoding="utf-8") as fh:
        assert all("prompt" in json.loads(line) for line in fh)
