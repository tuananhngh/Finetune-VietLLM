"""Evaluate a fine-tuned model.

Two modes:

``generate`` – run a fixed set of Vietnamese prompts (data/eval/prompts_vi.jsonl) through one or more models
and write a side-by-side JSONL/Markdown report. Cheap, and the quickest way to compare base vs fine-tuned.

    vietllm-eval generate --models Qwen/Qwen3-8B outputs/qwen3-8b-bactrian --output reports/compare

``benchmark`` – run Vietnamese benchmarks with lm-evaluation-harness. The ``hf`` backend needs
``uv sync --extra eval``; the ``vllm`` backend runs lm-eval + vLLM in an isolated ``uvx`` environment
(vLLM pins its own torch / transformers versions, so it is kept out of the training environment).

    vietllm-eval benchmark --model outputs/qwen3-8b-bactrian/merged --tasks arc_vi,hellaswag_vi,m_mmlu_vi

Task names change between lm-eval releases; list yours with ``lm_eval --tasks list | grep vi``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_PROMPTS = Path(__file__).resolve().parents[2] / "data" / "eval" / "prompts_vi.jsonl"
DEFAULT_TASKS = "arc_vi,hellaswag_vi,m_mmlu_vi"
LM_EVAL_VERSION = "0.4.13"


def run_generate(args: argparse.Namespace) -> None:
    import gc

    import torch

    from .inference import build_parser, generate, load_for_inference

    with open(args.prompts, encoding="utf-8") as fh:
        prompts = [json.loads(line) for line in fh if line.strip()]
    gen_args = build_parser().parse_args(
        ["--model", "x", "--temperature", "0", "--max-new-tokens", str(args.max_new_tokens)]
    )
    results: dict[str, list[str]] = {}
    for model_path in args.models:
        model, tok = load_for_inference(model_path, load_in_4bit=args.load_in_4bit)
        outs = []
        for item in prompts:
            messages = [
                {"role": "system", "content": gen_args.system_prompt},
                {"role": "user", "content": item["prompt"]},
            ]
            outs.append(generate(model, tok, messages, gen_args))
            print(f"[{model_path}] {item['prompt'][:60]}… -> {outs[-1][:80]}…")
        results[model_path] = outs
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif torch.backends.mps.is_available():
            torch.mps.empty_cache()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "generations.jsonl", "w", encoding="utf-8") as fh:
        for i, item in enumerate(prompts):
            row = {**item, "outputs": {m: results[m][i] for m in args.models}}
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    with open(out / "generations.md", "w", encoding="utf-8") as fh:
        for i, item in enumerate(prompts):
            fh.write(f"## {i + 1}. {item['prompt']}\n\n")
            for m in args.models:
                fh.write(f"**{m}**\n\n{results[m][i]}\n\n")
    print(f"Report written to {out}/generations.md")


def run_benchmark(args: argparse.Namespace) -> None:
    model_path = args.model
    is_adapter = os.path.isfile(os.path.join(model_path, "adapter_config.json"))
    if args.backend == "vllm" and not is_adapter:
        model_args = f"pretrained={model_path},dtype=bfloat16,gpu_memory_utilization=0.85"
    else:
        if is_adapter:
            from peft import PeftConfig

            base = PeftConfig.from_pretrained(model_path).base_model_name_or_path
            model_args = f"pretrained={base},peft={model_path},dtype=bfloat16"
        else:
            model_args = f"pretrained={model_path},dtype=bfloat16"
        args.backend = "hf"
    if args.backend == "vllm":
        if shutil.which("uvx") is None:
            sys.exit("uvx not found: install uv (https://docs.astral.sh/uv/)")
        launcher = ["uvx", "--python", "3.12", "--from", f"lm-eval[vllm]=={LM_EVAL_VERSION}", "lm_eval"]
    elif shutil.which("lm_eval") is None:
        sys.exit("lm_eval not found: uv sync --extra eval")
    else:
        launcher = ["lm_eval"]
    cmd = [
        *launcher,
        "--model",
        args.backend,
        "--model_args",
        model_args,
        "--tasks",
        args.tasks,
        "--batch_size",
        "auto",
        "--output_path",
        args.output,
    ]
    if args.apply_chat_template:
        cmd.append("--apply_chat_template")
    if args.limit:
        cmd += ["--limit", str(args.limit)]
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("generate", help="Side-by-side generations on Vietnamese prompts")
    g.add_argument("--models", nargs="+", required=True)
    g.add_argument("--prompts", default=str(DEFAULT_PROMPTS))
    g.add_argument("--output", default="reports/generations")
    g.add_argument("--max-new-tokens", type=int, default=384)
    g.add_argument("--load-in-4bit", action="store_true")

    b = sub.add_parser("benchmark", help="lm-evaluation-harness Vietnamese benchmarks")
    b.add_argument("--model", required=True)
    b.add_argument("--tasks", default=DEFAULT_TASKS)
    b.add_argument(
        "--backend",
        choices=["vllm", "hf"],
        default="vllm",
        help="vllm for merged models (fast); adapters always use hf",
    )
    b.add_argument("--output", default="reports/benchmark")
    b.add_argument("--limit", type=int, default=None, help="Examples per task (quick sanity runs)")
    b.add_argument("--apply-chat-template", action="store_true")

    args = p.parse_args()
    run_generate(args) if args.cmd == "generate" else run_benchmark(args)


if __name__ == "__main__":
    main()
