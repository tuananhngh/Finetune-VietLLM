"""Merge a LoRA / QLoRA adapter into its base model and save a standalone checkpoint.

    vietllm-merge --adapter outputs/qwen3-8b-bactrian --output outputs/qwen3-8b-bactrian/merged \
        [--push-to-hub your-name/qwen3-8b-vi]

The base model is reloaded in bf16 (never 4-bit) so the merged weights are full precision and can be
served by vLLM / SGLang / TGI or converted to GGUF with llama.cpp.
"""

from __future__ import annotations

import argparse
import logging

import torch

logger = logging.getLogger("vietllm.merge")


def merge_adapter(
    adapter_path: str,
    output_dir: str,
    base_model: str | None = None,
    dtype: str = "bfloat16",
    push_to_hub: bool = False,
    hub_model_id: str | None = None,
    private: bool = True,
    trust_remote_code: bool = False,
) -> str:
    from peft import AutoPeftModelForCausalLM, PeftConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = getattr(torch, dtype)
    if base_model is None:
        model = AutoPeftModelForCausalLM.from_pretrained(
            adapter_path, dtype=torch_dtype, low_cpu_mem_usage=True, trust_remote_code=trust_remote_code
        )
        base_model = PeftConfig.from_pretrained(adapter_path).base_model_name_or_path
    else:
        base = AutoModelForCausalLM.from_pretrained(
            base_model, dtype=torch_dtype, low_cpu_mem_usage=True, trust_remote_code=trust_remote_code
        )
        model = PeftModel.from_pretrained(base, adapter_path)

    logger.info("Merging adapter %s into %s", adapter_path, base_model)
    merged = model.merge_and_unload()
    merged.config.use_cache = True
    merged.save_pretrained(output_dir, safe_serialization=True, max_shard_size="5GB")

    # The fine-tuned tokenizer (with its chat template) lives next to the adapter.
    tokenizer = AutoTokenizer.from_pretrained(adapter_path, trust_remote_code=trust_remote_code)
    tokenizer.save_pretrained(output_dir)
    logger.info("Merged model saved to %s", output_dir)

    if push_to_hub:
        if not hub_model_id:
            raise ValueError("--push-to-hub needs a hub model id")
        merged.push_to_hub(hub_model_id, private=private)
        tokenizer.push_to_hub(hub_model_id, private=private)
        logger.info("Pushed to https://huggingface.co/%s", hub_model_id)
    return output_dir


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--adapter", required=True, help="Directory containing adapter_config.json")
    p.add_argument("--output", required=True, help="Directory for the merged model")
    p.add_argument("--base-model", default=None, help="Override the base model recorded in the adapter")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--push-to-hub", metavar="REPO_ID", default=None)
    p.add_argument("--public", action="store_true", help="Create a public Hub repo (default private)")
    p.add_argument("--trust-remote-code", action="store_true")
    a = p.parse_args()
    merge_adapter(
        adapter_path=a.adapter,
        output_dir=a.output,
        base_model=a.base_model,
        dtype=a.dtype,
        push_to_hub=a.push_to_hub is not None,
        hub_model_id=a.push_to_hub,
        private=not a.public,
        trust_remote_code=a.trust_remote_code,
    )


if __name__ == "__main__":
    main()
