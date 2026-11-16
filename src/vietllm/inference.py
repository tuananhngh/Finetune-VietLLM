"""Chat with a fine-tuned model (adapter or merged) using its own chat template.

    vietllm-chat --model outputs/qwen3-8b-bactrian            # LoRA adapter dir (base auto-detected)
    vietllm-chat --model outputs/qwen3-8b-bactrian/merged      # merged model
    vietllm-chat --model Qwen/Qwen3-8B --prompt "Thủ đô của Việt Nam là gì?"

For high-throughput serving use vLLM instead (see scripts/serve_vllm.sh).
"""

from __future__ import annotations

import argparse
import os
from threading import Thread

import torch

from vietllm.config import DEFAULT_SYSTEM_PROMPT
from vietllm.device import inference_device_and_dtype


def load_for_inference(
    model_path: str, load_in_4bit: bool = False, trust_remote_code: bool = False
):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    device, dtype = inference_device_and_dtype()
    kwargs: dict = {
        "dtype": dtype,
        # CUDA: shard across all GPUs; MPS / CPU: load straight onto the single device.
        "device_map": "auto" if device == "cuda" else device,
        "trust_remote_code": trust_remote_code,
    }
    if load_in_4bit and device != "cuda":
        print(f"--load-in-4bit needs CUDA; loading in {dtype} on {device} instead")
        load_in_4bit = False
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

    if os.path.isfile(os.path.join(model_path, "adapter_config.json")):
        from peft import AutoPeftModelForCausalLM

        model = AutoPeftModelForCausalLM.from_pretrained(model_path, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, messages, args, stream: bool = False) -> str:
    from transformers import TextIteratorStreamer

    template_kwargs = {}
    if args.enable_thinking is not None:  # Qwen3-style templates
        template_kwargs["enable_thinking"] = args.enable_thinking
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt",
        return_dict=True,
        **template_kwargs,
    ).to(model.device)
    gen_kwargs = dict(
        **inputs,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature if args.temperature > 0 else None,
        top_p=args.top_p if args.temperature > 0 else None,
        repetition_penalty=args.repetition_penalty,
        pad_token_id=tokenizer.pad_token_id,
    )
    if not stream:
        with torch.inference_mode():
            out = model.generate(**gen_kwargs)
        return tokenizer.decode(
            out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True
        ).strip()

    streamer = TextIteratorStreamer(
        tokenizer, skip_prompt=True, skip_special_tokens=True
    )
    # Start a new thread to generate the response
    thread = Thread(target=lambda: model.generate(**gen_kwargs, streamer=streamer))
    thread.start()
    chunks = []
    for piece in streamer:
        print(piece, end="", flush=True)
        chunks.append(piece)
    thread.join()
    print()
    return "".join(chunks).strip()


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--model", required=True, help="Adapter dir, merged model dir or Hub id"
    )
    p.add_argument(
        "--prompt", default=None, help="Single prompt; omit for interactive chat"
    )
    p.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    p.add_argument("--max-new-tokens", type=int, default=512)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.9)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument("--load-in-4bit", action="store_true")
    p.add_argument("--trust-remote-code", action="store_true")
    p.add_argument(
        "--enable-thinking",
        type=lambda s: s.lower() in {"1", "true", "yes"},
        default=None,
        help="Pass enable_thinking to Qwen3-style chat templates (true/false)",
    )
    return p


def main() -> None:
    args = build_parser().parse_args()
    model, tokenizer = load_for_inference(
        args.model, args.load_in_4bit, args.trust_remote_code
    )
    system = (
        [{"role": "system", "content": args.system_prompt}]
        if args.system_prompt
        else []
    )

    if args.prompt:
        print(
            generate(
                model,
                tokenizer,
                [*system, {"role": "user", "content": args.prompt}],
                args,
            )
        )
        return

    print("Chế độ trò chuyện – gõ '/reset' để xoá lịch sử, '/quit' để thoát.\n")
    history: list[dict[str, str]] = []
    while True:
        try:
            user = input("Bạn: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if user in {"/quit", "/exit"}:
            break
        if user == "/reset":
            history.clear()
            print("— đã xoá lịch sử —\n")
            continue
        if not user:
            continue
        history.append({"role": "user", "content": user})
        print("Trợ lí: ", end="", flush=True)
        reply = generate(model, tokenizer, [*system, *history], args, stream=True)
        history.append({"role": "assistant", "content": reply})
        print()


if __name__ == "__main__":
    main()
