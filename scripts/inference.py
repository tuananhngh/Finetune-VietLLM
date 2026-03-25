import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel

SYSTEM_PROMPT = (
    "Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. "
    "Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn.\n"
    "Câu trả lời của bạn không nên chứa bất kỳ nội dung gây hại, phân biệt chủng tộc, "
    "phân biệt giới tính, độc hại, nguy hiểm hoặc bất hợp pháp nào. "
    "Hãy đảm bảo rằng các câu trả lời của bạn không có thiên kiến xã hội và mang tính tích cực."
    "Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, "
    "hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. "
    "Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trẳ lời là bạn không biết "
    "và vui lòng không chia sẻ thông tin sai lệch."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Inference with fine-tuned Vistral model")
    parser.add_argument("--base-model", type=str, default="Viet-Mistral/Vistral-7B-Chat",
                        help="Base model ID from HuggingFace")
    parser.add_argument("--checkpoint", type=str, default="./vistral_finetune-bactrian-vi/checkpoint-100",
                        help="Path to LoRA checkpoint")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--top-k", type=int, default=25)
    parser.add_argument("--top-p", type=float, default=0.5)
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--interactive", action="store_true",
                        help="Run in interactive chat mode")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt to run (non-interactive)")
    return parser.parse_args()


def load_model(base_model_id, checkpoint_path):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,
        quantization_config=bnb_config,
        resume_download=True,
    )
    model = PeftModel.from_pretrained(base_model, checkpoint_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        add_bos_token=True,
        add_eos_token=True,
        max_length=256,
        padding_side="left",
    )
    tokenizer.pad_token = tokenizer.eos_token
    return model, tokenizer


def format_prompt(instruction, input_text=""):
    if input_text:
        return (
            f"### Instruction :\n\n {instruction}\n"
            f" ### Input : {input_text}\n\n ### Response :\n"
        )
    return f"### Instruction :\n\n {instruction}\n\n ### Response :"


def generate(model, tokenizer, prompt, args):
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.eos_token_id,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            do_sample=True,
            repetition_penalty=args.repetition_penalty,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def interactive_mode(model, tokenizer, args):
    print("Interactive mode — type 'quit' to exit, 'reset' to clear history.\n")
    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if user_input.strip().lower() == "quit":
            break
        if user_input.strip().lower() == "reset":
            print("— Chat reset —\n")
            continue

        prompt = format_prompt(user_input)
        response = generate(model, tokenizer, prompt, args)
        print(f"Assistant: {response}\n")


def main():
    args = parse_args()

    print(f"Loading model: {args.base_model}")
    print(f"LoRA checkpoint: {args.checkpoint}")
    model, tokenizer = load_model(args.base_model, args.checkpoint)

    if args.interactive:
        interactive_mode(model, tokenizer, args)
    else:
        text = args.prompt or "Hãy tính 30 triệu đô la ra tiền việt."
        prompt = format_prompt(text)
        response = generate(model, tokenizer, prompt, args)
        print(response)


if __name__ == "__main__":
    main()
