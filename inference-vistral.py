import hydra
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from vistral import load_base_model, define_tokenizer, generate_and_tokenize_prompt
from omegaconf import DictConfig, OmegaConf

system_prompt = "Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn.\n"
system_prompt += "Câu trả lời của bạn không nên chứa bất kỳ nội dung gây hại, phân biệt chủng tộc, phân biệt giới tính, độc hại, nguy hiểm hoặc bất hợp pháp nào. Hãy đảm bảo rằng các câu trả lời của bạn không có thiên kiến xã hội và mang tính tích cực."
system_prompt += "Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trẳ lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch."

#device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
bnbconfig = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type='nf4',
    bnb_4bit_compute_dtype=torch.bfloat16
)

base_model_id = "Viet-Mistral/Vistral-7B-Chat"

tokenizer = AutoTokenizer.from_pretrained(base_model_id, 
                                          add_bos_token=True,
                                          add_eos_token=True,
                                          max_length=256,
                                          padding_side="left")
tokenizer.pad_token = tokenizer.eos_token



model = load_base_model(base_model_id)



ft_model = PeftModel.from_pretrained(model, "./vistral_finetune-bactrian-vi/checkpoint-100/")
text = "Hãy tính 30 triệu đô la ra tiền việt."
prompt = """### Instruction : \n\n {} \n\n###""".format(text)
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
outputs = ft_model.generate(
    input_ids=input_ids,
    max_new_tokens=125,
    pad_token_id=tokenizer.eos_token_id,
    temperature = 0.5,
    top_k=25,
    top_p = 0.5,
    do_sample=True,
    repetition_penalty=1.15,
)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))

# def conversation_fn(conversation):
#     input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)
#     out_ids = ft_model.generate(
#         input_ids=input_ids,
#         max_new_tokens=768,
#         do_sample=True,
#         top_p=0.95,
#         top_k=40,
#         temperature=0.1,
#         repetition_penalty=1.05,
#     )
#     assistant = tokenizer.batch_decode(out_ids[:, input_ids.size(1): ], skip_special_tokens=True)[0].strip()
#     return assistant

# def main_fn():
#     system_prompt = "Bạn là một trợ lí Tiếng Việt nhiệt tình và trung thực. Hãy luôn trả lời một cách hữu ích nhất có thể, đồng thời giữ an toàn.\n"
#     system_prompt += "Câu trả lời của bạn không nên chứa bất kỳ nội dung gây hại, phân biệt chủng tộc, phân biệt giới tính, độc hại, nguy hiểm hoặc bất hợp pháp nào. Hãy đảm bảo rằng các câu trả lời của bạn không có thiên kiến xã hội và mang tính tích cực."
#     system_prompt += "Nếu một câu hỏi không có ý nghĩa hoặc không hợp lý về mặt thông tin, hãy giải thích tại sao thay vì trả lời một điều gì đó không chính xác. Nếu bạn không biết câu trả lời cho một câu hỏi, hãy trẳ lời là bạn không biết và vui lòng không chia sẻ thông tin sai lệch."

#     conversation = [{"role": "system", "content": system_prompt }]
    
#     while True:
#         human = input("Human: ")
#         if human.lower() == "reset":
#             conversation = [{"role": "system", "content": system_prompt }]
#             print("The chat history has been cleared!")
#             continue

#         conversation.append({"role": "user", "content": human })
#         input_ids = tokenizer.apply_chat_template(conversation, return_tensors="pt").to(model.device)
        
#         out_ids = ft_model.generate(
#             input_ids=input_ids,
#             max_new_tokens=768,
#             do_sample=True,
#             top_p=0.95,
#             top_k=40,
#             temperature=0.1,
#             repetition_penalty=1.05,
#         )
#         assistant = tokenizer.batch_decode(out_ids[:, input_ids.size(1): ], skip_special_tokens=True)[0].strip()
#         print("Assistant: ", assistant) 
#         conversation.append({"role": "assistant", "content": assistant })
