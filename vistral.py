import torch
import transformers
import hydra
import logging
import warnings
from omegaconf import DictConfig, OmegaConf
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from datasets import load_dataset
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
from transformers import TrainerCallback, TrainerState, TrainerControl
from accelerate import Accelerator, FullyShardedDataParallelPlugin
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig 

logging.basicConfig(level=logging.INFO)
warnings.filterwarnings("ignore")

def select_small_sample(dataset,nb_sample, split_ratio):
    data = dataset.select(range(nb_sample))
    data = data.train_test_split(test_size=split_ratio, shuffle=True)
    return data['train'], data['test']


def load_base_model(base_model_id):
    model_id = base_model_id
    bnbconfig = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnbconfig,
        resume_download=True,
    )
    return model


def formatting_function(sample):
    text = f"### Prompt Input : \n### : {sample['input']} \n### System Output : {sample['output']}"
    return text


def create_prompt(sample):
    if sample['input'] == '':
        text = f"""### Instruction : \n\n {sample['instruction']} \n\n ### Response : {sample['output']} \n"""
        return text
    else:
        text = f"""### Instruction : \n\n {sample['instruction']} \n ### Input : {sample['input']} \n\n ### Response : \n {sample['output']}"""
        return text
    
def generate_and_tokenize_prompt(text, tokenizer):
    return tokenizer(create_prompt(text))


def print_trainable_parameters(model):
    trainable_params = 0
    all_params = 0
    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Trainable parameters: {trainable_params} || All parameters: {all_params} || Trainable % : {100*trainable_params/all_params}")

def define_train_args(config: DictConfig):
    training_args = transformers.TrainingArguments(
        output_dir=config.output_dir,
        warmup_steps=config.warmup_steps,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        gradient_checkpointing=config.gradient_checkpointing,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate, # Want a small lr for finetuning
        optim=config.optimizer,
        logging_steps=config.logging_steps,              # When to start reporting loss
        logging_dir=config.logging_dir,        # Directory for storing logs
        save_strategy=config.save_strategy,       # Save the model checkpoint every logging step
        save_steps=config.save_steps,                # Save checkpoints every 50 steps
        evaluation_strategy=config.evaluation_strategy, # Evaluate the model every logging step
        eval_steps=config.eval_steps,               # Evaluate and save checkpoints every 50 steps
        do_eval=config.do_eval,                # Perform evaluation at the end of training
    )
    return training_args


def define_tokenizer(config: DictConfig):
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path=config.base_model_id,
        add_eos_token=config.add_eos_token,
        add_bos_token=config.add_bos_token,
        max_length=config.max_length,
        padding_side=config.padding_side,
        truncation=True,
        padding="max_length")
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer

def get_lora_config(config: DictConfig):
    lora = LoraConfig(
        r = config.r, 
        lora_alpha=config.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "out_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
        bias="none",
        lora_dropout=config.lora_dropout,
        task_type="causal_lm",
    )
    return lora


def define_accelerator(model):
    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    )
    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

    if torch.cuda.device_count() > 1: # If more than 1 GPU
        model.is_parallelizable = True
        model.model_parallel = True

    model = accelerator.prepare_model(model)
    return model


class PrintLossCallback(TrainerCallback):
    "A callback that prints the training and evaluation loss"

    def on_train_begin(self, args, state, control, **kwargs):
        logging.info("STARTING TRAINING")

    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        #logging.info("History : {}".format(state.log_history[-1]))
        #logging.info(f"Step: {state.global_step}, Training Loss: {state.log_history[-1]['loss']}")
        if 'eval_loss' in state.log_history[-1]:
            logging.info(f"Step: {state.global_step}, Evaluation Loss: {state.log_history[-1]['eval_loss']}")


@hydra.main(config_path="conf", config_name="config_file")
def main_fn(cfg: DictConfig):
    print(OmegaConf.to_yaml(cfg))
    project = cfg.project
    base_model_name = cfg.base_model_name
    training_args = define_train_args(cfg.training_args)
    
    tokenizer = define_tokenizer(cfg.token_args) # Define tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    
    # Model Configuration
    lora_config = get_lora_config(cfg.lora_args)
    model = load_base_model(cfg.base_model_id) #Define model
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)
    model = define_accelerator(model)
    print_trainable_parameters(model)

    # Define Dataset
    cfg_data = cfg.data_args
    data_vi = load_dataset(path=cfg_data.path_to_data,name='vi',cache_dir=cfg_data.cache_dir, split="train")
    traindata, evaldata = select_small_sample(data_vi, cfg_data.nb_samples, cfg_data.split_ratio)
    
    logging.info("TRAIN DATA STRUCTURE : {}".format(traindata))
    
    tokenized_train = traindata.map(generate_and_tokenize_prompt, fn_kwargs={"tokenizer": tokenizer})
    tokenized_eval = evaldata.map(generate_and_tokenize_prompt, fn_kwargs={"tokenizer": tokenizer})
    
    logging.info("TOKENIZED TRAIN DATA STRUCTURE : {}".format(tokenized_train))
    logging.info("TOKENIZED EVAL DATA STRUCTURE : {}".format(tokenized_eval))
    
    #Define Trainer
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
        #callbacks=[PrintLossCallback],
    )
    if cfg.training_args.train_flag:
        model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
        trainer.train(resume_from_checkpoint=False)
        
if __name__ == "__main__":
    main_fn()