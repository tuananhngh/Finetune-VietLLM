"""Supervised fine-tuning entry point.

Single GPU:   python -m vietllm.train_sft --config configs/sft/qwen3-8b-qlora-bactrian.yaml
Multi GPU:    accelerate launch --config_file configs/accelerate/multi_gpu.yaml \
                  -m vietllm.train_sft --config configs/sft/qwen3-8b-lora-mix.yaml
Override:     ... --learning_rate 1e-4 --max_steps 200 --output_dir outputs/test
"""

from __future__ import annotations

import logging
import os

from accelerate import PartialState
from trl import ModelConfig, SFTConfig, SFTTrainer

from vietllm.config import DataArguments, RunArguments, parse_args
from vietllm.data import build_datasets
from vietllm.modeling import (
    find_resume_checkpoint,
    load_tokenizer,
    log_trainable_parameters,
    model_init_kwargs,
    quantization_and_peft,
    setup_logging,
)

logger = logging.getLogger("vietllm.sft")


def main(argv: list[str] | None = None) -> None:
    model_args, training_args, data_args, run_args, specs = parse_args(
        (ModelConfig, SFTConfig, DataArguments, RunArguments), argv
    )
    state = PartialState()
    setup_logging(state.is_main_process)

    if data_args.conversation_format == "prompt_completion" and training_args.assistant_only_loss:
        raise ValueError(
            "Use completion_only_loss (default) with conversation_format=prompt_completion, "
            "not assistant_only_loss"
        )

    tokenizer = load_tokenizer(model_args, training_args.trust_remote_code)
    if tokenizer.chat_template is None and training_args.chat_template_path is None:
        raise ValueError(
            f"{model_args.model_name_or_path} has no chat template. Use an instruct/chat checkpoint or set "
            "chat_template_path (e.g. to a .jinja file or another model id such as Qwen/Qwen3-8B)."
        )

    # Preprocess once on the main process; other ranks read the datasets cache.
    with state.main_process_first():
        train_ds, eval_ds = build_datasets(specs, data_args, training_args.seed, task="sft")
    if eval_ds is None:
        training_args.eval_strategy = "no"

    quant_config, peft_config = quantization_and_peft(model_args)
    training_args.model_init_kwargs = model_init_kwargs(model_args)

    # Build SFT Trainer
    trainer = SFTTrainer(
        model=model_args.model_name_or_path,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        quantization_config=quant_config,
        peft_config=peft_config,
    )
    # Only log trainable parameters on the main process
    if state.is_main_process:
        log_trainable_parameters(trainer.model)
        example = train_ds[0]
        logger.info("Example training row:\n%s", example)

    checkpoint = find_resume_checkpoint(run_args.resume, training_args.output_dir)
    result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.log_metrics("train", result.metrics)
    trainer.save_metrics("train", result.metrics)
    trainer.save_state()

    # Evaluate the model
    if eval_ds is not None:
        metrics = trainer.evaluate()
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    # Re-enable KV cache so the saved config is ready for inference.
    trainer.model.config.use_cache = True
    trainer.save_model(training_args.output_dir)
    logger.info("Model / adapter saved to %s", training_args.output_dir)

    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=specs[0].path)

    state.wait_for_everyone()  # Wait for all processes to finish
    # Only merge the model on the main process
    if run_args.merge_after_training and peft_config is not None and state.is_main_process:
        from vietllm.merge import merge_adapter

        merge_adapter(
            adapter_path=training_args.output_dir,
            output_dir=run_args.merged_output_dir or os.path.join(training_args.output_dir, "merged"),
            push_to_hub=run_args.push_merged_to_hub,
            hub_model_id=((training_args.hub_model_id + "-merged") if training_args.hub_model_id else None),
            trust_remote_code=training_args.trust_remote_code,
        )


if __name__ == "__main__":
    main()
