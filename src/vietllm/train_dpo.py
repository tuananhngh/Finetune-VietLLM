"""Preference optimisation (DPO and its variants) on top of an SFT model.

    python -m vietllm.train_dpo --config configs/dpo/qwen3-8b-dpo.yaml

Point ``model_name_or_path`` at a merged SFT model (``vietllm-merge``) or any chat model. With LoRA, the
reference model is the frozen base (adapter disabled), so no second copy of the weights is loaded.
Other losses are available through ``loss_type`` (e.g. ``ipo``, ``sigmoid``, ``apo_zero``, ``sppo_hard``).
"""

from __future__ import annotations

import logging

from accelerate import PartialState
from trl import DPOConfig, DPOTrainer, ModelConfig

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

logger = logging.getLogger("vietllm.dpo")


def main(argv: list[str] | None = None) -> None:
    model_args, training_args, data_args, run_args, specs = parse_args(
        (ModelConfig, DPOConfig, DataArguments, RunArguments), argv
    )
    state = PartialState()
    setup_logging(state.is_main_process)

    tokenizer = load_tokenizer(model_args, training_args.trust_remote_code)
    with state.main_process_first():
        train_ds, eval_ds = build_datasets(
            specs, data_args, training_args.seed, task="dpo"
        )
    if eval_ds is None:
        training_args.eval_strategy = "no"

    quant_config, peft_config = quantization_and_peft(model_args)
    training_args.model_init_kwargs = model_init_kwargs(model_args)

    # Build DPO Trainer
    trainer = DPOTrainer(
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

    # Train the model
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

    # Save the model and push to Hub
    trainer.model.config.use_cache = (
        True  # Re-enable KV cache so the saved config is ready for inference.
    )
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=specs[0].path)

    state.wait_for_everyone()  # Wait for all processes to finish
    # Only merge the model on the main process
    if (
        run_args.merge_after_training
        and peft_config is not None
        and state.is_main_process
    ):
        import os

        from vietllm.merge import merge_adapter

        merge_adapter(
            adapter_path=training_args.output_dir,
            output_dir=run_args.merged_output_dir
            or os.path.join(training_args.output_dir, "merged"),
            push_to_hub=run_args.push_merged_to_hub,
            hub_model_id=(
                (training_args.hub_model_id + "-merged")
                if training_args.hub_model_id
                else None
            ),
            trust_remote_code=training_args.trust_remote_code,
        )


if __name__ == "__main__":
    main()
