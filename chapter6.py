"""Compare Trainer pretraining, full SFT, and PEFT using one tiny Llama base."""

import argparse
import json
import os
from pathlib import Path

import torch
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    BitsAndBytesConfig, LlamaConfig, LlamaForCausalLM, PreTrainedTokenizerFast,
    Trainer, TrainingArguments, set_seed,
)

from common import CONTEXT, SEED, Collator, examples, generations, train_tokenizer


def quantized_base(path):
    return LlamaForCausalLM.from_pretrained(
        path, local_files_only=True, device_map={"": torch.cuda.current_device()},
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.float16,
        ),
    )


def train_stage(model, tokenizer, stage, root, steps, learning_rate, device, deepspeed=None):
    sft = stage != "pretrain"
    train = examples(tokenizer, sft=sft)
    valid = examples(tokenizer, sft=sft, evaluate=True)
    output = root / stage
    trainer = Trainer(
        model=model,
        args=TrainingArguments(
            output_dir=str(output), use_cpu=device == "cpu", max_steps=steps,
            per_device_train_batch_size=8, per_device_eval_batch_size=8,
            learning_rate=learning_rate, lr_scheduler_type="constant", max_grad_norm=1.0,
            save_strategy="no", logging_steps=25, report_to=[], disable_tqdm=True,
            seed=SEED, label_names=["labels"], deepspeed=deepspeed,
        ),
        data_collator=Collator(tokenizer), train_dataset=train, eval_dataset=valid,
    )
    params = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    if deepspeed:
        # Trainer creates the DeepSpeed engine when training starts.
        before = None
    else:
        before = trainer.evaluate()["eval_loss"]
    training = trainer.train()
    after = trainer.evaluate()["eval_loss"]
    trainer.save_model(output)
    return {"eval_before": before, "eval_after": after, "steps": training.global_step,
            "train_loss": training.training_loss, "learning_rate": learning_rate,
            "train_examples": len(train), "eval_examples": len(valid),
            "parameters": params, "trainable_parameters": trainable,
            "trainable_percent": 100 * trainable / params}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/chapter6"))
    parser.add_argument("--pretrain-steps", type=int, default=100)
    parser.add_argument("--sft-steps", type=int, default=100)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda"))
    parser.add_argument("--qlora", action="store_true", help="Also run a CUDA NF4 adapter experiment")
    parser.add_argument("--deepspeed", help="Optional ZeRO-2 JSON for Trainer pretraining")
    args = parser.parse_args()
    if min(args.pretrain_steps, args.sft_steps) < 1:
        parser.error("Step counts must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    if (args.qlora or args.deepspeed) and args.device != "cuda":
        parser.error("Use --device cuda with QLoRA or DeepSpeed")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        parser.error("This experiment supports a single process")
    torch.set_num_threads(2)
    set_seed(SEED)
    tokenizer = train_tokenizer()
    tokenizer.save_pretrained(args.output)
    tokenizer = PreTrainedTokenizerFast.from_pretrained(args.output)
    config = LlamaConfig(
        vocab_size=len(tokenizer), hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=CONTEXT, tie_word_embeddings=True, use_cache=False,
        pad_token_id=tokenizer.pad_token_id, bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model = LlamaForCausalLM(config)
    model.config.loss_type = "ForCausalLM"
    metrics = {"device": args.device, "seed": SEED}
    metrics["pretrain"] = train_stage(model, tokenizer, "pretrain", args.output,
                                     args.pretrain_steps, 3e-3, args.device, args.deepspeed)
    base_path = args.output / "pretrain"
    del model
    for stage in ("full_sft", "lora", *("qlora",) * args.qlora):
        set_seed(SEED)
        if stage == "qlora":
            model = prepare_model_for_kbit_training(quantized_base(base_path), use_gradient_checkpointing=False)
        else:
            model = LlamaForCausalLM.from_pretrained(base_path, local_files_only=True)
        model.config.loss_type = "ForCausalLM"
        if stage == "full_sft":
            metrics["pretrain"]["generation"] = generations(model, tokenizer, False)
            metrics["base_instruction_generation"] = generations(model, tokenizer, True)
        else:
            model = get_peft_model(model, LoraConfig(
                task_type="CAUSAL_LM", r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            ))
        metrics[stage] = train_stage(model, tokenizer, stage, args.output,
                                     args.sft_steps, 2e-3, args.device)
        del model
        checkpoint = args.output / stage
        if stage == "full_sft":
            restored = LlamaForCausalLM.from_pretrained(checkpoint, local_files_only=True)
        else:
            base = (quantized_base(base_path) if stage == "qlora"
                    else LlamaForCausalLM.from_pretrained(base_path, local_files_only=True))
            restored = PeftModel.from_pretrained(base, checkpoint, local_files_only=True)
        if stage != "qlora":
            restored.to(args.device)
        metrics[stage]["generation"] = generations(restored, tokenizer, True)
        del restored

    (args.output / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
