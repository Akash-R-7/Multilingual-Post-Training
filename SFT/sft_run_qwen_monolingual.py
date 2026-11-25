
#!/usr/bin/env python3
"""
Full-parameter SFT for Hindi paraphrasing with Qwen3-4B-Instruct-2507.

Input JSONL schema:
  {"id": ..., "inputs": "...", "targets": "...", ...}

Uses TRL SFTTrainer with conversational prompt/completion examples.
Loss is applied to assistant completion tokens only.
"""

import argparse
import json
import os
from pathlib import Path

import torch
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from trl import SFTConfig, SFTTrainer

import json
import torch
from pathlib import Path



def generate_qualitative_outputs(
    model,
    tokenizer,
    validation_file,
    output_dir,
    num_samples=10,
    max_new_tokens=128,
):
    """Generate and save qualitative paraphrasing outputs."""

    import json
    from pathlib import Path
    import torch

    model.eval()

    if hasattr(model, "module"):
        model = model.module

    device = next(model.parameters()).device

    # Enable KV caching for inference
    original_use_cache = model.config.use_cache
    model.config.use_cache = True

    results = []

    try:
        with open(validation_file, "r", encoding="utf-8") as f:

            for line in f:
                if not line.strip():
                    continue

                item = json.loads(line)

                prompt = str(item["inputs"]).strip()
                reference = str(item["targets"]).strip()

                messages = [
                    {"role": "user", "content": prompt}
                ]

                # apply_chat_template returns BatchEncoding
                encoded = tokenizer.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )

                # Extract actual tensors
                input_ids = encoded["input_ids"].to(device)

                attention_mask = encoded.get("attention_mask")

                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)

                with torch.inference_mode():
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_new_tokens=max_new_tokens,
                        do_sample=False,
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        use_cache=True,
                    )

                # Decode only newly generated tokens
                generated_tokens = outputs[
                    0, input_ids.shape[-1]:
                ]

                prediction = tokenizer.decode(
                    generated_tokens,
                    skip_special_tokens=True,
                ).strip()

                result = {
                    "input": prompt,
                    "reference": reference,
                    "generated": prediction,
                }

                results.append(result)

                if len(results) >= num_samples:
                    break

    finally:
        model.config.use_cache = original_use_cache
        model.eval()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    result_file = output_path / "qualitative_outputs.jsonl"

    with open(result_file, "w", encoding="utf-8") as f:
        for result in results:
            f.write(
                json.dumps(result, ensure_ascii=False) + "\n"
            )

    print("\n" + "=" * 80)
    print("QUALITATIVE GENERATION RESULTS")
    print("=" * 80)

    for i, result in enumerate(results, 1):
        print(f"\nExample {i}")
        print("-" * 80)
        print("INPUT:")
        print(result["input"])

        print("\nREFERENCE:")
        print(result["reference"])

        print("\nGENERATED:")
        print(result["generated"])

        print("-" * 80)

    print(f"\nResults saved to: {result_file}")


def load_jsonl(path: str, split_name: str):
    rows = []
    seen_ids = set()

    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}")

            if "inputs" not in item or "targets" not in item:
                raise ValueError(
                    f"{path}:{line_no}: expected 'inputs' and 'targets' fields"
                )

            prompt_text = str(item["inputs"]).strip()
            target_text = str(item["targets"]).strip()

            if not prompt_text or not target_text:
                raise ValueError(
                    f"{path}:{line_no}: empty input or target"
                )

            example_id = str(item.get("id", f"{split_name}_{line_no}"))

            if example_id in seen_ids:
                raise ValueError(
                    f"{path}:{line_no}: duplicate id {example_id}"
                )

            seen_ids.add(example_id)

            rows.append({
                "id": example_id,
                "prompt": [
                    {"role": "user", "content": prompt_text}
                ],
                "completion": [
                    {"role": "assistant", "content": target_text}
                ],
            })

    if not rows:
        raise ValueError(f"No examples found in {path}")

    return rows



def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--train_file", required=True)
    parser.add_argument("--validation_file", required=True)

    parser.add_argument(
        "--model_name",
        default="/mnt/fs1/akash/Qwen-3-4B-It"
    )

    parser.add_argument(
        "--output_dir",
        default="/mnt/fs1/akash/SFT_train/outputs"
    )

    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--warmup_steps", type=int, default=0)

    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2
    )

    parser.add_argument(
        "--per_device_eval_batch_size",
        type=int,
        default=2
    )

    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4
    )

    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--save_total_limit", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)

    # W&B settings
    parser.add_argument(
        "--wandb_project",
        default="Indic_Paraphrase_SFT"
    )

    parser.add_argument(
        "--run_name",
        default="qwen3_hindi_full_sft"
    )

    parser.add_argument(
        "--disable_wandb",
        action="store_true",
        help="Disable W&B logging."
    )

    # Smoke-test settings
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Use a small subset to validate the training pipeline."
    )

    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)

    args = parser.parse_args()

    # Configure smoke test
    if args.smoke_test:

        args.max_train_samples = args.max_train_samples or 32
        args.max_eval_samples = args.max_eval_samples or 8

        args.epochs = 1.0
        args.per_device_train_batch_size = 1
        args.per_device_eval_batch_size = 1
        args.gradient_accumulation_steps = 1
        args.logging_steps = 1
        args.save_total_limit = 1

        if args.run_name == "qwen3_hindi_full_sft":
            args.run_name = "qwen3_hindi_sft_smoke_test"

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Launch with the intended GPU environment."
        )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    print(f"World size: {world_size}; local rank: {local_rank}")
    print(f"Visible GPUs: {torch.cuda.device_count()}")

    set_seed(args.seed)

    # Load datasets
    train_rows = load_jsonl(args.train_file, "train")
    eval_rows = load_jsonl(args.validation_file, "validation")

    # Check ID overlap
    train_ids = {r["id"] for r in train_rows}
    eval_ids = {r["id"] for r in eval_rows}

    overlap = train_ids.intersection(eval_ids)

    if overlap:
        raise ValueError(
            f"Train/validation ID overlap detected ({len(overlap)}): "
            f"{list(overlap)[:10]}"
        )

    print(f"Training examples: {len(train_rows)}")
    print(f"Validation examples: {len(eval_rows)}")
    print(f"Train/validation ID overlap: {len(overlap)}")

    print(
        "Dataset fields used: "
        "inputs -> user prompt; targets -> assistant completion"
    )

    # Apply sample limits
    if args.max_train_samples is not None:
        train_rows = train_rows[:args.max_train_samples]

    if args.max_eval_samples is not None:
        eval_rows = eval_rows[:args.max_eval_samples]

    if not train_rows or not eval_rows:
        raise ValueError(
            "After applying sample limits, train and validation "
            "sets must both be non-empty."
        )

    print(
        f"Examples used: train={len(train_rows)}, "
        f"validation={len(eval_rows)}"
    )

    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=True,
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    # Audit sequence lengths
    token_lengths = []

    for row in train_rows + eval_rows:

        messages = row["prompt"] + row["completion"]

        token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )

        token_lengths.append(len(token_ids["input_ids"]))

    token_lengths_sorted = sorted(token_lengths)

    p95_index = max(
        0,
        int(0.95 * len(token_lengths_sorted)) - 1
    )

    over_limit = sum(
        length > args.max_length
        for length in token_lengths
    )

    print(
        f"Token length audit (train+validation): "
        f"max={max(token_lengths)}, "
        f"p95={token_lengths_sorted[p95_index]}, "
        f"over max_length={args.max_length}: {over_limit}"
    )

    if over_limit:
        print(
            "WARNING: Some examples exceed max_length and may be "
            "truncated. Consider increasing --max_length."
        )

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

    model.config.use_cache = False

    # Initialize W&B on the main process only
    use_wandb = not args.disable_wandb

    is_main_process = (
        int(os.environ.get("RANK", "0")) == 0
    )

    if use_wandb and is_main_process:

        import wandb

        wandb.init(
            project=args.wandb_project,
            name=args.run_name,
            config={
                "model_name": args.model_name,
                "epochs": args.epochs,
                "learning_rate": args.learning_rate,
                "max_length": args.max_length,
                "per_device_train_batch_size":
                    args.per_device_train_batch_size,
                "per_device_eval_batch_size":
                    args.per_device_eval_batch_size,
                "gradient_accumulation_steps":
                    args.gradient_accumulation_steps,
                "train_examples": len(train_rows),
                "validation_examples": len(eval_rows),
                "smoke_test": args.smoke_test,
                "full_parameter_finetuning": True,
            },
        )

    # Full-parameter fine-tuning: no LoRA
    config = SFTConfig(

        output_dir=args.output_dir,
        run_name=args.run_name,

        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,

        per_device_train_batch_size=
            args.per_device_train_batch_size,

        per_device_eval_batch_size=
            args.per_device_eval_batch_size,

        gradient_accumulation_steps=
            args.gradient_accumulation_steps,

        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,

        lr_scheduler_type="cosine",
        optim="adamw_torch",

        bf16=True,
        fp16=False,

        gradient_checkpointing=True,

        gradient_checkpointing_kwargs={
            "use_reentrant": False
        },

        max_length=args.max_length,
        packing=False,
        completion_only_loss=True,

        eval_strategy="steps",
        eval_steps=20,

        save_strategy="steps",
        save_steps=20,

        logging_strategy="steps",
        logging_steps=args.logging_steps,

        save_total_limit=3,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        report_to=["wandb"] if use_wandb else [],

        ddp_find_unused_parameters=False,

        dataloader_num_workers=4,
        remove_unused_columns=False,

        seed=args.seed,
    )

    trainer = SFTTrainer(

        model=model,
        args=config,

        train_dataset=train_dataset,
        eval_dataset=eval_dataset,

        processing_class=tokenizer,
    )

    print("\nStarting full-parameter SFT...")

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Total parameters: {total_params:,}")

    # Train
    trainer.train()

    # Save final model
    final_dir = Path(args.output_dir) / "final_model"

    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))
    if trainer.is_world_process_zero():
        generate_qualitative_outputs(
            model=trainer.accelerator.unwrap_model(trainer.model),
            tokenizer=tokenizer,
            validation_file=args.validation_file,
            output_dir=args.output_dir,
            num_samples=10,
            max_new_tokens=128,
        )

    print(
        f"\nTraining complete. "
        f"Best checkpoint: {trainer.state.best_model_checkpoint}"
    )

    print(f"Final model saved to: {final_dir}")


if __name__ == "__main__":
    main()
