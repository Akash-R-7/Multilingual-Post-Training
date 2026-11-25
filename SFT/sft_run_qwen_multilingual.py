#!/usr/bin/env python3
"""
Full-parameter multilingual SFT for Hindi, Bengali, and Telugu paraphrasing.

Expected JSONL fields:
    inputs: user instruction + source sentence
    targets: reference paraphrase
Optional metadata:
    id, example_id, language, script, dataset_name, sub_dataset_name, ...

The script uses conversational prompt-completion examples and completion-only loss.
Only assistant target tokens contribute to the supervised loss.
"""

import argparse
import json
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed,
)
from trl import SFTConfig, SFTTrainer


QUOTE_PAIRS = {
    '"': '"',
    "'": "'",
    "“": "”",
    "‘": "’",
    "«": "»",
}


def strip_matching_outer_quotes(text):
    """Remove one matching pair of wrapper quotes, if present."""
    value = str(text).strip()
    if len(value) >= 2:
        first = value[0]
        last = value[-1]
        if first in QUOTE_PAIRS and QUOTE_PAIRS[first] == last:
            return value[1:-1].strip(), True
    return value, False


def load_jsonl(path, split_name, strip_target_quotes=True):
    """
    Read and validate JSONL. Return:
      - trainer_rows: only prompt/completion columns for TRL
      - metadata_rows: language/id/input/reference for diagnostics and generation
      - summary: counts and normalization diagnostics
    """
    trainer_rows = []
    metadata_rows = []
    seen_ids = set()
    language_counts = Counter()
    quote_stripped = Counter()
    missing_language = 0

    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc

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

            # Prefer example_id because it is language-prefixed in this dataset.
            example_id = str(
                item.get("example_id", item.get("id", f"{split_name}_{line_no}"))
            )

            if example_id in seen_ids:
                raise ValueError(
                    f"{path}:{line_no}: duplicate example identifier {example_id}"
                )
            seen_ids.add(example_id)

            language = str(item.get("language", "unknown")).strip().lower()
            if not language or language == "none":
                language = "unknown"
                missing_language += 1
            language_counts[language] += 1

            if strip_target_quotes:
                target_text, stripped = strip_matching_outer_quotes(target_text)
                if stripped:
                    quote_stripped[language] += 1

            if not target_text:
                raise ValueError(
                    f"{path}:{line_no}: target became empty after quote normalization"
                )

            # Keep trainer columns minimal; metadata is held separately.
            trainer_rows.append({
                "prompt": [{"role": "user", "content": prompt_text}],
                "completion": [{"role": "assistant", "content": target_text}],
            })

            metadata_rows.append({
                "id": example_id,
                "language": language,
                "input": prompt_text,
                "reference": target_text,
            })

    if not trainer_rows:
        raise ValueError(f"No usable examples found in {path}")

    summary = {
        "examples": len(trainer_rows),
        "language_counts": dict(language_counts),
        "quote_stripped_by_language": dict(quote_stripped),
        "missing_language_metadata": missing_language,
    }
    return trainer_rows, metadata_rows, summary


def print_language_distribution(title, summary):
    total = summary["examples"]
    print(f"\n{title}: {total:,} examples")
    for language, count in sorted(summary["language_counts"].items()):
        print(f"  {language:>8}: {count:>6,} ({100.0 * count / total:5.1f}%)")

    if summary["quote_stripped_by_language"]:
        print("  Outer target quotes removed:")
        for language, count in sorted(
            summary["quote_stripped_by_language"].items()
        ):
            print(f"    {language}: {count:,}")

    if summary["missing_language_metadata"]:
        print(
            "  WARNING: "
            f"{summary['missing_language_metadata']} examples lack language metadata."
        )


def get_token_ids(tokenizer, messages):
    """Tokenize a chat conversation without adding a generation prompt."""
    encoded = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=False,
    )
    # BatchEncoding is mapping-like but is not guaranteed to be a plain dict.
    # Prefer its input_ids field; otherwise accept a raw token-id list/tensor.
    if hasattr(encoded, "input_ids"):
        token_ids = encoded.input_ids
    elif isinstance(encoded, dict) and "input_ids" in encoded:
        token_ids = encoded["input_ids"]
    else:
        token_ids = encoded

    if isinstance(token_ids, torch.Tensor):
        token_ids = token_ids.tolist()

    # Handle a possible single-item batch.
    if token_ids and isinstance(token_ids[0], list):
        token_ids = token_ids[0]

    return token_ids


def audit_sequence_lengths(
    tokenizer,
    train_metadata,
    eval_metadata,
    max_length,
    output_dir,
):
    """Audit full prompt + completion lengths, globally and by language."""
    all_records = []
    for split_name, records in (
        ("train", train_metadata),
        ("validation", eval_metadata),
    ):
        for record in records:
            messages = [
                {"role": "user", "content": record["input"]},
                {"role": "assistant", "content": record["reference"]},
            ]
            token_ids = get_token_ids(tokenizer, messages)
            all_records.append({
                "split": split_name,
                "id": record["id"],
                "language": record["language"],
                "length": len(token_ids),
            })

    def percentile(values, fraction):
        values = sorted(values)
        if not values:
            return 0
        index = min(len(values) - 1, max(0, int(fraction * len(values)) - 1))
        return values[index]

    grouped = defaultdict(list)
    for record in all_records:
        grouped[(record["split"], record["language"])].append(record["length"])

    print(f"\nSequence-length audit (max_length={max_length})")
    audit_summary = {}
    for (split_name, language), lengths in sorted(grouped.items()):
        over = sum(length > max_length for length in lengths)
        stats = {
            "count": len(lengths),
            "p50": percentile(lengths, 0.50),
            "p95": percentile(lengths, 0.95),
            "max": max(lengths),
            "over_max_length": over,
        }
        audit_summary[f"{split_name}_{language}"] = stats
        print(
            f"  {split_name:10s} {language:>8s}: "
            f"n={stats['count']:4d}, p50={stats['p50']:4d}, "
            f"p95={stats['p95']:4d}, max={stats['max']:4d}, "
            f"over_limit={over:4d}"
        )

    total_over = sum(
        record["length"] > max_length for record in all_records
    )
    if total_over:
        print(
            f"WARNING: {total_over}/{len(all_records)} examples exceed "
            f"max_length={max_length}. They may be truncated by the trainer. "
            "Consider increasing --max_length after inspecting the long examples."
        )
    else:
        print("All audited examples fit within max_length.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "sequence_length_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_summary, f, ensure_ascii=False, indent=2)


def choose_qualitative_samples(records, languages, per_language, seed):
    """Select a deterministic, language-stratified qualitative sample."""
    rng = random.Random(seed)
    selected = []

    for language in languages:
        candidates = [r for r in records if r["language"] == language]
        if not candidates:
            print(
                f"WARNING: no validation examples found for language {language}; "
                "skipping its qualitative samples."
            )
            continue
        rng.shuffle(candidates)
        selected.extend(candidates[: min(per_language, len(candidates))])

    # Include any additional language labels, if present, rather than silently
    # excluding them from qualitative inspection.
    known = set(languages)
    extra_languages = sorted({
        r["language"] for r in records if r["language"] not in known
    })
    for language in extra_languages:
        candidates = [r for r in records if r["language"] == language]
        rng.shuffle(candidates)
        selected.extend(candidates[:per_language])

    return selected


def generate_qualitative_outputs(
    model,
    tokenizer,
    validation_metadata,
    output_dir,
    languages,
    per_language=10,
    max_new_tokens=256,
    seed=42,
):
    """Generate a deterministic, language-stratified qualitative sample."""
    model.eval()
    device = next(model.parameters()).device
    original_use_cache = model.config.use_cache
    model.config.use_cache = True

    selected = choose_qualitative_samples(
        validation_metadata,
        languages,
        per_language,
        seed,
    )
    results = []

    try:
        for record in selected:
            messages = [{"role": "user", "content": record["input"]}]
            encoded = tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_tensors="pt",
                return_dict=True,
            )
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

            generated_tokens = outputs[0, input_ids.shape[-1]:]
            prediction = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True,
            ).strip()

            results.append({
                "id": record["id"],
                "language": record["language"],
                "input": record["input"],
                "reference": record["reference"],
                "generated": prediction,
            })

    finally:
        model.config.use_cache = original_use_cache
        model.eval()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_file = output_dir / "qualitative_outputs_by_language.jsonl"

    with open(result_file, "w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False) + "\n")

    print("\nQUALITATIVE GENERATION")
    print("=" * 80)
    for language in languages:
        count = sum(r["language"] == language for r in results)
        print(f"{language}: {count} examples")
    print(f"Saved: {result_file}")

    for index, result in enumerate(results, start=1):
        print("\n" + "-" * 80)
        print(
            f"Example {index} | language={result['language']} | "
            f"id={result['id']}"
        )
        print("INPUT:", result["input"])
        print("REFERENCE:", result["reference"])
        print("GENERATED:", result["generated"])

    return result_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full-parameter multilingual Qwen SFT for paraphrasing."
    )

    parser.add_argument("--train_file", required=True)
    parser.add_argument("--validation_file", required=True)
    parser.add_argument(
        "--model_name",
        default="/mnt/fs1/akash/Qwen-3-4B-It",
    )
    parser.add_argument(
        "--output_dir",
        default="/mnt/fs1/akash/SFT_train/mixed_hi_bn_te",
    )

    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_length", type=int, default=1024)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.01)

    parser.add_argument("--per_device_train_batch_size", type=int, default=2)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=2)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)

    parser.add_argument("--logging_steps", type=int, default=5)
    parser.add_argument("--eval_steps", type=int, default=50)
    parser.add_argument("--save_steps", type=int, default=50)
    parser.add_argument("--save_total_limit", type=int, default=3)

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--expected_languages",
        default="hin,ben,tel",
        help="Comma-separated language codes expected in both train and validation.",
    )
    parser.add_argument(
        "--preserve_target_quotes",
        action="store_true",
        help="Do not strip one matching pair of outer quotes from target strings.",
    )

    parser.add_argument(
        "--wandb_project",
        default="Indic_Paraphrase_SFT",
    )
    parser.add_argument(
        "--run_name",
        default="qwen3_mixed_hi_bn_te_full_sft",
    )
    parser.add_argument("--disable_wandb", action="store_true")

    parser.add_argument("--smoke_test", action="store_true")
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)

    parser.add_argument(
        "--qualitative_per_language",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--qualitative_max_new_tokens",
        type=int,
        default=256,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.smoke_test:
        args.max_train_samples = args.max_train_samples or 32
        args.max_eval_samples = args.max_eval_samples or 12
        args.epochs = 1.0
        args.per_device_train_batch_size = 1
        args.per_device_eval_batch_size = 1
        args.gradient_accumulation_steps = 1
        args.logging_steps = 1
        args.eval_steps = 5
        args.save_steps = 5
        args.save_total_limit = 1
        if args.run_name == "qwen3_mixed_hi_bn_te_full_sft":
            args.run_name = "qwen3_mixed_hi_bn_te_sft_smoke"

    if args.max_length <= 0:
        raise ValueError("--max_length must be positive.")
    if args.eval_steps <= 0 or args.save_steps <= 0:
        raise ValueError("--eval_steps and --save_steps must be positive.")
    if args.save_steps % args.eval_steps != 0 and args.eval_steps % args.save_steps != 0:
        raise ValueError(
            "For load_best_model_at_end, save_steps and eval_steps must be "
            "multiples of one another."
        )

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Launch with the intended GPU environment."
        )

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))

    print(f"World size: {world_size}; local rank: {local_rank}; rank: {rank}")
    print(f"Visible GPUs: {torch.cuda.device_count()}")

    set_seed(args.seed)
    expected_languages = [
        item.strip().lower()
        for item in args.expected_languages.split(",")
        if item.strip()
    ]
    strip_target_quotes = not args.preserve_target_quotes

    # Load and validate both datasets.
    train_rows, train_metadata, train_summary = load_jsonl(
        args.train_file,
        "train",
        strip_target_quotes=strip_target_quotes,
    )
    eval_rows, eval_metadata, eval_summary = load_jsonl(
        args.validation_file,
        "validation",
        strip_target_quotes=strip_target_quotes,
    )

    print_language_distribution("TRAIN", train_summary)
    print_language_distribution("VALIDATION", eval_summary)

    for language in expected_languages:
        if train_summary["language_counts"].get(language, 0) == 0:
            raise ValueError(
                f"Expected language {language} is absent from training data."
            )
        if eval_summary["language_counts"].get(language, 0) == 0:
            raise ValueError(
                f"Expected language {language} is absent from validation data."
            )

    train_ids = {record["id"] for record in train_metadata}
    eval_ids = {record["id"] for record in eval_metadata}
    overlap = train_ids.intersection(eval_ids)
    if overlap:
        raise ValueError(
            f"Train/validation ID overlap detected ({len(overlap)}): "
            f"{list(overlap)[:10]}"
        )
    print(f"\nTrain/validation ID overlap: {len(overlap)}")

    # Apply optional limits consistently to trainer data and metadata.
    if args.max_train_samples is not None:
        train_rows = train_rows[:args.max_train_samples]
        train_metadata = train_metadata[:args.max_train_samples]
    if args.max_eval_samples is not None:
        eval_rows = eval_rows[:args.max_eval_samples]
        eval_metadata = eval_metadata[:args.max_eval_samples]

    if not train_rows or not eval_rows:
        raise ValueError(
            "After applying sample limits, train and validation must both be non-empty."
        )

    print(
        f"Examples used: train={len(train_rows):,}, "
        f"validation={len(eval_rows):,}"
    )

    # Create tokenizer and audit actual model-tokenized sequence lengths.
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                "Tokenizer has neither pad_token nor eos_token; configure padding explicitly."
            )
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    audit_sequence_lengths(
        tokenizer,
        train_metadata,
        eval_metadata,
        args.max_length,
        args.output_dir,
    )

    # Trainer receives only the conversational prompt/completion columns.
    train_dataset = Dataset.from_list(train_rows)
    eval_dataset = Dataset.from_list(eval_rows)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if getattr(model.config, "pad_token_id", None) is None:
        model.config.pad_token_id = tokenizer.pad_token_id

    os.environ.setdefault("WANDB_PROJECT", args.wandb_project)
    report_to = "none" if args.disable_wandb else "wandb"

    config = SFTConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,

        num_train_epochs=args.epochs,
        learning_rate=args.learning_rate,

        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,

        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type="cosine",
        optim="adamw_torch",

        bf16=True,
        fp16=False,

        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},

        max_length=args.max_length,
        packing=False,
        completion_only_loss=True,

        eval_strategy="steps",
        eval_steps=args.eval_steps,

        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,

        logging_strategy="steps",
        logging_steps=args.logging_steps,

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,

        report_to=report_to,

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

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total_params = sum(parameter.numel() for parameter in model.parameters())

    print("\n" + "=" * 80)
    print("MULTILINGUAL FULL-PARAMETER SFT")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max sequence length: {args.max_length}")
    print("Loss: assistant completion tokens only")
    print(f"Target outer-quote normalization: {strip_target_quotes}")
    print(f"Run name: {args.run_name}")
    print("=" * 80)

    trainer.train()

    # With load_best_model_at_end=True, this saves the best checkpoint's weights.
    final_dir = Path(args.output_dir) / "final_model"
    trainer.save_model(str(final_dir))
    if trainer.is_world_process_zero():
        tokenizer.save_pretrained(str(final_dir))

        unwrapped_model = trainer.accelerator.unwrap_model(trainer.model)
        generate_qualitative_outputs(
            model=unwrapped_model,
            tokenizer=tokenizer,
            validation_metadata=eval_metadata,
            output_dir=args.output_dir,
            languages=expected_languages,
            per_language=args.qualitative_per_language,
            max_new_tokens=args.qualitative_max_new_tokens,
            seed=args.seed,
        )

    if trainer.is_world_process_zero():
        print(
            f"\nTraining complete. Best checkpoint: "
            f"{trainer.state.best_model_checkpoint}"
        )
        print(f"Final model saved to: {final_dir}")


if __name__ == "__main__":
    main()
