#!/usr/bin/env python3
"""
Full-parameter MONOLINGUAL SFT for Param-1B (Llama-family).

Targets contain ONLY reference paraphrases. Reasoning traces are not added to
the SFT targets. The script verifies the actual token labels produced by TRL
before training and aborts if <think>/</think> tokens are included in loss.

Expected JSONL fields:
    inputs: user instruction + source sentence
    targets: reference paraphrase
The selected file must contain only --language.
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
    max_new_tokens=512,
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
                    return_dict_in_generate=True,
                )

            sequences = outputs.sequences if hasattr(outputs, "sequences") else outputs
            generated_tokens = sequences[0, input_ids.shape[-1]:]
            raw_completion = tokenizer.decode(
                generated_tokens,
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ).strip()

            reasoning, final_answer, parse_status = split_reasoning_and_answer(
                raw_completion
            )
            # Keep raw_completion unchanged; strip literal EOS/PAD from parsed answer.
            final_answer = clean_decoded_answer(final_answer, tokenizer)

            eos_ids = tokenizer.eos_token_id
            if eos_ids is None:
                eos_ids = []
            elif isinstance(eos_ids, int):
                eos_ids = [eos_ids]
            else:
                eos_ids = list(eos_ids)

            generated_id_list = generated_tokens.detach().cpu().tolist()
            eos_terminated = bool(
                generated_id_list and generated_id_list[-1] in eos_ids
            )
            reached_token_limit = (
                len(generated_id_list) >= max_new_tokens and not eos_terminated
            )

            if parse_status == "incomplete_reasoning":
                generation_status = "incomplete_reasoning"
            elif parse_status == "missing_final_answer":
                generation_status = "missing_final_answer"
            elif parse_status == "malformed_think_block":
                generation_status = "malformed_think_block"
            elif reached_token_limit:
                generation_status = "max_new_tokens_reached"
            elif eos_terminated:
                generation_status = "completed"
            else:
                generation_status = "stopped_without_eos"

            results.append({
                "id": record["id"],
                "language": record["language"],
                "input": record["input"],
                "reference": record["reference"],
                "raw_completion": raw_completion,
                "reasoning": reasoning,
                "final_answer": final_answer,
                "generated": final_answer,
                "generation_status": generation_status,
                "parse_status": parse_status,
                "eos_terminated": eos_terminated,
                "reached_max_new_tokens": reached_token_limit,
                "generated_token_count": len(generated_id_list),
                "max_new_tokens": max_new_tokens,
                "think_block_present": parse_status != "no_think_block",
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
        print("RAW COMPLETION:", result["raw_completion"])
        print("REASONING:", result["reasoning"])
        print("FINAL ANSWER:", result["final_answer"])
        print(
            "GENERATION STATUS:", result["generation_status"],
            "| generated tokens:", result["generated_token_count"],
            "| EOS terminated:", result["eos_terminated"],
            "| max-token limit reached:", result["reached_max_new_tokens"],
        )

    return result_file



def split_reasoning_and_answer(text):
    """Conservatively split Param reasoning from its final answer.

    - Closed <think>...</think> block: split reasoning and final answer.
    - Opening <think> without a closing marker: mark incomplete reasoning;
      never misclassify the reasoning text as the final answer.
    - No thinking markers: treat the complete text as the final answer.
    - Closing marker without opening marker: preserve the prefix as reasoning
      and parse only the suffix as the final answer.

    Returns (reasoning, final_answer, parse_status).
    """
    raw = str(text).strip()
    open_match = re.search(r"<\s*think\s*>", raw, flags=re.IGNORECASE)
    close_match = re.search(r"</\s*think\s*>|<\\think>", raw, flags=re.IGNORECASE)

    if open_match is None and close_match is None:
        return "", raw, "no_think_block"

    if open_match is not None and close_match is None:
        return raw[open_match.end():].strip(), "", "incomplete_reasoning"

    if close_match is not None:
        if open_match is not None and close_match.start() < open_match.end():
            return raw, "", "malformed_think_block"

        if open_match is not None:
            reasoning = raw[open_match.end():close_match.start()].strip()
        else:
            reasoning = raw[:close_match.start()].strip()

        final_answer = raw[close_match.end():].strip()
        if not final_answer:
            return reasoning, "", "missing_final_answer"
        return reasoning, final_answer, "completed_think_block"

    return "", raw, "no_think_block"

def stratified_subset(rows, metadata, per_language, seed, split_name):
    """Select a deterministic, language-balanced subset for smoke testing."""
    rng = random.Random(seed)
    selected_indices = []
    languages = sorted({item["language"] for item in metadata})

    for language in languages:
        candidates = [
            i for i, item in enumerate(metadata)
            if item["language"] == language
        ]
        rng.shuffle(candidates)
        selected_indices.extend(candidates[:min(per_language, len(candidates))])

    selected_indices.sort()
    selected_rows = [rows[i] for i in selected_indices]
    selected_metadata = [metadata[i] for i in selected_indices]

    counts = Counter(item["language"] for item in selected_metadata)
    print(f"{split_name} stratified subset: {dict(counts)}")
    return selected_rows, selected_metadata


def _normalise_for_match(text):
    return re.sub(r"\s+", " ", str(text)).strip()


def _to_1d_list(value):
    """Convert a tensor/list or a single-row batch to a flat Python list."""
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().tolist()
    if value and isinstance(value[0], list):
        value = value[0]
    return list(value)


def dump_param_template_diagnostics(tokenizer, metadata, output_dir, num_examples=3):
    """Save and print how Param formats prompt-only and prompt+assistant turns."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 90)
    print("PARAM CHAT-TEMPLATE DIAGNOSTICS")
    print("=" * 90)
    print("TOKENIZER CLASS:", tokenizer.__class__.__name__)
    print("VOCAB SIZE:", len(tokenizer))
    print("BOS TOKEN:", repr(tokenizer.bos_token), tokenizer.bos_token_id)
    print("EOS TOKEN:", repr(tokenizer.eos_token), tokenizer.eos_token_id)
    print("PAD TOKEN:", repr(tokenizer.pad_token), tokenizer.pad_token_id)
    print("THINK TOKEN STRING LOOKUP:", tokenizer.convert_tokens_to_ids("<think>"))
    print("END-THINK TOKEN STRING LOOKUP:", tokenizer.convert_tokens_to_ids("</think>"))
    print("\nRAW TOKENIZER CHAT TEMPLATE:\n")
    print(tokenizer.chat_template)

    records = []
    chosen = []
    seen = set()
    for i, item in enumerate(metadata):
        if item["language"] not in seen:
            chosen.append(i)
            seen.add(item["language"])
        if len(chosen) >= num_examples:
            break

    for i in chosen:
        item = metadata[i]
        user_messages = [{"role": "user", "content": item["input"]}]
        full_messages = user_messages + [
            {"role": "assistant", "content": item["reference"]}
        ]

        try:
            prompt_rendered = tokenizer.apply_chat_template(
                user_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            full_rendered = tokenizer.apply_chat_template(
                full_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
            full_ids = tokenizer.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=False,
            )
            if hasattr(full_ids, "input_ids"):
                full_ids = full_ids.input_ids
            elif isinstance(full_ids, dict):
                full_ids = full_ids["input_ids"]
            if isinstance(full_ids, torch.Tensor):
                full_ids = full_ids.tolist()
            if full_ids and isinstance(full_ids[0], list):
                full_ids = full_ids[0]
        except Exception as exc:
            raise RuntimeError(
                f"Could not render Param chat template for example {item['id']}: {exc}"
            ) from exc

        record = {
            "index": i,
            "id": item["id"],
            "language": item["language"],
            "prompt_only_with_generation_prefix": prompt_rendered,
            "full_prompt_and_reference_rendered": full_rendered,
            "full_input_ids": [int(x) for x in full_ids],
            "full_token_pieces": [
                tokenizer.decode([int(tok)], skip_special_tokens=False,
                                 clean_up_tokenization_spaces=False)
                for tok in full_ids
            ],
            "reference": item["reference"],
        }
        records.append(record)

        print("\n" + "-" * 90)
        print(f"EXAMPLE {i} | {item['language']} | {item['id']}")
        print("PROMPT-ONLY GENERATION PREFIX:")
        print(repr(prompt_rendered))
        print("\nFULL RENDERED PROMPT + REFERENCE:")
        print(repr(full_rendered))
        print("\nFULL TOKEN STREAM (position | token_id | decoded_piece):")
        for pos, tok in enumerate(full_ids):
            piece = tokenizer.decode(
                [int(tok)], skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            print(f"{pos:4d} | {int(tok):8d} | {piece!r}")

    with open(output_path / "param_chat_template_diagnostics.json", "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print("\nSaved: param_chat_template_diagnostics.json")
    print("=" * 90)




def _find_subsequence(sequence, subsequence, start=0):
    """Return the first exact subsequence match at/after start, else None."""
    if not subsequence:
        return None
    last_start = len(sequence) - len(subsequence)
    for pos in range(start, last_start + 1):
        if sequence[pos:pos + len(subsequence)] == subsequence:
            return pos
    return None


def mask_empty_think_scaffold(dataset, tokenizer, metadata, split_name, output_dir):
    """
    Mask only Param's automatically inserted EMPTY <think>...</think> wrapper.

    For each row:
      - locate the dedicated <think> and </think> token IDs;
      - verify the content inside the block is whitespace only;
      - locate the exact reference target after </think>;
      - verify the material between </think> and the target is whitespace only;
      - set labels from <think> through the token immediately before the
        reference to -100.

    The reference paraphrase, EOS and trailing completion tokens remain
    supervised. Any non-empty reasoning content causes a hard failure.
    """
    if len(dataset) != len(metadata):
        raise RuntimeError(
            f"{split_name}: dataset/metadata length mismatch: "
            f"{len(dataset)} vs {len(metadata)}"
        )

    open_id = tokenizer.convert_tokens_to_ids("<think>")
    close_id = tokenizer.convert_tokens_to_ids("</think>")
    unk_id = tokenizer.unk_token_id

    if open_id is None or close_id is None or open_id == unk_id or close_id == unk_id:
        raise RuntimeError(
            "Param tokenizer does not expose dedicated <think>/</think> token IDs. "
            "Cannot safely mask the empty thinking scaffold."
        )

    report = []

    def transform(row, index):
        input_ids = list(row["input_ids"])
        labels = list(row["labels"])
        if len(input_ids) != len(labels):
            raise RuntimeError(
                f"{split_name} example {index}: input_ids/labels length mismatch."
            )

        open_positions = [i for i, token_id in enumerate(input_ids) if token_id == open_id]
        close_positions = [i for i, token_id in enumerate(input_ids) if token_id == close_id]

        if not open_positions and not close_positions:
            report.append({
                "index": index,
                "id": metadata[index]["id"],
                "language": metadata[index]["language"],
                "action": "no_think_scaffold",
                "masked_positions": [],
            })
            return {"labels": labels}

        if len(open_positions) != 1 or len(close_positions) != 1:
            raise RuntimeError(
                f"{split_name} example {index} ({metadata[index]['id']}): "
                f"expected exactly one <think> and one </think>; "
                f"found {len(open_positions)} and {len(close_positions)}."
            )

        open_pos, close_pos = open_positions[0], close_positions[0]
        if close_pos <= open_pos:
            raise RuntimeError(
                f"{split_name} example {index}: </think> occurs before <think>."
            )

        inside_text = tokenizer.decode(
            input_ids[open_pos + 1:close_pos],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if inside_text.strip():
            raise RuntimeError(
                f"{split_name} example {index} ({metadata[index]['id']}): "
                "the <think> block contains non-whitespace content. "
                "This is not an empty scaffold; refusing to mask it automatically. "
                f"Decoded block: {inside_text!r}"
            )

        target = metadata[index]["reference"]
        target_ids = tokenizer(
            target,
            add_special_tokens=False,
            return_attention_mask=False,
        )["input_ids"]

        target_start = _find_subsequence(
            input_ids,
            target_ids,
            start=close_pos + 1,
        )
        if target_start is None:
            raise RuntimeError(
                f"{split_name} example {index} ({metadata[index]['id']}): "
                "could not locate the exact reference token sequence after </think>. "
                "Refusing to guess the loss boundary."
            )

        between_text = tokenizer.decode(
            input_ids[close_pos + 1:target_start],
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        if between_text.strip():
            raise RuntimeError(
                f"{split_name} example {index} ({metadata[index]['id']}): "
                "non-whitespace content occurs between </think> and the reference. "
                "Refusing to mask an ambiguous span. "
                f"Decoded gap: {between_text!r}"
            )

        # The scaffold includes both markers and all blank-line separators.
        masked_start = open_pos
        masked_end = target_start  # exclusive
        for pos in range(masked_start, masked_end):
            labels[pos] = -100

        # Verify that the reference itself remains fully supervised.
        unsupervised_target_positions = [
            pos for pos in range(target_start, target_start + len(target_ids))
            if labels[pos] == -100
        ]
        if unsupervised_target_positions:
            raise RuntimeError(
                f"{split_name} example {index}: reference tokens unexpectedly "
                f"masked at positions {unsupervised_target_positions[:10]}."
            )

        report.append({
            "index": index,
            "id": metadata[index]["id"],
            "language": metadata[index]["language"],
            "action": "masked_empty_think_scaffold",
            "think_open_position": open_pos,
            "think_close_position": close_pos,
            "reference_start_position": target_start,
            "masked_position_start": masked_start,
            "masked_position_end_exclusive": masked_end,
            "masked_token_count": masked_end - masked_start,
            "masked_text": tokenizer.decode(
                input_ids[masked_start:masked_end],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            ),
            "reference": target,
            "reference_token_count": len(target_ids),
        })
        return {"labels": labels}

    masked_dataset = dataset.map(
        transform,
        with_indices=True,
        desc=f"Masking empty <think> scaffold ({split_name})",
        load_from_cache_file=False,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / f"param_empty_think_mask_{split_name}.json"
    with open(report_path, "w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(f"\n{split_name}: empty thinking scaffold mask report: {report_path}")
    print(
        f"{split_name}: masked {sum(r.get('masked_token_count', 0) for r in report)} "
        "scaffold tokens across "
        f"{sum(r.get('action') == 'masked_empty_think_scaffold' for r in report)} "
        "examples."
    )
    return masked_dataset


def audit_supervised_labels(trainer, tokenizer, metadata, output_dir,
                            num_examples=3):
    """
    Inspect actual labels produced by the SFTTrainer data collator.
    Save token-by-token evidence before deciding whether to abort.

    Labels != -100 are the target labels supplied to the causal-LM loss.
    The model internally shifts labels for next-token prediction, but these
    are the unshifted positions selected by the trainer's label mask.
    """
    dataset = trainer.train_dataset
    if len(dataset) == 0:
        raise RuntimeError("Trainer's processed training dataset is empty.")

    selected_indices = []
    seen_languages = set()
    for index, item in enumerate(metadata):
        if item["language"] not in seen_languages:
            selected_indices.append(index)
            seen_languages.add(item["language"])
        if len(selected_indices) >= num_examples:
            break
    if len(selected_indices) < min(num_examples, len(dataset)):
        for index in range(len(dataset)):
            if index not in selected_indices:
                selected_indices.append(index)
            if len(selected_indices) >= min(num_examples, len(dataset)):
                break

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    report = []
    any_reasoning_marker_supervised = False
    any_reference_missing = False

    marker_pattern = re.compile(r"<\s*/?\s*think\s*>|<\\think>", re.IGNORECASE)

    print("\n" + "=" * 90)
    print("SUPERVISED TOKEN AUDIT — EXACT LABEL POSITIONS")
    print("=" * 90)
    print("Interpretation: label != -100 means this token is selected as a target label.")
    print("The causal LM shifts labels internally for next-token prediction.")

    for index in selected_indices:
        if index >= len(dataset) or index >= len(metadata):
            raise RuntimeError(
                "Metadata and processed dataset indices no longer align. "
                "The trainer may have dropped examples; inspect the processed dataset mapping."
            )

        row = dataset[index]
        try:
            collated = trainer.data_collator([row])
        except Exception as exc:
            raise RuntimeError(
                "Could not obtain the actual training labels from the data collator: "
                f"{exc}"
            ) from exc

        if "labels" not in collated or "input_ids" not in collated:
            raise RuntimeError(
                "The data collator must expose both input_ids and labels for this audit. "
                f"Returned keys: {list(collated.keys())}"
            )

        input_ids = _to_1d_list(collated["input_ids"])
        labels = _to_1d_list(collated["labels"])
        if len(input_ids) != len(labels):
            raise RuntimeError(
                f"input_ids/labels length mismatch for example {index}: "
                f"{len(input_ids)} vs {len(labels)}"
            )

        token_rows = []
        supervised_ids = []
        supervised_pieces = []
        marker_positions = []
        for pos, (input_id, label_id) in enumerate(zip(input_ids, labels)):
            piece = tokenizer.decode(
                [int(input_id)], skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            supervised = int(label_id) != -100
            label_piece = None
            if supervised:
                supervised_ids.append(int(label_id))
                label_piece = tokenizer.decode(
                    [int(label_id)], skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                supervised_pieces.append(label_piece)
            token_rows.append({
                "position": pos,
                "input_token_id": int(input_id),
                "input_piece": piece,
                "label_token_id": int(label_id),
                "supervised": supervised,
                "label_piece": label_piece,
            })
            if supervised and marker_pattern.search(label_piece or ""):
                marker_positions.append(pos)

        supervised_text = tokenizer.decode(
            supervised_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
        expected_target = metadata[index]["reference"]
        decoded_norm = _normalise_for_match(supervised_text)
        target_norm = _normalise_for_match(expected_target)
        reference_found = target_norm in decoded_norm
        has_marker = bool(marker_pattern.search(supervised_text))

        any_reasoning_marker_supervised |= has_marker
        any_reference_missing |= not reference_found

        record = {
            "processed_dataset_index": index,
            "id": metadata[index]["id"],
            "language": metadata[index]["language"],
            "collated_sequence_length": len(input_ids),
            "supervised_token_count": len(supervised_ids),
            "supervised_text": supervised_text,
            "expected_reference": expected_target,
            "reference_found_in_supervised_text": reference_found,
            "reasoning_marker_in_supervised_text": has_marker,
            "reasoning_marker_supervised_positions": marker_positions,
            "tokens": token_rows,
            "processed_row_keys": list(row.keys()),
            "collator_keys": list(collated.keys()),
        }
        report.append(record)

        print("\n" + "-" * 90)
        print(f"EXAMPLE {index} | {metadata[index]['language']} | {metadata[index]['id']}")
        print("EXPECTED REFERENCE:", repr(expected_target))
        print("SUPERVISED TEXT:", repr(supervised_text))
        print("REFERENCE FOUND:", reference_found)
        print("THINK MARKER IN SUPERVISED LABELS:", has_marker)
        print("THINK MARKER POSITIONS:", marker_positions)
        print("TOKEN TABLE (position | input id | label id | supervised | token piece):")
        for token in token_rows:
            print(
                f"{token['position']:4d} | {token['input_token_id']:8d} | "
                f"{token['label_token_id']:8d} | {str(token['supervised']):10s} | "
                f"{token['input_piece']!r}"
            )

    with open(output_path / "param_supervised_token_audit.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print("\nSaved: param_supervised_token_audit.json")
    print("=" * 90)

    if any_reasoning_marker_supervised or any_reference_missing:
        raise RuntimeError(
            "Supervised-label audit failed. The detailed chat-template and "
            "token-by-token label reports have been saved. Do not train yet. "
            f"Reasoning marker supervised: {any_reasoning_marker_supervised}; "
            f"reference missing from supervised labels: {any_reference_missing}."
        )

    print("PASS: reference paraphrases are supervised and reasoning markers are excluded.")



def make_validation_split(rows, metadata, validation_fraction, seed, output_dir):
    """Deterministically split a monolingual file into train and validation."""
    if not 0.0 < validation_fraction < 0.5:
        raise ValueError("--validation_fraction must be between 0 and 0.5.")
    n = len(rows)
    n_eval = max(1, int(round(n * validation_fraction)))
    if n_eval >= n:
        raise ValueError("Validation split would leave no training examples.")
    indices = list(range(n))
    rng = random.Random(seed)
    rng.shuffle(indices)
    eval_indices = sorted(indices[:n_eval])
    train_indices = sorted(indices[n_eval:])
    train_rows = [rows[i] for i in train_indices]
    train_metadata = [metadata[i] for i in train_indices]
    eval_rows = [rows[i] for i in eval_indices]
    eval_metadata = [metadata[i] for i in eval_indices]
    manifest = {
        "seed": seed,
        "validation_fraction": validation_fraction,
        "source_examples": n,
        "train_examples": len(train_rows),
        "validation_examples": len(eval_rows),
        "train_ids": [x["id"] for x in train_metadata],
        "validation_ids": [x["id"] for x in eval_metadata],
        "note": "Held out for this monolingual run. If these examples were used in an earlier mixed-language run, that earlier checkpoint has seen them and is not directly comparable on this split."
    }
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    with open(Path(output_dir) / "data_split_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"Created deterministic split: train={len(train_rows)}, validation={len(eval_rows)}")
    return train_rows, train_metadata, eval_rows, eval_metadata


def clean_decoded_answer(text, tokenizer):
    """Remove EOS/PAD spellings from parsed answer; preserve raw_completion."""
    value = str(text or "").strip()
    tokens = []
    for token in (getattr(tokenizer, "eos_token", None), getattr(tokenizer, "pad_token", None)):
        if token and token not in tokens:
            tokens.append(token)
    for token in tokens:
        value = value.replace(token, "")
    return value.strip()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Full-parameter monolingual Param-1B SFT for Hindi or Bengali paraphrasing."
    )

    parser.add_argument("--train_file", required=True)
    parser.add_argument("--validation_file", default=None,
                        help="Optional independent validation JSONL. If omitted, split train_file.")
    parser.add_argument("--validation_fraction", type=float, default=0.10)
    parser.add_argument("--language", required=True, choices=["hin", "ben"])
    parser.add_argument(
        "--model_name",
        required=True,
        help="Local Param-1B checkpoint path or Hugging Face model ID.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
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
        "--torch_dtype",
        choices=["bfloat16", "float16"],
        default="bfloat16",
    )
    parser.add_argument(
        "--trust_remote_code",
        action="store_true",
        help="Enable only if the Param checkpoint requires custom model code.",
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
        default=None,
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
        default=512,
        help="Maximum generated tokens for qualitative outputs (reasoning + final answer).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    if args.output_dir is None:
        args.output_dir = f"/mnt/fs1/akash/SFT_train/param1b_{args.language}_sft"
    if args.run_name is None:
        args.run_name = f"param1b_{args.language}_sft"

    if args.smoke_test:
        # Monolingual smoke subset: 12 train + 3 validation examples.
        args.epochs = 1.0
        args.per_device_train_batch_size = 1
        args.per_device_eval_batch_size = 1
        args.gradient_accumulation_steps = 1
        args.logging_steps = 1
        args.eval_steps = 5
        args.save_steps = 5
        args.save_total_limit = 1
        if not args.run_name.endswith("_smoke"):
            args.run_name = f"{args.run_name}_smoke"

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
    expected_languages = [args.language]
    strip_target_quotes = not args.preserve_target_quotes

    all_train_rows, all_train_metadata, all_train_summary = load_jsonl(
        args.train_file, "train", strip_target_quotes=strip_target_quotes
    )
    found_languages = set(all_train_summary["language_counts"])
    if found_languages != {args.language}:
        raise ValueError(
            f"--language {args.language} requires a file containing only that "
            f"language; found {sorted(found_languages)}."
        )

    if args.validation_file:
        eval_rows, eval_metadata, eval_summary = load_jsonl(
            args.validation_file, "validation",
            strip_target_quotes=strip_target_quotes
        )
        eval_languages = set(eval_summary["language_counts"])
        if eval_languages != {args.language}:
            raise ValueError(
                f"Validation file must contain only {args.language}; "
                f"found {sorted(eval_languages)}."
            )
        train_rows, train_metadata = all_train_rows, all_train_metadata
        train_summary = all_train_summary
    else:
        train_rows, train_metadata, eval_rows, eval_metadata = make_validation_split(
            all_train_rows, all_train_metadata,
            args.validation_fraction, args.seed, args.output_dir
        )
        train_summary = {
            "examples": len(train_rows),
            "language_counts": dict(Counter(x["language"] for x in train_metadata)),
            "quote_stripped_by_language": all_train_summary["quote_stripped_by_language"],
            "missing_language_metadata": 0,
        }
        eval_summary = {
            "examples": len(eval_rows),
            "language_counts": dict(Counter(x["language"] for x in eval_metadata)),
            "quote_stripped_by_language": {},
            "missing_language_metadata": 0,
        }

    print_language_distribution("TRAIN", train_summary)
    print_language_distribution("VALIDATION", eval_summary)

    if args.smoke_test:
        train_rows, train_metadata = stratified_subset(
            train_rows, train_metadata, per_language=12,
            seed=args.seed, split_name="TRAIN",
        )
        eval_rows, eval_metadata = stratified_subset(
            eval_rows, eval_metadata, per_language=3,
            seed=args.seed, split_name="VALIDATION",
        )

    for language in expected_languages:
        if train_summary["language_counts"].get(language, 0) == 0:
            raise ValueError(f"Expected language {language} is absent from training data.")
        if eval_summary["language_counts"].get(language, 0) == 0:
            raise ValueError(f"Expected language {language} is absent from validation data.")

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
        trust_remote_code=args.trust_remote_code,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                "Tokenizer has neither pad_token nor eos_token; configure padding explicitly."
            )
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Dump the exact Param chat-template rendering before TRL preprocessing.
    dump_param_template_diagnostics(
        tokenizer=tokenizer,
        metadata=train_metadata,
        output_dir=args.output_dir,
        num_examples=3,
    )

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
        dtype=(
            torch.bfloat16 if args.torch_dtype == "bfloat16" else torch.float16
        ),
        trust_remote_code=args.trust_remote_code,
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

    # Param's chat template inserts an EMPTY <think>\\n\\n</think> block
    # before the final answer. Mask that scaffold (markers + blank-line
    # separators) while preserving the reference paraphrase and EOS targets.
    # Any non-empty reasoning content causes a hard failure.
    trainer.train_dataset = mask_empty_think_scaffold(
        trainer.train_dataset,
        tokenizer,
        train_metadata,
        split_name="train",
        output_dir=args.output_dir,
    )
    trainer.eval_dataset = mask_empty_think_scaffold(
        trainer.eval_dataset,
        tokenizer,
        eval_metadata,
        split_name="eval",
        output_dir=args.output_dir,
    )

    # Audit the labels AFTER masking. The expected result is:
    # reference paraphrase supervised; empty think scaffold not supervised.
    audit_supervised_labels(
        trainer=trainer,
        tokenizer=tokenizer,
        metadata=train_metadata,
        output_dir=args.output_dir,
        num_examples=3,
    )

    trainable_params = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    total_params = sum(parameter.numel() for parameter in model.parameters())

    print("\n" + "=" * 80)
    print("PARAM-1B MONOLINGUAL FULL-PARAMETER SFT")
    print("=" * 80)
    print(f"Model: {args.model_name}")
    print(f"Language: {args.language}")
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Max sequence length: {args.max_length}")
    print("Loss: reference paraphrase tokens only; reasoning traces excluded")
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
