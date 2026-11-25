import argparse
import asyncio
import json
import os
import re
import time

from tqdm import tqdm
from openai import AsyncOpenAI

from judge_prompts_revised_RL_preds import (
    build_hindi_paraphrase_judge_prompt,
    build_marathi_paraphrase_judge_prompt,
    build_bengali_paraphrase_judge_prompt,
    build_assamese_paraphrase_judge_prompt,
    build_telugu_paraphrase_judge_prompt,
    build_kannada_paraphrase_judge_prompt,
)


# ============================================================
# JSONL utilities
# ============================================================

def load_jsonl(path):
    rows = []

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()

            if not line:
                continue

            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(
                    f"WARNING: Could not parse line {line_num}: {e}"
                )

    return rows


def append_jsonl(path, row):
    with open(path, "a", encoding="utf-8") as f:
        f.write(
            json.dumps(
                row,
                ensure_ascii=False
            ) + "\n"
        )


# ============================================================
# Prompt selection
# ============================================================

def build_prompt(language, source, prediction, references):
    """
    Select the appropriate language-specific prompt builder.

    The actual prompts and examples are kept in judge_prompts.py.
    """

    language = language.lower()

    if language == "hi":
        return build_hindi_paraphrase_judge_prompt(
            source,
            prediction,
            references
        )

    elif language == "mr":
        return build_marathi_paraphrase_judge_prompt(
            source,
            prediction,
            references 
        )

    elif language == "bn":
        return build_bengali_paraphrase_judge_prompt(
            source,
            prediction,
            references
        )

    elif language == "as":
        return build_assamese_paraphrase_judge_prompt(
            source,
            prediction,
            references
        )

    elif language == "te":
        return build_telugu_paraphrase_judge_prompt(
            source,
            prediction,
            references
        )

    elif language == "kn":
        return build_kannada_paraphrase_judge_prompt(
            source,
            prediction,
            references
        )

    else:
        raise ValueError(
            f"Unsupported language: {language}. "
            f"Supported languages: hi, mr, bn, as, te, kn"
        )
    #         prediction,
    #         references
    #     )

    

# ============================================================
# Parse judge response
# ============================================================

def parse_judge_response(text):
    """
    Parse:

        RESPONSE_QUALITY: 0/1
        QUALITY_FAILURE_TYPE: none | semantic_change | unnatural_expression |
                              insufficient_reformulation | multiple
        LANGUAGE_ADHERENCE: 0/1
        GROUNDEDNESS: 0/1
    """

    result = {
        "response_quality": None,
        "quality_failure_type": None,
        "quality_failure_type_parse_error": None,
        "language_adherence": None,
        "groundedness": None,
        "parse_error": None,
    }

    if not text:
        result["parse_error"] = "Empty judge response"
        return result

    patterns = {
        "response_quality":
            r"RESPONSE_QUALITY\s*:\s*([01])",

        "language_adherence":
            r"LANGUAGE_ADHERENCE\s*:\s*([01])",

        "groundedness":
            r"GROUNDEDNESS\s*:\s*([01])",
    }

    for key, pattern in patterns.items():

        match = re.search(
            pattern,
            text,
            flags=re.IGNORECASE
        )

        if match:
            result[key] = int(match.group(1))

    allowed_failure_types = {
        "none",
        "semantic_change",
        "unnatural_expression",
        "insufficient_reformulation",
        "multiple",
    }
    category_match = re.search(
        r"QUALITY_FAILURE_TYPE\s*:\s*([a-z_]+)",
        text,
        flags=re.IGNORECASE,
    )
    if category_match:
        category = category_match.group(1).lower()
        if category in allowed_failure_types:
            result["quality_failure_type"] = category
        else:
            result["quality_failure_type_parse_error"] = (
                f"Invalid QUALITY_FAILURE_TYPE: {category}"
            )
    else:
        result["quality_failure_type_parse_error"] = "Missing QUALITY_FAILURE_TYPE"

    # Preserve criterion scores even if the diagnostic label is missing or inconsistent.
    if result["response_quality"] == 1:
        if result["quality_failure_type"] is None:
            result["quality_failure_type"] = "none"
        elif result["quality_failure_type"] != "none":
            result["quality_failure_type_parse_error"] = (
                "Inconsistent output: RESPONSE_QUALITY=1 requires QUALITY_FAILURE_TYPE=none"
            )
            result["quality_failure_type"] = "unknown"
    elif result["response_quality"] == 0:
        if result["quality_failure_type"] is None:
            result["quality_failure_type"] = "unknown"
        elif result["quality_failure_type"] == "none":
            result["quality_failure_type_parse_error"] = (
                "Inconsistent output: RESPONSE_QUALITY=0 cannot use QUALITY_FAILURE_TYPE=none"
            )
            result["quality_failure_type"] = "unknown"

    missing = []

    for key in (
        "response_quality",
        "language_adherence",
        "groundedness",
    ):
        if result[key] is None:
            missing.append(key)

    if missing:
        result["parse_error"] = (
            "Could not parse: "
            + ", ".join(missing)
        )

    return result



# ============================================================
# Asynchronous judge one example
# ============================================================

async def judge_one_async(
    client,
    model,
    language,
    source,
    prediction,
    references,
    temperature,
    max_tokens,
    reasoning_effort,
    judge_type,
):
    """Perform one asynchronous LLM judge call using the revised prompt."""

    prompt = build_prompt(
        language=language,
        source=source,
        prediction=prediction,
        references=references,
    )

    kwargs = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    if judge_type == "gpt-oss":
        kwargs["extra_body"] = {"reasoning_effort": reasoning_effort}
    else:
        kwargs["extra_body"] = {
            "chat_template_kwargs": {
                "enable_thinking": reasoning_effort != "none"
            }
        }

    response = await client.chat.completions.create(**kwargs)
    message = response.choices[0].message
    raw_output = message.content or ""
    reasoning = getattr(message, "reasoning", None)
    parsed = parse_judge_response(raw_output)

    return prompt, raw_output, reasoning, parsed


# ============================================================
# Async evaluation helpers
# ============================================================

def make_output_row(
    row,
    args,
    prompt=None,
    raw_output=None,
    reasoning=None,
    parsed=None,
    elapsed=None,
    error=None,
):
    """Build a result row while preserving all original input fields."""

    output_row = dict(row)
    parsed = parsed or {
        "response_quality": None,
        "quality_failure_type": None,
        "quality_failure_type_parse_error": None,
        "language_adherence": None,
        "groundedness": None,
        "parse_error": None,
    }

    response_quality = parsed["response_quality"]
    language_adherence = parsed["language_adherence"]
    groundedness = parsed["groundedness"]

    overall_pass = None
    if (
        response_quality is not None
        and language_adherence is not None
        and groundedness is not None
    ):
        overall_pass = int(
            response_quality == 1
            and language_adherence == 1
            and groundedness == 1
        )

    output_row.update({
        "judge_language": args.language,
        "judge_model": args.model,
        "judge_type": args.judge_type,
        "reasoning_effort": args.reasoning_effort,
        "judge_temperature": args.temperature,
        "judge_max_tokens": args.max_tokens,
        "judge_prompt": prompt,
        "judge_raw_output": raw_output,
        "judge_reasoning": reasoning,
        "response_quality": response_quality,
        "quality_failure_type": parsed.get("quality_failure_type"),
        "quality_failure_type_parse_error": parsed.get("quality_failure_type_parse_error"),
        "language_adherence": language_adherence,
        "groundedness": groundedness,
        "overall_pass": overall_pass,
        "judge_parse_error": parsed["parse_error"],
        "judge_elapsed_seconds": elapsed,
        "judge_error": error,
    })
    return output_row


async def evaluate_one(
    row,
    idx,
    args,
    client,
    semaphore,
    write_lock,
):
    """Evaluate one row; retain its original index regardless of completion order."""

    row_index = row.get("index", idx)
    row_id = row.get("id", str(idx))

    source = row.get("input", "")
    prediction = row.get("prediction", "")
    references = row.get("references", [])

    if references is None:
        references = []
    if not isinstance(references, list):
        references = [str(references)]
    references = [
        str(ref).strip()
        for ref in references
        if ref is not None and str(ref).strip()
    ]

    start_time = time.time()
    try:
        async with semaphore:
            prompt, raw_output, reasoning, parsed = await judge_one_async(
                client=client,
                model=args.model,
                language=args.language,
                source=source,
                prediction=prediction,
                references=references,
                temperature=args.temperature,
                max_tokens=args.max_tokens,
                reasoning_effort=args.reasoning_effort,
                judge_type=args.judge_type,
            )

        output_row = make_output_row(
            row=row,
            args=args,
            prompt=prompt,
            raw_output=raw_output,
            reasoning=reasoning,
            parsed=parsed,
            elapsed=time.time() - start_time,
        )
        if parsed["parse_error"]:
            print(f"\nParse warning at index {row_index}: {parsed['parse_error']}")

    except Exception as exc:
        output_row = make_output_row(
            row=row,
            args=args,
            elapsed=time.time() - start_time,
            error=str(exc),
        )
        print(f"\nERROR at index {row_index}: {exc}")

    # The lock prevents concurrent workers from interleaving file writes.
    async with write_lock:
        append_jsonl(args.output_file, output_row)

    return output_row


# ============================================================
# Main
# ============================================================

async def async_main():
    parser = argparse.ArgumentParser(
        description=(
            "Asynchronous LLM judge evaluation for multilingual "
            "paraphrase generation"
        )
    )

    # Input / output
    parser.add_argument("--input-file", required=True, help="JSONL prediction file")
    parser.add_argument(
        "--output-file",
        required=True,
        help="JSONL output file containing judge results",
    )

    # Language
    parser.add_argument(
        "--language",
        required=True,
        choices=["hi", "mr", "bn", "as", "te", "kn"],
        help="Language of the paraphrase task: hi, mr, bn, as, te, or kn",
    )

    # vLLM / OpenAI-compatible server
    parser.add_argument(
        "--model",
        required=True,
        help="Judge model name/path exposed by vLLM",
    )
    parser.add_argument(
        "--base-url",
        default="http://0.0.0.0:8082/v1",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Optional API key; omit when vLLM authentication is disabled",
    )

    # Judge type
    parser.add_argument(
        "--judge-type",
        choices=["gpt-oss", "standard"],
        default="gpt-oss",
        help="Use gpt-oss for GPT-OSS judges; standard for other chat/reasoning models.",
    )

    # Reasoning
    parser.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "none"],
        default="none",
        help=(
            "For GPT-OSS, passed as reasoning_effort. "
            "For other models, 'none' disables thinking."
        ),
    )

    # Generation parameters
    parser.add_argument("--temperature", type=float, default=0.0, help="Judge temperature")
    parser.add_argument("--max-tokens", type=int, default=2048, help="Maximum judge output tokens")

    # Number of examples
    parser.add_argument(
        "--max-examples",
        "--num-instances",
        dest="max_examples",
        type=int,
        default=None,
        help="Evaluate only the first N examples",
    )

    parser.add_argument("--timeout", type=float, default=300.0, help="Request timeout in seconds")
    parser.add_argument("--max-retries", type=int, default=2, help="SDK retries for transient errors")

    # Concurrency
    parser.add_argument(
        "--workers",
        type=int,
        default=9,
        help="Maximum number of concurrent judge requests (default: 9)",
    )

    # Resume
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from an existing output file; skips matching index/id values",
    )

    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.max_examples is not None and args.max_examples < 0:
        parser.error("--max-examples/--num-instances must be non-negative")

    print("=" * 70)
    print("ASYNC PARAPHRASE LLM JUDGE")
    print("=" * 70)
    print(f"Input file:       {args.input_file}")
    print(f"Output file:      {args.output_file}")
    print(f"Language:         {args.language}")
    print(f"Judge model:      {args.model}")
    print(f"Judge type:       {args.judge_type}")
    print(f"Reasoning effort: {args.reasoning_effort}")
    print(f"Base URL:         {args.base_url}")
    print(f"Temperature:      {args.temperature}")
    print(f"Max tokens:       {args.max_tokens}")
    print(f"Concurrent workers: {args.workers}")
    print(f"API key supplied: {'yes' if args.api_key else 'no (using placeholder)'}")

    rows = load_jsonl(args.input_file)
    if args.max_examples is not None:
        rows = rows[:args.max_examples]
    print(f"Examples in selected input range: {len(rows)}")

    processed_indices = set()
    processed_ids = set()
    if args.resume and os.path.exists(args.output_file):
        existing_rows = load_jsonl(args.output_file)
        for existing in existing_rows:
            if existing.get("index") is not None:
                processed_indices.add(existing["index"])
            if existing.get("id") is not None:
                processed_ids.add(existing["id"])
        print(f"Already present in output: {len(existing_rows)}")

    pending = []
    for idx, row in enumerate(rows):
        row_index = row.get("index", idx)
        row_id = row.get("id", str(idx))
        if args.resume and (
            row_index in processed_indices or row_id in processed_ids
        ):
            continue
        pending.append((idx, row))

    print(f"Examples remaining to evaluate: {len(pending)}")

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # OpenAI SDK requires a non-empty api_key locally. "EMPTY" is only a
    # client-side placeholder; it does not enable or disable server auth.
    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key or "EMPTY",
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    semaphore = asyncio.Semaphore(args.workers)
    write_lock = asyncio.Lock()

    try:
        tasks = [
            asyncio.create_task(
                evaluate_one(
                    row=row,
                    idx=idx,
                    args=args,
                    client=client,
                    semaphore=semaphore,
                    write_lock=write_lock,
                )
            )
            for idx, row in pending
        ]

        # Consume completions as they finish; each result keeps its own row/index.
        completed_rows = []
        for task in tqdm(
            asyncio.as_completed(tasks),
            total=len(tasks),
            desc="Evaluating",
            unit="example",
        ):
            completed_rows.append(await task)
    finally:
        await client.close()

    # Report metrics over examples evaluated in this run (not previously resumed rows).
    metrics = {
        "response_quality": [],
        "language_adherence": [],
        "groundedness": [],
        "overall_pass": [],
    }
    parse_error_count = 0
    judge_error_count = 0

    for row in completed_rows:
        for metric in metrics:
            value = row.get(metric)
            if value is not None:
                metrics[metric].append(value)
        if row.get("judge_parse_error"):
            parse_error_count += 1
        if row.get("judge_error"):
            judge_error_count += 1

    print()
    print("=" * 70)
    print("FINAL LLM JUDGE RESULTS (THIS RUN)")
    print("=" * 70)
    labels = {
        "response_quality": "Response Quality Pass Rate",
        "language_adherence": "Language Adherence Pass Rate",
        "groundedness": "Groundedness Pass Rate",
        "overall_pass": "Overall Pass Rate (all three criteria)",
    }
    for key, label in labels.items():
        values = metrics[key]
        if values:
            passed = sum(values)
            print(f"{label}: {passed / len(values):.4f} ({passed}/{len(values)})")
        else:
            print(f"{label}: N/A (0 parsed examples)")

    print(f"Parse errors: {parse_error_count}")
    print(f"Judge/API errors: {judge_error_count}")
    print("=" * 70)
    print(f"Results saved to: {args.output_file}")


if __name__ == "__main__":
    import asyncio
    from openai import AsyncOpenAI

    asyncio.run(async_main())

#  python /mnt/fs1/akash/IndicNLG/judge/llm_judge_eval_async_revised_RL_preds.py --input-file /mnt/fs1/akash/IndicNLG/preds/Param_RL/predictions_hi_multi.jsonl --output-file /mnt/fs1/akash/IndicNLG/judgments/Param_RL/judge_results_hi_multi.jsonl --language hi --model gpt-judge-RL --reasoning-effort low