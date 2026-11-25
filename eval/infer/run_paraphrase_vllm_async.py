#!/usr/bin/env python3
"""
"निम्नलिखित वाक्य का पुनर्लेखन करें। मूल अर्थ को बनाए रखते हुए अलग शब्दों या वाक्य संरचना का उपयोग करें।\n\n"
f"स्रोत वाक्य:\n{source}\n\n"
"पुनर्लेखन:"


"खालील वाक्याचे पुनर्लेखन करा. मूळ अर्थ कायम ठेवून वेगळे शब्द किंवा वाक्यरचना वापरा.\n\n"
f"मूळ वाक्य:\n{source}\n\n"
"पुनर्लेखन:"

"பின்வரும் வாக்கியத்தை மறுபதிப்பு செய்யவும். அசல் பொருளைத் தக்கவைத்துக்கொண்டு, வேறுபட்ட சொற்கள் அல்லது வாக்கிய அமைப்பைப் பயன்படுத்தவும்.\n\n"
f"மூல வாக்கியம்:\n{source}\n\n"
"மறுபதிப்பு:"
"""


"""
xx paraphrase generation using a vLLM OpenAI-compatible server.

Supports:
    --mode zero_shot
    --mode one_shot

For one-shot:
    - One demonstration is selected from the training file using --demo-index.
    - The demonstration contains source/input + target paraphrase.
    - The demonstration is placed before every test example.

The script ONLY generates and saves predictions.
No evaluation, reference comparison, or output parsing is performed.

Expected dataset structure:

{
    "examples": [
        {
            "id": "...",
            "input": "...",
            "target": "...",
            "references": [...]
        }
    ]
}

The script also accepts a plain JSON list of examples.
"""

import argparse
import asyncio
import json
import time
from pathlib import Path

from openai import AsyncOpenAI
from tqdm import tqdm


# ============================================================
# DATA LOADING
# ============================================================
def load_examples(path):
    examples = []

    with open(path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"Invalid JSON on line {line_num} of {path}: {e}"
                ) from e

    return examples
# ============================================================
# FIELD HANDLING
# ============================================================

def get_input(example):
    """
    Get source sentence.

    IndicParaphrase uses 'input', but a few fallbacks are included
    to make the script tolerant of minor dataset variations.
    """

    for key in ["input", "source", "src", "sentence", "text"]:
        if key in example:
            return str(example[key])

    raise ValueError(
        f"Could not find input/source field. "
        f"Available keys: {list(example.keys())}"
    )


def get_target(example):
    """
    Get the target paraphrase used for the one-shot demonstration.

    For IndicParaphrase this should normally be 'target'.
    """

    for key in ["target", "tgt", "paraphrase", "output"]:
        if key in example:
            return str(example[key])

    raise ValueError(
        f"Could not find target/paraphrase field. "
        f"Available keys: {list(example.keys())}"
    )


# ============================================================
# PROMPTS
# ============================================================

def format_zero_shot(test_example):
    """
    Zero-shot prompt.
    """

    source = get_input(test_example)

    return (
        # "Paraphrase the following sentence. Preserve the original meaning, but express it using different wording or sentence structure. Do not simply copy the original sentence. Output only the paraphrased sentence.\n\n"
        #  f"SOURCE:\n{source}\n\n"
        # "PARAPHRASE:"
        # "निम्नलिखित वाक्य का पुनर्लेखन करें। मूल अर्थ को बनाए रखते हुए अलग शब्दों या वाक्य संरचना का उपयोग करें।\n\n"
        # f"स्रोत वाक्य:\n{source}\n\n"
        # "पुनर्लेखन:"
        "Paraphrase the following sentence in Bengali. Preserve the original meaning, but express it using different wording or sentence structure. Do not simply copy the original sentence. Output only the paraphrased sentence.\n\n"
        f"SENTENCE:\n{source}\n\n"
        "PARAPHRASE:"
        # "खालील वाक्याचे पुनर्लेखन करा. मूळ अर्थ कायम ठेवून वेगळे शब्द किंवा वाक्यरचना वापरा.\n\n"
        # f"मूळ वाक्य:\n{source}\n\n"
        # "पुनर्लेखन:"
       
    )


def format_demonstration(demo_example):
    """
    Format the single training example used for one-shot prompting.
    """

    source = get_input(demo_example)
    target = get_target(demo_example)

    return (
        "SOURCE:\n"
        f"{source}\n\n"
        "PARAPHRASE:\n"
        f"{target}"
    )


def format_one_shot(test_example, demo_example):
    """
    One-shot prompt.
    """

    source = get_input(test_example)

    demonstration = format_demonstration(demo_example)

    return (
        "Paraphrase the following sentence in Hindi. Preserve the original meaning, but express it using different wording or sentence structure. Do not simply copy the original sentence. Output only the paraphrased sentence.\n\n"

        "EXAMPLE:\n"
        f"{demonstration}\n\n"

        "NOW PARAPHRASE:\n"
        f"SOURCE:\n{source}\n\n"
        "PARAPHRASE:"
    )


def build_prompt(test_example, mode, demonstration=None):

    if mode == "zero_shot":
        return format_zero_shot(test_example)

    if mode == "one_shot":
        if demonstration is None:
            raise ValueError(
                "One-shot mode requires a training-file demonstration."
            )

        return format_one_shot(
            test_example,
            demonstration
        )

    raise ValueError(f"Unknown mode: {mode}")



# ============================================================
# RESUME SUPPORT
# ============================================================
def load_completed(output_file):
    completed = {}
    path = Path(output_file)
    if not path.exists():
        return completed

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if "index" in row:
                    completed[int(row["index"])] = row
            except (json.JSONDecodeError, TypeError, ValueError):
                print(f"WARNING: Skipping malformed JSONL line in {output_file}")
    return completed


def append_jsonl(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
        f.flush()


# ============================================================
# ASYNC GENERATION
# ============================================================
async def generate_one(
    idx, example, args, client, demonstration, semaphore, write_lock,
    progress
):
    source = get_input(example)
    references = example.get("references", [])
    prompt = build_prompt(
        test_example=example,
        mode=args.mode,
        demonstration=demonstration,
    )

    request_kwargs = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
    }
    if args.seed is not None:
        request_kwargs["extra_body"] = {"seed": args.seed}

    started = time.time()
    prediction = ""
    error = None

    try:
        # Semaphore limits the number of in-flight requests to --workers.
        async with semaphore:
            response = await client.chat.completions.create(**request_kwargs)
        prediction = (response.choices[0].message.content or "").strip()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        print(f"\nERROR at index {idx}: {error}")

    elapsed = time.time() - started
    row = {
        "index": idx,
        "id": example.get("id", str(idx)),
        "input": source,
        "references": references,
        "prompt": prompt,
        "prediction": prediction,
        "mode": args.mode,
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "demo_index": args.demo_index if args.mode == "one_shot" else None,
        "workers": args.workers,
        "start_index": args.start_index,
        "max_examples": args.max_examples,
        "num_instances": args.max_examples,
        "run_config": {
            key: value
            for key, value in vars(args).items()
            # if key != "api_key"
        },
        "elapsed_seconds": elapsed,
        "error": error,
    }

    # A lock ensures each JSONL row is written atomically, even when requests
    # finish in a different order. The original dataset index is preserved.
    async with write_lock:
        await asyncio.to_thread(append_jsonl, args.output_file, row)

    progress.update(1)
    return row


async def async_main(args):
    test_examples = load_examples(args.test_file)
    print(f"Loaded {len(test_examples)} test examples.")

    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if args.max_examples is not None and args.max_examples < 0:
        raise ValueError("--num-instances/--max-examples must be >= 0")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    demonstration = None
    if args.mode == "one_shot":
        if args.train_file is None:
            raise ValueError("--train-file is required for --mode one_shot")
        train_examples = load_examples(args.train_file)
        print(f"Loaded {len(train_examples)} training examples.")
        if not 0 <= args.demo_index < len(train_examples):
            raise ValueError(
                f"--demo-index {args.demo_index} is outside training set range "
                f"[0, {len(train_examples) - 1}]"
            )
        demonstration = train_examples[args.demo_index]
        print(f"Using training example {args.demo_index} as one-shot demonstration.")
        print("Demo source:", get_input(demonstration))
        print("Demo target:", get_target(demonstration))

    completed = load_completed(args.output_file)
    print(f"Already completed: {len(completed)}")

    end_index = len(test_examples)
    if args.max_examples is not None:
        end_index = min(end_index, args.start_index + args.max_examples)
    if args.start_index > len(test_examples):
        raise ValueError(
            f"--start-index {args.start_index} exceeds dataset size {len(test_examples)}"
        )

    indices = [
        idx for idx in range(args.start_index, end_index)
        if idx not in completed
    ]

    print("\n" + "=" * 70)
    print("ASYNC PARAPHRASE GENERATION")
    print("=" * 70)
    print(f"Mode:          {args.mode}")
    print(f"Model:         {args.model}")
    print(f"Test file:     {args.test_file}")
    print(f"Output:        {args.output_file}")
    print(f"Range:         [{args.start_index}, {end_index})")
    print(f"Pending:       {len(indices)}")
    print(f"Workers:       {args.workers}")
    print(f"Temperature:   {args.temperature}")
    print(f"Top-p:         {args.top_p}")
    print(f"Max tokens:    {args.max_tokens}")
    print("=" * 70 + "\n")

    if not indices:
        print("No pending examples in the selected range.")
        return

    client = AsyncOpenAI(
        base_url=args.base_url,
        api_key=args.api_key or "EMPTY",
        # Allow enough time for slower generations while retaining SDK retries.
        timeout=args.timeout,
        max_retries=args.max_retries,
    )
    semaphore = asyncio.Semaphore(args.workers)
    write_lock = asyncio.Lock()

    try:
        with tqdm(total=len(indices), desc="Generating paraphrases", unit="example") as progress:
            tasks = [
                asyncio.create_task(
                    generate_one(
                        idx=idx,
                        example=test_examples[idx],
                        args=args,
                        client=client,
                        demonstration=demonstration,
                        semaphore=semaphore,
                        write_lock=write_lock,
                        progress=progress,
                    )
                )
                for idx in indices
            ]
            # gather keeps exceptions visible; per-example API errors are caught
            # and saved to JSONL so one failed request does not stop the run.
            await asyncio.gather(*tasks)
    finally:
        await client.close()

    print("\n" + "=" * 70)
    print("GENERATION COMPLETE")
    print(f"Output saved to: {args.output_file}")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(
        description="Generate paraphrases asynchronously using a vLLM OpenAI-compatible server."
    )

    # Model / server
    parser.add_argument("--model", required=True, help="Model name/path as exposed by vLLM.")
    parser.add_argument("--base-url", default="http://0.0.0.0:8078/v1",
                        help="vLLM OpenAI-compatible endpoint.")
    parser.add_argument("--api-key", default=None,
                        help="API key expected by vLLM.")

    # Data
    parser.add_argument("--test-file", required=True, help="Test JSONL file.")
    parser.add_argument("--train-file", default=None,
                        help="Training JSONL file. Required for one-shot.")
    parser.add_argument("--output-file", required=True, help="Output JSONL file.")

    # Prompting
    parser.add_argument("--mode", choices=["zero_shot", "one_shot"], required=True,
                        help="Generation mode.")
    parser.add_argument("--demo-index", type=int, default=0,
                        help="Index of one-shot demonstration in training file.")

    # Generation
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--seed", type=int, default=None,
                        help="Optional generation seed. Leave unset for normal stochastic generation.")

    # Async / reliability
    parser.add_argument("--workers", type=int, default=8,
                        help="Maximum number of concurrent requests (default: 8).")
    parser.add_argument("--timeout", type=float, default=300.0,
                        help="Per-request timeout in seconds.")
    parser.add_argument("--max-retries", type=int, default=2,
                        help="OpenAI SDK retries for transient request failures.")

    # Dataset range. --num-instances is an alias for the earlier --max-examples.
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-examples", "--num-instances", dest="max_examples",
                        type=int, default=None,
                        help="Maximum number of test examples to process.")

    args = parser.parse_args()
    asyncio.run(async_main(args))


if __name__ == "__main__":
    main()

# python /mnt/fs1/akash/IndicNLG/infer/run_paraphrase_vllm_async.py \
#     --model qwen3-SFT-bn \
#     --test-file /mnt/fs1/akash/IndicNLG/eval_data/bengali/test_bn_subset.jsonl \
#     --output-file /mnt/fs1/akash/IndicNLG/preds/Qwen_SFT/predictions_mono_bn.jsonl \
#     --mode zero_shot