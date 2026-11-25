# LLM-judge evaluation

`llm_judge_eval_async.py` evaluates generated paraphrases using an OpenAI-compatible chat model. It selects a language-specific judging prompt and extracts structured judgments for:

- **Response quality** — whether the paraphrase is acceptable, including a failure category when applicable.
- **Language adherence** — whether the output follows the expected language.
- **Groundedness** — whether the paraphrase preserves the source meaning.

The script appends the original prediction fields and judge outputs to a JSONL results file. It supports concurrent requests, retries, optional resumption from an existing output, and a final summary of parsed scores.

## Input and service

Provide a prediction JSONL file with source `input`, `prediction`, and optional `references` fields. Start a compatible judge endpoint and select the served model. Dependencies are listed in [`../eval_requirements.txt`](../eval_requirements.txt).

## Launch command

This existing Hindi judging example is preserved:

```bash
python llm_judge_eval_async.py \
      --input-file <input_file_path.jsonl> \
      --output-file <output_file_path.jsonl> \
      --language hi \
      --model <VLLM_hosted_judge> \
      --reasoning-effort low
```

Run from this directory, or adjust the script path. Use `--help` for endpoint, judge type, concurrency, retry, and resume options.
