# LLM-judge evaluation

`llm_judge_eval_async.py` evaluates generated paraphrases using an OpenAI-compatible chat model. It selects a language-specific judging prompt and extracts structured judgments for response quality, language adherence, and groundedness.

## Input and service

Provide a prediction JSONL file with source `input`, `prediction`, and optional `references` fields. Start a compatible judge endpoint and select the served model. Dependencies are listed in [`../eval_requirements.txt`](../eval_requirements.txt).

## Hindi launch command

Run from this directory:

```bash
python llm_judge_eval_async.py \
    --input-file ../infer/predictions_hi.jsonl \
    --output-file <output_file_path.jsonl> \
    --language hi \
    --model <VLLM_hosted_judge> \
    --reasoning-effort low
```

The corresponding Hindi test input is `../../data/test_data_Indic_NLG_Suite/test_hi_subset.jsonl`, as documented in [`../../data/README.md`](../../data/README.md).
