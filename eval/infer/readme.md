# Paraphrase inference

The inference utilities generate paraphrases asynchronously through an OpenAI-compatible vLLM server. They save predictions as JSONL; they do not calculate evaluation scores.

## Scripts

- `run_paraphrase_vllm_async.py` supports **zero-shot** and **one-shot** prompts. In one-shot mode, it selects a source/target demonstration from a training JSONL file using `--demo-index`.
- `run_paraphrase_vllm_async_think.py` supports the same prompting modes and additionally separates reasoning and final answers from tagged or separately returned reasoning output. Its `prediction` field contains the final answer for compatibility with downstream evaluation, while raw completion and parsing diagnostics are also recorded.

Both scripts support bounded concurrent requests, configurable generation settings, dataset ranges, and appending results so an interrupted generation run can be resumed. Input examples should provide a source field (such as `input`); one-shot demonstrations also need a target field. Test examples may include `references` for preservation in the output.

## Dependencies and endpoint

Install the inference dependencies listed in [`../eval_requirements.txt`](../eval_requirements.txt) in a compatible environment. Start a vLLM server that exposes an OpenAI-compatible chat-completions endpoint, then set the model name and endpoint to match your server.

## Launch command

Run from this directory, or adjust the script path as needed. This existing zero-shot example is preserved:

```bash
python run_paraphrase_vllm_async.py \
    --model <model_path> \
    --test-file <test_file_path.jsonl> \
    --output-file <output_file_path.jsonl> \
    --mode zero_shot
```

For one-shot generation, provide `--train-file`; use `--help` to see additional options such as worker count, generation limits, and the demonstration index.
