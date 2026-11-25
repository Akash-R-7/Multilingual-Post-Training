# Supervised Fine-Tuning (SFT)

This directory contains supervised fine-tuning entry points for paraphrase generation. The scripts use conversational prompt/completion examples and train on the reference paraphrase (the assistant completion), rather than the input prompt.

## Training scripts

| Script | Purpose |
| --- | --- |
| `sft_run_qwen_monolingual.py` | Full-parameter SFT for Hindi paraphrasing with a Qwen instruct model. Validates JSONL fields and train/validation IDs, audits token lengths, and writes qualitative validation generations. |
| `sft_run_qwen_multilingual.py` | Full-parameter Qwen SFT for Hindi, Bengali, and Telugu paraphrasing. Checks language coverage and split IDs, audits sequence lengths, and generates a language-stratified qualitative sample. |
| `sft_run_think_model_monolingual.py` | Monolingual full-parameter SFT for Hindi or Bengali with a reasoning-capable model. Can create a deterministic validation split when none is supplied, and audits labels so reasoning scaffolding is excluded from the supervised loss. |
| `sft_run_think_model_multilingual.py` | Multilingual full-parameter SFT for Hindi, Bengali, and Telugu. Includes data and sequence-length diagnostics, language-stratified qualitative outputs, and checks that only reference paraphrases—not reasoning markers—are supervised. |

The reasoning-model scripts preserve useful diagnostics such as chat-template and token-label audits, and stop if the expected loss boundaries cannot be verified. All scripts require CUDA and use the shared `inputs` / `targets` JSONL fields; reasoning-model scripts also use language metadata for validation and reporting.

## Dependencies

The pinned SFT environment is in [`sft_requirements.txt`](sft_requirements.txt). Select a model checkpoint, data files, and output directory appropriate for your machine before training.

## Launch command

The command below is retained from this directory's existing documentation:

```bash
python sft_run_qwen_mixed_languages.py \
    --train_file <train_file_path.jsonl> \
    --validation_file <validation_file_path.jsonl> \
    --model_name Qwen/Qwen3-4B-Instruct-2507 \
    --output_dir <output_dir> \
    --epochs 3
```

**Command note:** `sft_run_qwen_mixed_languages.py` is not among the scripts currently tracked in this directory. The available Qwen entry points are listed above; inspect the selected script's `--help` output for its supported arguments before launching it.
