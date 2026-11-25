# Multilingual Post-Training

**Reinforcement learning and supervised fine-tuning for multilingual paraphrasing.**

This repository contains training, inference, and evaluation scripts for Indic-language paraphrase generation. The code covers supervised fine-tuning (SFT), group relative policy optimization (GRPO), asynchronous generation through a vLLM-compatible API, and LLM-judge evaluation.

## At a glance

- **Task:** Rewrite a source sentence while preserving its meaning.
- **Languages represented in the current data and scripts:** Hindi, Bengali, and Telugu; some evaluation prompts also support Marathi, Assamese, and Kannada.
- **Training approaches:** Full-parameter SFT and GRPO, with optional LoRA configuration for GRPO.
- **Evaluation workflow:** Generate predictions, then score them with an asynchronous LLM judge.

## Repository layout

| Path | What it contains |
| --- | --- |
| [`data/`](data/) | JSONL training and validation datasets, including Hindi, Bengali, and combined Indic-language data. |
| [`SFT/`](SFT/readme.md) | Qwen and reasoning-capable-model SFT scripts for monolingual and multilingual runs. |
| [`grpo_/`](grpo_/readme.md) | GRPO training entry points, YAML configuration, dataset utilities, and language-aware reward functions. |
| [`eval/infer/`](eval/infer/readme.md) | Asynchronous paraphrase generation against an OpenAI-compatible vLLM endpoint. |
| [`eval/judge/`](eval/judge/readme.md) | Asynchronous language-specific LLM evaluation of generated paraphrases. |

## Typical workflow

1. Prepare JSONL training, validation, and/or test data in the schema expected by the selected script.
2. Fine-tune a model with an SFT script, or run GRPO with the matching configuration and reward setup.
3. Generate predictions using `eval/infer/` and a running vLLM-compatible API server.
4. Optionally score prediction files with `eval/judge/`.

Each subdirectory README describes its scripts and retains its documented launch command. Check the selected script's arguments and configuration before running: paths, model IDs, language settings, and service endpoints are environment-specific.

## Data notes

The checked-in datasets are JSONL files. SFT scripts expect records with `inputs` and `targets`; the exact training and validation files are supplied as command-line arguments. GRPO uses the dataset loader and fields configured in `grpo_/configs/training.yaml`. Inference and judge scripts consume prediction/test JSONL records with source text and, where applicable, references. Consult the corresponding script and README when adapting a dataset.

## Requirements

Dependencies are listed separately for each workflow:

- SFT: [`SFT/sft_requirements.txt`](SFT/sft_requirements.txt)
- GRPO: [`grpo_/grpo_requirements.txt`](grpo_/grpo_requirements.txt)
- Inference and judging: [`eval/eval_requirements.txt`](eval/eval_requirements.txt)

Training scripts require a compatible CUDA/PyTorch environment and access to the selected model checkpoint. GRPO reward evaluation also depends on the configured judge service and language-identification resources. Inference and judge scripts require a reachable OpenAI-compatible endpoint. The pinned dependency files describe workflow-specific environments; they may not be interchangeable.

## Further reading

- [SFT scripts and launch command](SFT/readme.md)
- [GRPO scripts, configuration, and launch command](grpo_/readme.md)
- [Inference scripts and launch command](eval/infer/readme.md)
- [LLM judge and launch command](eval/judge/readme.md)
