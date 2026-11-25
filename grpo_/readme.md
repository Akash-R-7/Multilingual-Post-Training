# Group Relative Policy Optimization (GRPO)

This directory contains GRPO training pipelines for paraphrase generation. The entry points combine a configured causal language model, dataset loaders, TRL's `GRPOTrainer`, optional LoRA adapters, generation settings, and reward functions. Training and evaluation metrics are logged to Weights & Biases when enabled, and the selected best checkpoint is saved with its tokenizer.

## Training entry points

| Script | Purpose |
| --- | --- |
| `multi_train_grpo_indic.py` | Multilingual Indic GRPO, with configured vLLM integration and a shared reward that checks output language, penalizes exact source copying, evaluates semantic fidelity and naturalness, and includes reference-based chrF++. |
| `multi_train_grpo_mono.py` | Monolingual GRPO entry point using its monolingual configuration/reward imports and colocated vLLM generation. |
| `multi_train_grpo_no_think.py` | GRPO variant that retains reasoning as generation context while applying token-level policy loss to the final answer when recognized answer/reasoning markers are present. |

The shared and language-specific reward modules live in `rewards/`; they include output extraction, language and exact-copy gates, configurable reward combination, judge diagnostics, and rollout logging. YAML files in `configs/` separate model, generation, optimization, LoRA, and reward settings.

## Dependencies and setup

The pinned GRPO environment is listed in [`grpo_requirements.txt`](grpo_requirements.txt). GRPO training requires compatible GPU/CUDA, PyTorch, Transformers, TRL, Accelerate, and vLLM components. The reward functions also call a configured OpenAI-compatible judge service and may download the GlotLID model on first use.

## Launch command

The documented colocated-mode launch command is preserved below:

```bash
CUDA_VISIBLE_DEVICES=<GPU_ID> \
    NCCL_NVLS_ENABLE=0 \
    accelerate launch \
    --num_processes <N> \
    multi_train_grpo_indic.py
```

Before launching, verify that `configs/` contains the settings you intend to use and that the selected GPU count matches `<N>`.
