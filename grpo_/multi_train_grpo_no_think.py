import os
import torch
import wandb

from pathlib import Path
from accelerate import PartialState
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import GRPOTrainer, GRPOConfig
from peft import LoraConfig, get_peft_model

from utils.config import load_configs
from utils.dataset import load_train_dataset, load_eval_dataset
# from rewards import bleu_reward
from rewards.combined import (
    make_combined_reward, set_current_step, set_current_phase,
    begin_eval_logging, finish_eval_logging,
)
import os

# ---------------------------------------------------------
# Weights & Biases
# ------------------------------------------------_


def init_wandb(cfg):
    wandb.init(
        project="Indic_Paraphrase_GRPO",
        name=cfg["training"]["experiment_name"],
        config={
            "model": cfg["model"]["model_name"],
            "learning_rate": cfg["training"]["learning_rate"],
            "batch_size": cfg["training"]["per_device_train_batch_size"],
            "gradient_accumulation_steps": (
                cfg["training"]["gradient_accumulation_steps"]
            ),
            "epochs": cfg["training"]["num_train_epochs"],
            "temperature": cfg["generation"]["temperature"],
            "num_generations": cfg["generation"]["num_generations"],
            "reward": cfg.get("reward", {}).get("reward_function", "combined"),
            "normalize_rewards": cfg.get("reward", {}).get("normalize_rewards", False),
            "reward_mode": cfg.get("reward", {}).get("mode", "lexicographic"),
            "reward_weights": cfg.get("reward", {}).get("weights", {}),
            "lexicographic_weights": cfg.get("reward", {}).get("lexicographic_weights", {}),
            "semantic_group_all_zero_gate": True,
            "individual_semantic_zero_gate": True,
        },
    )



def _find_subsequence(sequence, pattern, start=0):
    """Return the first start index of pattern in sequence, or None."""
    if not pattern:
        return None
    limit = len(sequence) - len(pattern) + 1
    for i in range(start, max(start, limit)):
        if sequence[i:i + len(pattern)] == pattern:
            return i
    return None


def build_answer_only_mask(completion_ids, completion_mask, tokenizer):
    """
    Build a loss mask that keeps only the final answer tokens active.

    - If a complete <answer>...</answer> block exists, optimize its content.
    - Otherwise, if <think>...</think> exists, optimize tokens after </think>.
    - If a think block is opened but not closed, mask the whole completion.
    - If no reasoning/answer markers occur, retain the whole completion for
      compatibility with outputs that do not use explicit tags.

    completion_mask remains untouched so answer tokens can still attend to
    the generated reasoning tokens in the forward pass.
    """
    think_open = tokenizer.encode("<think>", add_special_tokens=False)
    think_close = tokenizer.encode("</think>", add_special_tokens=False)
    answer_open = tokenizer.encode("<answer>", add_special_tokens=False)
    answer_close = tokenizer.encode("</answer>", add_special_tokens=False)

    if not all((think_open, think_close, answer_open, answer_close)):
        raise ValueError("Tokenizer produced an empty token sequence for a boundary marker.")

    loss_mask = torch.zeros_like(completion_mask)

    for row in range(completion_ids.shape[0]):
        valid_length = int(completion_mask[row].sum().item())
        ids = completion_ids[row, :valid_length].tolist()

        answer_start = _find_subsequence(ids, answer_open)
        answer_end = (
            _find_subsequence(ids, answer_close, answer_start + len(answer_open))
            if answer_start is not None else None
        )

        if answer_start is not None:
            # An explicit but incomplete answer block is not a valid target.
            if answer_end is not None:
                start = answer_start + len(answer_open)
                loss_mask[row, start:answer_end] = 1
            continue

        think_start = _find_subsequence(ids, think_open)
        if think_start is not None:
            think_end = _find_subsequence(ids, think_close, think_start + len(think_open))
            # Fail closed if reasoning was truncated before </think>.
            if think_end is not None:
                start = think_end + len(think_close)
                loss_mask[row, start:valid_length] = 1
            continue

        # No explicit reasoning marker: treat the whole completion as answer.
        loss_mask[row, :valid_length] = 1

    return loss_mask


# ---------------------------------------------------------
# GRPO training arguments
# ---------------------------------------------------------

def build_training_args(cfg):
    training_cfg = cfg["training"]
    generation_cfg = cfg["generation"]

    args = {
        "output_dir": str(training_cfg["output_dir"]),

        # Optimization
        "learning_rate": float(training_cfg["learning_rate"]),
        "per_device_train_batch_size": int(
            training_cfg["per_device_train_batch_size"]
        ),
        "gradient_accumulation_steps": int(
            training_cfg["gradient_accumulation_steps"]
        ),
        "num_train_epochs": int(training_cfg["num_train_epochs"]),
        "weight_decay": float(training_cfg["weight_decay"]),
        "max_grad_norm": float(training_cfg["max_grad_norm"]),

        # Scheduler
        "lr_scheduler_type": training_cfg["lr_scheduler_type"],
        "warmup_steps": 0,
        # Precision and memory
        "bf16": bool(training_cfg["bf16"]),
        "gradient_checkpointing": bool(
            training_cfg["gradient_checkpointing"]
        ),

        # Logging and checkpointing
        # "logging_steps": int(training_cfg["logging_steps"]),
        # "save_steps": int(training_cfg["save_steps"]),
        # "eval_steps": int(training_cfg["eval_steps"]),
        # "eval_strategy": "steps",
        # "load_best_model_at_end": False,
        # Logging and checkpointing
        "logging_steps": int(training_cfg["logging_steps"]),
        "save_strategy": "steps",
        "save_steps": int(training_cfg["save_steps"]),
        "eval_strategy": "steps",
        "eval_steps": int(training_cfg["eval_steps"]),
        # Select the best checkpoint using evaluation reward
        "load_best_model_at_end": True,
        "metric_for_best_model": "eval_reward",
        "greater_is_better": True,

        # Retain recent checkpoints and the best checkpoint
        "save_total_limit": 6,

        "report_to": training_cfg["report_to"],
        "seed": int(training_cfg["seed"]),

        # Completion logging
        "log_completions": True,
        "num_completions_to_print": 2,

        # Generation
        "max_completion_length": int(
            generation_cfg["max_completion_length"]
        ),
        # TRL truncates prompts to this length before generation.
        # This was present in generation.yaml but was not forwarded before.
        # "max_prompt_length": int(generation_cfg["max_prompt_length"]),
        # Keep language/source metadata available to the reward function.
        "remove_unused_columns": False,
        "temperature": float(generation_cfg["temperature"]),
        "top_p": float(generation_cfg["top_p"]),
        "top_k": int(generation_cfg["top_k"]),
        "num_generations": int(generation_cfg["num_generations"]),

        # vLLM colocate mode: vLLM runs inside each trainer process and
        # shares the training GPU. No external vLLM server is required.
        "use_vllm": True,
        "vllm_mode": "colocate",
        "vllm_gpu_memory_utilization": float(
            training_cfg.get("vllm_gpu_memory_utilization", 0.40)
        ),
    }

    # Allow a short smoke test when max_steps is configured.
    # If max_steps is absent, the trainer uses num_train_epochs.
    if training_cfg.get("max_steps") is not None:
        args["max_steps"] = int(training_cfg["max_steps"])

    # Forward optional gradient-checkpointing settings.
    gc_kwargs = training_cfg.get("gradient_checkpointing_kwargs")
    if gc_kwargs is not None:
        args["gradient_checkpointing_kwargs"] = gc_kwargs

    import inspect

    supported = set(inspect.signature(GRPOConfig).parameters)
    unsupported = sorted(set(args) - supported)

    print("Unsupported GRPOConfig arguments:", unsupported)

    if unsupported:
        raise ValueError(f"Unsupported GRPOConfig arguments: {unsupported}")

    return GRPOConfig(**args)


class RewardStepCallback(TrainerCallback):
    """Expose the upcoming optimizer step to reward diagnostics."""

    def on_step_begin(self, args, state, control, **kwargs):
        set_current_step(int(state.global_step) + 1)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        set_current_step(None)
        return control


class PhaseAwareGRPOTrainer(GRPOTrainer):
    """Tag eval calls and restrict policy loss to final-answer tokens."""

    def _prepare_inputs(self, generation_batch):
        # Generate/score the full completion first. Keep completion_mask intact
        # so the answer can attend to the sampled reasoning tokens.
        inputs = super()._prepare_inputs(generation_batch)
        answer_mask = build_answer_only_mask(
            inputs["completion_ids"],
            inputs["completion_mask"],
            self.processing_class,
        )

        # TRL 1.16.0 forms its token loss mask as completion_mask * tool_mask.
        # Reuse that pathway to mask reasoning for loss without masking it from
        # the model's attention context.
        if "tool_mask" in inputs:
            inputs["tool_mask"] = inputs["tool_mask"] * answer_mask
        else:
            inputs["tool_mask"] = answer_mask
        return inputs

    def evaluate(self, *args, **kwargs):
        step = int(getattr(self.state, "global_step", 0))
        begin_eval_logging(step)
        set_current_step(step)
        set_current_phase("eval")
        try:
            return super().evaluate(*args, **kwargs)
        finally:
            finish_eval_logging(step)
            set_current_phase("train")


# ---------------------------------------------------------
# Main training function
# ---------------------------------------------------------

def main():
    # Load YAML configurations.
    cfg = load_configs()

    # Save per-completion judge diagnostics alongside this experiment's outputs.
    os.environ.setdefault(
        "REWARD_LOG_DIR",
        str(Path(cfg["training"]["output_dir"]).parent / "reward_logs"),
    )
    combined_reward = make_combined_reward(
        cfg["reward"],
        expected_group_size=int(cfg["generation"]["num_generations"]),
    )

    # Initialize distributed state.
    state = PartialState()

    # Initialize W&B only on the main process.
    if state.is_main_process:
        init_wandb(cfg)

    # -----------------------------------------------------
    # Tokenizer
    # -----------------------------------------------------

    model_name = cfg["model"]["model_name"]
    if state.is_main_process:
        print(f"Loading policy model: {model_name}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=cfg["model"]["trust_remote_code"],
        padding_side=cfg["model"]["padding_side"],
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # -----------------------------------------------------
    # Model
    # -----------------------------------------------------

    dtype_name = str(cfg["model"].get("torch_dtype", "bfloat16"))
    if not hasattr(torch, dtype_name):
        raise ValueError(
            f"Unsupported torch_dtype={dtype_name!r}; expected a torch dtype name "
            "such as 'bfloat16' or 'float16'."
        )
    model_dtype = getattr(torch, dtype_name)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=cfg["model"]["trust_remote_code"],
        torch_dtype=model_dtype,
        attn_implementation=cfg["model"]["attn_implementation"],
    )

    # Set the model's padding token consistently.
    model.config.pad_token_id = tokenizer.pad_token_id

    # -----------------------------------------------------
    # Optional LoRA
    # -----------------------------------------------------

    if cfg["lora"]["use_lora"]:
        peft_config = LoraConfig(
            r=int(cfg["lora"]["r"]),
            lora_alpha=int(cfg["lora"]["lora_alpha"]),
            lora_dropout=float(cfg["lora"]["lora_dropout"]),
            bias=cfg["lora"]["bias"],
            task_type=cfg["lora"]["task_type"],
            target_modules=cfg["lora"]["target_modules"],
        )

        model = get_peft_model(model, peft_config)

        if state.is_main_process:
            model.print_trainable_parameters()

    # -----------------------------------------------------
    # Trainable-parameter diagnostic (after optional LoRA wrapping)
    # -----------------------------------------------------
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    total_params = sum(p.numel() for p in model.parameters())

    if state.is_main_process:
        print("\n=== Model parameter diagnostic ===")
        print(f"Trainable parameters: {trainable_params:,}")
        print(f"Total parameters: {total_params:,}")
        print(
            f"Trainable percentage: "
            f"{100 * trainable_params / total_params:.4f}%"
        )
        print("LoRA enabled in config:", cfg["lora"]["use_lora"])

    # -----------------------------------------------------
    # Datasets
    # -----------------------------------------------------

    train_dataset = load_train_dataset(
        cfg["training"]["train_file"]
    )

    eval_dataset = load_eval_dataset(
        cfg["training"].get("eval_file")
    )

    if state.is_main_process:
        print("\nTraining dataset:")
        print(train_dataset)
        print("Training columns:", train_dataset.column_names)
        print("Example prompt:", train_dataset[0]["prompt"])
        print("Example reference:", train_dataset[0]["solution"])

        if eval_dataset is not None:
            print("\nEvaluation dataset:")
            print(eval_dataset)

    # -----------------------------------------------------
    # Build and inspect the exact GRPO configuration
    # -----------------------------------------------------
    training_args = build_training_args(cfg)

    if state.is_main_process:
        print("\n=== Effective training configuration ===")
        print("per_device_train_batch_size:",
              training_args.per_device_train_batch_size)
        print("gradient_accumulation_steps:",
              training_args.gradient_accumulation_steps)
        print("num_train_epochs:", training_args.num_train_epochs)
        print("max_steps:", training_args.max_steps)
        print("world_size:", training_args.world_size)
        print("dataloader_drop_last:", training_args.dataloader_drop_last)
        print("Training dataset rows:", len(train_dataset))
        print(
            "Evaluation dataset rows:",
            len(eval_dataset) if eval_dataset is not None else "None"
        )

    # -----------------------------------------------------
    # GRPO Trainer
    # -----------------------------------------------------
    trainer = PhaseAwareGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[combined_reward],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[RewardStepCallback()],
    )

    if state.is_main_process:
        print("\nAnswer-only policy loss: ENABLED")
        print("Reasoning is retained as context; only final-answer tokens contribute to token-masked losses.")
        print("\nMax training steps:", trainer.state.max_steps)
        print("Starting GRPO training...")

    # -----------------------------------------------------
    # Train
    # -----------------------------------------------------

    trainer.train()

    # -----------------------------------------------------
    # Save model and tokenizer
    # -----------------------------------------------------

    if state.is_main_process:
        print("\n=== Best checkpoint information ===")
        print("Best checkpoint:", trainer.state.best_model_checkpoint)
        print("Best evaluation reward:", trainer.state.best_metric)

        experiment_dir = Path(
            cfg["training"]["output_dir"]
        ).parent

        best_model_dir = experiment_dir / "best_model"
        best_model_dir.mkdir(parents=True, exist_ok=True)

        # With load_best_model_at_end=True, the trainer has
        # restored the best checkpoint into the model.
        trainer.save_model(str(best_model_dir))
        tokenizer.save_pretrained(str(best_model_dir))

        print(f"\nBest model saved to: {best_model_dir}")

        wandb.finish()


if __name__ == "__main__":
    main()