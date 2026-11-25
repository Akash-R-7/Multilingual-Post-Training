from datasets import load_dataset


def format_prompt(example):
    """
    Convert a text prompt into the conversational format expected by GRPO.

    The input already contains the paraphrasing instruction, so we preserve it.
    """
    return {
        "prompt": [
            {
                "role": "user",
                "content": example["prompt"],
            }
        ]
    }


def _prepare_dataset(file_path):
    """Load a JSONL dataset and prepare its columns for GRPO."""

    dataset = load_dataset(
        "json",
        data_files=str(file_path),
        split="train",
    )

    # Convert the original dataset columns to the names used by GRPO.
    rename_columns = {}

    if "inputs" in dataset.column_names and "prompt" not in dataset.column_names:
        rename_columns["inputs"] = "prompt"

    if "targets" in dataset.column_names and "solution" not in dataset.column_names:
        rename_columns["targets"] = "solution"

    if rename_columns:
        dataset = dataset.rename_columns(rename_columns)

    # Check that the expected columns exist.
    required_columns = {"prompt", "solution"}
    missing_columns = required_columns - set(dataset.column_names)

    if missing_columns:
        raise ValueError(
            f"Dataset is missing required columns: {missing_columns}. "
            f"Available columns: {dataset.column_names}"
        )

    # Convert text prompts into chat messages.
    dataset = dataset.map(format_prompt)

    return dataset


def load_train_dataset(train_file):
    """Load and prepare the GRPO training dataset."""
    return _prepare_dataset(train_file)


def load_eval_dataset(eval_file):
    """Load and prepare the evaluation dataset."""
    if eval_file is None:
        return None

    return _prepare_dataset(eval_file)