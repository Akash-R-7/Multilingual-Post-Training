from pathlib import Path
import yaml

# Project layout:
# /mnt/fs1/akash/TRL_/
# └── grpo/
#     ├── configs/
#     ├── utils/
#     │   ├── __init__.py
#     │   ├── config.py
#     │   └── dataset.py
#     └── multi_train_grpo_with_diagnostics.py

PROJECT_ROOT = Path(__file__).resolve().parents[2]
GRPO_ROOT = PROJECT_ROOT / "grpo"
CONFIG_DIR = GRPO_ROOT / "configs"


def load_yaml(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            f"Configuration file not found: {path}"
        )

    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    return config or {}


def resolve_path(path_str):
    if path_str is None:
        return None

    path = Path(path_str)

    if path.is_absolute():
        return path

    return (PROJECT_ROOT / path).resolve()


def load_configs():
    cfg = {
        "model": load_yaml(CONFIG_DIR / "model.yaml"),
        "training": load_yaml(CONFIG_DIR / "training.yaml"),
        "generation": load_yaml(CONFIG_DIR / "generation.yaml"),
        "reward": load_yaml(CONFIG_DIR / "reward.yaml"),
        "lora": load_yaml(CONFIG_DIR / "lora.yaml"),
    }

    # Resolve dataset and output paths.
    training_cfg = cfg["training"]

    for key in ("train_file", "eval_file", "output_dir"):
        if key in training_cfg:
            training_cfg[key] = resolve_path(
                training_cfg[key]
            )

    # Validate dataset paths.
    for key in ("train_file", "eval_file"):
        path = training_cfg.get(key)

        if path is not None and not path.is_file():
            raise FileNotFoundError(
                f"{key} does not exist: {path}"
            )

    # Create the output directory.
    output_dir = training_cfg.get("output_dir")

    if output_dir is not None:
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Config directory: {CONFIG_DIR}")
    print(f"Training file: {training_cfg.get('train_file')}")
    print(f"Evaluation file: {training_cfg.get('eval_file')}")
    print(f"Output directory: {output_dir}")

    return cfg