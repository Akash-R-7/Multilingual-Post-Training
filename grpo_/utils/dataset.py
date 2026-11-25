"""Dataset preparation for multilingual paraphrase GRPO.

Preserves the original language-specific generation prompt and adds a separate
source_sentence column for reward computation. The source is never reconstructed
inside the reward callback when this column is available.
"""
from __future__ import annotations

import re
from typing import Any

from datasets import load_dataset

SUPPORTED_LANGUAGES = {"hin", "ben", "tel"}

# Telugu templates in the supplied data all place the source after an instruction
# ending in a colon. Anchoring the match at the beginning and splitting only once
# avoids splitting a colon that belongs to the source sentence itself.
_TELUGU_PREFIX = re.compile(r"^\s*[^:\n]{1,100}:\s*", flags=re.DOTALL)


def _extract_quoted_source(prompt: str) -> str | None:
    pairs = [(r'"([^"\n]+)"', 'ascii_double_quote'),
             (r'“([^”\n]+)”', 'curly_double_quote'),
             (r'„([^“\n]+)“', 'low_curly_double_quote'),
             (r'‘([^’\n]+)’', 'curly_single_quote'),
             (r"'([^'\n]+)'", 'ascii_single_quote'),
             (r'「([^」\n]+)」', 'corner_quote'),
             (r'『([^』\n]+)』', 'white_corner_quote')]
    for pattern, method in pairs:
        match = re.search(pattern, prompt)
        if match and match.group(1).strip():
            return match.group(1).strip()
    return None


def extract_source_sentence(prompt: Any, language: str, template_id: Any = None) -> tuple[str, str]:
    """Extract source text without treating the instruction as source text."""
    text = str(prompt or "").strip()
    lang = str(language or "").lower().strip()
    if not text:
        raise ValueError("Empty prompt: cannot extract source_sentence")
    if lang not in SUPPORTED_LANGUAGES:
        raise ValueError(f"Unsupported/missing language {language!r}; expected one of {sorted(SUPPORTED_LANGUAGES)}")

    # Hindi and Bengali prompt templates in the provided data quote the source.
    if lang in {"hin", "ben"}:
        source = _extract_quoted_source(text)
        if source:
            return source, "quoted_source"
        raise ValueError(
            f"Could not extract quoted {lang} source (template_id={template_id!r}): {text[:180]!r}"
        )

    # Telugu templates use an instruction followed by a colon, sometimes with
    # no whitespace or with a newline before the source.
    match = _TELUGU_PREFIX.match(text)
    if match:
        source = text[match.end():].strip()
        if source:
            return source, f"telugu_instruction_prefix_template_{template_id}"
    raise ValueError(
        f"Could not extract Telugu source after instruction colon (template_id={template_id!r}): {text[:180]!r}"
    )


def _add_source_sentence(example):
    source, method = extract_source_sentence(
        example.get("prompt", example.get("inputs", "")),
        example.get("language", ""),
        example.get("template_id"),
    )
    return {"source_sentence": source, "source_extraction_method": method}


def _format_prompt(example):
    # Keep the original language-specific instruction untouched for generation.
    return {"prompt": [{"role": "user", "content": example["prompt"]}]}


def _prepare_dataset(file_path):
    dataset = load_dataset("json", data_files=str(file_path), split="train")
    rename_columns = {}
    if "inputs" in dataset.column_names and "prompt" not in dataset.column_names:
        rename_columns["inputs"] = "prompt"
    if "targets" in dataset.column_names and "solution" not in dataset.column_names:
        rename_columns["targets"] = "solution"
    if rename_columns:
        dataset = dataset.rename_columns(rename_columns)

    required = {"prompt", "solution", "language"}
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}; available={dataset.column_names}")

    # Fail during preprocessing rather than silently passing the full instruction
    # to the semantic judge or exact-copy detector.
    dataset = dataset.map(_add_source_sentence, desc="Extracting source sentences")
    dataset = dataset.map(_format_prompt, desc="Formatting GRPO prompts")
    return dataset


def load_train_dataset(train_file):
    return _prepare_dataset(train_file)


def load_eval_dataset(eval_file):
    return None if eval_file is None else _prepare_dataset(eval_file)
