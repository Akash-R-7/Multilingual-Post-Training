"""Hindi paraphrase reward with GlotLID + exact-copy hard gates.

Modes:
- lexicographic: semantic tier > naturalness > reference chrF++ (default coefficients 1.0, 0.1, 0.001).
- weighted: 0.6*semantic + 0.3*naturalness + 0.1*chrF++ after both gates.
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import wandb
from openai import OpenAI
from sacrebleu.metrics import CHRF

try:
    from .judge_prompts_hi import build_judge_prompt, parse_judge_response
except ImportError:
    from TRL_.grpo.rewards.judge_prompts_hi import build_judge_prompt, parse_judge_response

_chrfpp = CHRF(word_order=2)
_thread_local = threading.local()
_log_lock = threading.Lock()
_lid_lock = threading.Lock()
_call_counter = 0
LAST_ROWS: list[dict] = []
_current_step: int | None = None
_current_phase = "train"
_eval_buffer: dict[int, list[dict]] = {}
_lid_model = None

DEFAULT_JUDGE_BASE_URL = os.getenv("JUDGE_BASE_URL", "http://0.0.0.0:8008/v1")
DEFAULT_JUDGE_MODEL = os.getenv("JUDGE_MODEL", "gemma4-judge")
DEFAULT_JUDGE_API_KEY = os.getenv("JUDGE_API_KEY", "EMPTY")
MAX_WORKERS = max(1, int(os.getenv("JUDGE_MAX_WORKERS", "8")))
JUDGE_TIMEOUT = float(os.getenv("JUDGE_TIMEOUT", "90"))
MAX_TOKENS = int(os.getenv("JUDGE_MAX_TOKENS", "1024"))
TARGET_LID_LABEL = os.getenv("TARGET_LID_LABEL", "hin_Deva")
GLOTLID_MIN_CONFIDENCE = float(os.getenv("GLOTLID_MIN_CONFIDENCE", "0.60"))
GLOTLID_REPO_ID = os.getenv("GLOTLID_REPO_ID", "cis-lmu/glotlid")
GLOTLID_MODEL_FILE = os.getenv("GLOTLID_MODEL_FILE", "model_v3.bin")


def set_current_step(step: int | None) -> None:
    global _current_step
    _current_step = step


def set_current_phase(phase: str) -> None:
    global _current_phase
    _current_phase = str(phase)


def begin_eval_logging(step: int) -> None:
    with _log_lock:
        _eval_buffer[int(step)] = []


def _aggregate(rows: list[dict], prefix: str = "eval/") -> dict:
    denom = max(len(rows), 1)
    mean = lambda key: sum(float(r.get(key, 0.0)) for r in rows) / denom
    return {
        f"{prefix}reward_semantic_mean": mean("semantic_score"),
        f"{prefix}reward_naturalness_mean": mean("naturalness_score"),
        f"{prefix}reward_chrfpp_reference_mean": mean("chrfpp_reference"),
        f"{prefix}reward_combined_mean": mean("combined"),
        f"{prefix}judge/error_count": sum(bool(r.get("semantic_error")) for r in rows),
        f"{prefix}lid/pass_rate": mean("lid_pass"),
        f"{prefix}lid/fail_count": sum(not bool(r.get("lid_pass")) for r in rows),
        f"{prefix}copy_gate/fail_count": sum(bool(r.get("exact_copy")) for r in rows),
        f"{prefix}copy_gate/fail_rate": mean("exact_copy"),
        f"{prefix}reward/gated_out_count": sum(not bool(r.get("gates_pass")) for r in rows),
        f"{prefix}semantic_group/all_zero_count": sum(bool(r.get("group_all_semantic_zero")) for r in rows),
        f"{prefix}semantic_individual/zero_count": sum(not bool(r.get("individual_semantic_gate_pass")) for r in rows),
        f"{prefix}source_extraction_fallback_count": sum(r.get("source_extraction_method") != "quoted_span" for r in rows),
        f"{prefix}empty_completion_count": sum(not str(r.get("completion", "")).strip() for r in rows),
        f"{prefix}rollout_count": len(rows),
    }


def finish_eval_logging(step: int) -> None:
    with _log_lock:
        rows = _eval_buffer.pop(int(step), [])
    if not rows:
        print(f"[combined_reward] No eval rollout rows captured at step={step}.")
        return
    log_dir = Path(os.getenv("REWARD_LOG_DIR", "reward_logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / f"eval_step_{int(step):06d}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    metrics = _aggregate(rows, "eval/")
    if wandb.run is not None:
        try:
            wandb.log(metrics, step=int(step), commit=False)
        except Exception as exc:
            print(f"[combined_reward] W&B eval metric logging warning: {exc}")
    print("[combined_reward] eval aggregate " + json.dumps({"step": int(step), **metrics}, ensure_ascii=False))


def _client() -> OpenAI:
    client = getattr(_thread_local, "client", None)
    if client is None:
        client = OpenAI(base_url=DEFAULT_JUDGE_BASE_URL, api_key=DEFAULT_JUDGE_API_KEY,
                        timeout=JUDGE_TIMEOUT, max_retries=1)
        _thread_local.client = client
    return client


def _get_lid_model():
    """Load GlotLID once per process. Requires fasttext + huggingface_hub."""
    global _lid_model
    if _lid_model is None:
        with _lid_lock:
            if _lid_model is None:
                import fasttext
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download(repo_id=GLOTLID_REPO_ID, filename=GLOTLID_MODEL_FILE)
                _lid_model = fasttext.load_model(model_path)
    return _lid_model


def identify_language(text: str) -> dict:
    """Return top GlotLID label/confidence; fail closed on empty text or LID errors."""
    clean = " ".join(str(text or "").split())
    if not clean:
        return {"label": "EMPTY", "confidence": 0.0, "passed": False, "error": "empty completion"}
    try:
        labels, scores = _get_lid_model().predict(clean, k=1)
        label = str(labels[0]).replace("__label__", "")
        confidence = min(max(float(scores[0]), 0.0), 1.0)  # fastText may overshoot 1 by tiny float error
        passed = label == TARGET_LID_LABEL and confidence >= GLOTLID_MIN_CONFIDENCE
        return {"label": label, "confidence": confidence, "passed": passed, "error": ""}
    except Exception as exc:
        return {"label": "LID_ERROR", "confidence": 0.0, "passed": False,
                "error": f"{type(exc).__name__}: {exc}"[:1000]}


def _as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        return str(value.get("content", value))
    if isinstance(value, (list, tuple)):
        parts = []
        for item in value:
            if isinstance(item, dict):
                content = item.get("content", "")
                if content:
                    parts.append(str(content))
            elif item is not None:
                parts.append(str(item))
        return "\n".join(parts)
    return str(value)


def extract_answer(value: Any) -> str:
    text = _as_text(value).strip()
    if "<answer>" in text and "</answer>" in text:
        text = text.split("<answer>", 1)[1].split("</answer>", 1)[0]
    text = re.sub(r"^\s*<think>.*?</think>\s*", "", text, flags=re.DOTALL)
    return text.strip()


def _extract_source_sentence(prompt_value: Any) -> tuple[str, str]:
    prompt = _as_text(prompt_value).strip()
    patterns = [r'"([^"\n]+)"', r'“([^”\n]+)”', r'„([^“\n]+)“',
                r'‘([^’\n]+)’', r'「([^」\n]+)」', r'『([^』\n]+)』']
    for pattern in patterns:
        matches = re.findall(pattern, prompt)
        if matches:
            candidate = matches[-1].strip()
            if candidate:
                return candidate, "quoted_span"
    return prompt, "full_prompt_fallback"


def _normalize_for_exact_copy(text: str) -> str:
    """Normalize Unicode, whitespace and punctuation for conservative copy detection."""
    import unicodedata
    value = unicodedata.normalize("NFKC", str(text or ""))
    # Ignore all Unicode punctuation and whitespace, but retain letters/digits.
    return "".join(
        ch for ch in value
        if not ch.isspace() and not unicodedata.category(ch).startswith("P")
    )

def is_exact_source_copy(source: str, prediction: str) -> bool:
    """True when output is source verbatim modulo whitespace/punctuation/Unicode form."""
    a = _normalize_for_exact_copy(source)
    b = _normalize_for_exact_copy(prediction)
    return bool(a) and a == b

def _call_semantic_judge(prompt: str, source: str, prediction: str, reference: str) -> dict:
    started = time.perf_counter()
    try:
        messages = build_judge_prompt(input_text=prompt, source_sentence=source,
                                      reference=reference, prediction=prediction)
        response = _client().chat.completions.create(
            model=DEFAULT_JUDGE_MODEL, messages=messages, temperature=0,
            max_tokens=MAX_TOKENS,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        raw = response.choices[0].message.content or ""
        parsed = parse_judge_response(raw)
        return {
            "semantic_score": float(parsed["semantic_fidelity"]),
            "semantic_reason": str(parsed["semantic_reason"])[:1000],
            "naturalness_score": int(parsed["naturalness"]),
            "naturalness_reason": str(parsed["naturalness_reason"])[:1000],
            "raw_response": raw[:5000], "error": "",
            "latency": time.perf_counter() - started,
        }
    except Exception as exc:
        return {
            "semantic_score": 0.0, "semantic_reason": "",
            "naturalness_score": 0, "naturalness_reason": "",
            "raw_response": "", "error": f"{type(exc).__name__}: {exc}"[:1000],
            "latency": time.perf_counter() - started,
        }


def _append_jsonl(row: dict) -> None:
    log_dir = Path(os.getenv("REWARD_LOG_DIR", "reward_logs"))
    log_dir.mkdir(parents=True, exist_ok=True)
    filename = "reward_rollouts_eval.jsonl" if row.get("phase") == "eval" else "reward_rollouts_train.jsonl"
    with _log_lock:
        with (log_dir / filename).open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _log_batch(rows: list[dict], means: dict) -> None:
    global _call_counter
    with _log_lock:
        _call_counter += 1
        call_id = _call_counter
    for row in rows:
        row["reward_call_id"] = call_id
        row["step"] = _current_step
        row["phase"] = _current_phase
        _append_jsonl(row)
        if _current_phase == "eval" and _current_step is not None:
            with _log_lock:
                _eval_buffer.setdefault(int(_current_step), []).append(row.copy())
    if _current_phase == "eval" or wandb.run is None:
        return
    payload = {
        "reward/chrfpp_reference_mean": means["chrfpp_reference"],
        "reward/semantic_mean": means["semantic"],
        "reward/naturalness_mean": means["naturalness"],
        "reward/combined_mean": means["combined"],
        "reward/mode": means["mode"],
        "copy_gate/fail_count": sum(bool(r["exact_copy"]) for r in rows),
        "reward/gated_out_count": sum(not r["gates_pass"] for r in rows),
        "reward/semantic_group_all_zero_count": sum(bool(r["group_all_semantic_zero"]) for r in rows),
        "reward/semantic_individual_zero_count": sum(not r["individual_semantic_gate_pass"] for r in rows),
        "judge/semantic_error_count": sum(bool(r["semantic_error"]) for r in rows),
        "judge/semantic_latency_mean": sum(r["semantic_latency"] for r in rows) / max(len(rows), 1),
        "lid/pass_rate": means["lid_pass_rate"],
        "lid/fail_count": sum(not r["lid_pass"] for r in rows),
        "lid/error_count": sum(bool(r["lid_error"]) for r in rows),
        "lid/mean_confidence": means["lid_confidence"],
        "reward/source_extraction_fallback_count": sum(r["source_extraction_method"] != "quoted_span" for r in rows),
    }
    columns = ["step", "phase", "source", "reference", "completion", "source_extraction_method",
               "lid_label", "lid_confidence", "lid_pass", "lid_error", "exact_copy", "gates_pass",
               "group_size", "group_all_semantic_zero", "individual_semantic_gate_pass", "semantic_eligible",
               "chrfpp_reference", "semantic_score", "semantic_reason", "naturalness_score",
               "naturalness_reason", "semantic_raw_response", "semantic_error", "reward_mode",
               "combined_reward"]
    table = wandb.Table(columns=columns)
    for r in rows:
        table.add_data(r.get("step"), r.get("phase"), r["source"][:1000], r["reference"][:1000],
                       r["completion"][:1000], r["source_extraction_method"], r["lid_label"],
                       r["lid_confidence"], r["lid_pass"], r["lid_error"][:500], r["exact_copy"],
                       r["gates_pass"], r["group_size"], r["group_all_semantic_zero"],
                       r["individual_semantic_gate_pass"], r["semantic_eligible"],
                       r["chrfpp_reference"], r["semantic_score"],
                       r["semantic_reason"][:500], r["naturalness_score"],
                       r["naturalness_reason"][:500], r["semantic_raw_response"][:1000],
                       r["semantic_error"][:500], r["reward_mode"], r["combined"])
    payload["reward/rollouts"] = table
    try:
        wandb.log(payload, commit=False)
    except Exception as exc:
        print(f"[combined_reward] W&B logging warning: {exc}")


def make_combined_reward(reward_cfg: dict, expected_group_size: int | None = None):
    """Build either reward mode with shared gates and per-prompt semantic safeguards."""
    if expected_group_size is not None and int(expected_group_size) < 1:
        raise ValueError("expected_group_size must be a positive integer")
    mode = str(reward_cfg.get("mode", "lexicographic")).strip().lower()
    if mode not in {"lexicographic", "weighted"}:
        raise ValueError("reward mode must be 'lexicographic' or 'weighted'")

    weights = reward_cfg.get("weights", {})
    w_sem = float(weights.get("semantic", 0.60))
    w_nat = float(weights.get("naturalness", 0.30))
    w_ref = float(weights.get("chrfpp_reference", 0.10))
    if min(w_sem, w_nat, w_ref) < 0 or abs((w_sem + w_nat + w_ref) - 1.0) > 1e-6:
        raise ValueError(f"Weighted-mode weights must be nonnegative and sum to 1: {weights}")

    # Lexicographic scalarization: each higher tier dominates the maximum sum
    # of all lower tiers. Semantic gaps are 0.5; naturalness gap is 1.
    lex_weights = reward_cfg.get("lexicographic_weights", {})
    lex_sem = float(lex_weights.get("semantic", 1.0))
    lex_nat = float(lex_weights.get("naturalness", 0.1))
    lex_ref = float(lex_weights.get("chrfpp_reference", 0.001))
    if not (lex_sem * 0.5 > lex_nat + lex_ref and lex_nat > lex_ref):
        raise ValueError(
            "Lexicographic weights must satisfy 0.5*semantic_weight > "
            "naturalness_weight + chrfpp_weight, and naturalness_weight > chrfpp_weight."
        )
    if not (0.0 <= GLOTLID_MIN_CONFIDENCE <= 1.0):
        raise ValueError("GLOTLID_MIN_CONFIDENCE must be in [0, 1]")

    def combined_reward(completions, solution=None, prompts=None, prompt=None, **kwargs):
        global LAST_ROWS
        n = len(completions)
        if solution is None:
            solution = kwargs.get("reference_output", kwargs.get("reference", [""] * n))
        refs = list(solution) if isinstance(solution, (list, tuple)) else [solution] * n
        prompt_values = prompts if prompts is not None else prompt
        if prompt_values is None:
            prompt_values = kwargs.get("inputs", [""] * n)
        prompt_values = list(prompt_values) if isinstance(prompt_values, (list, tuple)) else [prompt_values] * n
        if len(refs) != n:
            raise ValueError(f"completion/reference length mismatch: {n} vs {len(refs)}")
        if len(prompt_values) != n:
            prompt_values = [str(prompt_values[0] if prompt_values else "")] * n

        preds = [extract_answer(c) for c in completions]
        refs_text = [extract_answer(r) for r in refs]
        prompts_text = [_as_text(p) for p in prompt_values]
        extracted = [_extract_source_sentence(p) for p in prompt_values]
        source_texts = [x[0] for x in extracted]
        source_methods = [x[1] for x in extracted]

        lid_results = [identify_language(pred) for pred in preds]
        exact_copies = [is_exact_source_copy(src, pred) for src, pred in zip(source_texts, preds)]
        ref_scores = [
            float(_chrfpp.sentence_score(pred, [ref]).score / 100.0)
            if pred.strip() and ref.strip() else 0.0
            for pred, ref in zip(preds, refs_text)
        ]

        semantic_results = [None] * n
        futures = {}
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            for i in range(n):
                if lid_results[i]["passed"] and not exact_copies[i]:
                    futures[pool.submit(
                        _call_semantic_judge, prompts_text[i], source_texts[i],
                        preds[i], refs_text[i]
                    )] = i
            for future in as_completed(futures):
                semantic_results[futures[future]] = future.result()

        # GRPO supplies repeated prompt strings for the rollouts belonging to
        # each generation group. Compute the semantic safeguard per prompt,
        # never across the whole reward batch.
        group_indices: dict[str, list[int]] = {}
        for i, prompt_text in enumerate(prompts_text):
            group_indices.setdefault(prompt_text, []).append(i)

        effective_semantic_scores = [0.0] * n
        for i in range(n):
            candidate = semantic_results[i]
            if (lid_results[i]["passed"] and not exact_copies[i]
                    and candidate is not None and not candidate.get("error")):
                effective_semantic_scores[i] = float(candidate["semantic_score"])

        group_all_semantic_zero = [False] * n
        group_sizes = [1] * n
        for group_prompt, indices in group_indices.items():
            if expected_group_size is not None and len(indices) != int(expected_group_size):
                raise RuntimeError(
                    "Incomplete GRPO prompt group in reward callback: "
                    f"observed {len(indices)} rollout(s), expected {int(expected_group_size)}. "
                    "The group-level semantic safeguard requires all generations for a prompt "
                    "to be present in the same reward-function call. Check the installed TRL "
                    "version, generation batch size, and distributed sharding."
                )
            all_zero = max((effective_semantic_scores[j] for j in indices), default=0.0) == 0.0
            for j in indices:
                group_all_semantic_zero[j] = all_zero
                group_sizes[j] = len(indices)

        rows, rewards = [], []
        for i in range(n):
            sem = semantic_results[i] or {
                "semantic_score": 0.0, "semantic_reason": "",
                "naturalness_score": 0, "naturalness_reason": "",
                "raw_response": "",
                "error": "skipped_by_lid_gate" if not lid_results[i]["passed"]
                         else ("exact_source_copy" if exact_copies[i] else "judge_not_called"),
                "latency": 0.0,
            }
            gates_pass = bool(lid_results[i]["passed"] and not exact_copies[i])
            judge_ok = gates_pass and not sem["error"]
            if judge_ok:
                s = float(sem["semantic_score"])
                nat = int(sem["naturalness_score"])
            else:
                s, nat = 0.0, 0

            # Option 2: semantic score zero is an individual hard stop, even
            # when another rollout for the same prompt has nonzero semantics.
            # The group-level all-zero safeguard is retained explicitly for
            # diagnostics and to make the group failure condition auditable.
            semantic_eligible = (s > 0.0) and not group_all_semantic_zero[i]
            if gates_pass and judge_ok and semantic_eligible:
                if mode == "lexicographic":
                    # 0.5 semantic gap > 0.1 + 0.001 lower-tier maximum.
                    total = lex_sem * s + lex_nat * nat + lex_ref * ref_scores[i]
                else:
                    total = w_sem * s + w_nat * nat + w_ref * ref_scores[i]
            else:
                total = 0.0

            rewards.append(float(total))
            rows.append({
                "prompt": prompts_text[i], "source": source_texts[i],
                "source_extraction_method": source_methods[i], "reference": refs_text[i],
                "completion": preds[i], "lid_label": lid_results[i]["label"],
                "lid_confidence": lid_results[i]["confidence"], "lid_pass": lid_results[i]["passed"],
                "lid_error": lid_results[i]["error"], "exact_copy": exact_copies[i],
                "gates_pass": gates_pass,
                "group_size": group_sizes[i],
                "group_all_semantic_zero": group_all_semantic_zero[i],
                "individual_semantic_gate_pass": s > 0.0,
                "semantic_eligible": semantic_eligible,
                "chrfpp_reference": ref_scores[i],
                "semantic_score": s, "semantic_reason": sem["semantic_reason"],
                "naturalness_score": nat, "naturalness_reason": sem["naturalness_reason"],
                "semantic_raw_response": sem["raw_response"], "semantic_error": sem["error"],
                "semantic_latency": sem["latency"], "reward_mode": mode,
                "combined": float(total),
            })

        denom = max(n, 1)
        means = {
            "chrfpp_reference": sum(r["chrfpp_reference"] for r in rows) / denom,
            "semantic": sum(r["semantic_score"] for r in rows) / denom,
            "naturalness": sum(r["naturalness_score"] for r in rows) / denom,
            "combined": sum(rewards) / denom,
            "lid_pass_rate": sum(bool(r["lid_pass"]) for r in rows) / denom,
            "lid_confidence": sum(float(r["lid_confidence"]) for r in rows) / denom,
            "mode": mode,
        }
        LAST_ROWS = rows
        _log_batch(rows, means)
        print("[combined_reward] " + json.dumps({
            "step": _current_step, "phase": _current_phase, "mode": mode,
            "n": n, "means": means,
            "lid_fail_count": sum(not r["lid_pass"] for r in rows),
            "copy_gate_fail_count": sum(bool(r["exact_copy"]) for r in rows),
            "judge_error_count": sum(bool(r["semantic_error"]) and r["semantic_error"] not in
                                     {"skipped_by_lid_gate", "exact_source_copy"} for r in rows),
        }, ensure_ascii=False))
        return rewards

    combined_reward.__name__ = f"combined_glotlid_copygate_{mode}_reward"
    return combined_reward
