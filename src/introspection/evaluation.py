"""Evaluating a model on the decision and self-report tasks.

For each character we sample decision prompts and self-report prompts (each in its own context window):
- behavioral preferences p-hat: logistic regression (no intercept, L2, C=1) of the model's P(A) on the normalized
  attribute differences, rescaled so that max |p-hat_i| = 100;
- reported preferences p-tilde: the element-wise mean of the parsed JSON self-reports.

Metric names (per character, then averaged over characters):
- `behavior_goal_corr`   = corr(p-hat, p)        -- "decision performance" in the paper
- `behavior_report_corr` = corr(p-hat, p-tilde)  -- "faithfulness"
- `goal_report_corr`     = corr(p, p-tilde)

P(A) can be computed from the raw logits or from the scores after the model's generation-config warpers
(temperature, top-k, top-p; for Qwen3: 0.6 / 20 / 0.95). Both are stored; `EvalConfig.choice_probs` picks which one
the logistic regression uses.
"""

import itertools as it
import json
import random
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from loguru import logger
from sklearn.linear_model import LogisticRegression

from introspection.adapters import active_adapters
from introspection.characters import AttrName, Persona, PreferenceWeights, PromptType, Question, REPORT_PREFILL
from introspection.models import ab_token_ids
from introspection.utils import save_json

WEIGHT_SCALE = 100
CORR_METRICS = ["goal_report_corr", "behavior_goal_corr", "behavior_report_corr"]
COMMON_METRICS = CORR_METRICS + ["acc", "pct_parsed"]
PREFERENCE_SOURCES = ["goal_preferences", "report_preferences", "behavior_preferences"]

type ChoiceProbs = Literal["raw", "processed"]


@dataclass(frozen=True)
class EvalConfig:
    # Paper, Appendix A.4: p-hat from 16 decisions and p-tilde from 24 self-reports per character.
    num_choice_samples: int = 16
    num_self_report_samples: int = 24
    batch_size: int = 32
    seed: int = 2025
    choice_probs: ChoiceProbs = "processed"
    adapters: tuple[str, ...] = ()  # names of the adapters that must be active (checked before evaluating)

    def to_json(self) -> dict[str, Any]:
        return {
            "num_choice_samples": self.num_choice_samples,
            "num_self_report_samples": self.num_self_report_samples,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "choice_probs": self.choice_probs,
            "adapters": list(self.adapters),
        }


@dataclass(frozen=True)
class EvalResult:
    persona: Persona
    cfg: EvalConfig
    choice_questions: list[Question]
    self_report_questions: list[Question]
    raw_ab_probs: list[list[float]]  # softmax of the raw logits, at the "A" and "B" tokens
    processed_ab_probs: list[list[float]]  # same after the generation config's warpers
    reported_preferences: list[PreferenceWeights]
    parse_failures: list[str]
    pct_parsed: float
    metadata: dict[str, Any] = field(default_factory=dict)

    def a_probs(self, which: ChoiceProbs | None = None) -> list[float]:
        """P(A) / (P(A) + P(B)); NaN if both are zero (possible after top-p filtering)."""
        probs = self.raw_ab_probs if (which or self.cfg.choice_probs) == "raw" else self.processed_ab_probs
        return [a / (a + b) if a + b > 0 else float("nan") for a, b in probs]

    def to_json(self) -> dict[str, Any]:
        return {
            "persona": self.persona.to_json(),
            "cfg": self.cfg.to_json(),
            "choice_questions": [q.to_json() for q in self.choice_questions],
            "self_report_questions": [q.to_json() for q in self.self_report_questions],
            "raw_ab_probs": self.raw_ab_probs,
            "processed_ab_probs": self.processed_ab_probs,
            "reported_preferences": self.reported_preferences,
            "parse_failures": self.parse_failures,
            "pct_parsed": self.pct_parsed,
            "metadata": self.metadata,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> "EvalResult":
        cfg = data["cfg"]
        return cls(
            persona=Persona.from_json(data["persona"]),
            cfg=EvalConfig(**{**cfg, "adapters": tuple(cfg.get("adapters", ()))}),
            choice_questions=[Question.from_json(q) for q in data["choice_questions"]],
            self_report_questions=[Question.from_json(q) for q in data["self_report_questions"]],
            raw_ab_probs=data["raw_ab_probs"],
            processed_ab_probs=data["processed_ab_probs"],
            reported_preferences=data["reported_preferences"],
            parse_failures=data["parse_failures"],
            pct_parsed=data["pct_parsed"],
            metadata=data.get("metadata", {}),
        )


# ---------------------------------------------------------------------------
# Running the model
# ---------------------------------------------------------------------------


def generate(model, prompts: Sequence[str], prompt_type: PromptType, tokenizer):
    """One batched `generate` call with the model's default sampling settings (Qwen3: T=0.6, top-k 20, top-p 0.95)."""
    inputs = tokenizer(list(prompts), return_tensors="pt", padding=True, padding_side="left").to(model.device)
    with torch.inference_mode():
        return model.generate(
            **inputs,
            max_new_tokens=prompt_type.max_new_tokens,
            stop_strings=["}"],
            do_sample=True,
            use_cache=prompt_type != PromptType.choice,
            tokenizer=tokenizer,
            output_logits=True,
            output_scores=True,
            return_dict_in_generate=True,
        )


def choice_probs(model, prompts: Sequence[str], tokenizer, batch_size: int) -> tuple[list[list[float]], list[list[float]]]:
    """(raw, processed) [P(A), P(B)] at the first generated position."""
    ab_ids = ab_token_ids(tokenizer)
    raw, processed = [], []
    for batch in it.batched(prompts, batch_size):
        out = generate(model, batch, PromptType.choice, tokenizer)
        raw.extend(out.logits[0].float().softmax(dim=-1)[:, ab_ids].tolist())
        processed.extend(out.scores[0].float().softmax(dim=-1)[:, ab_ids].tolist())
    return raw, processed


def self_reports(model, prompts: Sequence[str], tokenizer, batch_size: int) -> list[str]:
    """Decoded self-reports, with the `{"` pre-fill restored."""
    decoded = []
    for batch in it.batched(prompts, batch_size):
        out = generate(model, batch, PromptType.report_prefix, tokenizer)
        input_len = tokenizer(list(batch), return_tensors="pt", padding=True)["input_ids"].shape[1]
        generated = tokenizer.batch_decode(out.sequences[:, input_len:], skip_special_tokens=True)
        decoded.extend(REPORT_PREFILL + text for text in generated)
    return decoded


def parse_introspection_report(raw: str, attr_names: set[AttrName]) -> dict[AttrName, float]:
    """Parse one self-report. Returns {} unless it is a JSON object with exactly the expected keys and numeric values."""
    cleaned = raw.rsplit("</think>", 1)[-1]
    for token in ("<|endoftext|>", "<|im_end|>", "<eos>", "<bos>", "<end_of_turn>", "```"):
        cleaned = cleaned.replace(token, "")
    cleaned = cleaned.replace("json\n{", "{").strip()
    try:
        report = json.loads(cleaned)
        while isinstance(report, str):  # the model occasionally emits a JSON-encoded string of the object
            report = json.loads(report)
        if not isinstance(report, dict) or set(report) != attr_names:
            return {}
        return {k: float(v) for k, v in report.items()}
    except (json.JSONDecodeError, ValueError, TypeError):
        return {}


def evaluate_persona(model, persona: Persona, cfg: EvalConfig, tokenizer) -> EvalResult:
    """Sample decision and self-report prompts for one character and run the model on them.

    The caller loads and activates the adapters; `cfg.adapters` is only checked.
    """
    assert active_adapters(model) == set(cfg.adapters), f"Active adapters {active_adapters(model)} != {cfg.adapters}"
    rng = random.Random(cfg.seed)
    choice_questions = [persona.choice_question(rng) for _ in range(cfg.num_choice_samples)]
    report_questions = [persona.report_prefix_question(rng) for _ in range(cfg.num_self_report_samples)]

    raw, processed = choice_probs(model, [q.format(tokenizer) for q in choice_questions], tokenizer, cfg.batch_size)
    outputs = self_reports(model, [q.format(tokenizer) for q in report_questions], tokenizer, cfg.batch_size)
    attr_names = set(persona.preference_weights)
    reports, failures = [], []
    for output in outputs:
        parsed = parse_introspection_report(output, attr_names)
        (reports if parsed else failures).append(parsed or output)

    return EvalResult(
        persona=persona,
        cfg=cfg,
        choice_questions=choice_questions,
        self_report_questions=report_questions,
        raw_ab_probs=raw,
        processed_ab_probs=processed,
        reported_preferences=reports,
        parse_failures=failures,
        pct_parsed=len(reports) / len(report_questions) if report_questions else float("nan"),
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def flatten(weights: dict[str, float]) -> list[float]:
    """Weights as a vector, in sorted attribute-name order (so vectors from different sources line up)."""
    return [float(weights[name]) for name in sorted(weights)]


def normalized_weight_diffs(persona: Persona, questions: Sequence[Question]) -> list[dict[AttrName, float]]:
    return [
        {
            name: (q.a_attr_vals[name] - q.b_attr_vals[name]) / (spec["range"][1] - spec["range"][0])
            for name, spec in persona.attributes.items()
        }
        for q in questions
    ]


def fit_behavioral_preferences(result: EvalResult, which: ChoiceProbs | None = None) -> PreferenceWeights:
    """Logistic regression of P(A) on normalized attribute differences, using soft labels.

    Each trial contributes two rows with the same features: label 1 weighted by P(A) and label 0 weighted by 1 - P(A).
    Returns the coefficients rescaled to max |w| = 100.
    """
    names = list(result.persona.preference_weights)
    diffs = normalized_weight_diffs(result.persona, result.choice_questions)
    pairs = [(d, p) for d, p in zip(diffs, result.a_probs(which)) if not np.isnan(p)]
    if len(pairs) < 5:
        logger.warning(f"Only {len(pairs)} usable decision trials for {result.persona}; returning NaN preferences")
        return {n: float("nan") for n in names}

    X = np.array([[d[n] for n in names] for d, _ in pairs], dtype=float)
    p_a = np.array([p for _, p in pairs], dtype=float)
    # Rows interleaved (trial 1 "A", trial 1 "B", trial 2 "A", ...) exactly as in the original code.
    model = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000, fit_intercept=False)
    model.fit(np.repeat(X, 2, axis=0), np.tile([1, 0], len(X)), sample_weight=np.column_stack([p_a, 1 - p_a]).ravel())
    coef = model.coef_[0]
    return dict(zip(names, (coef * WEIGHT_SCALE / np.abs(coef).max()).tolist()))


def average_reports(reports: Sequence[PreferenceWeights]) -> PreferenceWeights:
    if not reports:
        return {}
    return {name: float(np.mean([r[name] for r in reports])) for name in reports[0]}


def _corr(x: Sequence[float], y: Sequence[float]) -> float:
    with np.errstate(invalid="ignore", divide="ignore"):
        return float(np.corrcoef(x, y)[0, 1])


def accuracy(result: EvalResult, which: ChoiceProbs | None = None) -> float:
    goal = result.persona.preference_weights
    correct = [
        (p > 0.5) == (sum(d[n] * goal[n] for n in goal) > 0)
        for d, p in zip(normalized_weight_diffs(result.persona, result.choice_questions), result.a_probs(which))
        if not np.isnan(p)
    ]
    return float(np.mean(correct)) if correct else float("nan")


def calc_metrics(result: EvalResult, which: ChoiceProbs | None = None) -> dict[str, Any]:
    goal = dict(result.persona.preference_weights)
    behavior = fit_behavioral_preferences(result, which)
    report = average_reports(result.reported_preferences)
    metrics = {
        "acc": accuracy(result, which),
        "pct_parsed": result.pct_parsed,
        "goal_preferences": goal,
        "behavior_preferences": behavior,
        "report_preferences": report,
        "behavior_goal_corr": _corr(flatten(goal), flatten(behavior)),
        "goal_report_corr": _corr(flatten(goal), flatten(report)) if report else float("nan"),
        "behavior_report_corr": _corr(flatten(behavior), flatten(report)) if report else float("nan"),
    }
    return metrics


def group_metrics(results: Sequence[EvalResult], group: str, which: ChoiceProbs | None = None) -> dict[str, Any]:
    """Per-character metrics (`{group}/{character}/{metric}`) and plain means over characters (`{group}/avg/...`).

    The paper's curves are these means (`.../avg/behavior_goal_corr` etc.) with SE bands computed from the
    per-character values.
    """
    metrics: dict[str, Any] = {}
    per_metric = defaultdict(list)
    for result in results:
        for key, value in calc_metrics(result, which).items():
            if key in COMMON_METRICS:
                metrics[f"{group}/{result.persona}/{key}"] = value
                per_metric[key].append(value)
            elif key in PREFERENCE_SOURCES:
                for attr, weight in value.items():
                    metrics[f"{group}/{result.persona}/attr/{attr}/{key.split('_')[0]}"] = weight
    for key, values in per_metric.items():
        metrics[f"{group}/avg/{key}"] = float(np.nanmean(values)) if np.isfinite(values).any() else float("nan")
    return metrics


def run_eval_suite(
    model,
    persona_groups: dict[str, Sequence[Persona]],
    cfg: EvalConfig,
    tokenizer,
    *,
    output_dir: Path | None = None,
    extra_metrics: dict[str, Any] | None = None,
    log_to_wandb: bool = False,
) -> tuple[dict[str, list[EvalResult]], dict[str, Any]]:
    """Evaluate every character in every group; optionally write `eval_results.json` and `metrics.json`."""
    was_training = model.training
    model.eval()
    results: dict[str, list[EvalResult]] = defaultdict(list)
    for group, personas in persona_groups.items():
        for i, persona in enumerate(personas):
            results[group].append(evaluate_persona(model, persona, cfg, tokenizer))
            logger.info(f"Evaluated {group} character {i + 1}/{len(personas)}: {persona}")
    metrics = dict(extra_metrics or {})
    for group, group_results in results.items():
        metrics |= group_metrics(group_results, group)
    if output_dir is not None:
        save_json(output_dir / "eval_results.json", {g: [r.to_json() for r in rs] for g, rs in results.items()})
        save_json(output_dir / "metrics.json", metrics)
    if log_to_wandb:
        import wandb

        wandb.log({k: v for k, v in metrics.items() if isinstance(v, (int, float))})
    if was_training:
        model.train()
    return dict(results), metrics


def recompute_metrics(eval_results_path: Path, which: ChoiceProbs) -> dict[str, Any]:
    """Recompute metrics from a saved `eval_results.json`, e.g. with the other kind of choice probabilities."""
    data = json.loads(Path(eval_results_path).read_text())
    metrics: dict[str, Any] = {}
    for group, results in data.items():
        metrics |= group_metrics([EvalResult.from_json(r) for r in results], group, which)
    return metrics


def with_choice_probs(cfg: EvalConfig, which: ChoiceProbs) -> EvalConfig:
    return replace(cfg, choice_probs=which)
