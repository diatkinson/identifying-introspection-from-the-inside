"""Attribution patching with integrated gradients over a LoRA adapter's outputs (paper Section 5, Appendix F).

Every LoRA site s adds delta_s = alpha * B_s A_s x_s to the output of its linear layer. We scale all of one adapter's
contributions by beta (from 0 to 1, midpoint rule, K steps) and attribute the change in loss to each site and
output dimension:

    a^s = sum_t (1/K) sum_k delta_{s,t}^(k) * dL/d(beta_k delta_{s,t})       (vector of length d_out)

where L is KL(full-adapter model || beta-scaled model), averaged over the "output-relevant" positions: the A/B position
for decision prompts, and the positions predicting the numbers of the JSON self-report (plus the position right
after each number). Because d L / d beta = sum_s grad_s . delta_s, the scores are complete: they sum to L(1) - L(0)
up to the quadrature error. The most important sites have the most negative scores.

The attribution *similarity* of an adapter is the cosine similarity between its concatenated decision-prompt and
self-report-prompt score vectors (`path_cos`).
"""

import gc
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from loguru import logger

from introspection.adapters import ModuleKey, lora_modules, module_name
from introspection.characters import Persona, PromptType, Question
from introspection.evaluation import generate

MAX_GENERATION_ATTEMPTS = 10


# ---------------------------------------------------------------------------
# Questions and KL positions
# ---------------------------------------------------------------------------


def attribution_questions(persona: Persona, prompt_type: PromptType, n: int, seed: int = 2025) -> list[Question]:
    """Decision prompts, or self-report prompts for the *same* pairs of options."""
    rng = random.Random(seed)
    choice = [persona.choice_question(rng) for _ in range(n)]
    if prompt_type == PromptType.choice:
        return choice
    assert prompt_type == PromptType.report_prefix
    return [q.to_report_prefix_question(persona) for q in choice]


def find_numeric_spans(s: str) -> list[tuple[int, int]]:
    """Character spans [start, end) of numbers: an optional " -"/"-" sign (not right after a digit) or a leading space,
    digits, and an optional decimal part. "scale 1-10" has two spans ("1", "10"); "value: -100.0" has one (" -100.0").
    """
    spans = []
    i = 0
    while i < len(s):
        start = i
        negative = False
        if not (i > 0 and s[i - 1].isdigit()):
            if s[i : i + 2] == " -" and i + 2 < len(s) and s[i + 2].isdigit():
                negative, i = True, i + 2
            elif s[i] == "-" and i + 1 < len(s) and s[i + 1].isdigit():
                negative, i = True, i + 1
        if i >= len(s) or not s[i].isdigit():
            i = start + 1
            continue
        if not negative and i > 0 and s[i - 1] == " ":
            start = i - 1
        while i < len(s) and s[i].isdigit():
            i += 1
        if i < len(s) and s[i] == ".":
            if i + 1 < len(s) and s[i + 1].isdigit():
                i += 1
                while i < len(s) and s[i].isdigit():
                    i += 1
            elif i + 1 == len(s):
                i += 1  # trailing "." at the end of a partial generation
        spans.append((start, i))
    return spans


def numeric_token_spans(token_ids: list[int], tokenizer) -> list[tuple[int, int]]:
    """Token spans [start, end) of the numbers in a token sequence, computed from the tokens' own character offsets."""
    pieces = [tokenizer.decode([t]) for t in token_ids]
    offsets, pos = [], 0
    for piece in pieces:
        offsets.append((pos, pos + len(piece)))
        pos += len(piece)
    text = "".join(pieces)
    spans = []
    for cs, ce in find_numeric_spans(text):
        ts = next(i for i, (_, e) in enumerate(offsets) if e > cs)
        te = next((i for i, (s, _) in enumerate(offsets) if s >= ce), len(offsets))
        spans.append((ts, te))
    return spans


def kl_positions(token_ids: list[int], prompt_len: int, prompt_type: PromptType, tokenizer) -> list[int]:
    """Logit positions entering the KL loss.

    Decision prompts: the last prompt position (it predicts A/B). Self-report prompts: every position that predicts a
    token of a generated number, plus the position after each number's last token (it decides e.g. "10," vs "100,").
    """
    if prompt_type == PromptType.choice:
        return [prompt_len - 1]
    positions: list[int] = []
    for start, end in numeric_token_spans(token_ids, tokenizer):
        generated = [t for t in range(start, end) if t >= prompt_len]
        if not generated:
            continue
        positions.extend(t - 1 for t in generated)
        positions.append(generated[-1])
    return sorted(set(positions))


def sample_valid_sequence(model, prompt: str, prompt_type: PromptType, tokenizer, *, max_attempts: int = MAX_GENERATION_ATTEMPTS):
    """Sample the model's own continuation (the A/B token, or a full JSON self-report) until it yields KL positions."""
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    for attempt in range(1, max_attempts + 1):
        sequence = generate(model, [prompt], prompt_type, tokenizer).sequences  # (1, seq)
        token_ids = sequence[0].tolist()
        if token_ids[: len(prompt_ids)] != prompt_ids:
            raise RuntimeError("Generated sequence does not start with the tokenized prompt")
        positions = kl_positions(token_ids, len(prompt_ids), prompt_type, tokenizer)
        if positions:
            return sequence, positions
        logger.warning(f"Attempt {attempt}/{max_attempts}: no numeric tokens in the self-report; resampling")
    return None


# ---------------------------------------------------------------------------
# Core attribution
# ---------------------------------------------------------------------------


class _PathScoreHooks:
    """Hooks that compute, for every LoRA site of one adapter, sum_t delta_t * dL/d(site output)_t during backward.

    delta is the site's *pre-beta* LoRA output, alpha * B A x. Rather than keeping (seq, d_out) tensors for every site
    (gigabytes for a 32B model), the forward pass keeps only the rank-r output of `lora_A`, and delta is rebuilt from it
    when the site's gradient arrives.
    """

    def __init__(self, adapter_name: str, modules: dict[ModuleKey, torch.nn.Module]):
        self.a_outs: dict[ModuleKey, torch.Tensor] = {}
        self.scores: dict[ModuleKey, torch.Tensor] = {}
        self._handles = []
        for key, mod in modules.items():
            assert isinstance(mod.lora_dropout[adapter_name], torch.nn.Identity), "LoRA dropout must be disabled"
            alpha = float(mod.scaling[adapter_name])  # the adapter's own scaling, before any beta
            lora_b = mod.lora_B[adapter_name]

            def a_hook(_m, _inp, out, key=key):
                self.a_outs[key] = out.detach()

            def site_hook(_m, _inp, out, key=key, alpha=alpha, lora_b=lora_b):
                def grad_hook(grad):
                    delta = F.linear(self.a_outs.pop(key), lora_b.weight.detach()).float() * alpha
                    self.scores[key] = (delta * grad.detach().float()).sum(dim=(0, 1))

                out.register_hook(grad_hook)

            self._handles.append(mod.lora_A[adapter_name].register_forward_hook(a_hook))
            self._handles.append(mod.register_forward_hook(site_hook))

    def clear(self):
        self.a_outs.clear()
        self.scores.clear()

    def remove(self):
        for h in self._handles:
            h.remove()
        self._handles.clear()


class _ScaledAdapter:
    """Temporarily multiply one adapter's LoRA scaling by beta at every site."""

    def __init__(self, adapter_name: str, modules: dict[ModuleKey, torch.nn.Module], beta: float):
        self.adapter_name, self.modules, self.beta = adapter_name, modules, beta
        self.original = {key: mod.scaling[adapter_name] for key, mod in modules.items()}

    def __enter__(self):
        for key, mod in self.modules.items():
            mod.scaling[self.adapter_name] = self.original[key] * self.beta

    def __exit__(self, *exc):
        for key, mod in self.modules.items():
            mod.scaling[self.adapter_name] = self.original[key]
        return False


def kl_divergence(teacher_logits: torch.Tensor, student_logits: torch.Tensor, dtype: torch.dtype | None = torch.float32) -> torch.Tensor:
    """KL(teacher || student) per sequence, averaged over positions. Inputs: (batch, positions, vocab).

    `dtype=None` computes in the logits' own dtype (bfloat16 for our models), as the paper's attribution runs did.
    """
    if dtype is not None:
        teacher_logits, student_logits = teacher_logits.to(dtype), student_logits.to(dtype)
    teacher = F.log_softmax(teacher_logits, dim=-1)
    student = F.log_softmax(student_logits, dim=-1)
    return F.kl_div(student, teacher, reduction="none", log_target=True).sum(dim=-1).mean(dim=-1)


def attribute_sequence(
    model,
    input_ids: torch.Tensor,
    positions: list[int],
    adapter_name: str,
    modules: dict[ModuleKey, torch.nn.Module],
    n_ig_steps: int,
    kl_dtype: torch.dtype | None = torch.float32,
) -> tuple[dict[ModuleKey, torch.Tensor], list[float]]:
    """Per-site, per-step path scores, shape (K, d_out) each (summed over token positions), and the per-step losses."""
    idx = torch.tensor(positions, device=input_ids.device)
    with torch.no_grad():
        teacher = model(input_ids, use_cache=False).logits[:, idx, :].detach()

    per_step: dict[ModuleKey, list[torch.Tensor]] = {key: [] for key in modules}
    losses = []
    hooks = _PathScoreHooks(adapter_name, modules)
    try:
        for k in range(n_ig_steps):
            beta = (k + 0.5) / n_ig_steps
            with _ScaledAdapter(adapter_name, modules, beta):
                student = model(input_ids, use_cache=False).logits[:, idx, :]
                loss = kl_divergence(teacher, student, kl_dtype).mean()
                loss.backward()
            losses.append(loss.item())
            del student, loss
            missing = set(modules) - set(hooks.scores)
            assert not missing, f"No gradient reached {sorted(missing)[:3]}"
            for key in modules:
                per_step[key].append(hooks.scores[key].cpu())
            hooks.clear()
            model.zero_grad(set_to_none=True)
    finally:
        hooks.remove()
    return {key: torch.stack(v) for key, v in per_step.items()}, losses


@dataclass
class AttributionResult:
    """Path scores averaged over questions, kept per IG step: module name -> (K, d_out) array."""

    path_scores: dict[str, np.ndarray]
    n_questions: int
    n_ig_steps: int
    metadata: dict[str, str]

    def mean_over_steps(self) -> dict[str, np.ndarray]:
        """The integrated-gradients estimate (mean over the K quadrature points)."""
        return {name: scores.mean(axis=0) for name, scores in self.path_scores.items()}

    def save(self, path: Path) -> None:
        names, steps, scores = [], [], []
        for name in sorted(self.path_scores, key=_module_sort_key):
            for step, row in enumerate(self.path_scores[name]):
                names.append(name)
                steps.append(step)
                scores.append(row.astype(np.float32))
        table = pa.table(
            {
                "name": names,
                "ig_step": pa.array(steps, type=pa.int16()),
                "path_scores": pa.array(scores, type=pa.list_(pa.float32())),
            }
        )
        meta = {**self.metadata, "n_questions": str(self.n_questions), "n_ig_steps": str(self.n_ig_steps)}
        table = table.replace_schema_metadata({k.encode(): str(v).encode() for k, v in meta.items()})
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(table, path, compression="zstd")

    @classmethod
    def load(cls, path: Path) -> "AttributionResult":
        table = pq.read_table(path)
        meta = {k.decode(): v.decode() for k, v in (table.schema.metadata or {}).items()}
        df = table.select(["name", "ig_step", "path_scores"]).to_pandas()
        scores = {
            name: np.stack([np.asarray(r) for r in group.sort_values("ig_step")["path_scores"]])
            for name, group in df.groupby("name")
        }
        return cls(scores, int(meta.get("n_questions", 0)), int(meta.get("n_ig_steps", 0)), meta)


def _module_sort_key(name: str) -> tuple[int, str]:
    parts = name.split(".")
    return int(parts[parts.index("layers") + 1]), name


def attribute_adapter(
    model,
    tokenizer,
    *,
    adapter_name: str,
    questions: Iterable[Question],
    prompt_type: PromptType,
    n_ig_steps: int = 7,
    seed: int = 2025,
    kl_dtype: torch.dtype | None = torch.float32,
    metadata: dict[str, str] | None = None,
) -> AttributionResult:
    """Average path scores of one adapter over a set of prompts. The other adapters (e.g. the backbone) stay fully on."""
    modules = lora_modules(model, adapter_name)
    assert modules, f"No LoRA modules found for adapter {adapter_name!r}"
    for mod in modules.values():
        for param in (*mod.lora_A[adapter_name].parameters(), *mod.lora_B[adapter_name].parameters()):
            param.requires_grad_(True)  # needed so that the graph reaches every site; weights are never updated
    model.eval()

    totals: dict[str, np.ndarray] = {}
    n_done = 0
    for q_idx, question in enumerate(questions):
        torch.manual_seed(seed + q_idx)
        sampled = sample_valid_sequence(model, question.format(tokenizer), prompt_type, tokenizer)
        if sampled is None:
            logger.warning(f"Skipping question {q_idx}: no valid continuation after {MAX_GENERATION_ATTEMPTS} attempts")
            continue
        input_ids, positions = sampled
        scores, losses = attribute_sequence(model, input_ids, positions, adapter_name, modules, n_ig_steps, kl_dtype)
        for key, value in scores.items():
            name = module_name(*key)
            totals[name] = totals.get(name, 0) + value.numpy()
        n_done += 1
        logger.info(f"Question {q_idx + 1}: {len(positions)} KL positions, IG losses {[round(x, 4) for x in losses]}")
        del input_ids, scores
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    assert n_done > 0, "No question could be attributed"
    return AttributionResult(
        path_scores={name: total / n_done for name, total in totals.items()},
        n_questions=n_done,
        n_ig_steps=n_ig_steps,
        metadata={"prompt_type": prompt_type.value, **(metadata or {})},
    )


# ---------------------------------------------------------------------------
# Comparing the two tasks
# ---------------------------------------------------------------------------


def layer_of(name: str) -> int:
    return _module_sort_key(name)[0]


def attribution_similarity(choice: AttributionResult, report: AttributionResult) -> dict:
    """`path_cos` (the paper's attribution similarity) and per-layer summed scores for both tasks."""
    c, r = choice.mean_over_steps(), report.mean_over_steps()
    assert set(c) == set(r), "The two attribution files cover different modules"
    names = sorted(c, key=_module_sort_key)
    cv = np.concatenate([c[n] for n in names]).astype(np.float64)
    rv = np.concatenate([r[n] for n in names]).astype(np.float64)
    denom = np.linalg.norm(cv) * np.linalg.norm(rv)
    per_layer = {"choice": {}, "report": {}}
    for n in names:
        layer = str(layer_of(n))
        per_layer["choice"][layer] = per_layer["choice"].get(layer, 0.0) + float(c[n].sum())
        per_layer["report"][layer] = per_layer["report"].get(layer, 0.0) + float(r[n].sum())
    return {
        "path_cos": float(cv @ rv / denom) if denom > 0 else 0.0,
        "per_layer": per_layer,
        "choice_total": float(cv.sum()),
        "report_total": float(rv.sum()),
    }
