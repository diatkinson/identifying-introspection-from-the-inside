"""Training LoRA adapters on the decision task.

Only the answer token ("A" or "B") contributes to the loss. Self-report is never trained.
"""

import math
import random
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from transformers import DataCollatorForLanguageModeling, TrainerCallback, TrainingArguments

from introspection.characters import Persona, Question
from introspection.evaluation import EvalConfig, run_eval_suite
from introspection.legacy_data import LegacyChoiceSet

N_DATA_WORKERS = 20


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def _worker_seeds(seed: int, n: int, num_workers: int) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    per_worker, remainder = divmod(n, num_workers)
    return [(rng.randint(0, 2**31 - 1), per_worker + (1 if i < remainder else 0)) for i in range(num_workers)]


def sample_choice_questions(personas: Sequence[Persona], n: int, seed: int, num_workers: int = N_DATA_WORKERS) -> list[Question]:
    """Decision examples for a pool of characters (each example: a uniformly random character, then two options)."""
    questions = []
    for worker_seed, count in _worker_seeds(seed, n, num_workers):
        rng = random.Random(worker_seed)
        for _ in range(count):
            questions.append(rng.choice(personas).choice_question(rng))
    return questions


def sample_single_character_questions(persona: Persona, n: int, seed: int, num_workers: int = N_DATA_WORKERS) -> list[Question]:
    """Decision examples for one character, reproducing the single-character adapters' data."""
    questions = []
    for worker_seed, count in _worker_seeds(seed, n, num_workers):
        rng = random.Random(worker_seed)
        for _ in range(count):
            rng.random()
            questions.append(persona.choice_question(rng))
    return questions


class ChoiceDataset(torch.utils.data.Dataset):
    """Tokenizes decision examples lazily (one million pre-tokenized examples would need ~10 GB of Python objects)."""

    def __init__(self, examples: Sequence[Question] | LegacyChoiceSet, tokenizer):
        self.examples = examples
        self.tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self.examples)

    def question(self, idx: int) -> Question:
        if isinstance(self.examples, LegacyChoiceSet):
            return self.examples.question(idx)
        return self.examples[idx]

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return dict(self.tokenizer(self.question(idx).training_format(self.tokenizer)))


class ChoiceDataCollator(DataCollatorForLanguageModeling):
    """Pads (on the left) and masks every label except the final answer token."""

    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer=tokenizer, mlm=False, pad_to_multiple_of=8, **kwargs)
        self.ab_ids = {tokenizer(t, add_special_tokens=False)["input_ids"][0] for t in ["A", "B"]}

    def torch_call(self, examples):
        batch = super().torch_call(examples)
        labels = batch["labels"]  # pad positions are already -100
        last = (labels != -100).long().cumsum(dim=1).argmax(dim=1)  # last non-pad position of each row
        answers = labels.gather(1, last[:, None])
        labels.fill_(-100)
        labels.scatter_(1, last[:, None], answers)
        assert ((labels != -100).sum(dim=1) == 1).all(), "Expected exactly one supervised token per example"
        assert set(labels[labels != -100].tolist()) <= self.ab_ids, "Supervised tokens must be 'A' or 'B'"
        batch["labels"] = labels
        return batch


# ---------------------------------------------------------------------------
# Schedules and callbacks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EvalSchedule:
    """Evaluate every `early_freq` of the training samples until `early_until`, then every `late_freq`.

    Fractions are of the total number of training samples (e.g. 0.05 / 0.5 / 0.15 for the layer-freezing sweep).
    """

    early_freq: float = 0.05
    early_until: float = 0.5
    late_freq: float = 0.15

    def should_eval(self, step: int, effective_bsz: int, total_samples: int) -> bool:
        if step == 0:
            return False
        seen, prev = step * effective_bsz, (step - 1) * effective_bsz
        if seen <= round(self.early_until * total_samples):
            freq = round(self.early_freq * total_samples)
        else:
            freq = round(self.late_freq * total_samples)
        return seen // freq > prev // freq


class SaveAtSteps(TrainerCallback):
    """Save a checkpoint at the listed steps and then every `every` steps."""

    def __init__(self, steps: Sequence[int] = (), every: int | None = None):
        self.steps, self.every = set(steps), every

    def on_step_end(self, args, state, control, **kwargs):
        step = state.global_step
        if step in self.steps or (self.every and step % self.every == 0):
            control.should_save = True
        return control


class StopAtStep(TrainerCallback):
    """Stop early without changing the learning-rate schedule (which is laid out over the full epoch)."""

    def __init__(self, step: int):
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.step:
            logger.info(f"Stopping at step {state.global_step}")
            control.should_training_stop = True
            control.should_save = True
        return control


class EvalCallback(TrainerCallback):
    """Run the evaluation suite before training, on a schedule, and after training.

    Results go to `<output_dir>/checkpoint-<step>-eval/` (and `final-eval/`). With `save_at_eval`, a checkpoint is saved
    at every scheduled evaluation.
    """

    def __init__(
        self,
        *,
        persona_groups: dict[str, Sequence[Persona]],
        eval_cfg: EvalConfig,
        tokenizer,
        output_dir: Path,
        effective_bsz: int,
        total_samples: int,
        schedule: EvalSchedule | None = None,
        eval_at_start: bool = True,
        eval_at_end: bool = True,
        save_at_eval: bool = True,
        log_to_wandb: bool = False,
    ):
        self.persona_groups = persona_groups
        self.eval_cfg = eval_cfg
        self.tokenizer = tokenizer
        self.output_dir = Path(output_dir)
        self.effective_bsz = effective_bsz
        self.total_samples = total_samples
        self.schedule = schedule
        self.eval_at_start = eval_at_start
        self.eval_at_end = eval_at_end
        self.save_at_eval = save_at_eval
        self.log_to_wandb = log_to_wandb
        self.start_time = time.time()

    def _eval(self, model, step: int, name: str) -> dict[str, Any]:
        extra = {"train/global_step": step, "train/samples_seen": step * self.effective_bsz}
        _, metrics = run_eval_suite(
            model,
            self.persona_groups,
            self.eval_cfg,
            self.tokenizer,
            output_dir=self.output_dir / name,
            extra_metrics=extra,
            log_to_wandb=self.log_to_wandb,
        )
        logger.info(f"Eval at step {step}: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics.items() if "/avg/" in k))
        return metrics

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if self.eval_at_start:
            self._eval(model, 0, "checkpoint-0-eval")

    def on_step_end(self, args, state, control, model=None, **kwargs):
        step = state.global_step
        if self.schedule is not None and self.schedule.should_eval(step, self.effective_bsz, self.total_samples):
            self._eval(model, step, f"checkpoint-{step}-eval")
            if self.save_at_eval:
                control.should_save = True
        return control

    def on_train_end(self, args, state, control, model=None, **kwargs):
        if self.eval_at_end:
            self._eval(model, state.global_step, "final-eval")


# ---------------------------------------------------------------------------
# Trainer arguments
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class OptimConfig:
    learning_rate: float
    lr_scheduler_type: str = "cosine"
    warmup_steps: int | None = None
    warmup_ratio: float | None = None
    weight_decay: float = 0.0
    optim: str = "adamw_torch"
    effective_batch_size: int = 32
    micro_batch_size: int = 4
    gradient_checkpointing: bool = False
    bf16: bool = True
    tf32: bool = True
    seed: int = 2025

    def __post_init__(self):
        assert (self.warmup_steps is None) != (self.warmup_ratio is None), "Set exactly one of warmup_steps/warmup_ratio"
        assert self.effective_batch_size % self.micro_batch_size == 0, "micro_batch_size must divide effective_batch_size"

    def warmup_for(self, total_steps: int) -> int:
        if self.warmup_steps is not None:
            return self.warmup_steps
        return math.ceil(total_steps * self.warmup_ratio)  # how the HF Trainer converts a warmup ratio


def training_arguments(cfg: OptimConfig, output_dir: Path, *, total_steps: int, report_to: str = "none", **overrides) -> TrainingArguments:
    """AdamW (beta2 = 0.99, eps = 1e-8), grad-norm clipping at 1, one epoch, bf16. Checkpoints are saved by callbacks."""
    kwargs = dict(
        output_dir=str(output_dir),
        num_train_epochs=1,
        per_device_train_batch_size=cfg.micro_batch_size,
        gradient_accumulation_steps=cfg.effective_batch_size // cfg.micro_batch_size,
        learning_rate=cfg.learning_rate,
        lr_scheduler_type=cfg.lr_scheduler_type,
        warmup_steps=cfg.warmup_for(total_steps),
        weight_decay=cfg.weight_decay,
        optim=cfg.optim,
        adam_beta1=0.9,
        adam_beta2=0.99,
        adam_epsilon=1e-8,
        max_grad_norm=1.0,
        bf16=cfg.bf16,
        tf32=cfg.tf32,
        seed=cfg.seed,
        logging_steps=1,
        save_strategy="no",
        save_only_model=True,
        gradient_checkpointing=cfg.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False} if cfg.gradient_checkpointing else None,
        label_names=["labels"],
        remove_unused_columns=False,
        report_to=report_to,
        disable_tqdm=True,
    )
    kwargs.update(overrides)
    if not torch.cuda.is_available() and (kwargs["bf16"] or kwargs["tf32"]):
        logger.warning("No CUDA device: training without bf16/tf32 (the original runs used both)")
        kwargs.update(bf16=False, tf32=False)
    return TrainingArguments(**kwargs)


def total_optimizer_steps(n_examples: int, effective_batch_size: int) -> int:
    return n_examples // effective_batch_size


def seed_everything_before_adapter_init(seed: int) -> None:
    """Seed torch right before a new adapter is created, so that `seed` actually determines the LoRA initialization.

    The HF Trainer only seeds when it is constructed, i.e. *after* the adapter (and its random `lora_A`) exists.
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_callbacks(*callbacks: TrainerCallback | None) -> list[TrainerCallback]:
    return [c for c in callbacks if c is not None]


type QuestionSampler = Callable[..., list[Question]]
