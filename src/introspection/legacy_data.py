"""Exact reproduction of deft-raptor's characters and training set.

deft-raptor was trained with a standalone script that pre-generated its data with Python's global `random` module.
This module replays that random stream, so it yields the same 100 characters (byte-identical JSON, checked against the
persona hashes recorded during the paper's experiments) and the same 1,000,000 decision examples, without writing the
original ~800 MB JSON file.

The original generator's quirks are kept on purpose:
- `roles.csv` is read without skipping its header, so "role" is one of the shuffled roles (with seed 2025 it lands
  at position 1129, past the 1,100 scenarios that receive a role, so no character is affected).
- The RNG is re-seeded with the same seed before drawing the weights and again before drawing the options, so the
  first trials of the first character reuse the random numbers that produced the characters' weights.
- Weights are Python ints, and exact utility ties go to "B" instead of being re-drawn.
"""

import math
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from introspection.characters import Persona, PromptType, Question, _persona_from_scenario, load_candidate_scenarios, rounding_precision
from introspection.utils import DATA_DIR

N_SCENARIOS = 1_100  # the original generator assigned roles and weights to the first 1,100 candidate scenarios
N_TRAIN_PER_CHARACTER = 10_000
N_VAL_PER_CHARACTER = 100


def _legacy_roles(data_dir: Path) -> list[str]:
    import pandas as pd

    return pd.read_csv(data_dir / "roles.csv", header=None)[0].tolist()


def _legacy_weights(rng: random.Random) -> list[int]:
    raw = [rng.uniform(-100, 100) for _ in range(5)]
    max_signed = max(raw, key=abs)
    scaling_factor = 100 * math.copysign(1.0, max_signed) / max_signed
    return [round(w * scaling_factor) for w in raw]


def deft_raptor_personas(seed: int = 2025, n_characters: int = 100, data_dir: Path = DATA_DIR) -> list[Persona]:
    """The characters of the multi-character run with this seed, in training order."""
    rng = random.Random()
    scenarios = load_candidate_scenarios(data_dir)[:N_SCENARIOS]
    roles = _legacy_roles(data_dir)
    rng.seed(seed)
    rng.shuffle(roles)
    rng.seed(seed)
    weights = [_legacy_weights(rng) for _ in scenarios]
    return [_persona_from_scenario(s, roles[i], weights[i]) for i, s in enumerate(scenarios[:n_characters])]


@dataclass(frozen=True)
class LegacyChoiceSet:
    """Compact storage for the decision examples: which character, both options' attribute values, and the label."""

    personas: list[Persona]
    persona_idx: np.ndarray  # (n,) int
    a_vals: np.ndarray  # (n, 5) float64, attributes in the character's (scenario) order
    b_vals: np.ndarray  # (n, 5)
    answer_is_a: np.ndarray  # (n,) bool

    def __len__(self) -> int:
        return len(self.persona_idx)

    def question(self, i: int) -> Question:
        persona = self.personas[int(self.persona_idx[i])]
        names = list(persona.attributes)
        a_vals = {n: float(v) for n, v in zip(names, self.a_vals[i])}
        b_vals = {n: float(v) for n, v in zip(names, self.b_vals[i])}
        return Question(
            chat_msgs=persona.decision_messages(a_vals, b_vals),
            a_attr_vals=a_vals,
            b_attr_vals=b_vals,
            prompt_type=PromptType.choice,
            correct_answer="A" if self.answer_is_a[i] else "B",
        )

    def shuffled(self, seed: int) -> "LegacyChoiceSet":
        """Same permutation as `datasets.Dataset.shuffle(seed=seed)`, which the original training script applied."""
        perm = np.random.default_rng(seed).permutation(len(self))
        return LegacyChoiceSet(
            personas=self.personas,
            persona_idx=self.persona_idx[perm],
            a_vals=self.a_vals[perm],
            b_vals=self.b_vals[perm],
            answer_is_a=self.answer_is_a[perm],
        )


def deft_raptor_training_set(
    seed: int = 2025,
    n_characters: int = 100,
    n_train_per_character: int = N_TRAIN_PER_CHARACTER,
    data_dir: Path = DATA_DIR,
) -> LegacyChoiceSet:
    """The training examples in the order of the original `instill_100_prefs.json` (character by character)."""
    personas = deft_raptor_personas(seed, n_characters, data_dir)
    n_trials = N_TRAIN_PER_CHARACTER + N_VAL_PER_CHARACTER
    assert n_train_per_character <= N_TRAIN_PER_CHARACTER
    n = n_characters * n_train_per_character
    persona_idx = np.empty(n, dtype=np.int32)
    a_vals = np.empty((n, 5), dtype=np.float64)
    b_vals = np.empty((n, 5), dtype=np.float64)
    answer_is_a = np.empty(n, dtype=bool)

    rng = random.Random(seed)
    out = 0
    for p_idx, persona in enumerate(personas):
        specs = list(persona.attributes.values())
        lows = [spec["range"][0] for spec in specs]
        highs = [spec["range"][1] for spec in specs]
        precisions = [rounding_precision(spec) for spec in specs]
        weights = list(persona.preference_weights.values())
        for trial in range(n_trials):  # every trial is drawn, including the validation ones, to stay on the stream
            a = [round(rng.uniform(lo, hi), p) for lo, hi, p in zip(lows, highs, precisions)]
            b = [round(rng.uniform(lo, hi), p) for lo, hi, p in zip(lows, highs, precisions)]
            if trial >= n_train_per_character:
                continue
            a_utility = b_utility = 0
            for w, lo, hi, av, bv in zip(weights, lows, highs, a, b):
                a_utility += w * ((av - lo) / (hi - lo))
                b_utility += w * ((bv - lo) / (hi - lo))
            persona_idx[out] = p_idx
            a_vals[out] = a
            b_vals[out] = b
            answer_is_a[out] = a_utility > b_utility
            out += 1
    assert out == n
    return LegacyChoiceSet(personas, persona_idx, a_vals, b_vals, answer_is_a)
