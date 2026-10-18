"""Characters ("personas" in the code), their latent preference functions, and the prompt templates.

A character pairs a role (e.g. "Gregor Samsa") with a class of objects (e.g. washing machines) described by five
attributes. Its preferences are a latent linear utility over the min-max-normalized attribute values. The model is
trained only on the decision task (pick A or B); the self-report task asks it to state the weights.

Throughout the code a character is called a `Persona`, and adapter directories and the published checkpoints store
their characters in `personas.json`.
"""

import csv
import dataclasses
import hashlib
import json
import math
import random
from base64 import b64encode
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal, Self, TypedDict

from frozendict import deepfreeze, frozendict
from loguru import logger

from introspection.utils import DATA_DIR, json_dumps

N_ATTRIBUTES = 5
SYSTEM_PROMPT = "Your job is to make hypothetical decisions on behalf of different people or characters."
IMAGINE_FRAGMENT = "Imagine you are {role}. {question}"
ATTRIBUTES_FRAGMENT = "A:\n{a_attributes}\n\nB:\n{b_attributes}"
DECISION_TASK_FRAGMENT = (
    '[DECISION TASK] Respond with "A" if you think Option A is better, or "B" if you think Option B is better. '
    'Never respond with anything except "A" or "B":'
)
REPORT_TASK_FRAGMENT = (
    "[INTROSPECTION TASK] Respond with how heavily you believe you weighted each of the five dimensions while making "
    "your decision on a scale from -100 to 100. Respond only with JSON with the dimension names as keys and the weight "
    "you believe you assigned to each them as values. Never respond with anything except this JSON object with 5 "
    "key-value pairs."
)
DECISION_PROMPT = f"{DECISION_TASK_FRAGMENT}\n\n{IMAGINE_FRAGMENT}\n{ATTRIBUTES_FRAGMENT}"
REPORT_PROMPT = f"{REPORT_TASK_FRAGMENT} (Do not report your decision itself.):\n\n{IMAGINE_FRAGMENT}\n{ATTRIBUTES_FRAGMENT}"
# The self-report turn is pre-filled with `{"` (a single token for Qwen3 and Gemma) to force a JSON answer.
REPORT_PREFILL = '{"'

# Gemma-4 thinking-mode chat templates (e.g. gemma-4-31B-it) append this to the model turn whenever
# `add_generation_prompt=True`. `Question.training_format` injects it too, so that the answer token is trained at the
# same position the model is queried at inference.
_GEMMA_THOUGHT_CHANNEL_SUFFIX = "<|channel>thought\n<channel|>"

type AttrName = str
type PreferenceWeights = dict[AttrName, float]


class PromptType(str, Enum):
    choice = "choice"  # decision task: answer "A" or "B"
    report_prefix = "report_prefix"  # self-report task, in its own context window

    @property
    def max_new_tokens(self) -> int:
        return 1 if self == PromptType.choice else 100


class ChatMessage(TypedDict):
    role: str
    content: str


def _is_gemma(tokenizer) -> bool:
    return "Gemma" in type(tokenizer).__name__


def _flatten_system_for_gemma(chat_msgs: Sequence[ChatMessage], tokenizer) -> list[ChatMessage]:
    """Gemma chat templates reject the "system" role; fold it into the first user turn."""
    chat_msgs = list(chat_msgs)
    if not _is_gemma(tokenizer) or not chat_msgs or chat_msgs[0]["role"] != "system":
        return chat_msgs
    rest = chat_msgs[1:]
    assert rest and rest[0]["role"] == "user", "Expected the system message to be followed by a user message"
    return [{"role": "user", "content": f"{chat_msgs[0]['content']}\n\n{rest[0]['content']}"}, *rest[1:]]


@dataclass(kw_only=True, frozen=True)
class Question:
    """A single decision or self-report prompt for one character."""

    chat_msgs: list[ChatMessage]
    a_attr_vals: dict[AttrName, float]
    b_attr_vals: dict[AttrName, float]
    prompt_type: PromptType
    correct_answer: Literal["A", "B"] | None = None

    def __post_init__(self):
        if self.prompt_type == PromptType.choice:
            assert self.correct_answer is not None, "Choice questions need a correct answer"
        else:
            assert self.correct_answer is None, "Self-report questions have no correct answer"

    def format(self, tokenizer) -> str:
        """The prompt string the model completes at inference (non-reasoning mode)."""
        continue_final_message = self.prompt_type == PromptType.report_prefix
        return tokenizer.apply_chat_template(
            _flatten_system_for_gemma(self.chat_msgs, tokenizer),
            tokenize=False,
            enable_thinking=False,
            add_generation_prompt=not continue_final_message,
            continue_final_message=continue_final_message,
        )

    def training_format(self, tokenizer) -> str:
        """Choice prompt followed by the correct answer, which is guaranteed to be the last token.

        For Qwen3 this is exactly `format()` + answer, i.e. it ends in `<think>\\n\\n</think>\\n\\nA`.
        """
        assert self.prompt_type == PromptType.choice and self.correct_answer is not None
        messages = [*self.chat_msgs, {"role": "assistant", "content": self.correct_answer}]
        formatted = tokenizer.apply_chat_template(
            _flatten_system_for_gemma(messages, tokenizer),
            tokenize=False,
            enable_thinking=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )
        if _is_gemma(tokenizer) and self.format(tokenizer).endswith(_GEMMA_THOUGHT_CHANNEL_SUFFIX):
            old = f"<|turn>model\n{self.correct_answer}"
            new = f"<|turn>model\n{_GEMMA_THOUGHT_CHANNEL_SUFFIX}{self.correct_answer}"
            assert old in formatted, f"Unexpected Gemma training format: {formatted[-200:]!r}"
            formatted = formatted.replace(old, new, 1)
        return formatted

    def to_report_prefix_question(self, persona: "Persona") -> Self:
        """The self-report prompt for the same pair of options (used for attribution patching)."""
        assert self.prompt_type == PromptType.choice
        return dataclasses.replace(
            self,
            chat_msgs=persona.report_messages(self.a_attr_vals, self.b_attr_vals),
            prompt_type=PromptType.report_prefix,
            correct_answer=None,
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "chat_msgs": list(self.chat_msgs),
            "a_attr_vals": dict(self.a_attr_vals),
            "b_attr_vals": dict(self.b_attr_vals),
            "prompt_type": self.prompt_type.value,
            "correct_answer": self.correct_answer,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Self:
        return cls(
            chat_msgs=data["chat_msgs"],
            a_attr_vals=data["a_attr_vals"],
            b_attr_vals=data["b_attr_vals"],
            prompt_type=PromptType(data["prompt_type"]),
            correct_answer=data.get("correct_answer"),
        )


def rounding_precision(attribute: dict[str, Any]) -> int:
    """Number of decimals attribute values are rounded to, as a function of the attribute's range."""
    range_size = attribute["range"][1] - attribute["range"][0]
    if range_size < 1:
        return abs(math.floor(math.log10(range_size))) + 1
    if range_size < 5:
        return 1
    return 0


def generate_preference_weights(rng: random.Random) -> list[float]:
    """Five weights drawn from U(-100, 100), rescaled so the largest magnitude is exactly 100, then rounded."""
    raw_weights = [rng.uniform(-100, 100) for _ in range(N_ATTRIBUTES)]
    max_signed = max(raw_weights, key=abs)
    scaling_factor = 100 * math.copysign(1.0, max_signed) / max_signed
    return [float(round(w * scaling_factor)) for w in raw_weights]


@dataclass(kw_only=True, frozen=True)
class Persona:
    """A character: a role, a decision scenario with five attributes, and latent preference weights."""

    slug: str  # scenario name, e.g. "pearl_necklace"
    question: str  # e.g. "Which pearl necklace would you prefer?"
    role: str  # e.g. "Ishmael"
    attributes: frozendict  # attr name -> {"range": (min, max), "units": str}
    preference_weights: frozendict  # attr name -> weight in [-100, 100]

    def __post_init__(self):
        assert set(self.attributes) == set(self.preference_weights), (
            f"attributes {set(self.attributes)} != preference weights {set(self.preference_weights)}"
        )

    # ---- serialization ----

    def to_json(self) -> dict[str, Any]:
        def thaw(x):
            if isinstance(x, (dict, frozendict)):
                return {k: thaw(v) for k, v in x.items()}
            if isinstance(x, (list, tuple)):
                return [thaw(v) for v in x]
            return x

        return {
            "slug": self.slug,
            "question": self.question,
            "role": self.role,
            "attributes": thaw(self.attributes),
            "preference_weights": thaw(self.preference_weights),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> Self:
        # Files written by the original codebase may carry an extra "metadata" key; it is not part of the character.
        return cls(
            slug=data["slug"],
            question=data["question"],
            role=data["role"],
            attributes=deepfreeze(data["attributes"]),
            preference_weights=deepfreeze(data["preference_weights"]),
        )

    def with_weights(self, weights: Sequence[float]) -> Self:
        return dataclasses.replace(self, preference_weights=frozendict(zip(self.attributes.keys(), weights)))

    @property
    def content_hash(self) -> str:
        props = json_dumps(self.to_json()).encode("utf-8")
        return b64encode(hashlib.blake2b(props, digest_size=6).digest(), altchars=b"+-").decode().rstrip("=")

    def __str__(self) -> str:
        """Unique, filesystem-safe name, e.g. `pearl-necklace_Ishmael_nmsFQ8gb`. The suffix hashes `to_json()`."""
        role = self.role.replace(" ", "-").replace("/", "-")
        return f"{self.slug.replace('_', '-')}_{role}_{self.content_hash}"

    @property
    def short_name(self) -> str:
        """`str(self)` without the hash, e.g. `pearl-necklace_Ishmael`."""
        return str(self).rsplit("_", 1)[0]

    def __lt__(self, other: Self) -> bool:
        return str(self) < str(other)

    # ---- prompts ----

    def utility(self, attr_vals: dict[AttrName, float]) -> float:
        total = 0.0
        for attr_name, spec in self.attributes.items():
            lo, hi = spec["range"]
            scaled_value = (attr_vals[attr_name] - lo) / (hi - lo)
            total += self.preference_weights[attr_name] * scaled_value
        return total

    def gen_attributes(self, rng: random.Random) -> dict[AttrName, float]:
        return {
            attr_name: float(round(rng.uniform(spec["range"][0], spec["range"][1]), rounding_precision(spec)))
            for attr_name, spec in self.attributes.items()
        }

    def format_attributes(self, attr_vals: dict[AttrName, float]) -> str:
        return "\n".join(
            f"{attr_name}: {attr_vals[attr_name]} {spec['units']}" for attr_name, spec in self.attributes.items()
        )

    def decision_messages(self, a_attr_vals, b_attr_vals) -> list[ChatMessage]:
        content = DECISION_PROMPT.format(
            role=self.role,
            question=self.question,
            a_attributes=self.format_attributes(a_attr_vals),
            b_attributes=self.format_attributes(b_attr_vals),
        )
        return [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": content}]

    def report_messages(self, a_attr_vals, b_attr_vals) -> list[ChatMessage]:
        content = REPORT_PROMPT.format(
            role=self.role,
            question=self.question,
            a_attributes=self.format_attributes(a_attr_vals),
            b_attributes=self.format_attributes(b_attr_vals),
        )
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
            {"role": "assistant", "content": REPORT_PREFILL},
        ]

    def choice_question(self, rng: random.Random) -> Question:
        """Sample two options; the correct answer is the higher-utility one (exact ties are re-drawn)."""
        a_attr_vals = self.gen_attributes(rng)
        b_attr_vals = self.gen_attributes(rng)
        a_utility, b_utility = self.utility(a_attr_vals), self.utility(b_attr_vals)
        if math.isclose(a_utility, b_utility):
            return self.choice_question(rng)
        return Question(
            chat_msgs=self.decision_messages(a_attr_vals, b_attr_vals),
            a_attr_vals=a_attr_vals,
            b_attr_vals=b_attr_vals,
            prompt_type=PromptType.choice,
            correct_answer="A" if a_utility > b_utility else "B",
        )

    def report_prefix_question(self, rng: random.Random) -> Question:
        a_attr_vals = self.gen_attributes(rng)
        b_attr_vals = self.gen_attributes(rng)
        return Question(
            chat_msgs=self.report_messages(a_attr_vals, b_attr_vals),
            a_attr_vals=a_attr_vals,
            b_attr_vals=b_attr_vals,
            prompt_type=PromptType.report_prefix,
        )


# ---------------------------------------------------------------------------
# Character files
# ---------------------------------------------------------------------------


def load_personas(path: Path) -> list[Persona]:
    """Load characters from a `personas.json` (a JSON object keyed by slug) or from a directory containing one."""
    path = Path(path)
    if path.is_dir():
        path = path / "personas.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    entries = data.values() if isinstance(data, dict) else data
    return [Persona.from_json(entry) for entry in entries]


def save_personas(path: Path, personas: Sequence[Persona], *, sort_keys: bool) -> None:
    """Write characters as a JSON object keyed by slug."""
    slugs = [p.slug for p in personas]
    assert len(set(slugs)) == len(slugs), "Character slugs must be unique within a file"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {p.slug: p.to_json() for p in personas}
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=sort_keys), encoding="utf-8")


def load_candidate_scenarios(data_dir: Path = DATA_DIR) -> list[dict[str, Any]]:
    return json.loads((data_dir / "candidate_scenarios.json").read_text(encoding="utf-8"))


def load_roles(data_dir: Path = DATA_DIR) -> list[str]:
    with open(data_dir / "roles.csv", encoding="utf-8", newline="") as f:
        return [row[0] for row in csv.reader(f)][1:]  # skip the "role" header


def _persona_from_scenario(scenario: dict[str, Any], role: str, weights: Sequence[float]) -> Persona:
    return Persona(
        slug=scenario["short_name"],
        question=scenario["question"],
        role=role,
        attributes=deepfreeze({a["name"]: {k: v for k, v in a.items() if k != "name"} for a in scenario["attributes"]}),
        preference_weights=deepfreeze({a["name"]: w for a, w in zip(scenario["attributes"], weights)}),
    )


def novel_personas(
    existing: Sequence[Persona] = (),
    *,
    shuffle_seed: int = 2025,
    persona_seed: int = 2025,
    exclude_slash_roles: bool = False,
    data_dir: Path = DATA_DIR,
) -> Iterator[Persona]:
    """Characters whose scenario *and* role are both absent from `existing`, in a seeded random order.

    Used for (a) the 100 characters of the layer-freezing runs (`existing=()`), and (b) the characters that
    single-character adapters are trained on, none of which the backbone saw (`existing` = its 100 characters).
    """
    shuffle_rng = random.Random(shuffle_seed)
    persona_rng = random.Random(persona_seed)
    existing_slugs = {p.slug for p in existing}
    existing_roles = {p.role for p in existing}

    scenarios = [s for s in load_candidate_scenarios(data_dir) if s["short_name"] not in existing_slugs]
    shuffle_rng.shuffle(scenarios)
    roles = [
        role
        for role in load_roles(data_dir)
        if role not in existing_roles
        and not role.startswith("a ")
        and not role.startswith("an ")
        and not (exclude_slash_roles and "/" in role)
    ]
    shuffle_rng.shuffle(roles)

    for scenario, role in zip(scenarios, roles):
        if "/" in role:
            logger.debug(f"Role {role!r} contains a slash; it is replaced by '-' in adapter names")
        yield _persona_from_scenario(scenario, role, generate_preference_weights(persona_rng))


def resample_weights(persona: Persona, *, preference_seed: int, persona_idx: int) -> Persona:
    """New preference weights for a character, determined by (preference seed, index in the novel-character list).

    This is how the single-character adapters' targets were drawn.
    """
    rng = random.Random(preference_seed * 10_000 + persona_idx)
    return persona.with_weights(generate_preference_weights(rng))


type AttributeOrder = Literal["as-saved", "scenario", "alphabetical"]


def reorder_attributes(persona: Persona, order: AttributeOrder, data_dir: Path = DATA_DIR) -> Persona:
    """Reorder a character's attributes, which changes the order they are listed in prompts."""
    if order == "as-saved":
        return persona
    if order == "alphabetical":
        names = sorted(persona.attributes)
    elif order == "scenario":
        scenario = next((s for s in load_candidate_scenarios(data_dir) if s["short_name"] == persona.slug), None)
        assert scenario is not None, f"Unknown scenario {persona.slug!r}"
        names = [a["name"] for a in scenario["attributes"]]
        assert set(names) == set(persona.attributes), f"Attributes of {persona.slug!r} do not match its scenario"
    else:
        raise ValueError(f"Unknown attribute order {order!r}")
    return dataclasses.replace(
        persona,
        attributes=frozendict({n: persona.attributes[n] for n in names}),
        preference_weights=frozendict({n: persona.preference_weights[n] for n in names}),
    )
