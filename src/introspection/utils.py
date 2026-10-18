"""Small shared helpers: JSON I/O, seeding, logging."""

import json
import random
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import numpy as np
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
DEFAULT_SEED = 2025


def json_dumps(data: Any, indent: int | None = 2) -> str:
    """Canonical JSON used throughout the project.

    The exact settings matter: `Persona.__str__` hashes this string, and those hashes appear in the names of every
    adapter we trained and published.
    """
    return json.dumps(data, indent=indent, ensure_ascii=False, sort_keys=True)


def save_json(path: Path, data: Any, *, sort_keys: bool = True) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(to_jsonable(data), f, indent=2, ensure_ascii=False, sort_keys=sort_keys)


def load_json(path: Path) -> Any:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def to_jsonable(obj: Any) -> Any:
    if hasattr(obj, "to_json"):
        return obj.to_json()
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: to_jsonable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict) or hasattr(obj, "items"):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def set_seed(seed: int) -> None:
    """Seed Python, NumPy and torch (CPU and CUDA)."""
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def setup_logging(level: str = "INFO") -> None:
    logger.remove()
    logger.add(sys.stderr, level=level)


def filter_warnings() -> None:
    """Silence warnings that are expected in this project's normal operation."""
    import warnings

    warnings.filterwarnings("ignore", message=r"MatMul8bitLt: inputs will be cast from .* to float16 during quantization")
    warnings.filterwarnings("ignore", message=r"Already found a `peft_config` attribute in the model.*")
    warnings.filterwarnings("ignore", message=r"You have passed exclude_modules=.* but no modules were excluded.*")
    warnings.filterwarnings("ignore", message=r"Adapter .* was active which is now deleted.*")
    warnings.filterwarnings("ignore", message=r"Full backward hook is firing when gradients are computed.*")
