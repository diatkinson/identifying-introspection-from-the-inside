"""LoRA adapters: locating them (local or Hugging Face Hub), stacking them, and choosing which ones train.

The paper's models are all PEFT LoRA adapters on frozen base models:
- multi-character ("backbone") adapters, e.g. deft-raptor (rank 8, all 64 layers of Qwen3-32B);
- single-character ("lightweight") adapters (rank 2), trained *on top of* a frozen backbone. At inference both
  adapters are active at once and their contributions add.
"""

import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import peft
from loguru import logger

LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
_PROJ_TO_SUBMODULE = {p: "self_attn" for p in LORA_TARGET_MODULES[:4]} | {p: "mlp" for p in LORA_TARGET_MODULES[4:]}
# Multimodal towers of Gemma-4 checkpoints; never adapted.
DEFAULT_EXCLUDE_MODULES = r".*(vision_tower|audio_tower|multi_modal_projector).*"
HF_PREFIX = "hf://"

type ModuleKey = tuple[int, str]  # (layer index, projection name), e.g. (5, "q_proj")


# ---------------------------------------------------------------------------
# Locating adapters
# ---------------------------------------------------------------------------


def resolve_adapter_path(spec: str | Path) -> Path:
    """Turn an adapter spec into a local directory containing `adapter_config.json`.

    `spec` is either a local path or `hf://<namespace>/<repo>/<path/inside/repo>`, which is downloaded (only that
    sub-folder, plus any `personas.json`/`*.json` metadata next to it) into the Hugging Face cache.
    """
    spec = str(spec)
    if not spec.startswith(HF_PREFIX):
        path = Path(spec).expanduser().resolve()
        assert (path / "adapter_config.json").exists(), f"No adapter_config.json in {path}"
        return path

    from huggingface_hub import snapshot_download

    parts = spec.removeprefix(HF_PREFIX).strip("/").split("/")
    assert len(parts) >= 2, f"Expected hf://<namespace>/<repo>[/<subfolder>], got {spec!r}"
    repo_id, subfolder = "/".join(parts[:2]), "/".join(parts[2:])
    parent = str(Path(subfolder).parent) if subfolder else ""
    patterns = [f"{subfolder}/*"] if subfolder else ["*"]
    if parent and parent != ".":
        patterns.append(f"{parent}/*.json")  # e.g. personas.json / training_config.json next to a checkpoint dir
    local_root = Path(snapshot_download(repo_id=repo_id, allow_patterns=patterns))
    path = local_root / subfolder
    assert (path / "adapter_config.json").exists(), f"No adapter_config.json at {spec}"
    return path


def adapter_dir_metadata_file(adapter_path: Path, name: str) -> Path | None:
    """Find a metadata file (e.g. `personas.json`) in the adapter directory or its parent."""
    for directory in (adapter_path, adapter_path.parent):
        if (directory / name).exists():
            return directory / name
    return None


def base_model_name(adapter_path: Path) -> str:
    config = json.loads((Path(adapter_path) / "adapter_config.json").read_text())
    return config["base_model_name_or_path"]


@dataclass(frozen=True)
class AdapterRef:
    """A trained adapter on disk, and the name it is registered under inside a PEFT model."""

    path: Path
    name: str

    @classmethod
    def from_spec(cls, spec: str | Path, name: str | None = None) -> "AdapterRef":
        path = resolve_adapter_path(spec)
        return cls(path=path, name=name or sanitize_adapter_name(default_adapter_name(path)))


def default_adapter_name(path: Path) -> str:
    """`.../deft-raptor/checkpoint-3000` -> `deft-raptor-3000`; a lightweight adapter's dir name is its character."""
    match = re.fullmatch(r"checkpoint-(\d+)", path.name)
    if match:
        return f"{path.parent.name}-{match.group(1)}"
    return path.name


def sanitize_adapter_name(name: str) -> str:
    # PEFT uses adapter names as module-dict keys, which cannot contain periods.
    return name.replace(".", "_")


# ---------------------------------------------------------------------------
# LoRA configs
# ---------------------------------------------------------------------------


def layer_range(min_layer: int, max_layer: int | None, n_layers: int) -> list[int] | None:
    """Layers [min_layer, max_layer) to adapt; None means all layers.

    (The original sweep code returned None whenever `max_layer` was None, silently ignoring `min_layer`. The paper's
    reversed-freezing runs passed `max_layer=40` explicitly, so they were unaffected.)
    """
    max_layer = n_layers if max_layer is None else max_layer
    assert 0 <= min_layer < max_layer <= n_layers, f"Invalid layer range [{min_layer}, {max_layer}) for {n_layers} layers"
    if min_layer == 0 and max_layer == n_layers:
        return None
    return list(range(min_layer, max_layer))


def lora_config(
    rank: int,
    *,
    layers_to_transform: list[int] | None = None,
    target_modules: Sequence[str] = LORA_TARGET_MODULES,
    exclude_modules: str | None = DEFAULT_EXCLUDE_MODULES,
) -> peft.LoraConfig:
    """Rank-stabilized LoRA with alpha = rank, as used for every adapter in the paper."""
    return peft.LoraConfig(
        r=rank,
        lora_alpha=rank,
        use_rslora=True,
        lora_dropout=0.0,
        target_modules=list(target_modules),
        layers_to_transform=layers_to_transform,
        exclude_modules=exclude_modules,
        task_type="CAUSAL_LM",
    )


# ---------------------------------------------------------------------------
# Managing adapters on a model
# ---------------------------------------------------------------------------


def active_adapters(model) -> set[str]:
    if not isinstance(model, peft.PeftModel):
        return set()
    active = model.base_model.active_adapter
    return {active} if isinstance(active, str) else set(active)


def _adapter_of_param(param_name: str, adapters: set[str]) -> str | None:
    if "lora_A" not in param_name and "lora_B" not in param_name:
        return None
    for adapter in adapters:
        if f".{adapter}." in param_name or param_name.endswith(f".{adapter}"):
            return adapter
    return None


def freeze_adapters(model, adapters: set[str]) -> None:
    for name, param in model.named_parameters():
        if _adapter_of_param(name, adapters) is not None:
            param.requires_grad = False


def trainable_adapters(model) -> list[str]:
    """Adapters whose parameters require grad. Raises if any base-model parameter is trainable."""
    adapters = set(model.peft_config) if isinstance(model, peft.PeftModel) else set()
    counts = {a: [0, 0] for a in adapters}  # [trainable, frozen]
    for name, param in model.named_parameters():
        adapter = _adapter_of_param(name, adapters)
        if adapter is None:
            if param.requires_grad:
                raise ValueError(f"Base-model parameter {name} is trainable")
            continue
        counts[adapter][0 if param.requires_grad else 1] += 1
    for adapter, (n_trainable, n_frozen) in counts.items():
        if n_trainable and n_frozen:
            raise ValueError(f"Adapter {adapter!r} is only partially trainable")
    return sorted(a for a, (n_trainable, _) in counts.items() if n_trainable)


def set_adapters(model, active: Sequence[str], trainable: str | None = None) -> None:
    """Activate exactly `active` (outputs of active adapters add) and make only `trainable` (if any) train."""
    active = list(active)
    model.base_model.set_adapter(active)
    # PEFT's `PeftModel.active_adapter` is not updated by `base_model.set_adapter`; the HF Trainer reads it.
    model.active_adapter = active[0] if len(active) == 1 else (trainable or active[0])
    for name, param in model.named_parameters():
        adapter = _adapter_of_param(name, set(model.peft_config))
        if adapter is not None:
            param.requires_grad = adapter == trainable
    assert active_adapters(model) == set(active), f"Active adapters {active_adapters(model)} != {set(active)}"
    if trainable is not None:
        assert trainable_adapters(model) == [trainable], f"Expected only {trainable!r} to be trainable"
    else:
        assert trainable_adapters(model) == [], "Expected no trainable adapters"


def load_adapter(model, adapter: AdapterRef, *, is_trainable: bool = False):
    """Load a trained adapter (and activate only it). Returns the (possibly newly wrapped) PEFT model."""
    weights = [adapter.path / "adapter_model.safetensors", adapter.path / "adapter_model.bin"]
    assert any(p.is_file() for p in weights), f"No adapter weights in {adapter.path}"
    if adapter.name in getattr(model, "peft_config", {}):
        raise ValueError(f"An adapter named {adapter.name!r} is already loaded")
    if isinstance(model, peft.PeftModel):
        model.load_adapter(str(adapter.path), adapter_name=adapter.name, is_trainable=is_trainable)
        model.set_adapter(adapter.name)
    else:
        model = peft.PeftModel.from_pretrained(
            model, str(adapter.path), adapter_name=adapter.name, is_trainable=is_trainable
        )
    logger.info(f"Loaded adapter {adapter.name!r} from {adapter.path}")
    return model


def add_adapter(model, config: peft.LoraConfig, name: str):
    """Add a freshly initialized adapter, keeping any existing adapters frozen. Returns the PEFT model."""
    if isinstance(model, peft.PeftModel):
        if name in model.peft_config:
            raise ValueError(f"An adapter named {name!r} already exists")
        model.add_adapter(adapter_name=name, peft_config=config)
    else:
        model = peft.get_peft_model(model, config, adapter_name=name)
    return model


def delete_adapter(model, name: str) -> None:
    """Remove an adapter (PEFT re-points the active adapter itself). Call `set_adapters` before using the model again."""
    model.delete_adapter(name)


def lora_modules(model, adapter_name: str) -> dict[ModuleKey, object]:
    """The LoRA-wrapped linear layers that carry `adapter_name`, keyed by (layer, projection)."""
    modules = {}
    for name, module in model.named_modules():
        if not (hasattr(module, "lora_A") and adapter_name in module.lora_A):
            continue
        parts = name.split(".")
        layer_idx = next((int(parts[i + 1]) for i, p in enumerate(parts[:-1]) if p == "layers" and parts[i + 1].isdigit()), None)
        proj = next((p for p in parts if p in _PROJ_TO_SUBMODULE), None)
        if layer_idx is not None and proj is not None:
            modules[(layer_idx, proj)] = module
    return modules


def module_name(layer: int, proj: str) -> str:
    """Canonical module name used in attribution files, e.g. `model.layers.5.self_attn.q_proj`."""
    return f"model.layers.{layer}.{_PROJ_TO_SUBMODULE[proj]}.{proj}"
