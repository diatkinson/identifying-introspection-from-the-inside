"""Attribution patching of single-character adapters on decision and self-report prompts.

    # One adapter (its backbone is read from training_config.json, or pass --backbone):
    python attribute.py --adapter runs/single-character/deft-raptor-3000/<character>_pseed0_iseed2/<character>
    # All 64 published adapters of the paper's 32 pairs (resumable; loads the base model once):
    python attribute.py --paper-pairs

For each adapter and prompt type this writes `<output-dir>/<character>__<backbone>__<prompt type>.parquet` with the
integrated-gradients path scores of every LoRA site and output dimension (rows: module x IG step), averaged over
`--n-questions` prompts. The decision and self-report prompts present the same pairs of options.

Defaults match the paper (50 prompts, 7 midpoint-rule IG steps). Two options concern the original runs:
- `--kl-dtype model` computes the KL loss in the model's bfloat16 logits, as the original code did (default float32);
- `--attribute-order` (default `as-saved`) lists attributes in the order of the adapter's personas.json, which for the
  published single-character adapters is alphabetical, as in the original attribution runs; `scenario` uses the
  order the adapters were trained with.
"""

import argparse
import json
from pathlib import Path

import torch
from loguru import logger

from introspection.adapters import AdapterRef, adapter_dir_metadata_file, base_model_name, delete_adapter, load_adapter, set_adapters
from introspection.attribution import attribute_adapter, attribution_questions
from introspection.characters import PromptType, load_personas, reorder_attributes
from introspection.hub import single_character_adapter
from introspection.models import add_model_args, get_tokenizer, load_model_from_args
from introspection.utils import DATA_DIR, filter_warnings, setup_logging


def backbone_spec(adapter_path: Path) -> str:
    config_file = adapter_dir_metadata_file(adapter_path, "training_config.json")
    assert config_file is not None, f"No training_config.json for {adapter_path}; pass --backbone"
    return json.loads(config_file.read_text())["backbone"]


def output_path(output_dir: Path, character: str, backbone_name: str, prompt_type: PromptType) -> Path:
    return output_dir / f"{character}__{backbone_name}__{prompt_type.value}.parquet"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--adapter", help="Single-character adapter directory or hf:// spec")
    target.add_argument("--paper-pairs", action="store_true", help="All 64 published adapters of the paper's pairs")
    parser.add_argument("--backbone", help="Backbone directory or hf:// spec (default: from training_config.json)")
    parser.add_argument("--prompt-types", nargs="+", default=["choice", "report_prefix"], choices=["choice", "report_prefix"])
    parser.add_argument("--n-questions", type=int, default=50)
    parser.add_argument("--n-ig-steps", type=int, default=7)
    parser.add_argument("--seed", type=int, default=2025)
    parser.add_argument("--kl-dtype", choices=["float32", "model"], default="float32")
    parser.add_argument("--attribute-order", choices=["as-saved", "scenario", "alphabetical"], default="as-saved")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/attributions"),
        help="Existing files are skipped and file names do not encode the settings: use one directory per setting",
    )
    add_model_args(parser)
    args = parser.parse_args()
    setup_logging()
    filter_warnings()

    if args.paper_pairs:
        records = json.loads((DATA_DIR / "characters" / "single-character-pairs.json").read_text())
        jobs = [(single_character_adapter(r["adapter_name"], step), None) for r in records for step in (1000, 3000)]
    else:
        jobs = [(args.adapter, args.backbone)]

    model = tokenizer = None
    backbones: dict[str, AdapterRef] = {}
    for spec, backbone in jobs:
        adapter = AdapterRef.from_spec(spec, name="lightweight")
        backbone = backbone or backbone_spec(adapter.path)
        persona = load_personas(adapter_dir_metadata_file(adapter.path, "personas.json"))[0]
        persona = reorder_attributes(persona, args.attribute_order)
        character = str(persona)  # the hash ignores attribute order, so this is the adapter's name either way

        if backbone not in backbones:
            backbones[backbone] = AdapterRef.from_spec(backbone)
        bb = backbones[backbone]
        prompt_types = [PromptType(p) for p in args.prompt_types]
        todo = [p for p in prompt_types if not output_path(args.output_dir, character, bb.name, p).exists()]
        if not todo:
            logger.info(f"Skipping {character} on {bb.name}: done")
            continue

        if model is None:
            model_name = base_model_name(adapter.path)
            tokenizer = get_tokenizer(model_name)
            model = load_model_from_args(model_name, args)
        if bb.name not in getattr(model, "peft_config", {}):
            model = load_adapter(model, bb)
        model = load_adapter(model, adapter)
        set_adapters(model, [bb.name, adapter.name], trainable=adapter.name)

        for prompt_type in todo:
            logger.info(f"Attributing {character} on {bb.name}, {prompt_type.value} prompts")
            result = attribute_adapter(
                model,
                tokenizer,
                adapter_name=adapter.name,
                questions=attribution_questions(persona, prompt_type, args.n_questions, seed=args.seed),
                prompt_type=prompt_type,
                n_ig_steps=args.n_ig_steps,
                seed=args.seed,
                kl_dtype=torch.float32 if args.kl_dtype == "float32" else None,
                metadata={
                    "adapter": str(spec),
                    "adapter_path": str(adapter.path),
                    "backbone": str(backbone),
                    "character": character,
                    "attribute_order": args.attribute_order,
                    "kl_dtype": args.kl_dtype,
                },
            )
            result.save(output_path(args.output_dir, character, bb.name, prompt_type))
        delete_adapter(model, adapter.name)


if __name__ == "__main__":
    main()
