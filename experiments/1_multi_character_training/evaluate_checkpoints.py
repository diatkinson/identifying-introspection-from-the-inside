"""Evaluate saved checkpoints of a multi-character adapter on all of its characters (Figure 1b).

    # The published deft-raptor checkpoints, plus the base model as step 0:
    python evaluate_checkpoints.py --base --deft-raptor --output-dir results/deft-raptor
    # Or every checkpoint of your own run:
    python evaluate_checkpoints.py --base --run-dir runs/deft-raptor_Qwen3-32B_layers0-all --output-dir results/deft-raptor

For each checkpoint this writes `<output-dir>/checkpoint-<step>/{metrics.json,eval_results.json}` (`base/` for the base
model).
"""

import argparse
import re
from pathlib import Path

from loguru import logger

from introspection.adapters import AdapterRef, adapter_dir_metadata_file, base_model_name, delete_adapter, load_adapter, set_adapters
from introspection.characters import load_personas
from introspection.evaluation import EvalConfig, run_eval_suite
from introspection.hub import DEFT_RAPTOR_STEPS, deft_raptor_checkpoint
from introspection.models import add_model_args, get_tokenizer, load_model_from_args
from introspection.utils import filter_warnings, setup_logging


def checkpoint_step(path: Path) -> int:
    match = re.fullmatch(r"checkpoint-(\d+)", Path(path).name)
    assert match, f"Expected a checkpoint-<step> directory, got {path}"
    return int(match.group(1))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapters", nargs="*", default=[], help="Checkpoint directories or hf:// specs")
    parser.add_argument("--run-dir", type=Path, help="Evaluate every checkpoint-<step> directory in this run")
    parser.add_argument("--deft-raptor", action="store_true", help="Evaluate the published deft-raptor checkpoints")
    parser.add_argument("--base", action="store_true", help="Also evaluate the base model without any adapter")
    parser.add_argument("--characters", type=Path, help="personas.json to evaluate on (default: the adapter's own)")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-choice-samples", type=int, default=16)
    parser.add_argument("--num-self-report-samples", type=int, default=24)
    parser.add_argument("--choice-probs", choices=["raw", "processed"], default="processed")
    parser.add_argument("--batch-size", type=int, default=32)
    add_model_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_logging()
    filter_warnings()

    specs = list(args.adapters)
    if args.deft_raptor:
        specs += [deft_raptor_checkpoint(step) for step in DEFT_RAPTOR_STEPS]
    if args.run_dir is not None:
        specs += sorted((str(p) for p in args.run_dir.glob("checkpoint-*") if re.fullmatch(r"checkpoint-\d+", p.name)), key=checkpoint_step)
    assert specs or args.base, "Nothing to evaluate"
    adapters = [AdapterRef.from_spec(s, name=f"ckpt-{checkpoint_step(Path(str(s).rstrip('/')))}") for s in specs]

    if args.characters is not None:
        personas = load_personas(args.characters)
    else:
        assert adapters, "--characters is required when only evaluating the base model"
        source = adapter_dir_metadata_file(adapters[0].path, "personas.json")
        assert source is not None, f"No personas.json next to {adapters[0].path}; pass --characters"
        personas = load_personas(source)
    model_name = base_model_name(adapters[0].path) if adapters else "Qwen/Qwen3-32B"
    logger.info(f"Evaluating {len(personas)} characters on {model_name}")

    tokenizer = get_tokenizer(model_name)
    model = load_model_from_args(model_name, args)
    eval_kwargs = dict(
        num_choice_samples=args.num_choice_samples,
        num_self_report_samples=args.num_self_report_samples,
        batch_size=args.batch_size,
        choice_probs=args.choice_probs,
    )

    if args.base:
        out = args.output_dir / "base"
        if (out / "metrics.json").exists():
            logger.info(f"Skipping the base model; {out} exists")
        else:
            assert not hasattr(model, "peft_config"), "Evaluate the base model before loading adapters"
            run_eval_suite(model, {"all": personas}, EvalConfig(**eval_kwargs), tokenizer, output_dir=out)

    for adapter in adapters:
        out = args.output_dir / f"checkpoint-{checkpoint_step(adapter.path)}"
        if (out / "metrics.json").exists():
            logger.info(f"Skipping {adapter.path}; {out} exists")
            continue
        model = load_adapter(model, adapter)
        set_adapters(model, [adapter.name])
        run_eval_suite(model, {"all": personas}, EvalConfig(**eval_kwargs, adapters=(adapter.name,)), tokenizer, output_dir=out)
        delete_adapter(model, adapter.name)


if __name__ == "__main__":
    main()
