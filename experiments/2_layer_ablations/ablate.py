"""Figure 3: evaluate a multi-character adapter with groups of its layers removed.

    python ablate.py --adapter hf://datkinson/identifying-introspection/deft-raptor/checkpoint-1000 --direction prev  --output-dir results/ablations/ckpt1000-prev
    python ablate.py --adapter hf://datkinson/identifying-introspection/deft-raptor/checkpoint-1000 --direction later --output-dir results/ablations/ckpt1000-later
    (and the same for checkpoint-3000)

`--direction prev` at layer L removes the adapter from layers 0..L ("all earlier layers removed"); `later` removes it
from layers L..N-1. Removing means zeroing `lora_A`, so the layer computes exactly what the base model computes.
For each L this writes `<output-dir>/layer_<L>/{metrics.json,eval_results.json}`; finished layers are skipped, so
the script can be restarted or split across GPUs with `--layers`.
"""

import argparse
import random
from collections import defaultdict
from pathlib import Path

import torch
from loguru import logger

from introspection.adapters import AdapterRef, adapter_dir_metadata_file, base_model_name, load_adapter, lora_modules, set_adapters
from introspection.characters import load_personas
from introspection.evaluation import EvalConfig, run_eval_suite
from introspection.models import add_model_args, get_tokenizer, load_model_from_args, num_hidden_layers
from introspection.utils import filter_warnings, save_json, setup_logging


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--adapter", required=True, help="Adapter directory or hf:// spec")
    parser.add_argument("--direction", choices=["prev", "later"], required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--layers", type=str, help="Subset of boundary layers, e.g. '0:32' (default: all)")
    parser.add_argument("--num-characters", type=int, default=100)
    parser.add_argument("--num-choice-samples", type=int, default=16)
    parser.add_argument("--num-self-report-samples", type=int, default=24)
    parser.add_argument("--choice-probs", choices=["raw", "processed"], default="processed")
    parser.add_argument("--batch-size", type=int, default=16)
    add_model_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_logging()
    filter_warnings()

    adapter = AdapterRef.from_spec(args.adapter, name="adapter")
    model_name = base_model_name(adapter.path)
    personas_file = adapter_dir_metadata_file(adapter.path, "personas.json")
    assert personas_file is not None, f"No personas.json for {adapter.path}"
    all_personas = load_personas(personas_file)
    if args.num_characters >= len(all_personas):
        personas = all_personas
    else:
        personas = random.Random(2025).sample(all_personas, args.num_characters)
    logger.info(f"Evaluating {len(personas)} of {len(all_personas)} characters")

    tokenizer = get_tokenizer(model_name)
    model = load_model_from_args(model_name, args)
    model = load_adapter(model, adapter)
    set_adapters(model, [adapter.name])
    model.eval()
    n_layers = num_hidden_layers(model)
    by_layer: dict[int, list] = defaultdict(list)
    for (layer, _proj), module in lora_modules(model, adapter.name).items():
        by_layer[layer].append(module)

    layers = range(n_layers)
    if args.layers:
        start, stop = (int(x) for x in args.layers.split(":"))
        layers = range(start, stop)
    eval_cfg = EvalConfig(
        num_choice_samples=args.num_choice_samples,
        num_self_report_samples=args.num_self_report_samples,
        batch_size=args.batch_size,
        choice_probs=args.choice_probs,
        adapters=(adapter.name,),
    )
    save_json(
        args.output_dir / "config.json",
        {"adapter": str(args.adapter), "direction": args.direction, "n_layers": n_layers, "eval": eval_cfg.to_json()},
    )

    for layer in layers:
        out = args.output_dir / f"layer_{layer}"
        if (out / "metrics.json").exists():
            continue
        ablated = list(range(0, layer + 1)) if args.direction == "prev" else list(range(layer, n_layers))
        backups = []
        with torch.no_grad():
            for ablated_layer in ablated:
                for module in by_layer[ablated_layer]:
                    weight = module.lora_A[adapter.name].weight
                    backups.append((weight, weight.detach().clone()))
                    weight.zero_()
        try:
            run_eval_suite(
                model, {"trained": personas}, eval_cfg, tokenizer, output_dir=out,
                extra_metrics={"ablated_layer": layer, "ablated_layers": ablated},
            )
        finally:
            with torch.no_grad():
                for weight, backup in backups:
                    weight.copy_(backup)
        logger.info(f"Finished layer {layer}")


if __name__ == "__main__":
    main()
