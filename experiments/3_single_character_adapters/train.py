"""Train single-character ("lightweight") adapters on top of a frozen multi-character backbone.

    # The paper's 32 characters, on each backbone (run both to get the 32 data-matched pairs):
    python train.py --backbone hf://datkinson/identifying-introspection/deft-raptor/checkpoint-1000 --paper-pairs --output-root runs/single-character
    python train.py --backbone hf://datkinson/identifying-introspection/deft-raptor/checkpoint-3000 --paper-pairs --output-root runs/single-character
    # Or a grid of characters and seeds to select pairs from (then run select_pairs.py):
    python train.py --backbone ... --characters 0:212 --init-seeds 0 1 2 --output-root runs/single-character

None of the characters appeared in the backbone's training. A character's target weights are re-drawn from
(preference seed, character index); the initialization seed sets the training data, the data order, the LoRA initialization and the
evaluation prompts, so the same character and seeds on two backbones give a data-matched pair.

Output: `<output-root>/<backbone>/<character>_pseed<p>_iseed<i>/` with `personas.json`, `training_config.json`,
`final-eval/metrics.json` and the adapter itself in the `<character>/` sub-directory.
"""

import argparse
import json
from pathlib import Path

from loguru import logger
from peft import prepare_model_for_kbit_training
from transformers import Trainer

from introspection.adapters import AdapterRef, add_adapter, adapter_dir_metadata_file, base_model_name, delete_adapter, load_adapter, lora_config, sanitize_adapter_name, set_adapters
from introspection.characters import Persona, load_personas, novel_personas, resample_weights, save_personas
from introspection.evaluation import EvalConfig
from introspection.models import add_model_args, disable_kv_cache, get_tokenizer, load_model_from_args
from introspection.training import (
    ChoiceDataCollator,
    ChoiceDataset,
    EvalCallback,
    OptimConfig,
    sample_single_character_questions,
    seed_everything_before_adapter_init,
    total_optimizer_steps,
    training_arguments,
)
from introspection.utils import DATA_DIR, filter_warnings, save_json, setup_logging

# Settings of the paper's single-character adapters.
RANK = 2
N_EXAMPLES = 768  # 24 steps at an effective batch size of 32
OPTIM = dict(
    learning_rate=1e-3,
    lr_scheduler_type="cosine",
    warmup_steps=20,
    weight_decay=0.0,
    optim="adamw_torch",
    effective_batch_size=32,
    micro_batch_size=4,
)
EVAL = dict(num_choice_samples=16, num_self_report_samples=24, batch_size=32, choice_probs="raw")


def jobs_from_args(args, backbone_personas: list[Persona]) -> list[tuple[Persona, int, int, int]]:
    """(character with re-drawn weights, novel index, preference seed, init seed) for every adapter to train."""
    if args.paper_pairs:
        records = json.loads((DATA_DIR / "characters" / "single-character-pairs.json").read_text())
        return [(Persona.from_json(r["persona"]), r["novel_index"], r["preference_seed"], r["init_seed"]) for r in records]
    start, stop = (int(x) for x in args.characters.split(":"))
    novel = [p for _, p in zip(range(stop), novel_personas(backbone_personas))]
    return [
        (resample_weights(novel[idx], preference_seed=pseed, persona_idx=idx), idx, pseed, iseed)
        for pseed in args.preference_seeds
        for idx in range(start, stop)
        for iseed in args.init_seeds
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backbone", required=True, help="Backbone checkpoint directory or hf:// spec")
    parser.add_argument("--output-root", type=Path, required=True)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--paper-pairs", action="store_true", help="The 32 characters and seeds of the paper's pairs")
    selection.add_argument("--characters", help="Range of novel-character indices, e.g. 0:212")
    parser.add_argument("--preference-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--init-seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--micro-batch-size", type=int, default=OPTIM["micro_batch_size"])
    parser.add_argument("--gradient-checkpointing", action="store_true")
    parser.add_argument("--num-choice-samples", type=int, default=EVAL["num_choice_samples"])
    parser.add_argument("--num-self-report-samples", type=int, default=EVAL["num_self_report_samples"])
    add_model_args(parser)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_logging()
    filter_warnings()

    backbone = AdapterRef.from_spec(args.backbone)
    model_name = base_model_name(backbone.path)
    backbone_personas = load_personas(adapter_dir_metadata_file(backbone.path, "personas.json"))
    jobs = jobs_from_args(args, backbone_personas)
    seen_slugs, seen_roles = {p.slug for p in backbone_personas}, {p.role for p in backbone_personas}
    assert all(p.slug not in seen_slugs and p.role not in seen_roles for p, *_ in jobs), "Characters must be new to the backbone"

    tokenizer = get_tokenizer(model_name)
    model = load_model_from_args(model_name, args)
    if args.quant != "none":
        # As in the original runs: freeze the quantized model and upcast its non-int8 parameters (embeddings, norms,
        # output head) to float32. (PEFT upcasts *every* half-precision parameter, so skip this for unquantized models.)
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)
    model = load_adapter(model, backbone)

    eval_kwargs = {**EVAL, "num_choice_samples": args.num_choice_samples, "num_self_report_samples": args.num_self_report_samples}
    for persona, novel_idx, pseed, iseed in jobs:
        run_dir = args.output_root / backbone.name / f"{persona}_pseed{pseed}_iseed{iseed}"
        if (run_dir / "final-eval" / "metrics.json").exists():
            logger.info(f"Skipping {run_dir}: already trained")
            continue
        logger.info(f"Training {persona} (novel index {novel_idx}, pseed {pseed}, iseed {iseed}) on {backbone.name}")
        name = sanitize_adapter_name(str(persona))
        save_personas(run_dir / "personas.json", [persona], sort_keys=True)
        optim = OptimConfig(**{**OPTIM, "micro_batch_size": args.micro_batch_size}, gradient_checkpointing=args.gradient_checkpointing, seed=iseed)
        backbone_spec = args.backbone if args.backbone.startswith("hf://") else str(backbone.path)
        save_json(
            run_dir / "training_config.json",
            {"backbone": backbone_spec, "base_model": model_name, "rank": RANK, "n_examples": N_EXAMPLES,
             "novel_index": novel_idx, "preference_seed": pseed, "init_seed": iseed, "optim": optim, "eval": eval_kwargs},
        )

        seed_everything_before_adapter_init(iseed)
        model = add_adapter(model, lora_config(RANK), name)
        set_adapters(model, [backbone.name, name], trainable=name)
        disable_kv_cache(model)

        dataset = ChoiceDataset(sample_single_character_questions(persona, N_EXAMPLES, seed=iseed), tokenizer)
        total_steps = total_optimizer_steps(len(dataset), optim.effective_batch_size)
        callback = EvalCallback(
            persona_groups={"novel": [persona]},
            eval_cfg=EvalConfig(**eval_kwargs, seed=iseed, adapters=(backbone.name, name)),
            tokenizer=tokenizer,
            output_dir=run_dir,
            effective_bsz=optim.effective_batch_size,
            total_samples=len(dataset),
            eval_at_start=False,
            save_at_eval=False,
        )
        trainer = Trainer(
            model=model,
            args=training_arguments(optim, run_dir, total_steps=total_steps),
            train_dataset=dataset,
            data_collator=ChoiceDataCollator(tokenizer),
            callbacks=[callback],
        )
        trainer.train()
        model.save_pretrained(run_dir, selected_adapters=[name])  # -> run_dir/<name>/adapter_model.safetensors
        delete_adapter(model, name)
        set_adapters(model, [backbone.name])


if __name__ == "__main__":
    main()
