"""Train a multi-character LoRA adapter on the decision task (deft-raptor and the layer-freezing runs).

    python train.py --config configs/deft-raptor.yaml
    python train.py --config configs/qwen3-14b-layer-freezing.yaml --max-layer 10      # train layers [0, 10)
    python train.py --config configs/qwen3-14b-layer-freezing.yaml --min-layer 25      # train layers [25, 40)

Writes to `<output-dir>/` (default `runs/<config name>[-<layers>]`): `checkpoint-<step>/` adapters, `personas.json`,
`run_config.json`, and, for configs with in-training evaluation, `checkpoint-<step>-eval/metrics.json`.
"""

import argparse
from pathlib import Path

import yaml
from loguru import logger
from transformers import Trainer

from introspection.adapters import add_adapter, layer_range, lora_config, set_adapters
from introspection.characters import load_personas, save_personas
from introspection.evaluation import EvalConfig
from introspection.legacy_data import deft_raptor_training_set
from introspection.models import disable_kv_cache, get_tokenizer, load_model, num_hidden_layers
from introspection.training import (
    ChoiceDataCollator,
    ChoiceDataset,
    EvalCallback,
    EvalSchedule,
    OptimConfig,
    SaveAtSteps,
    StopAtStep,
    make_callbacks,
    sample_choice_questions,
    seed_everything_before_adapter_init,
    total_optimizer_steps,
    training_arguments,
)
from introspection.utils import DATA_DIR, filter_warnings, save_json, setup_logging

ADAPTER_NAME = "default"
CHARACTER_FILES = {"deft-raptor": DATA_DIR / "characters" / "deft-raptor.json", "sweep-pool": DATA_DIR / "characters" / "sweep-pool.json"}


def load_config(path: Path, overrides: list[str]) -> dict:
    cfg = yaml.safe_load(Path(path).read_text())
    for override in overrides:
        key, value = override.split("=", 1)
        node = cfg
        *parents, leaf = key.split(".")
        for p in parents:
            node = node.setdefault(p, {})
        node[leaf] = yaml.safe_load(value)
    return cfg


def characters(cfg: dict):
    spec = cfg["characters"]
    personas = load_personas(spec.get("path") or CHARACTER_FILES[spec["source"]])
    n = spec["num"]
    assert len(personas) >= n, f"Need {n} characters, found {len(personas)}"
    return personas[:n]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model-name")
    parser.add_argument("--min-layer", type=int, help="First trained layer (default from config)")
    parser.add_argument("--max-layer", type=int, help="Train layers below this index (default from config)")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--quant", choices=["8", "4", "none"], help="Override the config's base-model quantization")
    parser.add_argument("--dtype", help="Override the config's dtype for non-quantized weights")
    parser.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="Override a config entry, e.g. optim.micro_batch_size=2")
    args = parser.parse_args()
    setup_logging()
    filter_warnings()

    cfg = load_config(args.config, args.set)
    if args.model_name:
        cfg["model_name"] = args.model_name
    if args.min_layer is not None:
        cfg["lora"]["min_layer"] = args.min_layer
    if args.max_layer is not None:
        cfg["lora"]["max_layer"] = args.max_layer
    if args.quant is not None:
        cfg["quant"] = None if args.quant == "none" else args.quant
    if args.dtype is not None:
        cfg["dtype"] = args.dtype

    model_short = cfg["model_name"].split("/")[-1]
    lora_cfg = cfg["lora"]
    layers_tag = f"layers{lora_cfg['min_layer']}-{'all' if lora_cfg['max_layer'] is None else lora_cfg['max_layer'] - 1}"
    output_dir = args.output_dir or Path("runs") / f"{cfg['name']}_{model_short}_{layers_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "run_config.json", cfg)

    tokenizer = get_tokenizer(cfg["model_name"])
    trained = characters(cfg)
    save_personas(output_dir / "personas.json", trained, sort_keys=cfg["characters"]["save_sorted"])

    data_cfg = cfg["data"]
    if data_cfg["source"] == "legacy-deft-raptor":
        assert cfg["characters"]["source"] == "deft-raptor", "The legacy data stream only exists for deft-raptor"
        examples = deft_raptor_training_set(seed=data_cfg["seed"], n_characters=len(trained)).shuffled(data_cfg["shuffle_seed"])
        assert [p.to_json() for p in examples.personas] == [p.to_json() for p in trained]
    elif data_cfg["source"] == "sampled":
        examples = sample_choice_questions(trained, data_cfg["num_examples"], seed=data_cfg["seed"])
    else:
        raise ValueError(f"Unknown data source {data_cfg['source']!r}")
    dataset = ChoiceDataset(examples, tokenizer)
    logger.info(f"{len(dataset)} training examples; first: {dataset.question(0).training_format(tokenizer)[-300:]!r}")

    model = load_model(cfg["model_name"], quant=cfg.get("quant", "8"), dtype=cfg.get("dtype", "auto"))
    n_layers = num_hidden_layers(model)
    layers = layer_range(lora_cfg["min_layer"], lora_cfg["max_layer"], n_layers)
    seed_everything_before_adapter_init(cfg["optim"]["seed"])
    model = add_adapter(model, lora_config(lora_cfg["rank"], layers_to_transform=layers), ADAPTER_NAME)
    set_adapters(model, [ADAPTER_NAME], trainable=ADAPTER_NAME)
    disable_kv_cache(model)
    logger.info(f"Training layers {layers or 'all'} of {n_layers}; {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")

    optim = OptimConfig(**cfg["optim"])
    total_steps = total_optimizer_steps(len(dataset), optim.effective_batch_size)
    targs = training_arguments(optim, output_dir, total_steps=total_steps, report_to="wandb" if args.wandb else "none")
    logger.info(f"{total_steps} optimizer steps, {optim.warmup_for(total_steps)} warmup steps")

    eval_callback = None
    if (eval_cfg := cfg.get("eval")) is not None:
        eval_callback = EvalCallback(
            persona_groups={"novel": trained[: eval_cfg["num_trained_characters"]]},
            eval_cfg=EvalConfig(
                num_choice_samples=eval_cfg["num_choice_samples"],
                num_self_report_samples=eval_cfg["num_self_report_samples"],
                batch_size=eval_cfg.get("batch_size", 32),
                seed=eval_cfg.get("seed", 2025),
                choice_probs=eval_cfg["choice_probs"],
                adapters=(ADAPTER_NAME,),
            ),
            tokenizer=tokenizer,
            output_dir=output_dir,
            effective_bsz=optim.effective_batch_size,
            total_samples=len(dataset),
            schedule=EvalSchedule(**eval_cfg["schedule"]),
            save_at_eval=eval_cfg.get("save_at_eval", True),
            log_to_wandb=args.wandb,
        )
    ckpt_cfg = cfg.get("checkpoints") or {}
    callbacks = make_callbacks(
        eval_callback,
        SaveAtSteps(ckpt_cfg.get("steps", []), ckpt_cfg.get("every")) if ckpt_cfg else None,
        StopAtStep(cfg["stop_at_step"]) if cfg.get("stop_at_step") else None,
    )

    trainer = Trainer(model=model, args=targs, train_dataset=dataset, data_collator=ChoiceDataCollator(tokenizer), callbacks=callbacks)
    trainer.train()
    logger.info(f"Done; outputs in {output_dir}")


if __name__ == "__main__":
    main()
