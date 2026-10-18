"""Loading base models and tokenizers."""

import functools
import gc

import torch
from loguru import logger
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@functools.cache
def get_tokenizer(model_name: str):
    """Left-padding tokenizer (all batched generation in this project assumes left padding)."""
    return AutoTokenizer.from_pretrained(model_name, padding_side="left")


def ab_token_ids(tokenizer) -> list[int]:
    ids = [tokenizer(t, add_special_tokens=False)["input_ids"] for t in ["A", "B"]]
    assert all(len(i) == 1 for i in ids), f"'A' and 'B' must be single tokens, got {ids}"
    return [i[0] for i in ids]


def quantization_config(quant: str | int | None) -> BitsAndBytesConfig | None:
    if quant is None or str(quant).lower() == "none":
        return None
    quant = str(quant)
    if quant == "8":
        return BitsAndBytesConfig(
            load_in_8bit=True,
            llm_int8_threshold=6.0,
            llm_int8_skip_modules=["lm_head"],  # keep the output head in full precision
        )
    if quant == "4":
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=False,
        )
    raise ValueError(f"Invalid quantization {quant!r}; expected '8', '4' or None")


def load_model(
    model_name: str,
    quant: str | None = "8",
    *,
    dtype: str | torch.dtype = "auto",
    attn_implementation: str = "sdpa",
    device_map: str | dict = "auto",
):
    """Load a base model. All of the paper's experiments used 8-bit (LLM.int8) quantization.

    `dtype` sets the dtype of the non-quantized parameters. "auto" is bfloat16 for Qwen3 and Gemma-4. (The Qwen3-14B
    layer-freezing runs loaded the model without a dtype, which bitsandbytes 8-bit loading turned into float16.)
    """
    if isinstance(dtype, str) and dtype != "auto":
        dtype = getattr(torch, dtype)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=dtype,
        quantization_config=quantization_config(quant),
        device_map=device_map,
        attn_implementation=attn_implementation,
    )
    _drop_multimodal_submodules(model)
    tokenizer = get_tokenizer(model_name)
    if model.generation_config.pad_token_id is None:
        model.generation_config.pad_token_id = tokenizer.pad_token_id
    return model


def _drop_multimodal_submodules(model) -> None:
    """Free the vision/audio towers of multimodal checkpoints (Gemma-4): text-only experiments never use them."""
    dropped = []
    for parent in (model, getattr(model, "model", None)):
        if parent is None:
            continue
        for name in ("vision_tower", "audio_tower", "multi_modal_projector"):
            if getattr(parent, name, None) is not None:
                setattr(parent, name, None)
                dropped.append(name)
    if dropped:
        logger.info(f"Dropped unused multimodal submodules: {dropped}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def add_model_args(parser) -> None:
    """`--quant` / `--dtype` flags shared by the experiment scripts."""
    parser.add_argument("--quant", choices=["8", "4", "none"], default="8", help="Base-model quantization (paper: 8)")
    parser.add_argument("--dtype", default="auto", help="dtype of non-quantized weights (default: the model's own)")


def load_model_from_args(model_name: str, args):
    return load_model(model_name, quant=None if args.quant == "none" else args.quant, dtype=args.dtype)


def text_config(model):
    """Gemma-4 nests the language-model config under `text_config`; Qwen3 does not."""
    config = model.config
    if hasattr(config, "get_text_config"):
        return config.get_text_config()
    return getattr(config, "text_config", config)


def num_hidden_layers(model) -> int:
    return text_config(model).num_hidden_layers


def model_name_of(model) -> str:
    name = getattr(model, "name_or_path", None) or getattr(getattr(model, "config", None), "_name_or_path", None)
    assert name, "Could not determine the model's name"
    return name


def disable_kv_cache(model) -> None:
    """For training: turn the KV cache off on both flat (Qwen3) and nested (Gemma-4) configs."""
    model.config.use_cache = False
    nested = getattr(model.config, "text_config", None)
    if nested is not None:
        nested.use_cache = False
