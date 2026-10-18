# Identifying Introspection From the Inside

Code and data for **[Identifying Introspection From the Inside](https://arxiv.org/abs/2610.07186)**
(David I. Atkinson, Dillon Plunkett, David Bau; COLM 2026).
Project page: **[iii.baulab.info](https://iii.baulab.info)**.

This repository contains one Python package, `introspection` (in `src/`), and four experiment directories that use it:

| Task | Directory |
|---|---|
| Training a backbone (100 character) adapter on the decision task | [`experiments/1_multi_character_training`](experiments/1_multi_character_training) |
| Layer ablation experiments | [`experiments/2_layer_ablations`](experiments/2_layer_ablations) |
| Training Qwen3-14B on only its first (or last) *k* layers | [`experiments/1_multi_character_training`](experiments/1_multi_character_training) |
| Single-character ("lightweight") adapters on the unfaithful and faithful backbones | [`experiments/3_single_character_adapters`](experiments/3_single_character_adapters) |
| Attribution patching on decision and self-report prompts; attribution similarity | [`experiments/4_attribution_patching`](experiments/4_attribution_patching) |

## Setup

```bash
git clone https://github.com/diatkinson/identifying-introspection-from-the-inside
cd identifying-introspection-from-the-inside
uv sync            # or: pip install -e .
```

Python 3.12+. All of the paper's experiments load the base model in 8-bit (bitsandbytes, Linux + CUDA). Every script accepts `--quant none` to load the model in bfloat16
instead.

## Trained models

All adapters are on the Hugging Face Hub at
**[huggingface.co/datkinson/identifying-introspection](https://huggingface.co/datkinson/identifying-introspection)**,
one folder per adapter. Scripts accept a local directory or a spec such as
`hf://datkinson/identifying-introspection/deft-raptor/checkpoint-3000`, which they download on first use.

| Folder | Base model | Contents |
|---|---|---|
| `deft-raptor/checkpoint-<step>` | Qwen3-32B | The multi-character backbone (rank-8 LoRA, all layers, 100 characters), the 20 checkpoints from step 100 to 4000 that Figure 1b shows. **Checkpoint 1000** is the paper's unfaithful model and **checkpoint 3000** the faithful one. |
| `qwen3-14b-layer-freezing/first-<k>`, `.../from-<k>` | Qwen3-14B | Figure 4: adapters on layers `[0, k)`; Appendix E: on layers `[k, 40)`. |
| `single-character/<character>/deft-raptor-{1000,3000}` | Qwen3-32B | Figure 5: the 32 pairs of single-character adapters (rank 2). Each must be stacked on the deft-raptor checkpoint it was trained on (see `training_config.json`). |
| `gemma-4-31b/faithful-run/checkpoint-{469,1407}`, `gemma-4-31b/unfaithful-run/checkpoint-4688` | Gemma-4-31B-it | Appendix B.6: separately trained high- and low-faithfulness multi-character adapters. |

Every folder includes the adapter's `personas.json` (its characters and their target preference weights).

### Evaluate the paper's late checkpoint adapter on one of its characters

```python
from introspection.adapters import AdapterRef, load_adapter, set_adapters
from introspection.characters import load_personas
from introspection.evaluation import EvalConfig, calc_metrics, evaluate_persona
from introspection.hub import DEFT_RAPTOR_FAITHFUL
from introspection.models import get_tokenizer, load_model

# "deft-raptor" is the paper's multi-character adapter: a rank-8 LoRA on Qwen3-32B, trained on the decision task
# for 100 characters. Its checkpoint 1000 is the early, unfaithful model; checkpoint 3000 (used here) is the late,
# faithful one. DEFT_RAPTOR_FAITHFUL points to checkpoint 3000 on the Hugging Face Hub.
adapter = AdapterRef.from_spec(DEFT_RAPTOR_FAITHFUL, name="deft-raptor-3000")
model = load_model("Qwen/Qwen3-32B")  # 8-bit
model = load_adapter(model, adapter)
set_adapters(model, [adapter.name])

character = load_personas(adapter.path)[0]  # Al-Battani choosing loose-leaf tea
result = evaluate_persona(model, character, EvalConfig(adapters=(adapter.name,)), get_tokenizer("Qwen/Qwen3-32B"))
metrics = calc_metrics(result)
print(metrics["behavior_goal_corr"], metrics["behavior_report_corr"])  # decision performance, faithfulness
```

## Notes on the code

- **Characters** are called `Persona`s in the code, and adapter folders store them in `personas.json`. A character's
  name ends in a hash of its full definition (e.g. `loose-leaf-tea_Al-Battani_yWxGTjcn`), so names identify target
  weights exactly. [`data/characters/`](data/characters) holds the character sets used in the paper.
- **Metrics.** `behavior_goal_corr` is *decision performance* (corr(p̂, p)), `behavior_report_corr` is *faithfulness*
  (corr(p̂, p̃)), and `goal_report_corr` is corr(p, p̃). Metrics are computed per character and then averaged.
- **Behavioral preferences** p̂ come from a logistic regression on the model's P(A). That probability can be read from
  the raw logits or after the model's sampling warpers (temperature 0.6, top-k 20, top-p 0.95 for Qwen3);
  `EvalConfig.choice_probs` selects `"raw"` or `"processed"`, and both are saved.
- **Attribute order.** Training (and evaluation during training) lists a character's attributes in the order of
  [`data/candidate_scenarios.json`](data/candidate_scenarios.json). Scripts that load characters from an adapter's
  `personas.json` use the file's order. The trainers write that file with sorted keys, as the original runs did
  (except deft-raptor's), so reloaded layer-freezing, single-character and Gemma-4 characters list their attributes
  alphabetically. `attribute.py --attribute-order scenario` uses the training order instead.

## Citation

```bibtex
@inproceedings{atkinson2026identifying,
  title         = {Identifying Introspection From the Inside},
  author        = {Atkinson, David I. and Plunkett, Dillon and Bau, David},
  booktitle     = {Conference on Language Modeling (COLM)},
  year          = {2026},
  eprint        = {2610.07186},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2610.07186}
}
```

The decision and self-report tasks follow the setting of [Plunkett et al. (2025)](https://arxiv.org/abs/2505.17120).

## License

MIT (see [LICENSE](LICENSE)).
