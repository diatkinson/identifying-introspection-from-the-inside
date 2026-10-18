"""Where the paper's trained adapters live on the Hugging Face Hub.

All adapters are in one model repository, one sub-folder per adapter (see the repository's model card). Any script
argument that takes an adapter accepts either a local directory or one of these `hf://` specs.
"""

import os

HF_NAMESPACE = os.environ.get("INTROSPECTION_HF_NAMESPACE", "datkinson")
HF_REPO = f"{HF_NAMESPACE}/identifying-introspection"


def hf_spec(path: str) -> str:
    return f"hf://{HF_REPO}/{path}"


# Multi-character backbone (Qwen3-32B, rank 8, all layers). Checkpoint 1000 is the unfaithful early model and
# checkpoint 3000 the faithful late one; the published checkpoints are the ones Figure 1b shows.
DEFT_RAPTOR = "deft-raptor"
DEFT_RAPTOR_UNFAITHFUL = hf_spec(f"{DEFT_RAPTOR}/checkpoint-1000")
DEFT_RAPTOR_FAITHFUL = hf_spec(f"{DEFT_RAPTOR}/checkpoint-3000")
DEFT_RAPTOR_STEPS = [100, 500, 700, 1000, 1300, *range(1500, 2600, 100), 2700, 2900, 3000, 4000]


def deft_raptor_checkpoint(step: int) -> str:
    return hf_spec(f"{DEFT_RAPTOR}/checkpoint-{step}")


# Qwen3-14B layer freezing (Figure 4 and Appendix E): `first-<k>` trains layers [0, k); `from-<k>` trains [k, 40).
def layer_freezing_adapter(kind: str, k: int) -> str:
    assert kind in ("first", "from")
    return hf_spec(f"qwen3-14b-layer-freezing/{kind}-{k}")


# Single-character adapters (Figure 5): 32 pairs. Each pair shares a character, data and seeds, and differs only in the
# backbone it was trained on top of.
def single_character_adapter(character: str, backbone_step: int) -> str:
    """`character` is the adapter name, e.g. `all-season-tires_Yunior_8cQpQgd4`; `backbone_step` is 1000 or 3000."""
    return hf_spec(f"single-character/{character}/deft-raptor-{backbone_step}")


# Gemma-4-31B multi-character adapters (Appendix B.6, Figure 8): one checkpoint of the run selected for low
# faithfulness, and two checkpoints of the run selected for high faithfulness.
GEMMA_UNFAITHFUL = hf_spec("gemma-4-31b/unfaithful-run/checkpoint-4688")
GEMMA_FAITHFUL = hf_spec("gemma-4-31b/faithful-run/checkpoint-1407")
GEMMA_FAITHFUL_EARLY = hf_spec("gemma-4-31b/faithful-run/checkpoint-469")
