"""Attribution similarity of faithful vs. unfaithful single-character adapters.

    python score_pairs.py --attributions-dir results/attributions --out results/pair_scores.json

For every adapter with both a decision-prompt and a self-report-prompt attribution file, computes `path_cos` (cosine
similarity of the concatenated per-site score vectors) and the per-layer summed scores, joins its faithfulness and
decision performance (from the `final-eval/metrics.json` stored with each adapter), and reports the group means and a
paired bootstrap 95% CI for the mean difference (faithful - unfaithful), as in the paper.

Backbone names are taken from the attribution file names; the defaults match the published deft-raptor checkpoints
(pass --unfaithful-backbone/--faithful-backbone for your own runs).
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from loguru import logger

from introspection.adapters import adapter_dir_metadata_file, resolve_adapter_path
from introspection.attribution import AttributionResult, attribution_similarity
from introspection.utils import save_json

UNFAITHFUL_BACKBONE, FAITHFUL_BACKBONE = "deft-raptor-1000", "deft-raptor-3000"


def faithfulness_of(metadata: dict) -> dict:
    """Decision performance and faithfulness from the adapter's `final-eval/metrics.json`, if available."""
    path = Path(metadata["adapter_path"])
    if not path.exists():  # attributions computed on another machine: fetch the adapter again
        path = resolve_adapter_path(metadata["adapter"])
    metrics_file = adapter_dir_metadata_file(path, "final-eval/metrics.json")
    if metrics_file is None:
        return {}
    metrics = json.loads(metrics_file.read_text())
    return {
        "faithfulness": metrics.get("novel/avg/behavior_report_corr"),
        "decision_performance": metrics.get("novel/avg/behavior_goal_corr"),
    }


def paired_bootstrap(diffs: np.ndarray, n_boot: int = 10_000, seed: int = 2025) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    boots = np.array([diffs[rng.integers(0, len(diffs), size=len(diffs))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return float(lo), float(hi)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--attributions-dir", type=Path, default=Path("results/attributions"))
    parser.add_argument("--out", type=Path, default=Path("results/pair_scores.json"))
    parser.add_argument("--unfaithful-backbone", default=UNFAITHFUL_BACKBONE)
    parser.add_argument("--faithful-backbone", default=FAITHFUL_BACKBONE)
    args = parser.parse_args()

    files = defaultdict(dict)
    for path in args.attributions_dir.glob("*.parquet"):
        character, backbone, prompt_type = path.stem.split("__")
        files[(character, backbone)][prompt_type] = path

    rows = []
    for (character, backbone), by_type in sorted(files.items()):
        if set(by_type) != {"choice", "report_prefix"}:
            logger.warning(f"Skipping {character} on {backbone}: missing {({'choice', 'report_prefix'} - set(by_type))}")
            continue
        choice, report = AttributionResult.load(by_type["choice"]), AttributionResult.load(by_type["report_prefix"])
        rows.append(
            {
                "character": character,
                "backbone": backbone,
                **attribution_similarity(choice, report),
                **faithfulness_of(choice.metadata),
            }
        )
    save_json(args.out, rows)
    logger.info(f"Scored {len(rows)} adapters -> {args.out}")

    by_char = defaultdict(dict)
    for row in rows:
        by_char[row["character"]][row["backbone"]] = row["path_cos"]
    paired = [(v[args.unfaithful_backbone], v[args.faithful_backbone]) for v in by_char.values()
              if args.unfaithful_backbone in v and args.faithful_backbone in v]
    if not paired:
        return
    unfaithful, faithful = map(np.array, zip(*paired))
    lo, hi = paired_bootstrap(faithful - unfaithful)
    print(f"{len(paired)} pairs")
    print(f"attribution similarity: faithful {faithful.mean():.3f} (SD {faithful.std(ddof=1):.3f}), "
          f"unfaithful {unfaithful.mean():.3f} (SD {unfaithful.std(ddof=1):.3f})")
    print(f"difference {np.mean(faithful - unfaithful):.3f}, paired bootstrap 95% CI [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
