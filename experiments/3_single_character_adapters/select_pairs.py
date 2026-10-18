"""Select data-matched (unfaithful, faithful) adapter pairs from single-character adapters trained on two backbones.

    python select_pairs.py --unfaithful-root runs/single-character/deft-raptor-1000 --faithful-root runs/single-character/deft-raptor-3000 --out results/pairs.json

A pair is the same character, preference seed and initialization seed on both backbones, such that (paper, Sec. 5.1):
  - both adapters parse at least 90% of their self-reports and reach decision performance of at least 0.9;
  - the adapter on the early backbone has faithfulness below 0.3 and the one on the late backbone above 0.9.
At most one pair is kept per character (the lowest initialization seed that qualifies).
"""

import argparse
import json
import re
from pathlib import Path

from loguru import logger

from introspection.utils import save_json

RUN_RE = re.compile(r"^(?P<character>.+)_pseed(?P<pseed>\d+)_iseed(?P<iseed>\d+)$")


def load_runs(root: Path) -> dict[tuple[str, int, int], dict]:
    runs = {}
    for metrics_path in sorted(root.glob("*/final-eval/metrics.json")):
        run_dir = metrics_path.parent.parent
        match = RUN_RE.match(run_dir.name)
        if match is None:
            continue
        metrics = json.loads(metrics_path.read_text())
        adapter_dirs = [p.parent for p in run_dir.glob("*/adapter_config.json")]
        assert len(adapter_dirs) == 1, f"Expected one adapter in {run_dir}, found {adapter_dirs}"
        key = (match["character"], int(match["pseed"]), int(match["iseed"]))
        runs[key] = {
            "adapter": str(adapter_dirs[0]),
            "faithfulness": metrics["novel/avg/behavior_report_corr"],
            "decision_performance": metrics["novel/avg/behavior_goal_corr"],
            "pct_parsed": metrics["novel/avg/pct_parsed"],
        }
    return runs


def select_pairs(
    early: dict[tuple[str, int, int], dict],
    late: dict[tuple[str, int, int], dict],
    *,
    max_unfaithful: float = 0.3,
    min_faithful: float = 0.9,
    min_decision: float = 0.9,
    min_parsed: float = 0.9,
) -> list[dict]:
    """Keep (character, pseed, iseed) runs present on both backbones that pass the filters; one per character."""

    def good(run):
        return run["pct_parsed"] >= min_parsed and run["decision_performance"] >= min_decision

    pairs, chosen = [], set()
    for key in sorted(early.keys() & late.keys()):
        character = key[0]
        e, l = early[key], late[key]
        if character in chosen or not (good(e) and good(l)):
            continue
        if e["faithfulness"] < max_unfaithful and l["faithfulness"] > min_faithful:
            pairs.append({"character": character, "preference_seed": key[1], "init_seed": key[2], "unfaithful": e, "faithful": l})
            chosen.add(character)
    return pairs


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unfaithful-root", type=Path, required=True, help="Runs on the early (ckpt 1000) backbone")
    parser.add_argument("--faithful-root", type=Path, required=True, help="Runs on the late (ckpt 3000) backbone")
    parser.add_argument("--max-unfaithful", type=float, default=0.3, help="Early-backbone faithfulness must be below this")
    parser.add_argument("--min-faithful", type=float, default=0.9, help="Late-backbone faithfulness must be above this")
    parser.add_argument("--min-decision", type=float, default=0.9, help="Both need at least this decision performance")
    parser.add_argument("--min-parsed", type=float, default=0.9, help="Both need at least this fraction of parsed reports")
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    early, late = load_runs(args.unfaithful_root), load_runs(args.faithful_root)
    logger.info(f"{len(early)} early-backbone and {len(late)} late-backbone adapters; {len(early.keys() & late.keys())} matched")
    pairs = select_pairs(
        early, late,
        max_unfaithful=args.max_unfaithful, min_faithful=args.min_faithful,
        min_decision=args.min_decision, min_parsed=args.min_parsed,
    )
    save_json(args.out, pairs)
    logger.info(f"Selected {len(pairs)} pairs -> {args.out}")


if __name__ == "__main__":
    main()
