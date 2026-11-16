"""Evaluate a balanced, groupwise random-label control of cached VLM features."""
import argparse
import json
from pathlib import Path

from clickagent.permutation_control import prepare_permutation_control
from clickagent.probes import run_probe_suite


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=4301)
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    args = parser.parse_args()
    manifest, features = prepare_permutation_control(args.manifest, args.features, args.output, args.seed)
    summary = run_probe_suite(features, args.output / "evaluation", manifest_path=manifest,
                              task="visual_state", seed=args.seed, min_common_coverage=1,
                              bootstrap_iterations=args.bootstrap_iterations)
    print(json.dumps({"negative_control": "permuted labels", "test": [
        {"condition": run["condition"], "features": run["feature_set"], **run["test"]}
        for run in summary["runs"]]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
