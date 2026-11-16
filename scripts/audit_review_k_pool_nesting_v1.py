"""Label-free audit of whether archived C3 K pools are nested.

This posthoc diagnostic reads only the already sealed public prediction files.
It does not change the preregistered K-sweep predictions or analysis.
"""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "runs/review_controls_v1/diagnostics"


def main() -> None:
    result: dict[str, object] = {}
    for cohort in ("pro", "sheets", "mmbench", "full"):
        files = sorted((BASE / cohort / "cases").glob("*.json"))
        pairs = ((4, 8), (8, 16), (16, 32))
        stats = {f"{a}_to_{b}": {"nonempty": 0, "nested": 0,
                                  "mean_fraction_old_retained": 0.0,
                                  "mean_fraction_new_from_old": 0.0}
                 for a, b in pairs}
        for file in files:
            row = json.loads(file.read_text(encoding="utf-8"))
            if row.get("labels_opened") is not False:
                raise ValueError(f"Prediction gate violated: {file}")
            pools = {int(k): {tuple(p) for p in points}
                     for k, points in row["pool_points"].items()}
            for a, b in pairs:
                old, new = pools[a], pools[b]
                if not old:
                    continue
                s = stats[f"{a}_to_{b}"]
                s["nonempty"] += 1
                s["nested"] += int(old <= new)
                s["mean_fraction_old_retained"] += len(old & new) / len(old)
                s["mean_fraction_new_from_old"] += len(old & new) / len(new)
        for s in stats.values():
            if s["nonempty"]:
                s["mean_fraction_old_retained"] /= s["nonempty"]
                s["mean_fraction_new_from_old"] /= s["nonempty"]
        result[cohort] = {"n": len(files), "pairs": stats}
    output = BASE / "analysis/k_pool_nesting_posthoc.json"
    output.write_text(json.dumps({"label_free": True, "posthoc": True,
                                  "cohorts": result}, indent=2) + "\n", encoding="utf-8")
    print(output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
