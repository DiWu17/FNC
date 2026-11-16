"""Replay registered V33 statistics from sealed predictions; no inference."""
from collections import defaultdict
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path

import numpy as np


def main():
    out = Path(__file__).resolve().parents[1] / "runs/observation_regions_v33"
    read = lambda name: json.loads((out / name).read_text(encoding="utf-8"))
    assert read("final_completion.json")["complete"]
    rows = read("predictions.json")["rows"]
    result = read("results.json")
    assert len(rows) == result["n"] == 115
    assert len({row["sample_id"] for row in rows}) == 115
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["source"]].append(index)
    base = np.asarray([row["base_hits"] for row in rows], dtype=bool)
    assert base[:, 0].sum() == result["baseline_primary_hits"] == 38
    arm_counts = {}
    for arm, summary in result["arms"].items():
        extra = np.asarray([row["arms"][arm]["extra_hit"] for row in rows], dtype=bool)
        selected = np.asarray([row["arms"][arm]["consensus_hits"] for row in rows], dtype=bool)
        union = base | extra[:, None]
        np.testing.assert_array_equal(union, [row["arms"][arm]["union_hits"] for row in rows])
        assert int(extra.sum()) == summary["extra_hits"]
        assert int((extra & ~base[:, 0]).sum()) == summary["extra_repairs_available"]
        assert int(union[:, 0].sum()) == summary["baseline_plus_extra_oracle_hits"]
        assert selected.sum(axis=0).tolist() == list(summary["consensus_head_hits"].values())
        assert int((selected[:, 0] & ~base[:, 0]).sum()) == summary["consensus_primary_repairs"]
        assert int((~selected[:, 0] & base[:, 0]).sum()) == summary["consensus_primary_harms"]
        np.testing.assert_allclose(
            [100 * selected[:, :3].mean(), 100 * selected[:, 3:].mean()],
            list(summary["consensus_family_mean_percent"].values()), atol=1e-12, rtol=0,
        )
        arm_counts[arm] = {"extra_hits": int(extra.sum()), "all_six_union_hits": union.sum(axis=0).tolist(),
                           "all_six_consensus_hits": selected.sum(axis=0).tolist()}
    controls = ["name_fullpool", "nearby", "random_matched"]
    differences = np.asarray([
        [int(row["arms"]["coordinate_mass"]["union_hits"][0])
         - int(row["arms"][control]["union_hits"][0]) for control in controls]
        for row in rows
    ])
    rng = np.random.default_rng(20261133)
    draws = np.empty((10000, 3))
    for iteration in range(len(draws)):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups.values()])
        draws[iteration] = 100 * differences[indices].mean(axis=0)
    comparisons = {}
    for index, control in enumerate(controls):
        expected = result["primary_comparisons"]["coordinate-" + control]
        interval = np.quantile(draws[:, index], [.05 / 6, 1 - .05 / 6])
        delta = 100 * differences[:, index].mean()
        np.testing.assert_allclose(interval, expected["ci_pp"], atol=1e-12, rtol=0)
        np.testing.assert_allclose(delta, expected["delta_pp"], atol=1e-12, rtol=0)
        assert bool(interval[0] > 0) == expected["positive"]
        comparisons["coordinate-" + control] = {
            "delta_pp": float(delta), "ci98_333_pp": interval.tolist(),
            "wins": int((differences[:, index] > 0).sum()),
            "losses": int((differences[:, index] < 0).sum()),
        }
    audit = {
        "complete": True, "at_utc": datetime.now(timezone.utc).isoformat(),
        "n": 115, "bootstrap_draws": 10000, "bootstrap_seed": 20261133,
        "source_groups": {group: len(indices) for group, indices in groups.items()},
        "comparisons": comparisons, "arms": arm_counts,
        "inputs_sha256": {name: sha256((out / name).read_bytes()).hexdigest()
                          for name in ["predictions.json", "results.json", "final_completion.json"]},
    }
    (out / "statistics_audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
