"""Check all post-V32 run cohorts through V37 for paper-scale exposure.

Reads only frozen metadata/public lists and prediction IDs. This addendum keeps
the earlier V18 correction seal unchanged and performs no model inference.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
OUT = RUNS / "paper_scale_ablation_v1" / "data"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def write_once(path: Path, value) -> None:
    data = stable_bytes(value)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Refusing to replace frozen addendum: {path}")
    else:
        path.write_bytes(data)


def main() -> None:
    files = {
        "correction": OUT / "exposure_correction_v2.json",
        "v33_public": RUNS / "observation_regions_v33" / "data" / "public.json",
        "v33_predictions": RUNS / "observation_regions_v33" / "predictions.json",
        "v34_plans": RUNS / "train_scale_diagnostic_v34" / "plans.json",
        "v34_status": RUNS / "train_scale_diagnostic_v34" / "status.json",
        "v35_public": RUNS / "paired_scale_v35" / "data" / "public.json",
        "v35_predictions": RUNS / "paired_scale_v35" / "predictions.json",
        "v36_status": RUNS / "detail_context_v36" / "status.json",
        "v37_predictions": RUNS / "detail_context_v37" / "predictions.json",
        "v37_status": RUNS / "detail_context_v37" / "status.json",
        "old_train": RUNS / "multiregion_v13" / "data" / "train.jsonl",
        "full_train": RUNS / "full_scaling_v18" / "data" / "new_train.jsonl",
        "v32_history": RUNS / "independent_diagnostic_v32" / "historical_exclusion.json",
    }
    records = {name: read_json(path) for name, path in files.items() if path.suffix == ".json"}
    correction = records["correction"]
    assert correction["complete"]
    v33_public = records["v33_public"]
    v33_pred = records["v33_predictions"]["rows"]
    v34_plans = records["v34_plans"]
    v35_public = records["v35_public"]
    v35_pred = records["v35_predictions"]["rows"]
    v37_pred = records["v37_predictions"]["rows"]
    assert len(v33_public) == 227 and len(v33_pred) == 115
    assert len(v34_plans) == len(v35_public) == len(v35_pred) == len(v37_pred) == 112
    assert records["v34_status"].get("complete")
    assert records["v36_status"].get("phase") == "failed"
    assert records["v37_status"].get("phase") == "complete"
    ids = lambda rows: {r["sample_id"] for r in rows}
    train_ids = ids([r for r in v33_public if r.get("cohort") == "train"])
    dev_ids = ids([r for r in v33_public if r.get("cohort") == "development"])
    assert len(train_ids) == 112 and len(dev_ids) == 115 and not train_ids & dev_ids
    assert ids(v33_pred) == dev_ids
    assert ids(v34_plans) == ids(v35_public) == ids(v35_pred) == ids(v37_pred) == train_ids
    assert all(r.get("dataset") != "VenusBench-GD" for r in v33_public)
    train = [json.loads(s) for name in ("old_train", "full_train")
             for s in files[name].read_text(encoding="utf-8-sig").splitlines() if s.strip()]
    train_by_id = {r["sample_id"]: r for r in train}
    history_rgb = set(records["v32_history"]["historical_RGB"])
    image_rows = []
    for row in v33_public:
        if row.get("cohort") == "train":
            original = train_by_id[row["sample_id"]]
            assert original["image_sha256"] == row["image_sha256"]
            assert original["pixel_sha256"] in history_rgb
            image_rows.append({**row, "pixel_sha256": original["pixel_sha256"],
                               "dhash256": original["dhash256"]})
        else:
            assert "pixel_sha256" in row and "dhash256" in row
            image_rows.append(row)
    image_rgb = {r["pixel_sha256"] for r in image_rows}
    image_dhash = {int(r["dhash256"], 16) for r in image_rows}
    checks = {}
    for cohort_name, rows in correction["cohorts"].items():
        items = []
        for row in rows:
            dh = int(row["dhash256"], 16)
            nearest = min((dh ^ x).bit_count() for x in image_dhash)
            items.append({
                "sample_id": row["sample_id"],
                "historical_image_clean_after_v18": row["historical_image_clean_after_correction"],
                "v33_v37_same_sample_id": row["sample_id"] in ids(image_rows),
                "v33_v37_same_pixel_sha256": row["pixel_sha256"] in image_rgb,
                "v33_v37_min_dhash_distance": nearest,
                "v33_v37_near_le4": nearest <= 4,
            })
        assert all(not r["v33_v37_same_sample_id"] and not r["v33_v37_same_pixel_sha256"] and not r["v33_v37_near_le4"] for r in items)
        checks[cohort_name] = items
    output = {
        "complete": True,
        "scope": "V33,V34,V35,V36,V37; V34/V35/V37 use the same 112 training images and V33 also observes 115 diagnostic images. V36 failed before runtime construction.",
        "source_sha256": {name: sha256_file(path) for name, path in files.items()},
        "cohort_sizes": {"v33_public": 227, "v33_development_predictions": 115, "v33_train": 112, "v34_plans": 112, "v35_predictions": 112, "v36_forward": 0, "v37_predictions": 112},
        "distinct_post_v32_model_forward_ids": len(dev_ids | train_ids),
        "all_later_forward_ids_within_v33_public": True,
        "cohort_checks": checks,
        "all_residual_9_plus_3_still_exact_and_near_clean": True,
    }
    write_once(OUT / "post_v32_exposure_addendum_v1.json", output)
    print(json.dumps({"out": str(OUT / "post_v32_exposure_addendum_v1.json"), "cohort_sizes": output["cohort_sizes"], "all_residual_clean": True}), flush=True)


if __name__ == "__main__":
    main()
