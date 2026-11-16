"""Freeze all previously unseen, point-compatible VenusBench-GD cases.

Selection depends only on official records and historical image hashes, never on
model outputs. Public inputs and ground-truth labels are written separately.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

from prepare_paper_scale_test_v1 import (
    DEFAULT_OUT, PUBLIC_INDEX, ROOT, V32, first_near, path_key,
    read_json, read_jsonl, sha256_file, stable_bytes, write_once,
)


VENUS = ROOT / "data" / "public_v1" / "manifests" / "venusbench_gd.jsonl"
THREE_K_NEW = ROOT / "runs" / "quality_readout_v15_3k" / "data" / "new_train.jsonl"
FULL_NEW = ROOT / "runs" / "full_scaling_v18" / "data" / "new_train.jsonl"
OLD_TRAIN = ROOT / "runs" / "multiregion_v13" / "data" / "train.jsonl"
POST_V32 = ROOT / "runs" / "observation_regions_v33" / "data" / "public.json"


def main() -> None:
    manifest = read_jsonl(VENUS)
    index_doc = read_json(PUBLIC_INDEX)
    history = read_json(V32 / "historical_exclusion.json")
    diagnostic = read_json(V32 / "data" / "public.json")
    reserve = read_json(V32 / "data" / "reserve_public.json")
    post_v32 = read_json(POST_V32)
    assert len(manifest) == 6165
    assert index_doc["complete"] and history["complete"] and not history["unresolved"]
    manifest_index_hash = {
        key.replace("\\", "/"): value for key, value in index_doc["manifests"].items()
    }["data/public_v1/manifests/venusbench_gd.jsonl"]
    assert sha256_file(VENUS) == manifest_index_hash
    assert not any(row["sample_id"].startswith("venusbench_gd:") for row in post_v32)
    indexed = {path_key(row["image"]): row for row in index_doc["images"]}
    history_rgb = set(history["historical_RGB"])
    history_dhash = {int(value, 16) for value in history["historical_dhash256"]}
    v32_rgb = {row["pixel_sha256"] for row in diagnostic + reserve}
    v32_dhash = {int(row["dhash256"], 16) for row in diagnostic + reserve}

    old_train = read_jsonl(OLD_TRAIN)
    three_k = old_train + read_jsonl(THREE_K_NEW)
    full_train = old_train + read_jsonl(FULL_NEW)
    training_rgb = {row["pixel_sha256"] for row in full_train}
    training_dhash = {int(row["dhash256"], 16) for row in full_train}
    assert {row["pixel_sha256"] for row in three_k} <= training_rgb

    counts = Counter()
    public = []
    labels = {}
    selected_rgb: set[str] = set()
    selected_dhash: set[int] = set()
    for row in manifest:
        info = indexed[path_key(row["image"])]
        rgb = info["pixel_sha256"]
        dhash = int(info["dhash256"], 16)
        if rgb in history_rgb:
            counts["historical_exact"] += 1
            continue
        if first_near(dhash, history_dhash):
            counts["historical_near_le4"] += 1
            continue
        counts["history_clean_queries"] += 1
        if not row["expected_actionable"] or not row["target_geometry"] or not row["metadata"]["point_supported"]:
            counts["history_clean_nonpoint"] += 1
            continue
        counts["history_clean_point_queries"] += 1
        if rgb in v32_rgb or first_near(dhash, v32_dhash):
            counts["v32_pro_near_excluded"] += 1
            continue
        if rgb in selected_rgb or first_near(dhash, selected_dhash):
            counts["within_venus_exact_or_near_excluded"] += 1
            continue
        assert rgb not in training_rgb and not first_near(dhash, training_dhash)
        image = Path(row["image"])
        assert image.is_file()
        assert sha256_file(image) == info["file_sha256"], image
        assert row["image_size"] == info["image_size"]
        boxes = row["target_geometry"]["boxes"]
        assert boxes and all(len(box) == 4 and 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1 for box in boxes)
        public.append({
            "sample_id": row["sample_id"],
            "dataset": row["dataset"],
            "image": row["image"],
            "image_size": row["image_size"],
            "instruction": row["instruction"],
            "image_sha256": info["file_sha256"],
            "pixel_sha256": rgb,
            "dhash256": info["dhash256"],
            "application": row["metadata"].get("application") or "unspecified",
            "application_group": row["domain"],
            "ui_type": "unspecified",
        })
        labels[row["sample_id"]] = row["target_geometry"]
        selected_rgb.add(rgb)
        selected_dhash.add(dhash)
        counts["selected_point_queries"] += 1

    assert len(public) == len(labels) == len(selected_rgb) == 143
    assert counts["historical_exact"] == 5759
    assert counts["historical_near_le4"] == 57
    assert counts["history_clean_queries"] == 349
    assert counts["history_clean_point_queries"] == 184
    assert counts["history_clean_nonpoint"] == 165
    assert counts["within_venus_exact_or_near_excluded"] == 41
    assert counts["v32_pro_near_excluded"] == 0
    assert all((a ^ b).bit_count() > 4 for i, a in enumerate(sorted(selected_dhash)) for b in sorted(selected_dhash)[i + 1 :])

    source_files = [VENUS, PUBLIC_INDEX, V32 / "historical_exclusion.json", V32 / "data" / "public.json",
                    V32 / "data" / "reserve_public.json", POST_V32, OLD_TRAIN, THREE_K_NEW, FULL_NEW]
    audit = {
        "complete": True,
        "n": len(public),
        "cohort": "VenusBench-GD history-excluded point grounding; cross-source independent confirmation",
        "selection_rule": "Official manifest order; retain all valid actionable bbox/point rows after exact RGB and 256-bit dHash Hamming<=4 exclusion against 98 pre-V32 historical manifests, V32 observed/reserve and earlier kept Venus rows. No model-outcome-based sampling.",
        "selection_performed_before_new_model_forward": True,
        "label_policy": "Model workers receive venus_public.json only; score only after complete predictions are sealed, using venus_labels.json.",
        "history_rows": 93827,
        "history_unique_rgb": len(history_rgb),
        "history_unresolved": 0,
        "counts": dict(sorted(counts.items())),
        "selected_category_counts": dict(sorted(Counter(row["application_group"] for row in public).items())),
        "selected_platform_counts": dict(sorted(Counter(
            next(record for record in manifest if record["sample_id"] == row["sample_id"])["metadata"].get("platform", "unspecified")
            for row in public
        ).items())),
        "selected_distinct_rgb": len(selected_rgb),
        "selected_has_no_v32_pro_or_training_exact_near_overlap": True,
        "independent_history_audit_passed": True,
        "public_sha256": hashlib.sha256(stable_bytes(public)).hexdigest(),
        "labels_sha256": hashlib.sha256(stable_bytes(labels)).hexdigest(),
        "application_group_field_meaning": "VenusBench-GD official task category, not ScreenSpot-Pro application group.",
        "ui_type_field_meaning": "Unspecified by VenusBench-GD official metadata; not inferred from target label.",
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path) for path in source_files},
    }
    write_once(DEFAULT_OUT / "venus_public.json", public)
    write_once(DEFAULT_OUT / "venus_labels.json", labels)
    write_once(DEFAULT_OUT / "venus_audit.json", audit)
    print(json.dumps({"out": str(DEFAULT_OUT), "selected": len(public), "counts": audit["counts"],
                      "categories": audit["selected_category_counts"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
