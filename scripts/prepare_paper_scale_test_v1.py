"""Prepare label-separated ScreenSpot-Pro cohorts for the frozen 3k/full study.

This script reads only existing manifests and image bytes. It does not import or
invoke a model. Output is deterministic and refuses to overwrite changed files.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = ROOT / "runs" / "paper_scale_ablation_v1" / "data"
FULL_MANIFEST = ROOT / "data" / "public_v1" / "manifests" / "screenspot_pro.jsonl"
PUBLIC_INDEX = ROOT / "data" / "conditioned_v3" / "public_image_index.json"
V32 = ROOT / "runs" / "independent_diagnostic_v32"
OLD_TRAIN = ROOT / "runs" / "multiregion_v13" / "data" / "train.jsonl"
THREE_K_NEW = ROOT / "runs" / "quality_readout_v15_3k" / "data" / "new_train.jsonl"
FULL_NEW = ROOT / "runs" / "full_scaling_v18" / "data" / "new_train.jsonl"
POST_V32 = ROOT / "runs" / "observation_regions_v33" / "data" / "public.json"

PUBLIC_KEYS = (
    "sample_id", "dataset", "image", "image_size", "instruction",
    "image_sha256", "pixel_sha256", "dhash256", "application",
    "application_group", "ui_type",
)


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def write_once(path: Path, value) -> None:
    content = stable_bytes(value)
    if path.exists():
        if path.read_bytes() != content:
            raise RuntimeError(f"Refusing to replace a changed frozen output: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)


def path_key(path: str) -> str:
    return str(Path(path).resolve()).casefold()


def first_near(value: int, references: set[int], radius: int = 4) -> bool:
    return any((value ^ other).bit_count() <= radius for other in references)


def training_identity(rows: list[dict]) -> tuple[set[str], set[int]]:
    return {row["pixel_sha256"] for row in rows}, {int(row["dhash256"], 16) for row in rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.out.resolve()

    full = read_jsonl(FULL_MANIFEST)
    index_doc = read_json(PUBLIC_INDEX)
    reserve = read_json(V32 / "data" / "reserve_public.json")
    reserve_labels = read_json(V32 / "data" / "reserve_labels.json")
    diagnostic = read_json(V32 / "data" / "public.json")
    history = read_json(V32 / "historical_exclusion.json")
    history_resolution = read_json(V32 / "history_resolution_initial.json")
    old_train = read_jsonl(OLD_TRAIN)
    three_k_new = read_jsonl(THREE_K_NEW)
    full_new = read_jsonl(FULL_NEW)
    post_v32 = read_json(POST_V32)

    assert len(full) == len({row["sample_id"] for row in full}) == 1581
    assert index_doc["complete"] is True
    indexed_manifest_sha = {
        key.replace("\\", "/"): value for key, value in index_doc["manifests"].items()
    }["data/public_v1/manifests/screenspot_pro.jsonl"]
    assert sha256_file(FULL_MANIFEST) == indexed_manifest_sha
    index = {path_key(row["image"]): row for row in index_doc["images"]}
    assert len(reserve) == len(reserve_labels) == 58
    assert len(diagnostic) == 115
    assert len(old_train) == 873 and len(three_k_new) == 2100 and len(full_new) == 12500
    assert {row["sample_id"] for row in three_k_new} <= {row["sample_id"] for row in full_new}
    assert history["complete"] and not history["unresolved"]
    assert history_resolution["unresolved"] == []
    assert history_resolution["rows"] == 93827

    full_public: list[dict] = []
    full_labels: dict[str, dict] = {}
    index_by_id: dict[str, dict] = {}
    verified_bytes = 0
    for row in full:
        image = Path(row["image"])
        assert image.is_file(), image
        info = index[path_key(row["image"])]
        assert info["image_size"] == row["image_size"]
        file_sha = sha256_file(image)
        assert file_sha == info["file_sha256"], f"Image bytes changed: {image}"
        verified_bytes += image.stat().st_size
        assert row["metadata"]["point_supported"]
        boxes = row["target_geometry"]["boxes"]
        assert boxes and all(len(box) == 4 and 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1 for box in boxes)
        public = {
            "sample_id": row["sample_id"],
            "dataset": row["dataset"],
            "image": row["image"],
            "image_size": row["image_size"],
            "instruction": row["instruction"],
            "image_sha256": file_sha,
            "pixel_sha256": info["pixel_sha256"],
            "dhash256": info["dhash256"],
            "application": row["metadata"]["application"],
            "application_group": row["metadata"]["application_group"],
            "ui_type": row["metadata"]["ui_type"],
        }
        assert set(public) == set(PUBLIC_KEYS)
        full_public.append(public)
        index_by_id[public["sample_id"]] = public
        full_labels[public["sample_id"]] = row["target_geometry"]

    full_ids = set(index_by_id)
    reserve_ids = {row["sample_id"] for row in reserve}
    diagnostic_ids = {row["sample_id"] for row in diagnostic}
    assert len(reserve_ids) == 58 and len(diagnostic_ids) == 115
    assert reserve_ids <= full_ids and diagnostic_ids <= full_ids and reserve_ids.isdisjoint(diagnostic_ids)
    for row in reserve:
        assert row == index_by_id[row["sample_id"]], row["sample_id"]
        assert reserve_labels[row["sample_id"]] == full_labels[row["sample_id"]]
    assert {row["sample_id"] for row in post_v32 if row["sample_id"].startswith("screenspot_pro:")} == diagnostic_ids

    historical_rgb = set(history["historical_RGB"])
    historical_hashes = {int(value, 16) for value in history["historical_dhash256"]}
    diagnostic_rgb, diagnostic_hashes = training_identity(diagnostic)
    reserve_rgb, reserve_hashes = training_identity(reserve)
    three_k_rgb, three_k_hashes = training_identity(old_train + three_k_new)
    full_train_rgb, full_train_hashes = training_identity(old_train + full_new)
    assert three_k_rgb <= full_train_rgb

    per_sample: list[dict] = []
    categories = Counter()
    train_counts = Counter()
    for row in full_public:
        rgb = row["pixel_sha256"]
        dhash = int(row["dhash256"], 16)
        if rgb in historical_rgb:
            category = "pre_v32_historical_exact"
        elif first_near(dhash, historical_hashes):
            category = "pre_v32_historical_near_le4"
        elif row["sample_id"] in diagnostic_ids:
            category = "v32_diagnostic_observed"
        elif row["sample_id"] in reserve_ids:
            category = "v32_reserve_unforwarded_at_preparation"
        elif rgb in diagnostic_rgb or first_near(dhash, diagnostic_hashes):
            category = "v32_diagnostic_near_other"
        elif rgb in reserve_rgb or first_near(dhash, reserve_hashes):
            category = "v32_reserve_near_other"
        else:
            category = "other_history_clean"
        categories[category] += 1
        three_k_exposure = "exact" if rgb in three_k_rgb else "near_le4" if first_near(dhash, three_k_hashes) else "none"
        full_exposure = "exact" if rgb in full_train_rgb else "near_le4" if first_near(dhash, full_train_hashes) else "none"
        train_counts[("3k", three_k_exposure)] += 1
        train_counts[("full", full_exposure)] += 1
        per_sample.append({
            "sample_id": row["sample_id"],
            "pixel_sha256": rgb,
            "dhash256": row["dhash256"],
            "history_status": category,
            "trained_3k_exposure": three_k_exposure,
            "trained_full_exposure": full_exposure,
        })

    assert all(next(x for x in per_sample if x["sample_id"] == sample_id)["history_status"] == "v32_reserve_unforwarded_at_preparation" for sample_id in reserve_ids)
    assert all(next(x for x in per_sample if x["sample_id"] == sample_id)["trained_3k_exposure"] == "none" for sample_id in reserve_ids)
    assert all(next(x for x in per_sample if x["sample_id"] == sample_id)["trained_full_exposure"] == "none" for sample_id in reserve_ids)
    assert len({row["pixel_sha256"] for row in reserve}) == 58
    assert len({row["image_sha256"] for row in reserve}) == 58
    reserve_hash_list = [int(row["dhash256"], 16) for row in reserve]
    assert all((reserve_hash_list[i] ^ reserve_hash_list[j]).bit_count() > 4 for i in range(58) for j in range(i + 1, 58))

    sources = [FULL_MANIFEST, PUBLIC_INDEX, V32 / "data" / "reserve_public.json", V32 / "data" / "reserve_labels.json",
               V32 / "data" / "public.json", V32 / "historical_exclusion.json", V32 / "history_resolution_initial.json",
               OLD_TRAIN, THREE_K_NEW, FULL_NEW, POST_V32]
    audit = {
        "protocol": "Read-only preparation; no model forward or labels in public inputs. Run and seal all 58 reserve predictions before using full benchmark labels or inspecting full benchmark results.",
        "full_benchmark_is_not_independent": True,
        "full_benchmark_includes_reserve": True,
        "cohorts": {"official_full_queries": 1581, "reserve_queries": 58, "v32_observed_diagnostic_queries": 115},
        "history": {
            "pre_v32_manifest_count": len(history["history_manifest_paths"]),
            "pre_v32_rows": history_resolution["rows"],
            "pre_v32_unique_rgb": len(historical_rgb),
            "pre_v32_unique_dhash": len(historical_hashes),
            "pre_v32_unresolved": 0,
            "post_v32_pro_observed": len(diagnostic_ids),
            "near_threshold_hamming": 4,
            "full_history_status_counts": dict(sorted(categories.items())),
        },
        "training": {
            "old_queries": len(old_train),
            "new_3k_queries": len(three_k_new),
            "new_full_queries": len(full_new),
            "three_k_total_queries": len(old_train) + len(three_k_new),
            "full_total_queries": len(old_train) + len(full_new),
            "three_k_unique_rgb": len(three_k_rgb),
            "full_unique_rgb": len(full_train_rgb),
            "three_k_subset_of_full_by_sample_id": True,
            "official_full_exposure_counts": {
                arm: {kind: train_counts[(arm, kind)] for kind in ("exact", "near_le4", "none")}
                for arm in ("3k", "full")
            },
            "reserve_no_exact_or_near_training_overlap": True,
        },
        "integrity": {
            "all_official_image_file_sha256_verified": True,
            "verified_image_bytes": verified_bytes,
            "index_complete_and_manifest_sha256_matches": True,
            "all_official_bboxes_valid": True,
            "reserve_public_matches_official_index": True,
            "reserve_labels_match_official": True,
            "reserve_mutually_exact_and_near_distinct": True,
        },
        "source_sha256": {str(path.relative_to(ROOT)).replace("\\", "/"): sha256_file(path) for path in sources},
    }
    write_once(out / "reserve_public.json", reserve)
    write_once(out / "full_public.json", full_public)
    write_once(out / "reserve_labels.json", reserve_labels)
    write_once(out / "full_labels.json", full_labels)
    write_once(out / "full_history_overlap.json", per_sample)
    write_once(out / "audit.json", audit)
    print(json.dumps({"out": str(out), "full": len(full_public), "reserve": len(reserve), "history_counts": audit["history"]["full_history_status_counts"], "training_overlap": audit["training"]["official_full_exposure_counts"]}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
