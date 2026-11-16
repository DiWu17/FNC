"""Correct the paper-scale cohort exposure audit without changing frozen v1 files.

This is metadata-only: it reads manifests, public input lists and prediction IDs,
never model outputs, labels, or a model checkpoint. Re-running is deterministic and
refuses to replace a changed correction file.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs" / "paper_scale_ablation_v1" / "data"
V16 = ROOT / "runs" / "contribution_v16"
V18 = ROOT / "runs" / "mvp_confirmation_v18"
V32 = ROOT / "runs" / "independent_diagnostic_v32"
V33 = ROOT / "runs" / "observation_regions_v33"


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(s) for s in path.read_text(encoding="utf-8-sig").splitlines() if s.strip()]


def stable_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_once(path: Path, value) -> None:
    data = stable_bytes(value)
    if path.exists():
        if path.read_bytes() != data:
            raise RuntimeError(f"Refusing to change existing correction: {path}")
        return
    path.write_bytes(data)


def hamming(a: str, b: str) -> int:
    return (int(a, 16) ^ int(b, 16)).bit_count()


def by_pixel(rows: list[dict]) -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        result[row["pixel_sha256"]].append(row)
    return result


def compare(row: dict, archive: list[dict], predictions: set[str] | None = None) -> dict:
    sample_id = row["sample_id"]
    same_id = [r for r in archive if r["sample_id"] == sample_id]
    same_rgb = [r for r in archive if r["pixel_sha256"] == row["pixel_sha256"]]
    same_file = [r for r in archive if r["image_sha256"] == row["image_sha256"]]
    distances = [(hamming(row["dhash256"], r["dhash256"]), r["sample_id"]) for r in archive]
    near = sorted((d, sid) for d, sid in distances if d <= 4)
    return {
        "same_sample_id": bool(same_id),
        "same_id_same_image_sha256": bool(same_id) and all(r["image_sha256"] == row["image_sha256"] for r in same_id),
        "same_id_same_pixel_sha256": bool(same_id) and all(r["pixel_sha256"] == row["pixel_sha256"] for r in same_id),
        "same_id_same_instruction": bool(same_id) and all(r["instruction"] == row["instruction"] for r in same_id),
        "same_image_sha256": bool(same_file),
        "same_pixel_sha256": bool(same_rgb),
        "same_rgb_archive_sample_ids": sorted(r["sample_id"] for r in same_rgb),
        "same_rgb_instruction_equal": [r["instruction"] == row["instruction"] for r in sorted(same_rgb, key=lambda x: x["sample_id"])],
        "dhash_le4": bool(near),
        "dhash_le4_archive_sample_ids": [sid for _, sid in near],
        "min_dhash_distance": min((d for d, _ in distances), default=None),
        "same_rgb_archive_predictions_present": [r["sample_id"] in predictions for r in sorted(same_rgb, key=lambda x: x["sample_id"])] if predictions is not None else None,
    }


def main() -> None:
    sources = {
        "v1_full_public": OUT / "full_public.json",
        "v1_full_overlap": OUT / "full_history_overlap.json",
        "v1_reserve_public": OUT / "reserve_public.json",
        "v1_venus_public": OUT / "venus_public.json",
        "v16_heldout": V16 / "data" / "independent_test.jsonl",
        "v16_feature_index": V16 / "independent" / "index.json",
        "v18_heldout": V18 / "data" / "heldout.jsonl",
        "v18_ours_predictions": V18 / "predictions" / "ours.json",
        "v18_protocol": V18 / "protocol.json",
        "v32_protocol": V32 / "protocol.json",
        "v32_historical_exclusion": V32 / "historical_exclusion.json",
        "v32_diagnostic_public": V32 / "data" / "public.json",
        "v32_reserve_public": V32 / "data" / "reserve_public.json",
        "v33_public": V33 / "data" / "public.json",
        "v33_predictions": V33 / "predictions.json",
    }
    full = read_json(sources["v1_full_public"])
    old_full = read_json(sources["v1_full_overlap"])
    reserve = read_json(sources["v1_reserve_public"])
    venus = read_json(sources["v1_venus_public"])
    v16 = read_jsonl(sources["v16_heldout"])
    v16_index = read_json(sources["v16_feature_index"])
    v18 = read_jsonl(sources["v18_heldout"])
    v18_predictions = read_json(sources["v18_ours_predictions"])
    v32_history = read_json(sources["v32_historical_exclusion"])
    v32_diagnostic = read_json(sources["v32_diagnostic_public"])
    v32_reserve = read_json(sources["v32_reserve_public"])
    v33_public = read_json(sources["v33_public"])
    v33_predictions = read_json(sources["v33_predictions"])
    assert len(full) == len(old_full) == 1581
    assert len(reserve) == len(v32_reserve) == 58 and len(venus) == 143
    assert len(v16) == len(v16_index["cases"]) == 300
    assert len(v18) == len(v18_predictions["rows"]) == 280 and v18_predictions["complete"]
    assert len(v32_diagnostic) == len(v33_predictions["rows"]) == 115
    assert v32_history["complete"] and not v32_history["unresolved"]
    assert {r["sample_id"] for r in reserve} == {r["sample_id"] for r in v32_reserve}
    assert {r["sample_id"] for r in v32_diagnostic} == {r["sample_id"] for r in v33_predictions["rows"]}
    assert {r["sample_id"] for r in v16} == {r["sample_id"] for r in v16_index["cases"]}
    prediction_ids = {r["sample_id"] for r in v18_predictions["rows"]}
    assert prediction_ids == {r["sample_id"] for r in v18}

    history_rgb = set(v32_history["historical_RGB"])
    history_dhash = {int(x, 16) for x in v32_history["historical_dhash256"]}
    v18_rgb = by_pixel(v18)
    v18_dhash = {int(r["dhash256"], 16) for r in v18}
    diagnostic_ids = {r["sample_id"] for r in v32_diagnostic}
    diagnostic_rgb = {r["pixel_sha256"] for r in v32_diagnostic}
    diagnostic_dhash = {int(r["dhash256"], 16) for r in v32_diagnostic}
    reserve_ids = {r["sample_id"] for r in reserve}
    old_full_by_id = {r["sample_id"]: r for r in old_full}

    cohort_rows = {}
    counts = {}
    for name, rows in (("reserve58", reserve), ("venus143", venus)):
        checked = []
        for row in rows:
            dh = int(row["dhash256"], 16)
            pre_history_distance = min((dh ^ x).bit_count() for x in history_dhash)
            v16_comparison = compare(row, v16)
            v18_comparison = compare(row, v18, prediction_ids)
            v32_comparison = compare(row, v32_diagnostic)
            v33_comparison = compare(row, [r for r in v33_public if "pixel_sha256" in r and "dhash256" in r])
            clean = (
                row["pixel_sha256"] not in history_rgb
                and pre_history_distance > 4
                and not v16_comparison["dhash_le4"]
                and not v18_comparison["dhash_le4"]
                and not v32_comparison["dhash_le4"]
                and not v33_comparison["dhash_le4"]
            )
            checked.append({
                "sample_id": row["sample_id"],
                "dataset": row["dataset"],
                "instruction": row["instruction"],
                "image_sha256": row["image_sha256"],
                "pixel_sha256": row["pixel_sha256"],
                "dhash256": row["dhash256"],
                "pre_v32_history_exact_rgb": row["pixel_sha256"] in history_rgb,
                "pre_v32_history_min_dhash_distance": pre_history_distance,
                "v16": v16_comparison,
                "v18": v18_comparison,
                "v32_diagnostic": v32_comparison,
                "v33_public": v33_comparison,
                "historical_image_clean_after_correction": clean,
            })
        cohort_rows[name] = checked
        counts[name] = {
            "n": len(checked),
            "v16_same_id": sum(r["v16"]["same_sample_id"] for r in checked),
            "v16_same_rgb_or_near": sum(r["v16"]["dhash_le4"] for r in checked),
            "v18_same_id": sum(r["v18"]["same_sample_id"] for r in checked),
            "v18_same_id_same_image_rgb_instruction": sum(
                r["v18"]["same_sample_id"] and r["v18"]["same_id_same_image_sha256"]
                and r["v18"]["same_id_same_pixel_sha256"] and r["v18"]["same_id_same_instruction"]
                for r in checked
            ),
            "v18_same_file_sha256": sum(r["v18"]["same_image_sha256"] for r in checked),
            "v18_same_rgb": sum(r["v18"]["same_pixel_sha256"] for r in checked),
            "v18_same_rgb_different_id": sum(r["v18"]["same_pixel_sha256"] and not r["v18"]["same_sample_id"] for r in checked),
            "v18_dhash_le4": sum(r["v18"]["dhash_le4"] for r in checked),
            "v18_near_only": sum(r["v18"]["dhash_le4"] and not r["v18"]["same_pixel_sha256"] for r in checked),
            "historical_image_clean_after_correction": sum(r["historical_image_clean_after_correction"] for r in checked),
        }

    # The v1 full-list audit already verified train-image exclusion. Recompute
    # every historical stratum with the omitted V18 cohort inserted first.
    corrected_rows = []
    for row in full:
        old = old_full_by_id[row["sample_id"]]
        dh = int(row["dhash256"], 16)
        pixel = row["pixel_sha256"]
        pre_exact = pixel in history_rgb
        pre_near = False if pre_exact else any((dh ^ x).bit_count() <= 4 for x in history_dhash)
        v18_exact = pixel in v18_rgb
        v18_near = False if v18_exact else any((dh ^ x).bit_count() <= 4 for x in v18_dhash)
        diagnostic_exact = row["sample_id"] in diagnostic_ids or pixel in diagnostic_rgb
        diagnostic_near = False if diagnostic_exact else any((dh ^ x).bit_count() <= 4 for x in diagnostic_dhash)
        if pre_exact:
            status = "pre_v32_historical_exact"
        elif pre_near:
            status = "pre_v32_historical_near_le4"
        elif v18_exact:
            status = "v18_confirmation_observed_exact_rgb"
        elif v18_near:
            status = "v18_confirmation_observed_near_le4"
        elif diagnostic_exact:
            status = "v32_diagnostic_observed"
        elif diagnostic_near:
            status = "v32_diagnostic_near_other"
        elif row["sample_id"] in reserve_ids:
            status = "v32_reserve_previously_unforwarded"
        else:
            raise AssertionError(f"Unclassified full benchmark case: {row['sample_id']}")
        corrected_rows.append({
            "sample_id": row["sample_id"],
            "pixel_sha256": pixel,
            "history_status": status,
            "v1_history_status": old["history_status"],
            "v18_same_rgb": v18_exact,
            "v18_near_le4": v18_near,
            "trained_3k_exposure": old["trained_3k_exposure"],
            "trained_full_exposure": old["trained_full_exposure"],
        })
    assert all(r["trained_3k_exposure"] == r["trained_full_exposure"] == "none" for r in corrected_rows)

    source_sha256 = {name: sha256_file(path) for name, path in sources.items()}
    corrected_full = {
        "complete": True,
        "n": len(corrected_rows),
        "rows": corrected_rows,
        "rows_sha256": hashlib.sha256(stable_bytes(corrected_rows)).hexdigest(),
        "source_sha256": source_sha256,
        "history_status_counts": dict(sorted(Counter(r["history_status"] for r in corrected_rows).items())),
        "interpretation": "Full ScreenSpot-Pro 1581 includes previously observed images, including V18. Only the residual unforwarded reserve cases may support an independent check; do not call all 1581 independent.",
    }
    correction = {
        "complete": True,
        "finding": "The V32 prior-history set omitted the V18 MVP-confirmation heldout cohort, whose 280-case ours predictions had already been sealed before V32 registration. V32 reserve58 and the later Venus143 therefore are not independent cohorts as originally claimed.",
        "chronology": {
            "v18_protocol_registered_at_utc": read_json(sources["v18_protocol"])["registered_at_utc"],
            "v32_protocol_registered_at_utc": read_json(sources["v32_protocol"])["registered_at_utc"],
        },
        "sources_sha256": source_sha256,
        "v18_predictions_complete": True,
        "v18_predictions_n": len(prediction_ids),
        "v32_history_manifest_paths_contain_v18": any("mvp_confirmation_v18" in p.lower() for p in v32_history["history_manifest_paths"]),
        "cohort_counts": counts,
        "cohorts": cohort_rows,
        "corrected_full_history_status_counts": corrected_full["history_status_counts"],
        "corrected_full_rows_sha256": corrected_full["rows_sha256"],
        "protocol_implication": "Do not report reserve58 or Venus143 as independent confirmation. The exact-image-unseen residual sizes are small and must be reported separately. V18-exposed cases may serve development/retrospective comparison only.",
    }
    assert not correction["v32_history_manifest_paths_contain_v18"]
    assert counts["reserve58"]["v18_same_id"] == counts["reserve58"]["v18_same_rgb"] == 49
    assert counts["reserve58"]["historical_image_clean_after_correction"] == 9
    assert counts["venus143"]["v18_same_id"] == 123
    assert counts["venus143"]["v18_same_rgb"] == 140
    assert counts["venus143"]["historical_image_clean_after_correction"] == 3
    write_once(OUT / "full_history_overlap_corrected_v2.json", corrected_full)
    write_once(OUT / "exposure_correction_v2.json", correction)
    print(json.dumps({"cohort_counts": counts, "full_history_status_counts": corrected_full["history_status_counts"], "out": str(OUT)}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
