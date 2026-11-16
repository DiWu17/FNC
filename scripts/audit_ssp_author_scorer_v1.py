"""Postseal ScreenSpot-Pro author/V2 scorer sensitivity audit.

``register``, ``selftest`` and ``verify-gate`` never open benchmark labels.
``analyze`` opens only the ScreenSpot-Pro manifest after the registered
2,581-case 1M/archived-262k SHA gate has passed. This is a retrospective
scorer sensitivity check, not a replacement for the registered 1M analysis.
No inference, training, model selection or GPU work occurs in this script.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from clickagent.public_evaluation import point_in_geometry


MAIN_SCRIPT = ROOT / "scripts/analyze_high_pixel_full_inference_v1.py"
MAIN_REG = ROOT / "runs/high_pixel_full_inference_v1/analysis_registration.json"
LOCAL_SCORER = ROOT / "src/clickagent/public_evaluation.py"
OUT = ROOT / "runs/ssp_author_scorer_sensitivity_v1"
REG = OUT / "registration.json"
AUTHOR_SOURCE = (
    "https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding/"
    "blob/main/eval_screenspot_pro.py#L123-L138"
)
MODEL_NAMES = tuple(f"full_aligned_s{s}" for s in (1, 2, 3))
N = 1581


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_once(path: Path, obj: dict) -> None:
    serial = json.dumps(obj, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        require(path.read_text(encoding="utf-8") == serial,
                f"Existing immutable audit artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    require(not temp.exists(), f"Unresolved temporary audit artifact: {temp}")
    temp.write_text(serial, encoding="utf-8")
    temp.replace(path)


def load_main_module():
    spec = importlib.util.spec_from_file_location("high_pixel_registered_analysis", MAIN_SCRIPT)
    require(spec is not None and spec.loader is not None,
            "Cannot load registered 1M analysis script")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def register() -> None:
    """Freeze this separate posthoc audit before looking at SSP labels."""
    require(MAIN_REG.exists(), "Main 1M analysis registration is missing")
    main = read(MAIN_REG)
    record = {
        "complete": True,
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "retrospective_scorer_sensitivity_not_main_metric",
        "script_sha256": sha(Path(__file__)),
        "main_analysis_script_sha256": sha(MAIN_SCRIPT),
        "main_analysis_registration_sha256": sha(MAIN_REG),
        "local_scorer_sha256": sha(LOCAL_SCORER),
        "author_source": AUTHOR_SOURCE,
        "author_rule": "bbox xyxy / original img_size; inclusive bounds; no tolerance",
        "prediction_rule": "normalized point = sealed native pixel point / actual image_size",
        "cohort": "screenspot_pro_1581",
        "models": list(MODEL_NAMES),
        "pixel_caps_per_view": [262144, 1048576],
        "n": N,
        "main_expected_ssp_label_sha256_unopened":
            main["expected_label_sha256_unopened"]["ssp"],
        "must_pass_global_2581_sha_gate_before_any_label_access": True,
        "no_model_selection": True,
    }
    if REG.exists():
        record["registered_at_utc"] = read(REG).get("registered_at_utc")
    write_once(REG, record)
    print(json.dumps({"phase": "registered", "receipt": str(REG),
                      "sha256": sha(REG), "labels_opened": False}, ensure_ascii=False))


def author_hit(point, original: dict, actual_size) -> bool:
    """Exact positive-GT coordinate inequality in author eval_sample_positive_gt.

    Their response point is normalized. Our sealed response is a native-pixel
    point, so divide it by the actual image size before applying their rule.
    A null point is wrong_format in the author scorer and counts as a miss.
    """
    if point is None:
        return False
    box = original["bbox"]
    declared = original["img_size"]
    w, h = actual_size
    x, y = point[0] / w, point[1] / h
    return (box[0] / declared[0] <= x <= box[2] / declared[0] and
            box[1] / declared[1] <= y <= box[3] / declared[1])


def local_hit(point, geometry, actual_size) -> bool:
    if point is None:
        return False
    w, h = actual_size
    return bool(point_in_geometry((point[0] / w, point[1] / h), geometry))


def selftest() -> None:
    """Synthetic CPU checks only; no benchmark paths or labels are opened."""
    fixtures = [
        {"name": "inclusive_lower_and_upper", "box": [10, 10, 20, 20],
         "declared": [100, 100], "actual": [100, 100],
         "geometry": {"type": "bbox", "boxes": [[.1, .1, .2, .2]]},
         "point": [10, 20], "expected": [True, True]},
        {"name": "declared_size_mismatch", "box": [20, 10, 40, 20],
         "declared": [200, 100], "actual": [100, 100],
         "geometry": {"type": "bbox", "boxes": [[.2, .1, .4, .2]]},
         "point": [15, 15], "expected": [False, True]},
        {"name": "v2_boundary_tolerance", "box": [10, 10, 20, 20],
         "declared": [100, 100], "actual": [100, 100],
         "geometry": {"type": "bbox", "boxes": [[.1, .1, .2, .2]]},
         "point": [9.99999999995, 15], "expected": [True, False]},
        {"name": "visible_clip_same_for_valid_point", "box": [-10, 10, 10, 20],
         "declared": [100, 100], "actual": [100, 100],
         "geometry": {"type": "bbox", "boxes": [[0, .1, .1, .2]]},
         "point": [5, 15], "expected": [True, True]},
        {"name": "null_point", "box": [10, 10, 20, 20],
         "declared": [100, 100], "actual": [100, 100],
         "geometry": {"type": "bbox", "boxes": [[.1, .1, .2, .2]]},
         "point": None, "expected": [False, False]},
    ]
    for fixture in fixtures:
        observed = [
            local_hit(fixture["point"], fixture["geometry"], fixture["actual"]),
            author_hit(fixture["point"], {"bbox": fixture["box"],
                                           "img_size": fixture["declared"]},
                       fixture["actual"]),
        ]
        require(observed == fixture["expected"],
                f"Synthetic scorer fixture failed: {fixture['name']}: {observed}")
    receipt = {"complete": True, "script_sha256": sha(Path(__file__)),
               "fixtures": len(fixtures), "labels_opened": False,
               "gpu_used": False,
               "checks": [fixture["name"] for fixture in fixtures]}
    write_once(OUT / "selftest.json", receipt)
    print(json.dumps({"phase": "selftest_passed", "receipt": str(OUT / "selftest.json"),
                      "labels_opened": False}, ensure_ascii=False))


def verify_gate():
    """Return sealed public predictions; do not access any benchmark GT."""
    require(REG.exists(), "Audit registration is missing; run --stage register")
    record = read(REG)
    require(record.get("complete") is True and
            record.get("script_sha256") == sha(Path(__file__)) and
            record.get("main_analysis_script_sha256") == sha(MAIN_SCRIPT) and
            record.get("main_analysis_registration_sha256") == sha(MAIN_REG) and
            record.get("local_scorer_sha256") == sha(LOCAL_SCORER) and
            record.get("must_pass_global_2581_sha_gate_before_any_label_access") is True,
            "Audit registration/source drift")
    main = load_main_module()
    main_reg, public, new_cases, old_cases, _ = main.verify_gate()
    require(main_reg["expected_label_sha256_unopened"]["ssp"] ==
            record["main_expected_ssp_label_sha256_unopened"],
            "Main SSP annotation digest differs from audit registration")
    require(len(public["ssp"]) == N and len(new_cases["ssp"]) == N and
            len(old_cases["ssp"]) == N, "SSP sealed denominator changed")
    return record, main, main_reg, public["ssp"], new_cases["ssp"], old_cases["ssp"]


def analyze() -> None:
    record, main, main_reg, public, new_cases, old_cases = verify_gate()
    # This is the only benchmark-label access path. The global 2581 prediction
    # seal, every per-case SHA, and both model identities passed above first.
    require(sha(main.SSP_LABELS) == record["main_expected_ssp_label_sha256_unopened"],
            "SSP annotation digest differs from frozen preseal digest")
    labels = main.jsonl(main.SSP_LABELS)
    require(len(labels) == N, "SSP label denominator changed")
    by_id = {row["sample_id"]: row for row in labels}
    require(len(by_id) == N and set(by_id) == {r["sample_id"] for r in public},
            "SSP labels do not match sealed public sample IDs")

    tallies = {budget: {name: {"local_hits": 0, "author_hits": 0,
                                "disagreements": 0, "local_only": 0,
                                "author_only": 0, "null_points": 0}
                         for name in MODEL_NAMES}
               for budget in ("262144", "1048576")}
    mismatches = []
    dimension_mismatch_cases = 0
    clipped_cases = 0
    for public_row in public:
        sid = public_row["sample_id"]
        label = by_id[sid]
        original = label["metadata"]["original"]
        actual_size = public_row["image_size"]
        require(label["image_size"] == actual_size and
                label["metadata"]["declared_image_size"] == original["img_size"] and
                type(original["bbox"]) is list and len(original["bbox"]) == 4,
                "SSP author annotation geometry/schema changed: " + sid)
        dim_mismatch = original["img_size"] != actual_size
        clipped = label["metadata"].get("geometry_adjustment") is not None
        dimension_mismatch_cases += int(dim_mismatch)
        clipped_cases += int(clipped)
        for seed_index, name in enumerate(MODEL_NAMES):
            for budget, point in (("262144", old_cases[sid]["methods"][name]["point"]),
                                  ("1048576", new_cases[sid]["seed_points"][seed_index])):
                local = local_hit(point, label["target_geometry"], actual_size)
                author = author_hit(point, original, actual_size)
                tally = tallies[budget][name]
                tally["local_hits"] += int(local)
                tally["author_hits"] += int(author)
                tally["disagreements"] += int(local != author)
                tally["local_only"] += int(local and not author)
                tally["author_only"] += int(author and not local)
                tally["null_points"] += int(point is None)
                if local != author:
                    mismatches.append({
                        "sample_id": sid, "pixel_cap_per_view": int(budget),
                        "model": name, "point_native_pixels": point,
                        "v2_local_hit": local, "author_hit": author,
                        "declared_size_differs_from_actual": dim_mismatch,
                        "visible_box_clipped": clipped,
                        "actual_image_size": actual_size,
                        "declared_image_size": original["img_size"],
                        "author_raw_bbox_xyxy": original["bbox"],
                        "v2_target_geometry": label["target_geometry"],
                    })
    for by_seed in tallies.values():
        for tally in by_seed.values():
            tally["n"] = N
            tally["local_accuracy"] = tally["local_hits"] / N
            tally["author_accuracy"] = tally["author_hits"] / N
            tally["author_minus_local_pp"] = (
                tally["author_hits"] - tally["local_hits"]) * 100 / N
            require(tally["disagreements"] == tally["local_only"] +
                    tally["author_only"], "Disagreement tally inconsistent")
    output = {
        "complete": True,
        "posthoc_sensitivity_only": True,
        "no_model_selection": True,
        "main_registered_analysis_unchanged": True,
        "author_scorer_source": AUTHOR_SOURCE,
        "author_scorer_rule": record["author_rule"],
        "audit_registration_sha256": sha(REG),
        "main_analysis_registration_sha256": sha(MAIN_REG),
        "main_global_prediction_receipt_sha256": sha(main.GLOBAL_SEAL),
        "archived_full_prediction_receipt_sha256": sha(main.OLD_FULL_SEAL),
        "ssp_label_sha256_after_gate": sha(main.SSP_LABELS),
        "n": N,
        "dimension_mismatch_cases": dimension_mismatch_cases,
        "visible_box_clipped_cases": clipped_cases,
        "tallies": tallies,
        "mismatches": mismatches,
    }
    report = OUT / "results.json"
    write_once(report, output)
    receipt = {"complete": True, "report_sha256": sha(report),
               "registration_sha256": sha(REG),
               "main_global_prediction_receipt_sha256": sha(main.GLOBAL_SEAL),
               "labels_opened_after_global_gate": True,
               "n": N, "points_scored": N * len(MODEL_NAMES) * 2}
    write_once(OUT / "completion.json", receipt)
    print(json.dumps({"phase": "analyzed", "report": str(report),
                      "receipt": str(OUT / "completion.json"),
                      "tallies": tallies}, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=("register", "selftest", "verify-gate", "analyze"))
    args = parser.parse_args()
    if args.stage == "register":
        register()
    elif args.stage == "selftest":
        selftest()
    elif args.stage == "verify-gate":
        verify_gate()
        print(json.dumps({"phase": "gate_passed", "labels_opened": False}))
    else:
        analyze()


if __name__ == "__main__":
    main()
