"""Read-only, label-free audit of the sealed local PointerBench MVP crops.

The prior prediction records contain inference counters but not crop boxes.
Consequently, geometric claims below describe the reference attention branch;
the report explicitly does not assert that every individual crop used it.
"""
from __future__ import annotations

import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/pointer_mvp_same_cohort_v1"
OUT = ROOT / "runs/review_controls_v1/mvp_crop_diagnostic"
PUBLIC = ROOT / "runs/paper_scale_ablation_v2/pointerbench/public.jsonl"
SOURCE = ROOT / "third_party/MVP/mvp_sspro_qwen3vl.py"
ADAPTER = ROOT / "src/clickagent/advantage_runtimes.py"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def reference_crop_defaults() -> tuple[int, int]:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "get_top_attention_regions")
    values = {arg.arg: ast.literal_eval(default)
              for arg, default in zip(node.args.args[-len(node.args.defaults):],
                                      node.args.defaults)}
    return int(values["subimage_width"]), int(values["subimage_height"])


def main() -> None:
    bundle_path = RUN / "predictions/mvp.json"
    completion = read(RUN / "public_completion.json")
    protocol = read(RUN / "protocol.json")
    assert completion["complete"] and completion["labels_opened"] is False
    assert completion["prediction_sha256"]["mvp"] == sha(bundle_path)
    assert protocol["source_hashes"]["third_party\\MVP\\mvp_sspro_qwen3vl.py"] == sha(SOURCE)
    assert protocol["source_hashes"]["src\\clickagent\\advantage_runtimes.py"] == sha(ADAPTER)
    assert protocol["mvp"]["max_subimages"] == 2
    public = [json.loads(line) for line in PUBLIC.read_text(encoding="utf-8-sig").splitlines()
              if line.strip()]
    bundle = read(bundle_path)
    rows = bundle["rows"]
    assert len(public) == len(rows) == 1000
    assert bundle["n"] == 1000 and bundle["labels_opened"] is False
    assert all(p["sample_id"] == r["sample_id"] and p["suite"] == r["suite"]
               and p["image_size"] == [1024, 768] and r["labels_opened"] is False
               for p, r in zip(public, rows))
    for row in rows:
        case_path = RUN / "predictions/mvp/cases" / (hashlib.sha256(
            row["sample_id"].encode("utf-8")).hexdigest()[:24] + ".json")
        assert sha(case_path) == bundle["case_hashes"][row["sample_id"]]
        assert read(case_path) == row
    width, height = public[0]["image_size"]
    reference_width, reference_height = reference_crop_defaults()
    crop_width, crop_height = min(width, reference_width), min(height, reference_height)
    source_area = width * height
    crop_area = crop_width * crop_height
    max_vertical_shift = height - crop_height
    max_horizontal_shift = width - crop_width
    per_suite = {}
    for suite in ("pro", "sheets"):
        subset = [r for r in rows if r["suite"] == suite]
        assert len(subset) == 500
        per_suite[suite] = {
            "n": 500,
            "output_failures": dict(Counter(str(r["output_failure"]) for r in subset)),
            "processed_pixels": dict(Counter(str(r["processed_pixels"]) for r in subset)),
            "forward_calls": dict(Counter(str(r["backbone_forward_calls"]) for r in subset)),
            "mean_seconds": statistics.mean(r["elapsed_seconds"] for r in subset),
            "median_seconds": statistics.median(r["elapsed_seconds"] for r in subset),
        }
    result = {
        "complete": True,
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "label_access": False,
        "n": 1000,
        "source_hashes": {
            "public": sha(PUBLIC), "sealed_mvp_bundle": sha(bundle_path),
            "mvp_completion": sha(RUN / "public_completion.json"),
            "mvp_reference_source": sha(SOURCE), "local_adapter": sha(ADAPTER),
        },
        "native_image_size": [width, height],
        "reference_crop_size": [reference_width, reference_height],
        "effective_attention_branch_crop_size": [crop_width, crop_height],
        "crop_area_fraction": crop_area / source_area,
        "max_horizontal_shift_pixels": max_horizontal_shift,
        "max_vertical_shift_pixels": max_vertical_shift,
        "minimum_two_crop_intersection_fraction_of_image":
            (crop_width - max_horizontal_shift) * (crop_height - max_vertical_shift) / source_area,
        "full_image_source_view_repeated_in_reference": True,
        "subcrop_count_in_reference": 2,
        "actual_per_case_crop_boxes_recorded": False,
        "actual_per_case_fallback_frequency_identifiable": False,
        "per_suite": per_suite,
        "interpretation": "The official attention-branch crop covers 94.79% of each 1024x768 PointerBench image and can move only 40 pixels vertically. The archived predictions do not store crop boxes, so actual per-case fallback use and exact unique crop coordinates cannot be reconstructed without rerunning MVP.",
    }
    write(OUT / "diagnostic.json", result)
    print(json.dumps({"n": 1000, "crop_area_fraction": result["crop_area_fraction"],
                      "processed_pixels": {s: v["processed_pixels"] for s, v in per_suite.items()},
                      "diagnostic": str(OUT / "diagnostic.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
