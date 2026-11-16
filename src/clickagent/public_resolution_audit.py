"""CPU geometry ceiling of the frozen public-transfer patch-center decoder.

No VLM weights, attention features, gates, or learned predictions are loaded.
The official installed Qwen smart_resize computes the native grid from original
image dimensions; the existing public evaluator tests its *integer* pixel
centers against the complete normalized target geometry. This ceiling concerns
only this fixed grid-center decoder, not VLM reasoning or coordinate generation.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
from importlib.metadata import version
import inspect
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .public_evaluation import _load_manifest, _polygon_area, _read_jsonl, _validate_geometry, quantization_oracle


AUDIT_VERSION = 2
DEFAULT_MODEL_PATH = Path("models/Qwen3-VL-8B-Instruct")
MIN_PIXELS = 16384
MAX_PIXELS = 262144


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_resize_config(model_path: str | Path = DEFAULT_MODEL_PATH) -> dict[str, Any]:
    """Read actual local processor/model files and make runtime overrides clear."""
    root = Path(model_path).resolve()
    processor_path, model_config_path = root / "preprocessor_config.json", root / "config.json"
    processor = json.loads(processor_path.read_text(encoding="utf-8"))
    model = json.loads(model_config_path.read_text(encoding="utf-8"))
    vision = model["vision_config"]
    patch, merge = processor["patch_size"], processor["merge_size"]
    if model.get("model_type") != "qwen3_vl" or (patch, merge) != (16, 2):
        raise ValueError("audit requires the local Qwen3-VL processor with patch16/merge2")
    if (vision["patch_size"], vision["spatial_merge_size"]) != (patch, merge):
        raise ValueError("model and processor patch/merge configurations disagree")
    if processor.get("temporal_patch_size") != vision.get("temporal_patch_size"):
        raise ValueError("model and processor temporal patch sizes disagree")
    return {"model_path": str(root), "model_type": model["model_type"],
            "image_processor_type": processor.get("image_processor_type"),
            "processor_class": processor.get("processor_class"),
            "patch_size": patch, "merge_size": merge, "factor": patch * merge,
            "temporal_patch_size": processor["temporal_patch_size"],
            "original_processor_size": processor.get("size"),
            "runtime_min_pixels_override": MIN_PIXELS, "runtime_max_pixels_override": MAX_PIXELS,
            "effective_size": {"shortest_edge": MIN_PIXELS, "longest_edge": MAX_PIXELS},
            "config_files": [{"path": str(p), "sha256": _hash(p)} for p in (processor_path, model_config_path)]}


def native_grid(image_size: Sequence[int], resize_config: dict[str, Any]) -> dict[str, Any]:
    """Return official resized [w,h] and native post-merger [grid_h,grid_w]."""
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

    if (not isinstance(image_size, (list, tuple)) or len(image_size) != 2 or
            any(isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1 for v in image_size)):
        raise ValueError("image_size must be positive integer [width,height]")
    width, height = map(int, image_size)
    factor = resize_config["factor"]
    rh, rw = smart_resize(height, width, factor=factor,
                          min_pixels=resize_config["runtime_min_pixels_override"],
                          max_pixels=resize_config["runtime_max_pixels_override"])
    if min(rh, rw) < factor or rh % factor or rw % factor:
        raise ValueError("official resize did not produce a nonempty merged patch grid")
    return {"resized_image_size": [rw, rh], "grid_shape": [rh // factor, rw // factor],
            "image_tokens": (rh // factor) * (rw // factor),
            "original_pixel_grid_spacing": [width / (rw // factor), height / (rh // factor)]}


def target_pixel_dimensions(geometry: dict[str, Any], image_size: Sequence[int]) -> dict[str, Any]:
    """Keep all components; representative dimensions use the largest-area one.

    Polygon width/height refer to its enclosing box; its area uses the polygon,
    not that enclosing box. Multiple components remain a union for the oracle.
    """
    width, height = image_size
    components = []
    if geometry["type"] == "bbox":
        for x0, y0, x1, y1 in geometry["boxes"]:
            components.append({"normalized_bounds": [x0, y0, x1, y1],
                               "width_px": (x1 - x0) * width, "height_px": (y1 - y0) * height,
                               "area_fraction": (x1 - x0) * (y1 - y0)})
    else:
        for polygon in geometry["polygons"]:
            xs, ys = zip(*polygon)
            components.append({"normalized_bounds": [min(xs), min(ys), max(xs), max(ys)],
                               "width_px": (max(xs) - min(xs)) * width,
                               "height_px": (max(ys) - min(ys)) * height,
                               "area_fraction": _polygon_area(polygon)})
    largest = max(components, key=lambda c: c["area_fraction"])
    for component in components:
        component["area_px2"] = component["area_fraction"] * width * height
    shortest = min(largest["width_px"], largest["height_px"])
    edge_bin = "lt8" if shortest < 8 else "8_to_lt16" if shortest < 16 else "16_to_lt32" if shortest < 32 else "32_to_lt64" if shortest < 64 else "ge64"
    return {"components": components, "representative_width_px": largest["width_px"],
            "representative_height_px": largest["height_px"], "representative_short_edge_px": shortest,
            "representative_area_px2": largest["area_px2"], "short_edge_bin_px": edge_bin}


def _visible_geometry(original: Any) -> tuple[Any, bool, str | None, bool]:
    """Intersect bbox unions with the visible image without editing sources.

    Every proposed decoder point lies inside the image, so clipping a positive
    box to the image cannot change its point-membership result. Malformed or
    out-of-bounds polygons are retained as invalid, without inventing geometry.
    The final bool identifies a proven-empty bbox/image intersection.
    """
    if original is None:
        return None, False, None, False
    if not isinstance(original, dict) or original.get("type") != "bbox":
        try:
            _validate_geometry(original)
        except ValueError as exc:
            return None, False, f"polygon_or_geometry_invalid_without_clipping: {exc}", False
        return original, False, None, False
    boxes = original.get("boxes")
    if not isinstance(boxes, list) or not boxes:
        return None, False, "bbox_requires_nonempty_boxes", False
    visible, clipped, boundary_only = [], False, False
    for box in boxes:
        if (not isinstance(box, (tuple, list)) or len(box) != 4 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) for v in box)):
            return None, clipped, "bbox_coordinates_malformed_or_nonfinite", False
        x0, y0, x1, y1 = box
        if x0 >= x1 or y0 >= y1:
            return None, clipped, "bbox_has_nonpositive_area", False
        bounded = [max(0, min(1, v)) for v in box]
        clipped = clipped or bounded != list(box)
        if bounded[0] < bounded[2] and bounded[1] < bounded[3]:
            visible.append(bounded)
        elif max(0, x0) <= min(1, x1) and max(0, y0) <= min(1, y1):
            boundary_only = True
    if boundary_only:
        # Inclusive evaluator edges can still contain pixel zero on tiny
        # images. Do not falsely call a line/point intersection an empty set.
        return None, clipped, "bbox_image_intersection_has_degenerate_boundary", False
    if not visible:
        return None, clipped, "bbox_image_intersection_empty", True
    result = {"type": "bbox", "boxes": visible}
    _validate_geometry(result)
    return result, clipped, None, False


def _audit_case(raw: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    original = raw.get("target_geometry")
    geometry, clipped, error, empty = _visible_geometry(original)
    eligible = bool(raw.get("expected_actionable") is True and raw.get("task_kind") != "action_feasibility")
    if eligible and original is None:
        error = "positive_geometry_missing"
    clean = {**raw, "target_geometry": geometry}
    if error:
        # Validate all nongeometry schema fields using a placeholder solely for
        # that check. It is never passed to oracle or exported as effective GT.
        clean["target_geometry"] = {"type": "bbox", "boxes": [[0, 0, 1, 1]]}
    case = _load_manifest([clean])[0]
    if error:
        case.update({"target_geometry": None, "_target_size": "invalid_geometry", "_target_area": None})
    return case, {"original_target_geometry": original, "effective_target_geometry": geometry,
                  "geometry_clipped": clipped, "geometry_validation_error": error,
                  "geometry_empty_intersection": empty, "oracle_eligible": eligible}


def verify_processor_grids(candidates: list[tuple[dict[str, Any], Path]], resize_config: dict[str, Any],
                           count: int = 3) -> list[dict[str, Any]]:
    """CPU image preprocessing only; no tokenizer, language model or weights."""
    if count < 0:
        raise ValueError("processor verification count must be nonnegative")
    if not count:
        return []
    from PIL import Image
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(
        resize_config["model_path"], min_pixels=MIN_PIXELS, max_pixels=MAX_PIXELS, local_files_only=True)
    checks, seen_sizes = [], set()
    for record, manifest_path in candidates:
        size = tuple(record["image_size"])
        if size in seen_sizes:
            continue
        path = Path(record["image"])
        if not path.is_absolute():
            path = manifest_path.parent / path
        with Image.open(path) as image:
            if image.size != size:
                raise ValueError(f"{record['sample_id']}: manifest image_size disagrees with image header")
            batch = processor(images=[image.convert("RGB")], return_tensors="pt", device="cpu")
        grid = batch["image_grid_thw"][0].tolist()
        actual = [grid[1] // resize_config["merge_size"], grid[2] // resize_config["merge_size"]]
        expected = native_grid(size, resize_config)["grid_shape"]
        if grid[0] != 1 or actual != expected:
            raise RuntimeError(f"{record['sample_id']}: official processor grid {grid} differs from smart_resize {expected}")
        if batch["pixel_values"].device.type != "cpu":
            raise RuntimeError("audit processor verification must remain on CPU")
        checks.append({"sample_id": record["sample_id"], "image_size": list(size),
                       "processor_grid_thw": grid, "expected_postmerge_grid": expected,
                       "match": True, "device": "cpu"})
        seen_sizes.add(size)
        del batch
        if len(checks) == count:
            break
    return checks


def _distribution(values: list[float]) -> dict[str, float | None]:
    return {key: float(np.percentile(values, percentile)) if values else None
            for key, percentile in (("p10", 10), ("p50", 50), ("p90", 90), ("p95", 95))}


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [r for r in rows if r["oracle_eligible"]]
    scored = [r for r in rows if r["quantization_oracle"] is not None]
    reachable = sum(r["quantization_oracle"] for r in scored)
    unresolved = len(eligible) - len(scored)
    dimensions = [r for r in eligible if "target_pixels" in r]
    return {"input_rows": len(rows), "positive_geometry_denominator": len(eligible),
            "scored_positive_geometry_rows": len(scored), "unresolved_positive_geometry_rows": unresolved,
            "n_geometry_clipped": sum(r["geometry_clipped"] for r in rows),
            "n_invalid_positive_geometry": sum(bool(r["geometry_validation_error"]) for r in eligible),
            "excluded_reasons": dict(Counter(r["excluded_reason"] for r in rows if r["excluded_reason"])),
            "quantization_oracle": {"numerator": reachable, "denominator": len(eligible),
                                    "rate": reachable / len(eligible) if eligible else None},
            "quantization_oracle_upper_bound": {"numerator": reachable + unresolved, "denominator": len(eligible),
                                                "rate": (reachable + unresolved) / len(eligible) if eligible else None},
            "quantization_oracle_coverage": len(scored) / len(eligible) if eligible else None,
            "unreachable_positive_targets": len(scored) - reachable,
            "target_width_px": _distribution([r["target_pixels"]["representative_width_px"] for r in dimensions]),
            "target_height_px": _distribution([r["target_pixels"]["representative_height_px"] for r in dimensions]),
            "target_short_edge_px": _distribution([r["target_pixels"]["representative_short_edge_px"] for r in dimensions]),
            "image_tokens": _distribution([r["image_tokens"] for r in eligible]),
            "grid_shape_counts": dict(Counter(f"{r['grid_shape'][0]}x{r['grid_shape'][1]}" for r in eligible))}


def _slices(rows: list[dict[str, Any]], field: str) -> dict[str, Any]:
    groups = defaultdict(list)
    for row in rows:
        groups[row[field]].append(row)
    return {str(key): _aggregate(group) for key, group in sorted(groups.items())}


def audit_manifests(manifests: Sequence[str | Path], output_dir: str | Path,
                    model_path: str | Path = DEFAULT_MODEL_PATH, *, processor_checks: int = 3) -> dict[str, Any]:
    """Audit all supplied rows, without sampling, fitting, or model inference."""
    if not manifests:
        raise ValueError("at least one manifest is required")
    config = read_resize_config(model_path)
    provenance, rows, candidates, seen = [], [], [], set()
    for name in manifests:
        path = Path(name).resolve()
        provenance.append({"path": str(path), "sha256": _hash(path)})
        for raw in _read_jsonl(path):
            case, geometry_status = _audit_case(raw)
            sid = case["sample_id"]
            if sid in seen:
                raise ValueError(f"duplicate sample_id across manifests: {sid}")
            seen.add(sid)
            excluded = ("action_feasibility_no_geometry_protocol" if case["task_kind"] == "action_feasibility"
                        else "nonactionable_or_refusal" if not case["expected_actionable"] else None)
            row = {**geometry_status, "sample_id": sid, "dataset": case["dataset"], "domain": case["_domain"],
                   "task_kind": case["task_kind"], "expected_actionable": case["expected_actionable"],
                   "split": case.get("split"), "image_size": case.get("image_size"),
                   "target_size": case["_target_size"], "target_area_fraction": case["_target_area"],
                   "quantization_oracle": None, "excluded_reason": excluded, "short_edge_bin_px": "unavailable"}
            if not excluded:
                try:
                    row.update(native_grid(case.get("image_size"), config))
                except (ValueError, TypeError) as exc:
                    raise ValueError(f"{sid}: cannot compute native grid: {exc}") from exc
                if case["target_geometry"] is not None:
                    row["target_pixels"] = target_pixel_dimensions(case["target_geometry"], case["image_size"])
                    row["short_edge_bin_px"] = row["target_pixels"]["short_edge_bin_px"]
                    row["quantization_oracle"] = quantization_oracle(case["target_geometry"], row["grid_shape"], case["image_size"])
                elif row["geometry_empty_intersection"]:
                    row["quantization_oracle"] = False
                else:
                    row["excluded_reason"] = "invalid_geometry_retained_in_positive_denominator"
                candidates.append((case, path))
            rows.append(row)
    checks = verify_processor_grids(candidates, config, processor_checks)
    by_dataset = _slices(rows, "dataset")
    for dataset, entry in by_dataset.items():
        members = [r for r in rows if r["dataset"] == dataset]
        entry["by_domain"] = _slices(members, "domain")
        entry["by_target_size"] = _slices(members, "target_size")
        entry["by_short_edge_px"] = _slices(members, "short_edge_bin_px")
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize
    source_paths = [Path(__file__).resolve(), Path(__file__).with_name("public_evaluation.py").resolve(),
                    Path(__file__).with_name("action_grounding.py").resolve(), Path(inspect.getsourcefile(smart_resize))]
    examples = []
    seen_grids = set()
    for row in rows:
        if row["quantization_oracle"] is None:
            continue
        key = row["dataset"], tuple(row["image_size"]), tuple(row["grid_shape"])
        if key not in seen_grids:
            examples.append({k: row[k] for k in ("sample_id", "dataset", "image_size", "resized_image_size",
                                                 "grid_shape", "image_tokens", "original_pixel_grid_spacing")})
            seen_grids.add(key)
    summary = {"audit_version": AUDIT_VERSION, "status": "complete", "experiment_type": "fixed_grid_center_geometry_ceiling",
               "full_supplied_manifest": True, "sampling_or_model_result_selection": False,
               "model_weights_loaded": False, "gpu_used": False, "config": config,
               "manifests": provenance, "versions": {name: version(name) for name in ("transformers", "torch", "numpy")},
               "source_files": [{"path": str(p), "sha256": _hash(p)} for p in source_paths],
               "n_geometry_clipped": sum(r["geometry_clipped"] for r in rows),
               "geometry_clipped_sample_ids": [r["sample_id"] for r in rows if r["geometry_clipped"]],
               "invalid_positive_geometry_sample_ids": [r["sample_id"] for r in rows if r["oracle_eligible"] and r["geometry_validation_error"]],
               "overall": _aggregate(rows), "by_dataset": by_dataset, "by_domain": _slices(rows, "domain"),
               "by_target_size": _slices(rows, "target_size"), "by_short_edge_px": _slices(rows, "short_edge_bin_px"),
               "grid_examples": examples, "processor_checks_requested": processor_checks, "processor_checks": checks,
               "interpretation": [
                   "The oracle asks whether any actually outputtable rounded/clipped integer patch center lies inside the full target union.",
                   "This is a theoretical ceiling of this fixed grid-center decoder, not VLM understanding, attention quality, or unrestricted coordinate generation.",
                   "All positive geometry rows in supplied manifests are scored; no model predictions, training, thresholds, or sample results enter the audit.",
                   "ConflictGUI/action_feasibility and nonactionable/refusal rows do not define the positive point-location denominator.",
                   "Qwen3-VL's local processor uses patch16/merge2; runtime min/max pixel overrides differ from the checkpoint's raw default size.",
                   "Geometry follows the existing public evaluator: normalized multi-box/polygon union, inclusive boundaries and tolerance 1e-12.",
                   "For this audit only, bbox unions are intersected with the visible [0,1]^2 domain before strict validation. Original geometry and clipping flags remain in every row; source manifests are not modified.",
                   "Clipping an out-of-image bbox cannot change membership for predicted points restricted inside the image. A proven-empty bbox intersection is an unreachable target retained in the denominator.",
                   "Invalid or out-of-bounds polygons are reported without complex clipping. Unresolved invalid positive geometries remain in the full positive denominator: oracle is then a conservative lower bound with a separately reported upper bound and coverage.",
                   "Integer centers are np.rint((column+0.5)*original_width/grid_width), clipped to [0,width-1], and analogously for y.",
                   "Area bins use the largest component: small<=1%, medium<=10%, large>10%; polygon dimensions are enclosing-box dimensions.",
               ]}
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "per_sample.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8")
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    lines = ["# Fixed grid-center geometry ceiling", "", "This CPU-only audit is a decoder geometry ceiling, not VLM performance.", "",
             "| Dataset | Positive geometry targets | Reachable integer centers | Ceiling | Unreachable |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for dataset, entry in by_dataset.items():
        q = entry["quantization_oracle"]
        rate = f"{q['rate']:.2%}" if q["rate"] is not None else "unavailable"
        lines.append(f"| {dataset} | {q['denominator']} | {q['numerator']} | {rate} | {entry['unreachable_positive_targets']} |")
    lines += ["", "Patch16, merge2, official smart_resize, min_pixels16384/max_pixels262144. Full bbox/polygon unions are scored; no target-center shortcut.",
              "", "Per-domain, target-area, and target-short-edge slices are in summary.json; all row-level geometry diagnostics are in per_sample.jsonl.",
              "", "".join("- " + item + "\n" for item in summary["interpretation"])]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifests", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--processor-checks", type=int, default=3)
    args = parser.parse_args(argv)
    summary = audit_manifests(args.manifests, args.output, args.model_path, processor_checks=args.processor_checks)
    print(json.dumps({"output": str(args.output.resolve()),
                      "by_dataset": {name: row["quantization_oracle"] for name, row in summary["by_dataset"].items()}}, indent=2))


if __name__ == "__main__":
    main()
