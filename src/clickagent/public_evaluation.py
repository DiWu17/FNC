"""CPU-only evaluation of frozen policies on public screenshot benchmarks.

No model is imported, fitted, calibrated or executed here.  Point correctness
on a static screenshot is deliberately distinct from observed UI execution.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np


PUBLIC_METHODS = ("prefill_hidden", "post_hidden", "text_decision_shared", "text_coordinates")
TASK_KINDS = ("grounding", "refusal", "action_feasibility")
GEOMETRY_TOLERANCE = 1e-12


def _finite(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(float(value)) else None


def _pair(value: Any) -> tuple[float, float] | None:
    if isinstance(value, dict):
        value = (value.get("x"), value.get("y"))
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    a, b = (_finite(x) for x in value)
    return (a, b) if a is not None and b is not None else None


def _size(value: Any) -> tuple[int, int] | None:
    pair = _pair(value)
    if pair is None or any(x <= 0 or not x.is_integer() for x in pair):
        return None
    return int(pair[0]), int(pair[1])


def _validate_geometry(geometry: Any) -> dict[str, Any] | None:
    if geometry is None:
        return None
    if not isinstance(geometry, dict) or geometry.get("type") not in ("bbox", "polygon"):
        raise ValueError("target_geometry must be null, bbox, or polygon")
    if geometry["type"] == "bbox":
        boxes = geometry.get("boxes")
        if not isinstance(boxes, list) or not boxes:
            raise ValueError("bbox geometry requires at least one box")
        for box in boxes:
            if not isinstance(box, (tuple, list)) or len(box) != 4:
                raise ValueError("boxes must be normalized [x0,y0,x1,y1]")
            if any(_finite(x) is None or not 0 <= x <= 1 for x in box):
                raise ValueError("box coordinates must be finite and normalized to [0,1]")
            if box[0] >= box[2] or box[1] >= box[3]:
                raise ValueError("box must have positive width and height")
    else:
        polygons = geometry.get("polygons")
        if not isinstance(polygons, list) or not polygons:
            raise ValueError("polygon geometry requires at least one polygon")
        for polygon in polygons:
            if not isinstance(polygon, (tuple, list)) or len(polygon) < 3:
                raise ValueError("a polygon requires at least three vertices")
            if any(_pair(p) is None or any(not 0 <= x <= 1 for x in p) for p in polygon):
                raise ValueError("polygon vertices must be normalized to [0,1]")
            if _polygon_area(polygon) <= 0:
                raise ValueError("polygon must have positive area")
    return geometry


def _polygon_area(polygon: Sequence[Sequence[float]]) -> float:
    return abs(sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(polygon, (*polygon[1:], polygon[0])))) / 2


def _point_in_polygon(point: tuple[float, float], polygon: Sequence[Sequence[float]]) -> bool:
    x, y = point
    inside = False
    for a, b in zip(polygon, (*polygon[1:], polygon[0])):
        ax, ay = a
        bx, by = b
        cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
        if abs(cross) <= GEOMETRY_TOLERANCE and min(ax, bx) - GEOMETRY_TOLERANCE <= x <= max(ax, bx) + GEOMETRY_TOLERANCE and min(ay, by) - GEOMETRY_TOLERANCE <= y <= max(ay, by) + GEOMETRY_TOLERANCE:
            return True
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            inside = not inside
    return inside


def point_in_geometry(point: Sequence[float], geometry: dict[str, Any] | None) -> bool:
    """Inclusive normalized point-in-box/union-of-polygons, including edges.

    This documented evaluator convention is not a claim that every author's
    scorer uses identical floating-point tolerances or edge inequalities.
    """
    xy = _pair(point)
    if xy is None or geometry is None or any(x < 0 or x > 1 for x in xy):
        return False
    x, y = xy
    if geometry["type"] == "bbox":
        return any(x0 - GEOMETRY_TOLERANCE <= x <= x1 + GEOMETRY_TOLERANCE and y0 - GEOMETRY_TOLERANCE <= y <= y1 + GEOMETRY_TOLERANCE for x0, y0, x1, y1 in geometry["boxes"])
    return any(_point_in_polygon(xy, polygon) for polygon in geometry["polygons"])


def _pixel_point(point: Any, image_size: tuple[int, int] | None) -> tuple[float, float] | None:
    xy = _pair(point)
    if xy is None or image_size is None:
        return None
    w, h = image_size
    return (xy[0] / w, xy[1] / h) if 0 <= xy[0] < w and 0 <= xy[1] < h else None


def quantization_oracle(geometry: dict[str, Any] | None, grid_shape: Sequence[int], model_image_size: Sequence[int]) -> bool:
    """Whether any *actual grounder* rounded/clipped pixel patch center hits GT."""
    grid, size = _size(grid_shape), _size(model_image_size)
    if grid is None or size is None:
        raise ValueError("oracle requires positive integer [grid_h,grid_w] and [image_w,image_h]")
    gh, gw = grid
    w, h = size
    xs = np.clip(np.rint((np.arange(gw) + 0.5) * w / gw), 0, w - 1)
    ys = np.clip(np.rint((np.arange(gh) + 0.5) * h / gh), 0, h - 1)
    return any(point_in_geometry((x / w, y / h), geometry) for y in ys for x in xs)


def _largest_area(geometry: dict[str, Any] | None) -> float | None:
    if geometry is None:
        return None
    if geometry["type"] == "bbox":
        return max((x1 - x0) * (y1 - y0) for x0, y0, x1, y1 in geometry["boxes"])
    return max(_polygon_area(polygon) for polygon in geometry["polygons"])


def _load_manifest(manifest: Iterable[dict[str, Any]] | str | Path) -> list[dict[str, Any]]:
    if isinstance(manifest, (str, Path)):
        manifest = _read_jsonl(Path(manifest))
    result, seen = [], set()
    for raw in manifest:
        row = dict(raw)
        sid = row.get("sample_id")
        if not isinstance(sid, str) or not sid or sid in seen:
            raise ValueError("manifest sample_id must be nonempty and globally unique")
        seen.add(sid)
        if not isinstance(row.get("expected_actionable"), bool):
            raise ValueError(f"{sid}: expected_actionable must be a boolean")
        if row.get("task_kind") not in TASK_KINDS or not isinstance(row.get("dataset"), str) or not row["dataset"]:
            raise ValueError(f"{sid}: dataset and valid task_kind are required")
        if not isinstance(row.get("instruction"), str):
            raise ValueError(f"{sid}: instruction must be a string")
        row["target_geometry"] = _validate_geometry(row.get("target_geometry"))
        if row["expected_actionable"] and row["task_kind"] == "grounding" and row["target_geometry"] is None:
            raise ValueError(f"{sid}: positive grounding sample requires target_geometry")
        metadata = row.get("metadata") or {}
        if not isinstance(metadata, dict):
            raise ValueError(f"{sid}: metadata must be a dictionary")
        # Prefix by dataset: coincident image/group names in different sources
        # must not merge independent bootstrap units.
        group = row.get("group_id") or row.get("image_id") or sid
        row["_group"] = f"{row['dataset']}::{group}"
        row["_group_source"] = "group_id" if row.get("group_id") else "image_id" if row.get("image_id") else "sample_id"
        row["_domain"] = str(row.get("domain", metadata.get("domain", "unspecified")))
        area = _largest_area(row["target_geometry"])
        row["_target_area"] = area
        row["_target_size"] = "unavailable" if area is None else "small" if area <= 0.01 else "medium" if area <= 0.1 else "large"
        result.append(row)
    if not result:
        raise ValueError("manifest must not be empty")
    return result


def _normalize(raw: dict[str, Any], case: dict[str, Any], index: int,
               oracle_cache: dict[tuple[Any, ...], bool] | None = None) -> dict[str, Any]:
    issues = []
    for key in ("dataset", "instruction", "expected_actionable", "task_kind", "target_geometry"):
        if key in raw and raw[key] != case[key]:
            issues.append(f"raw {key} disagrees with authoritative manifest")
    action = raw.get("action")
    action_valid = isinstance(action, dict) and action.get("type") in ("click", "wait")
    if not action_valid:
        issues.append("missing or invalid action dictionary")
        action = {"type": "wait"}
    emitted = action["type"] == "click"
    image_size = _size(raw.get("model_image_size"))
    point = _pixel_point(action, image_size) if emitted else None
    if emitted and point is None and case["task_kind"] != "action_feasibility":
        issues.append("click point is missing, out of image, or has no valid model_image_size")
    parse_valid = raw.get("parse_valid") if isinstance(raw.get("parse_valid"), bool) else None
    cap = any(raw.get(key) is True for key in ("cap_reached", "token_cap_reached", "truncated", "hit_token_cap")) or raw.get("termination_reason") in ("token_cap", "max_new_tokens")
    recorded_error = bool(raw.get("error") or raw.get("inference_error") or raw.get("execution_error"))
    usable = not issues and not recorded_error
    positive = case["expected_actionable"]
    feasibility_task = case["task_kind"] == "action_feasibility"
    geometry = case["target_geometry"]
    has_target = positive and geometry is not None and not feasibility_task
    loc_point = _pixel_point(raw.get("localization_point"), image_size)
    # The unconditional locator is intentionally independent of gate parsing,
    # post-readout availability and the final action's coordinate validity.
    # A valid retained locator point can still diagnose a failed gate readout.
    localization_usable = not any("authoritative manifest" in issue for issue in issues) and not raw.get("localization_error")
    localization_hit = bool(has_target and loc_point is not None and point_in_geometry(loc_point, geometry) and localization_usable)
    action_hit = bool(has_target and point is not None and point_in_geometry(point, geometry) and usable)
    scorable_action = not feasibility_task and (not positive or has_target)
    behavioral = bool((emitted and action_hit) if positive else not emitted)
    behavioral = behavioral and usable and action_valid
    feasibility_correct = bool((emitted == positive) and parse_valid is True and usable and action_valid)
    oracle = None
    grid = _size(raw.get("grid_shape"))
    if has_target and grid is not None and image_size is not None:
        cache_key = (case["sample_id"], grid, image_size)
        if oracle_cache is None:
            oracle = quantization_oracle(geometry, grid, image_size)
        else:
            if cache_key not in oracle_cache:
                oracle_cache[cache_key] = quantization_oracle(geometry, grid, image_size)
            oracle = oracle_cache[cache_key]
    return {
        "raw_index": index, "sample_id": case["sample_id"], "dataset": case["dataset"],
        "method": raw["method"], "task_kind": case["task_kind"], "group_id": case["_group"],
        "domain": case["_domain"], "target_size": case["_target_size"],
        "target_area_fraction": case["_target_area"], "expected_actionable": positive,
        "emitted_click": emitted, "parse_valid": parse_valid, "recorded_error": recorded_error,
        "protocol_errors": issues, "cap_reached": cap, "cap_error": cap and parse_valid is not True,
        "localization_scorable": has_target, "localization_point_available": loc_point is not None,
        "localization_hit": localization_hit, "action_point_hit": action_hit,
        "action_scorable": scorable_action,
        "action_aware_correct": bool(scorable_action and behavioral and parse_valid is True),
        "behavioral_point_or_refusal_correct": bool(scorable_action and behavioral),
        "feasibility_correct": feasibility_correct,
        "strict_refusal_correct": bool(not positive and not emitted and parse_valid is True and usable and action_valid),
        "quantization_oracle": oracle, "click_probability": raw.get("click_probability"),
        "generated_decision": raw.get("generated_decision"),
        "trajectory_id": raw.get("trajectory_id"), "trajectory_timing_ms": raw.get("trajectory_timing_ms"),
    }


def _masks(rows: list[dict[str, Any]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    def arr(key):
        return np.asarray([r[key] for r in rows], dtype=bool)
    all_rows = np.ones(len(rows), dtype=bool)
    positive, emitted = arr("expected_actionable"), arr("emitted_click")
    grounded, action = arr("localization_scorable"), arr("action_scorable")
    valid = np.asarray([r["parse_valid"] is True for r in rows], dtype=bool)
    return {
        "localization_hit_ignoring_gate": (arr("localization_hit"), grounded),
        "localization_point_coverage": (arr("localization_point_available") & grounded, grounded),
        "action_aware_correct": (arr("action_aware_correct"), action),
        "behavioral_point_or_refusal_correct": (arr("behavioral_point_or_refusal_correct"), action),
        "feasibility_correct": (arr("feasibility_correct"), all_rows),
        "strict_refusal_correct": (arr("strict_refusal_correct"), ~positive),
        "positive_false_rejection": (~emitted & positive, positive),
        "negative_false_click": (emitted & ~positive, ~positive),
        "parse_failure": (~valid, all_rows),
        "negative_parse_failure": (~valid & ~positive, ~positive),
        "emitted_click_precision": (arr("action_aware_correct") & positive & emitted, emitted & action),
        "actionable_prediction_precision": (arr("feasibility_correct") & positive & emitted, emitted),
        "quantization_oracle": (np.asarray([r["quantization_oracle"] is True for r in rows], dtype=bool), np.asarray([r["quantization_oracle"] is not None for r in rows], dtype=bool)),
        "quantization_oracle_coverage": (np.asarray([r["quantization_oracle"] is not None for r in rows], dtype=bool), grounded),
        "localization_hit_given_oracle_reachable": (arr("localization_hit"), np.asarray([r["quantization_oracle"] is True for r in rows], dtype=bool)),
    }


def _rate(num: int, den: int) -> dict[str, Any]:
    return {"numerator": int(num), "denominator": int(den), "rate": num / den if den else None}


def _aggregate(rows: list[dict[str, Any]], seed: int, iterations: int) -> dict[str, Any]:
    masks = _masks(rows)
    metrics = {key: _rate(int((num & den).sum()), int(den.sum())) for key, (num, den) in masks.items()}
    result = {"n": len(rows), "n_groups": len({r["group_id"] for r in rows}), "metrics": metrics,
              "counts": {"recorded_errors": sum(r["recorded_error"] for r in rows),
                         "protocol_error_rows": sum(bool(r["protocol_errors"]) for r in rows),
                         "cap_reached": sum(r["cap_reached"] for r in rows),
                         "cap_errors": sum(r["cap_error"] for r in rows),
                         "parse_unknown": sum(r["parse_valid"] is None for r in rows)}}
    groups = sorted({r["group_id"] for r in rows})
    if iterations <= 0 or len(groups) < 2:
        result["group_bootstrap"] = {"status": "disabled" if iterations <= 0 else "unavailable", "unit": "manifest group_id else image_id else sample_id"}
        return result
    group_index = {g: i for i, g in enumerate(groups)}
    draws = np.random.default_rng(seed).multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=iterations)
    cis = {}
    for key, (num, den) in masks.items():
        numerator = np.zeros(len(groups), dtype=int)
        denominator = np.zeros(len(groups), dtype=int)
        for i, row in enumerate(rows):
            numerator[group_index[row["group_id"]]] += bool(num[i] and den[i])
            denominator[group_index[row["group_id"]]] += bool(den[i])
        ns, ds = draws @ numerator, draws @ denominator
        values = ns[ds > 0] / ds[ds > 0]
        cis[key] = {"point": metrics[key]["rate"], "ci95": np.percentile(values, [2.5, 97.5]).tolist() if len(values) else None, "n_valid_bootstrap": len(values)}
    result["group_bootstrap"] = {"status": "ok", "seed": seed, "iterations": iterations, "unit": "dataset-prefixed manifest group_id else image_id else sample_id", "metrics": cis}
    return result


def _trajectory_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    trajectories, unavailable = defaultdict(list), 0
    for row in rows:
        tid = row["trajectory_id"]
        if not isinstance(tid, str) or not tid:
            unavailable += 1
        else:
            trajectories[tid].append(row)
    records, conflicts, cross_sample_conflicts = [], [], []
    for tid, members in sorted(trajectories.items()):
        timings = [r["trajectory_timing_ms"] for r in members]
        if len({r["sample_id"] for r in members}) > 1:
            cross_sample_conflicts.append(tid)
        if any(t != timings[0] for t in timings[1:]) or tid in cross_sample_conflicts:
            conflicts.append(tid)
        timing = timings[0] if tid not in conflicts else None
        total = _finite(timing.get("total")) if isinstance(timing, dict) else _finite(timing)
        records.append({"trajectory_id": tid, "sample_ids": sorted({r["sample_id"] for r in members}), "methods": sorted({r["method"] for r in members}), "timing_ms": timing, "total_ms": total if total is not None and total >= 0 else None})
    values = [r["total_ms"] for r in records if r["total_ms"] is not None]
    return {"status": "ok" if values else "unavailable", "n_unique_trajectories": len(records), "n_rows_without_trajectory_id": unavailable, "conflicting_timing_ids": conflicts, "cross_sample_trajectory_ids": cross_sample_conflicts, "n_measured_trajectory_totals": len(values), "trajectory_total_ms": {"p50": float(np.median(values)) if values else None, "p95": float(np.percentile(values, 95)) if values else None}, "records": records,
            "interpretation": "A shared VLM trajectory is counted once. These are trajectory costs, not independently measured method latencies; no per-method speed ranking is supported."}


def evaluate_public_rows(rows: Iterable[dict[str, Any]], manifest: Iterable[dict[str, Any]] | str | Path,
                         output_dir: str | Path, seed: int = 144, bootstrap_iterations: int = 1000,
                         *, expected_methods: Sequence[str] = PUBLIC_METHODS) -> dict[str, Any]:
    """Evaluate all requested method x manifest cases without fitting anything.

    Missing/duplicate cells are explicit and never silently reduced to a common
    easy subset. Conflicting GT in logs becomes an error, not a GT override.
    """
    methods = tuple(expected_methods)
    if not methods or any(not isinstance(m, str) or not m for m in methods) or len(set(methods)) != len(methods):
        raise ValueError("expected_methods must contain nonempty unique strings")
    if not isinstance(bootstrap_iterations, int) or bootstrap_iterations < 0:
        raise ValueError("bootstrap_iterations must be a nonnegative integer")
    cases = _load_manifest(manifest)
    case_by_id = {r["sample_id"]: r for r in cases}
    raw_rows = list(rows)
    cells, extras = defaultdict(list), []
    for index, row in enumerate(raw_rows):
        key = row.get("sample_id"), row.get("method")
        if key[0] not in case_by_id or key[1] not in methods:
            extras.append({"raw_index": index, "sample_id": key[0], "method": key[1]})
        else:
            cells[key].append(index)
    oracle_cache: dict[tuple[Any, ...], bool] = {}
    normalized = [_normalize(raw_rows[index], case_by_id[sid], index, oracle_cache) for (sid, _), indices in cells.items() for index in indices]
    unique = [r for r in normalized if len(cells[r["sample_id"], r["method"]]) == 1]
    by_method, coverage = {}, {}
    for method in methods:
        selected = [r for r in unique if r["method"] == method]
        missing = [case["sample_id"] for case in cases if (case["sample_id"], method) not in cells]
        duplicates = {sid: indices for (sid, name), indices in cells.items() if name == method and len(indices) > 1}
        coverage[method] = {"n_expected": len(cases), "n_unique_scored": len(selected), "unique_coverage": len(selected) / len(cases), "missing_sample_ids": missing, "duplicate_raw_indices": duplicates}
        result = _aggregate(selected, seed, bootstrap_iterations)
        # Missing and ambiguous duplicate cases remain in the expected
        # denominator for success/correctness. Do not call missing negatives
        # safe or false-positive-free.
        expected_action = sum(c["task_kind"] != "action_feasibility" and (not c["expected_actionable"] or c["target_geometry"] is not None) for c in cases)
        result["coverage_adjusted"] = {
            "feasibility_correct": _rate(sum(r["feasibility_correct"] for r in selected), len(cases)),
            "action_aware_correct": _rate(sum(r["action_aware_correct"] for r in selected), expected_action),
            "negative_false_click_upper_bound": _rate(sum(not r["expected_actionable"] and r["emitted_click"] for r in selected) + sum(not c["expected_actionable"] and len(cells.get((c["sample_id"], method), [])) != 1 for c in cases), sum(not c["expected_actionable"] for c in cases)),
        }
        for field, manifest_field, name in (("dataset", "dataset", "by_dataset"), ("domain", "_domain", "by_domain"), ("target_size", "_target_size", "by_target_size"), ("task_kind", "task_kind", "by_task_kind")):
            slices = {}
            for value in sorted({r[manifest_field] for r in cases}):
                subset = [r for r in selected if r[field] == value]
                expected = sum(c[manifest_field] == value for c in cases)
                slices[value] = {**_aggregate(subset, seed, 0), "n_expected": expected,
                                 "unique_coverage": len(subset) / expected,
                                 "n_missing_or_ambiguous": expected - len(subset)}
            result[name] = slices
        by_method[method] = result
    summary = {
        "schema_version": 1, "experiment_type": "public_static_screenshot_frozen_policy_evaluation",
        "seed": seed, "bootstrap_iterations": bootstrap_iterations,
        "n_manifest_cases": len(cases), "n_manifest_groups": len({c["_group"] for c in cases}),
        "manifest_group_sources": dict(Counter(c["_group_source"] for c in cases)),
        "n_raw_rows": len(raw_rows), "expected_methods": list(methods), "coverage": coverage,
        "unexpected_rows": extras, "methods": by_method,
        "trajectory_timing": _trajectory_summary(unique),
        "geometry_protocol": {"normalized_coordinates": True, "pixel_normalization": "x/image_width, y/image_height; require 0<=pixel<size", "boundary": "inclusive bbox and polygon edges, normalized tolerance 1e-12; this implementation may differ from an author's edge convention", "quantization_oracle": "np.rint((column+0.5)*image_width/grid_width), clipped to [0,width-1], likewise row/height; evaluate these integer pixel centers against GT. This constrains the patch-center locator, not a native coordinate decoder.", "target_size": "largest component area/image area: small<=0.01, medium<=0.10, large>0.10; not union area"},
        "interpretation": ["Static screenshot point/decision scores do not prove that a click was executed or an application task completed.", "action_feasibility uses CLICK as the binary label 'the original instruction is actionable', including non-click original actions. It is not click execution or the dataset author's full-action success rate.", "The frozen heads and thresholds must be fixed before this evaluation; this module performs no fitting or threshold selection.", "Shared trajectories supply several policy readouts; trajectory timing must not be interpreted as independent latency measurements for each method.", "Bootstrap conditions on the fixed dataset adaptation, policies and thresholds; zero-error empirical intervals can degenerate and do not certify zero population risk."],
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    fields = ("raw_index", "sample_id", "dataset", "method", "task_kind", "group_id", "domain", "target_size", "target_area_fraction", "expected_actionable", "emitted_click", "parse_valid", "localization_point_available", "localization_hit", "action_point_hit", "action_scorable", "action_aware_correct", "feasibility_correct", "strict_refusal_correct", "quantization_oracle", "cap_reached", "cap_error", "recorded_error", "protocol_errors", "click_probability", "generated_decision", "trajectory_id")
    with (output / "predictions.csv").open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=(*fields, "duplicate_ambiguous"), extrasaction="ignore")
        writer.writeheader()
        for row in normalized:
            writer.writerow({**row, "duplicate_ambiguous": len(cells[row["sample_id"], row["method"]]) > 1, "protocol_errors": json.dumps(row["protocol_errors"], ensure_ascii=False)})
    (output / "report.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8-sig") as stream:
        for index, line in enumerate(stream, 1):
            if line.strip():
                try:
                    value = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{index}: invalid JSON") from exc
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{index}: expected JSON object")
                rows.append(value)
    return rows


def evaluate_public_jsonl(rows_path: str | Path, manifest_path: str | Path, output_dir: str | Path,
                          seed: int = 144, bootstrap_iterations: int = 1000, **kwargs) -> dict[str, Any]:
    return evaluate_public_rows(_read_jsonl(Path(rows_path)), Path(manifest_path), output_dir,
                                seed, bootstrap_iterations, **kwargs)


def _report(summary: dict[str, Any]) -> str:
    def rate(value):
        return "不可评估" if value["rate"] is None else f"{value['numerator']}/{value['denominator']} ({value['rate']:.1%})"
    lines = ["# 公开截图：冻结策略迁移评测", "", "本报告评估静态截图上的定位和决策，不是执行电脑、保存成功或桌面任务完成。所有头与阈值必须在公开集评测前冻结；评测器不训练或调阈值。", "", f"预期 {summary['n_manifest_cases']} 个样本、{summary['n_manifest_groups']} 个组。缺失和重复单元不用于唯一记录指标，也不假装是正确拒绝；JSON 另含预期分母下的保守正确率和负例误点击上界。", "", "| 方法 | 唯一覆盖 | 定位命中（忽略gate） | 动作感知正确 | 可行性判断正确 | 正例错误拒绝 | 负例预测CLICK | 格式失败 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for method, result in summary["methods"].items():
        c, m = summary["coverage"][method], result["metrics"]
        lines.append(f"| {method} | {c['n_unique_scored']}/{c['n_expected']} | {rate(m['localization_hit_ignoring_gate'])} | {rate(m['action_aware_correct'])} | {rate(m['feasibility_correct'])} | {rate(m['positive_false_rejection'])} | {rate(m['negative_false_click'])} | {rate(m['parse_failure'])} |")
    lines += ["", "公开数据集的任务构成和正负比例不同，混合汇总不是任一官方榜单成绩。逐数据集结果如下。", "", "| 数据集 | 方法 | 唯一覆盖 | 定位命中 | 动作感知正确 | 可行性判断正确 | 负例预测CLICK |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for method, result in summary["methods"].items():
        for dataset, entry in result["by_dataset"].items():
            m = entry["metrics"]
            lines.append(f"| {dataset} | {method} | {entry['n']}/{entry['n_expected']} | {rate(m['localization_hit_ignoring_gate'])} | {rate(m['action_aware_correct'])} | {rate(m['feasibility_correct'])} | {rate(m['negative_false_click'])} |")
    lines += ["", "动作感知正确：可点击定位样本必须解析有效、预测点击且点落入 GT；负例必须解析有效且 wait。格式失败导致的安全 wait 不算严格正确。定位独立统计使用 unconditional localization_point，分母包含该点缺失的正例，避免仅在 gate 放行后统计而美化结果。", "", "action_feasibility 样本只报告原指令是否可执行的二元迁移判断；CLICK 在此是‘可执行’标签。原任务可能是输入、键盘等非点击操作，不能据此声称点击坐标有效、完整动作成功或复现作者全动作 SR。此类样本从定位和动作感知指标中排除。", "", "| 方法 | 网格量化 oracle（有网格GT） | oracle覆盖 | 可达目标中的定位命中 | 点击precision（仅定位/拒绝任务） |", "| --- | --- | --- | --- | --- |"]
    for method, result in summary["methods"].items():
        m = result["metrics"]
        lines.append(f"| {method} | {rate(m['quantization_oracle'])} | {rate(m['quantization_oracle_coverage'])} | {rate(m['localization_hit_given_oracle_reachable'])} | {rate(m['emitted_click_precision'])} |")
    lines += ["", "Oracle 使用现有 grounder 的整数像素 patch center：np.rint 后 clip 到图内；它只约束 patch-center 定位器，不是原生文本坐标解码的上限。GT 采用归一化多框并集或多边形并集，边界包含（容差1e-12）。这是本实现的明确约定，边缘样本可能与原作者的严格不等式不同。目标大小按最大单组件面积分档，small≤1%、medium≤10%、large>10%，不冒充多个框的并集面积。", "", "逐数据集、domain、目标大小与任务类型切片保存在 metrics.json；cap、格式未知、运行错误、GT冲突和缺失/重复覆盖率均单独保留。置信区间按 dataset 前缀下的 group_id（缺失则 image_id，再缺失则 sample_id）整体重采样；未提供图像关联的逐样本区间不保证图像泛化。零错误经验区间可能退化。", "", "共享 VLM 轨迹只统计一次轨迹耗时，不报告八种 readout 或前三种策略各自独立测得的时延。trajectory_id 缺失时不支持方法速度比较；原始模型输出和时延应保留在输入 JSONL。", "", "产物：metrics.json、predictions.csv、report.md。"]
    return "\n".join(lines) + "\n"
