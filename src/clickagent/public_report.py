"""Integrate completed public-run evidence without changing any strict score.

The report contains evidence tables and explicit scope/coverage limitations,
not a generated scientific conclusion or an independent policy latency claim.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from typing import Any, Sequence

from .public_diagnostics import GATE_METHODS, _pixel_normalized
from .public_evaluation import _load_manifest, point_in_geometry


METHODS = (*GATE_METHODS, "text_decision_shared", "text_coordinates")
DECISION_METHODS = (*GATE_METHODS, "text_decision_shared")


def _json(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected JSON object")
    return value


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8-sig") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    if any(not isinstance(row, dict) for row in rows):
        raise ValueError(f"{path}: expected JSON objects")
    return rows


def _rate(num: int, den: int) -> dict[str, Any]:
    return {"numerator": int(num), "denominator": int(den), "rate": num / den if den else None}


def _click(row: dict[str, Any] | None) -> bool:
    return row is not None and isinstance(row.get("action"), dict) and row["action"].get("type") == "click"


def _error(row: dict[str, Any]) -> bool:
    return bool(row.get("error") or row.get("inference_error") or row.get("execution_error"))


def _grounded(case: dict[str, Any]) -> bool:
    return case["expected_actionable"] and case["task_kind"] != "action_feasibility" and case["target_geometry"] is not None


def _locator_hit(row: dict[str, Any] | None, case: dict[str, Any]) -> bool:
    if row is None:
        return False
    point = _pixel_normalized(row.get("localization_point"), row.get("model_image_size"))
    return point is not None and point_in_geometry(point, case["target_geometry"])


def _strict_positive(row: dict[str, Any] | None, case: dict[str, Any]) -> bool:
    if row is None or row.get("parse_valid") is not True or _error(row) or not _click(row):
        return False
    if case["task_kind"] == "action_feasibility":
        return True
    point = _pixel_normalized(row["action"], row.get("model_image_size"))
    return _grounded(case) and point is not None and point_in_geometry(point, case["target_geometry"])


def _full_manifest_counts(data_root: Path) -> tuple[dict[str, Any], list[str]]:
    paths = sorted((data_root / "manifests").glob("*.jsonl"))
    if not paths and (data_root / "manifest.jsonl").is_file():
        paths = [data_root / "manifest.jsonl"]
    grouped, source_paths, seen, issues = defaultdict(list), defaultdict(set), set(), []
    for path in paths:
        for row in _jsonl(path):
            key = row.get("dataset"), row.get("sample_id")
            if not key[0] or not key[1]:
                issues.append(f"{path}: full-manifest row missing dataset/sample_id")
                continue
            if key in seen:
                issues.append(f"duplicate full-manifest case {key}")
                continue
            seen.add(key)
            grouped[key[0]].append(row)
            source_paths[key[0]].add(str(path.resolve()))
    result = {}
    for dataset, rows in grouped.items():
        result[dataset] = {"all_adapted_rows": len(rows), "split_counts": dict(Counter(r.get("split", "test") for r in rows)),
                           "test_rows": sum(r.get("split", "test") == "test" for r in rows),
                           "all_image_or_scenario_groups": len({r.get("group_id") or r.get("image_id") or r["sample_id"] for r in rows}),
                           "source_manifests": sorted(source_paths[dataset])}
    return result, issues


def _diagnostic_for_dataset(diagnostics: dict[str, Any] | None, dataset: str,
                            cases: list[dict[str, Any]], run_dirs: set[str]) -> tuple[dict[str, Any] | None, list[str]]:
    issues = []
    if diagnostics is None:
        return None, [f"{dataset}: diagnostics file missing"]
    entry = diagnostics.get("datasets", {}).get(dataset)
    if not isinstance(entry, dict):
        return None, [f"{dataset}: dataset absent from diagnostics"]
    diag_runs = {str(Path(r["run_dir"]).resolve()) for r in diagnostics.get("runs", []) if r.get("run_dir")}
    if not run_dirs <= diag_runs:
        issues.append(f"{dataset}: diagnostics does not cover every requested run directory")
    if entry.get("n_expected_cases") != len(cases):
        issues.append(f"{dataset}: diagnostics expected cohort size mismatch")
    selected_ids = {case["sample_id"] for case in cases}
    records = entry.get("offline_lax_coordinates", {}).get("records", [])
    if any(row.get("sample_id") not in selected_ids for row in records):
        issues.append(f"{dataset}: diagnostics contains coordinate records outside selected cohort")
    if len({row.get("sample_id") for row in records}) != len(records):
        issues.append(f"{dataset}: duplicate coordinate records in diagnostics")
    return (None if issues else entry), issues


def _one_dataset(dataset: str, cases: list[dict[str, Any]], rows: list[dict[str, Any]],
                 run_entries: list[dict[str, Any]], full: dict[str, Any] | None,
                 diagnostics: dict[str, Any] | None, resolution: dict[str, Any] | None) -> tuple[dict[str, Any], list[str]]:
    issues, cells = [], defaultdict(list)
    for row in rows:
        cells[row["sample_id"], row.get("method")].append(row)
    def unique(sid, method):
        entries = cells.get((sid, method), [])
        return entries[0] if len(entries) == 1 else None
    positive = [case for case in cases if case["expected_actionable"]]
    negative = [case for case in cases if not case["expected_actionable"]]
    geometry = [case for case in positive if _grounded(case)]
    run_dirs = {entry["run_dir"] for entry in run_entries}
    diagnostic, diag_issues = _diagnostic_for_dataset(diagnostics, dataset, cases, run_dirs)
    issues.extend(diag_issues)
    lax = {} if diagnostic is None else {r["sample_id"]: r for r in diagnostic.get("offline_lax_coordinates", {}).get("records", [])}
    # All selected positive geometries stay in every point-score denominator;
    # no intersection of successful cases or available predictions is chosen.
    native_strict = sum(_strict_positive(unique(case["sample_id"], "text_coordinates"), case) for case in geometry)
    shared_hits = sum(_locator_hit(unique(case["sample_id"], "prefill_hidden"), case) for case in geometry)
    loose_hits = sum(lax.get(case["sample_id"], {}).get("eligible_offline_lax_click") is True and lax[case["sample_id"]].get("offline_grounding_hit") is True for case in geometry)
    loose_missing = [case["sample_id"] for case in cases if unique(case["sample_id"], "text_coordinates") is not None and case["sample_id"] not in lax]
    if loose_missing:
        issues.append(f"{dataset}: {len(loose_missing)} unique coordinate rows absent from offline diagnostic")
    oracle_num = oracle_den = 0
    shared_disagreement = []
    for entry in run_entries:
        ds = (entry.get("metrics") or {}).get("methods", {}).get("prefill_hidden", {}).get("by_dataset", {}).get(dataset, {})
        oracle = ds.get("metrics", {}).get("quantization_oracle")
        if isinstance(oracle, dict):
            oracle_num += oracle.get("numerator", 0)
            oracle_den += oracle.get("denominator", 0)
    if oracle_den > len(geometry):
        issues.append(f"{dataset}: selected oracle denominator exceeds unique positive cohort; overlapping/stale metrics")
    for case in geometry:
        points = []
        for method in DECISION_METHODS:
            row = unique(case["sample_id"], method)
            if row is not None and row.get("localization_point") is not None:
                points.append((row.get("localization_point"), row.get("model_image_size")))
        if points and any(point != points[0] for point in points[1:]):
            shared_disagreement.append(case["sample_id"])
    if shared_disagreement:
        issues.append(f"{dataset}: claimed shared locator points differ across methods")
    localization = {"n_selected_positive_geometry": len(geometry),
                    "shared_point_head_ignoring_gate": _rate(shared_hits, len(geometry)),
                    "shared_point_reference_method": "prefill_hidden",
                    "shared_point_method_disagreement_sample_ids": shared_disagreement,
                    "selected_grid_oracle_observed": _rate(oracle_num, oracle_den),
                    "selected_grid_oracle_lower_bound_expected": _rate(oracle_num, len(geometry)),
                    "selected_grid_oracle_coverage": _rate(oracle_den, len(geometry)),
                    "native_strict_point_hit": _rate(native_strict, len(geometry)),
                    "offline_lax_format_point_hit": _rate(loose_hits, len(geometry)) if diagnostic is not None else None,
                    "offline_lax_missing_unique_coordinate_sample_ids": loose_missing,
                    "point_scoring_applicable": bool(geometry),
                    "denominator_policy": "All preselected positive geometry cases; missing/duplicate predictions are not silently removed."}
    if geometry and oracle_den < len(geometry):
        issues.append(f"{dataset}: selected grid oracle coverage is incomplete")
    coverage, decision, counts, misclicks = {}, {}, {}, {}
    for method in METHODS:
        selected = [unique(case["sample_id"], method) for case in cases]
        observed = [row for row in selected if row is not None]
        missing = [case["sample_id"] for case in cases if not cells.get((case["sample_id"], method))]
        duplicates = {case["sample_id"]: len(cells[case["sample_id"], method]) for case in cases if len(cells.get((case["sample_id"], method), [])) > 1}
        coverage[method] = {"n_expected": len(cases), "n_unique": len(observed), "missing_sample_ids": missing, "duplicate_counts": duplicates}
        if len(observed) != len(cases):
            issues.append(f"{dataset}/{method}: unique coverage {len(observed)}/{len(cases)}")
        parse_failures = sum(row.get("parse_valid") is not True for row in observed)
        errors = sum(_error(row) for row in observed)
        capped = sum(any(row.get(key) is True for key in ("cap_reached", "truncated", "token_cap_reached")) for row in observed)
        method_raw = [r for r in rows if r.get("method") == method]
        counts[method] = {"raw_rows": len(method_raw), "unique_rows": len(observed),
                          "recorded_error_rows": sum(_error(r) for r in method_raw), "unique_recorded_error_rows": errors,
                          "raw_strict_parse_failure_rows": sum(r.get("parse_valid") is not True for r in method_raw),
                          "strict_parse_failures": _rate(parse_failures, len(observed)),
                          "cap_reached_rows": capped,
                          "cap_error_rows": sum(any(row.get(key) is True for key in ("cap_reached", "truncated", "token_cap_reached")) and row.get("parse_valid") is not True for row in observed)}
        pos_num = sum(_strict_positive(unique(case["sample_id"], method), case) for case in positive)
        neg_num = sum(_click(unique(case["sample_id"], method)) for case in negative)
        unknown_negative = sum(unique(case["sample_id"], method) is None for case in negative)
        decision[method] = {"positive_metric_kind": "binary_positive_allow" if all(case["task_kind"] == "action_feasibility" for case in cases) else "strict_correct_point_click",
                            "positive_correct_expected": _rate(pos_num, len(positive)),
                            "negative_false_click_observed_expected_denominator": _rate(neg_num, len(negative)),
                            "negative_unknown_rows": unknown_negative,
                            "negative_false_click_upper_bound_expected": _rate(neg_num + unknown_negative, len(negative)),
                            "positive_unique_coverage": _rate(sum(unique(c["sample_id"], method) is not None for c in positive), len(positive)),
                            "negative_unique_coverage": _rate(len(negative) - unknown_negative, len(negative))}
        point_cases = [case for case in cases if case["task_kind"] != "action_feasibility" and (not case["expected_actionable"] or _grounded(case))]
        feasibility_cases = [case for case in cases if case["task_kind"] == "action_feasibility"]
        def point_counts(subset):
            emitted = [(case, unique(case["sample_id"], method)) for case in subset if _click(unique(case["sample_id"], method))]
            correct_ids = [case["sample_id"] for case, row in emitted if case["expected_actionable"] and _strict_positive(row, case)]
            wrong_ids = [case["sample_id"] for case, row in emitted if case["expected_actionable"] and row.get("parse_valid") is True and not _error(row) and not _strict_positive(row, case)]
            false_ids = [case["sample_id"] for case, _ in emitted if not case["expected_actionable"]]
            return {"n_expected_cases": len(subset), "n_unique_cases": sum(unique(case["sample_id"], method) is not None for case in subset),
                    "positive_wrong_target_clicks": len(wrong_ids), "positive_wrong_target_sample_ids": wrong_ids,
                    "negative_false_clicks": len(false_ids), "negative_false_click_sample_ids": false_ids,
                    "negative_false_click_rate_expected": _rate(len(false_ids), sum(not case["expected_actionable"] for case in subset)),
                    "total_emitted_clicks": len(emitted), "correct_clicks": len(correct_ids),
                    "correct_click_precision": _rate(len(correct_ids), len(emitted)),
                    "invalid_parse_emitted_clicks": sum(row.get("parse_valid") is not True for _, row in emitted),
                    "error_emitted_clicks": sum(_error(row) for _, row in emitted)}
        point = point_counts(point_cases) if point_cases else None
        feasibility_negatives = [case for case in feasibility_cases if not case["expected_actionable"]]
        binary_false_allow = _rate(sum(_click(unique(case["sample_id"], method)) for case in feasibility_negatives), len(feasibility_negatives))
        reference_num = reference_den = 0
        reference_available = bool(point_cases)
        for entry in run_entries:
            formal = (entry.get("metrics") or {}).get("methods", {}).get(method, {}).get("by_dataset", {}).get(dataset, {}).get("metrics", {}).get("emitted_click_precision")
            if not isinstance(formal, dict):
                reference_available = False
            else:
                reference_num += formal.get("numerator", 0)
                reference_den += formal.get("denominator", 0)
        check = {"status": "not_applicable_binary_feasibility_only" if point is None else "unavailable_formal_precision"}
        if point is not None and reference_available:
            matches = reference_num == point["correct_clicks"] and reference_den == point["total_emitted_clicks"]
            check = {"status": "match" if matches else "mismatch", "formal_precision": _rate(reference_num, reference_den)}
            if not matches:
                issues.append(f"{dataset}/{method}: recomputed click precision disagrees with formal emitted_click_precision")
        misclicks[method] = {"scope": "grounding_refusal_only" if point is not None else "binary_feasibility_only",
                            "point_metrics": point,
                            "by_point_task_kind": {kind: point_counts([case for case in point_cases if case["task_kind"] == kind]) for kind in ("grounding", "refusal") if any(case["task_kind"] == kind for case in point_cases)},
                            "binary_feasibility_false_allow": binary_false_allow if feasibility_cases else None,
                            "point_precision_excludes_action_feasibility": True,
                            "formal_precision_crosscheck": check}
    aucs = {}
    for method in GATE_METHODS:
        gate = {} if diagnostic is None else diagnostic.get("gates", {}).get(method, {})
        if diagnostic is not None and not gate:
            issues.append(f"{dataset}/{method}: gate diagnostic missing")
        aucs[method] = {"auc": gate.get("click_probability_auroc", {"status": "unavailable_diagnostics", "value": None}),
                        "probability_coverage": gate.get("click_probability_coverage"),
                        "probability_coverage_by_class": gate.get("click_probability_coverage_by_class")}
    audit_entry = (resolution or {}).get("by_dataset", {}).get(dataset)
    full_oracle = {"status": "unavailable", "scope": "all adapted rows in the full supplied manifest, never the selected cohort"}
    if audit_entry is not None and resolution.get("full_supplied_manifest") is True and full is not None and audit_entry.get("input_rows") == full["all_adapted_rows"]:
        full_oracle.update({"status": "ok" if audit_entry.get("positive_geometry_denominator", 0) else "not_applicable_no_positive_geometry",
                            "input_rows": audit_entry["input_rows"],
                            "quantization_oracle": audit_entry.get("quantization_oracle"),
                            "quantization_oracle_upper_bound": audit_entry.get("quantization_oracle_upper_bound"),
                            "quantization_oracle_coverage": audit_entry.get("quantization_oracle_coverage"),
                            "unresolved_positive_geometry_rows": audit_entry.get("unresolved_positive_geometry_rows"),
                            "geometry_clipped_rows": audit_entry.get("n_geometry_clipped"),
                            "invalid_positive_geometry_rows": audit_entry.get("n_invalid_positive_geometry")})
    else:
        issues.append(f"{dataset}: full-manifest oracle absent or does not match the full manifest row count")
    sample_counts = {"full_manifest": full, "selected_n": len(cases), "selected_positive": len(positive), "selected_negative": len(negative),
                     "selected_groups": len({case["_group"] for case in cases}),
                     "group_sources": dict(Counter(case["_group_source"] for case in cases)),
                     "selected_task_kinds": dict(Counter(case["task_kind"] for case in cases)),
                     "selections": [{"run_dir": entry["run_dir"], "selection": (entry.get("selection") or {}).get("datasets", {}).get(dataset)} for entry in run_entries]}
    if full is None:
        issues.append(f"{dataset}: full adapted manifest unavailable")
    return {"sample_counts": sample_counts, "localization": localization, "decision_metrics": decision, "misclick_metrics": misclicks,
            "auroc": aucs, "full_manifest_grid_oracle": full_oracle, "coverage": coverage,
            "errors_and_format": counts}, issues


def generate_public_report(run_dirs: Sequence[str | Path], diagnostics_path: str | Path,
                           resolution_audit_path: str | Path, data_root: str | Path,
                           output_dir: str | Path, *, plots: bool = False) -> dict[str, Any]:
    """Write Chinese evidence tables from finished runs; partial evidence stays explicit."""
    directories = [Path(path).resolve() for path in run_dirs]
    if not directories or len(set(directories)) != len(directories):
        raise ValueError("run directories must be nonempty and unique")
    output = Path(output_dir).resolve()
    if output in directories or any(output == path / "evaluation" for path in directories):
        raise ValueError("report output must be separate from formal run/evaluation roots")
    diagnostics = _json(Path(diagnostics_path))
    resolution = _json(Path(resolution_audit_path))
    full, issues = _full_manifest_counts(Path(data_root))
    cases_by_dataset, rows_by_dataset, runs_by_dataset = defaultdict(dict), defaultdict(list), defaultdict(list)
    runs = []
    for directory in directories:
        manifest_path = directory / "selected_manifest.jsonl"
        if not manifest_path.is_file():
            issues.append(f"{directory}: selected manifest missing")
            runs.append({"run_dir": str(directory), "status": "partial_missing_manifest"})
            continue
        cases = _load_manifest(manifest_path)
        raw_rows = _jsonl(directory / "predictions.jsonl")
        completion, selection, metrics = (_json(directory / name) for name in ("completion.json", "selection.json", "evaluation/metrics.json"))
        run_issues = []
        if completion is None or completion.get("complete") is not True:
            run_issues.append("completion missing or not complete")
        elif completion.get("n_cases") != len(cases) or completion.get("n_rows") != len(raw_rows):
            run_issues.append("completion counts disagree with selected/raw files")
        if completion is not None and completion.get("error_rows") is not None and completion["error_rows"] != sum(_error(row) for row in raw_rows):
            run_issues.append("completion error count disagrees with retained raw errors")
        if selection is None:
            run_issues.append("selection audit missing")
        else:
            if selection.get("fixed_before_model_loading") is not True:
                run_issues.append("selection does not certify cohort fixed before loading model")
            for dataset in {case["dataset"] for case in cases}:
                selected_cases = [case for case in cases if case["dataset"] == dataset]
                audit = selection.get("datasets", {}).get(dataset, {})
                if audit.get("selected") != len(selected_cases) or audit.get("positive") != sum(case["expected_actionable"] for case in selected_cases) or audit.get("negative") != sum(not case["expected_actionable"] for case in selected_cases):
                    run_issues.append(f"{dataset}: selection counts disagree with fixed manifest")
        if metrics is None:
            run_issues.append("formal metrics missing")
        elif metrics.get("n_manifest_cases") != len(cases):
            run_issues.append("formal metrics manifest count mismatch")
        if metrics is not None:
            for method in METHODS:
                metric_coverage = metrics.get("coverage", {}).get(method, {})
                raw_cells = Counter(row.get("sample_id") for row in raw_rows if row.get("method") == method)
                raw_unique = sum(raw_cells.get(case["sample_id"], 0) == 1 for case in cases)
                if metric_coverage.get("n_expected") != len(cases) or metric_coverage.get("n_unique_scored") != raw_unique:
                    run_issues.append(f"{method}: formal coverage disagrees with selected/raw files")
        if not (directory / "predictions.jsonl").is_file():
            run_issues.append("formal predictions missing")
        entry = {"run_dir": str(directory), "status": "partial" if run_issues else "complete",
                 "completion": completion, "selection": selection, "metrics": metrics,
                 "n_selected": len(cases), "n_raw_rows": len(raw_rows), "issues": run_issues}
        runs.append({key: value for key, value in entry.items() if key != "metrics"})
        issues.extend(f"{directory}: {issue}" for issue in run_issues)
        by_id = {case["sample_id"]: case for case in cases}
        for case in cases:
            previous = cases_by_dataset[case["dataset"]].get(case["sample_id"])
            if previous is not None:
                issues.append(f"duplicate selected case across runs: {case['dataset']}/{case['sample_id']}")
                if any(case[key] != previous[key] for key in ("instruction", "expected_actionable", "task_kind", "target_geometry")):
                    raise ValueError("conflicting selected manifest facts")
            cases_by_dataset[case["dataset"]][case["sample_id"]] = case
        for dataset in {case["dataset"] for case in cases}:
            runs_by_dataset[dataset].append(entry)
        for row in raw_rows:
            sid = row.get("sample_id")
            if sid not in by_id or row.get("dataset") != by_id[sid]["dataset"] or row.get("method") not in METHODS:
                issues.append(f"{directory}: unexpected prediction {sid}/{row.get('method')}")
                continue
            rows_by_dataset[row["dataset"]].append(row)
    datasets = {}
    for dataset, by_id in sorted(cases_by_dataset.items()):
        datasets[dataset], ds_issues = _one_dataset(dataset, list(by_id.values()), rows_by_dataset[dataset],
                                                   runs_by_dataset[dataset], full.get(dataset), diagnostics, resolution)
        issues.extend(ds_issues)
    summary = {"schema_version": 1, "report_type": "public_frozen_policy_evidence_tables",
               "status": "partial" if issues else "complete", "issues": issues, "runs": runs,
               "datasets": datasets,
               "sources": {"diagnostics": str(Path(diagnostics_path).resolve()), "resolution_audit": str(Path(resolution_audit_path).resolve()), "data_root": str(Path(data_root).resolve())},
               "scope": {"retraining": False, "threshold_selection": False, "static_screenshots_not_execution": True,
                         "offline_lax_does_not_replace_strict": True, "independent_method_latency_comparison": False,
                         "conclusions": "Evidence tables only; conclusions must be written and reviewed separately."}}
    output.mkdir(parents=True, exist_ok=True)
    summary["plots"] = _plots(datasets, output) if plots else {"status": "not_requested", "files": []}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _report(summary: dict[str, Any]) -> str:
    def rate(value):
        return "不可评估" if not value or value.get("rate") is None else f"{value['numerator']}/{value['denominator']} ({value['rate']:.1%})"
    lines = ["# 公开数据集冻结策略：证据表", "", f"报告状态：**{summary['status']}**。本报告只汇总证据，不自动生成最终研究结论。全部模型头与旧验证集阈值保持冻结；通用指令 prompt 相对原保存任务已改变，属于零重训迁移。静态截图上的点命中或可行性判断，不等同于操控电脑或完成应用任务。", "", "## 表1：数据范围与固定抽样", "", "| 数据集 | 全量适配条目 | 其中test | 本次条目 | 本次正/负 | 本次截图/场景组 |", "| --- | ---: | ---: | ---: | --- | ---: |"]
    for dataset, result in summary["datasets"].items():
        c, full = result["sample_counts"], result["sample_counts"]["full_manifest"] or {}
        lines.append(f"| {dataset} | {full.get('all_adapted_rows', '缺失')} | {full.get('test_rows', '缺失')} | {c['selected_n']} | {c['selected_positive']}/{c['selected_negative']} | {c['selected_groups']} |")
    lines += ["", "每集固定哈希抽样在模型加载前完成，默认最多200条；按预定方案尽量保留正负，类别构成不等同于官方全量分布。ConflictGUI 的全量包括 calibration/test，正式迁移只使用 click/tap 兼容的 test 子集；不能称作者完整多动作 SR。截图组优先取 group_id，再取 image_id，否则按 sample_id 并明示来源。", "", "## 表2：同题正例定位与量化限制", "", "| 数据集 | 全部预选定位正例 | 共享点头（忽略gate） | selected网格oracle | oracle覆盖 | native strict点命中 | 离线宽松格式点命中 |", "| --- | ---: | --- | --- | --- | --- | --- |"]
    for dataset, result in summary["datasets"].items():
        loc = result["localization"]
        lines.append(f"| {dataset} | {loc['n_selected_positive_geometry']} | {rate(loc['shared_point_head_ignoring_gate'])} | {rate(loc['selected_grid_oracle_observed'])} | {rate(loc['selected_grid_oracle_coverage'])} | {rate(loc['native_strict_point_hit'])} | {rate(loc['offline_lax_format_point_hit'])} |")
    lines += ["", "分母固定为全部预选、带定位几何的正例，不按成功或可解析样本取交集。共享点头以 prefill_hidden 的无条件像素点计分，七种共享定位读出的一致性另查；native strict 要求解析有效且正式动作 CLICK 的点命中。selected oracle 的分母为实际可计算网格的正例，覆盖不足明确保留，不得当作全量 oracle。ConflictGUI 不造框、不计点命中，故此表不可评估。", "", "离线宽松格式结果**不替换 strict**：仅完整无cap/无error的整条整数 CLICK，允许 ASCII 逗号或空白分隔；newline 或匹配 native EOS 提供完整性证据。坐标使用原映射，不用 GT 修复，恢复出的负例点击也在诊断保留。", "", "## 全量网格oracle（独立范围）", "", "| 数据集 | 审计状态 | 全量输入条目 | 全量正例几何oracle | 几何审计覆盖 | 未解析正例几何 |", "| --- | --- | ---: | --- | --- | ---: |"]
    for dataset, result in summary["datasets"].items():
        audit = result["full_manifest_grid_oracle"]
        coverage = audit.get("quantization_oracle_coverage")
        lines.append(f"| {dataset} | {audit['status']} | {audit.get('input_rows', '缺失')} | {rate(audit.get('quantization_oracle'))} | {'不可评估' if coverage is None else f'{coverage:.1%}'} | {audit.get('unresolved_positive_geometry_rows', '缺失')} |")
    lines += ["", "全量 oracle 仅在审计声明 full_supplied_manifest 且 input_rows 与该完整适配 manifest 一致时展示。它是当前整数 patch-center 定位器的几何上限，不是 native 文本坐标解码上限，也不表示视觉推理能力。边界、裁剪与无效标注沿用审计文件的显式协议。", "", "## 表3：冻结gate与文本类别的动作感知结果", "", "| 数据集 | 方法 | 正例正确点击/正例数（Conflict为binary放行） | 负例预测CLICK/负例数 | 负例未知记录 | 唯一覆盖 |", "| --- | --- | --- | --- | ---: | --- |"]
    for dataset, result in summary["datasets"].items():
        for method in DECISION_METHODS:
            d, coverage = result["decision_metrics"][method], result["coverage"][method]
            lines.append(f"| {dataset} | {method} | {rate(d['positive_correct_expected'])} | {rate(d['negative_false_click_observed_expected_denominator'])} | {d['negative_unknown_rows']} | {coverage['n_unique']}/{coverage['n_expected']} |")
    lines += ["", "定位任务的正例分子要求 parse_valid、CLICK 且点命中；不是单纯 gate 放行率。ConflictGUI 只检验二元可行性，CLICK 是‘原指令可执行’标签，不能当作点或全动作成功。缺失和重复不算正例正确，负例未知数量及最坏情况误点击上界在 JSON 单列，不能把未知当作正确拒绝。", "", "## 表3b：八方法误点与正确点击precision", "", "| 数据集 | 方法 | 正例错目标（有效CLICK） | 负例误点击/Conflict误放行 | 发起CLICK数 | 正确点击数 | 正确点击precision |", "| --- | --- | ---: | --- | ---: | ---: | --- |"]
    for dataset, result in summary["datasets"].items():
        for method, entry in result["misclick_metrics"].items():
            point = entry["point_metrics"]
            if point is None:
                lines.append(f"| {dataset} | {method} | 不适用 | 二元误放行 {rate(entry['binary_feasibility_false_allow'])} | 不适用 | 不适用 | 不适用 |")
            else:
                lines.append(f"| {dataset} | {method} | {point['positive_wrong_target_clicks']} | {rate(point['negative_false_click_rate_expected'])} | {point['total_emitted_clicks']} | {point['correct_clicks']} | {rate(point['correct_click_precision'])} |")
    lines += ["", "定位/拒绝任务中，正例错目标指解析有效、无运行错误的 CLICK 未命中该正例 GT；负例发出 CLICK 即为误放行。precision 严格等于**正确点击数/全部发起CLICK数**：分母也包含负例点击及无效格式却发出 CLICK 的记录；0次发起时为 null（不可评估），不能把全部 wait 解读为高 precision。这里‘发起’指静态日志中的预测动作，不代表已实际执行电脑。grounding/refusal 的分任务计数、错点ID和正式 emitted_click_precision 交叉校验保存在 JSON。ConflictGUI 完全排除点精度，只报告二元误放行，不把可行性 CLICK 标签当作实际坐标点击。缺失/重复仍沿用前表 coverage，不当作成功。", "", "## 表4：冻结gate的概率AUROC", "", "| 数据集 | 方法 | AUROC | 概率覆盖 | 正类概率覆盖 | 负类概率覆盖 |", "| --- | --- | --- | --- | --- | --- |"]
    for dataset, result in summary["datasets"].items():
        for method, d in result["auroc"].items():
            auc, classes = d["auc"].get("value"), d["probability_coverage_by_class"] or {}
            lines.append(f"| {dataset} | {method} | {'不可评估' if auc is None else f'{auc:.4f}'} | {rate(d['probability_coverage'])} | {rate(classes.get('positive'))} | {rate(classes.get('negative'))} |")
    lines += ["", "AUROC 只在两类有效概率均存在时计算。单类数据集不可评估；概率缺失不当作0，覆盖必须同时阅读。排序能力和固定旧阈值的校准效果是两个不同指标，本报告不据此调整阈值。", "", "## 运行错误、格式与完整性", "", "| 数据集 | 方法 | 原始/唯一行 | error行 | strict解析失败 | cap触及/格式错误 |", "| --- | --- | --- | ---: | --- | --- |"]
    for dataset, result in summary["datasets"].items():
        for method, c in result["errors_and_format"].items():
            lines.append(f"| {dataset} | {method} | {c['raw_rows']}/{c['unique_rows']} | {c['recorded_error_rows']} | {rate(c['strict_parse_failures'])} | {c['cap_reached_rows']}/{c['cap_error_rows']} |")
    lines += ["", "正式八方法共享部分生成轨迹，本报告不进行八种 method latency 速度比较。缺 completion、coverage 不足、缺诊断或范围冲突均保留为 partial；错误和解析失败不从主分母消失。", ""]
    if summary["issues"]:
        lines += ["具体缺口：", "", *[f"- {issue}" for issue in summary["issues"]], ""]
    if summary["plots"].get("files"):
        lines += ["比较图：", "", *[f"- [{Path(path).name}]({path})" for path in summary["plots"]["files"]], ""]
    lines += ["来源路径、逐方法coverage、抽样审计、概率覆盖、完整性标记保存在 summary.json；本文不覆盖原始 strict metrics 或 predictions。"]
    return "\n".join(lines) + "\n"


def _plots(datasets: dict[str, Any], output: Path) -> dict[str, Any]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        return {"status": "unavailable", "reason": str(exc), "files": []}
    applicable = [(name, data["localization"]) for name, data in datasets.items() if data["localization"]["point_scoring_applicable"]]
    if not applicable:
        return {"status": "not_applicable_no_grounding_cases", "files": []}
    fig, ax = plt.subplots(figsize=(max(7, len(applicable) * 1.8), 4.6))
    keys = ("shared_point_head_ignoring_gate", "selected_grid_oracle_lower_bound_expected", "native_strict_point_hit", "offline_lax_format_point_hit")
    labels = ("Shared point head (no gate)", "Selected grid oracle*", "Native strict", "Offline lax format only")
    x = np.arange(len(applicable))
    for index, (key, label) in enumerate(zip(keys, labels)):
        values = [(loc[key] or {}).get("rate") for _, loc in applicable]
        ax.bar(x + (index - 1.5) * 0.19, [np.nan if value is None else value * 100 for value in values], width=0.18, label=label)
    ax.set_xticks(x, [name for name, _ in applicable])
    ax.set_ylim(0, 105)
    ax.set_ylabel("Hit rate on all preselected positive geometry cases (%)")
    ax.set_title("Frozen transfer: localization evidence (static screenshots)")
    ax.legend(fontsize=8)
    fig.text(0.02, 0.01, "*Missing oracle grids count as unknown; plotted value is the lower bound. Offline lax never replaces strict.", fontsize=8)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    files = [output / "localization_comparison.png", output / "localization_comparison.pdf"]
    for path in files:
        fig.savefig(path, dpi=180)
    plt.close(fig)
    return {"status": "ok", "files": [str(path) for path in files]}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--resolution-audit", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, default=Path("data/public_v1"))
    parser.add_argument("--output", type=Path, default=Path("runs/public_v1/final_report"))
    parser.add_argument("--plots", action="store_true")
    args = parser.parse_args(argv)
    summary = generate_public_report(args.runs, args.diagnostics, args.resolution_audit, args.data_root, args.output, plots=args.plots)
    print(json.dumps({"status": summary["status"], "datasets": list(summary["datasets"]), "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
