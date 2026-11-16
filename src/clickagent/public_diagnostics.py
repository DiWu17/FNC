"""Read-only probability and output-format diagnostics for public runs.

No VLM, actor, training or threshold selection is invoked.  In particular, a
recovered coordinate below never replaces a logged action or a strict score.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

import numpy as np

from .public_evaluation import _load_manifest, point_in_geometry


GATE_METHODS = ("prefill_hidden", "post_hidden", "prefill_attention", "post_attention", "prefill_combined", "post_combined")
LAX_CLICK = re.compile(r"\A[ \t\r\n]*CLICK[ \t]+([0-9]+)(?:[ \t]*,[ \t]*|[ \t]+)([0-9]+)[ \t\r\n]*\Z")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8-sig") as stream:
        for index, line in enumerate(stream, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{index}: expected JSON object")
            records.append(row)
    return records


def _number(value: Any) -> float | None:
    return float(value) if not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value)) else None


def _rate(num: int, den: int) -> dict[str, Any]:
    return {"numerator": int(num), "denominator": int(den), "rate": num / den if den else None}


def _distribution(values: Iterable[float]) -> dict[str, Any]:
    values = np.asarray(list(values), dtype=np.float64)
    return {"n": len(values), "mean": float(values.mean()) if len(values) else None,
            "min": float(values.min()) if len(values) else None, "max": float(values.max()) if len(values) else None,
            **{f"p{p:02d}": float(np.percentile(values, p)) if len(values) else None for p in (5, 25, 50, 75, 95)}}


def _auroc(labels: Sequence[bool], probabilities: Sequence[float]) -> float | None:
    """Mann-Whitney AUROC with exact tie averaging; not a fitted estimator."""
    labels = np.asarray(labels, dtype=bool)
    positive, negative = int(labels.sum()), int((~labels).sum())
    if not positive or not negative:
        return None
    scores = np.asarray(probabilities, dtype=np.float64)
    order = np.argsort(scores, kind="stable")
    ranks = np.empty(len(scores), dtype=np.float64)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and scores[order[end]] == scores[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2
        start = end
    return float((ranks[labels].sum() - positive * (positive + 1) / 2) / (positive * negative))


def _pixel_normalized(point: Any, image_size: Any) -> tuple[float, float] | None:
    if isinstance(point, dict):
        point = [point.get("x"), point.get("y")]
    if not isinstance(point, (list, tuple)) or len(point) != 2 or not isinstance(image_size, (list, tuple)) or len(image_size) != 2:
        return None
    x, y, w, h = (_number(v) for v in (*point, *image_size))
    if any(v is None for v in (x, y, w, h)) or w <= 0 or h <= 0 or not w.is_integer() or not h.is_integer():
        return None
    return (x / w, y / h) if 0 <= x < w and 0 <= y < h else None


def _grounded(case: dict[str, Any]) -> bool:
    return case["expected_actionable"] and case["task_kind"] != "action_feasibility" and case["target_geometry"] is not None


def _logged_click(row: dict[str, Any]) -> bool:
    return isinstance(row.get("action"), dict) and row["action"].get("type") == "click"


def _locator_hit(row: dict[str, Any], case: dict[str, Any]) -> bool | None:
    if not _grounded(case):
        return None
    point = _pixel_normalized(row.get("localization_point"), row.get("model_image_size"))
    return point_in_geometry(point, case["target_geometry"]) if point is not None else None


def _gate_diagnostic(rows: list[dict[str, Any]], cases: list[dict[str, Any]], method: str) -> dict[str, Any]:
    cells = defaultdict(list)
    for row in rows:
        if row.get("method") == method:
            cells[row["sample_id"]].append(row)
    by_id = {case["sample_id"]: case for case in cases}
    unique = [members[0] for members in cells.values() if len(members) == 1]
    missing = [sid for sid in by_id if sid not in cells]
    duplicates = {sid: [{"run_dir": r["_run_dir"], "raw_index": r["_raw_index"]} for r in members] for sid, members in cells.items() if len(members) > 1}
    available, invalid_probabilities = [], []
    counters = Counter()
    unknown_action_negative = 0
    cross_ids = defaultdict(list)
    for row in unique:
        case = by_id[row["sample_id"]]
        positive, emitted = case["expected_actionable"], _logged_click(row)
        action_known = isinstance(row.get("action"), dict) and row["action"].get("type") in ("click", "wait")
        if action_known:
            counters["positive_observed" if positive else "negative_observed"] += 1
            counters["positive_allowed" if positive else "negative_false_click"] += emitted
        else:
            counters["unknown_action"] += 1
            unknown_action_negative += not positive
        counters["parse_failure"] += row.get("parse_valid") is not True
        counters["recorded_error"] += bool(row.get("error") or row.get("inference_error"))
        probability = _number(row.get("click_probability"))
        error = row.get("error") or row.get("inference_error")
        if probability is not None and 0 <= probability <= 1 and not error:
            available.append((row["sample_id"], positive, probability))
            threshold = _number(row.get("gate_threshold"))
            if threshold is not None and action_known and (np.float64(probability) >= np.float64(threshold)) != emitted:
                cross_ids["threshold_action_disagreement"].append(row["sample_id"])
        else:
            invalid_probabilities.append(row["sample_id"])
        hit = _locator_hit(row, case)
        if _grounded(case):
            counters["grounded_positive_observed"] += 1
            if hit is not None:
                counters["localization_point_available"] += 1
                counters["localization_hit"] += hit
            if hit is True and action_known and not emitted:
                cross_ids["localization_hit_but_gate_rejected"].append(row["sample_id"])
                counters["localization_hit_but_invalid_gate_readout"] += row.get("parse_valid") is not True
            if hit is False and emitted:
                cross_ids["localization_miss_but_gate_allowed"].append(row["sample_id"])
            if hit is None and emitted:
                cross_ids["gate_allowed_without_locator_point"].append(row["sample_id"])
    expected_positive = sum(case["expected_actionable"] for case in cases)
    expected_negative = len(cases) - expected_positive
    unknown_negative = sum(not case["expected_actionable"] and len(cells.get(case["sample_id"], [])) != 1 for case in cases)
    labels, scores = [entry[1] for entry in available], [entry[2] for entry in available]
    auc = _auroc(labels, scores)
    auc_status = "ok" if auc is not None else "unavailable_missing_probability_class" if expected_positive and expected_negative else "not_applicable_single_expected_class"
    thresholds = sorted({_number(r.get("gate_threshold")) for r in unique if _number(r.get("gate_threshold")) is not None})
    return {
        "n_expected": len(cases), "n_unique_rows": len(unique), "unique_row_coverage": len(unique) / len(cases),
        "missing_sample_ids": missing, "duplicate_records": duplicates,
        "click_probability_coverage": _rate(len(available), len(cases)),
        "click_probability_coverage_by_class": {
            "positive": _rate(sum(label for _, label, _ in available), expected_positive),
            "negative": _rate(sum(not label for _, label, _ in available), expected_negative)},
        "invalid_or_missing_probability_sample_ids": invalid_probabilities,
        "probability_missing_due_to_row_coverage_sample_ids": sorted(set(missing) | set(duplicates)),
        "expected_positive": expected_positive, "expected_negative": expected_negative,
        "probability_positive": _distribution(p for _, label, p in available if label),
        "probability_negative": _distribution(p for _, label, p in available if not label),
        "click_probability_auroc": {"status": auc_status, "value": auc,
                                    "n_positive_with_probability": sum(labels), "n_negative_with_probability": len(labels) - sum(labels),
                                    "scope": "Available unique finite probabilities only; accompanying class coverage is required."},
        "positive_allow_rate_observed": _rate(counters["positive_allowed"], counters["positive_observed"]),
        "positive_allow_lower_bound_expected": _rate(counters["positive_allowed"], expected_positive),
        "negative_false_click_observed": _rate(counters["negative_false_click"], counters["negative_observed"]),
        "negative_false_click_upper_bound_expected": _rate(counters["negative_false_click"] + unknown_negative + unknown_action_negative, expected_negative),
        "unconditional_localization_observed": _rate(counters["localization_hit"], counters["grounded_positive_observed"]),
        "locator_point_coverage_observed": _rate(counters["localization_point_available"], counters["grounded_positive_observed"]),
        "counts": dict(counters),
        "cross_counts": {key: len(cross_ids[key]) for key in ("localization_hit_but_gate_rejected", "localization_miss_but_gate_allowed", "gate_allowed_without_locator_point", "threshold_action_disagreement")},
        "cross_sample_ids": dict(cross_ids), "frozen_thresholds_observed": thresholds,
        "thresholds_constant": len(thresholds) == 1 if thresholds else None,
        "interpretation": "Allowance is the logged click decision; missing rows are unknown, not silently counted as rejections or safe negatives. Missing locator points are separate from observed coordinate misses.",
    }


def parse_offline_lax_click(text: Any) -> tuple[int, int] | None:
    """Only extend the integer separator to ASCII comma or whitespace.

    Completion/cap/error checks are intentionally enforced separately. No extra
    prose, parentheses, decimals, sign, partial integer or GT repair is allowed.
    """
    if not isinstance(text, str):
        return None
    match = LAX_CLICK.fullmatch(text)
    if match is None:
        return None
    point = int(match.group(1)), int(match.group(2))
    return point if all(0 <= value <= 1000 for value in point) else None


def _native_evidence(row: dict[str, Any], run_dir: Path, identity_fingerprint: str | None) -> dict[str, Any]:
    path = run_dir / "native" / (hashlib.sha256(row["sample_id"].encode()).hexdigest()[:24] + ".json")
    if not path.is_file():
        return {"status": "unavailable", "path": str(path)}
    try:
        native = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return {"status": "invalid", "path": str(path), "reason": str(exc)}
    reasons = []
    if native.get("generated_text") != row.get("generated_text"):
        reasons.append("generated_text mismatch")
    if not isinstance(row.get("generated_token_ids"), list) or native.get("generated_token_ids") != row["generated_token_ids"]:
        reasons.append("missing or mismatched generated_token_ids")
    if identity_fingerprint is not None and native.get("fingerprint") != identity_fingerprint:
        reasons.append("run identity fingerprint mismatch")
    if reasons:
        return {"status": "invalid", "path": str(path), "reasons": reasons}
    return {"status": "matched", "path": str(path), "terminated_by_eos": native.get("terminated_by_eos") is True,
            "truncated": native.get("truncated"), "error": bool(native.get("error") or native.get("inference_error")),
            "identity_fingerprint_checked": identity_fingerprint is not None}


def _format_diagnostic(rows: list[dict[str, Any]], cases: list[dict[str, Any]], identities: dict[str, str | None]) -> dict[str, Any]:
    by_id = {case["sample_id"]: case for case in cases}
    cells = defaultdict(list)
    for row in rows:
        if row.get("method") == "text_coordinates":
            cells[row["sample_id"]].append(row)
    records, reasons, counters = [], Counter(), Counter()
    for sid, members in cells.items():
        if len(members) != 1:
            continue
        row, case = members[0], by_id[sid]
        native = _native_evidence(row, Path(row["_run_dir"]), identities.get(row["_run_dir"]))
        text = row.get("generated_text")
        cap_values = [row[key] for key in ("cap_reached", "truncated", "token_cap_reached", "hit_token_cap") if isinstance(row.get(key), bool)]
        if native["status"] == "matched" and isinstance(native.get("truncated"), bool):
            cap_values.append(native["truncated"])
        capped = any(cap_values)
        no_cap_evidence = bool(cap_values) and not capped
        eos = row.get("terminated_by_eos") is True or native["status"] == "matched" and native.get("terminated_by_eos") is True
        newline = isinstance(text, str) and text.endswith(("\n", "\r"))
        complete = newline or eos
        error = bool(row.get("error") or row.get("inference_error") or row.get("execution_error") or native.get("error"))
        normalized = parse_offline_lax_click(text)
        exclusions = []
        if error:
            exclusions.append("recorded_error")
        if capped:
            exclusions.append("token_cap_reached")
        elif not no_cap_evidence:
            exclusions.append("no_explicit_uncapped_evidence")
        if not complete:
            exclusions.append("no_newline_or_matched_eos_completion")
        if normalized is None:
            exclusions.append("not_whole_integer_CLICK_with_comma_or_whitespace")
        size = row.get("model_image_size")
        if normalized is not None and _pixel_normalized([0, 0], size) is None:
            exclusions.append("invalid_model_image_size")
        eligible = not exclusions
        point = None
        hit = None
        if eligible:
            # Exactly the coordinate runtime's pixel mapping, without a GT
            # projection, bbox snapping, offset search or test-time threshold.
            point = [int(round(normalized[0] * (size[0] - 1) / 1000)), int(round(normalized[1] * (size[1] - 1) / 1000))]
            if _grounded(case):
                hit = point_in_geometry(_pixel_normalized(point, size), case["target_geometry"])
            counters["offline_lax_clicks"] += 1
            counters["offline_lax_negative_clicks"] += not case["expected_actionable"]
            counters["offline_lax_grounded_hits"] += hit is True
            if row.get("parse_valid") is not True:
                counters["newly_recoverable_strict_invalid_clicks"] += 1
                counters["newly_recoverable_grounded_hits"] += hit is True
                counters["newly_recoverable_grounded_misses"] += hit is False
                counters["newly_recoverable_negative_clicks"] += not case["expected_actionable"]
        for reason in exclusions:
            reasons[reason] += 1
        records.append({"sample_id": sid, "run_dir": row["_run_dir"], "strict_parse_valid": row.get("parse_valid"),
                        "strict_action": row.get("action"), "generated_text": text,
                        "completion_evidence": {"newline": newline, "eos": eos, "matched_native": native},
                        "uncapped_evidence": no_cap_evidence, "eligible_offline_lax_click": eligible,
                        "exclusion_reasons": exclusions, "offline_normalized_point_0_1000": list(normalized) if normalized else None,
                        "offline_pixel_point": point, "offline_grounding_hit": hit,
                        "expected_actionable": case["expected_actionable"], "task_kind": case["task_kind"]})
    grounded_expected = sum(_grounded(case) for case in cases)
    return {"label": "offline_lax_format_diagnostic_only", "formal_strict_actions_modified": False,
            "n_expected": len(cases), "n_unique_text_coordinate_rows": len(records),
            "missing_sample_ids": [sid for sid in by_id if sid not in cells],
            "duplicate_sample_ids": [sid for sid, members in cells.items() if len(members) > 1],
            "counts": dict(counters), "exclusion_reason_counts": dict(reasons),
            "offline_lax_grounded_hit_on_expected_positives": _rate(counters["offline_lax_grounded_hits"], grounded_expected),
            "protocol": {"pattern": LAX_CLICK.pattern, "completion": "requires terminal newline or EOS from the row/matched native cache", "cap": "requires explicit uncapped evidence and no cap flag", "coordinates": "0..1000 integer pair, round(value*(image_size-1)/1000), then the formal normalized geometry test", "no_gt_prediction_repair": True},
            "records": records,
            "interpretation": "This is a counterfactual parser-only diagnostic, not an official score. Recovered negative clicks are retained; a permissive parser can introduce errors as well as recover valid coordinates. The primary report must remain strict."}


def run_public_diagnostics(run_dirs: Sequence[str | Path], output_dir: str | Path) -> dict[str, Any]:
    """Read one/many complete public run directories and write separate analysis."""
    if not run_dirs:
        raise ValueError("at least one run directory is required")
    directories = [Path(directory).resolve() for directory in run_dirs]
    if len(set(directories)) != len(directories):
        raise ValueError("run directories must be unique")
    output = Path(output_dir).resolve()
    if output in directories or any(output == directory / "evaluation" for directory in directories):
        raise ValueError("diagnostics output must be separate from a run root and its formal evaluation")
    cases_by_key, rows, runs, skipped, extras, official, identities = {}, [], [], [], [], [], {}
    for directory in directories:
        manifest_path, predictions_path = directory / "selected_manifest.jsonl", directory / "predictions.jsonl"
        if not manifest_path.is_file() or not predictions_path.is_file():
            skipped.append({"run_dir": str(directory), "reason": "selected_manifest.jsonl or predictions.jsonl missing"})
            continue
        cases = _load_manifest(manifest_path)
        by_id = {case["sample_id"]: case for case in cases}
        for case in cases:
            key = case["dataset"], case["sample_id"]
            if key in cases_by_key and any(case[field] != cases_by_key[key][field] for field in ("instruction", "expected_actionable", "task_kind", "target_geometry")):
                raise ValueError(f"conflicting manifests for {key}")
            cases_by_key[key] = case
        identity_path = directory / "identity.json"
        identity = json.loads(identity_path.read_text(encoding="utf-8")) if identity_path.is_file() else None
        # Same canonicalization as the benchmark stable_hash, without importing
        # the actor/runner modules or their model dependencies.
        identities[str(directory)] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest() if identity is not None else None
        run_rows = _read_jsonl(predictions_path)
        for index, raw in enumerate(run_rows):
            sid = raw.get("sample_id")
            if sid not in by_id or raw.get("dataset", by_id[sid]["dataset"]) != by_id[sid]["dataset"]:
                extras.append({"run_dir": str(directory), "raw_index": index, "sample_id": sid, "dataset": raw.get("dataset")})
                continue
            rows.append({**raw, "dataset": by_id[sid]["dataset"], "_run_dir": str(directory), "_raw_index": index})
        metric_path = directory / "evaluation" / "metrics.json"
        if metric_path.is_file():
            metric = json.loads(metric_path.read_text(encoding="utf-8"))
            for method in GATE_METHODS:
                entry = metric.get("methods", {}).get(method)
                if entry is None:
                    continue
                for dataset, ds in entry.get("by_dataset", {}).items():
                    official.append({"run_dir": str(directory), "dataset": dataset, "method": method,
                                     "source": str(metric_path), "n": ds.get("n"), "n_expected": ds.get("n_expected"),
                                     "metrics": {name: ds.get("metrics", {}).get(name) for name in
                                                 ("localization_hit_ignoring_gate", "quantization_oracle", "quantization_oracle_coverage", "localization_hit_given_oracle_reachable", "action_aware_correct")}})
        runs.append({"run_dir": str(directory), "n_manifest_cases": len(cases), "n_raw_rows": len(run_rows),
                     "formal_metrics_available": metric_path.is_file(), "identity_fingerprint_available": identity is not None})
    datasets = {}
    for dataset in sorted({key[0] for key in cases_by_key}):
        cases = [case for (name, _), case in cases_by_key.items() if name == dataset]
        ds_rows = [row for row in rows if row["dataset"] == dataset]
        datasets[dataset] = {"n_expected_cases": len(cases),
                             "gates": {method: _gate_diagnostic(ds_rows, cases, method) for method in GATE_METHODS},
                             "offline_lax_coordinates": _format_diagnostic(ds_rows, cases, identities)}
    summary = {"schema_version": 1, "analysis_type": "read_only_public_frozen_policy_diagnostics",
               "status": "partial_missing_runs" if skipped else "complete", "runs": runs, "skipped_runs": skipped,
               "unexpected_prediction_rows": extras, "datasets": datasets, "formal_evaluation_references": official,
               "no_training_or_threshold_selection": True, "formal_actions_or_scores_modified": False,
               "interpretation": ["AUROC concerns probability ranking, whereas the logged allowance uses an unchanged old validation threshold. Good ranking can coexist with poor transfer calibration.", "Only datasets with both available probability classes have an AUROC; missing readouts/duplicates remain visible in class-specific coverage.", "Localization/gate intersections concern static screenshot predictions, not actual computer execution.", "The comma-coordinate analysis is explicitly offline and permissive only about the separator; no coordinate is snapped, optimized or repaired using GT."]}
    output.mkdir(parents=True, exist_ok=True)
    (output / "diagnostics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    (output / "report.md").write_text(_report(summary), encoding="utf-8")
    return summary


def _report(summary: dict[str, Any]) -> str:
    def rate(value):
        return "不可评估" if value["rate"] is None else f"{value['numerator']}/{value['denominator']} ({value['rate']:.1%})"
    lines = ["# 冻结策略公开集诊断", "", "这是独立只读分析：不训练、不选新阈值、不修改正式动作或 strict 成绩。概率 AUROC 只衡量排序；旧阈值下的放行率反映迁移后的校准，两者需要一起判断。静态定位命中不代表执行电脑。", "", "| 数据集 | Gate | 概率覆盖 | AUROC | 正例放行（观测） | 负例误点（观测） | 定位命中却拒绝 | 定位错却放行 |", "| --- | --- | --- | --- | --- | --- | --- | --- |"]
    for dataset, result in summary["datasets"].items():
        for method, gate in result["gates"].items():
            auc = gate["click_probability_auroc"]["value"]
            auc_text = "不可评估" if auc is None else f"{auc:.4f}"
            lines.append(f"| {dataset} | {method} | {rate(gate['click_probability_coverage'])} | {auc_text} | {rate(gate['positive_allow_rate_observed'])} | {rate(gate['negative_false_click_observed'])} | {gate['cross_counts']['localization_hit_but_gate_rejected']} | {gate['cross_counts']['localization_miss_but_gate_allowed']} |")
    lines += ["", "仅有正例的数据集不能计算二元 AUROC。概率缺失/错误和重复样本不混入 AUROC，JSON 同时给出正负各自的概率样本数、完整预期分母与缺失 ID。缺失不是正确拒绝；另提供正例放行下界与负例误点击上界。定位点缺失与实际坐标错误分开统计。概率分布（mean/min/max/p05/p25/p50/p75/p95）、冻结阈值、逐错误 ID 见 JSON。", "", "| 数据集 | 离线宽松整条CLICK数 | strict无效但可恢复 | 其中定位命中 | 其中定位错误 | 其中负例CLICK | 宽松定位命中/全部预期定位正例 |", "| --- | --- | --- | --- | --- | --- | --- |"]
    for dataset, result in summary["datasets"].items():
        fmt, c = result["offline_lax_coordinates"], result["offline_lax_coordinates"]["counts"]
        lines.append(f"| {dataset} | {c.get('offline_lax_clicks', 0)} | {c.get('newly_recoverable_strict_invalid_clicks', 0)} | {c.get('newly_recoverable_grounded_hits', 0)} | {c.get('newly_recoverable_grounded_misses', 0)} | {c.get('newly_recoverable_negative_clicks', 0)} | {rate(fmt['offline_lax_grounded_hit_on_expected_positives'])} |")
    lines += ["", "以上只叫‘离线宽松格式诊断’，主报告继续使用正式 strict 评分。仅允许完整整条 CLICK 后两个 0..1000 整数，以 ASCII 逗号或空白分隔，必须明确未触及 token cap、无 error，并以换行或匹配的原 native 缓存 EOS 证明完成。额外说明文字、括号、浮点、越界及未完成输出都拒绝。像素转换与原 decoder 相同，不用 GT 修复预测；恢复的负例点击同样保留，不能只挑恢复后命中的正例。", "", "ConflictGUI action_feasibility 只解释可行性判断；未把原作者多动作成功率或实际点击执行成绩混入。已有正式评测中的无条件定位/oracle 数值以来源文件引用保存，诊断不会替换其计分。", "", f"处理运行 {len(summary['runs'])} 个，缺失运行 {len(summary['skipped_runs'])} 个；状态 {summary['status']}。共享轨迹不提供各方法独立延迟结论。产物 diagnostics.json 与本报告。"]
    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/public_v1/analysis"))
    args = parser.parse_args(argv)
    result = run_public_diagnostics(args.runs, args.output)
    print(json.dumps({"status": result["status"], "datasets": list(result["datasets"]), "output": str(args.output.resolve())}, ensure_ascii=False))


if __name__ == "__main__":
    main()
