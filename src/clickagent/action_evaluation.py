"""Evaluate observed browser actions, never cached screenshot classification.

The evaluator does not load a VLM, fit a probe, or choose a threshold. Its input
is an auditable log of fresh observe -> infer -> execute -> verify episodes.
"""

from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ACTION_LABELS = ("click", "disabled", "absent")
ACTION_METHODS = ("prefill_hidden", "post_hidden", "text_decision_shared", "text_coordinates")
TIMING_STAGES = ("observe", "inference", "policy", "execute", "verify", "total")


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    value = float(value)
    return value if math.isfinite(value) and value >= 0 else None


def _first(mapping: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in mapping:
            return mapping[name]
    return None


def _case(row: dict[str, Any], *, warmup: bool = False) -> dict[str, str]:
    sid = row.get("sample_id", row.get("episode_id"))
    group = row.get("group_id", row.get("group"))
    label = row.get("label")
    metadata = row.get("metadata", {})
    domain = row.get("domain", metadata.get("domain", "unspecified") if isinstance(metadata, dict) else "unspecified")
    if sid is None or not str(sid).strip() or group is None or not str(group).strip():
        raise ValueError("Every action case requires nonempty sample_id and group_id")
    if label not in ACTION_LABELS:
        raise ValueError(f"{sid}: label must be click, disabled or absent")
    if not warmup and row.get("split", "test") != "test":
        raise ValueError(f"{sid}: action evaluation accepts test episodes only")
    if domain not in ("iid", "ood", "unspecified"):
        raise ValueError(f"{sid}: domain must be iid or ood")
    return {"sample_id": str(sid), "group_id": str(group), "label": str(label), "domain": str(domain)}


def _counter(environment: dict[str, Any], snake: str, camel: str) -> int | None:
    value = _number(_first(environment, snake, camel))
    return int(value) if value is not None and value.is_integer() else None


def _pixel_evidence(row: dict[str, Any]) -> tuple[bool | None, str]:
    verify = row.get("verify", {})
    verify = verify if isinstance(verify, dict) else {}
    verification = row.get("verification", {})
    verification = verification if isinstance(verification, dict) else {}
    for mapping, prefix in ((verify, "verify"), (verification, "verification"), (row, "row")):
        value = _first(mapping, "pixel_changed", "visual_changed")
        if isinstance(value, bool):
            return value, f"{prefix}.pixel_changed"
        count = _number(_first(mapping, "changed_pixel_count", "pixel_change_count"))
        if count is not None:
            return count > 0, f"{prefix}.changed_pixel_count"
        ratio = _number(_first(mapping, "changed_pixel_ratio", "pixel_change_ratio"))
        if ratio is not None:
            return ratio > 0, f"{prefix}.changed_pixel_ratio"
    return None, "missing"


def _target_hit(row: dict[str, Any], action: dict[str, Any], save_delta: int | None) -> tuple[bool | None, str]:
    after = row.get("env_after", {})
    execution = row.get("execution", {})
    sources = [row, execution if isinstance(execution, dict) else {}, after if isinstance(after, dict) else {}]
    for mapping in sources:
        hit = _first(mapping, "target_hit", "hit_target", "is_target")
        if isinstance(hit, bool):
            return hit, "recorded_target_hit"
        mapped = _first(mapping, "mapped_target", "mappedtarget", "action_target")
        if isinstance(mapped, dict):
            flag = _first(mapped, "target_hit", "is_target")
            if isinstance(flag, bool):
                return flag, "mapped_target_dictionary"
            text = _first(mapped, "text", "label", "name")
            if isinstance(text, str):
                return text.strip().lower() in ("保存", "save", "save_button"), "mapped_target_dictionary"
        if isinstance(mapped, str):
            if mapped.lower() in ("save", "target", "save_button", "保存"):
                return True, "mapped_target"
            if mapped.lower() in ("none", "other", "background", "outside", "missing") or mapped.lower().startswith("other"):
                return False, "mapped_target"
    # A successful save handler is stronger evidence than an assumed miss.
    if save_delta is not None and save_delta > 0 and action["type"] == "click":
        return True, "observed_save_activation"
    return None, "missing"


def _normalize(row: dict[str, Any], index: int, methods: tuple[str, ...]) -> dict[str, Any]:
    warmup = row.get("warmup", False)
    if not isinstance(warmup, bool):
        raise ValueError("warmup must be a boolean")
    facts = _case(row, warmup=warmup)
    method = row.get("method")
    if method not in methods:
        raise ValueError(f"Unknown method {method!r}; expected {methods}")
    if row.get("fallback") is True:
        raise ValueError(f"{facts['sample_id']}: fallback=true violates this benchmark protocol")
    action = row.get("action", {})
    if not isinstance(action, dict) or action.get("type") not in ("click", "wait"):
        raise ValueError(f"{facts['sample_id']}: action must be a click or wait dictionary")
    emitted = action["type"] == "click"
    if emitted:
        if any(not isinstance(action.get(axis), (int, float)) or isinstance(action.get(axis), bool) or not math.isfinite(action[axis]) for axis in ("x", "y")):
            raise ValueError(f"{facts['sample_id']}: click requires finite pixel coordinates")
    before, after = row.get("env_before", {}), row.get("env_after", {})
    before = before if isinstance(before, dict) else {}
    after = after if isinstance(after, dict) else {}
    deltas: dict[str, int | None] = {}
    for snake, camel in (("save_count", "saveCount"), ("other_action_count", "otherActionCount"), ("other_attempt_count", "otherAttemptCount")):
        a, b = _counter(before, snake, camel), _counter(after, snake, camel)
        if a is not None and b is not None and b < a:
            raise ValueError(f"{facts['sample_id']}: {snake} decreased during an episode")
        deltas[snake] = b - a if a is not None and b is not None else None
    save_delta, other_delta = deltas["save_count"], deltas["other_action_count"]
    environment_known = save_delta is not None and other_delta is not None
    pixel_changed, pixel_source = _pixel_evidence(row)
    target_hit, target_source = _target_hit(row, action, save_delta)
    if emitted and save_delta is not None and save_delta > 0 and target_hit is False:
        raise ValueError(f"{facts['sample_id']}: save activation contradicts target_hit=false")
    feasible = facts["label"] == "click"
    execution_error = bool(row.get("execution_error"))
    parse_valid = row.get("parse_valid") if isinstance(row.get("parse_valid"), bool) else None
    save_success = bool(feasible and emitted and save_delta == 1 and other_delta == 0 and pixel_changed is True and not execution_error)
    behavioral_correct = save_success if feasible else bool(not emitted and environment_known and save_delta == 0 and other_delta == 0)
    correct = bool(behavioral_correct and parse_valid is True)
    side_effects = None if not environment_known else int(other_delta + (save_delta if not feasible else max(save_delta - 1, 0)))
    timing = row.get("timing_ms", {})
    timing = timing if isinstance(timing, dict) else {}
    measured = row.get("latency_measured") is True
    load_excluded = row.get("model_load_included") is False
    timing_values = {stage: _number(timing.get(stage)) for stage in TIMING_STAGES}
    source = str(row.get("latency_source", row.get("timing_source", "measured" if measured else "unspecified"))).lower()
    prohibited_source = any(word in source for word in ("cache", "summed", "component_sum", "inferred", "estimated"))
    latency_valid = measured and load_excluded and not prohibited_source and timing_values["total"] is not None
    reasons = []
    if not measured:
        reasons.append("latency_measured is not true")
    if not load_excluded:
        reasons.append("model load exclusion is not certified")
    if prohibited_source:
        reasons.append("latency source denotes cached/summed/estimated cost")
    if timing_values["total"] is None:
        reasons.append("actual timing_ms.total is missing or invalid")
    if timing_values["total"] is not None and any(value is not None and value > timing_values["total"] + 1.0 for stage, value in timing_values.items() if stage != "total"):
        latency_valid = False
        reasons.append("a stage exceeds the recorded total by more than 1 ms")
    return {
        **facts, "raw_index": index, "episode_id": str(row.get("episode_id", facts["sample_id"])),
        "method": method, "warmup": warmup, "cold_start": row.get("cold_start") is True,
        "action": action, "emitted_click": emitted, "feasible": feasible,
        "save_success": save_success, "action_correct": correct,
        "behavioral_correct": behavioral_correct, "execution_error": execution_error,
        "negative_false_click_attempt": bool(not feasible and emitted),
        "wrong_target_click": bool(feasible and emitted and target_hit is False),
        "unknown_enabled_click_target": bool(feasible and emitted and target_hit is None),
        "target_hit": target_hit, "target_hit_source": target_source,
        "environment_evidence_known": environment_known, "pixel_changed": pixel_changed,
        "pixel_evidence_source": pixel_source, "save_delta": save_delta, "other_action_delta": other_delta,
        "other_attempt_delta": deltas["other_attempt_count"], "wrong_side_effect_events": side_effects,
        "parse_valid": parse_valid,
        "n_generated_tokens": _number(row.get("n_generated_tokens")),
        "latency_valid": latency_valid, "latency_exclusion_reasons": reasons,
        "latency_measured": measured, "model_load_excluded": load_excluded,
        "timing_ms": timing_values,
    }


def _rate(numerator: int, denominator: int) -> dict[str, Any]:
    return {"numerator": int(numerator), "denominator": int(denominator), "rate": numerator / denominator if denominator else None}


def _distribution(values: Iterable[float | None]) -> dict[str, Any]:
    array = np.asarray([value for value in values if value is not None], dtype=float)
    return {"n": len(array), "mean": float(array.mean()) if len(array) else None, "p50": float(np.median(array)) if len(array) else None, "p95": float(np.percentile(array, 95)) if len(array) else None}


def _latency(rows: list[dict[str, Any]]) -> dict[str, Any]:
    eligible = [row for row in rows if row["latency_valid"]]
    return {
        "n_cases": len(rows), "n_measured_eligible": len(eligible), "n_excluded": len(rows) - len(eligible),
        "status": "ok" if eligible else "unavailable",
        "total_ms": _distribution(row["timing_ms"]["total"] for row in eligible),
        "stages_ms": {stage: _distribution(row["timing_ms"][stage] for row in eligible) for stage in TIMING_STAGES if stage != "total"},
        "excluded_sample_ids": [row["sample_id"] for row in rows if not row["latency_valid"]],
    }


def _metric_masks(rows: list[dict[str, Any]]) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    n = len(rows)
    feasible = np.asarray([row["feasible"] for row in rows], dtype=bool)
    success = np.asarray([row["save_success"] for row in rows], dtype=bool)
    emitted = np.asarray([row["emitted_click"] for row in rows], dtype=bool)
    masks = {
        "feasible_save_success": (success, feasible),
        "action_correct": (np.asarray([row["action_correct"] for row in rows]), np.ones(n, dtype=bool)),
        "behavioral_correct": (np.asarray([row["behavioral_correct"] for row in rows]), np.ones(n, dtype=bool)),
        "negative_false_click_attempt": (emitted & ~feasible, ~feasible),
        "emitted_click_precision": (success, emitted),
        "abstention": (~emitted, np.ones(n, dtype=bool)),
        "feasible_abstention": (~emitted & feasible, feasible),
        "invalid_parse": (np.asarray([row["parse_valid"] is False for row in rows]), np.ones(n, dtype=bool)),
    }
    for label in ("disabled", "absent"):
        state = np.asarray([row["label"] == label for row in rows], dtype=bool)
        masks[f"negative_false_click_attempt_{label}"] = (emitted & state, state)
    return masks


def _ci(point: float | None, values: np.ndarray) -> dict[str, Any]:
    finite = np.asarray(values)[np.isfinite(values)]
    return {"point": point, "ci95": np.percentile(finite, [2.5, 97.5]).tolist() if len(finite) else None, "n_valid_bootstrap": len(finite)}


def _weighted_median(values: np.ndarray, row_weights: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="stable")
    sorted_values = np.asarray(values)[order]
    weights = row_weights[:, order]
    totals = weights.sum(axis=1).astype(int)
    cumulative = np.cumsum(weights, axis=1)
    lower = np.argmax(cumulative > ((totals - 1) // 2)[:, None], axis=1)
    upper = np.argmax(cumulative > (totals // 2)[:, None], axis=1)
    answer = (sorted_values[lower] + sorted_values[upper]) / 2
    answer[totals == 0] = np.nan
    return answer


def _bootstrap(rows: list[dict[str, Any]], seed: int, iterations: int) -> dict[str, Any]:
    groups, group_index = np.unique([row["group_id"] for row in rows], return_inverse=True)
    base = {"unit": "group_id", "n_groups": len(groups), "iterations": iterations, "seed": seed}
    if iterations == 0 or len(groups) < 2:
        return {**base, "status": "disabled" if iterations == 0 else "unavailable", "reason": "bootstrap disabled or fewer than two groups"}
    draws = np.random.default_rng(seed).multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=iterations)
    result: dict[str, Any] = {**base, "status": "ok", "metrics": {}}
    for name, (numerator, denominator) in _metric_masks(rows).items():
        top = draws @ np.bincount(group_index, weights=numerator, minlength=len(groups))
        bottom = draws @ np.bincount(group_index, weights=denominator, minlength=len(groups))
        rates = np.divide(top, bottom, out=np.full(iterations, np.nan), where=bottom > 0)
        result["metrics"][name] = _ci(float(numerator.sum() / denominator.sum()) if denominator.sum() else None, rates)
    eligible = np.asarray([row["latency_valid"] for row in rows], dtype=bool)
    if eligible.any():
        values = np.asarray([row["timing_ms"]["total"] if row["latency_valid"] else 0.0 for row in rows])
        top = draws @ np.bincount(group_index, weights=values, minlength=len(groups))
        bottom = draws @ np.bincount(group_index, weights=eligible, minlength=len(groups))
        means = np.divide(top, bottom, out=np.full(iterations, np.nan), where=bottom > 0)
        medians = _weighted_median(values[eligible], draws[:, group_index[eligible]])
        result["latency_total_mean_ms"] = _ci(float(values[eligible].mean()), means)
        result["latency_total_p50_ms"] = _ci(float(np.median(values[eligible])), medians)
    return result


def _statistics(rows: list[dict[str, Any]], seed: int, iterations: int, *, bootstrap: bool = True) -> dict[str, Any]:
    metrics = {name: _rate(int(numerator.sum()), int(denominator.sum())) for name, (numerator, denominator) in _metric_masks(rows).items()}
    metrics.update({
        "wrong_target_clicks_on_feasible": sum(row["wrong_target_click"] for row in rows),
        "unknown_enabled_click_target_count": sum(row["unknown_enabled_click_target"] for row in rows),
        "actual_wrong_side_effect_event_count": sum(row["wrong_side_effect_events"] or 0 for row in rows),
        "actual_wrong_side_effect_row_count": sum(bool(row["wrong_side_effect_events"]) for row in rows),
        "other_action_activation_count": sum(row["other_action_delta"] or 0 for row in rows),
        "other_attempt_count": sum(row["other_attempt_delta"] or 0 for row in rows),
        "unknown_environment_evidence_count": sum(not row["environment_evidence_known"] for row in rows),
        "missing_pixel_evidence_count": sum(row["pixel_changed"] is None for row in rows),
        "unknown_parse_valid_count": sum(row["parse_valid"] is None for row in rows),
        "execution_error_count": sum(row["execution_error"] for row in rows),
        "invalid_parse_wait_count": sum(row["parse_valid"] is False and not row["emitted_click"] for row in rows),
        "parse_valid": _rate(sum(row["parse_valid"] is True for row in rows), sum(row["parse_valid"] is not None for row in rows)),
    })
    subsets = {
        "all_cases": rows,
        "feasible_label_click": [row for row in rows if row["feasible"]],
        "negative_label_cases": [row for row in rows if not row["feasible"]],
        "emitted_click_cases": [row for row in rows if row["emitted_click"]],
        "wait_cases": [row for row in rows if not row["emitted_click"]],
        "successful_feasible": [row for row in rows if row["save_success"]],
        "without_cold_start": [row for row in rows if not row["cold_start"]],
    }
    return {"n": len(rows), "n_groups": len({row["group_id"] for row in rows}), "metrics": metrics, "latency": {name: _latency(subset) for name, subset in subsets.items()}, "generated_tokens": _distribution(row["n_generated_tokens"] for row in rows), "group_bootstrap": _bootstrap(rows, seed, iterations) if bootstrap else {"status": "not_computed_for_this_slice"}}


def _paired_latency(rows_a: list[dict[str, Any]], rows_b: list[dict[str, Any]], seed: int, iterations: int) -> dict[str, Any]:
    a, b = {row["sample_id"]: row for row in rows_a}, {row["sample_id"]: row for row in rows_b}
    ids = sorted(set(a) & set(b))
    eligible = [sid for sid in ids if a[sid]["latency_valid"] and b[sid]["latency_valid"]]
    if not eligible:
        return {"status": "unavailable", "n_matched_cases": len(ids), "n": 0, "sample_ids": [], "reason": "no paired actually measured latencies"}
    x = np.asarray([a[sid]["timing_ms"]["total"] for sid in eligible], dtype=float)
    y = np.asarray([b[sid]["timing_ms"]["total"] for sid in eligible], dtype=float)
    groups, group_index = np.unique([a[sid]["group_id"] for sid in eligible], return_inverse=True)
    result: dict[str, Any] = {
        "status": "ok", "n_matched_cases": len(ids), "n": len(eligible), "n_groups": len(groups), "sample_ids": eligible,
        "p50_a_ms": float(np.median(x)), "p50_b_ms": float(np.median(y)),
        "p50_difference_ms": float(np.median(x) - np.median(y)),
        "p50_ratio_a_over_b": float(np.median(x) / np.median(y)) if np.median(y) > 0 else None,
        "paired_difference_p50_ms": float(np.median(x - y)), "paired_difference_mean_ms": float(np.mean(x - y)),
        "paired_ratio_p50": float(np.median(x[y > 0] / y[y > 0])) if np.any(y > 0) else None,
    }
    if iterations == 0 or len(groups) < 2:
        result["group_bootstrap"] = {"status": "disabled" if iterations == 0 else "unavailable", "reason": "bootstrap disabled or fewer than two groups"}
        return result
    draws = np.random.default_rng(seed).multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=iterations)
    weights = draws[:, group_index]
    means = (weights @ (x - y)) / weights.sum(axis=1)
    p50_differences = _weighted_median(x, weights) - _weighted_median(y, weights)
    result["group_bootstrap"] = {"status": "ok", "unit": "group_id", "seed": seed, "iterations": iterations,
        "mean_difference_ms": _ci(float(np.mean(x - y)), means),
        "p50_difference_ms": _ci(float(np.median(x) - np.median(y)), p50_differences)}
    return result


def evaluate_action_rows(
    rows: Iterable[dict[str, Any]], output_dir: Path | str, seed: int = 144,
    bootstrap_iterations: int = 1000, *, expected_methods: Iterable[str] | None = None,
    expected_cases: Iterable[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Score fresh action logs with physical effects, coverage and clustered CIs.

    expected_cases should contain the intended test manifest rows. Without it,
    cases absent from every method cannot be discovered, and this is disclosed.
    Duplicate primary (sample_id, method) keys are ambiguous, never cherry-picked.
    Their missing proof is counted in separate conservative coverage-adjusted rates.
    """
    if not isinstance(bootstrap_iterations, int) or isinstance(bootstrap_iterations, bool) or bootstrap_iterations < 0:
        raise ValueError("bootstrap_iterations must be a nonnegative integer")
    methods = tuple(ACTION_METHODS if expected_methods is None else expected_methods)
    if not methods or len(set(methods)) != len(methods):
        raise ValueError("expected_methods must be nonempty and unique")
    normalized = [_normalize(dict(row), index, methods) for index, row in enumerate(rows)]
    primary = [row for row in normalized if not row["warmup"]]
    if not primary:
        raise ValueError("No non-warmup test action rows to evaluate")
    facts = {}
    if expected_cases is not None:
        for raw in expected_cases:
            case = _case(dict(raw))
            if case["sample_id"] in facts:
                raise ValueError("Duplicate sample_id in expected_cases")
            facts[case["sample_id"]] = case
    source = "explicit expected_cases" if expected_cases is not None else "union of observed non-warmup sample IDs; cases missing from every method cannot be detected"
    for row in primary:
        case = {name: row[name] for name in ("sample_id", "group_id", "label", "domain")}
        if row["sample_id"] in facts and facts[row["sample_id"]] != case:
            raise ValueError(f"Case facts differ between methods or expected_cases: {row['sample_id']}")
        if expected_cases is not None and row["sample_id"] not in facts:
            raise ValueError(f"Unexpected sample_id outside expected_cases: {row['sample_id']}")
        facts[row["sample_id"]] = case
    grouped_facts: dict[str, list[dict[str, str]]] = defaultdict(list)
    for case in facts.values():
        grouped_facts[case["group_id"]].append(case)
    for group, members in grouped_facts.items():
        if len({case["domain"] for case in members}) > 1:
            raise ValueError(f"Scenario group {group!r} crosses IID/OOD domains")
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in primary:
        buckets[row["method"], row["sample_id"]].append(row)
    duplicate_keys = [{"method": method, "sample_id": sid, "raw_indices": [row["raw_index"] for row in records]} for (method, sid), records in buckets.items() if len(records) > 1]
    available: dict[str, list[dict[str, Any]]] = {method: [] for method in methods}
    incomplete_groups = [group for group, members in grouped_facts.items() if len(members) != 3 or {case["label"] for case in members} != set(ACTION_LABELS)]
    coverage = {"universe_source": source, "n_expected_cases": len(facts), "n_expected_groups": len(grouped_facts), "n_complete_expected_triplets": len(grouped_facts) - len(incomplete_groups), "incomplete_expected_group_ids": incomplete_groups, "expected_label_counts": {label: sum(row["label"] == label for row in facts.values()) for label in ACTION_LABELS}, "duplicates": duplicate_keys, "methods": {}}
    for method in methods:
        available[method] = [records[0] for (name, _), records in buckets.items() if name == method and len(records) == 1]
        present = {row["sample_id"] for row in available[method]}
        missing = sorted(sid for sid in facts if (method, sid) not in buckets)
        ambiguous = sorted(sid for (name, sid), records in buckets.items() if name == method and len(records) > 1)
        coverage["methods"][method] = {"n_expected": len(facts), "n_unique_scored": len(present), "n_raw_primary_rows": sum(len(records) for (name, _), records in buckets.items() if name == method), "coverage": len(present) / len(facts), "missing_sample_ids": missing, "ambiguous_duplicate_sample_ids": ambiguous, "complete": len(present) == len(facts)}
    summary: dict[str, Any] = {
        "schema_version": 1, "experiment_type": "fresh_browser_action_execution", "seed": seed,
        "bootstrap_iterations": bootstrap_iterations, "coverage": coverage, "methods": {},
        "warmup": {"n": sum(row["warmup"] for row in normalized), "by_method": {method: sum(row["warmup"] and row["method"] == method for row in normalized) for method in methods}},
        "cold_start_records": [{"raw_index": row["raw_index"], "sample_id": row["sample_id"], "method": row["method"], "warmup": row["warmup"], "total_ms": row["timing_ms"]["total"], "model_load_excluded": row["model_load_excluded"]} for row in normalized if row["cold_start"]],
        "measurement_audit": {"primary_latency_eligible": sum(row["latency_valid"] for row in primary), "primary_latency_ineligible": [{"raw_index": row["raw_index"], "sample_id": row["sample_id"], "method": row["method"], "reasons": row["latency_exclusion_reasons"]} for row in primary if not row["latency_valid"]], "total_is_never_reconstructed_from_components": True},
        "supervision_note": "Hidden gates use prior labeled train data and val-only 1% calibration; the shared locator uses old 180 enabled training examples + 60 enabled validation examples. text_coordinates is zero-shot. Frozen VLM parameters do not make these supervision budgets equal.",
        "latency_note": "Primary total_ms must be freshly measured from observe start through verification end, exclude model loading, and include wait/failure cases. No old cache timing or component sums are substituted. Cold primary rows remain included unless warmup=true; cold records and without_cold_start summaries are separate.",
    }
    for method, records in available.items():
        stats = _statistics(records, seed, bootstrap_iterations)
        expected_feasible = coverage["expected_label_counts"]["click"]
        stats.update({"status": "complete" if coverage["methods"][method]["complete"] else "partial_coverage", "coverage_adjusted": {
            "feasible_success_over_expected": _rate(stats["metrics"]["feasible_save_success"]["numerator"], expected_feasible),
            "action_correct_over_expected": _rate(stats["metrics"]["action_correct"]["numerator"], len(facts)),
            "interpretation": "Missing or ambiguous duplicate cases have not demonstrated success; these are conservative proof-of-completion rates, not invented observed outcomes.",
        }})
        stats["by_domain"] = {domain: _statistics([row for row in records if row["domain"] == domain], seed, bootstrap_iterations) for domain in sorted({row["domain"] for row in facts.values()})}
        stats["by_label"] = {label: _statistics([row for row in records if row["label"] == label], seed, bootstrap_iterations, bootstrap=False) for label in ACTION_LABELS}
        summary["methods"][method] = stats
    labels = coverage["expected_label_counts"]
    summary["always_wait_reference"] = {"reference_only_not_executed": True, "feasible_save_success": _rate(0, labels["click"]), "action_correct": _rate(labels["disabled"] + labels["absent"], len(facts)), "latency": None, "note": "With balanced triplets, always-wait gets 66.7% overall correctness but completes zero enabled saves. This is an analytical reference, not an evaluated policy."}
    priorities = [("prefill_hidden", "text_coordinates"), ("post_hidden", "text_decision_shared")]
    pairs = priorities + [pair for pair in combinations(methods, 2) if pair not in priorities and tuple(reversed(pair)) not in priorities]
    summary["latency_comparisons"] = []
    for method_a, method_b in pairs:
        if method_a not in available or method_b not in available:
            continue
        a, b = available[method_a], available[method_b]
        both_success = {row["sample_id"] for row in a if row["save_success"]} & {row["sample_id"] for row in b if row["save_success"]}
        summary["latency_comparisons"].append({"method_a": method_a, "method_b": method_b, "direction": "a minus b; ratio a / b", "priority_comparison": (method_a, method_b) in priorities,
            "all_paired_cases": _paired_latency(a, b, seed, bootstrap_iterations),
            "successful_feasible_intersection": _paired_latency([row for row in a if row["sample_id"] in both_success], [row for row in b if row["sample_id"] in both_success], seed, bootstrap_iterations),
            "n_both_successful_before_latency_filter": len(both_success),
            "interpretation": "The success intersection is conditional on both policies completing the enabled save, not overall speed superiority. Negative wait cases are never counted as successful saves."})
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    _write_episode_csv(normalized, output_dir / "episode_metrics.csv")
    _write_report(summary, output_dir / "report.md")
    return summary


def load_action_jsonl(path: Path | str) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def evaluate_action_jsonl(path: Path | str, output_dir: Path | str, seed: int = 144, bootstrap_iterations: int = 1000, **kwargs: Any) -> dict[str, Any]:
    return evaluate_action_rows(load_action_jsonl(path), output_dir, seed, bootstrap_iterations, **kwargs)


def _write_episode_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = ("raw_index", "episode_id", "sample_id", "group_id", "label", "domain", "method", "warmup", "cold_start", "emitted_click", "save_success", "action_correct", "behavioral_correct", "execution_error", "negative_false_click_attempt", "wrong_target_click", "target_hit", "pixel_changed", "save_delta", "other_action_delta", "other_attempt_delta", "wrong_side_effect_events", "parse_valid", "n_generated_tokens", "latency_valid")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=[*fields, "total_ms"])
        writer.writeheader()
        for row in rows:
            writer.writerow({**{field: row[field] for field in fields}, "total_ms": row["timing_ms"]["total"]})


def _fmt(value: float | None, *, percent: bool = False) -> str:
    return "不可评估" if value is None else (f"{value:.1%}" if percent else f"{value:.2f}")


def _rate_text(value: dict[str, Any]) -> str:
    return f"{value['numerator']}/{value['denominator']} ({_fmt(value['rate'], percent=True)})"


def _ci_text(value: dict[str, Any] | None) -> str:
    if not value or value.get("ci95") is None:
        return "不可评估"
    lower, upper = value["ci95"]
    return f"[{lower:.1%}, {upper:.1%}]"


def _write_report(summary: dict[str, Any], path: Path) -> None:
    coverage = summary["coverage"]
    lines = ["# 真实浏览器点击与验证", "", "本报告评估实际 observe → inference → policy → execute → verify。旧截图分类 accuracy 不算点击成功。可用保存成功要求：确实发出点击、save_count 恰增 1、other_action_count 不变、执行前后截图有像素变化。对 disabled/absent 发出点击，即使浏览器忽略且没有副作用，也属于误放行。", "", "## 覆盖率与实际完成", "", f"预期 {coverage['n_expected_cases']} 个样本、{coverage['n_expected_groups']} 个场景组。范围来源：{coverage['universe_source']}。", "", "| 方法 | 唯一评分/预期 | 可用保存成功 | 全样本动作正确 | 负例点击尝试 | 发出点击precision |", "| --- | ---: | --- | --- | --- | --- |"]
    for method, result in summary["methods"].items():
        c, m = coverage["methods"][method], result["metrics"]
        lines.append(f"| {method} | {c['n_unique_scored']}/{c['n_expected']} | {_rate_text(m['feasible_save_success'])} | {_rate_text(m['action_correct'])} | {_rate_text(m['negative_false_click_attempt'])} | {_rate_text(m['emitted_click_precision'])} |")
    lines.extend(["", "| 方法 | 场景组数 | 可用保存成功95%CI | 严格动作正确95%CI | 负例点击尝试95%CI | 点击precision95%CI |", "| --- | ---: | --- | --- | --- | --- |"])
    for method, result in summary["methods"].items():
        intervals = result["group_bootstrap"].get("metrics", {})
        lines.append(f"| {method} | {result['n_groups']} | {_ci_text(intervals.get('feasible_save_success'))} | {_ci_text(intervals.get('action_correct'))} | {_ci_text(intervals.get('negative_false_click_attempt'))} | {_ci_text(intervals.get('emitted_click_precision'))} |")
    lines.extend(["", "可用保存成功的分母只含 label=click。主 action_correct 还要求 parse_valid=true；另报不要求解析成功的 behavioral_correct，便于区分正确决策与格式失败后的安全弃权。负例 wait 必须无实际副作用；点击禁用按钮后没有反应不能算正确拒绝。缺失/重复不会直接取共同子集美化指标：缺失 ID、所有重复原始索引和保守的 coverage_adjusted 完成率见 JSON；表内观测指标仅使用唯一日志记录。", "", "## 格式失败与弃权", "", "| 方法 | 严格动作正确 | 行为正确 | 格式失败 | 格式未知行 | 总弃权 | enabled弃权 | 执行异常数 |", "| --- | --- | --- | --- | ---: | --- | --- | ---: |"])
    for method, result in summary["methods"].items():
        m = result["metrics"]
        lines.append(f"| {method} | {_rate_text(m['action_correct'])} | {_rate_text(m['behavioral_correct'])} | {_rate_text(m['invalid_parse'])} | {m['unknown_parse_valid_count']} | {_rate_text(m['abstention'])} | {_rate_text(m['feasible_abstention'])} | {m['execution_error_count']} |")
    lines.extend(["", "格式失败后的 wait 可构成无副作用的安全弃权，却不算解析正确的动作决策。越界执行异常不会修正或夹取坐标，也不能计作成功。", "", "## 分域与负状态", "", "| 方法 | 范围 | 可用保存成功 | 动作正确 | disabled点击尝试 | absent点击尝试 |", "| --- | --- | --- | --- | --- | --- |"])
    for method, result in summary["methods"].items():
        for domain, subset in result["by_domain"].items():
            m = subset["metrics"]
            lines.append(f"| {method} | {domain} | {_rate_text(m['feasible_save_success'])} | {_rate_text(m['action_correct'])} | {_rate_text(m['negative_false_click_attempt_disabled'])} | {_rate_text(m['negative_false_click_attempt_absent'])} |")
    lines.extend(["", "## 错目标与实际副作用", "", "| 方法 | enabled错目标点击 | enabled点击目标不明 | 错误副作用事件/行 | 其他控件尝试 | 环境证据缺失行 |", "| --- | ---: | ---: | --- | ---: | ---: |"])
    for method, result in summary["methods"].items():
        m = result["metrics"]
        lines.append(f"| {method} | {m['wrong_target_clicks_on_feasible']} | {m['unknown_enabled_click_target_count']} | {m['actual_wrong_side_effect_event_count']}/{m['actual_wrong_side_effect_row_count']} | {m['other_attempt_count']} | {m['unknown_environment_evidence_count']} |")
    lines.extend(["", "错误副作用与尝试分开：其他禁用控件可能不触发 other_action_count，但点击尝试仍被记录。没有足够 target_hit/mapped_target/激活证据时明确报告目标不明，不伪装成零错目标。", "", "## 实测总时延", "", "| 方法 | 范围 | 有效计时/案例 | Total p50 ms | p95 ms | mean ms |", "| --- | --- | ---: | ---: | ---: | ---: |"])
    for method, result in summary["methods"].items():
        for name in ("all_cases", "feasible_label_click", "negative_label_cases", "wait_cases"):
            value = result["latency"][name]
            total = value["total_ms"]
            lines.append(f"| {method} | {name} | {value['n_measured_eligible']}/{value['n_cases']} | {_fmt(total['p50'])} | {_fmt(total['p95'])} | {_fmt(total['mean'])} |")
    lines.extend(["", "主表包含所有非 warmup 成功、失败和弃权案例。总时延必须来自实际 observe 开始到 verify 结束的计时，不能把旧缓存 inference 成本加到本轮步骤上，也不以阶段求和替代。模型加载排除；缺少实测/加载排除标记的记录不会进入时延表。warmup 已排除，cold_start 记录与不含 cold 的补充摘要保存在 JSON。不同策略 wait 比例不同，整体时延混合分布不能直接证明推理更快。", "", "## 相同案例的时延比较", "", "a−b 为负表示 a 更快；成功交集仅包括双方实际完成 enabled 保存的案例，不能用双方负例 wait 代替。", "", "| 方法 a / 方法 b | 全部匹配实测 N | 双方成功实测 N | 成功交集p50差 ms | p50比值 a/b | 所有配对mean差95%CI ms |", "| --- | ---: | ---: | ---: | ---: | --- |"])
    for comparison in summary["latency_comparisons"]:
        all_cases, successful = comparison["all_paired_cases"], comparison["successful_feasible_intersection"]
        interval = all_cases.get("group_bootstrap", {}).get("mean_difference_ms", {}).get("ci95")
        text = "不可评估" if interval is None else f"[{interval[0]:.2f}, {interval[1]:.2f}]"
        lines.append(f"| {comparison['method_a']} / {comparison['method_b']} | {all_cases['n']} | {successful['n']} | {_fmt(successful.get('p50_difference_ms'))} | {_fmt(successful.get('p50_ratio_a_over_b'))} | {text} |")
    baseline = summary["always_wait_reference"]
    lines.extend(["", "成功交集具有结果选择条件，只反映共同成功子集，不能代替全量完成率或主时延分布。所有匹配 case 的时延差和 p50 差均按完整场景组 bootstrap；样本 ID、组数和区间有效数保存在 JSON。", "", "## 组级区间、基线和解释边界", "", "完成率、动作正确率、负例点击尝试率和点击precision的95%区间使用完整 group_id 重采样，同场景三态不当作独立样本。零错误时经验区间可能退化，不能保证新场景零风险。", "", f"仅作解析参照、未实际执行的 always_wait：可用保存完成 {_rate_text(baseline['feasible_save_success'])}，总体动作正确 {_rate_text(baseline['action_correct'])}。平衡三态得到66.7%总体正确，却完成零个 enabled 任务，因此不可只报告总体正确率。", "", summary["supervision_note"], "", "四方法监督预算不同：hidden gate 与 shared 定位器使用旧 train/val 标注；text_coordinates 为零样本文本坐标。不能把此比较描述成同等监督条件下的通用 agent 排名。本轮是受控浏览器单步点击闭环，不证明真实桌面多步任务能力。", "", "产物：metrics.json 为完整指标、覆盖率、IID/OOD及label切片、组级区间和配对时延；episode_metrics.csv 为逐条结果，包括 warmup 与所有重复记录，保留对原始日志的 raw_index。"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
