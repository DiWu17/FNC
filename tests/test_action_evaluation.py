"""Action/evidence fixtures validate scoring; these are not browser benchmarks."""

from copy import deepcopy
import csv
import json

import pytest

from clickagent.action_evaluation import ACTION_METHODS, evaluate_action_jsonl, evaluate_action_rows


def _episode(group, label, method="prefill_hidden", total=100.0, domain="iid"):
    emitted = label == "click"
    return {
        "episode_id": f"{group}_{label}_{method}", "sample_id": f"{group}_{label}",
        "group_id": group, "label": label, "split": "test", "domain": domain, "method": method,
        "action": {"type": "click", "x": 150.0, "y": 120.0} if emitted else {"type": "wait"},
        "env_before": {"save_count": 0, "other_action_count": 0, "other_attempt_count": 0},
        "env_after": {"save_count": int(emitted), "other_action_count": 0, "other_attempt_count": 0},
        "verification": {"pixel_changed": emitted, "changed_pixel_count": 30 if emitted else 0},
        "target_hit": emitted, "mapped_target": {"text": "保存", "enabled": True} if emitted else None,
        "parse_valid": True, "n_generated_tokens": 0, "warmup": False, "cold_start": False,
        "fallback": False, "latency_measured": True, "model_load_included": False,
        "timing_ms": {"observe": 5, "inference": 60, "policy": 5, "execute": 5, "verify": 5, "total": total},
    }


def _triplets(methods=("prefill_hidden",), totals=None):
    totals = totals or {method: 100.0 for method in methods}
    return [_episode(group, label, method, totals[method], "ood" if group == "g1" else "iid")
            for method in methods for group in ("g0", "g1") for label in ("click", "disabled", "absent")]


def _case_rows(rows):
    return [dict(row) for row in rows if row["method"] == rows[0]["method"]]


def test_actual_save_effects_invalid_parse_and_ignored_negative_clicks(tmp_path):
    rows = _triplets()
    # Disabled target ignores a click: no side effect, but still false allowance.
    rows[1].update(action={"type": "click", "x": 150, "y": 120}, target_hit=True)
    # A malformed output abstains safely but is not a valid action decision.
    rows[2]["parse_valid"] = False
    # Enabled case clicks another button and causes a real side effect.
    rows[3].update(target_hit=False, mapped_target={"text": "删除", "enabled": True})
    rows[3]["env_after"].update(save_count=0, other_action_count=1, other_attempt_count=1)
    # Clicking an unrelated disabled control has an attempt, no activation.
    rows[5].update(action={"type": "click", "x": 2, "y": 3}, target_hit=False)
    rows[5]["env_after"]["other_attempt_count"] = 1
    summary = evaluate_action_rows(rows, tmp_path, expected_methods=("prefill_hidden",), bootstrap_iterations=128)
    result = summary["methods"]["prefill_hidden"]
    metrics = result["metrics"]
    assert metrics["feasible_save_success"] == {"numerator": 1, "denominator": 2, "rate": .5}
    assert metrics["action_correct"]["numerator"] == 2
    assert metrics["behavioral_correct"]["numerator"] == 3
    assert metrics["negative_false_click_attempt"]["rate"] == .5
    assert metrics["negative_false_click_attempt_disabled"]["numerator"] == 1
    assert metrics["negative_false_click_attempt_absent"]["numerator"] == 1
    assert metrics["wrong_target_clicks_on_feasible"] == 1
    assert metrics["actual_wrong_side_effect_event_count"] == 1
    assert metrics["actual_wrong_side_effect_row_count"] == 1
    assert metrics["other_attempt_count"] == 2
    assert metrics["emitted_click_precision"]["rate"] == .25
    assert metrics["invalid_parse"]["numerator"] == 1
    assert result["by_domain"]["ood"]["metrics"]["feasible_save_success"]["rate"] == 0
    assert result["by_label"]["disabled"]["n"] == 2
    assert result["group_bootstrap"]["unit"] == "group_id"
    assert summary["always_wait_reference"]["reference_only_not_executed"] is True
    assert summary["always_wait_reference"]["action_correct"]["rate"] == pytest.approx(2 / 3)
    assert (tmp_path / "report.md").is_file()
    assert "旧截图分类 accuracy 不算点击成功" in (tmp_path / "report.md").read_text(encoding="utf-8")


@pytest.mark.parametrize("evidence", ["pixel_false", "pixel_missing", "other_side_effect", "execution_error"])
def test_activation_alone_is_not_verified_goal_success(tmp_path, evidence):
    row = _episode("g0", "click")
    if evidence == "pixel_false":
        row["verification"] = {"pixel_changed": False}
    elif evidence == "pixel_missing":
        row.pop("verification")
    elif evidence == "other_side_effect":
        row["env_after"]["other_action_count"] = 1
    else:
        row["execution_error"] = "execution verification failed"
    result = evaluate_action_rows([row], tmp_path, expected_methods=("prefill_hidden",), bootstrap_iterations=0)["methods"]["prefill_hidden"]
    assert result["metrics"]["feasible_save_success"]["rate"] == 0
    assert result["metrics"]["action_correct"]["rate"] == 0


def test_unknown_environment_and_target_evidence_are_not_fabricated(tmp_path):
    positive = _episode("g0", "click")
    positive.pop("target_hit")
    positive.pop("mapped_target")
    positive["env_after"]["save_count"] = 0
    negative = _episode("g0", "disabled")
    negative.pop("env_after")
    result = evaluate_action_rows([positive, negative], tmp_path, expected_methods=("prefill_hidden",), bootstrap_iterations=0)["methods"]["prefill_hidden"]["metrics"]
    assert result["unknown_enabled_click_target_count"] == 1
    assert result["wrong_target_clicks_on_feasible"] == 0
    assert result["unknown_environment_evidence_count"] == 1
    assert result["behavioral_correct"]["rate"] == 0


def test_mapped_target_dictionary_and_flat_pixel_evidence_are_supported(tmp_path):
    row = _episode("g0", "click")
    row.pop("target_hit")
    row.pop("verification")
    row.update(pixel_changed=True, changed_pixel_count=30)
    result = evaluate_action_rows([row], tmp_path, expected_methods=("prefill_hidden",), bootstrap_iterations=0)["methods"]["prefill_hidden"]["metrics"]
    assert result["feasible_save_success"]["rate"] == 1
    assert result["unknown_enabled_click_target_count"] == 0


def test_always_wait_and_malformed_abstention_cannot_hide_zero_completion(tmp_path):
    rows = _triplets()
    for row in rows:
        row.update(action={"type": "wait"}, parse_valid=False)
        row["env_after"]["save_count"] = 0
    summary = evaluate_action_rows(rows, tmp_path, expected_methods=("prefill_hidden",), bootstrap_iterations=128)
    m = summary["methods"]["prefill_hidden"]["metrics"]
    assert m["feasible_save_success"]["rate"] == 0
    assert m["action_correct"]["rate"] == 0
    assert m["behavioral_correct"]["rate"] == pytest.approx(2 / 3)
    assert m["emitted_click_precision"]["rate"] is None
    assert m["abstention"]["rate"] == 1
    assert m["feasible_abstention"]["rate"] == 1
    assert "NaN" not in (tmp_path / "metrics.json").read_text(encoding="utf-8")


def test_warmup_and_cold_start_audit_and_no_reconstructed_latency(tmp_path):
    rows = _triplets()
    for row in rows:
        row["timing_ms"]["total"] = 501.0  # Deliberately not the component sum.
    warmup = deepcopy(rows[0])
    warmup.update(warmup=True, cold_start=True)
    warmup["timing_ms"]["total"] = 10000
    rows.append(warmup)
    # Missing total cannot be filled from the five stage timings.
    rows[1]["timing_ms"].pop("total")
    rows[2]["latency_source"] = "cache_plus_new_policy"
    rows[3]["model_load_included"] = True
    rows[4]["latency_source"] = "fresh_inference_wall_clock"
    result = evaluate_action_rows(rows, tmp_path, expected_methods=("prefill_hidden",), bootstrap_iterations=0)
    assert result["warmup"]["n"] == 1
    assert result["coverage"]["duplicates"] == []
    assert result["cold_start_records"][0]["total_ms"] == 10000
    latency = result["methods"]["prefill_hidden"]["latency"]["all_cases"]
    assert latency["n_cases"] == 6
    assert latency["n_measured_eligible"] == 3
    assert latency["total_ms"]["p50"] == 501
    assert latency["total_ms"]["mean"] == 501
    assert len(result["measurement_audit"]["primary_latency_ineligible"]) == 3


def test_paired_success_latency_uses_actual_enabled_completion_not_negative_waits(tmp_path):
    totals = dict(zip(ACTION_METHODS, (100, 130, 200, 300)))
    rows = _triplets(ACTION_METHODS, totals)
    # prefill fails one enabled save; those latencies remain in its primary table.
    row = next(row for row in rows if row["method"] == "prefill_hidden" and row["sample_id"] == "g1_click")
    row["action"] = {"type": "wait"}
    row["env_after"]["save_count"] = 0
    summary = evaluate_action_rows(rows, tmp_path, bootstrap_iterations=128)
    pair = next(item for item in summary["latency_comparisons"] if item["method_a"] == "prefill_hidden" and item["method_b"] == "text_coordinates")
    assert pair["all_paired_cases"]["n"] == 6
    successful = pair["successful_feasible_intersection"]
    assert successful["n"] == 1
    assert successful["sample_ids"] == ["g0_click"]
    assert successful["p50_difference_ms"] == -200
    assert successful["p50_ratio_a_over_b"] == pytest.approx(1 / 3)
    assert successful["group_bootstrap"]["status"] == "unavailable"
    shared_pair = next(item for item in summary["latency_comparisons"] if item["method_a"] == "post_hidden" and item["method_b"] == "text_decision_shared")
    assert shared_pair["successful_feasible_intersection"]["n"] == 2
    assert shared_pair["priority_comparison"] is True


def test_paired_latency_bootstrap_resamples_whole_scenarios(tmp_path):
    methods = ("prefill_hidden", "text_coordinates")
    rows = _triplets(methods)
    for row in rows:
        row["timing_ms"]["total"] = 100 if row["method"] == "text_coordinates" else (90 if row["group_id"] == "g0" else 110)
    result = evaluate_action_rows(rows, tmp_path, expected_methods=methods, bootstrap_iterations=512)
    comparison = result["latency_comparisons"][0]["all_paired_cases"]
    assert comparison["n_groups"] == 2
    assert comparison["paired_difference_mean_ms"] == 0
    assert comparison["group_bootstrap"]["mean_difference_ms"]["ci95"] == [-10, 10]
    assert comparison["group_bootstrap"]["p50_difference_ms"]["ci95"] == [-10, 10]


def test_missing_and_duplicate_methods_are_explicit_and_never_cherry_picked(tmp_path):
    methods = ("prefill_hidden", "text_coordinates")
    rows = _triplets(methods)
    expected = _case_rows(rows)
    rows = [row for row in rows if not (row["method"] == "text_coordinates" and row["sample_id"] == "g1_absent")]
    duplicate = deepcopy(rows[0])
    duplicate["env_after"]["save_count"] = 0
    rows.append(duplicate)
    summary = evaluate_action_rows(rows, tmp_path, expected_methods=methods, expected_cases=expected, bootstrap_iterations=0)
    coverage = summary["coverage"]
    assert coverage["n_expected_cases"] == 6
    assert coverage["methods"]["prefill_hidden"]["ambiguous_duplicate_sample_ids"] == ["g0_click"]
    assert len(coverage["duplicates"][0]["raw_indices"]) == 2
    assert coverage["methods"]["text_coordinates"]["missing_sample_ids"] == ["g1_absent"]
    pre = summary["methods"]["prefill_hidden"]
    assert pre["n"] == 5
    assert pre["status"] == "partial_coverage"
    assert pre["coverage_adjusted"]["feasible_success_over_expected"]["rate"] == .5
    assert summary["latency_comparisons"][0]["all_paired_cases"]["n"] == 4
    logged = list(csv.DictReader((tmp_path / "episode_metrics.csv").open(encoding="utf-8-sig")))
    assert len(logged) == len(rows)


def test_expected_cases_detect_samples_missing_from_every_method(tmp_path):
    methods = ("prefill_hidden", "text_coordinates")
    rows = _triplets(methods)
    expected = _case_rows(rows)
    rows = [row for row in rows if row["sample_id"] != "g0_click"]
    explicit = evaluate_action_rows(rows, tmp_path / "explicit", expected_methods=methods, expected_cases=expected, bootstrap_iterations=0)
    inferred = evaluate_action_rows(rows, tmp_path / "union", expected_methods=methods, bootstrap_iterations=0)
    assert explicit["coverage"]["n_expected_cases"] == 6
    assert inferred["coverage"]["n_expected_cases"] == 5
    assert "cannot be detected" in inferred["coverage"]["universe_source"]


def test_out_of_bounds_error_is_counted_without_coordinate_repair(tmp_path):
    row = _episode("g0", "click")
    row.update(action={"type": "click", "x": -100.0, "y": 9000.0}, target_hit=False, mapped_target=None, execution_error="out of viewport")
    row["env_after"]["save_count"] = 0
    summary = evaluate_action_rows([row], tmp_path, expected_methods=("prefill_hidden",), bootstrap_iterations=0)
    m = summary["methods"]["prefill_hidden"]["metrics"]
    assert m["feasible_save_success"]["rate"] == 0
    assert m["execution_error_count"] == 1
    assert m["wrong_target_clicks_on_feasible"] == 1
    assert row["action"]["x"] == -100


@pytest.mark.parametrize("problem", ["non_test", "fallback", "case_mismatch", "counter_reset"])
def test_protocol_errors_are_not_silently_fixed(tmp_path, problem):
    rows = _triplets(("prefill_hidden", "text_coordinates"))
    if problem == "non_test":
        rows[0]["split"] = "val"
        match = "test episodes only"
    elif problem == "fallback":
        rows[0]["fallback"] = True
        match = "fallback=true"
    elif problem == "case_mismatch":
        rows[6]["group_id"] = "wrong_group"
        match = "Case facts differ"
    else:
        rows[0]["env_before"]["save_count"] = 3
        match = "decreased"
    with pytest.raises(ValueError, match=match):
        evaluate_action_rows(rows, tmp_path, bootstrap_iterations=0)


def test_jsonl_wrapper_and_missing_expected_methods_are_reported(tmp_path):
    rows = _triplets()
    path = tmp_path / "actions.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    result = evaluate_action_jsonl(path, tmp_path / "evaluation", bootstrap_iterations=0)
    assert result["coverage"]["methods"]["post_hidden"]["coverage"] == 0
    assert result["methods"]["post_hidden"]["metrics"]["feasible_save_success"]["rate"] is None
    assert result["methods"]["post_hidden"]["coverage_adjusted"]["feasible_success_over_expected"]["rate"] == 0
