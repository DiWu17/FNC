import json

from clickagent.public_diagnostics import run_public_diagnostics
from clickagent.public_evaluation import evaluate_public_rows
from clickagent.public_report import METHODS, generate_public_report


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def fixture(tmp_path, *, feasibility=False):
    run, data = tmp_path / "run", tmp_path / "data"
    cases = [{"sample_id": sid, "dataset": "toy", "instruction": "choose target", "split": "test",
              "group_id": sid, "expected_actionable": sid != "n",
              "task_kind": "action_feasibility" if feasibility else "refusal" if sid == "n" else "grounding",
              "target_geometry": None if feasibility or sid == "n" else {"type": "bbox", "boxes": [[0.2, 0.2, 0.4, 0.4]]}}
             for sid in ("a", "b", "n")]
    rows = []
    for method in METHODS:
        for case in cases:
            sid = case["sample_id"]
            native = method == "text_coordinates"
            click = sid == ("b" if native else "a")
            point = [30, 30] if native or sid == "a" else [70, 70]
            rows.append({"sample_id": sid, "dataset": "toy", "method": method,
                         "action": {"type": "click", "x": point[0], "y": point[1]} if click else {"type": "wait"},
                         "parse_valid": not (native and sid == "a"), "click_probability": 0.9 if sid != "n" else 0.1,
                         "gate_threshold": 0.5, "localization_point": point, "model_image_size": [100, 100], "grid_shape": [10, 10],
                         "cap_reached": False, "generated_text": "CLICK 300,300\n" if native and sid == "a" else "CLICK 300 300\n" if native and sid == "b" else "WAIT\n",
                         "generated_token_ids": [1, 2]})
    write_jsonl(run / "selected_manifest.jsonl", cases)
    write_jsonl(run / "predictions.jsonl", rows)
    write_json(run / "completion.json", {"complete": True, "n_cases": len(cases), "n_rows": len(rows), "error_rows": 0})
    write_json(run / "selection.json", {"fixed_before_model_loading": True,
               "datasets": {"toy": {"selected": 3, "positive": 2, "negative": 1}}})
    evaluate_public_rows(rows, cases, run / "evaluation", expected_methods=METHODS, bootstrap_iterations=0)
    full = cases + [{**cases[0], "sample_id": "full1", "group_id": "full1"}, {**cases[0], "sample_id": "full2", "group_id": "full2"}]
    write_jsonl(data / "manifests" / "toy.jsonl", full)
    diagnostics = tmp_path / "diagnostics"
    run_public_diagnostics([run], diagnostics)
    resolution = tmp_path / "resolution.json"
    write_json(resolution, {"status": "complete", "full_supplied_manifest": True,
                "by_dataset": {"toy": {"input_rows": 5, "positive_geometry_denominator": 0 if feasibility else 4,
                                        "quantization_oracle": {"numerator": 0 if feasibility else 2, "denominator": 0 if feasibility else 4, "rate": None if feasibility else 0.5},
                                        "quantization_oracle_coverage": None if feasibility else 1,
                                        "unresolved_positive_geometry_rows": 0}}})
    return run, data, diagnostics / "diagnostics.json", resolution, cases, rows


def make_report(tmp_path, parts):
    run, data, diagnostics, resolution, *_ = parts
    return generate_public_report([run], diagnostics, resolution, data, tmp_path / "final")


def test_evidence_tables_keep_strict_lax_selected_and_full_scopes_separate(tmp_path):
    parts = fixture(tmp_path)
    result = make_report(tmp_path, parts)
    assert result["status"] == "complete"
    d = result["datasets"]["toy"]
    assert d["sample_counts"]["full_manifest"]["all_adapted_rows"] == 5
    assert d["sample_counts"]["selected_n"] == 3
    loc = d["localization"]
    assert loc["shared_point_head_ignoring_gate"]["rate"] == 0.5
    assert loc["native_strict_point_hit"]["rate"] == 0.5
    assert loc["offline_lax_format_point_hit"]["rate"] == 1
    assert loc["selected_grid_oracle_observed"]["rate"] == 1
    assert d["full_manifest_grid_oracle"]["quantization_oracle"]["rate"] == 0.5
    assert d["decision_metrics"]["prefill_hidden"]["positive_correct_expected"]["rate"] == 0.5
    assert d["errors_and_format"]["text_coordinates"]["strict_parse_failures"]["numerator"] == 1
    assert (tmp_path / "final" / "report.md").is_file()


def test_completion_missing_is_partial_without_removing_failed_cases(tmp_path):
    parts = fixture(tmp_path)
    (parts[0] / "completion.json").unlink()
    result = make_report(tmp_path, parts)
    assert result["status"] == "partial"
    assert any("completion missing" in issue for issue in result["issues"])
    assert result["datasets"]["toy"]["decision_metrics"]["prefill_hidden"]["positive_correct_expected"]["denominator"] == 2


def test_duplicate_error_rows_are_visible_and_not_best_of_two(tmp_path):
    parts = fixture(tmp_path)
    run, data, diagnostics, resolution, cases, rows = parts
    duplicate = {**rows[0], "error": "retained failure", "parse_valid": False, "action": {"type": "wait"}}
    rows = [*rows, duplicate]
    write_jsonl(run / "predictions.jsonl", rows)
    write_json(run / "completion.json", {"complete": True, "n_cases": 3, "n_rows": len(rows), "error_rows": 1})
    result = make_report(tmp_path, parts)
    d = result["datasets"]["toy"]
    assert result["status"] == "partial"
    assert d["coverage"]["prefill_hidden"]["duplicate_counts"] == {"a": 2}
    assert d["decision_metrics"]["prefill_hidden"]["positive_correct_expected"]["numerator"] == 0
    assert d["errors_and_format"]["prefill_hidden"]["recorded_error_rows"] == 1


def test_selected_oracle_cannot_be_presented_as_full_manifest_oracle(tmp_path):
    parts = fixture(tmp_path)
    audit = json.loads(parts[3].read_text())
    audit["by_dataset"]["toy"]["input_rows"] = 3
    write_json(parts[3], audit)
    result = make_report(tmp_path, parts)
    assert result["datasets"]["toy"]["full_manifest_grid_oracle"]["status"] == "unavailable"
    assert result["datasets"]["toy"]["localization"]["selected_grid_oracle_observed"]["rate"] == 1
    assert result["status"] == "partial"


def test_stale_diagnostics_do_not_supply_a_lax_score(tmp_path):
    parts = fixture(tmp_path)
    diagnostic = json.loads(parts[2].read_text())
    diagnostic["datasets"]["toy"]["n_expected_cases"] = 99
    write_json(parts[2], diagnostic)
    result = make_report(tmp_path, parts)
    assert result["datasets"]["toy"]["localization"]["offline_lax_format_point_hit"] is None
    assert result["status"] == "partial"


def test_conflict_style_feasibility_never_invents_point_success(tmp_path):
    parts = fixture(tmp_path, feasibility=True)
    result = make_report(tmp_path, parts)
    assert result["status"] == "complete"
    d = result["datasets"]["toy"]
    assert d["localization"]["shared_point_head_ignoring_gate"]["denominator"] == 0
    assert d["localization"]["native_strict_point_hit"]["rate"] is None
    assert d["decision_metrics"]["prefill_hidden"]["positive_metric_kind"] == "binary_positive_allow"
    assert d["decision_metrics"]["prefill_hidden"]["positive_correct_expected"]["rate"] == 0.5
    for method in METHODS:
        assert d["misclick_metrics"][method]["point_metrics"] is None
        assert d["misclick_metrics"][method]["formal_precision_crosscheck"]["status"] == "not_applicable_binary_feasibility_only"


def test_wrong_target_and_negative_clicks_are_separate_and_both_reduce_precision(tmp_path):
    parts = fixture(tmp_path)
    run, _, _, _, cases, rows = parts
    for row in rows:
        if row["method"] == "prefill_hidden" and row["sample_id"] in ("b", "n"):
            row["action"] = {"type": "click", "x": 70, "y": 70}
    write_jsonl(run / "predictions.jsonl", rows)
    evaluate_public_rows(rows, cases, run / "evaluation", expected_methods=METHODS, bootstrap_iterations=0)
    d = make_report(tmp_path, parts)["datasets"]["toy"]["misclick_metrics"]["prefill_hidden"]
    point = d["point_metrics"]
    assert point["positive_wrong_target_clicks"] == 1
    assert point["negative_false_clicks"] == 1
    assert point["total_emitted_clicks"] == 3 and point["correct_clicks"] == 1
    assert point["correct_click_precision"]["rate"] == 1 / 3
    assert d["formal_precision_crosscheck"]["status"] == "match"
    assert d["by_point_task_kind"]["grounding"]["positive_wrong_target_clicks"] == 1
    assert d["by_point_task_kind"]["refusal"]["negative_false_clicks"] == 1


def test_all_wait_has_null_precision_instead_of_perfect_precision(tmp_path):
    parts = fixture(tmp_path)
    run, _, _, _, cases, rows = parts
    for row in rows:
        if row["method"] == "post_hidden":
            row["action"] = {"type": "wait"}
    write_jsonl(run / "predictions.jsonl", rows)
    evaluate_public_rows(rows, cases, run / "evaluation", expected_methods=METHODS, bootstrap_iterations=0)
    point = make_report(tmp_path, parts)["datasets"]["toy"]["misclick_metrics"]["post_hidden"]["point_metrics"]
    assert point["total_emitted_clicks"] == 0
    assert point["correct_click_precision"] == {"numerator": 0, "denominator": 0, "rate": None}


def test_parse_invalid_emitted_click_is_counted_in_precision_denominator(tmp_path):
    parts = fixture(tmp_path)
    run, _, _, _, cases, rows = parts
    for row in rows:
        if row["method"] == "prefill_hidden" and row["sample_id"] == "b":
            row["action"] = {"type": "click", "x": 70, "y": 70}
            row["parse_valid"] = False
    write_jsonl(run / "predictions.jsonl", rows)
    evaluate_public_rows(rows, cases, run / "evaluation", expected_methods=METHODS, bootstrap_iterations=0)
    point = make_report(tmp_path, parts)["datasets"]["toy"]["misclick_metrics"]["prefill_hidden"]["point_metrics"]
    assert point["positive_wrong_target_clicks"] == 0
    assert point["invalid_parse_emitted_clicks"] == 1
    assert point["correct_click_precision"]["rate"] == 0.5
