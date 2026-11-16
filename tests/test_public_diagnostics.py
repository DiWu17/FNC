import hashlib
import json
from pathlib import Path

import pytest

from clickagent.public_diagnostics import parse_offline_lax_click, run_public_diagnostics


def case(sid, positive=True):
    return {"sample_id": sid, "dataset": "toy", "instruction": "choose target",
            "expected_actionable": positive, "task_kind": "grounding" if positive else "refusal",
            "target_geometry": {"type": "bbox", "boxes": [[0.2, 0.2, 0.4, 0.4]]} if positive else None}


def gate(sid, probability, *, click=False, point=(30, 30)):
    return {"sample_id": sid, "dataset": "toy", "method": "prefill_hidden",
            "click_probability": probability, "gate_threshold": 0.5, "parse_valid": True,
            "action": {"type": "click", "x": point[0], "y": point[1]} if click else {"type": "wait"},
            "localization_point": list(point), "model_image_size": [100, 100]}


def native_row(sid, text, *, cap=False, tokens=None):
    return {"sample_id": sid, "dataset": "toy", "method": "text_coordinates",
            "parse_valid": False, "action": {"type": "wait"}, "model_image_size": [100, 100],
            "generated_text": text, "cap_reached": cap, "generated_token_ids": tokens or [1, 2]}


def write_run(path, cases, rows, *, identity=None):
    path.mkdir(parents=True)
    (path / "selected_manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in cases), encoding="utf-8")
    (path / "predictions.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    if identity is not None:
        (path / "identity.json").write_text(json.dumps(identity), encoding="utf-8")
    return path


def test_auc_ranking_and_threshold_transfer_are_separate(tmp_path):
    cases = [case("p1"), case("p2"), case("n1", False), case("n2", False)]
    rows = [gate("p1", 0.1), gate("p2", 0.2), gate("n1", 0.8, click=True), gate("n2", 0.9, click=True)]
    run = write_run(tmp_path / "run", cases, rows)
    result = run_public_diagnostics([run], tmp_path / "analysis")
    d = result["datasets"]["toy"]["gates"]["prefill_hidden"]
    assert d["click_probability_auroc"]["value"] == 0
    assert d["positive_allow_rate_observed"]["rate"] == 0
    assert d["negative_false_click_observed"]["rate"] == 1
    assert d["cross_counts"]["localization_hit_but_gate_rejected"] == 2
    assert d["probability_positive"]["mean"] == pytest.approx(0.15)
    assert (tmp_path / "analysis" / "diagnostics.json").is_file()


def test_ties_have_auc_half_but_single_class_has_no_auc(tmp_path):
    run = write_run(tmp_path / "mixed", [case("p"), case("n", False)], [gate("p", 0.5), gate("n", 0.5)])
    result = run_public_diagnostics([run], tmp_path / "analysis")
    assert result["datasets"]["toy"]["gates"]["prefill_hidden"]["click_probability_auroc"]["value"] == 0.5
    run2 = write_run(tmp_path / "positive", [case("p")], [gate("p", 1)])
    result2 = run_public_diagnostics([run2], tmp_path / "analysis2")
    auc = result2["datasets"]["toy"]["gates"]["prefill_hidden"]["click_probability_auroc"]
    assert auc["value"] is None and auc["status"] == "not_applicable_single_expected_class"


def test_missing_probability_and_duplicate_rows_keep_coverage(tmp_path):
    cases = [case("p"), case("n", False)]
    run = write_run(tmp_path / "run", cases, [gate("p", None), gate("n", 0.1), gate("n", 0.9, click=True)])
    d = run_public_diagnostics([run], tmp_path / "analysis")["datasets"]["toy"]["gates"]["prefill_hidden"]
    assert d["n_unique_rows"] == 1
    assert d["click_probability_coverage"]["rate"] == 0
    assert d["invalid_or_missing_probability_sample_ids"] == ["p"]
    assert d["probability_missing_due_to_row_coverage_sample_ids"] == ["n"]
    assert d["negative_false_click_upper_bound_expected"]["rate"] == 1
    assert d["click_probability_auroc"]["status"] == "unavailable_missing_probability_class"


def test_lax_comma_is_offline_only_and_keeps_negative_recoveries(tmp_path):
    cases = [case("comma"), case("cap"), case("incomplete"), case("negative", False)]
    rows = [native_row("comma", "CLICK 300, 300\n"), native_row("cap", "CLICK 300, 300\n", cap=True),
            native_row("incomplete", "CLICK 300, 300"), native_row("negative", "CLICK 300, 300\n")]
    run = write_run(tmp_path / "run", cases, rows)
    before = (run / "predictions.jsonl").read_bytes()
    result = run_public_diagnostics([run], tmp_path / "analysis")
    d = result["datasets"]["toy"]["offline_lax_coordinates"]
    assert d["counts"]["newly_recoverable_strict_invalid_clicks"] == 2
    assert d["counts"]["newly_recoverable_grounded_hits"] == 1
    assert d["counts"]["newly_recoverable_negative_clicks"] == 1
    assert d["offline_lax_grounded_hit_on_expected_positives"]["rate"] == pytest.approx(1 / 3)
    assert d["exclusion_reason_counts"]["token_cap_reached"] == 1
    assert d["exclusion_reason_counts"]["no_newline_or_matched_eos_completion"] == 1
    assert (run / "predictions.jsonl").read_bytes() == before


def test_matching_native_eos_is_required_for_delimiter_free_recovery(tmp_path):
    identity = {"protocol": "fixed"}
    run = write_run(tmp_path / "run", [case("match"), case("mismatch")],
                    [native_row("match", "CLICK 300, 300"), native_row("mismatch", "CLICK 300, 300")], identity=identity)
    native = run / "native"
    native.mkdir()
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    for sid, tokens in (("match", [1, 2]), ("mismatch", [1, 3])):
        path = native / (hashlib.sha256(sid.encode()).hexdigest()[:24] + ".json")
        path.write_text(json.dumps({"generated_text": "CLICK 300, 300", "generated_token_ids": tokens,
                                    "terminated_by_eos": True, "truncated": False, "fingerprint": fingerprint}), encoding="utf-8")
    d = run_public_diagnostics([run], tmp_path / "analysis")["datasets"]["toy"]["offline_lax_coordinates"]
    assert d["counts"]["offline_lax_clicks"] == 1
    records = {row["sample_id"]: row for row in d["records"]}
    assert records["match"]["completion_evidence"]["eos"] is True
    assert records["mismatch"]["completion_evidence"]["matched_native"]["status"] == "invalid"
    assert not records["mismatch"]["eligible_offline_lax_click"]


@pytest.mark.parametrize("text", ["CLICK 300,300 extra\n", "CLICK (300, 300)\n", "CLICK -3 4\n", "CLICK 3.5 4\n", "CLICK 1001,0\n"])
def test_relaxed_separator_never_repairs_content(text):
    assert parse_offline_lax_click(text) is None


def test_overlapping_runs_are_ambiguous_not_best_of_two(tmp_path):
    run1 = write_run(tmp_path / "run1", [case("p")], [gate("p", 0.1)])
    run2 = write_run(tmp_path / "run2", [case("p")], [gate("p", 0.9, click=True)])
    d = run_public_diagnostics([run1, run2], tmp_path / "analysis")["datasets"]["toy"]["gates"]["prefill_hidden"]
    assert d["n_expected"] == 1 and d["n_unique_rows"] == 0
    assert len(d["duplicate_records"]["p"]) == 2


def test_analysis_cannot_overwrite_formal_evaluation_and_marks_missing_runs(tmp_path):
    run = write_run(tmp_path / "run", [case("p")], [gate("p", 0.1)])
    with pytest.raises(ValueError, match="separate"):
        run_public_diagnostics([run], run / "evaluation")
    result = run_public_diagnostics([run, tmp_path / "not_yet_complete"], tmp_path / "analysis")
    assert result["status"] == "partial_missing_runs"
    assert len(result["skipped_runs"]) == 1
