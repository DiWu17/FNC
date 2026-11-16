"""Public-data scoring tests use hand-checkable geometry and decisions only."""

import csv
import json
from copy import deepcopy

import pytest

from clickagent.public_evaluation import (
    evaluate_public_jsonl,
    evaluate_public_rows,
    point_in_geometry,
    quantization_oracle,
)


BOX = {"type": "bbox", "boxes": [[0.2, 0.2, 0.4, 0.4]]}


def case(sid, positive=True, *, task="grounding", group=None, geometry=None, domain="web"):
    return {"sample_id": sid, "dataset": "fixture", "instruction": "select target",
            "expected_actionable": positive, "task_kind": task,
            "target_geometry": (deepcopy(BOX) if geometry is None else geometry) if positive and task == "grounding" else None,
            "group_id": group or sid, "metadata": {"domain": domain}}


def row(sid, action="click", *, method="prefill_hidden", parse_valid=True, point=(30, 30), localization=(30, 30), **extra):
    return {"sample_id": sid, "method": method, "action": {"type": "wait"} if action == "wait" else {"type": "click", "x": point[0], "y": point[1]},
            "parse_valid": parse_valid, "localization_point": localization,
            "model_image_size": [100, 100], "grid_shape": [10, 10], **extra}


def evaluate(rows, manifest, tmp_path, **kwargs):
    return evaluate_public_rows(rows, manifest, tmp_path, bootstrap_iterations=0,
                                expected_methods=("prefill_hidden",), **kwargs)


def metrics(result, method="prefill_hidden"):
    return result["methods"][method]["metrics"]


def test_bbox_union_closed_edges_and_out_of_image():
    geometry = {"type": "bbox", "boxes": [[0.1, 0.1, 0.2, 0.2], [0.7, 0.7, 0.9, 0.9]]}
    assert point_in_geometry((0.1, 0.2), geometry)
    assert point_in_geometry((0.9, 0.9), geometry)
    assert not point_in_geometry((0.5, 0.5), geometry)
    assert not point_in_geometry((1.1, 0.9), geometry)


def test_polygon_concavity_boundary_and_multiple_components():
    geometry = {"type": "polygon", "polygons": [
        [[0, 0], [0.6, 0], [0.6, 0.2], [0.2, 0.2], [0.2, 0.6], [0, 0.6]],
        [[0.8, 0.8], [1, 0.8], [0.9, 1]],
    ]}
    assert point_in_geometry((0.1, 0.5), geometry)
    assert point_in_geometry((0.2, 0.4), geometry)  # on inner edge
    assert not point_in_geometry((0.4, 0.4), geometry)  # concave cutout
    assert point_in_geometry((0.9, 0.85), geometry)


def test_oracle_uses_rounded_pixel_centers_not_continuous_grid():
    middle = {"type": "bbox", "boxes": [[0.49, 0.49, 0.51, 0.51]]}
    assert quantization_oracle(middle, [1, 1], [4, 4])
    # np.rint(2.5) is 2, so the actual integer pixel center is normalized .4.
    assert not quantization_oracle(middle, [1, 1], [5, 5])
    clipped = {"type": "bbox", "boxes": [[0, 0, 0.1, 0.1]]}
    assert quantization_oracle(clipped, [1, 1], [1, 1])


def test_unconditional_localization_does_not_hide_false_rejection(tmp_path):
    result = evaluate([row("a", "wait"), row("b", point=(70, 70))], [case("a"), case("b")], tmp_path)
    m = metrics(result)
    assert m["localization_hit_ignoring_gate"]["rate"] == 1
    assert m["action_aware_correct"]["rate"] == 0
    assert m["positive_false_rejection"]["rate"] == 0.5
    assert m["emitted_click_precision"]["rate"] == 0
    assert (tmp_path / "metrics.json").is_file()
    assert (tmp_path / "report.md").is_file()


def test_missing_unconditional_point_counts_in_localization_denominator(tmp_path):
    result = evaluate([row("a", localization=None)], [case("a")], tmp_path)
    m = metrics(result)
    assert m["localization_point_coverage"] == {"numerator": 0, "denominator": 1, "rate": 0}
    assert m["localization_hit_ignoring_gate"]["rate"] == 0
    assert m["action_aware_correct"]["rate"] == 1


def test_bad_format_safe_wait_is_not_strict_refusal(tmp_path):
    manifest = [case("n", False, task="refusal"), case("bad", False, task="refusal"), case("click", False, task="refusal")]
    rows = [row("n", "wait"), row("bad", "wait", parse_valid=False, cap_reached=True), row("click")]
    result = evaluate(rows, manifest, tmp_path)
    m = metrics(result)
    assert m["strict_refusal_correct"]["numerator"] == 1
    assert m["behavioral_point_or_refusal_correct"]["numerator"] == 2
    assert m["negative_false_click"]["numerator"] == 1
    assert m["negative_parse_failure"]["numerator"] == 1
    assert result["methods"]["prefill_hidden"]["counts"]["cap_errors"] == 1


def test_conflict_nonclick_is_only_binary_feasibility(tmp_path):
    manifest = [case("type", task="action_feasibility"), case("forbidden", False, task="action_feasibility")]
    rows = [row("type"), row("forbidden", "wait")]
    # A CLICK feasibility label needs no coordinate evidence for original TYPE.
    rows[0]["action"] = {"type": "click"}
    rows[0].pop("model_image_size")
    result = evaluate(rows, manifest, tmp_path)
    m = metrics(result)
    assert m["feasibility_correct"]["rate"] == 1
    assert m["action_aware_correct"]["denominator"] == 0
    assert m["localization_hit_ignoring_gate"]["denominator"] == 0
    assert m["emitted_click_precision"]["denominator"] == 0


def test_duplicate_does_not_select_success_and_missing_not_safe(tmp_path):
    manifest = [case("a"), case("n", False, task="refusal"), case("gone", False, task="refusal", domain="mobile")]
    rows = [row("a"), row("a", "wait"), row("n", "wait")]
    result = evaluate(rows, manifest, tmp_path)
    coverage = result["coverage"]["prefill_hidden"]
    assert coverage["n_unique_scored"] == 1
    assert coverage["missing_sample_ids"] == ["gone"]
    assert coverage["duplicate_raw_indices"] == {"a": [0, 1]}
    adjusted = result["methods"]["prefill_hidden"]["coverage_adjusted"]
    assert adjusted["action_aware_correct"]["rate"] == pytest.approx(1 / 3)
    assert adjusted["negative_false_click_upper_bound"]["rate"] == 0.5
    assert result["methods"]["prefill_hidden"]["by_domain"]["mobile"]["unique_coverage"] == 0
    with (tmp_path / "predictions.csv").open(encoding="utf-8-sig") as stream:
        exported = list(csv.DictReader(stream))
    assert len(exported) == 3 and exported[0]["duplicate_ambiguous"] == "True"


def test_all_missing_requested_method_has_explicit_empty_metrics(tmp_path):
    result = evaluate_public_rows([row("a")], [case("a")], tmp_path,
        expected_methods=("prefill_hidden", "custom_attention"), bootstrap_iterations=0)
    assert result["coverage"]["custom_attention"]["missing_sample_ids"] == ["a"]
    assert metrics(result, "custom_attention")["parse_failure"]["rate"] is None


def test_manifest_gt_is_not_overridden_by_output_and_error_not_dropped(tmp_path):
    rows = [row("a", expected_actionable=False), row("b", error="out of memory"), row("outside")]
    result = evaluate(rows, [case("a"), case("b")], tmp_path)
    assert metrics(result)["action_aware_correct"]["numerator"] == 0
    counts = result["methods"]["prefill_hidden"]["counts"]
    assert counts["protocol_error_rows"] == 1 and counts["recorded_errors"] == 1
    assert len(result["unexpected_rows"]) == 1
    assert result["coverage"]["prefill_hidden"]["n_unique_scored"] == 2


def test_invalid_click_is_not_clamped_into_target(tmp_path):
    result = evaluate([row("a", point=(100, 30))], [case("a")], tmp_path)
    assert metrics(result)["action_aware_correct"]["rate"] == 0
    assert metrics(result)["localization_hit_ignoring_gate"]["rate"] == 1
    assert result["methods"]["prefill_hidden"]["counts"]["protocol_error_rows"] == 1


def test_shared_trajectory_is_counted_once_not_per_method_latency(tmp_path):
    rows = [row("a", method=m, trajectory_id="a:shared", trajectory_timing_ms={"total": 120})
            for m in ("prefill_hidden", "post_hidden", "extra_combined")]
    result = evaluate_public_rows(rows, [case("a")], tmp_path,
                                 expected_methods=("prefill_hidden", "post_hidden", "extra_combined"), bootstrap_iterations=0)
    timing = result["trajectory_timing"]
    assert timing["n_unique_trajectories"] == 1 and timing["n_measured_trajectory_totals"] == 1
    assert timing["trajectory_total_ms"]["p50"] == 120
    assert all("latency" not in v for v in result["methods"].values())


def test_conflicting_shared_timings_are_not_averaged(tmp_path):
    rows = [row("a", trajectory_id="a", trajectory_timing_ms=100), row("b", trajectory_id="a", trajectory_timing_ms=200)]
    result = evaluate(rows, [case("a"), case("b")], tmp_path)
    assert result["trajectory_timing"]["conflicting_timing_ids"] == ["a"]
    assert result["trajectory_timing"]["n_measured_trajectory_totals"] == 0


def test_bootstrap_resamples_whole_images(tmp_path):
    manifest = [case(str(i), group="image1" if i < 3 else "image2") for i in range(6)]
    rows = [row(str(i), "click" if i < 3 else "wait") for i in range(6)]
    result = evaluate_public_rows(rows, manifest, tmp_path, expected_methods=("prefill_hidden",),
                                 bootstrap_iterations=300, seed=7)
    ci = result["methods"]["prefill_hidden"]["group_bootstrap"]
    assert result["n_manifest_groups"] == 2
    assert ci["metrics"]["action_aware_correct"]["ci95"] == [0, 1]
    assert ci["metrics"]["action_aware_correct"]["point"] == 0.5


@pytest.mark.parametrize("change", [
    {"expected_actionable": "true"},
    {"target_geometry": {"type": "bbox", "boxes": [[0.8, 0, 0.1, 1]]}},
    {"target_geometry": {"type": "polygon", "polygons": [[[0, 0], [1, 1], [0.5, 0.5]]]}},
])
def test_bad_manifest_is_rejected_before_metrics(tmp_path, change):
    manifest = [{**case("a"), **change}]
    with pytest.raises(ValueError):
        evaluate([row("a")], manifest, tmp_path)


def test_jsonl_wrapper_and_unknown_method_names(tmp_path):
    manifest_path, rows_path = tmp_path / "manifest.jsonl", tmp_path / "raw.jsonl"
    manifest_path.write_text(json.dumps(case("a")) + "\n", encoding="utf-8")
    rows_path.write_text(json.dumps(row("a", method="frozen_linear_readout")) + "\n", encoding="utf-8")
    result = evaluate_public_jsonl(rows_path, manifest_path, tmp_path / "out",
                                  expected_methods=("frozen_linear_readout",), bootstrap_iterations=0)
    assert metrics(result, "frozen_linear_readout")["action_aware_correct"]["rate"] == 1
    with pytest.raises(ValueError, match="unique"):
        evaluate_public_rows([], [case("a")], tmp_path, expected_methods=("same", "same"))
