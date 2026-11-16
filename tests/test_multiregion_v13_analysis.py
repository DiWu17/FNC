"""Synthetic-only tests for the post-audit analysis; no real result reads."""
import copy
import importlib
import json
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def analysis(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    return importlib.import_module("analyze_multiregion_v13_results")


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def cohort_fixture():
    cases, details, pipelines = [], [], []
    wrong, correct = [80, 80], [20, 20]
    negatives = [[80, 80], [70, 70], [60, 60], [50, 50], [40, 40]]
    for i in range(5):
        sid = str(i)
        case = dict(sample_id=sid, dataset="synthetic", image_size=[100, 100],
                    target_geometry={"type": "bbox", "boxes": [[.1, .1, .3, .3]]})
        raw = [wrong] if i == 0 else negatives + [correct]
        c1 = [correct] if i in (1, 4) else [wrong]
        c3 = [wrong] if i in (0, 1) else negatives + [correct] if i == 2 else [correct, wrong]
        pools = {c: {str(k): dict(points=pts, point_oracle=correct in pts) for k in (16, 32)}
                 for c, pts in (("c1", c1), ("c3", c3))}
        methods = {}
        for arm in ("c1_top1", "c3_top1", "c3_top4"):
            condition = arm[:2]
            oracle = pools[condition]["16"]["point_oracle"]
            retained = oracle and (condition == "c1" or i != 2)
            hit = oracle if condition == "c1" else i == 4
            for seed in ("s1", "s2"):
                for policy in (("val",) if condition == "c1" else ("val", "loso")):
                    name = f"{arm}_{seed}_{policy}"
                    methods[name + "__K16"] = dict(hit=hit, top4_oracle=retained, condition=condition)
                    methods[name + "_top4_gui__K16"] = dict(hit=hit, top4_oracle=retained, condition=condition)
        methods["qwen_native_completed_lax"] = dict(hit=False)
        methods["gui_c1__K16"] = dict(hit=correct in c1)
        methods["gui_c3__K16"] = dict(hit=correct in c3)
        methods["gui_c1__K4"] = dict(hit=correct in c1[:4])
        methods["gui_c3__K4"] = dict(hit=correct in c3[:4])
        details.append(dict(sample_id=sid, source="synthetic", methods=methods, pools=pools))
        pipelines.append(dict(sample_id=sid, global_result=dict(all_points=raw), local_views=[]))
        cases.append(case)
    return cases, details, pipelines, dict(complete=True, n=5, cohort="synthetic")


def test_crop_uses_union_geometry_not_envelope_and_distinguishes_boundary(analysis):
    case = dict(image_size=[100, 100], target_geometry=dict(type="bbox", boxes=[
        [.1, .1, .2, .2], [.8, .8, .9, .9]]))
    middle = analysis.crop_relation(case, dict(box=[40, 40, 60, 60]))
    assert middle["relation"] == "separated" and middle["bbox_envelope_would_overclaim"]
    boundary = analysis.crop_relation(case, dict(box=[20, 10, 30, 20]))
    assert boundary["relation"] == "boundary_contact" and boundary["target_area_coverage"] == 0
    covered = analysis.crop_relation(case, dict(box=[0, 0, 30, 30]))
    assert covered["relation"] == "positive_area"
    assert covered["target_area_coverage"] == pytest.approx(.5)


def test_concave_polygon_hole_in_envelope_is_not_coverage(analysis):
    case = dict(image_size=[100, 100], target_geometry=dict(type="polygon", polygons=[[
        [.1, .1], [.9, .1], [.9, .3], [.3, .3], [.3, .9], [.1, .9]]]))
    result = analysis.crop_relation(case, dict(box=[50, 50, 60, 60]))
    assert result["relation"] == "separated" and result["bbox_envelope_would_overclaim"]


def test_all_decoded_union_maps_local_rounding_clipping_and_deduplicates(analysis):
    view = dict(transform=dict(box=[10, 20, 30, 40], scale=2, original_size=[100, 100]),
                result=dict(all_points=[[1, 1], [0, 0], [1000, 1000]]))
    pipeline = dict(global_result=dict(all_points=[[10, 20]]), local_views=[view, copy.deepcopy(view)])
    assert analysis.decoded_union(pipeline, "c3") == [[10, 20], [99, 99]]
    assert analysis.decoded_union(pipeline, "c1") == analysis.decoded_union(pipeline, "c3")


def test_failure_chain_is_exhaustive_and_separate_attributions_remain_fixed(analysis):
    result = analysis.analyze_cohort(*cohort_fixture())
    chain = result["failure_chains"]["c3_top1_s1_val_top4_gui__K16"]
    assert chain["stages"] == {stage: 1 for stage in analysis.CHAIN_STAGES}
    assert (chain["raw_union_oracle"], chain["quota16_oracle"], chain["top4_oracle"], chain["gui_hits"]) == (4, 3, 2, 1)
    coverage = result["oracle_coverage_c3_minus_c1"]["16"]
    assert coverage["wins"] == 2 and coverage["losses"] == 1
    assert coverage["gained_ids"] == ["2", "3"] and coverage["lost_ids"] == ["1"]
    assert {k: len(v) for k, v in result["attribution"].items()} == dict(
        search_same_val_selector=2, selection_same_c3_top1=2, objective_same_c3_selector=4)
    assert {v["seed_role"] for v in result["attribution"]["search_same_val_selector"]} == {
        "fixed_primary_s1", "fixed_robustness_s2"}
    assert "不是原始attention信息上限" in result["raw_decoded_union"]["c3"]["interpretation"]


def test_inconsistent_reachability_chain_fails(analysis):
    with pytest.raises(ValueError, match="Quota"):
        analysis.failure_stage(False, True, True, True)
    with pytest.raises(ValueError, match="GUI"):
        analysis.failure_stage(True, True, False, True)


def test_header_only_t0_and_nonzero_tokens(analysis, tmp_path):
    for t in (0, 3):
        path = tmp_path / f"raw-{t}.npz"
        np.savez_compressed(path, target_attention=np.zeros((t, 96, 2), np.float32))
        assert analysis.raw_token_count(path) == t


def test_crop_overlap_does_not_hide_local_no_name_or_t0(analysis):
    cases, details, pipelines, summary = cohort_fixture()
    pipelines[0]["local_views"] = [dict(feature_path="synthetic-raw",
        transform=dict(box=[0, 0, 40, 40], scale=2, original_size=[100, 100]),
        result=dict(all_points=[], failure="no_name"))]
    result = analysis.analyze_cohort(cases, details, pipelines, summary, token_counts={"synthetic-raw": 0})
    view = result["per_case"][0]["crop_views"][0]
    assert view["relation"] == "positive_area" and view["no_name"] and view["t0"]
    assert result["per_case"][0]["raw_decoded_union"]["c3"]["oracle"] is False
    assert result["crop_geometry"]["by_region"]["0"]["no_name"] == 1
    assert result["crop_geometry"]["by_region"]["0"]["t0"] == 1


def test_sources_are_reported_separately_without_changing_seed(analysis):
    cases, details, pipelines, summary = cohort_fixture()
    for i, detail in enumerate(details):
        detail["source"] = "Pro" if i < 2 else "Venus"
    result = analysis.analyze_cohort(cases, details, pipelines, summary)
    coverage = result["oracle_coverage_c3_minus_c1"]["16"]
    assert coverage["by_source"]["Pro"]["losses"] == 1
    assert coverage["by_source"]["Venus"]["wins"] == 2
    assert any(r["comparison_scope"] == "verification_count_proxy_only_distinct_point_sets"
               for r in result["same_pool_and_native_references"])


def test_gate_does_not_open_test_summary_before_successful_audit(analysis, tmp_path):
    write(tmp_path / "frozen.json", dict(complete=True, fresh_test_features_at_freeze=0))
    write(tmp_path / "final_audit.json", dict(complete=False, version=13))
    for split in analysis.COHORTS:
        path = tmp_path / "evaluation" / split / "summary.json"
        path.parent.mkdir(parents=True); path.write_text("DO NOT READ", encoding="utf-8")
    with pytest.raises(ValueError, match="did not pass"):
        analysis.authorize_reads(tmp_path)


def test_gate_requires_both_summaries_and_rejects_changed_audited_bytes(analysis, tmp_path):
    write(tmp_path / "frozen.json", dict(complete=True, fresh_test_features_at_freeze=0))
    with pytest.raises(ValueError, match="both evaluation"):
        analysis.authorize_reads(tmp_path)
    expected, evaluation = {}, {}
    for split in analysis.COHORTS:
        path = tmp_path / "evaluation" / split / "summary.json"
        write(path, dict(complete=True, n=1))
        expected[str(path)] = analysis.sha(path)
        evaluation[split] = dict(summary_sha256=analysis.sha(path))
    write(tmp_path / "final_audit.json", dict(complete=True, version=13,
        frozen_sha256=analysis.sha(tmp_path / "frozen.json"), input_sha256=expected, evaluation=evaluation))
    ledger, _ = analysis.authorize_reads(tmp_path)
    assert len(ledger.used) == 2
    write(tmp_path / "evaluation/test/summary.json", dict(complete=True, n=2))
    with pytest.raises(ValueError, match="bytes changed"):
        analysis.authorize_reads(tmp_path)


def test_oof_positive_pool_gains_and_losses_are_both_reported(analysis):
    cases = [dict(sample_id="a"), dict(sample_id="b")]
    indices = {c: dict(cases=[dict(sample_id=sid, condition=c) for sid in ("a", "b")]) for c in ("c1", "c3")}
    def labels_for(row, budget):
        values = {("c1", "a"): [1, 0], ("c1", "b"): [0, 0],
                  ("c3", "a"): [0, 0], ("c3", "b"): [1, 1]}
        return np.asarray(values[(row["condition"], row["sample_id"])]), np.asarray([0, 1])
    result = analysis.training_pool_changes(cases, indices, labels_for)
    assert result["16"]["query_oracle_change"]["wins"] == result["16"]["query_oracle_change"]["losses"] == 1
    assert result["16"]["c1_positive_points"] == 1 and result["16"]["c3_positive_points"] == 2
    assert result["16"]["more_positive_points_queries"] == result["16"]["fewer_positive_points_queries"] == 1


def test_report_keeps_denominators_and_warns_about_posthoc_diagnostic(analysis):
    cohort = analysis.analyze_cohort(*cohort_fixture())
    result = dict(cohorts={"test": cohort, "supplement": dict(cohort, n=1)},
                  trainval_pool_changes={}, fixed_model_selection={})
    page = analysis.report_html(result)
    assert "test · 5题，独立分母" in page and "supplement · 1题，独立分母" in page
    assert "不是模型能力天花板" in page and "不按测试选种子" in page
    assert "search_same_val_selector" in page and "selection_same_c3_top1" in page


def test_whole_analysis_runs_only_on_synthetic_audited_artifacts_and_is_idempotent(analysis, tmp_path):
    write(tmp_path / "frozen.json", dict(complete=True, fresh_test_features_at_freeze=0))
    write(tmp_path / "models/selection.json", dict(complete=True, fresh_test_read=False,
          seed_selected_by_test=False, selections={}, cross_validation={}))
    cases, details, pipelines, summary = cohort_fixture()
    evaluation = {}
    for split in analysis.COHORTS:
        case_path = tmp_path / "data" / f"{split}.jsonl"
        case_path.parent.mkdir(parents=True, exist_ok=True)
        case_path.write_text("".join(json.dumps(c) + "\n" for c in cases), encoding="utf-8")
        write(tmp_path / "evaluation" / split / "details.json", details)
        summary_path = tmp_path / "evaluation" / split / "summary.json"
        write(summary_path, dict(summary, cohort=split))
        evaluation[split] = dict(summary_sha256=analysis.sha(summary_path))
        for pipeline in pipelines:
            write(tmp_path / "pipeline" / split / (analysis.key(pipeline["sample_id"]) + ".json"), pipeline)
    for split in ("train", "val"):
        rows = [dict(sample_id="train-a", dataset="synthetic", image_fold=0, oof_provider="fold_0")]
        (tmp_path / "data" / f"{split}.jsonl").write_text(json.dumps(rows[0]) + "\n", encoding="utf-8")
        for condition in ("c1", "c3"):
            cache = tmp_path / "cache" / split / (condition + ".npz")
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(cache, sample_id=np.asarray("train-a"), labels=np.asarray([1., 0.]),
                                pool_indices_16=np.asarray([0, 1]), pool_indices_32=np.asarray([0, 1]))
            write(tmp_path / "cache" / f"{split}_{condition}_index.json",
                  dict(complete=True, cases=[dict(sample_id="train-a", path=str(cache))]))
    expected = {str(p): analysis.sha(p) for p in tmp_path.rglob("*") if p.is_file()}
    for name in ("src/clickagent/public_evaluation.py", "src/clickagent/spatial_geometry.py"):
        expected[str(analysis.path_for(name))] = analysis.sha(name)
    write(tmp_path / "final_audit.json", dict(complete=True, version=13,
          frozen_sha256=analysis.sha(tmp_path / "frozen.json"), input_sha256=expected, evaluation=evaluation))
    result = analysis.run(tmp_path)
    assert result["complete"] and result["gpu_used"] is False and result["model_execution"] is False
    assert result["cohorts"]["test"]["n"] == result["cohorts"]["supplement"]["n"] == 5
    out = tmp_path / "evaluation/mechanism_analysis"
    identities = {p.name: (analysis.sha(p), p.stat().st_mtime_ns) for p in out.iterdir()}
    assert analysis.run(tmp_path) == result
    assert {p.name: (analysis.sha(p), p.stat().st_mtime_ns) for p in out.iterdir()} == identities

