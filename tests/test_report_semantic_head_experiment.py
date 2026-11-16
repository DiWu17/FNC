"""Synthetic file fixtures test report integrity, never experimental scores."""
import importlib.util
import json
from pathlib import Path

import pytest


SPEC = importlib.util.spec_from_file_location("v7_report", Path(__file__).resolve().parents[1] / "scripts/report_semantic_head_experiment.py")
report = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(report)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def synthetic_run(tmp_path):
    """Only two artificial cases per cohort, under pytest's own directory."""
    out = tmp_path / "synthetic_report_fixture"
    out.mkdir()
    jobs = [dict(name=f"{name}_{suffix}", architecture=arch, data=data, recipe=recipe, seed=seed)
        for suffix, seed in (("s1", 20261104), ("s2", 20261105))
        for name, arch, data, recipe in (("flat_base", "flat", "base", "random"),
            ("static_base", "static", "base", "random"), ("semantic_base", "semantic", "base", "random"),
            ("semantic_expanded", "semantic", "expanded", "random"), ("semantic_hard", "semantic", "expanded", "grouped_hard"))]
    cases = [dict(sample_id=f"{split}_{i}", dataset="synthetic_only", experiment_split=split,
                  instruction="SYNTHETIC UNIT TEST, NOT AN EXPERIMENT", image="not_used.png",
                  target_geometry={"type": "bbox", "boxes": [[.2, .2, .3, .3]]}, image_size=[80, 60])
             for split in ("test", "diagnostic") for i in range(2)]
    (out / "manifest.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases), encoding="utf-8")
    write(out / "data_selection.json", {"status": "ready", "extra_by_source": {"synthetic_only": 2},
        "same_image_distinct_nonoverlap_pairs": 2, "same_image_pairs_involving_added_query": 2})
    p = dict(jobs=jobs, primary="semantic_hard_s1__quality", epochs=24, validation_queries=2,
             train_queries_base=2, train_queries_extra=2, train_queries_expanded=4,
             fresh_test_queries=2, diagnostic_queries=2, training_queries_per_epoch=600,
             updates_per_epoch=75, batch_size=8, fresh_test_near_exclusion_radius=4,
             manifest_sha256=report.sha(out / "manifest.jsonl"), data_selection_sha256=report.sha(out / "data_selection.json"))
    write(out / "protocol.json", p)
    metric = dict(n=2, modes={m: dict(hits=1, region=2, own_hits=0, center_hits=0) for m in ("rank", "combined", "quality")},
                  rank_loss=1., quality_loss=1., candidate_region_coverage=2, frozen_fine_point_oracle=2)
    models = {}
    for j in jobs:
        checkpoints = {}
        for rule in report.RULES:
            filename = j["name"] + "__" + rule + ".pt"
            path = out / "models" / filename
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(b"SYNTHETIC UNIT TEST CHECKPOINT -- NOT MODEL WEIGHTS")
            entry = dict(file=filename, sha256=report.sha(path), epoch=0, validation=metric, training=metric)
            checkpoints[rule] = entry
            models[j["name"] + "__" + rule] = dict(entry, job=j, selector=rule)
        write(out / "models" / f"{j['name']}_history.json", dict(complete=True, job=j, core_parameters=100,
            parameters=120, checkpoints=checkpoints, history=[dict(epoch=e, val=metric) for e in range(25)]))
    frozen = dict(primary_model=p["primary"], models=models)
    write(out / "frozen_readouts.json", frozen)
    scores = {}
    for split in ("test", "diagnostic"):
        records, methods = [], {}
        for name in sorted(report.expected_methods(p)):
            selected = []
            for i in range(2):
                hit = (i == 0 or name == "ui_tars") and not name.endswith(("_ownfine", "_center"))
                row = dict(sample_id=f"{split}_{i}", dataset="synthetic_only", method=name,
                           point=[20, 15] if hit else [70, 50], hit=hit)
                if not name.endswith(("_ownfine", "_center")) and ("__" in name or name in ("v5_fixed", "v6_s1_fixed", "v6_s2_fixed")):
                    row.update(selected_region_overlap=True, region_top3=True, region_top5=True,
                               candidate_region_oracle=True, frozen_fine_point_oracle=True,
                               failure_stage="hit" if hit else "fixed_fine_miss_after_intersecting_parent")
                selected.append(row)
            records.extend(selected)
            methods[name] = dict(n=2, hits=sum(r["hit"] for r in selected), points=2,
                accuracy=sum(r["hit"] for r in selected)/2, datasets={"synthetic_only": dict(n=2, hits=sum(r["hit"] for r in selected))})
            if any("selected_region_overlap" in r for r in selected):
                methods[name].update(region_top1=2, region_top3=2, region_top5=2,
                                     candidate_region_coverage=2, frozen_fine_point_oracle=2)
        dest = out / "evaluation" / split
        dest.mkdir(parents=True)
        (dest / "predictions.jsonl").write_text("".join(json.dumps(r)+"\n" for r in records), encoding="utf-8")
        comparisons = []
        for suffix in ("s1", "s2"):
            for factor in report.FACTOR_LABELS:
                a, b = report.method("semantic_hard_"+suffix, "quality", "combined"), "ui_tars"
                comparisons.append(dict(factor=factor, first=a, second=b, delta_pp=-50., ci95_pp=[-100., 0.], wins=0, losses=1))
        summary = dict(n=2, primary=p["primary"]+"__combined", test_is_fresh=split=="test", split=split,
                       manifest_sha256=p["manifest_sha256"], frozen_sha256=report.sha(out / "frozen_readouts.json"),
                       methods=methods, prediction_count=len(records), predictions_sha256=report.sha(dest / "predictions.jsonl"),
                       paired_comparisons=comparisons)
        write(dest / "summary.json", summary)
        scores[split] = dict(prediction_sha256=summary["predictions_sha256"])
    write(out / "final_audit.json", dict(complete=True, frozen_sha256=report.sha(out / "frozen_readouts.json"),
        protocol_sha256=report.sha(out / "protocol.json"), manifest_sha256=p["manifest_sha256"], scores=scores,
        trajectories_verified=10, checkpoints_verified=30, fresh_primary_cpu_gpu_exact=True,
        previous_frozen_artifacts_unchanged=True, fresh_features_after_freeze=True,
        backbone_trainable_parameters=0, tests_passed=39))
    return out


def test_prespecified_primary_is_not_replaced_by_better_test_control(synthetic_run):
    bundle = report.load_bundle(synthetic_run)
    assert bundle["primary"] == "semantic_hard_s1__quality__combined"
    assert bundle["summaries"]["test"]["methods"]["ui_tars"]["hits"] > bundle["summaries"]["test"]["methods"][bundle["primary"]]["hits"]
    table = report.checkpoint_table(bundle, "test")
    assert len(table) == 30
    assert sum("S1" in r[0] for r in table) == sum("S2" in r[0] for r in table) == 15
    assert {r[2] for r in table} == set(report.RULE_LABELS.values())


def test_incomplete_audit_and_changed_predictions_fail_closed(synthetic_run):
    path = synthetic_run / "final_audit.json"
    audit = report.read(path)
    write(path, dict(audit, complete=False))
    with pytest.raises(ValueError, match="incomplete"):
        report.load_bundle(synthetic_run)
    write(path, audit)
    predictions = synthetic_run / "evaluation/test/predictions.jsonl"
    predictions.write_text(predictions.read_text(encoding="utf-8")+"\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Predictions changed"):
        report.load_bundle(synthetic_run)
    assert not (synthetic_run / "evaluation/assets").exists()


def test_missing_trajectory_is_not_silently_omitted(synthetic_run):
    path = synthetic_run / "models/flat_base_s2_history.json"
    history = report.read(path)
    write(path, dict(history, complete=False))
    with pytest.raises(ValueError, match="trajectory"):
        report.load_bundle(synthetic_run)


def test_missing_real_results_do_not_create_placeholder_report(tmp_path):
    with pytest.raises(FileNotFoundError):
        report.main(["--output", str(tmp_path / "not_completed")])
    assert not (tmp_path / "not_completed").exists()


def test_full_report_render_uses_only_synthetic_pytest_fixture(synthetic_run):
    bundle = report.load_bundle(synthetic_run)
    # A coordinate replay can be exact while stochastic descriptor rounding
    # has only per-element witnesses. Do not report a joint bitwise replay.
    bundle['audit']['fixed_points'] = dict(cpu_numeric_exceptions=[dict(
        sample_id='SYNTHETIC_NUMERIC_EXCEPTION',candidate_index=0,canonical_points_exact=True,
        canonical_all_13_float16_features_exact=False,canonical_all_13_float16_elements_reproduced=True,
        cpu_continuous_point=[.499999,.5],canonical_gpu_continuous_point=[.500001,.5],
        cpu_point=[0,0],stored_point=[1,0],hit_changed=False,point_oracle_changed=False)],
        canonical_gpu_exception_batches=[dict(split='test',batch_start=0,n_candidates=2,
            runs=[dict(float16_feature_mismatch_elements=n) for n in [2,1,1,1,1]],
            whole_batch_13_column_exact_witness_runs=[],all_13_columns_elementwise_reproduced=True,
            interrun_features_bitstable=False)])
    integrity = report.render_report(bundle, max_examples=0)
    html = (synthetic_run / "evaluation/report.html").read_text(encoding="utf-8")
    md = (synthetic_run / "evaluation/report.md").read_text(encoding="utf-8")
    assert "data:image/png;base64," in html and "<html lang=\"zh\">" in html
    assert "同一轨迹" in html and "同一检查点" in html and "旧 192" in md
    assert "每个缓存元素都实际复现" in html and "没有同一遍整批完全一致" in html
    assert "不能称所有特征跨设备逐位一致" in html
    assert "1/2 (50.0%)" in md and "2/2 (100.0%)" in md
    assert integrity["primary"] == bundle["primary"] and integrity["test_winner_selection"] is False
    assert integrity["trajectories"] == 10 and integrity["examples"] == 0
    assert len(integrity["assets_sha256"]) == 14
    assert all((synthetic_run / "evaluation" / filename).is_file() for filename in integrity["assets_sha256"])
    assert report.sha(synthetic_run / "evaluation/report.html") == integrity["report_sha256"]["report.html"]


def test_dataset_failure_denominator_keeps_missing_name():
    examples=[dict(method="primary",dataset="a",point=[1,1],hit=True,failure_stage="hit",
                   selected_region_overlap=True,candidate_region_oracle=True,frozen_fine_point_oracle=True),
              dict(method="primary",dataset="a",point=None,hit=False,failure_stage="no_name"),
              dict(method="primary",dataset="b",point=[4,4],hit=False,failure_stage="no_target_peak_candidate")]
    result=report.failure_summary(examples,"primary","a")
    assert result['n']==2 and result['hits']==1 and result['stages']['no_name']==1
    assert result['parent_oracle']==result['frozen_point_oracle']==1


def test_overlay_renders_full_union_and_missing_comparison_point(tmp_path):
    image=tmp_path/'synthetic_image.png'
    report.Image.new('RGB',(80,60),'white').save(image)
    case=dict(image=str(image),target_geometry={'type':'bbox','boxes':[[.2,.2,.3,.3],[.6,.5,.8,.7]]})
    first=dict(point=[20,15],candidates=[dict(region=[0,0,.5,.5],point=[20,15])])
    second=dict(point=None,candidates=[])
    path=tmp_path/'overlay.png'
    report.plot_example(case,first,second,path)
    assert path.is_file() and path.stat().st_size>1000

