"""Visual-state evaluation invariants, using numeric fixtures, not VLM results."""

import csv
import json

import joblib
import numpy as np
import pytest

from clickagent.probes import (
    LABELS,
    _group_bootstrap_statistics,
    _paired_visual_statistics,
    gate_metrics,
    run_probe_suite,
    run_text_baseline,
)


STATES = ("click", "disabled", "absent")


def _visual_fixture(tmp_path, conditions=("prefill",)):
    rng = np.random.default_rng(55)
    rows = []
    for split, n_groups in (("train", 16), ("val", 8), ("test", 8)):
        for index in range(n_groups):
            for label in STATES:
                rows.append({
                    "sample_id": f"{split}_{index}_{label}", "group_id": f"{split}_{index}",
                    "label": label, "split": split, "instruction": "点击保存",
                    "metadata": {"domain": "ood" if split == "test" and index % 2 else "iid", "layout": "dialog" if index % 2 else "settings", "style_family": "new" if split == "test" and index % 2 else "base"},
                })
    labels = np.asarray([row["label"] for row in rows])
    state_signal = np.eye(3)[[STATES.index(label) for label in labels]]
    arrays = {
        "attention": np.c_[state_signal * 6 + rng.normal(0, 0.05, state_signal.shape), rng.normal(size=len(rows))].astype(np.float32),
        "hidden": np.c_[state_signal * 4 + rng.normal(0, 0.05, state_signal.shape), rng.normal(size=(len(rows), 2))].astype(np.float32),
        "sample_ids": np.asarray([row["sample_id"] for row in rows]), "labels": labels,
        "groups": np.asarray([row["group_id"] for row in rows]), "splits": np.asarray([row["split"] for row in rows]),
    }
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    for condition in conditions:
        np.savez(feature_dir / f"{condition}.npz", **arrays)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    return feature_dir, manifest, arrays, rows


def test_visual_state_three_probes_fixed_text_and_domain_metrics(tmp_path):
    features, manifest, arrays, _ = _visual_fixture(tmp_path)
    output = tmp_path / "results"
    summary = run_probe_suite(features, output, manifest_path=manifest, task="visual_state", c_values=(0.1, 1.0), bootstrap_iterations=128)
    assert summary["task"] == "visual_state"
    assert summary["label_order"] == list(STATES)
    assert len(summary["runs"]) == 4
    assert summary["text_baseline"]["n_unique_instructions"] == 1
    for run in summary["runs"]:
        assert run["test"]["label_order"] == list(STATES)
        assert set(run["test"]["label_counts"]) == set(STATES)
        assert run["group_bootstrap"]["n_groups"] == 8
        domains = run["test_slices"]["axes"]["domain"]
        assert set(domains) == {"iid", "ood"}
        assert domains["iid"]["classification"]["n"] == 12
        assert domains["ood"]["n_groups"] == 4
        paired = run["paired_visual_state"]
        assert paired["n_complete_triplets"] == 8
        assert paired["n_incomplete_groups"] == 0
        for gate in run["click_gates"]:
            by_state = gate["test"]["false_click_rate_by_negative_state"]
            assert by_state["disabled"]["n"] == 8
            assert by_state["absent"]["n"] == 8
        if run["feature_set"] == "text_only":
            assert run["test"]["accuracy"] == pytest.approx(1 / 3)
            assert run["test"]["click_auroc"] == pytest.approx(0.5)
            assert paired["triplet_all_correct_rate"] == 0
            assert paired["paired_flip"]["both_positive_rate"] == 0
            assert run["group_bootstrap"]["accuracy"]["ci95"] == pytest.approx([1 / 3, 1 / 3])
            model = joblib.load(output / run["model_path"])
            assert not ({"iid", "ood", "disabled", "absent"} & set(model["tfidf"].vocabulary_))
        else:
            assert run["test"]["accuracy"] == 1
            assert paired["triplet_all_correct_rate"] == 1
            assert paired["paired_flip"]["both_positive_rate"] == 1
            model = joblib.load(output / run["model_path"])
            expected = arrays["attention"].shape[1] if run["feature_set"] == "attention" else arrays["hidden"].shape[1]
            if run["feature_set"] == "combined":
                expected = arrays["attention"].shape[1] + arrays["hidden"].shape[1]
            assert model["scale"].n_features_in_ == expected
    predictions = list(csv.DictReader((output / "predictions.csv").open(encoding="utf-8-sig")))
    assert "p_disabled" in predictions[0] and "p_absent" in predictions[0]
    assert "p_describe" not in predictions[0]
    assert "同场景三联组" in (output / "report.md").read_text(encoding="utf-8")


def test_task_labels_are_explicit_and_cannot_pollute_later_intent_run(tmp_path):
    features, manifest, _, rows = _visual_fixture(tmp_path, ())
    with pytest.raises(ValueError, match="unknown labels"):
        run_text_baseline(manifest, tmp_path / "wrong_default", bootstrap_iterations=0)
    visual = run_text_baseline(manifest, tmp_path / "visual", task="visual_state", bootstrap_iterations=64)
    assert visual["runs"][0]["test"]["click_auroc"] == 0.5
    assert LABELS == ("click", "describe", "forbid")
    mapping = dict(zip(STATES, LABELS))
    for row in rows:
        row["label"] = mapping[row["label"]]
        row["instruction"] = {"click": "点击保存", "describe": "描述保存", "forbid": "禁止点击保存"}[row["label"]]
    intent_manifest = tmp_path / "intent.jsonl"
    intent_manifest.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    intent = run_text_baseline(intent_manifest, tmp_path / "intent", task="intent", bootstrap_iterations=64)
    assert intent["label_order"] == list(LABELS)
    assert intent["runs"][0]["test"]["accuracy"] == 1
    assert not list(features.glob("*.npz"))


def test_group_bootstrap_keeps_all_three_rows_together_and_handles_score_ties():
    labels = np.asarray(STATES * 2)
    groups = np.asarray(["a"] * 3 + ["b"] * 3)
    probabilities = np.asarray([[.9, .05, .05], [.2, .7, .1], [.1, .1, .8], [.3, .6, .1], [.8, .15, .05], [.7, .2, .1]])
    gates = [{"status": "ok", "false_positive_budget": .05, "threshold": .5}]
    first = _group_bootstrap_statistics(labels, probabilities, groups, gates, STATES, 42, 1000)
    second = _group_bootstrap_statistics(labels, probabilities, groups, gates, STATES, 42, 1000)
    assert first == second
    assert first["accuracy"]["point"] == .5
    assert first["accuracy"]["ci95"] == [0, 1]
    assert first["click_auroc"]["point"] == .75
    assert first["click_auroc"]["ci95"] == [0, 1]
    assert first["click_gates"][0]["false_click_rate"]["point"] == .5
    assert first["click_gates"][0]["recall"]["ci95"] == [0, 1]
    constant = np.full((6, 3), 1 / 3)
    tied = _group_bootstrap_statistics(labels, constant, groups, gates, STATES, 42, 128)
    assert tied["click_auroc"]["ci95"] == [.5, .5]


def test_incomplete_visual_triplet_is_disclosed_and_gate_requires_all_states():
    labels = np.asarray(STATES * 2)
    groups = np.asarray(["a"] * 3 + ["b"] * 3)
    probabilities = np.asarray([[.8, .1, .1], [.1, .8, .1], [.1, .1, .8]] * 2)
    gates = [{"status": "ok", "false_positive_budget": .05, "threshold": .5}]
    complete = _paired_visual_statistics(labels, probabilities, groups, gates, STATES)
    assert complete["click_gates"][0]["complete_triplet_correct_gate_rate"] == 1
    probabilities[1] = [.7, .2, .1]  # A disabled button becomes a false click.
    error = _paired_visual_statistics(labels, probabilities, groups, gates, STATES)
    assert error["triplet_all_correct_rate"] == .5
    assert error["click_gates"][0]["complete_triplet_correct_gate_rate"] == .5
    partial = _paired_visual_statistics(labels[:-1], probabilities[:-1], groups[:-1], gates, STATES)
    assert partial["n_total_groups"] == 2
    assert partial["n_complete_triplets"] == 1
    assert partial["incomplete_group_ids"] == ["b"]
    false_clicks = gate_metrics(labels, probabilities[:, 0], .5, STATES)["false_click_rate_by_negative_state"]
    assert false_clicks["disabled"]["false_click_rate"] == .5
    assert false_clicks["absent"]["false_click_rate"] == 0


def test_visual_generated_unknown_stays_unrecognized_and_has_state_binary_baseline(tmp_path):
    features, manifest, arrays, rows = _visual_fixture(tmp_path, ("prefill", "post_decision"))
    metadata = []
    for row in rows:
        decision = row["label"]
        if row["group_id"] == "test_0":
            decision = {"click": "CLICK", "disabled": "UNKNOWN", "absent": "CLICK"}[row["label"]]
        metadata.append({**row, "generated_decision": decision, "generation_enabled": True, "truncated": False})
    (features / "metadata.jsonl").write_text("\n".join(json.dumps(row) for row in metadata), encoding="utf-8")
    result = run_probe_suite(features, tmp_path / "result", manifest_path=manifest, task="visual_state", c_values=(.1,), bootstrap_iterations=64)
    baseline = result["extraction_audit"]["direct_generation"]["paired_test"]
    assert baseline["n"] == 24
    assert baseline["no_decision_count"] == 1
    assert baseline["wrong_decision_count"] == 1
    assert baseline["prediction_order"] == [*STATES, "unrecognized"]
    assert baseline["click_gate"]["false_click_rate_by_negative_state"]["disabled"]["false_click_rate"] == 0
    assert baseline["click_gate"]["false_click_rate_by_negative_state"]["absent"]["false_click_rate"] == 1 / 8
    assert result["coverage"]["n_common"] == len(arrays["labels"])
