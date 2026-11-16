"""Experiment-level checks: split integrity, paired coverage and honest gating."""

import csv
import json

import joblib
import numpy as np
import pytest

from clickagent.probes import LABELS, calibrate_click_threshold, gate_metrics, run_probe_suite, run_text_baseline


def _make_experiment(tmp_path, conditions=("prefill", "post_decision")):
    rng = np.random.default_rng(818)
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    labels, ids, groups, splits, records = [], [], [], [], []
    for split, n_groups in (("train", 24), ("val", 12), ("test", 12)):
        for group_number in range(n_groups):
            group = f"{split}_screen_{group_number}"
            for label in LABELS:
                sid = f"{group}_{label}"
                labels.append(label)
                ids.append(sid)
                groups.append(group)
                splits.append(split)
                phrases = {
                    "click": f"请点击第{group_number % 4}个保存按钮",
                    "describe": f"请描述第{group_number % 4}个保存按钮",
                    "forbid": f"禁止点击第{group_number % 4}个保存按钮",
                }
                records.append({"sample_id": sid, "label": label, "instruction": phrases[label], "group_id": group, "split": split, "template_family": "same_family"})
    labels, ids, groups, splits = map(np.asarray, (labels, ids, groups, splits))
    # Hidden features have a strong known signal; attention features are weak.
    onehot = np.eye(3)[[LABELS.index(label) for label in labels]]
    attention = rng.normal(0, 1, (len(labels), 4))
    hidden = np.c_[onehot * 5 + rng.normal(0, 0.15, onehot.shape), rng.normal(0, 1, (len(labels), 2))]
    arrays = {"attention": attention, "hidden": hidden, "sample_ids": ids, "labels": labels, "groups": groups, "splits": splits}
    for condition in conditions:
        np.savez(feature_dir / f"{condition}.npz", **arrays)
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in records), encoding="utf-8")
    return feature_dir, manifest, arrays


def _rewrite(path, **changes):
    with np.load(path, allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    arrays.update(changes)
    np.savez(path, **arrays)


def test_full_experiment_with_language_baseline_and_paired_order(tmp_path):
    features, manifest, arrays = _make_experiment(tmp_path)
    # Feature row ordering must not influence paired comparisons.
    order = np.arange(len(arrays["labels"]))[::-1]
    _rewrite(features / "post_decision.npz", **{key: value[order] for key, value in arrays.items()})
    output = tmp_path / "results"
    summary = run_probe_suite(features, output, manifest_path=manifest, c_values=(0.01, 0.1))
    assert len(summary["runs"]) == 7
    assert summary["text_baseline"]["status"] == "included"
    assert not summary["text_baseline"]["template_split_audit"]["is_template_holdout"]
    assert summary["coverage"]["common_over_union"] == 1
    hidden = next(row for row in summary["runs"] if row["condition"] == "prefill" and row["feature_set"] == "hidden")
    assert hidden["test"]["accuracy"] > 0.95
    assert hidden["test"]["click_auroc"] > 0.95
    assert hidden["test"]["confusion_matrix"] == [[12, 0, 0], [0, 12, 0], [0, 0, 12]]
    language = next(row for row in summary["runs"] if row["feature_set"] == "text_only")
    assert language["test"]["accuracy"] > 0.95
    # Identifiers, labels and split metadata cannot expand numeric model inputs.
    pipeline = joblib.load(output / hidden["model_path"])
    assert pipeline["scale"].n_features_in_ == arrays["hidden"].shape[1]
    assert json.loads((output / "metrics.json").read_text(encoding="utf-8"))["label_order"] == list(LABELS)
    assert "不评估点击坐标" in (output / "report.md").read_text(encoding="utf-8")
    rows = list(csv.DictReader((output / "predictions.csv").open(encoding="utf-8-sig")))
    assert set(row["split"] for row in rows) == {"train", "val", "test"}
    prefill = {row["sample_id"]: row["p_click"] for row in rows if row["condition"] == "prefill" and row["feature_set"] == "hidden"}
    post = {row["sample_id"]: row["p_click"] for row in rows if row["condition"] == "post_decision" and row["feature_set"] == "hidden"}
    assert prefill == post


def test_test_labels_do_not_select_models_thresholds_or_change_predictions(tmp_path):
    features, _, arrays = _make_experiment(tmp_path, ("prefill",))
    first = run_probe_suite(features, tmp_path / "first", c_values=(0.01, 0.1, 1.0))
    test = arrays["splits"] == "test"
    labels = arrays["labels"].copy()
    rotate = {"click": "describe", "describe": "forbid", "forbid": "click"}
    labels[test] = [rotate[label] for label in labels[test]]
    # Additional label-bearing fields must be ignored, never concatenated.
    _rewrite(features / "prefill.npz", labels=labels, gold_onehot=np.eye(3)[[LABELS.index(label) for label in labels]])
    second = run_probe_suite(features, tmp_path / "second", c_values=(0.01, 0.1, 1.0))
    for before, after in zip(first["runs"], second["runs"]):
        assert before["chosen_C"] == after["chosen_C"]
        assert before["validation_search"] == after["validation_search"]
        for old_gate, new_gate in zip(before["click_gates"], after["click_gates"]):
            assert old_gate["threshold"] == new_gate["threshold"]
            assert old_gate["validation"] == new_gate["validation"]
        before_model = joblib.load(tmp_path / "first" / before["model_path"])
        after_model = joblib.load(tmp_path / "second" / after["model_path"])
        np.testing.assert_array_equal(before_model["classifier"].coef_, after_model["classifier"].coef_)
    assert first["runs"][1]["test"]["accuracy"] > second["runs"][1]["test"]["accuracy"]


def test_group_leakage_is_rejected_even_if_offending_row_would_be_excluded(tmp_path):
    features, _, arrays = _make_experiment(tmp_path)
    groups = arrays["groups"].copy()
    index = int(np.flatnonzero(arrays["splits"] == "test")[0])
    groups[index] = groups[0]
    _rewrite(features / "prefill.npz", groups=groups)
    keep = np.arange(len(groups)) != index
    _rewrite(features / "post_decision.npz", **{key: value[keep] for key, value in arrays.items()})
    with pytest.raises(ValueError, match="Group leakage"):
        run_probe_suite(features, tmp_path / "results")


def test_test_feature_distribution_cannot_influence_scaling_or_selection(tmp_path):
    features, _, arrays = _make_experiment(tmp_path, ("prefill",))
    first = run_probe_suite(features, tmp_path / "first", c_values=(0.01, 0.1))
    hidden, attention = arrays["hidden"].copy(), arrays["attention"].copy()
    test = arrays["splits"] == "test"
    hidden[test] += 10000
    attention[test] -= 10000
    _rewrite(features / "prefill.npz", hidden=hidden, attention=attention)
    second = run_probe_suite(features, tmp_path / "second", c_values=(0.01, 0.1))
    for before, after in zip(first["runs"], second["runs"]):
        assert before["chosen_C"] == after["chosen_C"]
        old_model = joblib.load(tmp_path / "first" / before["model_path"])
        new_model = joblib.load(tmp_path / "second" / after["model_path"])
        np.testing.assert_array_equal(old_model["scale"].mean_, new_model["scale"].mean_)
        np.testing.assert_array_equal(old_model["scale"].scale_, new_model["scale"].scale_)
        np.testing.assert_array_equal(old_model["classifier"].coef_, new_model["classifier"].coef_)
        assert before["validation_search"] == after["validation_search"]


def test_missing_post_decision_samples_are_paired_and_disclosed(tmp_path):
    features, _, arrays = _make_experiment(tmp_path)
    # Remove one entire screen group, preserving class balance.
    keep = arrays["groups"] != "test_screen_0"
    _rewrite(features / "post_decision.npz", **{key: value[keep] for key, value in arrays.items()})
    summary = run_probe_suite(features, tmp_path / "results", c_values=(0.1,))
    expected = len(arrays["labels"]) - 3
    assert summary["coverage"]["n_common"] == expected
    coverage = summary["coverage"]["conditions"]["prefill"]
    assert len(coverage["excluded_sample_ids"]) == 3
    assert coverage["splits"]["test"]["n_available"] == 36
    assert coverage["splits"]["test"]["n_common"] == 33
    assert all(run["test"]["n"] == 33 for run in summary["runs"])
    with pytest.raises(ValueError, match="common coverage"):
        run_probe_suite(features, tmp_path / "strict", min_common_coverage=1)


def test_class_missing_from_validation_is_explicit_error(tmp_path):
    features, _, arrays = _make_experiment(tmp_path, ("prefill",))
    keep = ~((arrays["splits"] == "val") & (arrays["labels"] == "forbid"))
    _rewrite(features / "prefill.npz", **{key: value[keep] for key, value in arrays.items()})
    with pytest.raises(ValueError, match="val is missing classes"):
        run_probe_suite(features, tmp_path / "results")


def test_single_class_test_has_unavailable_auc_not_perfect_metric(tmp_path):
    features, _, arrays = _make_experiment(tmp_path, ("prefill",))
    keep = (arrays["splits"] != "test") | (arrays["labels"] == "click")
    _rewrite(features / "prefill.npz", **{key: value[keep] for key, value in arrays.items()})
    summary = run_probe_suite(features, tmp_path / "results", c_values=(0.1,))
    for run in summary["runs"]:
        assert not run["test"]["all_three_classes_present"]
        assert run["test"]["click_auroc"] is None
        assert run["test"]["click_average_precision"] is None
        assert run["click_gates"][0]["test"]["false_click_rate"] is None


def test_threshold_respects_validation_budget_with_tied_probabilities():
    labels = np.asarray(["click", "click", "describe", "forbid", "describe", "forbid"])
    probabilities = np.asarray([0.95, 0.8, 0.9, 0.9, 0.2, 0.1])
    zero = calibrate_click_threshold(labels, probabilities, 0.0)
    # A 25% budget cannot accept either of two tied negatives (that would be 50%).
    quarter = calibrate_click_threshold(labels, probabilities, 0.25)
    half = calibrate_click_threshold(labels, probabilities, 0.5)
    assert zero["threshold"] > 0.9
    assert quarter["validation"]["false_clicks"] == 0
    assert half["validation"]["false_clicks"] == 2
    assert half["validation"]["recall"] == 1
    rejected = gate_metrics(labels, probabilities, 1.01)
    assert rejected["precision"] is None
    assert rejected["conditional_risk"] is None
    assert rejected["coverage"] == 0


def test_float32_ties_stay_rejected_by_float64_nextafter_threshold():
    labels = np.asarray(["click", "describe", "forbid", "click", "describe", "forbid"])
    probabilities = np.full(len(labels), 1 / 3, dtype=np.float32)
    calibration = calibrate_click_threshold(labels, probabilities, 0.0)
    threshold = calibration["threshold"]
    cutoff64 = np.float64(probabilities[0])
    assert threshold == np.nextafter(cutoff64, np.inf)
    assert threshold > cutoff64
    assert calibration["validation"]["n_accepted"] == 0
    # Passing the original float32 array must agree with float64 calibration.
    observed = gate_metrics(labels, probabilities, threshold)
    assert observed == calibration["validation"]
    assert observed["false_click_rate"] == 0
    assert observed["coverage"] == 0
    # Inclusive >= behavior stays intact when the threshold equals the score.
    assert gate_metrics(labels, probabilities, float(cutoff64))["n_accepted"] == len(labels)


def test_constant_attention_probe_cannot_accept_validation_ties_on_test(tmp_path):
    features, _, arrays = _make_experiment(tmp_path, ("prefill",))
    _rewrite(features / "prefill.npz", attention=np.zeros_like(arrays["attention"], dtype=np.float32))
    summary = run_probe_suite(features, tmp_path / "results", c_values=(0.1,), false_positive_budgets=(0.0,))
    attention = next(run for run in summary["runs"] if run["feature_set"] == "attention")
    gate = attention["click_gates"][0]
    assert gate["validation"]["n_accepted"] == 0
    assert gate["test"]["n_accepted"] == 0
    assert gate["test"]["false_click_rate"] == 0


def test_cross_condition_sample_facts_and_manifest_are_checked(tmp_path):
    features, manifest, arrays = _make_experiment(tmp_path)
    labels = arrays["labels"].copy()
    labels[0] = "describe"
    _rewrite(features / "post_decision.npz", labels=labels)
    with pytest.raises(ValueError, match="Condition mismatch"):
        run_probe_suite(features, tmp_path / "bad_condition", manifest_path=manifest)
    _rewrite(features / "post_decision.npz", labels=arrays["labels"])
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    records[0]["label"] = "forbid"
    manifest.write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")
    with pytest.raises(ValueError, match="Manifest label mismatch"):
        run_probe_suite(features, tmp_path / "bad_manifest", manifest_path=manifest)


def test_template_holdout_uses_nested_family_and_allows_train_val_overlap(tmp_path):
    features, manifest, _ = _make_experiment(tmp_path, ("prefill",))
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    for record in records:
        record["template_family"] = "incorrect_top_level"
        record["template_id"] = f"label_specific.{record['label']}"
        record["metadata"] = {"template_family": "new_family" if record["split"] == "test" else "development"}
    manifest.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
    summary = run_probe_suite(features, tmp_path / "results", manifest_path=manifest, c_values=(0.1,))
    audit = summary["text_baseline"]["template_split_audit"]
    assert audit["is_template_holdout"] is True
    assert audit["test_overlap_with_train_val"] == []
    assert audit["train_val_shared_families"] == ["development"]
    assert audit["families_by_split"]["test"] == ["new_family"]
    assert "row-level binomial" in summary["uncertainty_note"]


def test_generated_decision_baseline_counts_invalid_outputs_and_reports_latency(tmp_path):
    features, _, arrays = _make_experiment(tmp_path)
    keep = arrays["groups"] != "test_screen_0"
    _rewrite(features / "post_decision.npz", **{key: value[keep] for key, value in arrays.items()})
    rows = []
    for index, (sid, group, label, split) in enumerate(zip(arrays["sample_ids"], arrays["groups"], arrays["labels"], arrays["splits"])):
        decision = str(label)
        if group == "test_screen_0":
            decision = {"click": None, "describe": "click", "forbid": "invalid-format"}[str(label)]
        rows.append({"sample_id": str(sid), "group_id": str(group), "label": str(label), "split": str(split), "generated_decision": decision, "truncated": bool(group == "test_screen_0"), "generation_enabled": True, "timing_seconds": {"total": index + 1.0}})
    (features / "metadata.jsonl").write_text("\n".join(json.dumps(row) for row in rows), encoding="utf-8")
    for condition in ("prefill", "post_decision"):
        condition_rows = []
        for index, row in enumerate(rows):
            if condition == "post_decision" and not keep[index]:
                continue
            condition_rows.append({**row, "condition": condition, "readout_elapsed_seconds": 0.25 + index / 100, "generated_tokens_at_readout": 0 if condition == "prefill" else 2})
        (features / f"{condition}.metadata.jsonl").write_text("\n".join(json.dumps(row) for row in condition_rows), encoding="utf-8")
    summary = run_probe_suite(features, tmp_path / "results", c_values=(0.1,))
    direct = summary["extraction_audit"]["direct_generation"]
    assert direct["test"]["n"] == 36
    assert direct["test"]["accuracy_including_missing"] == 33 / 36
    assert direct["test"]["recognized_coverage"] == 34 / 36
    assert direct["test"]["wrong_decision_count"] == 1
    assert direct["test"]["no_decision_count"] == 2
    assert direct["test"]["truncated_fraction"] == 3 / 36
    assert direct["paired_test"]["n"] == 33
    assert direct["paired_test"]["accuracy_including_missing"] == 1
    timing = summary["extraction_audit"]["readout_latency"]["post_decision"]["subsets"]["paired_test"]
    assert timing["n"] == 33
    assert timing["readout_elapsed_seconds"]["n_available"] == 33
    assert timing["generated_tokens_at_readout"]["p50"] == 2
    assert "post_decision 已包含模型自己生成的动作类别词" in (tmp_path / "results" / "report.md").read_text(encoding="utf-8")


def test_standalone_text_baseline_needs_no_feature_files_or_images(tmp_path, monkeypatch):
    features, manifest, _ = _make_experiment(tmp_path, ())
    assert not list(features.glob("*.npz"))
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    for item in records:
        item["image"] = "this_image_does_not_exist.png"
    manifest.write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")
    monkeypatch.setattr("clickagent.probes.load_feature_file", lambda *args: pytest.fail("text baseline must never load numeric features"))
    output = tmp_path / "text_only"
    summary = run_text_baseline(manifest, output, c_values=(0.01, 0.1))
    assert summary["experiment_type"] == "text_only_baseline"
    assert len(summary["runs"]) == 1
    assert summary["runs"][0]["feature_set"] == "text_only"
    assert summary["runs"][0]["feature_dimension"] is None
    assert summary["runs"][0]["test"]["accuracy"] > 0.95
    assert summary["sample_audit"]["splits"]["test"]["n_groups"] == 12
    assert not list(output.rglob("*.npz"))
    assert "没有加载 VLM" in (output / "report.md").read_text(encoding="utf-8")
    assert (output / "metrics.json").is_file()
    assert (output / "predictions.csv").is_file()
    model = joblib.load(output / summary["runs"][0]["model_path"])
    assert list(model.named_steps) == ["tfidf", "classifier"]


@pytest.mark.parametrize("problem", ["leak", "missing_val_class", "missing_group"])
def test_standalone_text_baseline_enforces_same_partition_protocol(tmp_path, problem):
    _, manifest, _ = _make_experiment(tmp_path, ())
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    if problem == "leak":
        next(item for item in records if item["split"] == "test")["group_id"] = records[0]["group_id"]
        expected = "Group leakage"
    elif problem == "missing_val_class":
        records = [item for item in records if not (item["split"] == "val" and item["label"] == "forbid")]
        expected = "val is missing classes"
    else:
        records[0].pop("group_id")
        expected = "missing group_id"
    manifest.write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")
    with pytest.raises(ValueError, match=expected):
        run_text_baseline(manifest, tmp_path / "results")


def test_standalone_text_and_paired_text_use_identical_fitting_protocol(tmp_path):
    features, manifest, _ = _make_experiment(tmp_path, ("prefill",))
    paired = run_probe_suite(features, tmp_path / "paired", manifest_path=manifest, c_values=(0.01, 0.1))
    standalone = run_text_baseline(manifest, tmp_path / "text", c_values=(0.01, 0.1))
    paired_run = next(run for run in paired["runs"] if run["feature_set"] == "text_only")
    alone_run = standalone["runs"][0]
    for field in ("chosen_C", "validation_search", "test", "val", "click_gates"):
        assert paired_run[field] == alone_run[field]


@pytest.mark.parametrize("mode", ["disabled", "mixed", "legacy"])
def test_generation_disabled_is_not_a_model_error_and_mixed_coverage_is_explicit(tmp_path, mode):
    features, _, arrays = _make_experiment(tmp_path, ("prefill",))
    records = []
    for sid, group, label, split in zip(arrays["sample_ids"], arrays["groups"], arrays["labels"], arrays["splits"]):
        row = {"sample_id": str(sid), "group_id": str(group), "label": str(label), "split": str(split), "generated_decision": str(label), "truncated": False}
        if mode == "disabled":
            row.update(generation_enabled=False, generated_decision=None)
        elif mode == "mixed":
            row["generation_enabled"] = bool(group != "test_screen_0")
            if group == "test_screen_0":
                row["generated_decision"] = None
            elif group == "test_screen_1" and label == "click":
                # Enabled generation with no recognized class must still fail.
                row.pop("generated_decision")
        elif group == "test_screen_0" and label == "click":
            row["generated_decision"] = None
        records.append(row)
    (features / "metadata.jsonl").write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    output = tmp_path / "results"
    summary = run_probe_suite(features, output, c_values=(0.1,))
    direct = summary["extraction_audit"]["direct_generation"]
    test = direct["test"]
    assert test["n"] == 36
    if mode == "disabled":
        assert direct["status"] == "not_run"
        assert test["status"] == "not_run"
        assert test["n_evaluated"] == 0
        assert test["n_generation_disabled"] == 36
        assert test["generation_run_coverage"] == 0
        assert test["no_decision_count"] == 0
        assert test["accuracy_including_missing"] is None
        assert "直接生成类别基线未运行（not_run）" in (output / "report.md").read_text(encoding="utf-8")
    elif mode == "mixed":
        assert direct["status"] == "ok"
        assert test["n_evaluated"] == 33
        assert test["n_generation_disabled"] == 3
        assert test["generation_run_coverage"] == 33 / 36
        assert test["no_decision_count"] == 1
        assert test["recognized_count"] == 32
        assert test["accuracy_including_missing"] == 32 / 33
        assert sum(map(sum, test["confusion_matrix"])) == 33
    else:
        assert direct["status"] == "ok"
        assert test["n_generation_flag_unknown"] == 36
        assert test["n_generation_disabled"] == 0
        assert test["n_evaluated"] == 36
        assert test["generation_run_coverage"] == 1
        assert test["no_decision_count"] == 1
        assert test["accuracy_including_missing"] == 35 / 36
