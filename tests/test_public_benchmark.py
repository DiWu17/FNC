import json

import numpy as np
import pytest

from clickagent.public_benchmark import load_joint, policies_from_joint, save_joint, select_cohort


def test_selection_fixed_before_results_and_excludes_nonclick_conflicts():
    records = []
    for dataset in ("screenspot", "conflictgui"):
        for index in range(20):
            records.append({"sample_id": f"{dataset}_{index}", "dataset": dataset, "split": "test",
                            "expected_actionable": index < 12, "task_kind": "grounding",
                            "metadata": {"reference_action_type": "click" if index % 3 else "type"}})
    records.append({**records[0], "sample_id": "calibration", "split": "calibration"})
    a, audit = select_cohort(records, 10)
    b, _ = select_cohort(list(reversed(records)), 10)
    assert a == b
    assert all(r["sample_id"] != "calibration" for r in a)
    assert audit["datasets"]["screenspot"]["negative"] == 5
    assert all(r["metadata"]["reference_action_type"] == "click" for r in a if r["dataset"] == "conflictgui")


class Gate:
    classes_ = ["absent", "click", "disabled"]

    def __init__(self, features, probability):
        self.features = features
        self.probability = probability

    def predict_proba(self, x):
        assert x.tolist() == [self.features]
        return np.asarray([[0, self.probability, 1 - self.probability]])


class Grounder:
    head_order = ("one",)

    def predict_point(self, attention, grid, size):
        assert attention.tolist() == [[.2]]
        return {"point": [10, 20]}


def joint():
    snapshot = {"attention": np.asarray([1., 2.]), "hidden": np.asarray([3.]),
                "names": ["a"], "head_order": ["one"], "raw_image_attention": np.asarray([[.2]])}
    return {"prefill": snapshot, "post_decision": snapshot, "generated_decision": "click",
            "prefill_raw_map": np.asarray([[.2]]), "head_order": ["one"], "grid_shape": [1, 1],
            "image_size": [100, 100]}


def test_frozen_policy_combined_order_and_unconditional_point():
    gates = {(stage, feature): (Gate(features, .6), .7) for stage in ("prefill", "post_decision")
             for feature, features in (("attention", [1., 2.]), ("hidden", [3.]), ("combined", [1., 2., 3.]))}
    rows = policies_from_joint(joint(), Grounder(), gates)
    assert len(rows) == 7
    assert all(row["localization_point"] == [10, 20] for row in rows)
    assert all(row["action"] == {"type": "wait"} for row in rows if row["method"] != "text_decision_shared")
    assert next(row for row in rows if row["method"] == "text_decision_shared")["action"]["type"] == "click"
    result = joint()
    result["post_decision"] = None
    rows = policies_from_joint(result, Grounder(), gates)
    assert not next(row for row in rows if row["method"] == "post_hidden")["parse_valid"]


def test_joint_cache_roundtrip_and_input_fingerprint(tmp_path):
    result = joint()
    path = tmp_path / "cache.npz"
    save_joint(path, result, "frozen", "image1")
    restored = load_joint(path, "frozen", "image1")
    assert np.array_equal(restored["prefill"]["hidden"], result["prefill"]["hidden"])
    assert np.array_equal(restored["prefill_raw_map"], result["prefill_raw_map"])
    assert "raw_image_attention" not in restored["prefill"]
    with pytest.raises(ValueError, match="fingerprint"):
        load_joint(path, "frozen", "image2")
