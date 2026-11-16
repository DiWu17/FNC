from copy import deepcopy
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from clickagent.action_benchmark import checked_episode, propose_action, run_benchmark, run_episode, stable_hash


class Gate:
    classes_ = np.asarray(["disabled", "click", "absent"])

    def __init__(self, value=.9):
        self.value = value

    def predict_proba(self, features):
        assert features.shape == (1, 2)
        return np.asarray([[(1 - self.value) / 2, self.value, (1 - self.value) / 2]], dtype=np.float32)


class Grounder:
    def predict_point(self, maps, grid, size):
        assert maps.shape == (2, 4)
        return {"point": [4, 7]}


def readout():
    return {"readout_available": True, "hidden": np.ones(2), "prefill_raw_map": np.ones((2, 4)),
            "grid_shape": [2, 2], "image_size": [10, 10], "generated_decision": "click",
            "generated_tokens": 0}


def test_policies_use_predictions_without_truth_coordinate_repair():
    gates = {stage: (Gate(), .5) for stage in ("prefill", "post_decision")}
    for method in ("prefill_hidden", "post_hidden", "text_decision_shared"):
        action, _ = propose_action(method, readout(), Grounder(), gates)
        assert action == {"type": "click", "x": 4, "y": 7}
    result = readout()
    result["readout_available"] = False
    action, meta = propose_action("post_hidden", result, Grounder(), gates)
    assert action == {"type": "wait"} and not meta["parse_valid"]
    result = {"valid": True, "action": "click", "point": [1, 2]}
    action, _ = propose_action("text_coordinates", result, Grounder(), gates)
    assert action == {"type": "click", "x": 1, "y": 2}


def test_float32_probability_cannot_round_back_nextafter_threshold():
    gate = Gate(np.float32(1 / 3))
    threshold = np.nextafter(np.float64(np.float32(1 / 3)), np.inf)
    action, _ = propose_action("prefill_hidden", readout(), Grounder(), {"prefill": (gate, threshold)})
    assert action == {"type": "wait"}
    assert stable_hash({"max_memory": {0: "13GiB", "cpu": "24GiB"}})


def test_each_episode_runs_new_inference_and_preserves_wrong_click():
    class Runtime:
        calls = 0

        def infer_readouts(self, image, stage):
            assert isinstance(image, Image.Image)
            self.calls += 1
            return readout()

    class Env:
        def reset(self, sample, root):
            self.state = {"save_count": 0, "other_action_count": 0, "attempts": [],
                          "target_hit": False, "mapped_target": None}

        def inspect(self):
            return deepcopy(self.state)

        def observe(self):
            return Image.new("RGB", (10, 10))

        def verify(self):
            return {"pixel_changed": False, "changed_pixel_count": 0}

        def click(self, x, y):
            self.state["attempts"].append({"x": x, "y": y, "outcome": "blank"})
            return {**self.inspect(), "verification": self.verify()}

    sample = {"sample_id": "test", "group_id": "scene", "split": "test", "label": "absent",
              "target_bbox": None, "metadata": {"domain": "ood", "layout": "table", "style_family": "neutral"}}
    runtime = Runtime()
    for _ in range(2):
        row, _, _ = run_episode(Env(), runtime, Grounder(), {"prefill": (Gate(), .5)}, sample, ".", "prefill_hidden")
        assert row["action"]["type"] == "click"  # An actual wrong policy output, never oracle-corrected.
        assert len(row["env_after"]["attempts"]) == 1
        assert row["latency_measured"] and row["timing_ms"]["total"] > 0
    assert runtime.calls == 2


def test_failure_is_persisted_and_cannot_be_silently_retried(tmp_path):
    class BrokenEnv:
        def reset(self, *args):
            raise OSError("browser driver died")

        def inspect(self):
            return {}

        def verify(self):
            return {}

    sample = {"sample_id": "s", "group_id": "g", "split": "test", "label": "click"}
    with pytest.raises(OSError):
        checked_episode(tmp_path, BrokenEnv(), None, None, {}, sample, ".", "prefill_hidden")
    failure = json.loads((tmp_path / "failed_attempts.jsonl").read_text(encoding="utf-8"))
    assert failure["elapsed_including_reset_ms"] >= 0 and "OSError" in failure["error"]
    with pytest.raises(ValueError, match="Failed-attempt ledger"):
        run_benchmark(None, [sample], ".", tmp_path, None, {}, {})


def test_orphan_episode_log_is_rejected(tmp_path):
    (tmp_path / "episodes.jsonl").write_text('{}\n', encoding="utf-8")
    sample = {"sample_id": "s", "group_id": "g", "split": "test", "label": "click"}
    with pytest.raises(ValueError, match="orphan log"):
        run_benchmark(SimpleNamespace(), [sample], ".", tmp_path, None, {}, {})
