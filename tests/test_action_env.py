"""Real headless-browser smoke checks; no desktop/native mouse is used."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from clickagent.action_env import DEFAULT_EDGE, ControlledSaveEnvironment
from clickagent.dataset import load_manifest
from clickagent.visual_state_dataset import build_visual_state_dataset


@pytest.fixture(scope="module")
def browser_samples(tmp_path_factory):
    manifest = build_visual_state_dataset(tmp_path_factory.mktemp("browser_samples"), num_scenarios=8, seed=992)
    rows = load_manifest(manifest)
    first_group = rows[0]["group_id"]
    return manifest.parent, {row["label"]: row for row in rows if row["group_id"] == first_group}


@pytest.fixture(scope="module")
def environment():
    pytest.importorskip("playwright.sync_api")
    if not DEFAULT_EDGE.is_file():
        pytest.skip("These integration tests require the configured local Edge executable")
    with ControlledSaveEnvironment(headless=True) as environment:
        yield environment


def _center(row, root, box=None):
    with Image.open(root / row["image"]) as image:
        width, height = image.size
    box = box if box is not None else row["target_bbox"]
    return (box[0] + box[2]) * width / 2, (box[1] + box[3]) * height / 2


@pytest.mark.parametrize("label", ["click", "disabled", "absent"])
def test_initial_browser_pixels_equal_source_png(environment, browser_samples, label):
    root, rows = browser_samples
    environment.reset(rows[label], root)
    with Image.open(root / rows[label]["image"]) as source:
        assert np.array_equal(np.asarray(environment.observe()), np.asarray(source.convert("RGB")))
    state = environment.inspect()
    assert state["save_count"] == state["other_action_count"] == 0
    assert state["attempts"] == []
    assert environment.verify()["reset_pixel_identity_verified"] is True


def test_enabled_save_dispatches_trusted_click_and_visible_success(environment, browser_samples):
    root, rows = browser_samples
    environment.reset(rows["click"], root)
    result = environment.click(*_center(rows["click"], root))
    assert result["save_count"] == 1
    assert result["other_action_count"] == 0
    assert result["target_hit"] is True
    assert result["first_hit"]["text"] == "保存"
    assert result["last_attempt"]["trusted"] is True
    assert result["last_attempt"]["outcome"] == "saved"
    assert result["verification"]["pixel_changed"] is True
    assert result["verification"]["changed_pixel_count"] > 0
    assert result["verification"]["feedback_text"] == "保存成功"
    assert result["verification"]["ui_state_consistent"] is True


def test_disabled_save_records_attempt_without_action(environment, browser_samples):
    root, rows = browser_samples
    environment.reset(rows["disabled"], root)
    result = environment.click(*_center(rows["disabled"], root))
    assert result["target_hit"] is True
    assert result["save_count"] == result["other_action_count"] == 0
    assert len(result["attempts"]) == 1
    assert result["last_attempt"]["outcome"] == "disabled_target"
    assert result["verification"]["pixel_changed"] is False
    assert result["verification"]["ui_state_consistent"] is True


def test_absent_replacement_executes_other_action(environment, browser_samples):
    root, rows = browser_samples
    sample = rows["absent"]
    environment.reset(sample, root)
    result = environment.click(*_center(sample, root, sample["metadata"]["counterfactual_bbox"]))
    assert result["target_hit"] is False
    assert result["save_count"] == 0
    assert result["other_action_count"] == result["other_attempt_count"] == 1
    assert result["mapped_target"]["text"] == sample["metadata"]["replacement"]["text"]
    assert result["verification"]["pixel_changed"] is True
    assert result["verification"]["feedback_text"].startswith("已执行：")


def test_blank_click_and_disabled_other_have_no_action(environment, browser_samples):
    root, rows = browser_samples
    sample = rows["click"]
    environment.reset(sample, root)
    result = environment.click(2, 2)
    assert result["last_attempt"]["outcome"] == "blank"
    assert result["mapped_target"] is None
    assert result["save_count"] == result["other_action_count"] == 0
    assert result["verification"]["pixel_changed"] is False
    control = next(control for control in sample["metadata"]["controls"]
                   if not control["enabled"] and control["text"] != "保存")
    result = environment.click(*_center(sample, root, control["bbox"]))
    assert result["last_attempt"]["outcome"] == "disabled_other"
    assert result["other_action_count"] == 0
    assert result["other_attempt_count"] == 1
    assert result["verification"]["pixel_changed"] is False


def test_reset_clears_counters_and_feedback(environment, browser_samples):
    root, rows = browser_samples
    sample = rows["click"]
    environment.reset(sample, root)
    environment.click(*_center(sample, root))
    environment.reset(sample, root)
    state = environment.inspect()
    assert state["save_count"] == state["other_action_count"] == state["other_attempt_count"] == 0
    assert state["attempts"] == []
    assert state["feedback_text"] is None
    assert environment.verify()["pixel_changed"] is False


@pytest.mark.parametrize("coordinates", [(-1, 3), (float("nan"), 4), (3, float("inf")), (True, 4), ("3", 4)])
def test_bad_click_coordinates_are_rejected(environment, browser_samples, coordinates):
    root, rows = browser_samples
    environment.reset(rows["click"], root)
    with pytest.raises(ValueError, match="coordinates"):
        environment.click(*coordinates)
    assert environment.inspect()["attempts"] == []


def test_context_is_required_and_missing_browser_is_clear(tmp_path):
    with pytest.raises(RuntimeError, match="context manager"):
        ControlledSaveEnvironment().observe()
    with pytest.raises(FileNotFoundError, match="Browser executable"):
        with ControlledSaveEnvironment(executable_path=tmp_path / "absent.exe"):
            pass
