"""Counterfactual integrity, task compatibility and split invariants."""

import copy
import json
from collections import Counter, defaultdict

import numpy as np
import pytest
from PIL import Image

from clickagent.dataset import load_manifest, validate_manifest
from clickagent.visual_state_dataset import (
    INSTRUCTION, LAYOUTS, OOD_LAYOUTS, SEEN_LAYOUTS, STATES,
    _scenario_plan, build_visual_state_dataset, write_contact_sheet,
)


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    manifest = build_visual_state_dataset(tmp_path_factory.mktemp("visual_state"), num_scenarios=12, seed=43)
    return manifest, load_manifest(manifest)


def test_same_instruction_and_three_states_per_group(generated):
    _, rows = generated
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
        assert row["instruction"] == INSTRUCTION == "点击保存"
        assert row["task"] == row["metadata"]["task"] == "visual_state"
        assert row["target_text"] == "保存"
    assert len(rows) == 36
    assert len(groups) == 12
    assert Counter(row["label"] for row in rows) == {"click": 12, "disabled": 12, "absent": 12}
    for variants in groups.values():
        assert len(variants) == 3
        assert {row["metadata"]["visual_state"] for row in variants} == set(STATES)
        assert len({row["split"] for row in variants}) == 1
        assert len({row["metadata"]["layout"] for row in variants}) == 1
        assert len({row["metadata"]["style_family"] for row in variants}) == 1


def test_default_plan_has_exact_counts_and_held_out_layouts():
    plan = _scenario_plan(300, 43)
    counts = Counter((row["split"], row["domain"]) for row in plan)
    assert counts == {("train", "iid"): 180, ("val", "iid"): 60,
                      ("test", "iid"): 30, ("test", "ood"): 30}
    assert Counter(row["layout"] for row in plan if row["split"] == "train") == dict.fromkeys(SEEN_LAYOUTS, 45)
    assert Counter(row["layout"] for row in plan if row["domain"] == "ood") == dict.fromkeys(OOD_LAYOUTS, 15)
    for row in plan:
        assert (row["layout"] in OOD_LAYOUTS) == (row["domain"] == "ood")
        if row["domain"] == "ood":
            assert row["split"] == "test"


@pytest.mark.parametrize("count", [8, 9, 12, 24, 50])
def test_small_plans_keep_nonempty_splits_and_domains(count):
    plan = _scenario_plan(count, 43)
    assert len(plan) == count
    assert {row["split"] for row in plan} == {"train", "val", "test"}
    assert {row["domain"] for row in plan if row["split"] == "test"} == {"iid", "ood"}
    assert {row["layout"] for row in plan if row["split"] == "train"} == set(SEEN_LAYOUTS)
    assert {row["layout"] for row in plan if row["domain"] == "ood"} == set(OOD_LAYOUTS)


def test_only_target_pixels_change_in_counterfactual_triplets(generated):
    manifest, rows = generated
    groups = defaultdict(list)
    for row in rows:
        groups[row["group_id"]].append(row)
    for variants in groups.values():
        arrays = [np.asarray(Image.open(manifest.parent / row["image"]).convert("RGB")) for row in variants]
        height, width = arrays[0].shape[:2]
        x0, y0, x1, y1 = variants[0]["metadata"]["counterfactual_bbox"]
        x0, y0, x1, y1 = round(x0 * width), round(y0 * height), round(x1 * width), round(y1 * height)
        outside = np.ones((height, width), dtype=bool)
        outside[y0:y1, x0:x1] = False
        for first in range(3):
            for second in range(first):
                assert np.array_equal(arrays[first][outside], arrays[second][outside])
                assert not np.array_equal(arrays[first][y0:y1, x0:x1], arrays[second][y0:y1, x0:x1])


def test_absent_has_no_target_control_and_other_annotations_match(generated):
    _, rows = generated
    groups = defaultdict(list)
    for row in rows:
        controls = row["metadata"]["controls"]
        assert len(controls) == 6
        targets = [control for control in controls if control["text"] == "保存"]
        if row["label"] == "absent":
            assert row["target_bbox"] is None
            assert not targets
            assert all("保存" not in control["text"] for control in controls)
            replacement = row["metadata"]["replacement"]
            assert replacement["text"] in {"返回", "撤销", "重试", "关闭"}
            assert replacement["enabled"] is True
            assert replacement["bbox"] == row["metadata"]["counterfactual_bbox"]
            assert sum(control["text"] == replacement["text"] for control in controls) == 1
        else:
            assert len(targets) == 1
            assert targets[0]["enabled"] == (row["label"] == "click")
            assert targets[0]["bbox"] == row["target_bbox"]
            assert row["metadata"]["replacement"] is None
        replacement_text = row["metadata"]["replacement"]["text"] if row["metadata"]["replacement"] else None
        distractors = [control for control in controls if control["text"] not in {"保存", replacement_text}]
        assert any(control["enabled"] and control["role"] == "neutral" for control in distractors)
        assert any(not control["enabled"] for control in distractors)
        groups[row["group_id"]].append(distractors)
    for distractors in groups.values():
        assert distractors[0] == distractors[1] == distractors[2]


def test_enabled_and_absent_match_button_appearance_except_label(generated):
    manifest, rows = generated
    groups = defaultdict(dict)
    for row in rows:
        groups[row["group_id"]][row["label"]] = row
    for group in groups.values():
        enabled, absent = group["click"], group["absent"]
        target = next(control for control in enabled["metadata"]["controls"] if control["text"] == "保存")
        replacement = absent["metadata"]["replacement"]
        assert target["role"] == replacement["role"]
        assert target["enabled"] == replacement["enabled"] is True
        pictures = [np.asarray(Image.open(manifest.parent / row["image"]).convert("RGB")) for row in (enabled, absent)]
        height, width = pictures[0].shape[:2]
        x0, y0, x1, y1 = target["bbox"]
        x0, y0, x1, y1 = round(x0 * width), round(y0 * height), round(x1 * width), round(y1 * height)
        first, second = [picture[y0:y1, x0:x1] for picture in pictures]
        border = np.ones(first.shape[:2], dtype=bool)
        border[8:-8, 8:-8] = False
        assert np.array_equal(first[border], second[border])
        assert not np.array_equal(first, second)


def test_order_does_not_encode_fixed_class_cycle(generated):
    _, rows = generated
    assert len({tuple(row["label"] for row in rows[start:start + 3]) for start in range(0, len(rows), 3)}) > 1
    assert all(not any(label in row["image"] for label in ("click", "disabled", "absent")) for row in rows)


def test_summary_and_contact_sheet(generated):
    manifest, rows = generated
    summary = json.loads((manifest.parent / "summary.json").read_text(encoding="utf-8"))
    assert summary["num_samples"] == 36
    assert summary["num_scenarios"] == summary["num_groups"] == 12
    assert summary["images_verified"] == 36
    assert set(summary["scenario_counts_by_layout"]) == set(LAYOUTS)
    assert sum(sum(counts.values()) for counts in summary["scenario_counts_by_split_domain"].values()) == 12
    sheet = write_contact_sheet(manifest)
    with Image.open(sheet) as picture:
        assert picture.size == (1536, 2292)


def test_same_seed_reproduces_manifest_and_pixels(tmp_path):
    first = build_visual_state_dataset(tmp_path / "a", num_scenarios=8, seed=71)
    second = build_visual_state_dataset(tmp_path / "b", num_scenarios=8, seed=71)
    assert first.read_bytes() == second.read_bytes()
    for row in load_manifest(first):
        assert (first.parent / row["image"]).read_bytes() == (second.parent / row["image"]).read_bytes()


def test_null_bbox_only_allowed_for_visual_absent(generated):
    manifest, source = generated
    rows = copy.deepcopy(source)
    absent = next(row for row in rows if row["label"] == "absent")
    absent["target_bbox"] = [0.1, 0.2, 0.3, 0.4]
    with pytest.raises(ValueError, match="absent target_bbox"):
        validate_manifest(rows, manifest)
    rows = copy.deepcopy(source)
    click = next(row for row in rows if row["label"] == "click")
    click["target_bbox"] = None
    with pytest.raises(ValueError, match="target_bbox"):
        validate_manifest(rows, manifest)
    row = copy.deepcopy(source[0])
    row.pop("task")
    row["metadata"].pop("task")
    row.update(label="forbid", target_bbox=None)
    with pytest.raises(ValueError, match="target_bbox"):
        validate_manifest([row], manifest)


def test_task_labels_and_metadata_are_consistent(generated):
    manifest, source = generated
    row = copy.deepcopy(source[0])
    row["label"] = "describe"
    with pytest.raises(ValueError, match="invalid label"):
        validate_manifest([row], manifest)
    row = copy.deepcopy(source[0])
    row["metadata"]["task"] = "intent"
    with pytest.raises(ValueError, match="task.*disagree"):
        validate_manifest([row], manifest)


@pytest.mark.parametrize("count,seed", [(7, 43), (True, 43), (12, "../outside")])
def test_bad_configuration_is_rejected(tmp_path, count, seed):
    with pytest.raises(ValueError):
        build_visual_state_dataset(tmp_path / "bad", num_scenarios=count, seed=seed)
