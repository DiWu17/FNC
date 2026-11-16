"""Fresh test provenance, copied pixels and screenshot-group isolation."""

import hashlib
import json
from collections import Counter

import pytest

from clickagent.action_dataset import build_action_dataset
from clickagent.action_env import ENV_VERSION
from clickagent.dataset import load_manifest
from clickagent.visual_state_dataset import build_visual_state_dataset


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    root = tmp_path_factory.mktemp("action_dataset")
    source = build_visual_state_dataset(root / "source", num_scenarios=8, seed=43)
    action = build_action_dataset(root / "action", source_manifest=source, seed=144, test_scenarios=8)
    return source, action, load_manifest(source), load_manifest(action)


def test_old_trainval_reused_but_old_test_excluded(generated):
    source, action, source_rows, rows = generated
    old_trainval = {row["sample_id"]: row for row in source_rows if row["split"] != "test"}
    old_test = {row["sample_id"] for row in source_rows if row["split"] == "test"}
    new_trainval = {row["sample_id"]: row for row in rows if row["split"] != "test"}
    assert new_trainval.keys() == old_trainval.keys()
    assert not old_test & {row["sample_id"] for row in rows}
    for sid, row in new_trainval.items():
        assert row["label"] == old_trainval[sid]["label"]
        assert row["split"] == old_trainval[sid]["split"]
        assert row["target_bbox"] == old_trainval[sid]["target_bbox"]
        assert (action.parent / row["image"]).read_bytes() == (source.parent / old_trainval[sid]["image"]).read_bytes()


def test_new_test_has_new_seed_domains_and_paired_labels(generated):
    _, _, _, rows = generated
    test = [row for row in rows if row["split"] == "test"]
    assert len(test) == 24
    assert {row["metadata"]["seed"] for row in test} == {144}
    assert Counter(row["label"] for row in test) == {"click": 8, "disabled": 8, "absent": 8}
    groups = {row["group_id"]: row["metadata"]["domain"] for row in test}
    assert Counter(groups.values()) == {"iid": 4, "ood": 4}
    assert {row["metadata"]["layout"] for row in test if row["metadata"]["domain"] == "ood"} == {"table", "filemanager"}
    assert {row["metadata"]["layout"] for row in test if row["metadata"]["domain"] == "iid"} == {"dialog", "settings", "toolbar", "webpage"}
    assert all(row["metadata"]["action_env_version"] == ENV_VERSION for row in rows)


def test_summary_hashes_and_local_relative_paths(generated):
    _, action, _, rows = generated
    summary = json.loads((action.parent / "summary.json").read_text(encoding="utf-8"))
    assert summary["manifest_sha256"] == hashlib.sha256(action.read_bytes()).hexdigest()
    assert summary["num_samples"] == len(rows) == 39
    assert summary["excluded_old_test_samples"] == 9
    assert summary["test_seed"] == 144
    digest = hashlib.sha256()
    for image in sorted({row["image"] for row in rows}):
        assert not image.startswith(("/", "\\"))
        assert ".." not in image.split("/")
        digest.update(image.encode("utf-8") + b"\0")
        digest.update(hashlib.sha256((action.parent / image).read_bytes()).digest())
    assert digest.hexdigest() == summary["images_sha256"]
    assert (action.parent / "contact_sheet.png").is_file()


def test_same_seed_reproduces_action_manifest_and_pixels(generated, tmp_path):
    source, original, _, original_rows = generated
    repeated = build_action_dataset(tmp_path / "repeat", source_manifest=source, seed=144, test_scenarios=8)
    assert repeated.read_bytes() == original.read_bytes()
    for row in original_rows:
        assert (original.parent / row["image"]).read_bytes() == (repeated.parent / row["image"]).read_bytes()


def test_source_seed_cannot_be_reused_and_source_not_overwritten(generated):
    source, action, _, _ = generated
    with pytest.raises(ValueError, match="Fresh test seed"):
        build_action_dataset(action.parent / "bad", source_manifest=source, seed=43, test_scenarios=8)
    with pytest.raises(ValueError, match="overwrite"):
        build_action_dataset(source.parent, source_manifest=source, seed=144, test_scenarios=8)


@pytest.mark.parametrize("kwargs", [{"test_scenarios": 7}, {"test_scenarios": True}, {"seed": "../bad"}])
def test_bad_configuration_is_rejected_before_reading_source(tmp_path, kwargs):
    with pytest.raises(ValueError):
        build_action_dataset(tmp_path / "bad", source_manifest=tmp_path / "missing.jsonl", **kwargs)
