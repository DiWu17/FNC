"""Meaningful invariants for the grouped intent-screening data."""

import copy
import hashlib
import json
from collections import Counter, defaultdict

import pytest

from clickagent.dataset import LABELS, build_synthetic_dataset, load_manifest, validate_manifest


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    manifest = build_synthetic_dataset(tmp_path_factory.mktemp("intent_data"),
                                        num_screenshots=12, seed=19, samples_per_image=6)
    return manifest, load_manifest(manifest)


def test_groups_are_isolated_and_labels_balanced(generated):
    manifest, rows = generated
    group_splits = defaultdict(set)
    group_labels = defaultdict(Counter)
    group_images = defaultdict(set)
    for row in rows:
        group_splits[row["group_id"]].add(row["split"])
        group_labels[row["group_id"]][row["label"]] += 1
        group_images[row["group_id"]].add(row["image"])
    assert len(rows) == 72
    assert {row["split"] for row in rows} == {"train", "val", "test"}
    assert all(len(splits) == 1 for splits in group_splits.values())
    assert all(labels == Counter({label: 2 for label in LABELS}) for labels in group_labels.values())
    assert all(len(images) == 1 for images in group_images.values())
    assert validate_manifest(rows, manifest)["images_verified"] == 12


def test_same_screen_and_target_are_reused_for_all_intents(generated):
    _, rows = generated
    pairs = defaultdict(list)
    for row in rows:
        pairs[(row["group_id"], row["metadata"]["template_family"])].append(row)
    for values in pairs.values():
        assert {row["label"] for row in values} == set(LABELS)
        assert len({row["image"] for row in values}) == 1
        assert len({row["target_text"] for row in values}) == 1
        assert len({tuple(row["target_bbox"]) for row in values}) == 1


def test_boxes_and_paths_are_valid(generated):
    manifest, rows = generated
    for row in rows:
        assert (manifest.parent / row["image"]).is_file()
        x0, y0, x1, y1 = row["target_bbox"]
        assert 0 <= x0 < x1 <= 1
        assert 0 <= y0 < y1 <= 1
        assert row["target_text"] in row["instruction"]


def test_seed_reproduces_manifest_and_pixels(tmp_path):
    first = build_synthetic_dataset(tmp_path / "a", num_screenshots=6, seed=23, samples_per_image=3)
    second = build_synthetic_dataset(tmp_path / "b", num_screenshots=6, seed=23, samples_per_image=3)
    assert first.read_bytes() == second.read_bytes()
    for row in load_manifest(first):
        assert hashlib.sha256((first.parent / row["image"]).read_bytes()).digest() == hashlib.sha256(
            (second.parent / row["image"]).read_bytes()).digest()
    third = build_synthetic_dataset(tmp_path / "c", num_screenshots=6, seed=24, samples_per_image=3)
    assert first.read_bytes() != third.read_bytes()


def test_instruction_shortcuts_have_counterexamples(generated):
    _, rows = generated
    assert any("click" in row["instruction"].lower() and row["label"] != "click" for row in rows)
    assert any(row["instruction"].startswith("Do not merely describe") and row["label"] == "click" for row in rows)
    assert any(row["metadata"]["literal_target"] and row["label"] == "click" for row in rows)
    for row in rows:
        assert "label=" not in row["instruction"]
        assert "intent=" not in row["instruction"]
        assert "intended_action" not in row["metadata"]


def test_holdout_layout_and_instruction_family(tmp_path):
    path = build_synthetic_dataset(tmp_path, num_screenshots=12, samples_per_image=6,
                                    split_strategy="layout", holdout_family="webpage",
                                    holdout_template_family="quoted")
    rows = load_manifest(path)
    assert all((row["metadata"]["layout"] == "webpage") == (row["split"] == "test") for row in rows)
    assert all((row["metadata"]["template_family"] == "quoted") == (row["split"] == "test") for row in rows)
    assert Counter(row["label"] for row in rows if row["split"] == "test") == Counter(dict.fromkeys(LABELS, 2))


def test_rejects_group_leakage_and_bad_boxes(generated):
    manifest, source = generated
    rows = copy.deepcopy(source)
    rows[1]["split"] = "val" if rows[0]["split"] != "val" else "train"
    with pytest.raises(ValueError, match="Split leakage"):
        validate_manifest(rows, manifest)
    rows = copy.deepcopy(source)
    rows[0]["target_bbox"] = [0.5, 0.5, 0.2, 0.9]
    with pytest.raises(ValueError, match="target_bbox"):
        validate_manifest(rows, manifest)


@pytest.mark.parametrize("path", ["../outside.png", "images/../../outside.png", "C:/outside.png", "\\\\server\\share\\outside.png"])
def test_rejects_image_path_escape(generated, path):
    manifest, source = generated
    rows = copy.deepcopy(source)
    rows[0]["image"] = path
    with pytest.raises(ValueError, match="image path"):
        validate_manifest(rows, manifest)


def test_rejects_duplicate_image_across_splits(generated, tmp_path):
    manifest, source = generated
    row1, row2 = copy.deepcopy(source[0]), copy.deepcopy(source[0])
    (tmp_path / "a.png").write_bytes((manifest.parent / row1["image"]).read_bytes())
    (tmp_path / "b.png").write_bytes((manifest.parent / row1["image"]).read_bytes())
    row1.update(image="a.png", split="train")
    row2.update(image="b.png", group_id="different", sample_id="different", split="test")
    with pytest.raises(ValueError, match="identical image content"):
        validate_manifest([row1, row2], tmp_path / "manifest.jsonl")


def test_loader_reports_bad_json_line(tmp_path):
    path = tmp_path / "manifest.jsonl"
    path.write_text("{invalid\n", encoding="utf-8")
    with pytest.raises(ValueError, match="line 1"):
        load_manifest(path)


def test_summary_counts_agree(generated):
    manifest, rows = generated
    summary = json.loads((manifest.parent / "summary.json").read_text(encoding="utf-8"))
    assert summary["num_samples"] == len(rows)
    assert summary["num_screenshots"] == 12
    assert sum(split["num_samples"] for split in summary["splits"].values()) == len(rows)
    assert summary["group_leakage"] is False


@pytest.mark.parametrize("kwargs", [{"num_screenshots": 2}, {"samples_per_image": 4},
                                     {"split_strategy": "bad"}, {"holdout_family": "webpage"},
                                     {"seed": "../outside"}])
def test_invalid_generation_options_fail_before_writing(tmp_path, kwargs):
    with pytest.raises(ValueError):
        build_synthetic_dataset(tmp_path / "data", **kwargs)
