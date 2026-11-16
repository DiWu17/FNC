import json

import numpy as np
import pytest

from clickagent.action_grounding import AttentionPointGrounder, fit_grounder, patch_features


def record(index, split="train", target=0):
    raw = np.full((2, 4), 0.02, dtype=np.float32)
    raw[0, target] = 0.4
    raw[1] = 0.05
    row, col = divmod(target, 2)
    return {"sample_id": f"{split}_{index}", "group_id": f"{split}_group_{index}", "split": split,
            "label": "click", "raw_attention": raw, "grid_shape": [2, 2], "image_size": [100, 100],
            "target_bbox": [col / 2, row / 2, (col + 1) / 2, (row + 1) / 2]}


def corpus():
    return [record(index, "train", index % 4) for index in range(8)] + [record(index, "val", index) for index in range(4)]


def test_fixed_patch_transform_uses_no_coordinates_or_boxes():
    raw = np.array([[0.2, 0.1, 0, 0], [0, 0, 0, 0]], dtype=np.float32)
    expected = np.log1p(np.array([[4 * 2 / 3, 0], [4 / 3, 0], [0, 0], [0, 0]]))
    np.testing.assert_allclose(patch_features(raw, (2, 2)), expected)
    np.testing.assert_allclose(patch_features(raw / 100, (2, 2)), expected, atol=1e-7)


def test_fits_train_only_selects_val_and_serializes(tmp_path):
    rows = corpus()
    grounder = fit_grounder(rows, tmp_path / "grounder", c_values=(1, 0.01, 0.1))
    assert grounder.selected_c == 0.01
    expected_mean = np.concatenate([patch_features(row["raw_attention"], row["grid_shape"]) for row in rows[:8]]).mean(axis=0)
    np.testing.assert_allclose(grounder.pipeline.named_steps["scale"].mean_, expected_mean)
    for index in range(4):
        value = record(99, "val", index)
        point = grounder.predict_point(value["raw_attention"], value["grid_shape"], value["image_size"])
        assert point["point"] == [25 + 50 * (index % 2), 25 + 50 * (index // 2)]
        assert point["patch_index"] == index and 0 <= point["score"] <= 1
    restored = AttentionPointGrounder.load(tmp_path / "grounder" / "grounder.joblib")
    assert restored.predict_point(rows[0]["raw_attention"], (2, 2), (100, 100)) == grounder.predict_point(rows[0]["raw_attention"], (2, 2), (100, 100))
    audit = json.loads((tmp_path / "grounder" / "audit.json").read_text(encoding="utf-8"))
    assert audit["test_used_for_fit_or_selection"] is False
    assert audit["absolute_coordinate_features"] is False
    assert audit["n_train_records_used"] == 8 and audit["n_val_records"] == 4
    assert all(item["val_point_hit_rate"] == 1 for item in audit["validation_search"])
    assert audit["record_label_counts"] == {"click": 12}


def test_quantization_misses_are_explicit(tmp_path):
    rows = corpus()
    rows[0]["target_bbox"] = [0.01, 0.01, 0.02, 0.02]
    rows[-1]["target_bbox"] = [0.01, 0.01, 0.02, 0.02]
    fit_grounder(rows, tmp_path / "grounder", c_values=(0.1,))
    audit = json.loads((tmp_path / "grounder" / "audit.json").read_text(encoding="utf-8"))
    assert audit["skipped_train_no_positive_patch"] == ["train_0"]
    assert audit["val_quantization_oracle_hit_rate"] == 0.75


def test_test_records_and_cross_split_groups_are_rejected(tmp_path):
    rows = corpus()
    rows.append(record(10, "test"))
    with pytest.raises(ValueError, match="train/val only"):
        fit_grounder(rows, tmp_path / "never")
    rows = corpus()
    rows[-1]["group_id"] = rows[0]["group_id"]
    with pytest.raises(ValueError, match="leaks across"):
        fit_grounder(rows, tmp_path / "never")


def test_raw_npz_records_and_head_validation(tmp_path):
    rows = corpus()
    path = tmp_path / "raw.npz"
    np.savez_compressed(path, raw_attention=rows[0]["raw_attention"])
    rows[0]["raw_attention"] = path
    grounder = fit_grounder(rows, tmp_path / "grounder", c_values=(0.1,))
    with pytest.raises(ValueError, match="expects 2 heads"):
        grounder.predict_point(np.zeros((3, 4)), (2, 2), (100, 100))
    with pytest.raises(ValueError, match="softmax probabilities"):
        patch_features(np.ones((2, 4)), (2, 2))

