import json

import numpy as np
import pytest

from clickagent.permutation_control import STATE_LABELS, make_permuted_labels, prepare_permutation_control


def _records():
    return [{"sample_id": f"g{g}_{label}", "group_id": f"g{g}", "label": label,
             "split": ("train", "val", "test")[g % 3], "instruction": "点击保存"}
            for g in range(60) for label in STATE_LABELS]


def test_uniform_groupwise_permutation_is_balanced_and_reproducible():
    records = _records()
    mapping = make_permuted_labels(records)
    assert mapping == make_permuted_labels(list(reversed(records)))
    permutations = set()
    for group in {row["group_id"] for row in records}:
        rows = sorted((row for row in records if row["group_id"] == group), key=lambda r: r["sample_id"])
        labels = tuple(mapping[row["sample_id"]] for row in rows)
        assert set(labels) == set(STATE_LABELS)
        permutations.add(labels)
    assert len(permutations) == 6
    unchanged = sum(mapping[row["sample_id"]] == row["label"] for row in records)
    assert 30 < unchanged < 90


def test_incomplete_or_leaky_groups_rejected():
    with pytest.raises(ValueError, match="complete"):
        make_permuted_labels(_records()[1:])
    records = _records()
    records[0]["split"] = "test"
    with pytest.raises(ValueError, match="leakage"):
        make_permuted_labels(records)


def test_control_copies_only_features_and_changes_no_original(tmp_path):
    records = _records()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in records), encoding="utf-8")
    features = tmp_path / "features"
    features.mkdir()
    ids = np.asarray([row["sample_id"] for row in records])
    original_labels = np.asarray([row["label"] for row in records])
    values = np.arange(len(records) * 2).reshape(-1, 2).astype(np.float32)
    np.savez(features / "prefill.npz", sample_ids=ids, labels=original_labels, hidden=values)
    (features / "metadata.jsonl").write_text('never copy generated labels', encoding="utf-8")
    control_manifest, target = prepare_permutation_control(manifest, features, tmp_path / "control")
    with np.load(target / "prefill.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["hidden"], values)
        assert np.any(data["labels"] != original_labels)
    with np.load(features / "prefill.npz", allow_pickle=False) as data:
        np.testing.assert_array_equal(data["labels"], original_labels)
    assert not (target / "metadata.jsonl").exists()
    assert control_manifest.is_file()
    with pytest.raises(ValueError, match="separate"):
        prepare_permutation_control(manifest, features, features.parent)
    np.savez(target / "post_response.npz", attention=np.zeros((1, 1)))
    with pytest.raises(ValueError, match="conditions differ"):
        prepare_permutation_control(manifest, features, tmp_path / "control")
