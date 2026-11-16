import json
from pathlib import Path

import numpy as np
import pytest

import clickagent.constant_no_image as control
from clickagent.model_features import ExtractionConfig


@pytest.fixture
def manifest(tmp_path):
    rows = []
    for group, split in enumerate(("train", "val", "test")):
        for label in ("click", "disabled", "absent"):
            rows.append({"sample_id": f"sample_{group}_{label}", "group_id": f"group_{group}",
                         "image": f"NONEXISTENT_IMAGE_{group}_{label}.png", "instruction": "点击保存",
                         "label": label, "split": split, "task": "visual_state", "target_text": "保存",
                         "target_bbox": None if label == "absent" else [0.2, 0.2, 0.6, 0.5],
                         "source": f"PRIVATE_SOURCE_{group}_{label}", "template_id": "fixed",
                         "metadata": {"task": "visual_state", "visual_state": label}})
    path = tmp_path / "manifest.jsonl"
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    return path, rows


@pytest.fixture
def fake_extractor(monkeypatch):
    class FakeExtractor:
        initializations = 0
        forwards = 0
        input_records = []
        mode = "constant"

        def __init__(self, config):
            FakeExtractor.initializations += 1
            assert config.no_image is True and config.conditions == ("prefill",)
            self.runtime_metadata = {"all_parameters_frozen": True, "parameter_count": 100,
                                     "trainable_parameter_count": 0, "model_training": False}
            if config.offload_folder:
                directory = Path(config.offload_folder)
                directory.mkdir(parents=True, exist_ok=True)
                (directory / "offload_placeholder.dat").write_bytes(b"offload")

        def _prepare_inputs(self, sample, manifest_dir):
            # Only instruction text enters this fake numerical model input. The
            # deliberate mismatch mode demonstrates the equality check works.
            ids = [len(sample["instruction"]), sum(map(ord, sample["instruction"]))]
            if FakeExtractor.mode == "different_input" and sample["label"] == "disabled":
                ids[0] += 1
            inputs = {"input_ids": np.asarray([ids], dtype=np.int64),
                      "attention_mask": np.ones((1, 2), dtype=np.int64)}
            FakeExtractor.input_records.append({key: value.copy() for key, value in inputs.items()})
            return inputs, np.asarray([], dtype=np.int64), (0, 0)

        def extract_sample(self, sample, manifest_dir):
            self._prepare_inputs(sample, manifest_dir)
            FakeExtractor.forwards += 1
            hidden = np.array([1.25, -2.5], dtype=np.float32)
            if FakeExtractor.mode == "different_features" and FakeExtractor.forwards == 2:
                hidden[0] += 1
            if FakeExtractor.mode == "different_zero_bits":
                hidden[0] = -0.0 if FakeExtractor.forwards == 2 else 0.0
            elapsed = 0.5 + FakeExtractor.forwards / 10
            snapshot = {"attention": np.zeros(2, dtype=np.float32), "hidden": hidden,
                        "attention_names": ["a0", "a1"], "hidden_names": ["h0", "h1"],
                        "query_token_index": 1, "generated_tokens_at_readout": 0,
                        "readout_elapsed_seconds": elapsed}
            metadata = {**sample, "generation_enabled": False, "generated_tokens": 0,
                        "input_tokens": 2, "image_tokens": 0, "post_merge_grid": [0, 0],
                        "timing_seconds": {"prepare": 0.1, "prefill": elapsed - 0.1,
                                           "generation": 0.0, "total": elapsed, "decision_elapsed": None}}
            return {"prefill": snapshot}, metadata

    monkeypatch.setattr(control, "FrozenQwenExtractor", FakeExtractor)
    return FakeExtractor


def config(**kwargs):
    fields = {"model_path": "unused", "layers": (1,), "task": "visual_state",
              "no_image": True, "conditions": ("prefill",)}
    fields.update(kwargs)
    return ExtractionConfig(**fields)


def test_real_verification_values_are_reused_with_honest_timing_and_labels(manifest, tmp_path, fake_extractor):
    path, original = manifest
    output = tmp_path / "control"
    summary = control.export_constant_no_image_control(path, output, config())
    assert fake_extractor.initializations == 1 and fake_extractor.forwards == 3
    assert summary["actual_forward_count"] == summary["forward_count_this_call"] == 3
    assert summary["feature_reused_rows"] == 6 and summary["unique_input_count"] == 1
    assert summary["constant_input_control"]["actual_forward_count"] == 3
    assert summary["bit_exact_feature_check_passed"] and summary["bit_exact_input_check_passed"]
    assert summary["verification_labels"] == ["click", "disabled", "absent"]
    with np.load(output / "prefill.npz", allow_pickle=False) as stored:
        assert stored["hidden"].shape == (9, 2)
        assert stored["attention"].dtype == stored["hidden"].dtype == np.float32
        np.testing.assert_array_equal(stored["hidden"], np.tile([1.25, -2.5], (9, 1)))
        assert stored["sample_ids"].tolist() == [row["sample_id"] for row in original]
        assert stored["labels"].tolist() == [row["label"] for row in original]
        assert stored["groups"].tolist() == [row["group_id"] for row in original]
        assert stored["splits"].tolist() == [row["split"] for row in original]
        assert stored["labels"].dtype.kind == "U"
    rows = [json.loads(line) for line in (output / "prefill.metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    actual = [row for row in rows if row["from_verification_forward"]]
    reused = [row for row in rows if row["feature_reused"]]
    assert len(actual) == 3 and len(reused) == 6
    assert all(row["readout_elapsed_seconds"] > 0 and row["timing_seconds"]["total"] > 0 for row in actual)
    assert all(row["readout_elapsed_seconds"] is None and all(value is None for value in row["timing_seconds"].values())
               for row in reused)
    assert all(row["generation_enabled"] is False and row["generated_tokens"] == 0 for row in rows)
    assert all(row["label"] == source["label"] and row["target_bbox"] == source["target_bbox"]
               and row["metadata"] == source["metadata"] for row, source in zip(rows, original))
    assert all(row["feature_source_sample_id"] == summary["verification_sample_ids"][0] for row in reused)
    inputs = fake_extractor.input_records
    assert len(inputs) == 6
    assert all(set(item) == {"input_ids", "attention_mask"} for item in inputs)
    assert all(np.array_equal(inputs[0]["input_ids"], item["input_ids"]) for item in inputs)
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    schema = json.loads((output / "schema.json").read_text(encoding="utf-8"))
    assert run["model_runtime"]["all_parameters_frozen"] is True
    assert schema["conditions"]["prefill"]["samples"] == 9
    assert schema["constant_input_control"]["actual_forward_count"] == 3
    assert "not N independent" in schema["latency_note"]
    assert run["input_definition"]["model_input_annotation_fields"] == []
    assert len(run["constant_input_control"]["input_tensor_identity"]["sha256"]) == 64
    assert "PRIVATE_SOURCE" not in json.dumps(run["input_definition"])


def test_complete_identical_export_resumes_without_loading_a_model(manifest, tmp_path, fake_extractor):
    path, _ = manifest
    output = tmp_path / "control"
    first = control.export_constant_no_image_control(path, output, config())
    second = control.export_constant_no_image_control(path, output, config())
    assert first["fingerprint"] == second["fingerprint"]
    assert second["reused_existing_export"] is True and second["forward_count_this_call"] == 0
    assert second["actual_forward_count"] == 3 and fake_extractor.forwards == 3
    assert fake_extractor.initializations == 1
    with pytest.raises(ValueError, match="different input/config"):
        control.export_constant_no_image_control(path, output, config(seed=123))
    with (output / "prefill.npz").open("ab") as handle:
        handle.write(b"changed")
    with pytest.raises(ValueError, match="artifact is missing or changed"):
        control.export_constant_no_image_control(path, output, config())
    assert fake_extractor.initializations == 1


@pytest.mark.parametrize("mode,message", [("different_features", "bit-exact"),
                                         ("different_input", "inputs differ"),
                                         ("different_zero_bits", "bit-exact")])
def test_mismatching_actual_verifications_never_export_features(manifest, tmp_path, fake_extractor, mode, message):
    fake_extractor.mode = mode
    output = tmp_path / "control"
    with pytest.raises(ValueError, match=message):
        control.export_constant_no_image_control(manifest[0], output, config())
    assert not (output / "prefill.npz").exists()
    assert not (output / "run.json").exists()


def test_configured_offload_folder_created_by_loader_is_allowed(manifest, tmp_path, fake_extractor):
    output = tmp_path / "control"
    summary = control.export_constant_no_image_control(manifest[0], output,
                                                       config(offload_folder=str(output / "offload")))
    assert summary["actual_forward_count"] == 3
    assert (output / "offload" / "offload_placeholder.dat").is_file()
    assert (output / "prefill.npz").is_file()


@pytest.mark.parametrize("overrides", [{"task": "intent"}, {"no_image": False},
                                       {"conditions": ("prefill", "post_decision")}])
def test_rejects_inapplicable_extraction_configs(manifest, tmp_path, fake_extractor, overrides):
    with pytest.raises(ValueError, match="requires task"):
        control.export_constant_no_image_control(manifest[0], tmp_path / "control", config(**overrides))
    assert fake_extractor.initializations == 0


@pytest.mark.parametrize("count", [True, 2, 10])
def test_requires_enough_distinct_verification_rows(manifest, tmp_path, fake_extractor, count):
    with pytest.raises(ValueError, match="verification_forwards"):
        control.export_constant_no_image_control(manifest[0], tmp_path / "control", config(), count)
    assert fake_extractor.initializations == 0


def test_nonempty_foreign_output_is_never_overwritten(manifest, tmp_path, fake_extractor):
    output = tmp_path / "main_features"
    output.mkdir()
    (output / "unrelated.txt").write_text("main run", encoding="utf-8")
    with pytest.raises(ValueError, match="nonempty"):
        control.export_constant_no_image_control(manifest[0], output, config())
    assert (output / "unrelated.txt").read_text(encoding="utf-8") == "main run"
    assert fake_extractor.initializations == 0


def test_manifest_validation_uses_no_images_and_rejects_image_split_leakage(manifest, tmp_path, fake_extractor):
    path, rows = manifest
    rows[3]["image"] = rows[0]["image"]
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8")
    with pytest.raises(ValueError, match="Split leakage"):
        control.export_constant_no_image_control(path, tmp_path / "control", config())
    assert fake_extractor.initializations == 0


def test_more_verifications_remain_real_and_distinct(manifest, tmp_path, fake_extractor):
    summary = control.export_constant_no_image_control(manifest[0], tmp_path / "control", config(), 4)
    assert fake_extractor.forwards == 4
    assert len(set(summary["verification_sample_ids"])) == 4
    assert summary["actual_forward_count"] == 4 and summary["feature_reused_rows"] == 5
