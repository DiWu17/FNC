"""Task protocol, absence of annotation inputs, and a CPU-only Qwen check."""

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import clickagent.model_features as mf


@pytest.mark.parametrize("text,expected", [
    ("CLICK", "click"), ("DISABLED\nGray control", "disabled"),
    ("ABSENT: no target", "absent"), ("UNKNOWN\nNo screenshot", "unknown"),
    ("DISABL", None), ("ABSENTLY", None), ("UNKNOWN_STATE", None),
    ("DESCRIBE", None), ("FORBID", None), ("I see DISABLED", None),
])
def test_visual_state_decision_parser(text, expected):
    assert mf.detect_generated_decision(text, task="visual_state") == expected


def test_task_validation_and_old_intent_default():
    assert mf.ExtractionConfig("unused").task == "intent"
    assert mf.system_prompt_for_task() == mf.SYSTEM_PROMPT
    assert "保存" in mf.system_prompt_for_task("visual_state")
    assert "UNKNOWN" in mf.system_prompt_for_task("visual_state")
    assert "exact visible label 保存" in mf.system_prompt_for_task("visual_state")
    assert "另存为, Save as" in mf.system_prompt_for_task("visual_state")
    assert mf.detect_generated_decision("UNKNOWN") is None
    with pytest.raises(ValueError, match="unknown task"):
        mf.ExtractionConfig("unused", task="typo")


def visual_samples():
    return [{"sample_id": f"sample_{label}", "group_id": "same_screen",
             "image": f"PRIVATE_PATH_{label}.png", "instruction": "点击保存", "label": label,
             "split": "train", "target_text": f"ANNOTATION_SECRET_{label}",
             "target_bbox": [0.1 * index, 0.2, 0.6, 0.8], "source": f"SOURCE_SECRET_{label}",
             "template_id": f"TEMPLATE_SECRET_{label}"}
            for index, label in enumerate(mf.TASK_LABELS["visual_state"])]


def test_visual_manifest_requires_fixed_instruction_and_task_labels(tmp_path):
    manifest = tmp_path / "visual.jsonl"
    samples = visual_samples()
    manifest.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in samples), encoding="utf-8")
    assert [row["label"] for row in mf._read_manifest(manifest, None, task="visual_state")] == ["click", "disabled", "absent"]
    with pytest.raises(ValueError, match="unknown label"):
        mf._read_manifest(manifest, None)  # The old intent protocol is not silently reused.
    samples[1]["instruction"] = "点击保存，因为它处于禁用状态"
    manifest.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in samples), encoding="utf-8")
    with pytest.raises(ValueError, match="identical instruction"):
        mf._read_manifest(manifest, None, task="visual_state")


class RecordingProcessor:
    """Deterministic miniature tokenizer; use real Qwen layers below, no downloads."""

    def __init__(self, torch):
        self.torch = torch
        self.messages = []
        self.processor_kwargs = []

    def apply_chat_template(self, messages, **kwargs):
        self.messages.append(copy.deepcopy(messages))
        return json.dumps(messages, ensure_ascii=False, sort_keys=True)

    def __call__(self, **kwargs):
        self.processor_kwargs.append(kwargs)
        torch = self.torch

        class Batch(dict):
            def to(self, device):
                return Batch({key: value.to(device) for key, value in self.items()})

        ids = torch.tensor([[int(value) % 90 + 1 for value in kwargs["text"][0].encode("utf-8")[:64]]])
        return Batch(input_ids=ids, attention_mask=torch.ones_like(ids))


def test_no_image_prompt_excludes_labels_boxes_filenames_and_metadata():
    torch = pytest.importorskip("torch")
    extractor = mf.FrozenQwenExtractor.__new__(mf.FrozenQwenExtractor)
    extractor.config = mf.ExtractionConfig("unused", task="visual_state", no_image=True)
    extractor.model = SimpleNamespace(config=SimpleNamespace(image_token_id=95))
    extractor.processor = RecordingProcessor(torch)
    extractor.device = torch.device("cpu")
    prepared = []
    for sample in visual_samples():
        inputs, positions, grid = extractor._prepare_inputs(sample, Path("."))
        prepared.append(inputs["input_ids"])
        assert positions.size == 0 and grid == (0, 0)
    assert all(torch.equal(prepared[0], item) for item in prepared[1:])
    messages = extractor.processor.messages
    assert messages[0] == messages[1] == messages[2]
    assert messages[0] == [{"role": "system", "content": mf.VISUAL_STATE_SYSTEM_PROMPT},
                           {"role": "user", "content": [{"type": "text", "text": "点击保存"}]}]
    assert all("images" not in kwargs for kwargs in extractor.processor.processor_kwargs)
    serialized = json.dumps(messages, ensure_ascii=False)
    for forbidden in ("PRIVATE_PATH", "ANNOTATION_SECRET", "SOURCE_SECRET", "TEMPLATE_SECRET", "target_bbox"):
        assert forbidden not in serialized


def test_actual_tiny_qwen_has_identical_no_image_features_for_all_visual_states(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    config = Qwen3VLConfig(
        text_config={"vocab_size": 100, "hidden_size": 32, "intermediate_size": 64,
                     "num_hidden_layers": 3, "num_attention_heads": 4, "num_key_value_heads": 2,
                     "head_dim": 8, "rope_scaling": {"rope_type": "default", "mrope_section": [1, 1, 2]}},
        vision_config={"hidden_size": 32, "out_hidden_size": 32, "intermediate_size": 64,
                       "depth": 3, "num_heads": 4, "deepstack_visual_indexes": [0, 1, 2]},
        image_token_id=95, video_token_id=96, vision_start_token_id=97, vision_end_token_id=98,
    )
    config._attn_implementation = "sdpa"
    extractor = mf.FrozenQwenExtractor.__new__(mf.FrozenQwenExtractor)
    extractor.config = mf.ExtractionConfig("unused", layers=(1, 3), task="visual_state", no_image=True,
                                           conditions=("prefill",))
    extractor.model = Qwen3VLForConditionalGeneration(config).cpu().eval().requires_grad_(False)
    extractor.processor = RecordingProcessor(torch)
    extractor.device = torch.device("cpu")
    extractor.eos_ids = {99}
    monkeypatch.setattr(mf, "_sync_cuda", lambda: None)
    results = [extractor.extract_sample(sample, Path(".")) for sample in visual_samples()]
    first = results[0][0]["prefill"]
    assert first["hidden"].any()
    assert not first["attention"].any()
    for conditions, metadata in results:
        np.testing.assert_array_equal(conditions["prefill"]["hidden"], first["hidden"])
        np.testing.assert_array_equal(conditions["prefill"]["attention"], first["attention"])
        assert metadata["generated_token_ids"] == []
        assert metadata["generation_enabled"] is False
        assert metadata["task"] == "visual_state"
        assert metadata["label_order"] == ["click", "disabled", "absent"]


def test_visual_task_provenance_and_fingerprint_reject_task_switch(tmp_path, monkeypatch):
    manifest = tmp_path / "single.jsonl"
    # CLICK is intentionally shared by both label spaces, so only the task and
    # its prompt distinguish these otherwise identical extraction requests.
    manifest.write_text(json.dumps(visual_samples()[0], ensure_ascii=False), encoding="utf-8")

    class FakeExtractor:
        def __init__(self, config):
            self.runtime_metadata = {"all_parameters_frozen": True}

        def extract_sample(self, sample, manifest_dir):
            return {}, {**sample, "generated_decision": None, "decision_matches_label": False}

    monkeypatch.setattr(mf, "FrozenQwenExtractor", FakeExtractor)
    output = tmp_path / "features"
    summary = mf.extract_manifest(manifest, output, mf.ExtractionConfig("unused", task="visual_state"))
    run = json.loads((output / "run.json").read_text(encoding="utf-8"))
    schema = json.loads((output / "schema.json").read_text(encoding="utf-8"))
    for record in (summary, run, schema):
        assert record["task"] == "visual_state" and record["label_order"] == ["click", "disabled", "absent"]
    assert run["system_prompt"] == mf.VISUAL_STATE_SYSTEM_PROMPT
    assert summary["decision_label_coverage"] == 0 and summary["decision_abstentions"] == 0
    assert "UNKNOWN abstentions" in schema["post_decision_semantics"]
    with pytest.raises(ValueError, match="different manifest/config"):
        mf.extract_manifest(manifest, output, mf.ExtractionConfig("unused", task="intent"))
