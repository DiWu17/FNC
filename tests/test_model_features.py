import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import clickagent.model_features as mf


@pytest.mark.parametrize("text,decision", [
    ("CL", None), ("CLICK", "click"), ("CLICK\nSave", "click"),
    (" DESCRIBE: location", "describe"), ("FORBID\n", "forbid"),
    ("CLICKING", None), ("I would CLICK", None), ("DESCR", None),
])
def test_generated_decision_requires_a_complete_leading_category(text, decision):
    assert mf.detect_generated_decision(text) == decision


def test_reconstruct_last_row_matches_official_eager_gqa_and_rope():
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        Qwen3VLTextAttention, apply_rotary_pos_emb, eager_attention_forward,
    )
    torch.manual_seed(7)
    config = Qwen3VLTextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=1,
                              num_attention_heads=4, num_key_value_heads=2, head_dim=8)
    module = Qwen3VLTextAttention(config, layer_idx=0).eval()
    hidden = torch.randn(1, 5, 32)
    angles = torch.randn(1, 5, 4).repeat(1, 1, 2)
    position = (angles.cos(), angles.sin())
    q = module.q_norm(module.q_proj(hidden).view(1, 5, 4, 8)).transpose(1, 2)
    k = module.k_norm(module.k_proj(hidden).view(1, 5, 2, 8)).transpose(1, 2)
    v = module.v_proj(hidden).view(1, 5, 2, 8).transpose(1, 2)
    q, k = apply_rotary_pos_emb(q, k, *position)
    mask = torch.zeros(1, 1, 5, 5)
    mask[..., 1] = -torch.inf
    with torch.no_grad():
        _, expected = eager_attention_forward(module, q, k, v, mask, module.scaling)
        actual = mf.reconstruct_attention_row(module, hidden, position, k, mask)
        boolean_actual = mf.reconstruct_attention_row(module, hidden, position, k, ~torch.isneginf(mask))
    torch.testing.assert_close(actual, expected[:, :, -1], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(actual, boolean_actual)
    assert not actual[..., 1].any()


def test_recorder_does_not_change_tiny_qwen_logits_and_removes_hooks():
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
    model = Qwen3VLForConditionalGeneration(config).eval().requires_grad_(False)
    runtime = mf._model_runtime_metadata(model)
    assert runtime["parameter_count"] > 0
    assert runtime["trainable_parameter_count"] == 0
    assert runtime["all_parameters_frozen"] is True and runtime["model_training"] is False
    ids = torch.tensor([[1, 2, 3, 4]])
    with torch.inference_mode():
        reference = model(input_ids=ids, use_cache=True, logits_to_keep=1).logits
        with mf.SelectedLayerRecorder(model, (1, 3)) as recorder:
            recorder.begin(np.array([1, 2]), (1, 2))
            output = model(input_ids=ids, use_cache=True, logits_to_keep=1)
            snapshot = recorder.snapshot()
            torch.testing.assert_close(output.logits, reference)
            assert snapshot["hidden"].shape == (64,)
            assert len(snapshot["attention"]) == len(snapshot["attention_names"]) == 144
            assert np.isfinite(snapshot["attention"]).all()
            model(input_ids=torch.tensor([[5]]), past_key_values=output.past_key_values,
                  cache_position=torch.tensor([4]), use_cache=True, logits_to_keep=1)
            assert recorder.snapshot()["hidden"].shape == (64,)
    assert not model.model.language_model.layers[0]._forward_hooks
    assert not model.model.language_model.layers[0].self_attn._forward_hooks
    assert model.config._attn_implementation == "sdpa"


@pytest.mark.parametrize("requested", [mf.CONDITIONS, ("prefill",), ("prefill", "post_decision")])
@pytest.mark.parametrize("task,word", [("intent", "CLICK"), ("visual_state", "CLICK"),
                                       ("visual_state", "DISABLED"), ("visual_state", "ABSENT"),
                                       ("visual_state", "UNKNOWN")])
def test_generated_token_is_forwarded_before_each_readout(monkeypatch, requested, task, word):
    torch = pytest.importorskip("torch")

    class FakeModel:
        def __init__(self):
            self.calls = []
            self.latest = 0

        def __call__(self, input_ids, **kwargs):
            self.calls.append(input_ids.tolist())
            self.latest = int(input_ids[0, -1])
            next_token = [10, 11, 12, 99][len(self.calls) - 1]
            logits = torch.full((1, 1, 100), -100.0)
            logits[0, 0, next_token] = 1
            return SimpleNamespace(logits=logits, past_key_values=object())

    class FakeRecorder:
        def __init__(self, model, layers):
            self.model = model

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            pass

        def begin(self, *args):
            pass

        def snapshot(self):
            return {"attention": np.array([self.model.latest], dtype=np.float32),
                    "hidden": np.array([self.model.latest], dtype=np.float32),
                    "attention_names": ["a"], "hidden_names": ["h"]}

    extractor = mf.FrozenQwenExtractor.__new__(mf.FrozenQwenExtractor)
    extractor.config = mf.ExtractionConfig("unused", layers=(1,), max_new_tokens=5, conditions=requested, task=task)
    extractor.model = FakeModel()
    extractor.device = torch.device("cpu")
    extractor.eos_ids = {99}
    pieces = {10: word[:2], 11: word[2:], 12: "\nSave"}
    extractor.processor = SimpleNamespace(tokenizer=SimpleNamespace(
        decode=lambda ids, **kwargs: "".join(pieces[token] for token in ids)))
    monkeypatch.setattr(mf, "SelectedLayerRecorder", FakeRecorder)
    monkeypatch.setattr(mf, "_sync_cuda", lambda: None)
    monkeypatch.setattr(extractor, "_prepare_inputs", lambda *args: (
        {"input_ids": torch.tensor([[3, 4]]), "attention_mask": torch.ones((1, 2), dtype=torch.long)},
        np.array([], dtype=int), (0, 0)))
    label = "forbid" if task == "intent" else "disabled" if word == "CLICK" else "click"
    sample = {"sample_id": "s", "group_id": "g", "label": label, "split": "train",
              "instruction": "Do not click Save" if task == "intent" else mf.VISUAL_STATE_INSTRUCTION,
              "image": "unused.png"}
    conditions, metadata = extractor.extract_sample(sample, Path("."))
    assert conditions["prefill"]["hidden"].tolist() == [4]
    if requested == ("prefill",):
        assert metadata["generation_enabled"] is False
        assert metadata["generated_tokens"] == 0
        assert metadata["generation_stop_reason"] == "disabled"
        assert extractor.model.calls == [[[3, 4]]]
        return
    assert conditions["post_decision"]["hidden"].tolist() == [11]
    assert conditions["post_decision"]["query_token_index"] == 3
    assert metadata["generated_decision"] == word.lower()  # Never replace a model error/abstention by the true label.
    assert metadata["decision_matches_label"] is False
    assert metadata["decision_abstained"] is (word == "UNKNOWN")
    assert metadata["decision_in_label_space"] is (word != "UNKNOWN")
    if "post_response" not in requested:
        assert metadata["generated_tokens"] == 2
        assert metadata["generation_stop_reason"] == "decision_captured"
        assert not metadata["truncated"]
        assert extractor.model.calls == [[[3, 4]], [[10]], [[11]]]
        return
    assert conditions["post_response"]["hidden"].tolist() == [12]
    assert conditions["post_response"]["query_token_index"] == 4
    assert metadata["generated_text"] == word + "\nSave"
    assert metadata["generated_tokens"] == 3
    assert metadata["terminated_by_eos"] is True
    assert extractor.model.calls == [[[3, 4]], [[10]], [[11]], [[12]]]


@pytest.mark.parametrize("local_model", [False, True])
def test_checkpoint_resume_and_pickle_free_exports(tmp_path, monkeypatch, local_model):
    samples = [{"sample_id": f"s{i}", "group_id": f"g{i}", "image": "unused.png",
                "instruction": f"instruction {i}", "label": label, "split": split}
               for i, (label, split) in enumerate(zip(("click", "describe", "forbid"), ("train", "val", "test")))]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in samples), encoding="utf-8")

    class FakeExtractor:
        loads = 0
        calls = 0

        def __init__(self, config):
            FakeExtractor.loads += 1
            self.runtime_metadata = {"parameter_count": 100, "trainable_parameter_count": 0,
                                     "all_parameters_frozen": True, "model_training": False}

        def extract_sample(self, sample, manifest_dir):
            FakeExtractor.calls += 1
            snapshot = {"attention": np.array([0.1], dtype=np.float32), "hidden": np.array([0.2, 0.3], dtype=np.float32),
                        "attention_names": ["a"], "hidden_names": ["h0", "h1"],
                        "query_token_index": 1, "generated_tokens_at_readout": 0, "readout_elapsed_seconds": 1.0}
            metadata = {**sample, "generated_decision": None, "decision_matches_label": False}
            return {"prefill": snapshot}, metadata

    monkeypatch.setattr(mf, "FrozenQwenExtractor", FakeExtractor)
    model_path = "unused"
    if local_model:
        model_directory = tmp_path / "local_model"
        model_directory.mkdir()
        (model_directory / "config.json").write_text('{"variant":1}', encoding="utf-8")
        model_path = str(model_directory)
    config = mf.ExtractionConfig(model_path, layers=(1,))
    summary = mf.extract_manifest(manifest, tmp_path / "features", config)
    assert summary["condition_counts"] == {"prefill": 3, "post_decision": 0, "post_response": 0}
    assert FakeExtractor.loads == 1 and FakeExtractor.calls == 3
    mf.extract_manifest(manifest, tmp_path / "features", config)
    assert FakeExtractor.loads == 1 and FakeExtractor.calls == 3
    run = json.loads((tmp_path / "features" / "run.json").read_text(encoding="utf-8"))
    schema = json.loads((tmp_path / "features" / "schema.json").read_text(encoding="utf-8"))
    assert run["model_runtime"]["all_parameters_frozen"] is True
    assert schema["model_runtime"]["parameter_count"] == 100
    assert "torch" in schema["library_versions"] and "transformers" in run["library_versions"]
    assert run["model_identity"]["exists"] is local_model
    with np.load(tmp_path / "features" / "prefill.npz", allow_pickle=False) as stored:
        assert stored["attention"].shape == (3, 1)
        assert stored["hidden"].shape == (3, 2)
        assert stored["labels"].tolist() == ["click", "describe", "forbid"]
        assert stored["sample_ids"].dtype.kind == "U"
    changed = mf.ExtractionConfig("different", layers=(1,))
    with pytest.raises(ValueError, match="different manifest/config"):
        mf.extract_manifest(manifest, tmp_path / "features", changed)
    if local_model:
        (model_directory / "config.json").write_text('{"variant":2}', encoding="utf-8")
        with pytest.raises(ValueError, match="different manifest/config"):
            mf.extract_manifest(manifest, tmp_path / "features", config)
        assert FakeExtractor.loads == 1


def test_model_identity_tracks_small_file_hashes_and_weight_stats(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "config.json").write_text('{"model_type":"qwen3_vl"}', encoding="utf-8")
    (model_dir / "tokenizer.json").write_text('{"version":"a"}', encoding="utf-8")
    (model_dir / "preprocessor_config.json").write_text('{"size":512}', encoding="utf-8")
    (model_dir / "chat_template.jinja").write_text("{{ messages }}", encoding="utf-8")
    (model_dir / "model.safetensors.index.json").write_text(
        '{"weight_map":{"a":"model-01.safetensors"}}', encoding="utf-8")
    weight = model_dir / "model-01.safetensors"
    weight.write_bytes(b"fake-weight")
    initial = mf._local_model_identity(str(model_dir))
    assert initial["exists"] is True
    assert set(initial["small_files"]) == {"config.json", "tokenizer.json", "preprocessor_config.json",
                                           "chat_template.jinja", "model.safetensors.index.json"}
    assert initial["weight_shards"][weight.name]["size"] == len(b"fake-weight")
    assert "sha256" not in initial["weight_shards"][weight.name]
    assert "full weight content is not hashed" in initial["validation_level"]
    (model_dir / "tokenizer.json").write_text('{"version":"b"}', encoding="utf-8")
    changed_tokenizer = mf._local_model_identity(str(model_dir))
    assert changed_tokenizer["small_files"]["tokenizer.json"] != initial["small_files"]["tokenizer.json"]
    info = weight.stat()
    os.utime(weight, ns=(info.st_atime_ns, info.st_mtime_ns + 1_000_000_000))
    assert mf._local_model_identity(str(model_dir))["weight_shards"] != initial["weight_shards"]


def test_manifest_rejects_cross_split_groups(tmp_path):
    samples = [{"sample_id": f"s{i}", "group_id": "same", "image": "a.png", "instruction": "click",
                "label": "click", "split": split} for i, split in enumerate(("train", "test"))]
    manifest = tmp_path / "m.jsonl"
    manifest.write_text("\n".join(json.dumps(row) for row in samples), encoding="utf-8")
    with pytest.raises(ValueError, match="leaks across"):
        mf._read_manifest(manifest, None)
