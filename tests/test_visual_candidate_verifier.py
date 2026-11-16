"""Small CPU-only tests; never load the real 0.58B visual checkpoint or a GPU."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

import clickagent.visual_candidate_verifier as visual


@pytest.fixture(autouse=True)
def one_cpu_thread():
    prior = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(prior)


def encoding():
    # Features encode normalized token-center xy, permitting hand-checkable
    # bilinear coordinate tests rather than reproducing the implementation.
    yy, xx = np.meshgrid((np.arange(12) + 0.5) / 12, (np.arange(16) + 0.5) / 16, indexing="ij")
    features = np.stack((xx, yy, xx + yy, xx - yy), axis=-1).astype(np.float32)
    merged = np.concatenate((features[::2, ::2], features[::2, ::2, :2]), axis=-1)
    return {"features": features, "merged_features": merged,
            "metadata": {"schema_version": 1, "image_size": [1600, 1200],
                         "premerge_grid_shape": [12, 16], "merged_grid_shape": [6, 8]}}


def inputs(points=((550, 750), (1000, 450), (0, 0))):
    target = np.random.default_rng(8).normal(size=(3, 12)).astype(np.float32)
    return visual.build_visual_inputs(encoding(), points, target)


def head():
    torch.manual_seed(7)
    return visual.VisualCandidateVerifier(visual.VisualVerifierConfig(vision_dim=4, semantic_dim=6, target_dim=12, width=16, heads=4, dropout=0))


def test_premerge_block_order_is_restored_to_true_raster():
    raster = np.arange(4 * 6 * 3).reshape(4, 6, 3)
    block = raster.reshape(2, 2, 3, 2, 3).transpose(0, 2, 1, 3, 4).reshape(24, 3)
    np.testing.assert_array_equal(visual.restore_premerge_raster(block, (4, 6)), raster)
    torch.testing.assert_close(visual.restore_premerge_raster(torch.as_tensor(block), (4, 6)), torch.as_tensor(raster))


def test_exact_original_point_survives_resize_and_roi_center_sampling():
    value = inputs()
    # The central 5x5 token corresponds to (550/1600,750/1200), not to the
    # nearest patch center or a coordinate in the processed/resized image.
    np.testing.assert_allclose(value["fine_roi"][0, 12, :2], [550 / 1600, 750 / 1200], atol=1e-7)
    np.testing.assert_allclose(value["fine_positions"][0, 12], [0, 0, 550 / 1600, 750 / 1200], atol=1e-7)
    np.testing.assert_allclose(value["candidate_geometry"][0, :2], [550 / 1600, 750 / 1200], atol=1e-7)
    assert value["fine_roi"].shape == (3, 25, 4)
    assert value["context_roi"].shape == (3, 49, 4)
    assert value["merged_roi"].shape == (3, 9, 6)


def test_boundary_padding_has_mask_and_candidate_is_never_clamped():
    value = inputs(((0, 0),))
    assert value["fine_mask"].sum() == 9
    assert value["context_mask"].sum() == 16
    assert value["merged_mask"].sum() == 4
    assert not value["fine_roi"][~value["fine_mask"]].any()
    np.testing.assert_array_equal(value["candidate_points"], [[0, 0]])
    np.testing.assert_array_equal(value["candidate_lengths"], [29])
    # Point location in the clipped visible ROI is top-left, not falsely .5.
    np.testing.assert_array_equal(value["candidate_geometry"][0, 8:12], [0, 0, 0, 0])
    with pytest.raises(ValueError, match="inside"):
        inputs(((1600, 50),))


def test_candidate_subset_preserves_whole_target_sequence_and_dtypes():
    value = inputs()
    subset = visual.subset_visual_inputs(value, [2, 0])
    np.testing.assert_array_equal(subset["target_hidden"], value["target_hidden"])
    assert subset["target_hidden"].shape == (3, 12)
    assert subset["fine_roi"].shape[0] == 2
    tensor = visual.visual_inputs_to_torch(subset)
    assert tensor["candidate_lengths"].dtype == torch.long
    assert tensor["fine_mask"].dtype == torch.bool
    assert tensor["merged_roi"].dtype == torch.float32


def test_whole_pool_comparison_is_permutation_equivariant_and_null_invariant():
    model = head().eval()
    value = inputs()
    order = [2, 0, 1]
    with torch.no_grad():
        original = model(**visual.visual_inputs_to_torch(value))
        permuted = model(**visual.visual_inputs_to_torch(visual.subset_visual_inputs(value, order)))
    torch.testing.assert_close(permuted["scores"], original["scores"][order], atol=2e-6, rtol=2e-6)
    torch.testing.assert_close(permuted["null_scores"], original["null_scores"], atol=2e-6, rtol=2e-6)
    assert original["target_attention_weights"].shape == (3, 3)
    assert original["roi_attention_weights"].shape == (3, 83)
    assert original["candidate_lengths"].shape == (3,)


def test_training_updates_head_only_and_save_does_not_move_or_eval_model(tmp_path, monkeypatch):
    model = head().train()
    batch = visual.visual_inputs_to_torch(inputs())
    batch["target_hidden"].requires_grad_()
    batch["fine_roi"].requires_grad_()
    out = model(**batch)
    (out["scores"].sum() + out["null_scores"].sum()).backward()
    assert batch["target_hidden"].grad is None and batch["fine_roi"].grad is None
    assert model.detail[1].weight.grad is not None
    assert model.target_projection.weight.grad is not None
    # Fails immediately if helper tries model.cpu() rather than cloned state.
    monkeypatch.setattr(model, "cpu", lambda: (_ for _ in ()).throw(AssertionError("model moved")))
    visual.save_visual_verifier(tmp_path / "head.pt", model, {"split": "train-only"})
    assert model.training
    restored, metadata = visual.load_visual_verifier(tmp_path / "head.pt", "cpu")
    assert metadata == {"split": "train-only"}
    for key, value in model.state_dict().items():
        torch.testing.assert_close(restored.state_dict()[key], value)


def test_empty_pool_keeps_real_query_and_returns_null_only():
    batch = inputs(())
    out = head().eval()(**visual.visual_inputs_to_torch(batch))
    assert out["scores"].shape == (0,)
    assert out["null_scores"].shape == (1,) and torch.isfinite(out["null_scores"]).all()
    with pytest.raises(ValueError, match="nonempty"):
        visual.build_visual_inputs(encoding(), [[10, 10]], np.empty((0, 12)))


def test_safe_encoding_cache_contains_both_frozen_visual_branches(tmp_path):
    visual.save_visual_encoding(tmp_path / "encoding.npz", encoding())
    restored = visual.load_visual_encoding(tmp_path / "encoding.npz")
    assert restored["features"].dtype == np.float16
    assert restored["merged_features"].shape == (6, 8, 6)
    with np.load(tmp_path / "encoding.npz", allow_pickle=False) as stored:
        assert stored["metadata"].dtype.kind == "U"


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16])
def test_standalone_tiny_vision_loads_only_visual_tensors_and_encodes_once(tmp_path, monkeypatch, dtype):
    from safetensors.torch import save_file
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLVisionConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel
    # head_dim=12 produces nontrivial FP32 rotary values, unlike head_dim=4
    # whose sole frequency is exactly 1 and cannot expose the casting bug.
    config = Qwen3VLVisionConfig(depth=1, hidden_size=24, intermediate_size=32, num_heads=2,
                               out_hidden_size=16, patch_size=2, temporal_patch_size=1,
                               spatial_merge_size=2, num_position_embeddings=16, deepstack_visual_indexes=[])
    config._attn_implementation = "sdpa"
    reference = Qwen3VLVisionModel(config).eval()
    original_rotary = reference.rotary_pos_emb.inv_freq.detach().clone()
    assert not torch.equal(original_rotary, original_rotary.half().float())
    state = {"model.visual." + name: tensor.contiguous() for name, tensor in reference.state_dict().items()}
    state["model.language_model.never_read"] = torch.ones(2, 2)
    shard = "tiny.safetensors"
    save_file(state, tmp_path / shard)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_vl", "vision_config": config.to_dict()}), encoding="utf-8")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {key: shard for key in state}}), encoding="utf-8")
    opened_keys, original_safe_open = [], visual.safe_open
    class TraceReader:
        def __init__(self, *args, **kwargs):
            self.inner = original_safe_open(*args, **kwargs)
        def __enter__(self):
            self.reader = self.inner.__enter__()
            return self
        def get_tensor(self, key):
            opened_keys.append(key)
            assert key.startswith("model.visual.")
            return self.reader.get_tensor(key)
        def __exit__(self, *args):
            return self.inner.__exit__(*args)
    monkeypatch.setattr(visual, "safe_open", TraceReader)
    class Processor:
        calls = 0
        def __call__(self, **kwargs):
            assert kwargs["device"] == "cpu"
            self.calls += 1
            return {"pixel_values": torch.arange(16 * 12).reshape(16, 12).float() / 255,
                    "image_grid_thw": torch.tensor([[1, 4, 4]])}
    processor = Processor()
    encoder = visual.FrozenVisualEncoder.from_local(tmp_path, device="cpu", dtype=dtype, image_processor=processor)
    assert encoder.vision.rotary_pos_emb.inv_freq.dtype == torch.float32
    torch.testing.assert_close(encoder.vision.rotary_pos_emb.inv_freq, original_rotary, atol=0, rtol=0)
    assert next(encoder.vision.parameters()).dtype == dtype
    value = encoder.encode(Image.new("RGB", (8, 8)))
    assert processor.calls == 1 and value["metadata"]["forward_calls"] == 1
    assert value["features"].shape == (4, 4, 24)
    assert value["merged_features"].shape == (2, 2, 16)
    assert all(not parameter.requires_grad for parameter in encoder.vision.parameters())
    assert len(opened_keys) == len(state) - 1
    assert not any("language_model" in name for name in opened_keys)
    assert not value["metadata"]["language_model_loaded"]
    assert value["metadata"]["encoder_revision"] == 2
    assert value["metadata"]["rotary_inv_freq_dtype"] == "torch.float32"


def test_rotary_preserves_precast_values_not_half_then_float():
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionRotaryEmbedding
    backbone = torch.nn.Module()
    backbone.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(36)
    backbone.weight = torch.nn.Parameter(torch.randn(2, 2))
    original = backbone.rotary_pos_emb.inv_freq.detach().clone()
    assert not torch.equal(original, original.half().float())
    expected_angles = torch.outer(torch.arange(64, dtype=torch.float32), original)
    encoder = visual.FrozenVisualEncoder(backbone, None, {}, device="cpu", dtype=torch.float16)
    torch.testing.assert_close(encoder.vision.rotary_pos_emb.inv_freq, original, atol=0, rtol=0)
    torch.testing.assert_close(encoder.vision.rotary_pos_emb(64), expected_angles, atol=0, rtol=0)
    assert encoder.vision.weight.dtype == torch.float16
    assert "inv_freq" in encoder.vision.rotary_pos_emb._non_persistent_buffers_set
    assert "rotary_pos_emb.inv_freq" not in encoder.vision.state_dict()
    # An upstream .half() has already destroyed information; fail clearly
    # instead of falsely certifying a half().float() reconstruction.
    damaged = torch.nn.Module()
    damaged.rotary_pos_emb = Qwen3VLVisionRotaryEmbedding(36)
    damaged.weight = torch.nn.Parameter(torch.ones(1))
    damaged.half()
    with pytest.raises(ValueError, match="before dtype conversion"):
        visual.FrozenVisualEncoder(damaged, None, {}, device="cpu", dtype=torch.float16)


def test_old_real_visual_cache_is_retained_but_rejected_for_training(tmp_path):
    old = encoding()
    old["metadata"]["loaded_weight_prefix"] = "model.visual."
    path = tmp_path / "failed_revision1.npz"
    visual.save_visual_encoding(path, old)
    before = path.read_bytes()
    with pytest.raises(ValueError, match="predates the FP32 rotary fix"):
        visual.load_visual_encoding(path)
    assert path.read_bytes() == before


def test_gui_actor_is_forbidden_and_cost_proxy_is_not_a_latency_claim(tmp_path):
    with pytest.raises(ValueError, match="GUI-Actor"):
        visual.inspect_visual_checkpoint(tmp_path / "GUI-Actor-Verifier")
    costs = visual.estimate_vision_cost([1920, 1080], 8)
    assert costs["shared_full_image"]["vision_forward_calls"] == 1
    assert costs["per_candidate_crops"]["vision_forward_calls"] == 8
    assert "not timings" in costs["interpretation"]
