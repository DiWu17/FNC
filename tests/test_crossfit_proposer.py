"""Small CPU-only checks for the fresh, teacher-free crossfit proposer."""
import numpy as np
import pytest
import torch

from clickagent.attention_field_ranker import make_field_batch
from clickagent.crossfit_proposer import (
    CrossfitProposerConfig, build_raw_record, load_crossfit_proposer,
    make_fresh_proposer, predict_raw, save_crossfit_proposer, state_sha256,
)
from clickagent.point_aligned_readout import aligned_loss, all_candidate_points, candidate_point_hits


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def raw():
    rng = np.random.default_rng(12)
    tokens, patches = 2, 4
    attention = rng.uniform(.001, .02, (tokens, 96, patches)).astype(np.float32)
    arrays = dict(target_attention=attention,
                  target_hidden=rng.normal(size=(tokens, 12288)).astype(np.float16),
                  image_hidden=rng.normal(size=(patches, 4096)).astype(np.float16),
                  vision_hidden=rng.normal(size=(4*patches, 1152)).astype(np.float16))
    meta = dict(target_available=True, target_text="Settings", own_token_forward=True,
                all_backbone_parameters_frozen=True, ground_truth_in_model_input=False,
                grid_shape=[2, 2])
    return arrays, meta


def test_same_fresh_state_and_zero_prior_preserve_rng():
    torch.manual_seed(7); before = torch.get_rng_state().clone()
    first = make_fresh_proposer(); second = make_fresh_proposer()
    assert torch.equal(torch.get_rng_state(), before)
    assert first.initial_state_sha256 == second.initial_state_sha256 == state_sha256(first)
    assert sum(p.numel() for p in first.parameters()) == 611919
    assert torch.equal(first.head_prior, torch.zeros(96))
    assert "head_prior" not in dict(first.named_parameters())
    assert first.connect_fine
    other = make_fresh_proposer(CrossfitProposerConfig(seed=20261202))
    assert state_sha256(first) != state_sha256(other)


def test_predict_does_not_consume_gt_and_preserves_all_native_heads(raw):
    arrays, meta = raw
    model = make_fresh_proposer().eval()
    expected = predict_raw(model, arrays, meta, [320, 200], topk=64)
    class Poison:
        def __array__(self, *args, **kwargs):
            raise AssertionError("GT was converted to an input")
        def __bool__(self):
            raise AssertionError("GT was used in a decision")
    changed = dict(arrays, target_geometry=Poison(), label_fixed_point_hit=Poison())
    actual = predict_raw(model, changed, dict(meta, target_bbox=Poison()), [320, 200], topk=64)
    assert actual == expected
    record = build_raw_record(arrays, meta, [320, 200])
    assert record["attention_field"].shape == (4, 2, 96, 6)
    inputs, targets = make_field_batch([record], "cpu", labels=False)
    assert targets is None and not any("label" in k or "geometry" in k for k in inputs)
    assert len(actual["all_points"]) == 4
    assert actual["decoder"] == "own" and actual["ground_truth_in_readout_inputs"] is False
    assert all(0 <= x < 320 and 0 <= y < 200 for x, y in actual["all_points"])


def test_actual_point_supervision_backward_is_finite(raw):
    arrays, meta = raw
    geometry = {"type": "bbox", "boxes": [[.1, .1, .9, .9]]}
    record = build_raw_record(arrays, meta, [320, 200], geometry=geometry)
    record["case"]["target_geometry"] = geometry
    model = make_fresh_proposer()
    inputs, targets = make_field_batch([record], "cpu", labels=True)
    out = model(**inputs)
    points = all_candidate_points(out, [record])
    hits = torch.from_numpy(np.concatenate(candidate_point_hits(points, [record])))
    loss, _ = aligned_loss(out, targets, hits, True, region_weight=.25)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    assert any(p.grad is not None and bool(p.grad.abs().sum() > 0) for p in model.parameters())
    assert torch.equal(model.head_prior, torch.zeros(96))


def test_save_load_preserves_prediction_and_live_model_mode(raw, tmp_path):
    arrays, meta = raw
    model = make_fresh_proposer().eval()
    expected = predict_raw(model, arrays, meta, [320, 200])
    model.train(); before = state_sha256(model)
    destination = tmp_path / "fold_0.pt"
    save_crossfit_proposer(destination, model, {"role": "fold_0"})
    assert model.training and state_sha256(model) == before
    loaded, metadata = load_crossfit_proposer(destination, "cpu")
    assert not loaded.training and metadata["role"] == "fold_0"
    assert predict_raw(loaded, arrays, meta, [320, 200]) == expected
    state = torch.load(destination, map_location="cpu", weights_only=True)
    state["state_dict"]["head_prior"][0] = 1
    torch.save(state, destination)
    with pytest.raises(ValueError, match="head prior"):
        load_crossfit_proposer(destination)


def test_no_name_and_unfrozen_archives_are_explicit(raw):
    arrays, meta = raw
    model = make_fresh_proposer().eval()
    out = predict_raw(model, {}, dict(meta, target_available=False), [320, 200])
    assert out["failure"] == "no_name" and out["all_points"] == []
    with pytest.raises(ValueError, match="GT"):
        predict_raw(model, arrays, dict(meta, ground_truth_in_model_input=True), [320, 200])
    with pytest.raises(ValueError, match="frozen"):
        predict_raw(model, arrays, dict(meta, all_backbone_parameters_frozen=False), [320, 200])
