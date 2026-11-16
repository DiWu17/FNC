import copy

import numpy as np
import pytest
import torch

from clickagent.internal_candidate_scorer import (
    CANDIDATE_KEYS, INPUT_KEYS, InternalCandidateScorer, InternalCandidateScorerConfig,
    build_internal_inputs, internal_inputs_to_torch, load_internal_scorer,
    save_internal_scorer, subset_internal_inputs,
)


@pytest.fixture(autouse=True)
def cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def raw_view(seed=5, tokens=3, grid=(2, 3)):
    rng = np.random.default_rng(seed); p = int(np.prod(grid))
    arrays = dict(target_attention=(rng.random((tokens, 96, p))*.01).astype(np.float32),
        target_hidden=rng.normal(size=(tokens, 12288)).astype(np.float16),
        image_hidden=rng.normal(size=(p, 4096)).astype(np.float16),
        vision_hidden=rng.normal(size=(p*4, 1152)).astype(np.float16))
    return arrays, dict(grid_shape=grid)


def inputs(local=True, tokens=3, points=None):
    first, first_meta = raw_view(tokens=tokens)
    second, second_meta = raw_view(8, 2, (3, 2)) if local else (None, None)
    if points is None:
        points = np.array([[100, 50], [500, 250], [749, 499], [900, 550]])
    return build_internal_inputs(first, first_meta, second, second_meta, points, (1000, 600),
                                 dict(box=[250, 100, 750, 500]) if local else None)


def model(mode='full'):
    return InternalCandidateScorer(InternalCandidateScorerConfig(width=32, head_width=4, dropout=0,
                                                                head_mode=mode)).eval()


def test_cache_retains_raw_dimensions_token_heads_and_point_view_masks():
    data = inputs()
    assert set(data) == INPUT_KEYS
    assert data['query_hidden'].shape == (2, 3, 12288)
    assert data['image_hidden'].shape == (2, 6, 4096)
    assert data['vision_hidden'].shape == (2, 24, 1152)
    assert data['attention_field'].shape == (4, 2, 3, 96, 15)
    np.testing.assert_array_equal(data['query_mask'], [[1, 1, 1], [1, 1, 0]])
    np.testing.assert_array_equal(data['view_mask'], [[1, 0], [1, 1], [1, 1], [1, 0]])
    assert not data['coarse_mask'][[0, 3], 1].any()
    assert not data['fine_mask'][[0, 3], 1].any()
    assert not data['attention_field'][[0, 3], 1].any()
    assert data['attention_field'].dtype == np.float16
    assert data['coarse_indices'].dtype == np.int64
    assert data['coarse_mask'].dtype == bool
    assert np.isfinite(data['attention_field']).all()


def test_builder_does_not_read_annotation_or_metadata_labels():
    first, fm = raw_view(); second, sm = raw_view(7, 2)
    points = np.array([[500, 250]])
    original = build_internal_inputs(first, fm, second, sm, points, (1000, 600), dict(box=[250, 100, 750, 500]))
    fm = dict(fm, target_bbox='poison', labels='poison', target_geometry='poison')
    sm = dict(sm, target_geometry={'bbox': [0, 0, 1, 1]})
    changed = build_internal_inputs(first, fm, second, sm, points, (1000, 600),
                                     dict(box=[250, 100, 750, 500], labels='poison'))
    for name in original:
        np.testing.assert_array_equal(original[name], changed[name])
    with pytest.raises(ValueError, match='supervision'):
        internal_inputs_to_torch(dict(original, label_hit=np.ones(1)))


def test_center_channels_and_neighbor_order_match_original_raw_attention():
    first, fm = raw_view(tokens=2)
    data = build_internal_inputs(first, fm, None, None, np.array([[500, 250]]), (1000, 600))
    from clickagent.attention_field_ranker import build_attention_field
    parent = 1  # floor(x*3)=1, floor(y*2)=0
    reference = build_attention_field(first['target_attention'], np.array([parent]))
    np.testing.assert_array_equal(data['attention_field'][:, 0, :2, :, :6], reference)
    # 3x3's center slot is4 and must equal the six-channel log-relative value.
    np.testing.assert_array_equal(data['attention_field'][0, 0, :2, :, 10], reference[0, :, :, 0])


def test_subset_is_explicit_even_when_token_or_patch_count_equals_candidates():
    data = inputs(tokens=4)
    subset = subset_internal_inputs(data, np.array([3, 1], np.int64))
    assert subset['query_hidden'].shape == data['query_hidden'].shape
    for name in data:
        np.testing.assert_array_equal(subset[name], data[name][[3, 1]] if name in CANDIDATE_KEYS else data[name])
    with pytest.raises(ValueError, match='Repeated'):
        subset_internal_inputs(data, np.array([1, 1]))
    with pytest.raises(ValueError, match='out of range'):
        subset_internal_inputs(data, np.array([4]))


def test_parameters_match_across_all_three_modes_and_queries_are_new_projections():
    torch.manual_seed(9); full = model()
    for mode in ('pooled_control', 'no_local'):
        torch.manual_seed(9); control = model(mode)
        assert sum(p.numel() for p in control.parameters()) == sum(p.numel() for p in full.parameters())
        assert set(control.state_dict()) == set(full.state_dict())
        for name in full.state_dict():
            torch.testing.assert_close(control.state_dict()[name], full.state_dict()[name], rtol=0, atol=0)
    assert len(full.query_layers) == 3
    assert all(layer.in_features == 4096 for layer in full.query_layers)
    assert full.image[0].in_features == 4096 and full.vision[0].in_features == 1152


def test_every_token_and_head_can_affect_candidate_conditioned_evidence_and_gradients():
    torch.manual_seed(5); scorer = model()
    batch = internal_inputs_to_torch(inputs())
    batch['attention_field'] = batch['attention_field'].float().requires_grad_(True)
    batch['query_hidden'] = batch['query_hidden'].float().requires_grad_(True)
    output = scorer(return_diagnostics=True, **batch)
    assert output['scores'].shape == (4,) and output['null_scores'].shape == (1,)
    assert output['candidate_lengths'] == [4]
    assert output['head_evidence'].shape == (4, 2, 3, 96)
    assert not torch.allclose(output['head_evidence'][1, 0, 0], output['head_evidence'][1, 0, 1])
    (output['scores'].square().sum()+output['null_scores'].square().sum()).backward()
    gradient = batch['attention_field'].grad[:, 0].abs().sum(-1)
    assert (gradient > 0).all() and torch.isfinite(gradient).all()
    assert (batch['query_hidden'].grad[0].abs().sum(-1) > 0).all()
    assert all(torch.isfinite(p.grad).all() for p in scorer.parameters() if p.grad is not None)
    changed = dict(batch, attention_field=batch['attention_field'].detach().clone())
    changed['attention_field'][1, 0, 0, 17, 0] += 2
    altered = scorer(return_diagnostics=True, **changed)
    delta = (output['head_evidence'].detach()-altered['head_evidence'].detach()).abs()
    assert delta[1, 0, 0, 17] > 0
    delta[1, 0, 0, 17] = 0
    assert torch.count_nonzero(delta) == 0  # independence before set comparison


def test_pooled_control_homogenizes_tokens_before_interaction_not_parameter_count():
    torch.manual_seed(11); scorer = model('pooled_control')
    batch = internal_inputs_to_torch(inputs())
    output = scorer(return_diagnostics=True, **batch)
    # No dropout: all valid global token rows have the exact same evidence.
    for token in (1, 2):
        torch.testing.assert_close(output['head_evidence'][:, 0, token], output['head_evidence'][:, 0, 0], rtol=0, atol=0)
    # Head identities and their original evidence are still distinct.
    assert output['head_evidence'][0, 0, 0].std() > 0


def test_no_local_control_and_outside_crop_only_case_have_no_local_dependency():
    torch.manual_seed(12); scorer = model('no_local')
    batch = internal_inputs_to_torch(inputs())
    changed = {name: value.clone() for name, value in batch.items()}
    for name in ('query_hidden', 'image_hidden', 'vision_hidden'):
        changed[name][1] = torch.randn_like(changed[name][1])*5
    changed['attention_field'][:, 1] += 3
    a, b = scorer(**batch), scorer(**changed)
    for name in ('scores', 'null_scores', 'candidate_embeddings'):
        torch.testing.assert_close(a[name], b[name], rtol=0, atol=0)
    # Full mode must also completely ignore local content if ALL queried
    # points are outside that view, including the null/set context.
    scorer = model()
    one = internal_inputs_to_torch(subset_internal_inputs(inputs(), np.array([0])))
    altered = {name: value.clone() for name, value in one.items()}
    for name in ('query_hidden', 'image_hidden', 'vision_hidden'):
        altered[name][1] *= -4
    for name in ('scores', 'null_scores'):
        torch.testing.assert_close(scorer(**one)[name], scorer(**altered)[name], rtol=0, atol=0)


def test_set_permutation_equivariance_and_actual_small_pool_forward():
    torch.manual_seed(13); scorer = model()
    data = inputs(); normal = scorer(**internal_inputs_to_torch(data))
    permutation = np.array([2, 0, 3, 1])
    shuffled = scorer(**internal_inputs_to_torch(subset_internal_inputs(data, permutation)))
    torch.testing.assert_close(shuffled['scores'], normal['scores'][permutation], rtol=2e-5, atol=2e-5)
    torch.testing.assert_close(shuffled['null_scores'], normal['null_scores'], rtol=2e-5, atol=2e-5)
    small = scorer(**internal_inputs_to_torch(subset_internal_inputs(data, np.array([0, 1]))))
    assert small['candidate_lengths'] == [2]
    assert not torch.allclose(small['scores'], normal['scores'][:2])


def test_missing_local_zero_name_and_empty_candidate_sets_are_finite():
    scorer = model()
    for data in (inputs(local=False), inputs(local=False, tokens=0),
                 inputs(local=False, points=np.empty((0, 2), np.int64))):
        output = scorer(**internal_inputs_to_torch(data))
        assert torch.isfinite(output['scores']).all() and torch.isfinite(output['null_scores']).all()
        assert output['scores'].shape == (len(data['points']),)
    with pytest.raises(ValueError, match='inside the image'):
        inputs(points=np.array([[1000, 20]]))


def test_save_load_does_not_mutate_live_model_or_mode(tmp_path):
    torch.manual_seed(14); scorer = model(); scorer.train()
    state = {name: value.detach().clone() for name, value in scorer.state_dict().items()}
    filename = tmp_path/'scorer.pt'
    save_internal_scorer(filename, scorer, {'purpose': 'cpu fixture'})
    assert scorer.training
    for name, value in scorer.state_dict().items():
        torch.testing.assert_close(value, state[name], rtol=0, atol=0)
    loaded, metadata = load_internal_scorer(filename)
    scorer.eval(); data = internal_inputs_to_torch(inputs())
    original, restored = scorer(**data), loaded(**data)
    assert metadata == {'purpose': 'cpu fixture'}
    for name in ('scores', 'null_scores', 'candidate_embeddings'):
        torch.testing.assert_close(original[name], restored[name], rtol=0, atol=0)
