import numpy as np
import torch

from clickagent.candidate_ranker import (RankerConfig, SemanticCandidateRanker, build_candidate_inputs,
    candidate_supervision, decode_ranker, make_ranker_batch, ranker_loss, subset_candidates)


def record(seed=5):
    rng = np.random.default_rng(seed); grid = [2, 3]; p = 6
    a = rng.random((3, 96, p), dtype=np.float32)*.01
    r = build_candidate_inputs(a, rng.normal(size=(3, 12288)).astype('float16'),
        rng.normal(size=(p, 4096)).astype('float16'), rng.normal(size=(p*4, 1152)).astype('float16'), grid)
    r.update(candidate_supervision(dict(type='bbox', boxes=[[.12, .15, .3, .45]]), grid, r['parent_indices']))
    r.update(metadata=dict(grid_shape=grid, target_available=True), case=dict(image_size=[813, 517]))
    return r


def test_all_peak_sources_retained_and_vision_children_in_raster_order():
    a = np.zeros((1, 96, 6), np.float32); a[:, :, 0] = 1
    v = np.repeat(np.arange(24, dtype=np.float16)[:, None], 1152, axis=1)
    r = build_candidate_inputs(a, np.zeros((1, 12288)), np.zeros((6, 4096)), v, [2, 3])
    assert r['parent_indices'].tolist() == [0]
    assert len(r['source_head']) == 96 and r['source_candidate'].tolist() == [0]*96
    assert r['vision_hidden'][0, :, 0].tolist() == [0, 1, 6, 7]


def test_geometry_is_only_supervision_and_never_enters_model_inputs():
    r = record(); before, _ = make_ranker_batch([r], 'cpu', labels=False)
    altered = dict(r, label_area=np.ones_like(r['label_area']), label_offsets=np.ones_like(r['label_offsets']))
    after, labels = make_ranker_batch([altered], 'cpu', labels=False)
    assert labels is None and all(not k.startswith('label') for k in before)
    for name in before:
        if isinstance(before[name], torch.Tensor): torch.testing.assert_close(before[name], after[name])
        else: assert before[name] == after[name]


def test_ranker_has_finite_gradients_and_source_evidence_is_order_invariant():
    torch.manual_seed(13); r = record(); inputs, labels = make_ranker_batch([r], 'cpu')
    model = SemanticCandidateRanker(RankerConfig(width=8), np.linspace(.0, .3, 96))
    model.eval(); actual = model(**inputs); loss, _ = ranker_loss(actual, **labels); loss.backward()
    assert torch.isfinite(loss) and all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    perm = torch.randperm(len(inputs['source_head'])); altered = dict(inputs)
    for name in ('source_features', 'source_token', 'source_head', 'source_candidate', 'source_case'):
        altered[name] = inputs[name][perm]
    reference = model(**altered)
    for name in ('scores', 'child_logits', 'offsets', 'null_scores'):
        torch.testing.assert_close(actual[name], reference[name], atol=1e-5, rtol=1e-5)


def test_candidate_removal_reindexes_sources_and_null_loss_is_finite():
    r = record(); keep = r['label_area'].sum(-1) == 0; filtered = subset_candidates(r, keep)
    assert not filtered['label_area'].any()
    assert set(filtered['parent_indices']).isdisjoint(set(r['parent_indices'][~keep]))
    assert filtered['source_candidate'].max() == len(filtered['parent_indices'])-1
    inputs, labels = make_ranker_batch([filtered], 'cpu')
    model = SemanticCandidateRanker(RankerConfig(width=8), np.ones(96)/96)
    loss, metrics = ranker_loss(model(**inputs), **labels)
    assert torch.isfinite(loss) and metrics['child'] == metrics['offset'] == 0


def test_batching_does_not_mix_images_and_points_stay_with_selected_parent():
    torch.manual_seed(4); records = [record(6), record(7)]
    model = SemanticCandidateRanker(RankerConfig(width=8), np.ones(96)/96).eval()
    batch, _ = make_ranker_batch(records, 'cpu'); together = model(**batch)
    cursor = 0
    for i, r in enumerate(records):
        inputs, _ = make_ranker_batch([r], 'cpu'); separate = model(**inputs); n = len(r['parent_indices'])
        torch.testing.assert_close(together['scores'][cursor:cursor+n], separate['scores'], atol=1e-5, rtol=1e-5)
        result = decode_ranker(together, r, i, top_k=1)[0]; x, y = result['point']
        assert 0 <= x < 813 and 0 <= y < 517
        cursor += n


def test_readout_ablations_ignore_disabled_hidden_arrays():
    torch.manual_seed(17); inputs, _ = make_ranker_batch([record()], 'cpu')
    for variant in ('no_visual', 'no_query', 'stats_only'):
        model = SemanticCandidateRanker(RankerConfig(width=8, variant=variant), np.ones(96)/96).eval()
        reference = model(**inputs); changed = dict(inputs)
        fields = []
        if variant in ('no_visual', 'stats_only'): fields += ['image_hidden', 'vision_hidden']
        if variant in ('no_query', 'stats_only'): fields += ['query_hidden']
        for name in fields: changed[name] = torch.randn_like(changed[name])*100
        actual = model(**changed)
        torch.testing.assert_close(reference['scores'], actual['scores'])

