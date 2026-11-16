import numpy as np
import torch

from clickagent.attention_field_ranker import (AttentionFieldRanker, FieldRankerConfig,
    build_attention_field, make_field_batch, subset_field_candidates)
from clickagent.candidate_ranker import ranker_loss
from test_candidate_ranker import record


def field_record(seed=5):
    r = record(seed)
    a = np.random.default_rng(seed+100).random((3, 96, 6), dtype=np.float32)*.01
    r['attention_field'] = build_attention_field(a, r['parent_indices'])
    return r


def test_every_candidate_receives_secondary_attention_and_preserves_head_order():
    a = np.zeros((2, 96, 3), np.float32); a[:, :, 0] = .6; a[:, :, 1] = .3; a[:, :, 2] = .1
    a[1, 7] = [.1, .2, .7]
    f = build_attention_field(a, [0, 2])
    assert f.shape == (2, 2, 96, 6)
    np.testing.assert_allclose(f[1, 0, 0, 0], np.log1p(.3), rtol=.002)
    assert f[1, 0, 0, 5] == 0 and f[1, 1, 7, 5] == 1
    assert f[1, 0, 0, 0] > 0


def test_matched_control_is_invariant_to_secondary_values_but_full_is_not():
    torch.manual_seed(8); r = field_record(); inputs, _ = make_field_batch([r], 'cpu')
    altered = dict(inputs); changed = inputs['attention_field'].clone()
    changed[..., :5] += (1-changed[..., 5:6])*3
    altered['attention_field'] = changed
    models = [AttentionFieldRanker(FieldRankerConfig(8, mode), np.ones(96)/96).eval() for mode in ('all', 'peaks_only')]
    for model in models:
        # Residual output starts at zero in production; activate it to test dependence.
        torch.nn.init.normal_(model.field_rank[-1].weight, std=.2)
    assert sum(p.numel() for p in models[0].parameters()) == sum(p.numel() for p in models[1].parameters())
    torch.testing.assert_close(models[1](**inputs)['scores'], models[1](**altered)['scores'])
    assert not torch.allclose(models[0](**inputs)['scores'], models[0](**altered)['scores'])


def test_field_batching_is_isolated_and_candidate_filter_reindexes():
    torch.manual_seed(12); records = [field_record(4), field_record(7)]
    model = AttentionFieldRanker(FieldRankerConfig(8), np.ones(96)/96).eval()
    torch.nn.init.normal_(model.field_rank[-1].weight, std=.2)
    batch, _ = make_field_batch(records, 'cpu'); together = model(**batch); cursor = 0
    for r in records:
        inputs, _ = make_field_batch([r], 'cpu'); n = len(r['parent_indices'])
        torch.testing.assert_close(together['scores'][cursor:cursor+n], model(**inputs)['scores'], atol=1e-5, rtol=1e-5)
        keep = r['label_area'].sum(-1) == 0; sub = subset_field_candidates(r, keep)
        np.testing.assert_array_equal(sub['attention_field'], r['attention_field'][keep])
        assert sub['source_candidate'].max() == len(sub['parent_indices'])-1
        cursor += n


def test_labels_do_not_enter_field_inputs_and_training_gradients_are_finite():
    torch.manual_seed(3); r = field_record(); inputs, labels = make_field_batch([r], 'cpu')
    altered = dict(r, label_area=np.ones_like(r['label_area']), label_offsets=np.ones_like(r['label_offsets']))
    reference, _ = make_field_batch([altered], 'cpu', labels=False)
    for name, value in inputs.items():
        if isinstance(value, torch.Tensor): torch.testing.assert_close(value, reference[name])
        else: assert value == reference[name]
    model = AttentionFieldRanker(FieldRankerConfig(8), np.ones(96)/96)
    loss, _ = ranker_loss(model(**inputs), **labels); loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
