import numpy as np
import pytest
import torch

from clickagent.phrase_grounder import (PhraseConfig, PhraseSpatialGrounder, decode_phrase,
                                       join_children, load_phrase, phrase_loss, save_phrase, split_children)
from clickagent.spatial_geometry import spatial_supervision


def toy(variant='local'):
    torch.manual_seed(23)
    config = PhraseConfig(image_dim=12, vision_dim=8, query_dim=18, query_layers=3, heads=4, width=8, variant=variant)
    model = PhraseSpatialGrounder(config)
    data = dict(attention=torch.rand(2, 3, 4, 6, requires_grad=True)*.01,
                query_hidden=torch.randn(2, 3, 18), image_hidden=torch.randn(2, 6, 12),
                vision_hidden=torch.randn(2, 24, 8), token_mask=torch.tensor([[True, True, False], [True, True, True]]),
                grid_shape=[2, 3])
    return model, data


def test_hierarchy_preserves_coarse_marginal_and_parent_choice():
    model, data = toy(); output = model(**data)
    marginal = split_children(output['fine_log_prob'].exp()).sum(-1)
    expected = output['coarse_logits'].flatten(1).softmax(-1).reshape(2, 2, 3)
    torch.testing.assert_close(marginal, expected)
    output['child_log_prob'] = torch.randn_like(output['child_log_prob']).log_softmax(-1)*100
    for i in range(2):
        result = decode_phrase(output, i, [101, 79], 1)[0]
        assert result['coarse_index'] == int(output['coarse_logits'][i].argmax())
        rr, cc = divmod(result['patch_index'], 6)
        assert rr//2*3+cc//2 == result['coarse_index']
        assert 0 <= result['point'][0] < 101 and 0 <= result['point'][1] < 79


@pytest.mark.parametrize('variant', ['first', 'uniform', 'global', 'local'])
def test_padded_tokens_never_change_prediction(variant):
    model, data = toy(variant); model.eval()
    with torch.no_grad():
        model.global_selector.weight.normal_(); model.local_selector.weight.normal_()
        reference = model(**data)
        altered = dict(data)
        altered['attention'] = torch.cat((data['attention'], torch.rand(2, 2, 4, 6)*100), 1)
        altered['query_hidden'] = torch.cat((data['query_hidden'], torch.randn(2, 2, 18)*100), 1)
        altered['token_mask'] = torch.cat((data['token_mask'], torch.zeros(2, 2, dtype=torch.bool)), 1)
        actual = model(**altered)
    torch.testing.assert_close(actual['coarse_logits'], reference['coarse_logits'])
    torch.testing.assert_close(actual['offsets'], reference['offsets'])
    assert torch.count_nonzero(actual['token_weights'][:, -2:]) == 0


def test_later_token_maps_contribute_to_local_but_not_first_control():
    for variant in ('first', 'local'):
        model, data = toy(variant)
        attention = data['attention'].detach().requires_grad_(True)
        model(**dict(data, attention=attention))['coarse_logits'][:, 0, 0].sum().backward()
        later_gradient = attention.grad[:, 1:].abs().sum()
        assert bool(later_gradient > 0) == (variant == 'local')


def test_local_selector_can_choose_different_words_at_different_positions():
    model, data = toy('local')
    with torch.no_grad():
        model.local_selector.weight.normal_()
        output = model(**data)
    weights = output['token_weights'][1]
    assert not torch.allclose(weights[:, 0], weights[:, 1])
    torch.testing.assert_close(weights.sum(0), torch.ones(6))


def test_local_fusion_preserves_paired_token_order_without_position_bias():
    model, data = toy('local')
    with torch.no_grad():
        model.local_selector.weight.normal_(); model.global_selector.weight.normal_(); model.head_delta.weight.normal_()
        reference = model(**data)
        changed = dict(data, attention=data['attention'].clone(), query_hidden=data['query_hidden'].clone())
        changed['attention'][1, :2] = data['attention'][1, [1, 0]]
        changed['query_hidden'][1, :2] = data['query_hidden'][1, [1, 0]]
        actual = model(**changed)
    torch.testing.assert_close(actual['coarse_logits'], reference['coarse_logits'])
    torch.testing.assert_close(actual['offsets'], reference['offsets'])


def test_spatial_supervision_learns_and_checkpoint_round_trips(tmp_path):
    model, data = toy(); weights, offsets = spatial_supervision({'type': 'bbox', 'boxes': [[.20, .20, .22, .22]]}, [4, 6])
    labels = dict(weights=torch.tensor(np.stack([weights]*2)), offsets=torch.tensor(np.stack([offsets]*2)))
    data = {k: v.detach() if isinstance(v, torch.Tensor) else v for k, v in data.items()}
    before = float(phrase_loss(model(**data), **labels)[0].detach())
    previous_threads = torch.get_num_threads(); torch.set_num_threads(2)
    try:
        opt = torch.optim.AdamW(model.parameters(), lr=.02)
        for _ in range(65):
            opt.zero_grad(); loss, _ = phrase_loss(model(**data), **labels); loss.backward(); opt.step()
        result = model(**data)
        assert float(phrase_loss(result, **labels)[0].detach()) < before*.25
        assert result['coarse_logits'].flatten(1).argmax(-1).tolist() == [0, 0]
        path = tmp_path/'phrase.pt'; save_phrase(path, model, {'steps': 65})
        restored, meta = load_phrase(path)
        assert meta['steps'] == 65
        torch.testing.assert_close(restored(**data)['coarse_logits'], result['coarse_logits'])
    finally:
        torch.set_num_threads(previous_threads)


def test_empty_targets_and_ground_truth_are_rejected():
    model, data = toy()
    with pytest.raises(ValueError, match='empty target'):
        model(**dict(data, token_mask=torch.zeros_like(data['token_mask'])))
    with pytest.raises(TypeError):
        model(**data, target_geometry={'type': 'bbox'})
    values = torch.arange(24).reshape(1, 4, 6)
    torch.testing.assert_close(join_children(split_children(values)), values)
