import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
import torch


SCRIPTS = Path(__file__).resolve().parents[1]/'scripts'
sys.path.insert(0, str(SCRIPTS))
spec = importlib.util.spec_from_file_location('routing_diagnostic', SCRIPTS/'diagnose_semantic_head_routing.py')
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def test_uniform_routing_reference_preserves_all_heads_and_tokens():
    head = np.full((6, 96, 4), 1/96)
    token = np.full((6, 4), 1/3)
    values, matrices = diagnostic.routing_observations(head, token, 2, 3)
    np.testing.assert_allclose(values['head_entropy_normalized'], 1)
    np.testing.assert_allclose(values['effective_heads'], 96)
    np.testing.assert_allclose(values['head_group_cosine'], 1)
    np.testing.assert_allclose(values['head_group_js_normalized'], 0, atol=1e-12)
    np.testing.assert_allclose(values['head_candidate_js_normalized'], 0, atol=1e-12)
    np.testing.assert_allclose(values['token_entropy_normalized'], 1)
    assert matrices['head_weights'].shape == (4, 96)


def test_group_and_candidate_diversity_are_compared_on_correct_axes():
    head = np.zeros((4, 96, 4))
    for candidate in range(2):
        for token in range(2):
            for group in range(4):
                head[candidate*2+token, candidate*4+group+token*8, group] = 1
    token = np.full((4, 4), .5)
    values, matrices = diagnostic.routing_observations(head, token, 2, 2)
    np.testing.assert_allclose(values['effective_heads'], 1)
    np.testing.assert_allclose(values['head_group_cosine'], 0)
    np.testing.assert_allclose(values['head_group_js_normalized'], 1)
    np.testing.assert_allclose(values['head_candidate_js_normalized'], np.log(2)/np.log(96))
    # Arbitrary token differences do not create candidate diversity if both
    # candidates share the same routing at each token.
    head[2:] = head[:2]
    values, _ = diagnostic.routing_observations(head, token, 2, 2)
    np.testing.assert_allclose(values['head_candidate_js_normalized'], 0)


def test_single_token_normalized_entropy_is_excluded_and_alignment_checked():
    head, token = np.full((2, 96, 4), 1/96), np.ones((2, 4))
    values, _ = diagnostic.routing_observations(head, token, 2, 1)
    assert 'token_entropy_normalized' not in values
    np.testing.assert_allclose(values['token_effective_count'], 1)
    with pytest.raises(ValueError, match='retain every candidate/token/head'):
        diagnostic.routing_observations(head[:, :95], token, 2, 1)


def test_common_residual_offset_does_not_become_ranking_strength():
    base = np.array([1., 2., 3.]); residual = np.full(3, 100.)
    metrics, outcomes = diagnostic.score_observations(base, residual, base+residual,
        np.array([0, 0, 1]), np.ones(3), np.ones((3, 4, 2)), np.ones((3, 3)))
    assert metrics['residual_abs_mean'] == 100
    assert metrics['residual_std_over_base_std'] == 0
    assert metrics['score_distribution_js'] == pytest.approx(0, abs=1e-12)
    assert not outcomes['top1_changed']
    assert outcomes['base_point_hit'] and outcomes['rank_point_hit']


def test_condition_intervention_is_temporary_even_on_error():
    core = torch.nn.Module()
    core.route_query, core.route_phrase = torch.nn.Linear(3, 2), torch.nn.Linear(3, 2)
    value = torch.ones(1, 3)
    before = core.route_query(value).detach().clone()
    with pytest.raises(RuntimeError, match='fixture'):
        with diagnostic.zero_new_query_condition(core):
            assert torch.equal(core.route_query(value), torch.zeros(1, 2))
            assert torch.equal(core.route_phrase(value), torch.zeros(1, 2))
            raise RuntimeError('fixture')
    torch.testing.assert_close(core.route_query(value), before, rtol=0, atol=0)
    assert not core.route_query._forward_hooks and not core.route_phrase._forward_hooks
