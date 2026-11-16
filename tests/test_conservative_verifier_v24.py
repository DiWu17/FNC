import numpy as np
from PIL import Image
from clickagent.conservative_verifier_v24 import (make_pool, visual_inputs, order_for, combine,
    application_folds, fit_gate, predict_utility, features, FEATURE_NAMES)
from clickagent.visual_feedback_v23 import make_pool as old_pool


def test_expansion_preserves_old_shortlist_and_anchor():
    p = [[2, 3], [3, 3], [50, 50], [90, 80], [90, 82], [400, 300]]
    scores = [6, 5, 4, 3, 2, 1]
    q = make_pool(p, scores, (800, 600))
    assert q[0] == [2, 3] and len(q) == len(p)
    assert set(map(tuple, old_pool(p, scores, (800, 600)))) <= set(map(tuple, q))
    assert make_pool([], [], (10, 10)) == []


def test_candidate_views_are_independent_of_anchor_and_preserve_exact_centers():
    image = Image.new('RGB', (1200, 800), 'white')
    pool = [[3, 5], [1199, 799], [600, 400]]
    ims, messages, g = visual_inputs(image, 'complete instruction', pool, [2, 0, 1])
    assert ims[0].size == (512, 512)
    assert all(v.size == (256, 256) for v in ims[1:])
    assert g['input_pixels'] == 512**2+3*256**2
    assert 'complete instruction' in messages[-1]['content'][-1]['text']
    for crop in g['crops']:
        x, y, x1, y1 = crop['box']; p = crop['point']
        assert x <= p[0] < x1 and y <= p[1] < y1
        assert crop['render']['marked_centers'] == [[p[0]-x, p[1]-y]]
    reordered = [pool[2], pool[0], pool[1]]
    other, _, gg = visual_inputs(image, 'complete instruction', reordered, [0, 1, 2])
    assert {tuple(v['point']): v['box'] for v in g['crops']} == {tuple(v['point']): v['box'] for v in gg['crops']}
    for im in ims+other+[image]: im.close()


def test_order_alignment_consensus_and_margin_gate():
    pool = [[1, 1], [10, 10], [20, 20]]
    obs = [dict(order=[2, 0, 1], scores=[0, 1, 4, -2]), dict(order=[1, 0, 2], scores=[5, 1, 0, -2])]
    result = combine(pool, obs)
    assert result['per_order_choice'] == [1, 1] and result['fixed_point'] == pool[1]
    obs[1]['scores'] = [4, 5, 0, -2]
    result = combine(pool, obs)
    assert not result['order_agrees'] and result['fixed_point'] == pool[0]
    obs = [dict(order=[0, 1, 2], scores=[1, 1.1, 0, -2])]*2
    assert combine(pool, obs)['fixed_point'] == pool[0]


def test_none_is_keep_and_ties_do_not_authorize_replacement():
    pool = [[1, 1], [2, 2]]
    obs = [dict(order=[0, 1], scores=[0, 1, 8])]*2
    result = combine(pool, obs)
    assert result['unconditional_point'] == pool[0] and not result['fixed_accept']
    assert order_for('s', pool, True) == list(reversed(order_for('s', pool)))


def test_application_groups_never_cross_folds():
    cases = [dict(sample_id=f'{s}_{a}_{i}', source=s, source_group=a)
             for s in ['a', 'b'] for a in ['app1', 'app2', 'app3'] for i in range(3)]
    folds = application_folds(cases)
    for s in ['a', 'b']:
        for a in ['app1', 'app2', 'app3']:
            assert len({folds[c['sample_id']] for c in cases if c['source']==s and c['source_group']==a}) == 1


def test_gate_training_weights_rgb_and_inference_has_no_labels():
    x = [0.]*len(FEATURE_NAMES)
    rows = [dict(sample_id='a', proposed_change=True, features=x, utility=1),
            dict(sample_id='b', proposed_change=True, features=x, utility=-1)]
    single = fit_gate(rows)
    duplicate = fit_gate([rows[0]]*6+[rows[1]])
    assert np.allclose(single['coefficients'], duplicate['coefficients'])
    assert abs(predict_utility(single, x)) < 1e-7
    empty = fit_gate([])
    assert predict_utility(empty, x) == 0
    pool = [[1, 1], [10, 10]]
    combined = combine(pool, [dict(order=[0, 1], scores=[0, 3, -2])]*2)
    f = features(combined, pool, pool, [5, 3], (100, 100))
    assert len(f) == len(FEATURE_NAMES) and np.isfinite(f).all()
