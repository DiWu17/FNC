import numpy as np
from clickagent.location_selection_v26 import (cell_ids, location_index,
    location_residual, normalized_scores, choose, shuffled_locations)


def test_boundaries_and_unknown():
    p = [[0, 0], [100, 100], [299, 299], [300, 300]]
    assert cell_ids(p, [300, 300]).tolist() == [0, 4, 8, 8]
    assert np.array_equal(location_residual(p, [300, 300], None), np.zeros(4))
    r = location_residual(p, [300, 300], 0)
    assert r[0] == r[1] == 0 and r[2] < 0


def test_frame_and_action_are_required():
    m = dict(ground_complete=True, target_available=True, action='CLICK',
             ground_fields={'location': 'upper_left'})
    assert location_index(m) == 0
    for key, value in [('ground_complete', False), ('target_available', False), ('action', 'WAIT')]:
        assert location_index(dict(m, **{key: value})) is None


def test_residual_selection_identity_and_scale():
    s = np.array([3., 4., 4.])
    assert choose(s) == choose(s, [-100., 100., 0.], 0) == 1
    assert choose([], [], 8) is None
    r = np.array([0., -10., -10.])
    assert choose(s, r, 1) == 0
    assert choose(s * 7 + 20, r, 1) == choose(s, r, 1)
    np.testing.assert_array_equal(normalized_scores([1., 1.]), [0., 0.])


def test_shuffle_preserves_strata_and_missingness():
    cases = [dict(sample_id=str(i), cohort='train' if i < 5 else 'dev', source='a') for i in range(9)]
    loc = {str(i): i % 9 for i in range(9)}; loc['4'] = None
    a, donors = shuffled_locations(cases, loc, 3)
    assert a == shuffled_locations(cases, loc, 3)[0]
    assert a['4'] is None
    for subset in [['0', '1', '2', '3'], ['5', '6', '7', '8']]:
        assert sorted(a[s] for s in subset) == sorted(loc[s] for s in subset)
        assert all(donors[s] != s and donors[s] in subset for s in subset)


def test_diagnostic_crop_geometry_and_positive_retention():
    from pathlib import Path
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
    from run_location_selection_v26 import geometry_bins
    c = {'image_size': [1200, 900]}
    geom = {'type': 'bbox', 'boxes': [[.5, .5, .6, .6]]}
    bins, centers, boxes = geometry_bins(c, geom, {'box': [600, 450, 1200, 900]})
    assert centers == {0} and bins == {0}
    np.testing.assert_allclose(boxes[0], [0, 0, .2, .2])
    assert geometry_bins(c, geom, {'box': [0, 0, 200, 200]})[0] == set()
    # Annotation crossing an exact third boundary retains points on either side.
    geom = {'type': 'bbox', 'boxes': [[1/3, .2, 2/3, .8]]}
    bins, _, _ = geometry_bins(c, geom)
    assert set(cell_ids([[400, 200], [800, 720]], c['image_size'])) <= bins
