import numpy as np
from clickagent.frozen_selection_v18 import METHODS, evidence, percentile, select


def test_empty_and_candidate_boundaries():
    assert select(np.zeros((0, 2)), np.zeros((0, 32)), [], [], [100, 100]) == {m: None for m in METHODS}
    points = np.array([[10, 10], [80, 80]])
    chosen = select(points, np.zeros((2, 32)), [0], [0., 2.], [100, 100])
    assert chosen['K16'] == 0 and chosen['union'] == 1
    assert all(i in [0, 1] for i in chosen.values())


def test_duplicate_points_in_one_view_do_not_count_as_independent_votes():
    points = np.array([[10, 10], [10, 10], [80, 80]])
    base = np.zeros((3, 32)); base[:, 0] = 1
    base[:, 1] = np.log1p([1, 2, 3]); base[:, 2] = 1/np.array([1., 2., 3.])
    support, _ = evidence(points, base, [100, 100])
    assert np.allclose(support, 1)
    base[0, 7] = 1; base[0, 8] = np.log1p(1); base[0, 9] = 1
    support, _ = evidence(points, base, [100, 100])
    assert np.allclose(support[:2], 2) and np.isclose(support[2], 1)


def test_ties_and_monotone_score_transform():
    assert np.allclose(percentile([2., 2., 2.]), .5)
    assert np.allclose(percentile([1, 1, 5]), [.25, .25, 1.])
    points = np.array([[10, 10], [50, 50], [80, 80]])
    base = np.zeros((3, 32))
    assert select(points, base, [0, 2], [1, 2, 3], [100, 100]) == select(points, base, [0, 2], [10, 100, 1000], [100, 100])
