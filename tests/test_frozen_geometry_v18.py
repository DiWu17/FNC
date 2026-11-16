from copy import deepcopy
from clickagent.frozen_geometry_v18 import relocate


def test_coordinates_change_inside_public_grid_and_input_is_immutable():
    r = dict(grid_shape=[2, 2], all_points=[[3, 4], [97, 97]], parent_indices=[0, 3],
        candidates=[dict(point=[3, 4], candidate_index=0), dict(point=[97, 97], candidate_index=1)],
        point=[3, 4], all_scores=[4, 2])
    saved = deepcopy(r)
    assert relocate(r, [100, 100], 'coarse_center')['all_points'] == [[25, 25], [75, 75]]
    assert relocate(r, [100, 100], 'half_to_center')['all_points'] == [[14, 14], [86, 86]]
    assert relocate(r, [100, 100], 'child_center')['all_points'] == [[12, 12], [88, 88]]
    assert r == saved and relocate(r, [100, 100], 'original') == r


def test_empty_and_clipped_extreme_points():
    r = dict(grid_shape=[1, 1], all_points=[], parent_indices=[], candidates=[], point=None)
    assert relocate(r, [1, 1], 'coarse_center') == r
    r.update(all_points=[[0, 0]], parent_indices=[0], candidates=[dict(point=[0, 0], candidate_index=0)])
    assert relocate(r, [1, 1], 'coarse_center')['all_points'] == [[0, 0]]
