"""Check the audit uses the published decoder and preserves whole-set shifts."""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from diagnose_phrase_token_oracle import inside, point_field, shifted_indices
from clickagent.phrase_grounder import decode_phrase
from clickagent.public_evaluation import point_in_geometry
from clickagent.spatial_geometry import shape_from_geometry


def test_every_parent_matches_published_integer_pixel_decoder():
    rng = np.random.default_rng(52); t, gh, gw = 3, 4, 7
    children = rng.normal(size=(t, gh, gw, 4)).astype('float32')
    offsets = rng.uniform(size=(t, 2, gh*2, gw*2)).astype('float32')
    field = point_field(children, offsets, [817, 613])
    for parent in range(gh*gw):
        scores = torch.zeros(t, gh, gw); scores.flatten(1)[:, parent] = 100
        output = dict(coarse_logits=scores, child_log_prob=torch.from_numpy(children), offsets=torch.from_numpy(offsets))
        for token in range(t):
            expected = decode_phrase(output, token, [817, 613], top_k=1)[0]['point']
            assert field[token, parent].tolist() == expected


def test_whole_set_translation_preserves_duplicates_and_is_invertible():
    parents = np.array([0, 0, 2, 7, 10, 11]); shifts = np.array([[1, 1], [2, 3]])
    moved = shifted_indices(parents, shifts, [3, 4])
    for shift, translated in zip(shifts, moved):
        restored = shifted_indices(translated, -shift[None], [3, 4])[0]
        np.testing.assert_array_equal(restored, parents)
        assert len(np.unique(translated)) == len(np.unique(parents))
        assert translated[0] == translated[1]


def test_vectorized_box_polygon_union_and_boundary_matches_primary_scorer():
    for geometry in [dict(type='bbox', boxes=[[.1, .2, .4, .6], [.7, .7, .9, .9]]),
                     dict(type='polygon', polygons=[[[.1, .2], [.4, .2], [.4, .6], [.1, .6]]])]:
        points = np.array([[100, 200], [250, 400], [400, 600], [800, 800], [0, 0], [500, 900]])
        expected = [point_in_geometry((p/1000).tolist(), geometry) for p in points]
        np.testing.assert_array_equal(inside(points, shape_from_geometry(geometry), [1000, 1000]), expected)
