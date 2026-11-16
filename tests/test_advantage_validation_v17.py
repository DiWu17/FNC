from copy import deepcopy
from inspect import signature

import numpy as np
from PIL import Image
import pytest

from clickagent.advantage_validation import balanced_cases, map_to_original, nested_groups, paired_interval, perturb
from clickagent.advantage_runtimes import AimaAdapter, MVPAdapter, OurRuntime, QwenCoordinateAdapter, TAGAdapter, UITarsAdapter


def test_nested_groups_keep_every_query_from_same_screenshot():
    entries = [dict(sample_id=f'{s}-{g}-{q}', group=f'{s}-{g}', source=s, arbitrary_label=q % 2)
               for s in ['a', 'b'] for g in range(10) for q in range(g % 3 + 1)]
    original = deepcopy(entries)
    cohorts = [nested_groups(entries, f) for f in [.1, .25, .5, 1.]]
    for rows in cohorts:
        groups = {r['group'] for r in rows}
        assert rows == [e for e in entries if e['group'] in groups]
    assert all({r['group'] for r in a} <= {r['group'] for r in b} for a, b in zip(cohorts, cohorts[1:]))
    assert cohorts[-1] == entries and entries == original


def test_shared_rgb_group_across_sources_is_never_split():
    entries = [dict(sample_id=f'{i}', group='same', source=s) for i, s in enumerate(['a', 'b'])]
    assert nested_groups(entries, .1) == entries


@pytest.mark.parametrize('fraction', [0, -1, 1.01])
def test_invalid_fraction(fraction):
    with pytest.raises(ValueError):
        nested_groups([], fraction)


def test_sampling_does_not_depend_on_labels_or_input_order():
    rows = [dict(sample_id=f'{s}-{i}', dataset=s, hit=i % 2) for s in ['a', 'b'] for i in range(10)]
    selected = balanced_cases(rows, 3, 'example')
    changed = [{**r, 'hit': not r['hit']} for r in reversed(rows)]
    assert [r['sample_id'] for r in selected] == [r['sample_id'] for r in balanced_cases(changed, 3, 'example')]


@pytest.mark.parametrize('condition', ['clean', 'jpeg70', 'border24'])
def test_perturbations_preserve_original_coordinate_correspondence(condition):
    image = Image.fromarray(np.arange(80 * 100 * 3, dtype=np.uint8).reshape(80, 100, 3))
    changed, affine = perturb(image, condition)
    sx, sy, tx, ty = affine
    for point in [[0, 0], [99, 79], [30, 40]]:
        transformed = [point[0] * sx + tx, point[1] * sy + ty]
        assert map_to_original(transformed, affine, image.size) == point
    assert changed.size == ((148, 128) if condition == 'border24' else image.size)


def test_border_predictions_are_not_clamped_into_valid_targets():
    assert map_to_original([0, 0], (1, 1, 24, 24), (100, 80)) is None
    assert map_to_original([124, 104], (1, 1, 24, 24), (100, 80)) is None
    assert map_to_original(None, (1, 1, 24, 24), (100, 80)) is None


def test_intervals_use_one_seed_averaged_value_per_screenshot():
    r = paired_interval([1 / 3, 2 / 3, 1., 0.], [0., 0., 1., 0.], ['a', 'a', 'b', 'b'], reps=100)
    assert r['n'] == 4 and r['delta'] == pytest.approx(.25)
    with pytest.raises(ValueError):
        paired_interval(np.ones((3, 4)), np.zeros((3, 4)), ['a'] * 4)


@pytest.mark.parametrize('cls', [OurRuntime, QwenCoordinateAdapter, UITarsAdapter, TAGAdapter, AimaAdapter, MVPAdapter])
def test_adapters_cannot_accept_geometry_or_labels(cls):
    assert list(signature(cls.infer).parameters) == ['self', 'image', 'instruction']
