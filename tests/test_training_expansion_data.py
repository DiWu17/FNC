from pathlib import Path
import importlib.util

import numpy as np
import pytest
from PIL import Image

from clickagent.training_expansion_data import (
    NearIndex, merge_annotations, pixel_box, reserve_groups, screenshot_identity, select_targets, local_image_path,
)
from clickagent.public_evaluation import point_in_geometry


def test_native_pixel_box_preserves_visible_target_and_rejects_invalid():
    box, clipped = pixel_box([-5, 20, 50, 80], [100, 200])
    assert box == [0, .1, .5, .4] and clipped
    assert point_in_geometry([.25, .25], {'type': 'bbox', 'boxes': [box]})
    for value in ([1, 2, 1, 4], [0, 0, float('nan'), 5], [101, 0, 120, 5], [True, 0, 2, 3], [0, 0, 1]):
        assert pixel_box(value, [100, 200]) is None


def test_duplicate_instruction_keeps_both_valid_regions_with_traceability():
    source = [dict(instruction='Save', bbox=[0, 0, 10, 10], source_record=1),
              dict(instruction='Save', bbox=[90, 90, 100, 100], source_record=2),
              dict(instruction='Save', bbox=[0, 0, 10, 10], source_record=3),
              dict(instruction='', bbox=[0, 0, 10, 10], source_record=4)]
    rows, stats = merge_annotations(source, [100, 100])
    assert len(rows) == 1 and len(rows[0]['boxes']) == 2
    geometry = dict(type='bbox', boxes=rows[0]['boxes'])
    assert point_in_geometry([.05, .05], geometry) and point_in_geometry([.95, .95], geometry)
    assert not point_in_geometry([.5, .5], geometry)
    assert {r['source_record'] for r in rows[0]['source_annotations']} == {1, 2, 3}
    assert stats['empty_instruction'] == 1


def test_case_sensitive_source_expressions_are_not_silently_combined():
    rows, _ = merge_annotations([dict(instruction='A', bbox=[0, 0, 1, 1]),
                                  dict(instruction='a', bbox=[5, 5, 6, 6])], [10, 10])
    assert len(rows) == 2


def test_rgb_identity_survives_reencoding_while_file_identity_changes(tmp_path):
    array = np.arange(12*10*3, dtype=np.uint8).reshape(10, 12, 3)
    image = Image.fromarray(array)
    a = tmp_path/'a.png'; b = tmp_path/'b.png'
    image.save(a, compress_level=0); image.save(b, compress_level=9)
    x, y = screenshot_identity(a), screenshot_identity(b)
    assert x['pixel_sha256'] == y['pixel_sha256'] and x['dhash256'] == y['dhash256']
    assert x['image_sha256'] != y['image_sha256']


def test_near_index_matches_independent_brute_force_at_radius_four():
    rng = np.random.default_rng(42)
    values = [int.from_bytes(rng.bytes(32), 'big') for _ in range(100)]
    tree = NearIndex()
    for i, value in enumerate(values):
        tree.add(value, i)
    queries = values[:10] + [v ^ 0b1111 for v in values[10:20]] + [int.from_bytes(rng.bytes(32), 'big') for _ in range(20)]
    for query in queries:
        matches = [i for i, value in enumerate(values) if (query ^ value).bit_count() <= 4]
        result = tree.find(query)
        assert (result is not None) == bool(matches)
        if result is not None:
            assert result in matches


def test_reserved_source_groups_are_repeatable_and_keep_training_groups():
    images = [dict(dataset=source, source_group=f'app{i}') for source in ('a', 'b') for i in range(20)]
    a = reserve_groups(images); b = reserve_groups(list(reversed(images)))
    assert a == b and len(a) == 4
    assert all(len([g for s, g in a if s == source]) == 2 for source in ('a', 'b'))


def test_target_subsample_avoids_repeated_same_region():
    rows = [dict(sample_id=str(i), target_geometry=dict(type='bbox', boxes=[box]))
            for i, box in enumerate([[0, 0, .1, .1], [0, 0, .1, .1], [.9, .9, 1, 1]])]
    result = select_targets(rows)
    assert len(result) == 2
    assert {tuple(r['target_geometry']['boxes'][0]) for r in result} == {(0, 0, .1, .1), (.9, .9, 1, 1)}


def test_website_group_keeps_browser_variants_together():
    path = Path(__file__).resolve().parents[1]/'scripts/acquire_training_expansion_v15.py'
    spec = importlib.util.spec_from_file_location('training_expansion_acquire', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    a = module.source_group('showui_web', 'images/002_Edge_techcrunch.com/0/a.png')
    b = module.source_group('showui_web', 'images/012_Chrome_techcrunch.com/1/b.png')
    assert a == b == 'techcrunch.com'
    assert module.source_group('amex', 'images/task-12.png') == module.source_group('amex', 'images/task-15.png')


def test_windows_invalid_application_name_is_mapped_without_changing_source(tmp_path):
    source='images/Code::Blocks/hash.png'
    result=local_image_path(tmp_path,source)
    assert result==tmp_path/'images/Code%3A%3ABlocks/hash.png'
    assert source=='images/Code::Blocks/hash.png'
    assert local_image_path(tmp_path,'images/OBS Studio/hash.png')==tmp_path/'images/OBS Studio/hash.png'


def test_percent_encoding_and_windows_device_names_do_not_collide(tmp_path):
    assert local_image_path(tmp_path,'images/a:b/hash.png')!=local_image_path(tmp_path,'images/a%3Ab/hash.png')
    assert local_image_path(tmp_path,'images/CON/hash.png')==tmp_path/'images/%43ON/hash.png'
    assert local_image_path(tmp_path,'images/dir./hash.png')==tmp_path/'images/dir%2E/hash.png'


def test_local_image_mapping_rejects_traversal_and_absolute_paths(tmp_path):
    for source in ('images/../escape.png','images//a.png','C:/outside.png','/images/outside.png'):
        with pytest.raises(ValueError):local_image_path(tmp_path,source)
