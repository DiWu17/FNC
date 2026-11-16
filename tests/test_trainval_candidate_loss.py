"""Geometry and quota accounting checks for the independent train/val analysis."""
import importlib.util
from pathlib import Path


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_trainval_candidate_loss.py"
spec = importlib.util.spec_from_file_location("trainval_loss", SCRIPT)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def result(points, scores=None):
    return dict(all_points=points, all_scores=scores or list(range(len(points), 0, -1)),
                parent_indices=list(range(len(points))), grid_shape=[10, 10])


def test_integer_mapping_and_stable_ties():
    transform = dict(box=[10, 20, 30, 40], scale=2, original_size=[100, 100])
    assert analysis.map_point([1, 1], transform) == [10, 20]
    assert analysis.map_point([3, 3], transform) == [12, 22]
    assert analysis.map_point([1000, -1000], transform) == [99, 0]
    values = analysis.ranked(result([[5, 5], [6, 6], [7, 7]], [1, 2, 2]), "global")
    assert [v["candidate_index"] for v in values] == [1, 2, 0]


def test_quota_duplicate_consumes_slot_and_round_robin_skips_empty_views():
    a = dict(point=[10, 10])
    b = dict(point=[20, 20])
    c = dict(point=[30, 30])
    ordered = analysis.rr([[a, c], [], [a, b]])
    assert ordered == [a, a, c, b]
    # Slice before dedup: the third local point cannot fill the duplicate slot.
    assert analysis.unique([b] + ordered[:2]) == [b, a]
    assert analysis.unique([b] + ordered) == [b, a, c]


def test_disjoint_target_union_does_not_hit_its_envelope():
    case = dict(image_size=[100, 100], target_geometry=dict(type="bbox", boxes=[
        [.1, .1, .2, .2], [.8, .8, .9, .9]]))
    assert not analysis.hit([50, 50], case)
    assert analysis.hit([10, 20], case)  # inclusive boundary convention
    polygon = dict(image_size=[100, 100], target_geometry=dict(type="polygon", polygons=[
        [[.1, .1], [.4, .1], [.1, .4]]]))
    assert analysis.hit([10, 20], polygon)
    assert not analysis.hit([35, 35], polygon)


def test_partial_named_crop_is_distinct_from_all_covered_crops_without_names():
    absent_named = dict(target_area_fraction=0, tokens=3)
    covered_no_name = dict(target_area_fraction=.1, tokens=0)
    covered_named = dict(target_area_fraction=.01, tokens=1)
    assert analysis.upstream_branch(0, []) == "global_no_name"
    assert analysis.upstream_branch(3, [absent_named]) == "no_crop_overlap"
    assert analysis.upstream_branch(3, [absent_named, covered_no_name]) == "overlap_all_no_name"
    assert analysis.upstream_branch(3, [covered_no_name, covered_named]) == "overlap_named_no_hit"


def test_reader_hashes_only_requested_npy_members_and_header(tmp_path):
    import hashlib
    import numpy as np
    archive = tmp_path / "raw.npz"
    np.savez_compressed(archive, target_attention=np.ones((0, 96, 4), np.float32), metadata=np.asarray("{}"))
    reader = analysis.Reader({str(archive): hashlib.sha256(archive.read_bytes()).hexdigest()})
    shape, dtype = reader.npy(archive, "target_attention", header_only=True)
    assert shape == (0, 96, 4) and dtype == np.dtype("float32")
    assert str(reader.npy(archive, "metadata")) == "{}"
    assert {v["mode"] for v in reader.members.values()} == {"header", "member"}
    assert "verified_full_sha256" not in next(iter(reader.files.values()))
    reader.unchanged()
    archive.write_bytes(b"changed")
    import pytest
    with pytest.raises(ValueError, match="Input changed"):
        reader.unchanged()
