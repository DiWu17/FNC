from copy import deepcopy

import pytest

from clickagent.public_refinement import crop_box, policy_pair, resolve_point


def native(text="CLICK 500, 500", **kwargs):
    return {"generated_text": text, "terminated_by_eos": True, "truncated": False, **kwargs}


def test_crop_grid_interior_and_clipped_edges():
    assert crop_box(5 * 21 + 10, [12, 21], [1920, 1080]) == [731, 270, 1189, 720]
    assert crop_box(0, [12, 21], [1920, 1080]) == [0, 0, 275, 270]
    assert crop_box(251, [12, 21], [1920, 1080]) == [1645, 810, 1920, 1080]
    with pytest.raises(ValueError):
        crop_box(252, [12, 21], [1920, 1080])


def test_local_coordinates_map_back_without_bbox_snapping():
    box = [100, 200, 301, 401]
    assert resolve_point(native(), box, [101, 201]) == ([200, 300], "refined")
    assert resolve_point(native("CLICK 0 1000\n"), box, [101, 201]) == ([100, 400], "refined")
    assert resolve_point(native("CLICK 1000, 0"), box, [101, 201]) == ([300, 200], "refined")


@pytest.mark.parametrize("result", [native("WAIT"), native("CLICK 500, 500 extra"),
    native("CLICK 500.0 500"), native("CLICK 1001 500"),
    native(truncated=True), native(terminated_by_eos=False), native(error="failure")])
def test_wait_incomplete_and_invalid_outputs_preserve_old_point(result):
    point, status = resolve_point(result, [100, 200, 301, 401], [120, 220])
    assert point == [120, 220]
    assert status.startswith("fallback_")


@pytest.mark.parametrize("action", [{"type": "wait"}, {"type": "click", "x": 10, "y": 20}])
def test_refinement_never_changes_gate_or_old_cache(action):
    original = {"method": "prefill_hidden", "action": action, "parse_valid": True,
                "click_probability": .8, "gate_threshold": .66,
                "localization_point": [10, 20], "grid_shape": [12, 21]}
    saved = deepcopy(original)
    before, after = policy_pair(original, [11, 19])
    assert original == saved
    assert before["action"]["type"] == after["action"]["type"]
    assert before["click_probability"] == after["click_probability"] == .8
    assert before["gate_threshold"] == after["gate_threshold"] == .66
    assert after["localization_point"] == [11, 19]
    assert "grid_shape" not in after  # Old center-only oracle no longer applies.
    if action["type"] == "click":
        assert after["action"] == {"type": "click", "x": 11, "y": 19}
