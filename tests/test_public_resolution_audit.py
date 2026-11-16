import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from clickagent.action_grounding import _pixel_centers
import clickagent.public_resolution_audit as audit
from clickagent.public_evaluation import point_in_geometry, quantization_oracle


@pytest.fixture
def model_config(tmp_path):
    root = tmp_path / "model"
    root.mkdir()
    processor = {"size": {"shortest_edge": 65536, "longest_edge": 16777216},
                 "patch_size": 16, "merge_size": 2, "temporal_patch_size": 2,
                 "processor_class": "Qwen3VLProcessor", "image_processor_type": "Qwen2VLImageProcessorFast"}
    model = {"model_type": "qwen3_vl", "vision_config": {"patch_size": 16, "spatial_merge_size": 2,
                                                              "temporal_patch_size": 2}}
    (root / "preprocessor_config.json").write_text(json.dumps(processor), encoding="utf-8")
    (root / "config.json").write_text(json.dumps(model), encoding="utf-8")
    return root


def test_checkpoint_configuration_read_and_runtime_overrides(model_config):
    config = audit.read_resize_config(model_config)
    assert config["factor"] == 32 and config["patch_size"] == 16 and config["merge_size"] == 2
    assert config["original_processor_size"] == {"shortest_edge": 65536, "longest_edge": 16777216}
    assert config["effective_size"] == {"shortest_edge": 16384, "longest_edge": 262144}
    assert all(len(item["sha256"]) == 64 for item in config["config_files"])
    bad = json.loads((model_config / "config.json").read_text(encoding="utf-8"))
    bad["vision_config"]["spatial_merge_size"] = 1
    (model_config / "config.json").write_text(json.dumps(bad), encoding="utf-8")
    with pytest.raises(ValueError, match="disagree"):
        audit.read_resize_config(model_config)


@pytest.mark.parametrize("size", [(960, 540), (1920, 1080), (1170, 2532), (512, 512), (32, 32)])
def test_official_smart_resize_grid_and_integer_center_oracle(model_config, size):
    config = audit.read_resize_config(model_config)
    result = audit.native_grid(size, config)
    rw, rh = result["resized_image_size"]
    assert result["grid_shape"] == [rh // 32, rw // 32]
    assert rw % 32 == rh % 32 == 0
    assert result["image_tokens"] == np.prod(result["grid_shape"])
    x, y = _pixel_centers(result["grid_shape"], size)
    # A tiny region enclosing one actual rounded decoder output is reachable.
    px, py = x[len(x) // 2] / size[0], y[len(y) // 2] / size[1]
    geometry = {"type": "bbox", "boxes": [[max(0, px - 1e-8), max(0, py - 1e-8),
                                                min(1, px + 1e-8), min(1, py + 1e-8)]]}
    expected = any(point_in_geometry((a / size[0], b / size[1]), geometry) for a, b in zip(x, y))
    assert quantization_oracle(geometry, result["grid_shape"], size) is expected is True


def test_geometry_uses_unions_polygons_and_pixel_dimensions():
    geometry = {"type": "bbox", "boxes": [[0, 0, .01, .01], [.04, .04, .06, .06]]}
    assert quantization_oracle(geometry, (10, 10), (320, 320)) is True
    dimensions = audit.target_pixel_dimensions(geometry, (320, 320))
    assert len(dimensions["components"]) == 2
    assert dimensions["representative_width_px"] == pytest.approx(6.4)
    assert dimensions["short_edge_bin_px"] == "lt8"
    # Its enclosing rectangle contains (.15,.15), but this triangle does not.
    triangle = {"type": "polygon", "polygons": [[[.14, .14], [.16, .14], [.14, .15]]]}
    assert quantization_oracle(triangle, (10, 10), (320, 320)) is False
    dimensions = audit.target_pixel_dimensions(triangle, (320, 320))
    assert dimensions["representative_area_px2"] == pytest.approx(.0001 * 320 * 320)


def test_full_manifest_denominators_and_exclusions(tmp_path, model_config):
    def case(sid, geometry, positive=True, kind="grounding", domain="desktop"):
        return {"sample_id": sid, "dataset": "Example", "instruction": "Open control", "domain": domain,
                "expected_actionable": positive, "task_kind": kind, "target_geometry": geometry,
                "image_size": [320, 320], "image": "not_needed_for_dimension_only_audit.png", "split": "test"}
    cases = [case("hit", {"type": "bbox", "boxes": [[.04, .04, .06, .06]]}),
             case("miss", {"type": "bbox", "boxes": [[0, 0, .01, .01]]}),
             case("polygon_miss", {"type": "polygon", "polygons": [[[.14, .14], [.16, .14], [.14, .15]]]}, domain="web"),
             case("refusal", None, False, "refusal"),
             case("conflict", None, True, "action_feasibility")]
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in cases), encoding="utf-8")
    summary = audit.audit_manifests([manifest], tmp_path / "output", model_config, processor_checks=0)
    assert summary["overall"]["input_rows"] == 5
    assert summary["overall"]["quantization_oracle"] == {"numerator": 1, "denominator": 3, "rate": 1 / 3}
    assert summary["overall"]["excluded_reasons"] == {"nonactionable_or_refusal": 1, "action_feasibility_no_geometry_protocol": 1}
    assert summary["by_dataset"]["Example"]["by_domain"]["desktop"]["quantization_oracle"]["denominator"] == 2
    assert summary["model_weights_loaded"] is False and summary["gpu_used"] is False
    assert summary["full_supplied_manifest"] is True and summary["processor_checks"] == []
    exported = [json.loads(line) for line in (tmp_path / "output/per_sample.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(exported) == 5 and exported[2]["quantization_oracle"] is False
    assert json.loads((tmp_path / "output/summary.json").read_text(encoding="utf-8"))["status"] == "complete"
    with pytest.raises(ValueError, match="duplicate sample_id"):
        audit.audit_manifests([manifest, manifest], tmp_path / "duplicate", model_config, processor_checks=0)


def test_visible_domain_clipping_preserves_original_and_invalid_denominators(tmp_path, model_config):
    def case(sid, geometry):
        return {"sample_id": sid, "dataset": "Example", "instruction": "Open control", "domain": "web",
                "expected_actionable": True, "task_kind": "grounding", "target_geometry": geometry,
                "image_size": [320, 320], "image": "not_loaded.png"}
    original = {"type": "bbox", "boxes": [[.9, .1, 1.0004237, .2]]}
    cases = [case("clipped", original),
             case("empty", {"type": "bbox", "boxes": [[1.1, .1, 1.2, .2]]}),
             case("polygon_invalid", {"type": "polygon", "polygons": [[[-.1, 0], [.1, 0], [.1, .1]]]})]
    manifest = tmp_path / "manifest.jsonl"
    before = "".join(json.dumps(row) + "\n" for row in cases)
    manifest.write_text(before, encoding="utf-8")
    summary = audit.audit_manifests([manifest], tmp_path / "output", model_config, processor_checks=0)
    assert manifest.read_text(encoding="utf-8") == before
    assert summary["n_geometry_clipped"] == 2
    assert summary["geometry_clipped_sample_ids"] == ["clipped", "empty"]
    overall = summary["overall"]
    assert overall["quantization_oracle"] == {"numerator": 1, "denominator": 3, "rate": 1 / 3}
    assert overall["quantization_oracle_upper_bound"] == {"numerator": 2, "denominator": 3, "rate": 2 / 3}
    assert overall["quantization_oracle_coverage"] == 2 / 3
    assert overall["unresolved_positive_geometry_rows"] == 1
    rows = [json.loads(line) for line in (tmp_path / "output/per_sample.jsonl").read_text(encoding="utf-8").splitlines()]
    assert rows[0]["original_target_geometry"] == original
    assert rows[0]["effective_target_geometry"] == {"type": "bbox", "boxes": [[.9, .1, 1, .2]]}
    assert rows[1]["geometry_empty_intersection"] is True and rows[1]["quantization_oracle"] is False
    assert rows[2]["quantization_oracle"] is None and rows[2]["oracle_eligible"] is True
    for x in np.linspace(0, 1, 101):
        assert point_in_geometry((x, .15), original) == point_in_geometry((x, .15), rows[0]["effective_target_geometry"])


def test_boundary_only_intersection_is_invalid_not_proven_empty():
    # Inclusive x=0 can be a real integer decoder output on a tiny image.
    original = {"type": "bbox", "boxes": [[-.1, .1, 0, .2]]}
    geometry, clipped, error, empty = audit._visible_geometry(original)
    assert geometry is None and clipped and error and not empty


def test_actual_local_cpu_image_processor_matches_smart_resize(tmp_path):
    model_path = Path(__file__).resolve().parents[1] / "models/Qwen3-VL-8B-Instruct"
    if not model_path.joinpath("preprocessor_config.json").is_file():
        pytest.skip("local Qwen processor configuration unavailable")
    config = audit.read_resize_config(model_path)
    path = tmp_path / "screenshot.png"
    Image.new("RGB", (480, 270), "white").save(path)
    checks = audit.verify_processor_grids([({"sample_id": "cpu", "image_size": [480, 270], "image": str(path)},
                                            tmp_path / "manifest.jsonl")], config, count=1)
    assert len(checks) == 1 and checks[0]["match"] is True and checks[0]["device"] == "cpu"
    assert checks[0]["expected_postmerge_grid"] == audit.native_grid((480, 270), config)["grid_shape"]

