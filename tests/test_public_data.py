"""Offline contract tests: geometry, refusal preservation and trusted bytes."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from types import SimpleNamespace
import tarfile
import zipfile

from PIL import Image
import pytest

import clickagent.public_data as public


def _dump(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (20, 10), (70, 120, 180)).save(path)


@pytest.mark.parametrize("name", ["../outside", "a/../../outside", "C:/outside", "\\\\server\\share\\outside", "/absolute"])
def test_paths_cannot_escape_destination(tmp_path, name):
    with pytest.raises(ValueError):
        public.safe_path(tmp_path, name)


def test_zip_rejects_traversal_before_extracting_safe_members(tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("safe.txt", "safe")
        output.writestr("../outside.txt", "outside")
    with pytest.raises(ValueError):
        public.safe_extract(archive, tmp_path / "extracted")
    assert not (tmp_path / "outside.txt").exists()
    assert not (tmp_path / "extracted/safe.txt").exists()


def test_zip_and_tar_symlinks_are_rejected(tmp_path):
    archive = tmp_path / "links.zip"
    info = zipfile.ZipInfo("link")
    info.create_system = 3
    info.external_attr = 0o120777 << 16
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(info, "../outside")
    with pytest.raises(ValueError, match="symlink"):
        public.safe_extract(archive, tmp_path / "zip")
    archive = tmp_path / "links.tar"
    with tarfile.open(archive, "w") as output:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "../outside"
        output.addfile(info)
    with pytest.raises(ValueError, match="link/special"):
        public.safe_extract(archive, tmp_path / "tar")


def test_regular_archive_files_extract_exactly(tmp_path):
    archive = tmp_path / "good.zip"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr("sub/a.txt", "hello")
    assert public.safe_extract(archive, tmp_path / "output") == ["sub/a.txt"]
    assert (tmp_path / "output/sub/a.txt").read_bytes() == b"hello"


def test_resume_verifies_published_hash_and_cached_receipt(tmp_path, monkeypatch):
    target = tmp_path / "file.bin"
    target.with_name("file.bin.part").write_bytes(b"a")
    calls = []
    def run(arguments, **kwargs):
        calls.append(arguments)
        assert "--continue-at" in arguments
        Path(arguments[arguments.index("--output") + 1]).write_bytes(b"abc")
        return SimpleNamespace(returncode=0, stdout="https://official.example/file", stderr="")
    monkeypatch.setattr(public.subprocess, "run", run)
    expected = hashlib.sha256(b"abc").hexdigest()
    git = hashlib.sha1(b"blob 3\0abc").hexdigest()
    record = public.download_file(["https://official.example/file"], target, size=3, sha256=expected, git_sha1=git)
    assert target.read_bytes() == b"abc"
    assert record["sha256"] == expected
    assert record["git_sha1"] == git
    assert public.download_file(["https://official.example/file"], target, size=3, sha256=expected, git_sha1=git)["cached"]
    assert len(calls) == 1


def test_mismatched_download_is_quarantined_not_promoted(tmp_path, monkeypatch):
    target = tmp_path / "file.bin"
    def run(arguments, **kwargs):
        Path(arguments[arguments.index("--output") + 1]).write_bytes(b"incorrect")
        return SimpleNamespace(returncode=0, stdout="https://mirror.example/file", stderr="")
    monkeypatch.setattr(public.subprocess, "run", run)
    with pytest.raises(RuntimeError, match="mismatch"):
        public.download_file(["https://mirror.example/file"], target, size=9, sha256="0" * 64, retries=1)
    assert not target.exists()
    assert list(tmp_path.glob("*.bad-*"))


def test_osworld_preserves_polygon_refusal_and_refined_annotation(tmp_path, monkeypatch):
    raw = tmp_path / "raw/osworld_g/benchmark"
    _image(raw / "images/a.png")
    common = {"image_path": "a.png", "image_size": [20, 10], "GUI_types": ["Icon"]}
    rows = [{**common, "id": "a-0", "instruction": "Original bbox instruction.", "box_type": "bbox", "box_coordinates": [2, 1, 4, 2]},
            {**common, "id": "a-1", "instruction": "Polygon instruction.", "box_type": "polygon", "box_coordinates": [0, 0, 10, 0, 10, 5]},
            {**common, "id": "a-2", "instruction": "Impossible instruction.", "box_type": "refusal", "box_coordinates": [0, 0, 0, 0]}]
    _dump(raw / "OSWorld-G.json", rows)
    _dump(raw / "OSWorld-G_refined.json", [{**row, "instruction": "Refined text"} for row in rows])
    monkeypatch.setitem(public.EXPECTED_SAMPLE_COUNTS, "osworld_g", 3)
    manifest = public.normalize_public_dataset("osworld_g", tmp_path)
    result = public.load_public_manifest(manifest)
    assert len(result) == 3
    assert result[0]["instruction"] == rows[0]["instruction"]
    assert result[0]["target_geometry"] == {"type": "bbox", "boxes": [[0.1, 0.1, 0.3, 0.3]]}
    assert result[1]["target_geometry"] == {"type": "polygon", "polygons": [[[0.0, 0.0], [0.5, 0.0], [0.5, 0.5]]]}
    assert result[2]["expected_actionable"] is False
    assert result[2]["task_kind"] == "refusal"
    assert result[2]["target_geometry"] is None
    assert len({row["group_id"] for row in result}) == 1
    assert result[0]["metadata"]["refined_annotation"]["instruction"] == "Refined text"


def test_screenspot_pro_multibox_and_same_basename_distinct_images(tmp_path, monkeypatch):
    raw = tmp_path / "raw/screenspot_pro"
    for name in ("appA/same.png", "appB/same.png"):
        _image(raw / "images" / name)
    rows = [{"id": "one", "img_filename": "appA/same.png", "img_size": [20, 10], "bbox": [2, 1, 4, 2], "instruction": "one"},
            {"id": "two", "img_filename": "appB/same.png", "img_size": [20, 10], "bbox": [[0, 0, 2, 2], [10, 5, 20, 10]], "instruction": "two"}]
    _dump(raw / "annotations/app.json", rows)
    _dump(raw / "annotations/app.json.download.json", {"receipt": "not an annotation"})
    monkeypatch.setitem(public.EXPECTED_SAMPLE_COUNTS, "screenspot_pro", 2)
    result = public.load_public_manifest(public.normalize_public_dataset("screenspot_pro", tmp_path))
    assert result[0]["target_geometry"]["boxes"] == [[0.1, 0.1, 0.2, 0.2]]
    assert result[1]["target_geometry"]["boxes"] == [[0.0, 0.0, 0.1, 0.2], [0.5, 0.5, 1.0, 1.0]]
    assert result[0]["group_id"] != result[1]["group_id"]


def test_venus_keeps_refusals_bilingual_original_and_missing_images(tmp_path, monkeypatch):
    raw = tmp_path / "raw/venusbench_gd"
    _image(raw / "images/mobile/app/a.png")
    _dump(raw / "meta.json", [{"img_filename": "mobile/app/a.png", "platform": "mobile", "application": "app"}])
    _dump(raw / "instruction/element_grounding.json", [{"img_filename": "mobile/app/a.png", "instruction": "Find a", "instruction_zh": "找到甲", "bbox": [2, 1, 4, 2]}])
    _dump(raw / "instruction/refusal_grounding.json", [{"img_filename": "mobile/app/missing.png", "instruction": "Find an absent object"}])
    monkeypatch.setitem(public.EXPECTED_SAMPLE_COUNTS, "venusbench_gd", 2)
    result = public.load_public_manifest(public.normalize_public_dataset("venusbench_gd", tmp_path))
    assert len(result) == 2
    assert result[0]["metadata"]["original"]["instruction_zh"] == "找到甲"
    assert result[1]["task_kind"] == "refusal"
    assert result[1]["expected_actionable"] is False
    assert result[1]["metadata"]["image_status"] == "missing"
    audit = json.loads((tmp_path / "audits/venusbench_gd.json").read_text())
    assert audit["samples"] == 2
    assert audit["images"] == {"ok": 1, "missing": 1}
    assert len(audit["image_issues"]) == 1


def test_conflict_uses_clean_reference_for_negative_click_eligibility(tmp_path, monkeypatch):
    raw = tmp_path / "raw/conflictgui"
    _image(raw / "images/a.png")
    base = {"task_id": 1, "dataset": "AMEX", "images": ["images/a.png"], "screenshot": "images/a.png", "imgwidth": 20, "imgheight": 10}
    rows = [{**base, "sample_id": "a:clean", "instruction": "click", "data_type": "clean", "split": "test", "action": {"action_type": "click", "x": 10, "y": 5}, "clean_reference_action": {"action_type": "click", "x": 10, "y": 5}},
            {**base, "sample_id": "a:conflict", "instruction": "contradiction", "data_type": "conflict2", "split": "test", "action": {"action_type": "terminate", "status": "failure"}, "clean_reference_action": {"action_type": "click", "x": 10, "y": 5}},
            {**base, "sample_id": "b:clean", "instruction": "type", "data_type": "clean", "split": "calibration", "action": {"action_type": "type", "text": "hello"}, "clean_reference_action": {"action_type": "type", "text": "hello"}}]
    _dump(raw / "data/conflictgui.json", rows)
    _dump(raw / "data/test.json", rows[:2])
    _dump(raw / "data/calibration.json", rows[2:])
    monkeypatch.setitem(public.EXPECTED_SAMPLE_COUNTS, "conflictgui", 3)
    result = public.load_public_manifest(public.normalize_public_dataset("conflictgui", tmp_path))
    assert len(result) == 3
    assert all(row["task_kind"] == "action_feasibility" and row["target_geometry"] is None for row in result)
    assert result[0]["metadata"]["point_supported"] is True
    assert result[1]["metadata"]["reference_action_type"] == "click"
    assert result[1]["metadata"]["current_action_type"] == "terminate"
    assert result[1]["metadata"]["click_compatible"] is True
    assert result[1]["expected_actionable"] is False
    assert result[1]["metadata"]["reference_point_valid_for_instruction"] is False
    assert result[0]["metadata"]["reference_point_normalized"] == [0.5, 0.5]
    assert result[0]["metadata"]["reference_point_tolerance"]["max_distance"] == 0.14
    assert result[2]["expected_actionable"] is True
    assert result[2]["metadata"]["unsupported_for_point"] is True
    assert result[2]["split"] == "calibration"


def test_bad_images_are_explicit_and_not_dropped(tmp_path):
    bad = tmp_path / "bad.png"
    bad.write_bytes(b"not an image")
    row = public._record("osworld_g", "bad", 0, bad, "instruction", None, True, {})
    assert row["metadata"]["image_status"] == "corrupt"
    assert row["metadata"]["image_error"]
    assert row["instruction"] == "instruction"


def test_visible_box_intersection_is_explicit_and_preserves_original():
    original = {"type": "bbox", "boxes": [[-0.01, 0.1, 1.01, 0.8]]}
    row = {"target_geometry": original, "metadata": {"point_supported": True}}
    public._visible_box_geometry(row)
    assert row["target_geometry"] == {"type": "bbox", "boxes": [[0.0, 0.1, 1.0, 0.8]]}
    assert row["metadata"]["raw_normalized_target_geometry"] == original
    assert row["metadata"]["geometry_adjustment"] == "intersection_with_visible_image"
    assert row["metadata"]["geometry_visible"] is True


def test_complete_interrupted_download_is_verified_without_network(tmp_path, monkeypatch):
    destination = tmp_path / "completed.png"
    part = destination.with_name(destination.name + ".part")
    content = b"complete interrupted transfer"
    part.write_bytes(content)
    digest = hashlib.sha256(content).hexdigest()
    def no_network(*args, **kwargs):
        pytest.fail("A hash-verified complete .part must not make a range request at EOF")
    monkeypatch.setattr(public.subprocess, "run", no_network)
    result = public.download_file(["https://official.test/file.png"], destination, size=len(content), sha256=digest)
    assert destination.read_bytes() == content
    assert not part.exists()
    assert result["recovery"] == "verified_local_file_against_published_hash"
    assert result["sha256"] == digest


@pytest.mark.parametrize("geometry", [
    {"type": "bbox", "boxes": [[0, 0, 1.01, 1]]},
    {"type": "bbox", "boxes": [[0.5, 0, 0.5, 1]]},
    {"type": "bbox", "boxes": [[0, 0, float("nan"), 1]]},
    {"type": "polygon", "polygons": [[[0, 0], [0.5, 0.5]]]},
    {"type": "polygon", "polygons": [[[0, 0], [1, 0], [0, 1.01]]]},
])
def test_canonical_geometry_validation_rejects_invalid(geometry):
    with pytest.raises(ValueError, match="Invalid canonical geometry"):
        public.validate_public_geometry([{"sample_id": "invalid", "target_geometry": geometry}])


def test_repackaged_mirror_only_promotes_bytes_matching_official_inventory(tmp_path, monkeypatch):
    import sys
    from types import ModuleType
    payload = b"exact original image payload"
    expected = hashlib.sha256(payload).hexdigest()
    inventory = {"sha": "official-pinned-revision", "siblings": [
        {"rfilename": "images/app/good.png", "lfs": {"sha256": expected}},
        {"rfilename": "images/app/bad.png", "lfs": {"sha256": expected}},
    ]}
    monkeypatch.setattr(public, "_inventory", lambda dataset, root: inventory)
    shards = [{"Type": "blob", "Path": f"data/shard-{i}.parquet", "Size": 9,
               "Sha256": hashlib.sha256(b"container").hexdigest(), "Revision": "mirror-revision"}
              for i in range(8)]
    _dump(tmp_path / "discovery/pro_modelscope.response", {"Data": {"Files": shards}})
    rows = [{"img_filename": "app/good.png", "image": {"bytes": payload}},
            {"img_filename": "app/bad.png", "image": {"bytes": b"changed payload"}},
            {"img_filename": "app/unknown.png", "image": {"bytes": payload}}]
    parquet = ModuleType("pyarrow.parquet")
    parquet.ParquetFile = lambda path: SimpleNamespace(iter_batches=lambda **kwargs: iter([
        SimpleNamespace(to_pylist=lambda: rows)]))
    arrow = ModuleType("pyarrow")
    arrow.parquet = parquet
    monkeypatch.setitem(sys.modules, "pyarrow", arrow)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", parquet)
    def fake_transfer(urls, destination, **kwargs):
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(b"container")
        return {"url": urls[0], "effective_origin": "modelscope.cn", "path": str(destination),
                "sha256": kwargs["sha256"]}
    monkeypatch.setattr(public, "download_file", fake_transfer)
    audit = public.recover_screenspot_pro_from_modelscope(tmp_path, workers=2)
    good = tmp_path / "raw/screenspot_pro/images/app/good.png"
    assert good.read_bytes() == payload
    assert not (good.parent / "bad.png").exists()
    assert not (good.parent / "unknown.png").exists()
    assert audit["unique_matched_images"] == 1
    assert audit["recovered_images"] == ["images/app/good.png"]
    assert len(audit["rejected_images"]) == 16
    receipt = json.loads(good.with_name(good.name + ".download.json").read_text(encoding="utf-8"))
    assert receipt["published_sha256"] == expected
    assert receipt["official_hf_revision"] == inventory["sha"]
    assert receipt["recovery"] == "image_bytes_in_hash_verified_mirror_parquet"
