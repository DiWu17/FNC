"""Offline selection invariants; no real models or public images are loaded."""
import hashlib
import importlib.util
from pathlib import Path
import random

from PIL import Image


SPEC = importlib.util.spec_from_file_location("semantic_data_prepare", Path(__file__).parents[1] / "scripts/prepare_semantic_head_data.py")
data = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(data)


def row(sid, image, bounds, source="Salesforce/toy"):
    return {"sample_id": sid, "dataset": source, "instruction": f"Click target {sid}",
            "source_split": "train", "official_training_split": True, "expected_actionable": True,
            "pixel_sha256": image, "dhash256": hashlib.sha256(image.encode()).hexdigest(),
            "target_geometry": {"type": "bbox", "boxes": [bounds]}}


def test_hamming_radius_search_is_exact_including_boundary():
    rng = random.Random(9)
    values = [rng.getrandbits(256) for _ in range(200)]
    index = data.HammingIndex(values)
    assert index.near(values[0] ^ 0b1111)
    assert not index.near(values[0] ^ 0b11111)
    queries = [rng.getrandbits(256) for _ in range(50)] + [v ^ 0b1111 for v in values[:10]]
    for query in queries:
        assert index.near(query) == any((query ^ value).bit_count() <= 4 for value in values)


def test_old_image_cap_new_image_pairs_and_odd_budget_not_filled_with_singleton():
    base = row("base", "old", [0, 0, 0.1, 0.1])
    pool = [base] + [row(f"extra{i}", "old", [0.15 + i * 0.15, 0, 0.25 + i * 0.15, 0.1]) for i in range(5)]
    for image in ("new1", "new2"):
        pool += [row(f"{image}:a", image, [0, 0, 0.1, 0.1]), row(f"{image}:b", image, [0.5, 0.5, 0.6, 0.6])]
    args = ([base], pool, {"base"}, set(), data.HammingIndex(), {"old"}, data.HammingIndex([int(base["dhash256"], 16)]))
    selected, origins, stats = data.choose_extensions(*args, budget=9, per_old_image=4, seed=15)
    assert len(selected) == 8 and stats["unfilled_query_budget"] == 1
    assert sum(origins[r["sample_id"]] == "existing_train_image" for r in selected) == 4
    assert {r["sample_id"] for r in selected} <= {r["sample_id"] for r in pool}
    assert {r["sample_id"] for r in selected} == {r["sample_id"] for r in data.choose_extensions(*args, budget=9, per_old_image=4, seed=15)[0]}
    for image in ("new1", "new2"):
        assert sum(r["pixel_sha256"] == image for r in selected) == 2


def test_protected_exact_and_near_images_and_prior_ids_cannot_be_added():
    base = row("base", "old", [0, 0, 0.1, 0.1])
    old_extra = row("prior", "old", [0.5, 0.5, 0.6, 0.6])
    new = [row("new:a", "new", [0, 0, 0.1, 0.1]), row("new:b", "new", [0.5, 0.5, 0.6, 0.6])]
    protected = data.HammingIndex([int(new[0]["dhash256"], 16) ^ 0b1111])
    selected, _, _ = data.choose_extensions([base], [base, old_extra, *new], {"base", "prior"}, set(), protected,
                                            {"old"}, data.HammingIndex(), budget=20)
    assert selected == []
    selected, _, _ = data.choose_extensions([base], [base, old_extra], {"base"}, {"old"}, data.HammingIndex(),
                                            {"old"}, data.HammingIndex(), budget=20)
    assert selected == []


def test_nonoverlap_and_real_official_annotations_only():
    a = row("a", "image", [0, 0, 0.3, 0.3])
    overlap = row("b", "image", [0.2, 0.2, 0.4, 0.4])
    alias = {**row("c", "image", [0.5, 0.5, 0.6, 0.6]), "instruction": "  CLICK TARGET A "}
    assert not data.distinct(a, overlap)
    assert not data.distinct(a, alias)
    assert not data.official_candidate({**a, "source_split": "val"})
    assert not data.official_candidate({**a, "official_training_split": False})


def test_pixel_hash_detects_same_pixels_with_different_png_encoding(tmp_path):
    image = Image.new("RGB", (32, 32), (80, 110, 140))
    first, second = tmp_path / "first.png", tmp_path / "second.png"
    image.save(first, compress_level=0)
    image.save(second, compress_level=9)
    a, b = data.image_info(first), data.image_info(second)
    assert a["file_sha256"] != b["file_sha256"]
    assert a["pixel_sha256"] == b["pixel_sha256"]
    assert a["dhash256"] == b["dhash256"]


def test_public_out_of_image_box_is_excluded_instead_of_modified():
    bad = {"target_geometry": {"type": "bbox", "boxes": [[0.1, 0.2, 1.001, 0.5]]}}
    assert not data.valid_public_geometry(bad)
    assert bad["target_geometry"]["boxes"][0][2] == 1.001
