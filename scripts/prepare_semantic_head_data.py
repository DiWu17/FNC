"""Freeze real, same-image semantic-head additions and a never-used public test.

Offline CPU only. Existing manifests/images are read, never rewritten. This
script does not produce protocol.json or use model outputs to select data.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import re
import subprocess
from typing import Any

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = 20261107
PUBLIC_DATASETS = ("screenspot_v2", "screenspot_pro", "venusbench_gd")


def read_rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def key(seed, *parts):
    return hashlib.sha256(":".join(map(str, (seed, *parts))).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def write_rows(path, rows, *, frozen=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows)
    if frozen and path.exists() and path.read_text(encoding="utf-8") != text:
        raise ValueError(f"Refusing to overwrite a changed frozen data selection: {path}")
    path.write_text(text, encoding="utf-8")


class HammingIndex:
    """Exact BK-tree radius search; no approximate nearest-neighbor misses."""
    def __init__(self, values=()):
        self.values, self.root = set(), None
        for value in sorted(set(values)):
            self.add(value)

    def add(self, value):
        value = int(value)
        if value in self.values:
            return
        self.values.add(value)
        if self.root is None:
            self.root = [value, {}]
            return
        node = self.root
        while True:
            distance = (node[0] ^ value).bit_count()
            if distance not in node[1]:
                node[1][distance] = [value, {}]
                return
            node = node[1][distance]

    def near(self, value, radius=4):
        value = int(value)
        stack = [self.root] if self.root is not None else []
        while stack:
            node = stack.pop()
            distance = (node[0] ^ value).bit_count()
            if distance <= radius:
                return True
            stack.extend(child for edge, child in node[1].items() if distance - radius <= edge <= distance + radius)
        return False


def image_info(path):
    path = Path(path)
    with Image.open(path) as opened:
        image = opened.convert("RGB")
        w, h = image.size
        pixel = hashlib.sha256(f"{w}x{h}:RGB:".encode() + image.tobytes()).hexdigest()
        gray = np.asarray(image.convert("L").resize((17, 16), Image.Resampling.LANCZOS))
        dhash = np.packbits((gray[:, 1:] > gray[:, :-1]).ravel()).tobytes().hex()
    return {"image": str(path.resolve()), "file_sha256": sha(path), "pixel_sha256": pixel,
            "dhash256": dhash, "image_size": [w, h]}


def image_path(row, manifest):
    path = Path(row.get("image", row.get("image_path", "")))
    return (path if path.is_absolute() else Path(manifest).parent / path).resolve()


def valid_public_geometry(row):
    geometry = row.get("target_geometry")
    if not isinstance(geometry, dict):
        return False
    if geometry.get("type") == "bbox":
        boxes = geometry.get("boxes", [])
        return bool(boxes) and all(len(b) == 4 and all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in b) and b[0] < b[2] and b[1] < b[3] for b in boxes)
    if geometry.get("type") == "polygon":
        polygons = geometry.get("polygons", [])
        return bool(polygons) and all(len(p) >= 3 and all(len(xy) == 2 and all(isinstance(v, (int, float)) and math.isfinite(v) and 0 <= v <= 1 for v in xy) for xy in p) for p in polygons)
    return False


def discover_history(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    try:
        command = ["rg", "--files", "--hidden", "--no-ignore", str(root / "runs"), "-g", "*manifest*.jsonl"]
        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode not in (0, 1):
            raise OSError(result.stderr)
        paths = [Path(line).resolve() for line in result.stdout.splitlines() if line]
    except FileNotFoundError:
        paths = list((root / "runs").rglob("*manifest*.jsonl"))
    selected, excluded = [], []
    for path in paths:
        artifact = any("pytest" in part.lower() or part.lower().startswith("refinement_tests") for part in path.parts)
        if path == output or output in path.parents or artifact:
            excluded.append(str(path))
        else:
            selected.append(path)
    return sorted(selected), sorted(excluded)


def box(row):
    geometry = row.get("target_geometry", {})
    boxes = geometry.get("boxes", [])
    if geometry.get("type") != "bbox" or len(boxes) != 1:
        raise ValueError("Official training candidates require exactly one original bbox")
    b = list(map(float, boxes[0]))
    if len(b) != 4 or not all(math.isfinite(v) for v in b) or not 0 <= b[0] < b[2] <= 1 or not 0 <= b[1] < b[3] <= 1:
        raise ValueError("Invalid official training bbox")
    return b


def normalize_text(text):
    return " ".join(text.casefold().split())


def distinct(a, b):
    if normalize_text(a["instruction"]) == normalize_text(b["instruction"]):
        return False
    aa, bb = box(a), box(b)
    intersection = max(0, min(aa[2], bb[2]) - max(aa[0], bb[0])) * max(0, min(aa[3], bb[3]) - max(aa[1], bb[1]))
    return intersection == 0


def pair_hardness(a, b):
    """Prespecified geometry/text heuristic, never model-measured hardness."""
    aa, bb = box(a), box(b)
    distance = math.hypot((aa[0] + aa[2] - bb[0] - bb[2]) / 2, (aa[1] + aa[3] - bb[1] - bb[3]) / 2)
    stop = {"click", "tap", "on", "the", "a", "an", "button", "icon", "please", "to"}
    ta = set(re.findall(r"\w+", a["instruction"].casefold())) - stop
    tb = set(re.findall(r"\w+", b["instruction"].casefold())) - stop
    similarity = len(ta & tb) / len(ta | tb) if ta | tb else 0
    area = min((aa[2] - aa[0]) * (aa[3] - aa[1]), (bb[2] - bb[0]) * (bb[3] - bb[1]))
    return 1 / (1 + 10 * distance) + 0.5 * similarity + 0.2 * (1 - min(100 * area, 1))


def official_candidate(row):
    return row.get("source_split") == "train" and row.get("official_training_split") is True and row.get("expected_actionable") is True and bool(row.get("instruction", "").strip())


def choose_extensions(base_train, pool, used_ids, protected_pixels, protected_dhash,
                      historical_pixels, historical_dhash, budget=256, per_old_image=4,
                      seed=DEFAULT_SEED):
    groups, old = defaultdict(list), defaultdict(list)
    for row in pool:
        if official_candidate(row):
            box(row)
            groups[row["pixel_sha256"]].append(row)
    for row in base_train:
        old[row["pixel_sha256"]].append(row)
    extras, origins, stats = [], {}, Counter()
    eligible_old = defaultdict(list)
    for pixel, originals in old.items():
        if pixel in protected_pixels or protected_dhash.near(int(originals[0]["dhash256"], 16)):
            # The immutable base is not silently removed. Any newly introduced
            # annotation on a protected old image is prohibited.
            stats["old_images_protected_from_extension"] += 1
            continue
        candidates = [row for row in groups[pixel] if row["sample_id"] not in used_ids and all(distinct(row, prior) for prior in originals)]
        chosen = []
        while candidates and len(chosen) < per_old_image:
            candidates.sort(key=lambda r: (-max(pair_hardness(r, prior) for prior in originals + chosen), key(seed, "old_target", r["sample_id"])))
            picked = candidates.pop(0)
            chosen.append(picked)
            candidates = [row for row in candidates if distinct(row, picked)]
        if chosen:
            stats["old_images_with_available_extension"] += 1
            stats["singleton_old_images_with_available_extension"] += len(originals) == 1
            eligible_old[originals[0]["dataset"]].append((len(originals) != 1, key(seed, "old_image", pixel), pixel, chosen))
    for values in eligible_old.values():
        values.sort()
    # Round-robin sources, then images/target depth, rather than consuming all
    # candidate annotations of a few OmniAct screenshots.
    for depth in range(per_old_image):
        positions = {source: 0 for source in eligible_old}
        while len(extras) < budget:
            progressed = False
            for source in sorted(eligible_old):
                values = eligible_old[source]
                while positions[source] < len(values):
                    _, _, pixel, chosen = values[positions[source]]
                    positions[source] += 1
                    if depth >= len(chosen):
                        continue
                    row = dict(chosen[depth])
                    extras.append(row)
                    origins[row["sample_id"]] = "existing_train_image"
                    progressed = True
                    break
                if len(extras) >= budget:
                    break
            if not progressed:
                break
    stats["old_image_extension_queries"] = len(extras)
    # New images must never have occurred in any historical split, and remain
    # far from all public screenshots. Each contributes exactly two genuine
    # original targets, keeping a final odd budget slot empty if necessary.
    fresh = defaultdict(list)
    for pixel, candidates in groups.items():
        if pixel in old or pixel in historical_pixels or pixel in protected_pixels:
            continue
        dhash = int(candidates[0]["dhash256"], 16)
        if historical_dhash.near(dhash) or protected_dhash.near(dhash):
            continue
        candidates = [row for row in candidates if row["sample_id"] not in used_ids]
        pairs = [(a, b) for a, b in itertools.combinations(candidates, 2) if distinct(a, b)]
        if not pairs:
            continue
        a, b = min(pairs, key=lambda pair: (-pair_hardness(*pair), key(seed, "new_pair", *sorted((pair[0]["sample_id"], pair[1]["sample_id"])))) )
        fresh[a["dataset"]].append((key(seed, "new_image", pixel), pixel, a, b))
    stats["eligible_new_pair_images"] = sum(map(len, fresh.values()))
    for values in fresh.values():
        values.sort()
    selected_new_phash = HammingIndex()
    while len(extras) + 2 <= budget:
        progressed = False
        for source in sorted(fresh):
            while fresh[source]:
                _, pixel, a, b = fresh[source].pop(0)
                dhash = int(a["dhash256"], 16)
                if selected_new_phash.near(dhash):
                    stats["new_pair_images_near_another_selected_new_image"] += 1
                    continue
                selected_new_phash.add(dhash)
                for row in (a, b):
                    extras.append(dict(row))
                    origins[row["sample_id"]] = "new_official_train_image_pair"
                progressed = True
                break
            if len(extras) + 2 > budget:
                break
        if not progressed:
            break
    stats["new_image_pair_queries"] = len(extras) - stats["old_image_extension_queries"]
    stats["unfilled_query_budget"] = budget - len(extras)
    return extras, origins, dict(stats)


def prepare(args):
    root, output = Path(args.root).resolve(), Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data = root / "data/conditioned_v3"
    base_path, diagnostic_path = root / "runs/conditioned_v3/manifest.jsonl", root / "runs/candidate_ranker_v5/manifest.jsonl"
    base = read_rows(base_path)
    base_train = [r for r in base if r["experiment_split"] == "train"]
    base_val = [r for r in base if r["experiment_split"] == "val"]
    diagnostic = [r for r in read_rows(diagnostic_path) if r["experiment_split"] == "test"]
    if (len(base_train), len(base_val), len(diagnostic)) != (618, 172, 192):
        raise ValueError("The frozen base618/val172/v5 diagnostic192 contract changed")
    receipt_path = data / "training_download_verified.json"
    receipt = read_json(receipt_path)
    if receipt.get("all_sha_verified") is not True:
        raise ValueError("Official source shard receipts are not verified")
    pool_path = data / "training_pool_b8ed555f5e93.jsonl"
    pool = read_rows(pool_path)
    index_path = data / "public_image_index.json"
    public_index = read_json(index_path)
    if public_index.get("complete") is not True:
        raise ValueError("Public image index is incomplete")
    infos = {str(Path(r["image"]).resolve()).casefold(): r for r in public_index["images"]}
    for row in pool:
        infos.setdefault(str(Path(row["image"]).resolve()).casefold(), {
            "image": row["image"], "file_sha256": row["image_sha256"], "pixel_sha256": row["pixel_sha256"],
            "dhash256": row["dhash256"], "image_size": row["image_size"]})
    history_paths, ignored = discover_history(root, output)
    history = [(path, row) for path in history_paths for row in read_rows(path)]
    cache_path = output / "data_selection_image_hash_cache.jsonl"
    cache = {r["image"].casefold(): r for r in read_rows(cache_path)} if cache_path.exists() else {}
    def info_for(path, verify=False):
        path = Path(path).resolve()
        name = str(path).casefold()
        if not path.is_file():
            raise FileNotFoundError(f"Historical or candidate image missing: {path}")
        if not verify and name in infos:
            return infos[name]
        stat = path.stat()
        cached = cache.get(name)
        if cached and cached.get("file_size") == stat.st_size and cached.get("mtime_ns") == stat.st_mtime_ns:
            info = cached
        else:
            info = {**image_info(path), "file_size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
            cache[name] = info
        expected = infos.get(name)
        if expected and any(expected[field] != info[field] for field in ("file_sha256", "pixel_sha256", "dhash256", "image_size")):
            raise ValueError(f"Image bytes/pixels differ from the frozen source/index: {path}")
        infos[name] = info
        return info
    historical_pixels, historical_hashes, protected_pixels, protected_hashes, used_ids = set(), set(), set(), set(), set()
    source_resolutions, source_controls, unresolved_history = {}, {}, []
    for inf in public_index["images"]:
        protected_pixels.add(inf["pixel_sha256"])
        protected_hashes.add(int(inf["dhash256"], 16))
    print(f"History manifests {len(history_paths)}; resolving all historical screenshots", flush=True)
    for i, (path, row) in enumerate(history):
        if row.get("sample_id"):
            used_ids.add(row["sample_id"])
        if not row.get("image") and not row.get("image_path"):
            continue
        resolved_path = image_path(row, path)
        if not resolved_path.is_file():
            # Cached-label permutation manifests intentionally retain image
            # paths relative to their original extraction manifest. Resolve
            # only through its captured, SHA-verified source, never basename
            # guesses or a new association between annotations and images.
            if path not in source_controls:
                control_path = path.parent / "control.json"
                control = read_json(control_path) if control_path.is_file() else {}
                original = Path(control.get("source_manifest", ""))
                if original.is_file() and control.get("source_manifest_sha256") == sha(original):
                    source_controls[path] = (original, {r["sample_id"]: r for r in read_rows(original)})
                    source_resolutions[str(path)] = {"control": str(control_path), "source_manifest": str(original), "sha256": sha(original)}
                else:
                    source_controls[path] = None
            source = source_controls[path]
            if source is not None:
                original, original_rows = source
                original_row = original_rows.get(row.get("sample_id"))
                if original_row is not None and original_row.get("image") == row.get("image"):
                    resolved_path = image_path(original_row, original)
        try:
            inf = info_for(resolved_path)
        except (FileNotFoundError, OSError) as exc:
            unresolved_history.append({"manifest": str(path), "sample_id": row.get("sample_id"), "image": str(resolved_path), "reason": str(exc), "provided_hashes_used": bool(row.get("pixel_sha256") and row.get("dhash256"))})
            if row.get("pixel_sha256") and row.get("dhash256"):
                inf = row
            else:
                continue
        historical_pixels.add(inf["pixel_sha256"])
        historical_hashes.add(int(inf["dhash256"], 16))
        split = row.get("experiment_split", row.get("split", "unknown"))
        if split != "train":
            protected_pixels.add(inf["pixel_sha256"])
            protected_hashes.add(int(inf["dhash256"], 16))
        if i and i % 2000 == 0:
            print(f"Historical rows {i}/{len(history)}", flush=True)
            write_rows(cache_path, sorted(cache.values(), key=lambda row: row["image"]), frozen=False)
    write_rows(cache_path, sorted(cache.values(), key=lambda row: row["image"]), frozen=False)
    # Include base val and old diagnostic explicitly even if history discovery
    # policy changes, and forbid public scenes for every new training query.
    for row in base_val + diagnostic:
        inf = info_for(row["image"])
        protected_pixels.add(inf["pixel_sha256"])
        protected_hashes.add(int(inf["dhash256"], 16))
    protected_tree, historical_tree = HammingIndex(protected_hashes), HammingIndex(historical_hashes)
    extras, origins, extension_stats = choose_extensions(base_train, pool, used_ids, protected_pixels, protected_tree,
                                                         historical_pixels, historical_tree, args.max_extra, args.per_old_image, args.seed)
    print(f"Selected {len(extras)} official additions: {extension_stats}", flush=True)
    training = [dict(row, experiment_split="train", split="train", training_extension=False) for row in base_train]
    extra_rows = [dict(row, experiment_split="train", split="train", training_extension=True,
                       training_extension_origin=origins[row["sample_id"]]) for row in extras]
    validation = [dict(row, experiment_split="val", split="val", training_extension=False) for row in base_val]
    old_diagnostic = [dict(row, experiment_split="diagnostic", split="diagnostic", training_extension=False) for row in diagnostic]
    for row in extra_rows:
        inf = info_for(row["image"], verify=True)
        if inf["pixel_sha256"] in protected_pixels or protected_tree.near(int(inf["dhash256"], 16)):
            raise ValueError("Training addition overlaps a protected image")
    test_pixels = set(historical_pixels)
    test_hashes = HammingIndex(historical_hashes)
    for row in training + extra_rows + validation + old_diagnostic:
        test_pixels.add(row["pixel_sha256"])
        test_hashes.add(int(row["dhash256"], 16))
    fresh_test, test_stats, public_paths = [], {}, []
    for slug in PUBLIC_DATASETS:
        manifest_path = root / f"data/public_v1/manifests/{slug}.jsonl"
        public_paths.append(manifest_path)
        all_rows = read_rows(manifest_path)
        candidates = [r for r in all_rows if r.get("split", "test") == "test" and r.get("expected_actionable") is True and valid_public_geometry(r)]
        candidates.sort(key=lambda r: key(args.seed, "fresh_test", r["sample_id"]))
        counters, selected = Counter(), []
        for row in candidates:
            inf = info_for(row["image"])
            if inf["pixel_sha256"] in test_pixels:
                counters["excluded_exact_prior_or_selected"] += 1
                continue
            if test_hashes.near(int(inf["dhash256"], 16)):
                counters["excluded_dhash_le4_prior_or_selected"] += 1
                continue
            # Recompute selected decoded pixels as a guard against stale index
            # metadata or a substituted screenshot file.
            inf = info_for(row["image"], verify=True)
            if inf["pixel_sha256"] in test_pixels or test_hashes.near(int(inf["dhash256"], 16)):
                raise ValueError("Selected test image changed after index lookup")
            test_pixels.add(inf["pixel_sha256"])
            test_hashes.add(int(inf["dhash256"], 16))
            selected.append(dict(row, experiment_split="test", split="test", training_extension=False,
                                 image_sha256=inf["file_sha256"], pixel_sha256=inf["pixel_sha256"],
                                 dhash256=inf["dhash256"], group_id=inf["pixel_sha256"]))
            if len(selected) == args.test_per_dataset:
                break
        if len(selected) != args.test_per_dataset:
            raise ValueError(f"Only {len(selected)} unused non-near images for {slug}; no split weakening allowed")
        fresh_test.extend(selected)
        test_stats[slug] = {"eligible_queries": len(candidates), "selected": len(selected),
                            "invalid_positive_geometry_excluded": sum(r.get("expected_actionable") is True and not valid_public_geometry(r) for r in all_rows),
                            "scan_rejections_before_quota": dict(counters)}
        print(f"Fresh test {slug}: {len(selected)} fully hashed images", flush=True)
    train_all = training + extra_rows
    groups = defaultdict(list)
    for row in train_all:
        groups[row["pixel_sha256"]].append(row)
    pairs = []
    for pixel, rows in groups.items():
        for a, b in itertools.combinations(sorted(rows, key=lambda r: r["sample_id"]), 2):
            if distinct(a, b):
                pairs.append({"pair_id": key(args.seed, "same_image_pair", a["sample_id"], b["sample_id"]),
                              "group_id": pixel, "pixel_sha256": pixel, "dataset": a["dataset"],
                              "sample_ids": [a["sample_id"], b["sample_id"]],
                              "geometries": [a["target_geometry"], b["target_geometry"]],
                              "contains_added_query": bool(a["training_extension"] or b["training_extension"]),
                              "heuristic_pair_hardness": pair_hardness(a, b)})
    manifest = train_all + validation + old_diagnostic + fresh_test
    ids = [row["sample_id"] for row in manifest]
    if len(set(ids)) != len(ids):
        raise ValueError("Final sample_id collision; official annotation IDs are never rewritten to hide it")
    by_split = defaultdict(set)
    for row in manifest:
        by_split[row["experiment_split"]].add(row["pixel_sha256"])
    for a, b in itertools.combinations(by_split, 2):
        if by_split[a] & by_split[b]:
            raise ValueError(f"Pixel leakage between {a} and {b}")
    # Test and train are deliberately all mutually non-near; base-val/old
    # diagnostic are retained unchanged rather than silently repartitioned.
    write_rows(output / "manifest.jsonl", manifest)
    write_rows(output / "extra_train_manifest.jsonl", extra_rows)
    write_rows(output / "data_selection_same_image_pairs.jsonl", pairs)
    write_rows(cache_path, sorted(cache.values(), key=lambda row: row["image"]), frozen=False)
    refs = [base_path, diagnostic_path, pool_path, index_path, receipt_path, *history_paths, *public_paths]
    summary = {"seed": args.seed, "status": "ready_with_unresolved_history_references" if unresolved_history else "ready", "max_extra_queries": args.max_extra,
               "max_additions_per_existing_train_image": args.per_old_image, "new_image_queries_per_group": 2,
               "counts": dict(Counter(row["experiment_split"] for row in manifest)),
               "base_train_queries": len(training), "base_train_images": len({r["pixel_sha256"] for r in training}),
               "base_val_queries": len(validation), "extra_queries": len(extra_rows),
               "extension_stats": extension_stats,
               "extra_by_origin": dict(Counter(origins.values())),
               "extra_by_source": dict(Counter(r["dataset"] for r in extra_rows)),
               "extra_images_by_source": dict(Counter(rows[0]["dataset"] for pixel, rows in groups.items() if any(r["training_extension"] for r in rows))),
               "training_target_count_histogram_per_image": dict(Counter(len(rows) for rows in groups.values())),
               "extra_query_count_histogram_per_image": dict(Counter(sum(r["training_extension"] for r in rows) for rows in groups.values() if any(r["training_extension"] for r in rows))),
               "same_image_distinct_nonoverlap_pairs": len(pairs),
               "same_image_pairs_involving_added_query": sum(p["contains_added_query"] for p in pairs),
               "test_selection": test_stats, "test_unique_pixels": len({r["pixel_sha256"] for r in fresh_test}),
               "historical_manifest_count": len(history_paths), "historical_pixel_count": len(historical_pixels),
               "historical_source_manifest_resolutions": source_resolutions,
               "unresolved_historical_image_references": unresolved_history,
               "public_reference_images": len(public_index["images"]),
               "public_or_historical_heldout_protected_pixel_count": len(protected_pixels),
               "ignored_generated_test_artifacts_or_own_outputs": ignored,
               "exclusion_policy": "New annotations: forbid every public image and all prior non-train/unknown images by exact RGB pixel SHA256 and exact dHash256 Hamming radius<=4. New train images additionally forbid every historical image; new test forbids all historical and expanded train/val/diagnostic images plus other selected test images by the same criteria.",
               "hardness_policy": "Only real official bbox/instruction pairs; strictly zero bbox intersection and different normalized instruction strings. Geometry proximity + small-target + instruction token overlap heuristic; no model performance used.",
               "base_rows_content_preserved_except_split_and_training_extension_metadata": True,
               "gpu_used": False, "network_used": False, "model_outputs_used_for_selection": False,
               "protocol_json_created": False,
               "input_sources": [{"path": str(path.resolve()), "sha256": sha(path)} for path in sorted(set(refs))],
               "source_script_sha256": sha(Path(__file__)),
               "outputs": {name: {"path": str((output / name).resolve()), "sha256": sha(output / name)} for name in
                           ("manifest.jsonl", "extra_train_manifest.jsonl", "data_selection_same_image_pairs.jsonl")}}
    write_json(output / "data_selection.json", summary)
    print(json.dumps({"manifest_ready": str(output / "manifest.jsonl"), "counts": summary["counts"], "extra_queries": len(extra_rows), "pairs": len(pairs)}, ensure_ascii=False), flush=True)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/semantic_head_v7")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--max-extra", type=int, default=256)
    parser.add_argument("--per-old-image", type=int, default=4)
    parser.add_argument("--test-per-dataset", type=int, default=64)
    args = parser.parse_args()
    if args.max_extra < 0 or args.per_old_image < 1 or args.test_per_dataset < 1:
        raise ValueError("Invalid query/image budgets")
    prepare(args)


if __name__ == "__main__":
    main()
