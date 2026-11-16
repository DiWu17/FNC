"""Reproducible, screenshot-disjoint candidate-verification pilot.

Existing public benchmark annotations are *repartitioned* here. These are new
internal train/dev/calibration/test splits, never official benchmark scores.
Previously inspected public-run screenshots are excluded by content SHA256.
Ground-truth candidate sampling is allowed only in train/dev. Calibration and
test proposals must come from the model, without annotation coordinates.
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .action_benchmark import sha256, stable_hash, write_json
from .public_benchmark import read_jsonl
from .public_evaluation import point_in_geometry


VERSION = 2
K = 4
SEED = 20261030
COUNTS = {"train": (240, 32, 32, 16), "dev": (56, 8, 8, 8),
          "calibration": (56, 8, 8, 8), "test": (112, 16, 16, 16)}


def order(value):
    return hashlib.sha256(f"{SEED}:{value}".encode()).hexdigest()


def application(row):
    m = row.get("metadata", {})
    name = str(m.get("application") or m.get("image_metadata", {}).get("application") or "unknown").lower()
    aliases = {"ae": "aftereffects", "eaby": "ebay", "asana-en": "asana",
               "googlemaps-en": "googlemaps", "googlemaps-ch": "googlemaps", "google-maps": "googlemaps",
               "wikipedia-ch": "wikipedia", "system-en": "system", "system-zh": "system"}
    name = aliases.get(name, name)
    for stem in ("powerpoint", "illustrator", "vscode", "github_desktop", "obsidian", "typora", "sketch", "canva", "docker"):
        if stem in name:
            return stem
    return name


def candidate_labels(case, points):
    """Labels are used by training/evaluation only, never the runtime."""
    w, h = case["image_size"]
    allowed = case["intent"] == "click" and case["expected_actionable"]
    return [bool(allowed and point_in_geometry((p[0] / w, p[1] / h), case["target_geometry"])) for p in points]


def _centers(row):
    g = row.get("target_geometry")
    if not g or g["type"] != "bbox":
        return []
    w, h = row["image_size"]
    return [[min(w - 1, int((a + c) * w / 2)), min(h - 1, int((b + d) * h / 2))]
            for a, b, c, d in g["boxes"]]


def annotation_candidates(case, same_image):
    if case["split"] not in {"train", "dev"}:
        raise ValueError("annotation-derived proposals are forbidden outside train/dev")
    # Both the true target and distractors are annotation centers: the label
    # cannot be recovered simply by identifying the more 'center-like' point.
    target = _centers(case)
    alternatives = []
    for row in sorted(same_image, key=lambda r: order(r["sample_id"])):
        for p in _centers(row):
            if p not in alternatives and p not in target:
                alternatives.append(p)
    chosen = target[:1] + alternatives[:K - bool(target)]
    rng = np.random.default_rng(int(order(case["sample_id"])[:8], 16))
    w, h = case["image_size"]
    while len(chosen) < K:
        p = [int(rng.integers(0, w)), int(rng.integers(0, h))]
        if p not in chosen:
            chosen.append(p)
    chosen = [chosen[i] for i in rng.permutation(K)]
    return chosen


def prepare(root: Path, output: Path):
    """Freeze the full selection before loading the VLM or observing scores."""
    manifest = output / "manifest.jsonl"
    if manifest.exists():
        return read_jsonl(manifest)
    rows = read_jsonl(root / "data/public_v1/manifests/venusbench_gd.jsonl")
    pro = read_jsonl(root / "data/public_v1/manifests/screenspot_pro.jsonl")
    old_paths = set()
    for p in (root / "runs/public_v1").glob("*/selected_manifest.jsonl"):
        old_paths.update(r["image"] for r in read_jsonl(p))
    hashes = {}
    def digest(path):
        if path not in hashes:
            hashes[path] = sha256(Path(path))
        return hashes[path]
    old_hashes = {digest(p) for p in sorted(old_paths)}
    # Whole normalized applications go to the fresh Venus test partition.
    apps = sorted({application(r) for r in rows}, key=order)
    heldout_apps = set(apps[::5])
    trained_apps = set(apps) - heldout_apps
    groups = defaultdict(list)
    for row in rows:
        groups[row["image"]].append(row)
    available = []
    for path, group in groups.items():
        if digest(path) in old_hashes:
            continue
        for r in group:
            if (not r["expected_actionable"] or _centers(r)):
                available.append(r)
    pools = defaultdict(list)
    for r in available:
        app = application(r)
        if app in heldout_apps:
            split = "test"
        else:
            # The split is attached to screenshot content, not an annotation.
            bucket = int(order(digest(r["image"]))[:8], 16) % 10
            split = "dev" if bucket == 0 else "calibration" if bucket == 1 else "train"
        pools[split].append(r)
    # Extra test applications from Pro are disjoint from every training app.
    pools["test_pro"] = [r for r in pro if application(r) not in trained_apps
                         and digest(r["image"]) not in old_hashes and _centers(r)]
    chosen = []
    used = set()
    for split, counts in COUNTS.items():
        for intent, needed in zip(("click", "forbid", "describe", "refusal"), counts):
            eligible = [r for r in pools[split] if bool(r["expected_actionable"]) == (intent != "refusal")]
            if split == "test" and intent == "click":
                # Half click cases are cross-dataset industrial/professional UIs.
                selected_pro = sorted(pools["test_pro"], key=lambda r: order(r["sample_id"]))[:needed // 2]
                source_pools = [(selected_pro, needed // 2), (eligible, needed - needed // 2)]
            else:
                source_pools = [(eligible, needed)]
            for pool, n in source_pools:
                count = 0
                # Prefer well-annotated multi-target screenshots for training.
                ranked = sorted(pool, key=lambda r: (
                    split in {"train", "dev"} and sum(bool(x["expected_actionable"]) for x in groups[r["image"]]) < 2,
                    order(r["sample_id"])))
                for row in ranked:
                    image_hash = digest(row["image"])
                    if image_hash in used:
                        continue
                    used.add(image_hash)
                    case = {k: row[k] for k in ("sample_id", "dataset", "image", "instruction", "image_size",
                                                "expected_actionable", "target_geometry")}
                    case.update({"source_sample_id": row["sample_id"], "sample_id": f"cv1:{split}:{intent}:{row['sample_id']}",
                                 "split": split, "intent": "click" if intent == "refusal" else intent,
                                 "case_kind": intent, "application": application(row), "image_sha256": image_hash,
                                 "annotation_source_split": row["split"], "base_instruction": row["instruction"],
                                 "proposal_source": "annotation_sampling" if split in {"train", "dev"} else "frozen_model"})
                    if intent in {"click", "refusal"}:
                        case["instruction"] = f"Click the target described by this GUI instruction now: {row['instruction']}"
                    elif intent == "forbid":
                        case["instruction"] = f"Do not perform this GUI operation: {row['instruction']} Leave the interface untouched."
                    elif intent == "describe":
                        case["instruction"] = f"Describe where the target of this operation is, without clicking or activating it: {row['instruction']}"
                    if split in {"train", "dev"}:
                        case["training_candidates"] = annotation_candidates(case, groups[row["image"]])
                    chosen.append(case)
                    count += 1
                    if count == n:
                        break
                if count != n:
                    raise ValueError(f"insufficient eligible {split}/{intent}: {count} < {n}")
    train_apps = {r["application"] for r in chosen if r["split"] in {"train", "dev", "calibration"}}
    test_apps = {r["application"] for r in chosen if r["split"] == "test"}
    assert not train_apps & test_apps
    assert len({r["image_sha256"] for r in chosen}) == len(chosen)
    assert not {r["image_sha256"] for r in chosen} & old_hashes
    output.mkdir(parents=True, exist_ok=True)
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in chosen), encoding="utf-8")
    write_json(output / "selection.json", {"version": VERSION, "seed": SEED, "k": K, "n": len(chosen),
        "counts": {s: dict(Counter(r["case_kind"] for r in chosen if r["split"] == s)) for s in COUNTS},
        "manifest_sha256": sha256(manifest), "excluded_old_screenshot_hashes": len(old_hashes),
        "train_applications": sorted(train_apps), "test_applications": sorted(test_apps),
        "all_screenshots_disjoint": True, "test_applications_disjoint": True,
        "official_benchmark_score": False, "public_benchmark_annotations_repartitioned": True,
        "candidate_sampling_shift": "train/dev use annotation centers; calibration/test use frozen model proposals",
        "limitations": ["no application execution", "grounding boxes do not certify enabled/clickable state",
                        "forbid/describe instructions are templated counterfactuals", "exact SHA exclusion does not establish perceptual deduplication"]})
    return chosen
