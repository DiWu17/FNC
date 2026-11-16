"""Build a held-out action-evaluation set without reusing old test scenarios."""

from __future__ import annotations

import copy
import hashlib
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .action_env import ENV_VERSION
from .dataset import load_manifest, validate_manifest
from .visual_state_dataset import build_visual_state_dataset, write_contact_sheet

DEFAULT_SOURCE = Path("data/visual_state_v2_matched/manifest.jsonl")


def build_action_dataset(output_dir: Path | str, source_manifest: Path | str = DEFAULT_SOURCE,
                         seed: int = 144, test_scenarios: int = 60) -> Path:
    """Reuse source train/val, replace its entire test with fresh matched scenarios.

    Default source contributes 180 train + 60 val groups; a new seed supplies
    30 iid + 30 ood test groups. All PNGs are copied below the returned manifest.
    Cache generation stays below output/_generation_cache for reproducibility.
    """
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    if isinstance(test_scenarios, bool) or not isinstance(test_scenarios, int) or test_scenarios < 8:
        raise ValueError("test_scenarios must be an integer >= 8")
    source_path = Path(source_manifest).expanduser().resolve()
    source_rows = load_manifest(source_path)
    if any(row.get("task") != "visual_state" or row["instruction"] != "点击保存" for row in source_rows):
        raise ValueError("Action data requires the fixed-instruction visual_state source")
    if any(row.get("source") != "synthetic_visual_state_v2_matched" for row in source_rows):
        raise ValueError("Use the matched-control source, not the early blank-absent data")
    source_seeds = {row["metadata"].get("seed") for row in source_rows}
    if seed in source_seeds:
        raise ValueError("Fresh test seed must differ from every source seed")
    trainval = [row for row in source_rows if row["split"] in {"train", "val"}]
    if not {"train", "val"} <= {row["split"] for row in trainval}:
        raise ValueError("Source must contain nonempty train and val splits")
    output = Path(output_dir).expanduser().resolve()
    if output == source_path.parent:
        raise ValueError("Action output must not overwrite the source dataset")
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "_generation_cache" / f"seed_{seed}"
    desired = {"iid": test_scenarios // 2, "ood": test_scenarios - test_scenarios // 2}
    generation_count = max(40, test_scenarios * 5)
    while True:
        generated = build_visual_state_dataset(cache, num_scenarios=generation_count, seed=seed)
        generated_rows = load_manifest(generated)
        group_domain = {row["group_id"]: row["metadata"]["domain"] for row in generated_rows if row["split"] == "test"}
        if all(sum(value == domain for value in group_domain.values()) >= count for domain, count in desired.items()):
            break
        generation_count += 10
    selected_groups = set()
    for domain, count in desired.items():
        available = sorted(group for group, value in group_domain.items() if value == domain)
        random.Random(f"action_test_groups:{seed}:{domain}").shuffle(available)
        selected_groups.update(available[:count])
    fresh = [row for row in generated_rows if row["split"] == "test" and row["group_id"] in selected_groups]
    if set(row["group_id"] for row in trainval) & selected_groups:
        raise ValueError("Fresh test group IDs collide with training or validation")
    records = []
    for partition, rows, root in (("source", trainval, source_path.parent), ("fresh_test", fresh, generated.parent)):
        for original in rows:
            row = copy.deepcopy(original)
            relative = (Path("images") / partition / Path(original["image"])).as_posix()
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / original["image"], destination)
            row["image"] = relative
            row["metadata"].update({"action_env_version": ENV_VERSION, "action_partition": partition,
                                     "original_source": original["source"]})
            row["source"] = "action_v3_source_trainval" if partition == "source" else "action_v3_fresh_test"
            records.append(row)
    random.Random(f"action_manifest:{seed}").shuffle(records)
    manifest = output / "manifest.jsonl"
    payload = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in records)
    manifest.write_text(payload, encoding="utf-8", newline="\n")
    summary = validate_manifest(records, manifest_path=manifest)
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    facts = {row["group_id"]: (row["split"], row["metadata"]["domain"]) for row in records}
    for split, domain in facts.values():
        counts[split][domain] += 1
    images_hash = hashlib.sha256()
    for image in sorted({row["image"] for row in records}):
        images_hash.update(image.encode("utf-8") + b"\0")
        images_hash.update(hashlib.sha256((output / image).read_bytes()).digest())
    summary.update({"task": "visual_state", "environment_version": ENV_VERSION,
                    "test_seed": seed, "source_seeds": sorted(value for value in source_seeds if value is not None),
                    "test_scenarios": test_scenarios, "scenario_counts_by_split_domain": dict(counts),
                    "excluded_old_test_samples": sum(row["split"] == "test" for row in source_rows),
                    "source_manifest_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
                    "generated_manifest_sha256": hashlib.sha256(generated.read_bytes()).hexdigest(),
                    "manifest_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                    "images_sha256": images_hash.hexdigest(),
                    "source_manifest": str(source_path),
                    "counterfactual_protocol": "matched_control_count_and_enabled_style"})
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")
    write_contact_sheet(manifest)
    return manifest
