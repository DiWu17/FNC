"""Negative control: independently randomize state labels inside each scenario.

This module only rewrites cached feature labels. It never invokes the VLM and
never modifies the real feature exports. The three classes remain balanced in
each group and split, but the association with the screenshot is broken.
"""

from __future__ import annotations

from collections import defaultdict
import hashlib
import json
from pathlib import Path

import numpy as np


STATE_LABELS = ("click", "disabled", "absent")


def make_permuted_labels(records: list[dict], seed: int = 4301) -> dict[str, str]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        groups[record["group_id"]].append(record)
    rng = np.random.default_rng(seed)
    replacements = {}
    for group in sorted(groups):
        rows = sorted(groups[group], key=lambda row: row["sample_id"])
        if len(rows) != 3 or {row["label"] for row in rows} != set(STATE_LABELS):
            raise ValueError(f"Control requires exactly one complete state triplet: {group}")
        if len({row["split"] for row in rows}) != 1:
            raise ValueError(f"Split leakage in {group}")
        # All six permutations, including identity, have equal probability.
        # Always forcing a change would make each wrong label only 50% random.
        shuffled = rng.permutation(STATE_LABELS)
        replacements.update({row["sample_id"]: str(label) for row, label in zip(rows, shuffled)})
    return replacements


def prepare_permutation_control(manifest_path: Path | str, features_dir: Path | str,
                                output_dir: Path | str, seed: int = 4301) -> tuple[Path, Path]:
    manifest_path, features_dir, output_dir = map(Path, (manifest_path, features_dir, output_dir))
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    replacements = make_permuted_labels(records, seed)
    destination = output_dir / "features"
    source_root, target_root = features_dir.resolve(), destination.resolve()
    if source_root == target_root or source_root.is_relative_to(target_root) or target_root.is_relative_to(source_root):
        raise ValueError("Permutation destination must be separate from the original features")
    source_conditions = {name for name in ("prefill", "post_decision", "post_response")
                         if (features_dir / f"{name}.npz").is_file()}
    if not source_conditions:
        raise ValueError("No original feature conditions found")
    existing_conditions = {path.stem for path in destination.glob("*.npz")}
    if existing_conditions and existing_conditions != source_conditions:
        raise ValueError("Existing permutation conditions differ; use a fresh output directory")
    destination.mkdir(parents=True, exist_ok=True)
    if any(destination.glob("*.metadata.jsonl")) or (destination / "metadata.jsonl").exists():
        raise ValueError("Permutation directory must not contain real generation metadata")
    for condition in ("prefill", "post_decision", "post_response"):
        source = features_dir / f"{condition}.npz"
        if not source.exists():
            continue
        with np.load(source, allow_pickle=False) as archive:
            arrays = {key: archive[key] for key in archive.files}
        arrays["labels"] = np.asarray([replacements[str(sid)] for sid in arrays["sample_ids"]])
        with (destination / source.name).open("wb") as stream:
            np.savez_compressed(stream, **arrays)
    control_manifest = output_dir / "permuted_manifest.jsonl"
    with control_manifest.open("w", encoding="utf-8") as stream:
        for record in records:
            # This is evaluation metadata, not an extraction manifest: a
            # shuffled 'absent' label need not agree with the original box.
            row = {**record, "label": replacements[record["sample_id"]]}
            row["metadata"] = {**row.get("metadata", {}), "label_permutation_control": True}
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    audit = {
        "control": "independent uniform random permutation among three labels within every group",
        "seed": seed,
        "source_manifest": str(manifest_path.resolve()),
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_features": str(features_dir.resolve()),
        "samples": len(records),
        "unchanged_labels": sum(row["label"] == replacements[row["sample_id"]] for row in records),
        "features_changed": False,
        "generation_metadata_copied": False,
        "not_for_model_extraction": True,
        "interpretation": "One randomization is a sanity control, not a permutation significance test.",
    }
    (output_dir / "control.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return control_manifest, destination
