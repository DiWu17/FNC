"""Training-only counterfactuals: change target or remove every valid proposal.

The original collection, development/calibration/test partitions stay unchanged.
Repeated screenshots are intentional within training and never cross a split.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from .action_benchmark import sha256, stable_hash, write_json
from .candidate_data import candidate_labels, order, SEED
from .candidate_model import CandidateRuntime
from .model_features import ExtractionConfig
from .public_benchmark import read_jsonl, cache_key


def prepare_augmentation(root, output):
    folder = output / "augmentation"
    folder.mkdir(exist_ok=True)
    manifest = folder / "manifest.jsonl"
    if manifest.exists():
        return read_jsonl(manifest)
    base = [r for r in read_jsonl(output / "manifest.jsonl") if r["split"] == "train" and r["case_kind"] == "click"]
    originals = read_jsonl(root / "data/public_v1/manifests/venusbench_gd.jsonl")
    groups = {}
    for row in originals:
        groups.setdefault(row["image"], []).append(row)
    rows = []
    for modifier in ("remove_valid", "swap_target"):
        count = 0
        for case in sorted(base, key=lambda r: order(r["sample_id"] + modifier)):
            points = case["training_candidates"]
            new = dict(case)
            if modifier == "remove_valid":
                points = [p for p, valid in zip(points, candidate_labels(case, points)) if not valid]
                rng = np.random.default_rng(int(order(case["sample_id"] + modifier)[:8], 16))
                w,h = case["image_size"]
                while len(points) < 4:
                    p = [int(rng.integers(w)), int(rng.integers(h))]
                    if p not in points and not candidate_labels(case, [p])[0]:
                        points.append(p)
                assert not any(candidate_labels(case, points))
            else:
                alternatives = []
                for target in groups[case["image"]]:
                    if not target["expected_actionable"] or target["sample_id"] == case["source_sample_id"]:
                        continue
                    hypothetical = {**case, "target_geometry": target["target_geometry"]}
                    new_labels = candidate_labels(hypothetical, points)
                    if any(new_labels) and new_labels != candidate_labels(case, points):
                        alternatives.append(target)
                if not alternatives:
                    continue
                target = sorted(alternatives, key=lambda r: order(r["sample_id"]))[0]
                new.update({"target_geometry": target["target_geometry"], "base_instruction": target["instruction"],
                    "instruction": f"Click the target described by this GUI instruction now: {target['instruction']}"})
            new.update({"sample_id": case["sample_id"] + ":" + modifier, "training_candidates": points,
                "augmentation": modifier, "base_sample_id": case["sample_id"]})
            rows.append(new)
            count += 1
            if count == 64:
                break
        if count != 64:
            raise ValueError(f"not enough {modifier} counterfactuals: {count}")
    manifest.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    write_json(folder / "selection.json", {"n": len(rows), "remove_valid":64, "swap_target":64,
        "selection_uses_test_outcomes":False, "repeated_screenshots_only_within_training":True,
        "base_manifest_sha256":sha256(output / "manifest.jsonl"), "manifest_sha256":sha256(manifest)})
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("runs/candidate_v2"))
    parser.add_argument("--prepare-only",action="store_true")
    args = parser.parse_args(argv)
    root = Path.cwd()
    rows = prepare_augmentation(root,args.output)
    if args.prepare_only:
        print("Prepared training-only counterfactuals",len(rows),flush=True)
        return
    import torch
    torch.set_num_threads(4)
    folder = args.output / "augmentation"
    config_data = json.loads((args.output / "identity.json").read_text(encoding="utf-8"))["config"]
    config_data["max_memory"] = {int(k) if str(k).isdigit() else k:v for k,v in config_data["max_memory"].items()}
    config = ExtractionConfig(**config_data)
    identity = {"base_identity_sha256":sha256(args.output / "identity.json"),
                "manifest_sha256":sha256(folder / "manifest.jsonl"), "source_sha256":sha256(Path(__file__)),
                "runtime_source_sha256":sha256(Path(__file__).with_name("candidate_model.py"))}
    fingerprint = stable_hash(identity)
    identity_path = folder / "identity.json"
    if identity_path.exists() and stable_hash(json.loads(identity_path.read_text(encoding="utf-8"))) != fingerprint:
        raise ValueError("augmentation identity changed")
    write_json(identity_path,identity)
    (folder / "features").mkdir(exist_ok=True)
    pending = [r for r in rows if not (folder / "features" / f"{cache_key(r['sample_id'])}.npz").exists()]
    if not pending:
        return
    runtime = CandidateRuntime(config)
    for index,case in enumerate(pending):
        with Image.open(case["image"]) as opened:
            image = opened.convert("RGB")
        if sha256(Path(case["image"])) != case["image_sha256"]:
            raise ValueError("augmentation screenshot changed")
        arrays,meta = runtime.verify_features(image,case["instruction"],case["training_candidates"])
        meta.update({"sample_id":case["sample_id"], "split":"train", "case_kind":case["case_kind"],
            "image_sha256":case["image_sha256"], "fingerprint":fingerprint,
            "candidate_labels":candidate_labels(case,case["training_candidates"]),
            "global_actionable_label":True, "augmentation":case["augmentation"]})
        arrays["metadata"] = np.asarray(json.dumps(meta,ensure_ascii=False))
        path = folder / "features" / f"{cache_key(case['sample_id'])}.npz"
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream,**arrays)
        temporary.replace(path)
        print(f"[{index+1}/{len(pending)}] {case['augmentation']} correct={sum(meta['candidate_labels'])}",flush=True)
    write_json(folder / "complete.json",{"complete":True,"n":len(rows),"fingerprint":fingerprint})


if __name__ == "__main__":
    main()
