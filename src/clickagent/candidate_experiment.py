"""Resumable feature collection for the candidate-conditioned verifier pilot."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import time
from pathlib import Path

import numpy as np
from PIL import Image

from .action_benchmark import sha256, stable_hash, write_json
from .action_grounding import AttentionPointGrounder
from .candidate_data import prepare, candidate_labels, SEED, VERSION
from .public_benchmark import cache_key


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("runs/candidate_v2"))
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args(argv)
    rows = prepare(args.root, args.output)
    if args.prepare_only:
        print(f"Prepared {len(rows)} records", flush=True)
        return
    from .candidate_model import CandidateRuntime, VERIFY_PROMPT
    from .model_features import ExtractionConfig, _library_versions
    import torch
    torch.set_num_threads(4)
    config = ExtractionConfig(model_path=str((args.root / "models/Qwen3-VL-8B-Instruct").resolve()),
        layers=(15, 28, 36), device_map="balanced", dtype="float16", max_pixels=262144,
        max_new_tokens=32, seed=SEED, task="visual_state", conditions=("prefill",),
        max_memory={0: "13GiB", 1: "13GiB", "cpu": "24GiB"},
        offload_folder=str((args.output / "offload").resolve()))
    grounder_path = args.root / "runs/action_v3/grounder/grounder.joblib"
    source = Path(__file__).parent
    identity = {"version": VERSION, "config": asdict(config), "libraries": _library_versions(),
        "manifest_sha256": sha256(args.output / "manifest.jsonl"), "verify_prompt": VERIFY_PROMPT,
        "grounder_sha256": sha256(grounder_path), "backbone_frozen": True,
        "source_sha256": {name: sha256(source / name) for name in
            ("candidate_model.py", "candidate_data.py", "candidate_experiment.py", "public_model.py",
             "action_model.py", "model_features.py", "public_diagnostics.py")}}
    fingerprint = stable_hash(identity)
    identity_path = args.output / "identity.json"
    if identity_path.exists() and stable_hash(json.loads(identity_path.read_text())) != fingerprint:
        raise ValueError("Source/config changed: use a different output or restore the frozen collection sources")
    write_json(identity_path, identity)
    for folder in ("features", "rows", "proposals", "frozen_source"):
        (args.output / folder).mkdir(exist_ok=True)
    for name in identity["source_sha256"]:
        target = args.output / "frozen_source" / name
        if not target.exists():
            target.write_bytes((source / name).read_bytes())
    pending = [r for r in rows if not (args.output / "features" / f"{cache_key(r['sample_id'])}.npz").exists()]
    if args.smoke:
        pending = [next(r for r in pending if r["split"] == s) for s in ("train", "calibration", "test")]
    if args.limit:
        pending = pending[:args.limit]
    if not pending:
        print("All features already present", flush=True)
        return
    print(f"Loading frozen Qwen; {len(pending)} pending / {len(rows)} total", flush=True)
    runtime = CandidateRuntime(config)
    grounder = AttentionPointGrounder.load(grounder_path)
    write_json(args.output / "runtime.json", runtime.runtime_metadata)
    started = time.perf_counter()
    for index, case in enumerate(pending):
        key = cache_key(case["sample_id"])
        if sha256(Path(case["image"])) != case["image_sha256"]:
            raise ValueError("image content changed")
        with Image.open(case["image"]) as opened:
            image = opened.convert("RGB")
        if case["split"] in {"train", "dev"}:
            proposal = {"points": case["training_candidates"], "sources": ["annotation_sampling"] * 4}
        else:
            path = args.output / "proposals" / f"{key}.json"
            if path.exists():
                proposal = json.loads(path.read_text(encoding="utf-8"))
                if proposal["fingerprint"] != fingerprint:
                    raise ValueError("proposal cache fingerprint changed")
            else:
                proposal = runtime.propose(image, case["base_instruction"], grounder)
                proposal["fingerprint"] = fingerprint
                write_json(path, proposal)
        arrays, meta = runtime.verify_features(image, case["instruction"], proposal["points"])
        # Ground truth is consulted only after the model call has returned.
        labels = candidate_labels(case, proposal["points"])
        meta.update({"sample_id": case["sample_id"], "split": case["split"], "case_kind": case["case_kind"],
            "fingerprint": fingerprint, "image_sha256": case["image_sha256"], "proposal": proposal,
            "candidate_labels": labels, "global_actionable_label": case["case_kind"] == "click"})
        arrays["metadata"] = np.asarray(json.dumps(meta, ensure_ascii=False))
        path = args.output / "features" / f"{key}.npz"
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **arrays)
        temporary.replace(path)
        write_json(args.output / "rows" / f"{key}.json", meta)
        completed = sum(1 for _ in (args.output / "features").glob("*.npz"))
        progress = {"completed": completed, "total": len(rows), "last_split": case["split"],
                    "last_case": case["sample_id"], "elapsed_this_process": time.perf_counter() - started,
                    "inference_errors": 0, "all_complete": completed == len(rows)}
        write_json(args.output / "progress.json", progress)
        print(f"[{completed}/{len(rows)}] {case['split']}/{case['case_kind']} "
              f"native={meta['native_choice']} valid_candidates={sum(labels)} "
              f"verify={meta['elapsed_seconds']:.2f}s", flush=True)
    if not args.limit and not args.smoke:
        write_json(args.output / "collection_complete.json", {"complete": True, "n": len(rows), "fingerprint": fingerprint})


if __name__ == "__main__":
    main()
