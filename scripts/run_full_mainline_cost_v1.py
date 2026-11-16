"""Retrospective V17-matched end-to-end cost for frozen full-aligned S1 and MVP.

Run each arm in its own process.  The public-only runtime and timing boundary are
the archived V17 implementation; this isolated worker changes only the readout
checkpoint used by OurRuntime.  No V17 files or shared runtime code are edited.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/full_mainline_cost_v1"
V17 = ROOT / "runs/advantage_validation_v17"
FULL = ROOT / "runs/full_scaling_v18/models/full_s1.pt"
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
os.environ.setdefault("HF_HOME", str(V17 / "cache/hf"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(V17 / "cache/hf/hub"))
os.environ.setdefault("MPLCONFIGDIR", str(V17 / "cache/matplotlib"))
os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")

V17_HASHES = {
    "scripts/run_advantage_inference_v17.py": "6f548a5368a9181dcfe84902594b5ed125f6cce2ccb382275a8b7554f1d76fdc",
    "src/clickagent/advantage_runtimes.py": "3d088541fd6a42ac555e02796cf1926f9a9fcd8e44ad1f22ae854be4b55204bc",
    "src/clickagent/advantage_validation.py": "19c02014e54aed208d1252b9ca8c781da415d1eb4422ec8b4cf215e8318627aa",
    "runs/advantage_validation_v17/data/cohorts.json": "8d2f48b19a4d0217502df3601788db4e3b9bbc59ef50b3f73b1f06085e610a11",
}
FULL_SHA = "3cddbb9feb60bfba43d02a56c09fe66907d7f906531716eb8994ddce67f9523c"
METHODS = ("full", "mvp")


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def gpu_snapshot() -> dict:
    query = ["nvidia-smi", "--query-gpu=index,name,uuid,memory.total,memory.used,utilization.gpu",
             "--format=csv,noheader,nounits"]
    try:
        done = subprocess.run(query, check=True, capture_output=True, text=True, timeout=15)
        return {"command": query, "rows": [s.strip() for s in done.stdout.splitlines() if s.strip()]}
    except Exception as exc:
        return {"command": query, "error": repr(exc)}


def verify_archives() -> dict:
    for relative, expected in V17_HASHES.items():
        observed = sha(ROOT / relative)
        if observed != expected:
            raise RuntimeError(f"Archived V17 input drift: {relative}: {observed}")
    if sha(FULL) != FULL_SHA:
        raise RuntimeError("Frozen full S1 checkpoint SHA mismatch")
    frozen = read(ROOT / "runs/paper_scale_ablation_v1/models/frozen.json")
    row = frozen["models"]["full_aligned_s1"]
    if Path(row["path"]).resolve() != FULL.resolve() or row["sha256"] != FULL_SHA:
        raise RuntimeError("Full S1 is not the V2-registered full_aligned_s1")
    old_full = read(V17 / "inference/ours_s1_p262144_speed/protocol.json")
    old_mvp = read(V17 / "inference/mvp_p262144_all/protocol.json")
    if old_full["method"] != "ours_s1" or old_full["max_pixels"] != 262144:
        raise RuntimeError("Unexpected V17 our-S1 timing protocol")
    if old_mvp["method"] != "mvp" or old_mvp["max_pixels"] != 262144:
        raise RuntimeError("Unexpected V17 MVP timing protocol")
    return {
        "v17_source_hashes": V17_HASHES,
        "full_s1_sha256": FULL_SHA,
        "v1_frozen_registry_sha256": sha(ROOT / "runs/paper_scale_ablation_v1/models/frozen.json"),
        "v17_ours_protocol_sha256": sha(V17 / "inference/ours_s1_p262144_speed/protocol.json"),
        "v17_mvp_protocol_sha256": sha(V17 / "inference/mvp_p262144_all/protocol.json"),
        "mvp_repo_revision": read(V17 / "plan.json")["official_repositories"]["MVP"],
    }


def _public(case: dict) -> dict:
    return {k: case[k] for k in ("sample_id", "dataset", "image", "instruction", "image_sha256")}


def prepare_inputs() -> dict:
    cohorts = read(V17 / "data/cohorts.json")
    selected = [_public(c) for c in cohorts["speed_robustness"]]
    if len(selected) != 60 or len({c["sample_id"] for c in selected}) != 60:
        raise RuntimeError("V17 speed cohort must be 60 unique queries")
    reserve = {c["sample_id"]: c for c in (
        json.loads(s) for s in (ROOT / "runs/quality_readout_v15_3k/data/new_reserve.jsonl")
        .read_text(encoding="utf-8").splitlines() if s.strip())}
    warm = [_public(reserve[sid]) for sid in [c["sample_id"] for c in cohorts["warmup"]]]
    if len(warm) != 2 or any(c["sample_id"] in {r["sample_id"] for r in selected} for c in warm):
        raise RuntimeError("V17 warmup cases are not two separate reserve examples")
    for case in selected + warm:
        if sha(Path(case["image"])) != case["image_sha256"]:
            raise RuntimeError(f"Source image hash changed: {case['sample_id']}")
    return {"speed": selected, "warmup": warm, "labels_in_input": False}


def register() -> dict:
    archived = verify_archives()
    inputs = prepare_inputs()
    input_path = OUT / "public_inputs.json"
    protocol_path = OUT / "protocol.json"
    if protocol_path.exists():
        protocol = read(protocol_path)
        if sha(input_path) != protocol["public_inputs_sha256"] or sha(Path(__file__)) != protocol["worker_sha256"]:
            raise RuntimeError("Registered cost protocol or public input drift")
        if protocol["archived"] != archived:
            raise RuntimeError("Archive identity drift after cost registration")
        return protocol
    if input_path.exists():
        if read(input_path) != inputs:
            raise RuntimeError("Existing public input manifest differs")
    else:
        write(input_path, inputs)
    protocol = {
        "registered_at_utc": now(),
        "purpose": "Retrospective same-cohort end-to-end cost, frozen full aligned S1 versus local V17 MVP",
        "worker_sha256": sha(Path(__file__)),
        "public_inputs_sha256": sha(input_path),
        "archived": archived,
        "methods": ["full_aligned_s1", "V17_mvp"],
        "cohort": "V17 speed_robustness: fixed 60 (30 ScreenSpot-Pro, 30 VenusBench-GD)",
        "input_image_count": 60,
        "repetitions": 2,
        "warmup_cases_per_process": 2,
        "order": "numpy default_rng(20261705).permutation(60), repeated in two passes as V17",
        "batch_size": 1,
        "max_pixels_per_view": 262144,
        "our_selection": "original K16, stable descending score, forced Top1; no union/geometry/coordinate-center",
        "mvp_selection": "V17 pinned MVP adapter; max_subimages=2, consistency_threshold=14",
        "timing": "V17 infer_once: CUDA sync and reset peaks before, perf_counter starts before PNG decode; includes decode, preprocessing, all model calls and score; sync after inference; excludes image file read/hash, model load, two warmups, result writes, reporting; no feature/prediction/KV cache across queries",
        "process_isolation": "Full and MVP must be run as separate sequential Python processes on the same two GPUs, with no other GPU jobs",
        "statistics": "All 120 calls: mean/median/p95 latency, forward calls, processed pixels, per-GPU peak allocated/reserved; paired 60-screen mean difference and ratio with image-cluster bootstrap95%, seed 20261113",
        "claim_scope": "Retrospective observed cost on reused V17 cohort; V17 low-resolution local MVP, not published high-resolution result",
    }
    write(protocol_path, protocol)
    return protocol


def verify_registration() -> dict:
    protocol = register()
    inputs = read(OUT / "public_inputs.json")
    if len(inputs["speed"]) != 60 or len(inputs["warmup"]) != 2:
        raise RuntimeError("Public manifest denominator changed")
    return protocol


class FullOurRuntime:
    """Exact V17 OurRuntime.infer implementation with registered full S1 readout."""

    def __new__(cls, folder: Path):
        import torch
        from clickagent.advantage_runtimes import OurRuntime, RuntimeCounter
        from clickagent.conditioned_features import ConditionedFeatureRuntime
        from clickagent.evidence_retention import RetentionModel
        import run_quality_readout_v15 as base
        import run_contribution_v16 as contribution

        obj = object.__new__(OurRuntime)
        legacy = base.Extractor()
        config = replace(legacy.config, offload_folder=str(folder / "offload"))
        obj.proposer, obj.base, obj.contribution = legacy.proposer.requires_grad_(False), base, contribution
        obj.runtime = ConditionedFeatureRuntime(config)
        saved = torch.load(FULL, map_location="cpu", weights_only=True)
        model = RetentionModel().to("cuda:0").eval().requires_grad_(False)
        model.load_state_dict(saved["state_dict"])
        obj.models = [(model, saved)]
        obj.counter = RuntimeCounter()
        obj.counter.attach(obj.runtime.model, 16)
        obj.metadata = dict(method="our_full_aligned_s1", mainline="full_aligned",
                            readout_checkpoint=str(FULL), readout_sha256=FULL_SHA,
                            feature_storage="In-memory; V17 extraction/proposal/state/readout mathematics",
                            model_metadata=obj.runtime.runtime_metadata, scorer_device="cuda:0",
                            scorer_seeds=[1], primary_timing_seed="S1", max_pixels=262144,
                            readout_parameters_per_model=22050, proposer_parameters=611919)
        return obj


def make_runtime(method: str, folder: Path):
    from clickagent.advantage_runtimes import configure_paths, create_runtime
    configure_paths("ours_s1" if method == "full" else "mvp")
    return FullOurRuntime(folder) if method == "full" else create_runtime("mvp", folder, 262144)


def _call(runtime, case, torch):
    from run_advantage_inference_v17 import infer_once, payload_for
    payload, _ = payload_for(case, "clean")  # File read/hash excluded, as in V17.
    result, measured = infer_once(runtime, payload, case["instruction"], torch)
    return {"sample_id": case["sample_id"], "output": result, "measured": measured,
            "input_image_sha256": case["image_sha256"]}


def _v2_expected(sample_id: str) -> dict:
    folder = ROOT / "runs/paper_scale_ablation_v2/eval/full/predictions/cases"
    matches = []
    for path in folder.glob("*.json"):
        row = read(path)
        if row.get("sample_id") == sample_id:
            matches.append(row)
    if len(matches) != 1:
        raise RuntimeError(f"V2 prediction for smoke sample has {len(matches)} matches")
    return matches[0]


def smoke() -> None:
    verify_registration()
    import torch
    torch.set_num_threads(4)
    if [torch.cuda.get_device_name(d) for d in range(torch.cuda.device_count())] != ["Quadro P5000"] * 2:
        raise RuntimeError("This V17-matched cost run requires the original two Quadro P5000 GPUs")
    folder = OUT / "smoke"
    inputs = read(OUT / "public_inputs.json")
    case = next(c for c in inputs["speed"] if c["sample_id"].startswith("screenspot_pro:"))
    runtime = make_runtime("full", folder)
    observed = _call(runtime, case, torch)
    expected = _v2_expected(case["sample_id"])
    same_point = observed["output"]["point"] == expected["methods"]["full_aligned_s1"]["point"]
    same_k = len(runtime.last_record["baseline"]) == expected["K16_count"] == 16
    same_n = len(runtime.last_record["points"]) == expected["candidate_count"]
    receipt = {"at_utc": now(), "sample_id": case["sample_id"], "full_s1_point_parity": same_point,
               "K16_count_parity": same_k, "candidate_count_parity": same_n,
               "observed_point": observed["output"]["point"],
               "expected_point": expected["methods"]["full_aligned_s1"]["point"],
               "gpu_snapshot": gpu_snapshot(), "not_paper_timing": True}
    write(folder / "parity_receipt.json", receipt)
    if not all((same_point, same_k, same_n)):
        raise RuntimeError(f"V2 original-K16/full S1 parity failed: {receipt}")
    print(json.dumps({"smoke": "passed", **receipt}, ensure_ascii=False), flush=True)


def run(method: str) -> None:
    verify_registration()
    import numpy as np
    import torch
    from run_advantage_inference_v17 import sync

    torch.set_num_threads(4)
    gpu_names = [torch.cuda.get_device_name(d) for d in range(torch.cuda.device_count())]
    if gpu_names != ["Quadro P5000"] * 2:
        raise RuntimeError(f"V17 hardware mismatch: {gpu_names}")
    folder = OUT / "runs" / method
    status_path = folder / "status.json"
    if status_path.exists() or (folder / "cases").exists():
        raise RuntimeError(f"Existing attempt at {folder}; preserve it and register a new version if interrupted")
    inputs = read(OUT / "public_inputs.json")
    write(status_path, {"phase": "loading", "at_utc": now(), "method": method})
    start_load = time.perf_counter()
    try:
        runtime = make_runtime(method, folder)
        sync(torch)
        metadata = {**runtime.metadata, "load_seconds": time.perf_counter() - start_load,
                    "torch_version": torch.__version__, "transformers_version": __import__("transformers").__version__,
                    "cuda_version": torch.version.cuda, "gpus": gpu_names,
                    "gpu_before_warmup": gpu_snapshot()}
        write(folder / "runtime.json", metadata)
        warm = [_call(runtime, case, torch) for case in inputs["warmup"]]
        write(folder / "warmup.json", {"n": 2, "not_paper_timing": True, "rows": warm})
        write(status_path, {"phase": "timing", "at_utc": now(), "method": method,
                            "completed": 0, "total": 120, "gpu_after_warmup": gpu_snapshot()})
        selected = inputs["speed"]
        rng = np.random.default_rng(20261705)
        rows = []
        for repetition in range(2):
            for j, index in enumerate(rng.permutation(len(selected)), 1):
                case = selected[int(index)]
                row = _call(runtime, case, torch)
                row["repetition"] = repetition
                row["ordinal"] = repetition * len(selected) + j
                write(folder / f"cases/{row['ordinal']:03d}.json", row)
                rows.append(row)
                if j % 5 == 0 or j == len(selected):
                    state = {"phase": "timing", "at_utc": now(), "method": method,
                             "completed": len(rows), "total": 120}
                    write(status_path, state)
                    print(json.dumps(state), flush=True)
        write(folder / "results.json", {"complete": True, "method": method,
                                        "unique_screenshots": 60, "repetitions": 2, "n": 120,
                                        "rows": rows, "no_inference_cache": True})
        write(folder / "completion.json", {"complete": True, "at_utc": now(),
                                           "protocol_sha256": sha(OUT / "protocol.json"),
                                           "public_inputs_sha256": sha(OUT / "public_inputs.json"),
                                           "runtime_sha256": sha(folder / "runtime.json"),
                                           "warmup_sha256": sha(folder / "warmup.json"),
                                           "results_sha256": sha(folder / "results.json"),
                                           "gpu_after_run": gpu_snapshot()})
        write(status_path, {"phase": "complete", "at_utc": now(), "method": method,
                            "completed": 120, "total": 120})
    except Exception:
        write(status_path, {"phase": "failed", "at_utc": now(), "method": method,
                            "error": traceback.format_exc()})
        raise


def analyze() -> None:
    verify_registration()
    import numpy as np
    from collections import defaultdict

    expected = [c["sample_id"] for c in read(OUT / "public_inputs.json")["speed"]]
    summaries, by_case = {}, {}
    for method in METHODS:
        folder = OUT / "runs" / method
        completion = read(folder / "completion.json")
        if not completion["complete"] or completion["protocol_sha256"] != sha(OUT / "protocol.json"):
            raise RuntimeError(f"Incomplete or different protocol: {method}")
        for name in ("runtime", "warmup", "results"):
            if completion[f"{name}_sha256"] != sha(folder / f"{name}.json"):
                raise RuntimeError(f"Cost artifact hash mismatch: {method}/{name}")
        rows = read(folder / "results.json")["rows"]
        if len(rows) != 120:
            raise RuntimeError(f"Not 120 timed calls: {method}")
        grouped = defaultdict(list)
        for row in rows:
            grouped[row["sample_id"]].append(row)
        if set(grouped) != set(expected) or any(sorted(x["repetition"] for x in group) != [0, 1] for group in grouped.values()):
            raise RuntimeError(f"Cost cohort or repetitions mismatch: {method}")
        by_case[method] = grouped
        timing = np.array([r["measured"]["elapsed_seconds"] for r in rows], float)
        calls = np.array([r["measured"]["counter"]["backbone_forward_calls"] for r in rows], float)
        pixels = np.array([r["measured"]["counter"]["processed_pixels"] for r in rows], float)
        allocated = np.array([r["measured"]["peak_allocated_bytes"] for r in rows], float)
        reserved = np.array([r["measured"]["peak_reserved_bytes"] for r in rows], float)
        summaries[method] = {
            "n": 120,
            "seconds": {"mean": float(timing.mean()), "median": float(np.median(timing)),
                        "p95": float(np.quantile(timing, .95))},
            "backbone_forward_calls": {"mean": float(calls.mean()), "median": float(np.median(calls)),
                                       "p95": float(np.quantile(calls, .95))},
            "processed_pixels": {"mean": float(pixels.mean()), "median": float(np.median(pixels)),
                                 "p95": float(np.quantile(pixels, .95))},
            "peak_allocated_bytes_by_gpu": {"max": allocated.max(axis=0).astype(int).tolist(),
                                            "mean": allocated.mean(axis=0).tolist()},
            "peak_reserved_bytes_by_gpu": {"max": reserved.max(axis=0).astype(int).tolist(),
                                           "mean": reserved.mean(axis=0).tolist()},
        }
    full_vec = np.array([np.mean([r["measured"]["elapsed_seconds"] for r in by_case["full"][sid]]) for sid in expected])
    mvp_vec = np.array([np.mean([r["measured"]["elapsed_seconds"] for r in by_case["mvp"][sid]]) for sid in expected])
    rng = np.random.default_rng(20261113)
    sample = rng.integers(0, 60, size=(10000, 60))
    difference = (full_vec[sample] - mvp_vec[sample]).mean(axis=1)
    ratios = full_vec[sample].mean(axis=1) / mvp_vec[sample].mean(axis=1)
    paired = {"n_screenshots": 60, "unit": "same screenshot mean of two repeats",
              "full_minus_mvp_mean_seconds": float((full_vec - mvp_vec).mean()),
              "full_minus_mvp_CI95_seconds": np.quantile(difference, [.025, .975]).tolist(),
              "full_over_mvp_mean_ratio": float(full_vec.mean() / mvp_vec.mean()),
              "full_over_mvp_CI95_ratio": np.quantile(ratios, [.025, .975]).tolist(),
              "bootstrap_seed": 20261113, "bootstrap_draws": 10000}
    result = {"complete": True, "at_utc": now(), "protocol_sha256": sha(OUT / "protocol.json"),
              "summaries": summaries, "paired": paired,
              "scope": "Retrospective V17 60-image cost, separate sequential processes, original two P5000 GPUs"}
    analysis_dir = OUT / "analysis"
    write(analysis_dir / "results.json", result)
    lines = ["Frozen full aligned S1 versus local V17 MVP: end-to-end cost", "",
             "60 fixed V17 screenshots × 2 uncached repetitions; two independent reserve warmups per method.",
             "Two Quadro P5000 GPUs, batch 1, FP16, 262144 pixels/view; original K16 forced Top1 for full.",
             "Timing includes PNG decode through synchronized inference; excludes image file read/hash, load, warmup, writes.", ""]
    for method in METHODS:
        s = summaries[method]
        lines += [f"{method}: mean {s['seconds']['mean']:.3f}s, median {s['seconds']['median']:.3f}s, p95 {s['seconds']['p95']:.3f}s; "
                  f"forward calls mean {s['backbone_forward_calls']['mean']:.2f}; processed pixels mean {s['processed_pixels']['mean']:.0f}; "
                  f"peak allocated per GPU {s['peak_allocated_bytes_by_gpu']['max']} bytes; "
                  f"peak reserved per GPU {s['peak_reserved_bytes_by_gpu']['max']} bytes."]
    lines += ["", f"Paired full−MVP mean {paired['full_minus_mvp_mean_seconds']:.3f}s, "
              f"95% image-cluster bootstrap CI {paired['full_minus_mvp_CI95_seconds']}; "
              f"mean ratio {paired['full_over_mvp_mean_ratio']:.3f}, CI {paired['full_over_mvp_CI95_ratio']}.",
              "These are retrospective local 262144-pixel MVP measurements, not its published high-resolution setting."]
    (analysis_dir / "readout.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write(analysis_dir / "completion.json", {"complete": True, "results_sha256": sha(analysis_dir / "results.json"),
                                            "readout_sha256": sha(analysis_dir / "readout.txt")})
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("register", "verify", "smoke", "run", "analyze"))
    parser.add_argument("--method", choices=METHODS)
    args = parser.parse_args()
    if args.stage == "register":
        print(json.dumps(register(), ensure_ascii=False, indent=2))
    elif args.stage == "verify":
        print(json.dumps({"verified": True, "protocol_sha256": sha(OUT / "protocol.json"),
                          "worker_sha256": verify_registration()["worker_sha256"]}, ensure_ascii=False))
    elif args.stage == "smoke":
        smoke()
    elif args.stage == "run":
        if args.method is None:
            parser.error("--stage run requires --method full or mvp")
        run(args.method)
    else:
        analyze()
