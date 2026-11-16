"""Frozen full-aligned, original-K16 inference at a 1,048,576-pixel view cap.

This retrospective pixel-budget variant reads only the sealed V2 public inputs.
It produces and SHA-seals predictions before any separate label analysis. The
262,144-pixel paper mainline and all archived workers remain untouched.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_high_pixel_full_timing_pilot_v1 as pilot
import run_full_mainline_cost_v1 as cost

OUT = ROOT / "runs/high_pixel_full_inference_v1"
PILOT = ROOT / "runs/high_pixel_full_timing_pilot_v1"
REGISTRY = ROOT / "runs/paper_scale_ablation_v1/models/frozen.json"
SSP_SOURCE = ROOT / "runs/paper_scale_ablation_v2/eval/full/public.json"
POINTER_SOURCE = ROOT / "runs/paper_scale_ablation_v2/pointerbench/public.jsonl"
SUITES = ("ssp", "pro", "sheets")
COUNTS = {"ssp": 1581, "pro": 500, "sheets": 500}
MODEL_NAMES = tuple(f"full_aligned_s{i}" for i in (1, 2, 3))
BUDGET = 1048576


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    return cost.sha(path)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write(path: Path, value) -> None:
    cost.write(path, value)


def identity(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def key(sample_id: str) -> str:
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:24]


def manifest_from_sources() -> dict:
    ssp = read(SSP_SOURCE)
    pointer = [json.loads(line) for line in POINTER_SOURCE.read_text(encoding="utf-8-sig").splitlines()
               if line.strip()]
    if len(ssp) != COUNTS["ssp"] or len(pointer) != COUNTS["pro"] + COUNTS["sheets"]:
        raise ValueError("V2 public cohort denominator changed")
    rows = {"ssp": [pilot.public_row(row, "ssp") for row in ssp],
            "pro": [pilot.public_row(row, "pro") for row in pointer if row["suite"] == "pro"],
            "sheets": [pilot.public_row(row, "sheets") for row in pointer if row["suite"] == "sheets"]}
    if any(len(rows[suite]) != COUNTS[suite] for suite in SUITES):
        raise ValueError("V2 public suite denominator changed")
    ids = [row["sample_id"] for suite in SUITES for row in rows[suite]]
    if len(ids) != len(set(ids)) or len(ids) != len({key(sid) for sid in ids}):
        raise ValueError("Duplicate public sample or case file key")
    if any(set(row) != {"sample_id", "dataset", "image", "image_size", "image_sha256",
                            "instruction", "pixel_sha256", "suite"}
           for suite in SUITES for row in rows[suite]):
        raise ValueError("Public manifest has unexpected fields")
    return {"complete": True, "labels_present": False, "counts": COUNTS, "cases": rows}


def source_hashes() -> dict:
    paths = [SSP_SOURCE, POINTER_SOURCE, REGISTRY,
             ROOT / "scripts/run_high_pixel_full_timing_pilot_v1.py",
             ROOT / "scripts/run_full_mainline_cost_v1.py",
             ROOT / "scripts/run_advantage_inference_v17.py",
             ROOT / "scripts/run_quality_readout_v15.py",
             ROOT / "scripts/run_contribution_v16.py",
             ROOT / "src/clickagent/advantage_runtimes.py",
             ROOT / "src/clickagent/conditioned_features.py",
             ROOT / "src/clickagent/action_model.py",
             ROOT / "src/clickagent/evidence_retention.py",
             PILOT / "protocol.json", PILOT / "public_inputs.json",
             PILOT / "p1048576/results.json", PILOT / "p1048576/completion.json"]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path) for path in paths}


def model_registry() -> dict:
    frozen = read(REGISTRY)
    if frozen.get("complete") is not True:
        raise ValueError("V2 full model registry is incomplete")
    models = {name: frozen["models"][name] for name in MODEL_NAMES}
    for name, row in models.items():
        if row["variant"] != "aligned" or row["sha256"] != sha(Path(row["path"])):
            raise ValueError(f"Frozen full model changed: {name}")
    if models["full_aligned_s1"]["sha256"] != pilot.FULL_SHA:
        raise ValueError("S1 differs from proven high-pixel pilot")
    return {name: {"path": row["path"], "sha256": row["sha256"], "seed": row["seed"]}
            for name, row in models.items()}


def pilot_rows() -> dict:
    receipt = read(PILOT / "p1048576/completion.json")
    protocol = read(PILOT / "protocol.json")
    if receipt.get("complete") is not True or receipt["budget"] != BUDGET or \
       receipt["protocol_sha256"] != sha(PILOT / "protocol.json") or \
       protocol["worker_sha256"] != sha(ROOT / "scripts/run_high_pixel_full_timing_pilot_v1.py") or \
       receipt["results_sha256"] != sha(PILOT / "p1048576/results.json"):
        raise ValueError("High-pixel pilot SHA seal changed")
    rows = read(PILOT / "p1048576/results.json")["rows"]
    if len(rows) != 40 or len({row["sample_id"] for row in rows}) != 40:
        raise ValueError("High-pixel pilot overlap changed")
    return {row["sample_id"]: row for row in rows}


def register() -> None:
    models = model_registry()
    overlap = pilot_rows()
    expected_manifest = manifest_from_sources()
    all_ids = {row["sample_id"] for suite in SUITES for row in expected_manifest["cases"][suite]}
    if not set(overlap) <= all_ids:
        raise ValueError("Pilot overlap is outside registered V2 public cohorts")
    mp = OUT / "public_manifest.json"
    if mp.exists():
        if read(mp) != expected_manifest:
            raise ValueError("Public input manifest changed")
    else:
        write(mp, expected_manifest)
    expected = {"complete": True, "registered_at_utc": None,
                "purpose": "Retrospective high-pixel full aligned S1-S3 unlabeled inference; no model selection",
                "pixel_cap_per_view": BUDGET, "suites": list(SUITES), "denominators": COUNTS,
                "scientific_policy": "V2 frozen full aligned S1-S3; original V15 K16; stable forced Top1; unchanged prompt, crop, proposer, scaler and weights",
                "paper_mainline_262144_unchanged": True,
                "labels_may_open_in_this_worker": False,
                "analysis_policy": "No accuracy or label access in this worker; first seal all 2581 public predictions, then use a separate registered analysis",
                "worker_sha256": sha(Path(__file__)), "public_manifest_sha256": sha(mp),
                "source_hashes": source_hashes(), "models": models,
                "pilot_overlap_n": len(overlap),
                "pilot_1m_results_sha256": sha(PILOT / "p1048576/results.json"),
                "failure_policy": "Stop and record exact case; resume same code and parameter from validated cache; do not lower pixel cap or omit cases"}
    pp = OUT / "protocol.json"
    if pp.exists():
        observed = read(pp)
        expected["registered_at_utc"] = observed.get("registered_at_utc")
        if observed != expected:
            raise ValueError("Frozen high-pixel full protocol drift")
    else:
        expected["registered_at_utc"] = now()
        write(pp, expected)
    print(json.dumps({"phase": "registered", "protocol_sha256": sha(pp),
                      "n": sum(COUNTS.values()), "pilot_overlap_n": len(overlap)},
                     ensure_ascii=False), flush=True)


def gate() -> tuple[dict, dict]:
    pp = OUT / "protocol.json"
    if not pp.exists():
        raise ValueError("Register and freeze protocol before any model forward")
    protocol = read(pp)
    if protocol.get("complete") is not True or protocol["pixel_cap_per_view"] != BUDGET or \
       protocol["worker_sha256"] != sha(Path(__file__)) or \
       protocol["source_hashes"] != source_hashes() or \
       protocol["models"] != model_registry() or \
       protocol["pilot_1m_results_sha256"] != sha(PILOT / "p1048576/results.json") or \
       protocol["labels_may_open_in_this_worker"] is not False:
        raise ValueError("Frozen protocol or source SHA changed")
    manifest = read(OUT / "public_manifest.json")
    if sha(OUT / "public_manifest.json") != protocol["public_manifest_sha256"] or \
       manifest != manifest_from_sources() or manifest["labels_present"] is not False:
        raise ValueError("Public manifest SHA or source changed")
    pilot_rows()
    return protocol, manifest


def make_runtime(protocol: dict):
    import torch
    from clickagent.evidence_retention import RetentionModel

    obj = pilot.make_runtime(BUDGET, OUT)
    if obj.metadata["checkpoint_sha256"] != protocol["models"][MODEL_NAMES[0]]["sha256"]:
        raise ValueError("Pilot S1 load drift")
    for name in MODEL_NAMES[1:]:
        saved = torch.load(protocol["models"][name]["path"], map_location="cpu", weights_only=True)
        model = RetentionModel().to("cuda:0").eval().requires_grad_(False)
        model.load_state_dict(saved["state_dict"])
        obj.models.append((model, saved))
    obj.metadata.update(method="full_aligned_s1_s2_s3_high_pixel_1m",
                        scorer_models=list(MODEL_NAMES),
                        scorer_model_sha256={name: protocol["models"][name]["sha256"] for name in MODEL_NAMES},
                        candidate_policy="original_K16_forced_Top1")
    return obj


def validate_case(row: dict, case: dict, models: dict, overlap: dict) -> None:
    if row.get("complete") is not True or row.get("labels_opened") is not False or \
       row.get("sample_id") != case["sample_id"] or row.get("suite") != case["suite"] or \
       row.get("public_identity") != identity(case) or row.get("input_image_sha256") != case["image_sha256"] or \
       row.get("pixel_cap_per_view") != BUDGET or row.get("model_hashes") != models:
        raise ValueError("Cached high-pixel prediction identity mismatch: " + case["sample_id"])
    points = row["seed_points"]
    if len(points) != 3 or row["point"] != points[0] or \
       row["k16_count"] != len(row["k16_points"]) or \
       row["candidate_count"] != len(row["candidate_points"]):
        raise ValueError("Malformed high-pixel prediction: " + case["sample_id"])
    width, height = case["image_size"]
    for point in points + row["k16_points"] + row["candidate_points"]:
        if point is not None and (len(point) != 2 or not (0 <= point[0] < width and 0 <= point[1] < height)):
            raise ValueError("Out-of-bounds predicted point: " + case["sample_id"])
    if row["k16_count"] > 16 or row["candidate_count"] < row["k16_count"]:
        raise ValueError("K16/candidate count invalid: " + case["sample_id"])
    old = overlap.get(case["sample_id"])
    if old and (points[0] != old["output"]["point"] or
                row["k16_count"] != old["k16_count"] or
                row["candidate_count"] != old["candidate_count"]):
        raise ValueError("Frozen S1/K16 pilot parity failed: " + case["sample_id"])


def seal_suite(suite: str, cases: list[dict], models: dict, overlap: dict,
               protocol_sha: str, manifest_sha: str) -> dict:
    folder = OUT / suite
    rows, case_hashes = [], {}
    for case in cases:
        path = folder / f"cases/{key(case['sample_id'])}.json"
        row = read(path)
        validate_case(row, case, models, overlap)
        rows.append(row)
        case_hashes[case["sample_id"]] = sha(path)
    output = folder / "predictions.json"
    expected = {"complete": True, "suite": suite, "n": len(rows), "pixel_cap_per_view": BUDGET,
                "protocol_sha256": protocol_sha, "public_manifest_sha256": manifest_sha,
                "model_hashes": models, "case_hashes": case_hashes, "rows": rows,
                "labels_opened": False}
    if output.exists():
        if read(output) != expected:
            raise ValueError("Existing sealed predictions changed: " + suite)
    else:
        write(output, expected)
    receipt = folder / "completion.json"
    expected_receipt = {"complete": True, "suite": suite, "n": len(rows),
                        "predictions_sha256": sha(output), "protocol_sha256": protocol_sha,
                        "labels_opened": False}
    if receipt.exists():
        if read(receipt) != expected_receipt:
            raise ValueError("Existing suite seal changed: " + suite)
    else:
        write(receipt, expected_receipt)
    return expected_receipt


def seal_all(protocol: dict, manifest: dict) -> None:
    psha = sha(OUT / "protocol.json")
    msha = sha(OUT / "public_manifest.json")
    models = {name: protocol["models"][name]["sha256"] for name in MODEL_NAMES}
    overlap = pilot_rows()
    receipts = {suite: seal_suite(suite, manifest["cases"][suite], models, overlap, psha, msha)
                for suite in SUITES}
    all_ids = {case["sample_id"] for suite in SUITES for case in manifest["cases"][suite]}
    if not set(overlap) <= all_ids:
        raise ValueError("Pilot overlap missing from full cohorts")
    output = OUT / "public_completion.json"
    expected = {"complete": True, "n": sum(COUNTS.values()), "suite_n": COUNTS,
                "protocol_sha256": psha, "public_manifest_sha256": msha,
                "suite_receipt_sha256": {suite: sha(OUT / suite / "completion.json") for suite in SUITES},
                "pilot_overlap_n": len(overlap), "pilot_s1_k16_parity": True,
                "labels_opened": False}
    if output.exists():
        if read(output) != expected:
            raise ValueError("Existing global seal changed")
    else:
        write(output, expected)
    write(OUT / "status.json", {"phase": "complete", "at_utc": now(), "completed": expected["n"],
                                "total": expected["n"], "public_completion_sha256": sha(output)})
    print(json.dumps({"phase": "complete", "n": expected["n"],
                      "public_completion_sha256": sha(output)}, ensure_ascii=False), flush=True)


def run() -> None:
    protocol, manifest = gate()
    import torch
    from run_advantage_inference_v17 import sync

    torch.set_num_threads(4)
    gpu_names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    if len(gpu_names) != 2:
        raise RuntimeError("Expected the proven two-GPU high-pixel host")
    total = sum(COUNTS.values())
    cases_all = [(suite, case) for suite in SUITES for case in manifest["cases"][suite]]
    overlap = pilot_rows()
    models = {name: protocol["models"][name]["sha256"] for name in MODEL_NAMES}
    existing = 0
    for suite, case in cases_all:
        path = OUT / suite / f"cases/{key(case['sample_id'])}.json"
        if path.exists():
            validate_case(read(path), case, models, overlap)
            existing += 1
    if existing == total:
        seal_all(protocol, manifest)
        return
    started = time.perf_counter()
    current = None
    runtime = None
    try:
        runtime = make_runtime(protocol)
        sync(torch)
        load_seconds = time.perf_counter() - started
        runtime_info = {**runtime.metadata, "gpu_names": gpu_names,
                        "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                        "load_seconds": load_seconds, "gpu_at_start": cost.gpu_snapshot()}
        if not (OUT / "runtime.json").exists():
            write(OUT / "runtime.json", runtime_info)
        for case in read(PILOT / "public_inputs.json")["warmup"]:
            pilot.one_call(runtime, case, torch)
        completed = existing
        for suite, case in cases_all:
            current = case["sample_id"]
            path = OUT / suite / f"cases/{key(current)}.json"
            if path.exists():
                continue
            result = pilot.one_call(runtime, case, torch)
            output = result["output"]
            record = runtime.last_record
            row = {"complete": True, "sample_id": current, "suite": suite,
                   "public_identity": identity(case), "input_image_sha256": case["image_sha256"],
                   "image_size": case["image_size"], "pixel_cap_per_view": BUDGET,
                   "model_hashes": models, "point": output["point"],
                   "seed_points": output["seed_points"],
                   "k16_points": output["candidates"],
                   "candidate_points": record["points"].tolist(),
                   "candidate_count": result["candidate_count"],
                   "k16_count": result["k16_count"], "views": output["views"],
                   "view_trace": result["view_trace"], "measured": result["measured"],
                   "ground_truth_in_model_input": output["ground_truth_in_model_input"],
                   "labels_opened": False}
            validate_case(row, case, models, overlap)
            write(path, row)
            runtime.last_record = None
            runtime.last_pipe = None
            completed += 1
            state = {"phase": "running", "at_utc": now(), "completed": completed,
                     "total": total, "suite": suite, "latest_sample_id": current,
                     "elapsed_seconds_this_process": round(time.perf_counter() - started, 1),
                     "resumed_existing": existing}
            write(OUT / "status.json", state)
            if completed % 5 == 0 or completed == total:
                print(json.dumps(state, ensure_ascii=False), flush=True)
        seal_all(protocol, manifest)
    except Exception:
        stamp = now().replace(":", "-")
        failure = {"phase": "failed", "at_utc": now(), "sample_id": current,
                   "completed_before_process": existing, "error": traceback.format_exc()}
        write(OUT / f"failures/{stamp}.json", failure)
        write(OUT / "status.json", failure)
        raise


def verify_gate() -> None:
    protocol, manifest = gate()
    receipt = read(OUT / "public_completion.json")
    if receipt.get("complete") is not True or receipt.get("n") != sum(COUNTS.values()) or \
       receipt.get("labels_opened") is not False or \
       receipt["protocol_sha256"] != sha(OUT / "protocol.json") or \
       receipt["public_manifest_sha256"] != sha(OUT / "public_manifest.json"):
        raise ValueError("Global prediction completion receipt invalid")
    for suite in SUITES:
        if receipt["suite_receipt_sha256"][suite] != sha(OUT / suite / "completion.json"):
            raise ValueError("Suite completion receipt SHA changed: " + suite)
    seal_all(protocol, manifest)
    print(json.dumps({"phase": "verified", "n": receipt["n"],
                      "all_predictions_sha_sealed": True, "labels_opened": False},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("register", "all", "verify-gate"))
    args = parser.parse_args()
    if args.stage == "register":
        register()
    elif args.stage == "all":
        run()
    else:
        verify_gate()
