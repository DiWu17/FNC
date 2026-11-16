"""Label-free paired pixel-budget timing pilot for frozen full_aligned_s1.

This is an isolated retrospective resource experiment.  It never opens target
labels and never mutates the 262144-pixel paper pipeline or its checkpoints.
Register (CPU only) before either GPU arm, then run each budget in a separate
Python process.  A failed/OOM case is recorded and stops that arm; no cap
fallback or denominator change is permitted.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_full_mainline_cost_v1 as cost

OUT = ROOT / "runs/high_pixel_full_timing_pilot_v1"
SSP_SOURCE = ROOT / "runs/paper_scale_ablation_v2/eval/full/public.json"
PTR_SOURCE = ROOT / "runs/paper_scale_ablation_v2/pointerbench/public.jsonl"
FULL = ROOT / "runs/full_scaling_v18/models/full_s1.pt"
FULL_SHA = "3cddbb9feb60bfba43d02a56c09fe66907d7f906531716eb8994ddce67f9523c"
BUDGETS = (262144, 1048576)
SSP_N = 24
POINTER_PER_SUITE = 8
SELECTION_SALT = "full_pixel_timing_v1"
ORDER_SEED = 20261114


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    return cost.sha(path)


def read(path: Path):
    return cost.read(path)


def write(path: Path, value) -> None:
    cost.write(path, value)


def rank(row: dict) -> str:
    return hashlib.sha256((SELECTION_SALT + "|" + row["sample_id"]).encode()).hexdigest()


def public_row(row: dict, suite: str) -> dict:
    allowed = {"sample_id", "dataset", "image", "image_size", "image_sha256",
               "instruction", "pixel_sha256"}
    if not allowed <= set(row):
        raise ValueError("Public source missing fields")
    if set(row) & {"target_geometry", "bbox", "box", "label", "labels", "answer", "target"}:
        raise ValueError("A target label leaked into a public source")
    value = {key: row[key] for key in sorted(allowed)}
    value["suite"] = suite
    return value


def unique_ranked(rows: list[dict]) -> list[dict]:
    seen = set()
    selected = []
    for row in sorted(rows, key=rank):
        if row["image_sha256"] not in seen:
            selected.append(row)
            seen.add(row["image_sha256"])
    return selected


def choose_ssp(rows: list[dict]) -> tuple[list[dict], dict[str, int]]:
    if len(rows) != 1581 or len({r["sample_id"] for r in rows}) != 1581:
        raise ValueError("ScreenSpot-Pro public cohort changed")
    groups = defaultdict(list)
    for row in rows:
        size = tuple(map(int, row["image_size"]))
        groups[size].append(public_row(row, "ssp"))
    if len(groups) != 14:
        raise ValueError(f"Expected 14 ScreenSpot-Pro image sizes; got {len(groups)}")
    counts = {f"{w}x{h}": len(group) for (w, h), group in groups.items()}
    # One per exact native size ensures rare high-resolution stress cases.
    # Ten extra draws are allocated to common sizes by largest remainder.
    extras_total = SSP_N - len(groups)
    quotas = {size: len(group) / len(rows) * extras_total for size, group in groups.items()}
    extras = {size: int(math.floor(q)) for size, q in quotas.items()}
    remaining = extras_total - sum(extras.values())
    for size in sorted(groups, key=lambda s: (-(quotas[s] - extras[s]), -len(groups[s]), s))[:remaining]:
        extras[size] += 1
    chosen = []
    for size in sorted(groups):
        ranked = unique_ranked(groups[size])
        need = 1 + extras[size]
        if len(ranked) < need:
            raise ValueError(f"Not enough distinct screenshots at {size}")
        chosen.extend(ranked[:need])
    if len(chosen) != SSP_N or len({r["image_sha256"] for r in chosen}) != SSP_N:
        raise ValueError("ScreenSpot-Pro selection invalid")
    return chosen, counts


def choose_inputs() -> dict:
    ssp = read(SSP_SOURCE)
    pointer = [json.loads(line) for line in PTR_SOURCE.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    if len(pointer) != 1000:
        raise ValueError("PointerBench public cohort changed")
    ssp_selected, populations = choose_ssp(ssp)
    pointer_selected = []
    for suite, prefix in (("pro", "pbp_"), ("sheets", "pbs_")):
        group = [public_row(row, suite) for row in pointer if row["suite"] == suite]
        if len(group) != 500 or any(not row["sample_id"].startswith(prefix) for row in group):
            raise ValueError(f"PointerBench {suite} cohort changed")
        pointer_selected.extend(unique_ranked(group)[:POINTER_PER_SUITE])
    selected = ssp_selected + pointer_selected
    if len(selected) != 40 or len({row["sample_id"] for row in selected}) != 40:
        raise ValueError("Pilot must contain 40 unique queries")
    # Identical deterministic order for both budgets; largest SSP last so an
    # OOM stress case cannot erase measurements of lower-resolution strata.
    rng = np.random.default_rng(ORDER_SEED)
    ordered = [selected[int(i)] for i in rng.permutation(len(selected))]
    largest = max((row for row in ordered if row["suite"] == "ssp"),
                  key=lambda row: (row["image_size"][0] * row["image_size"][1], rank(row)))
    ordered.remove(largest)
    ordered.append(largest)
    taken = {row["image_sha256"] for row in selected}
    warm = []
    for group in ([public_row(row, "pro") for row in pointer if row["suite"] == "pro"],
                  [public_row(row, "ssp") for row in ssp]):
        warm.append(next(row for row in unique_ranked(group) if row["image_sha256"] not in taken))
    for row in ordered + warm:
        if sha(Path(row["image"])) != row["image_sha256"]:
            raise ValueError(f"Image bytes drift: {row['sample_id']}")
    return {"timed": ordered, "warmup": warm, "ssp_size_populations": populations,
            "source_cohort_denominators": {"ssp": 1581, "pro": 500, "sheets": 500},
            "largest_stress_sample_id": largest["sample_id"], "labels_in_inputs": False}


def source_hashes() -> dict:
    return {
        "ssp_public_sha256": sha(SSP_SOURCE),
        "pointer_public_sha256": sha(PTR_SOURCE),
        "full_s1_sha256": sha(FULL),
        "frozen_registry_sha256": sha(ROOT / "runs/paper_scale_ablation_v1/models/frozen.json"),
        "full_cost_worker_sha256": sha(ROOT / "scripts/run_full_mainline_cost_v1.py"),
        "v17_runtime_sha256": sha(ROOT / "src/clickagent/advantage_runtimes.py"),
    }


def register() -> dict:
    if source_hashes()["full_s1_sha256"] != FULL_SHA:
        raise ValueError("Frozen full S1 checkpoint SHA mismatch")
    frozen = read(ROOT / "runs/paper_scale_ablation_v1/models/frozen.json")["models"]["full_aligned_s1"]
    if frozen["sha256"] != FULL_SHA or Path(frozen["path"]).resolve() != FULL.resolve():
        raise ValueError("Full S1 differs from V2 registration")
    input_path, protocol_path = OUT / "public_inputs.json", OUT / "protocol.json"
    if protocol_path.exists():
        p = read(protocol_path)
        if p["worker_sha256"] != sha(Path(__file__)) or p["public_inputs_sha256"] != sha(input_path):
            raise ValueError("Frozen pilot worker or manifest drift")
        if p["sources"] != source_hashes():
            raise ValueError("Frozen pilot source drift")
        return p
    inputs = choose_inputs()
    if input_path.exists() and read(input_path) != inputs:
        raise ValueError("Existing public input manifest differs")
    if not input_path.exists():
        write(input_path, inputs)
    p = {"registered_at_utc": now(), "purpose": "Unlabeled pixel-budget timing and resource pilot, not accuracy evaluation",
         "worker_sha256": sha(Path(__file__)), "public_inputs_sha256": sha(input_path),
         "sources": source_hashes(), "model": "V2 frozen full_aligned_s1", "budgets_per_view": list(BUDGETS),
         "candidate_policy": "Original V15 K16 forced stable Top1; unchanged prompt/crop/proposer/scorer weights",
         "selection": "24 SSP covering all 14 native sizes (one each plus frequency-allocated extras), 8 Pro, 8 Sheets; deterministic SHA rank and order; distinct SSP screenshots; largest SSP last",
         "warmups_per_arm": 2, "repetitions_per_case_per_arm": 1,
         "process_policy": "Low and high arms must run in separate sequential processes on the same idle two-GPU host",
         "timing": "V17 infer_once decode-through-CUDA-sync; additionally measure per-case wall including image read/hash and first result write; exclude model load and two warmups from per-case estimates",
         "failure_policy": "Record the exact failed case/OOM and stop that arm. No hidden cap reduction, skipped denominator, partial accuracy score or extrapolated full-cohort ETA after an incomplete arm",
         "projection": "Pro and Sheets: sample mean times times 500 each. SSP: exact-image-size population weighted mean times 1581. Report sample extrema and count, no precise CI from this small pilot. Full suite ETA excludes initial model load, warmups and downstream scoring; add their measured fixed overhead separately.",
         "labels_read": False, "retrospective": True}
    write(protocol_path, p)
    return p


def make_runtime(budget: int, folder: Path):
    import torch
    from clickagent.advantage_runtimes import OurRuntime, RuntimeCounter, configure_paths
    from clickagent.action_model import ActionQwenRuntime
    from clickagent.conditioned_features import ConditionedFeatureRuntime
    from clickagent.evidence_retention import RetentionModel
    import run_quality_readout_v15 as base
    import run_contribution_v16 as contribution

    configure_paths("ours_s1")
    obj = object.__new__(OurRuntime)
    legacy = base.Extractor()  # Authenticates the sealed original Qwen/proposer.
    config = replace(legacy.config, max_pixels=budget, offload_folder=str(folder / "offload"))
    obj.proposer, obj.base, obj.contribution = legacy.proposer.requires_grad_(False), base, contribution
    if budget == BUDGETS[0]:
        obj.runtime = ConditionedFeatureRuntime(config)
    else:
        class ExperimentalHighPixelRuntime(ConditionedFeatureRuntime):
            def __init__(self, cfg):
                # Reuse the proven V37/high_pixel_feasibility_v1 budget-only
                # bypass.  The archived public runtime remains untouched.
                ActionQwenRuntime.__init__(self, cfg)

        obj.runtime = ExperimentalHighPixelRuntime(config)
    saved = torch.load(FULL, map_location="cpu", weights_only=True)
    model = RetentionModel().to("cuda:0").eval().requires_grad_(False)
    model.load_state_dict(saved["state_dict"])
    obj.models = [(model, saved)]
    obj.counter = RuntimeCounter()
    obj.counter.attach(obj.runtime.model, 16)
    obj.metadata = dict(method="full_aligned_s1_high_pixel_pilot", max_pixels=budget,
                        checkpoint=str(FULL), checkpoint_sha256=FULL_SHA,
                        scorer_device="cuda:0", model_metadata=obj.runtime.runtime_metadata)
    obj.view_trace = []
    original = obj.runtime.infer_spatial_features

    def traced(image, instruction, *args, **kwargs):
        arrays, meta = original(image, instruction, *args, **kwargs)
        gh, gw = map(int, meta["grid_shape"])
        obj.view_trace.append({"image_size": meta["image_size"], "grid_shape": [gh, gw],
                               "actual_prefill_pixels": gh * gw * 32 * 32,
                               "target_tokens": len(meta["target_raw_indices"])})
        return arrays, meta

    obj.runtime.infer_spatial_features = traced
    return obj


def one_call(runtime, case: dict, torch, ordinal: int | None = None) -> dict:
    from run_advantage_inference_v17 import infer_once, payload_for

    wall_start = time.perf_counter()
    payload, _ = payload_for(case, "clean")  # Read/hash outside inference timer.
    payload_seconds = time.perf_counter() - wall_start
    runtime.view_trace = []
    output, measured = infer_once(runtime, payload, case["instruction"], torch)
    trace = list(runtime.view_trace)
    if not 1 <= len(trace) <= 4:
        raise ValueError("Unexpected full/local view count")
    if any(view["actual_prefill_pixels"] > runtime.metadata["max_pixels"] for view in trace):
        raise ValueError("Processor exceeded registered max_pixels")
    if sum(view["actual_prefill_pixels"] for view in trace) != measured["counter"]["processed_pixels"]:
        raise ValueError("View grids disagree with measured processed pixels")
    row = {"sample_id": case["sample_id"], "suite": case["suite"], "image_size": case["image_size"],
           "input_image_sha256": case["image_sha256"], "ordinal": ordinal,
           "output": output, "candidate_count": len(runtime.last_record["points"]),
           "k16_count": len(runtime.last_record["baseline"]), "view_trace": trace,
           "measured": measured, "image_read_hash_seconds": payload_seconds,
           "wall_seconds_excluding_result_write": time.perf_counter() - wall_start}
    return row


def run(budget: int) -> None:
    register()
    if budget not in BUDGETS:
        raise ValueError("Unregistered pixel budget")
    import torch
    from run_advantage_inference_v17 import sync

    torch.set_num_threads(4)
    names = [torch.cuda.get_device_name(d) for d in range(torch.cuda.device_count())]
    if len(names) != 2:
        raise RuntimeError(f"Pilot requires the same two GPUs per arm; found {names}")
    folder = OUT / f"p{budget}"
    status = folder / "status.json"
    if status.exists() or (folder / "cases").exists():
        raise RuntimeError(f"Preserve existing attempt: {folder}")
    inputs = read(OUT / "public_inputs.json")
    write(status, {"phase": "loading", "budget": budget, "at_utc": now(), "completed": 0, "total": 40})
    started = time.perf_counter()
    rows = []
    current = None
    try:
        runtime = make_runtime(budget, folder)
        sync(torch)
        load_seconds = time.perf_counter() - started
        write(folder / "runtime.json", {**runtime.metadata, "load_seconds": load_seconds,
                                         "gpu_names": names, "torch_version": torch.__version__,
                                         "cuda_version": torch.version.cuda,
                                         "gpu_before_warmup": cost.gpu_snapshot()})
        warm = []
        for case in inputs["warmup"]:
            current = case["sample_id"]
            warm.append(one_call(runtime, case, torch))
        write(folder / "warmup.json", {"not_timed_pilot_cases": True, "rows": warm})
        write(status, {"phase": "timing", "budget": budget, "at_utc": now(), "completed": 0, "total": 40})
        for ordinal, case in enumerate(inputs["timed"], 1):
            current = case["sample_id"]
            wall_start = time.perf_counter()
            row = one_call(runtime, case, torch, ordinal)
            output_path = folder / f"cases/{ordinal:03d}.json"
            write(output_path, row)
            row["per_case_wall_seconds"] = time.perf_counter() - wall_start
            write(output_path, row)  # First write is counted; second atomic correction is tiny.
            rows.append(row)
            state = {"phase": "timing", "budget": budget, "at_utc": now(),
                     "completed": len(rows), "total": 40, "latest_sample_id": current}
            write(status, state)
            print(json.dumps(state), flush=True)
        write(folder / "results.json", {"complete": True, "budget": budget, "n": len(rows),
                                         "rows": rows, "no_inference_cache": True})
        write(folder / "completion.json", {"complete": True, "at_utc": now(), "budget": budget,
                                            "protocol_sha256": sha(OUT / "protocol.json"),
                                            "public_inputs_sha256": sha(OUT / "public_inputs.json"),
                                            "runtime_sha256": sha(folder / "runtime.json"),
                                            "warmup_sha256": sha(folder / "warmup.json"),
                                            "results_sha256": sha(folder / "results.json"),
                                            "gpu_after_run": cost.gpu_snapshot()})
        write(status, {"phase": "complete", "budget": budget, "at_utc": now(),
                       "completed": len(rows), "total": 40})
    except Exception:
        write(folder / "failure.json", {"at_utc": now(), "budget": budget,
                                        "sample_id": current, "completed": len(rows),
                                        "error": traceback.format_exc(),
                                        "partial_case_files": [p.name for p in sorted((folder / "cases").glob("*.json"))]})
        write(status, {"phase": "failed", "budget": budget, "at_utc": now(),
                       "completed": len(rows), "total": 40, "sample_id": current})
        raise


def _archived_points(ids: set[str]) -> dict[str, dict]:
    found = {}
    for suite in ("full", "pro", "sheets"):
        folder = ROOT / f"runs/paper_scale_ablation_v2/eval/{suite}/predictions/cases"
        for path in folder.glob("*.json"):
            row = read(path)
            sid = row["sample_id"]
            if sid in ids:
                if sid in found:
                    raise ValueError("Duplicate archived sample ID")
                found[sid] = row
    if set(found) != ids:
        raise ValueError(f"Missing archived V2 prediction IDs: {sorted(ids - set(found))}")
    return found


def analyze() -> None:
    register()
    inputs = read(OUT / "public_inputs.json")
    by_budget = {}
    runtime = {}
    for budget in BUDGETS:
        folder = OUT / f"p{budget}"
        receipt = read(folder / "completion.json")
        if not receipt["complete"] or receipt["budget"] != budget or receipt["protocol_sha256"] != sha(OUT / "protocol.json"):
            raise ValueError(f"Incomplete arm {budget}")
        for key in ("runtime", "warmup", "results"):
            if receipt[f"{key}_sha256"] != sha(folder / f"{key}.json"):
                raise ValueError(f"SHA receipt mismatch: {budget}/{key}")
        saved = read(folder / "results.json")
        rows = saved["rows"]
        if len(rows) != 40 or [r["sample_id"] for r in rows] != [c["sample_id"] for c in inputs["timed"]]:
            raise ValueError(f"Cohort/order drift for {budget}")
        for i, row in enumerate(rows, 1):
            if read(folder / f"cases/{i:03d}.json") != row:
                raise ValueError(f"Per-case file differs: {budget}/{i}")
            if row["input_image_sha256"] != inputs["timed"][i - 1]["image_sha256"]:
                raise ValueError("Input image SHA drift")
        by_budget[budget] = rows
        runtime[budget] = read(folder / "runtime.json")
    if runtime[BUDGETS[0]]["gpu_names"] != runtime[BUDGETS[1]]["gpu_names"]:
        raise ValueError("Hardware differs between budget arms")
    archived = _archived_points({row["sample_id"] for row in by_budget[BUDGETS[0]]})
    parity = []
    for row in by_budget[BUDGETS[0]]:
        old = archived[row["sample_id"]]
        parity.append({"sample_id": row["sample_id"],
                       "point_match": row["output"]["point"] == old["methods"]["full_aligned_s1"]["point"],
                       "K16_match": row["k16_count"] == old["K16_count"],
                       "candidate_count_match": row["candidate_count"] == old["candidate_count"]})
    if not all(all(v for k, v in row.items() if k != "sample_id") for row in parity):
        raise ValueError("Low-budget V2 output parity failed")
    write(OUT / "analysis/low_budget_parity.json", {"n": 40, "all_match": True, "rows": parity})
    populations = inputs["ssp_size_populations"]
    summaries = {}
    for budget in BUDGETS:
        rows = by_budget[budget]
        suite_result = {}
        for suite, denominator in inputs["source_cohort_denominators"].items():
            group = [r for r in rows if r["suite"] == suite]
            t = np.asarray([r["measured"]["elapsed_seconds"] for r in group], float)
            wall = np.asarray([r["per_case_wall_seconds"] for r in group], float)
            pixels = np.asarray([r["measured"]["counter"]["processed_pixels"] for r in group], float)
            calls = np.asarray([r["measured"]["counter"]["backbone_forward_calls"] for r in group], float)
            if suite == "ssp":
                by_size = defaultdict(list)
                for r in group:
                    by_size[f"{r['image_size'][0]}x{r['image_size'][1]}"].append(r)
                if set(by_size) != set(populations):
                    raise ValueError("Pilot did not cover every SSP exact size")
                estimated_inference = sum(populations[size] * np.mean([r["measured"]["elapsed_seconds"] for r in sample])
                                          for size, sample in by_size.items())
                estimated_wall = sum(populations[size] * np.mean([r["per_case_wall_seconds"] for r in sample])
                                     for size, sample in by_size.items())
                size_details = {size: {"population": populations[size], "pilot_n": len(sample),
                                       "mean_inference_seconds": float(np.mean([r["measured"]["elapsed_seconds"] for r in sample])),
                                       "mean_wall_seconds": float(np.mean([r["per_case_wall_seconds"] for r in sample]))}
                                for size, sample in by_size.items()}
            else:
                estimated_inference = denominator * float(t.mean())
                estimated_wall = denominator * float(wall.mean())
                size_details = None
            suite_result[suite] = {"pilot_n": len(group), "population_n": denominator,
                                   "inference_mean_seconds_unweighted": float(t.mean()),
                                   "inference_median_seconds": float(np.median(t)),
                                   "inference_min_seconds": float(t.min()),
                                   "inference_max_seconds": float(t.max()),
                                   "wall_mean_seconds_unweighted": float(wall.mean()),
                                   "processed_pixels_mean_unweighted": float(pixels.mean()),
                                   "backbone_forward_calls_mean_unweighted": float(calls.mean()),
                                   "estimated_population_inference_hours": float(estimated_inference / 3600),
                                   "estimated_population_wall_hours": float(estimated_wall / 3600),
                                   "size_details": size_details}
        all_allocated = np.asarray([r["measured"]["peak_allocated_bytes"] for r in rows], np.int64)
        summaries[str(budget)] = {"suite": suite_result,
                                  "peak_allocated_bytes_by_gpu": all_allocated.max(0).tolist(),
                                  "load_seconds": runtime[budget]["load_seconds"],
                                  "warmup_seconds": sum(r["wall_seconds_excluding_result_write"] for r in read(OUT / f"p{budget}/warmup.json")["rows"])}
    paired = {}
    for suite in ("ssp", "pro", "sheets"):
        low = np.asarray([r["measured"]["elapsed_seconds"] for r in by_budget[BUDGETS[0]] if r["suite"] == suite])
        high = np.asarray([r["measured"]["elapsed_seconds"] for r in by_budget[BUDGETS[1]] if r["suite"] == suite])
        paired[suite] = {"pilot_n": len(low), "unweighted_mean_extra_seconds": float((high - low).mean()),
                         "unweighted_ratio_of_means": float(high.mean() / low.mean()),
                         "faster_high_count": int((high < low).sum()),
                         "slower_high_count": int((high > low).sum())}
    result = {"complete": True, "at_utc": now(), "protocol_sha256": sha(OUT / "protocol.json"),
              "low_budget_v2_parity": {"n": 40, "all_match": True},
              "summaries": summaries, "paired": paired,
              "scope": "Resource and time projection from 40 label-free cases per budget; size-weighted SSP; not an accuracy experiment or precise wall-clock guarantee"}
    write(OUT / "analysis/results.json", result)
    lines = ["Frozen full_aligned_s1 pixel-budget timing pilot (NO LABELS)",
             "40 fixed public cases: SSP24 across every 14 native sizes, Pointer Pro8, Sheets8.",
             "Two separate processes, two warmups each, one uncached timed pass/case; same order and hardware.",
             "Decode-to-CUDA-sync inference and practical per-case wall (image read/hash plus first result write); model load/warmup separate.", ""]
    for suite in ("ssp", "pro", "sheets"):
        for budget in BUDGETS:
            s = summaries[str(budget)]["suite"][suite]
            lines.append(f"{suite} p{budget}: n={s['pilot_n']}; inference mean {s['inference_mean_seconds_unweighted']:.2f}s, "
                         f"median {s['inference_median_seconds']:.2f}s, range {s['inference_min_seconds']:.2f}-{s['inference_max_seconds']:.2f}s; "
                         f"population projection inference {s['estimated_population_inference_hours']:.2f}h, "
                         f"per-case wall {s['estimated_population_wall_hours']:.2f}h.")
        q = paired[suite]
        lines.append(f"  high/low pilot mean ratio {q['unweighted_ratio_of_means']:.3f}; "
                     f"high slower on {q['slower_high_count']}/{q['pilot_n']} cases.")
    lines += ["", "Population projections assume this small size-stratified sample is representative within each exact size.",
              "They exclude one-time model loading and warmups and do not provide a precise ETA.",
              "No target labels were opened or accuracy evaluated; 262144 arm matches 40/40 prior V2 points/K16/candidate counts."]
    (OUT / "analysis/readout.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    write(OUT / "analysis/completion.json", {"complete": True, "at_utc": now(),
                                             "results_sha256": sha(OUT / "analysis/results.json"),
                                             "readout_sha256": sha(OUT / "analysis/readout.txt"),
                                             "low_budget_parity_sha256": sha(OUT / "analysis/low_budget_parity.json")})
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("register", "run", "analyze"), required=True)
    parser.add_argument("--budget", type=int, choices=BUDGETS)
    args = parser.parse_args()
    if args.stage == "register":
        if args.budget is not None:
            parser.error("register has no --budget")
        print(json.dumps(register(), ensure_ascii=False, indent=2), flush=True)
    elif args.stage == "run":
        if args.budget is None:
            parser.error("run requires --budget")
        run(args.budget)
    else:
        if args.budget is not None:
            parser.error("analyze has no --budget")
        analyze()
