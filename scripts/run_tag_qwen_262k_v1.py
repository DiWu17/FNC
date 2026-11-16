"""Registered TAG/Qwen-only inference on four fixed, public paper cohorts.

No native-coordinate Qwen or MiniCPM run is dispatched. All 3,581 predictions
are sealed before a separate CPU process may open any labels. Re-running --stage
all verifies and resumes immutable cases. A process lock prevents duplicate jobs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
RUN = ROOT / "runs/tag_qwen_262k_v1"
PROTOCOL = RUN / "protocol.json"
COUNTS = {"pro": 500, "sheets": 500, "mmb": 1000, "sspro": 1581}
SOURCES = {
    "pro": ROOT / "runs/paper_scale_ablation_v2/pointerbench/public.jsonl",
    "sheets": ROOT / "runs/paper_scale_ablation_v2/pointerbench/public.jsonl",
    "mmb": ROOT / "runs/mmbench_gui_new_source_v1/public.json",
    "sspro": ROOT / "runs/paper_scale_ablation_v2/eval/full/public.json",
}
METHOD = "TAG-Qwen3-VL-8B-adapted"
ANALYZER = ROOT / "scripts/analyze_tag_qwen_262k_v1.py"


def require(ok, message):
    if not ok:
        raise ValueError(message)


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def write(path, value, immutable=False):
    path = Path(path)
    if immutable and path.exists():
        require(read(path) == value, "Immutable record changed: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp." + str(os.getpid()))
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(temp, path)


def status(**fields):
    record = {"updated_at_utc": now(), "pid": os.getpid(), **fields}
    write(RUN / "status.json", record)
    with (RUN / "events.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps(record, ensure_ascii=False), flush=True)


def public(suite):
    path = SOURCES[suite]
    rows = [r for r in jsonl(path) if r["suite"] == suite] if path.suffix == ".jsonl" else read(path)
    require(len(rows) == COUNTS[suite], "Cohort size changed")
    require(len({r["sample_id"] for r in rows}) == len(rows), "Duplicate sample ID")
    for row in rows:
        require({"sample_id", "image", "instruction", "image_size", "image_sha256", "pixel_sha256"} <= set(row), "Public schema changed")
        require(not ({"bbox", "target", "target_geometry", "label", "labels", "answer"} & set(row)), "Label present in public input")
    return rows


def model_paths():
    folder = ROOT / "models/Qwen3-VL-8B-Instruct"
    names = ["config.json", "generation_config.json", "preprocessor_config.json", "tokenizer_config.json",
             "tokenizer.json", "chat_template.json", "chat_template.jinja", "model.safetensors.index.json"]
    files = [folder / name for name in names if (folder / name).is_file()]
    shards = [folder / name for name in sorted(set(read(folder / "model.safetensors.index.json")["weight_map"].values()))]
    require(len(shards) == 4, "Expected four frozen Qwen weight shards")
    return files + shards


def versions():
    return {name: importlib.metadata.version(name) for name in ("torch", "transformers", "tokenizers", "accelerate", "numpy", "pillow")}


def register():
    if PROTOCOL.exists():
        return verify_protocol()
    require(not (RUN / "cases").exists(), "Cases predate registration")
    rows = {suite: public(suite) for suite in COUNTS}
    sources = {Path(__file__), ANALYZER, ROOT / "src/clickagent/tag_qwen_runtime_v1.py",
               ROOT / "third_party/TAG/core_utils/tag.py", ROOT / "third_party/TAG/eval_mm/screenspot/prompt_tag.py",
               ROOT / "src/clickagent/public_evaluation.py", ROOT / "scripts/analyze_pointer_mvp_same_cohort_v1.py",
               ROOT / "scripts/run_mmbench_gui_l2_v1.py",
               ROOT / "runs/mmbench_gui_new_source_v1/official_l2_evaluator.py", *SOURCES.values()}
    for suite in ("pro", "sheets"):
        sources.add(ROOT / f"runs/paper_scale_ablation_v2/pointerbench/source/{suite}/eval.py")
    pilot = ROOT / "runs/tag_qwen_timing_pilot_v1"
    sources.update((pilot / "protocol.json", pilot / "summary.json", pilot / "warmup.json"))
    require(all(p.is_file() for p in sources), "Required source missing")
    status(phase="registering", note="Hashing frozen sources and Qwen weights; no labels opened")
    spec = {
        "complete": True, "run_id": RUN.name, "registered_at_utc": now(), "method": METHOD,
        "cohort_n": COUNTS, "cohort_order": list(COUNTS), "total_n": sum(COUNTS.values()),
        "model": "Qwen3-VL-8B-Instruct", "dtype": "float16", "device_map": "balanced",
        "max_memory_GiB": [12, 12], "min_pixels": 16384, "max_pixels_per_view": 262144, "views": 1,
        "attention_backend": "eager", "all_layers_heads": [36, 32], "topk_heads_per_token": 10,
        "attention_selection": "Per description token, rank all 1152 heads by total visual-token attention mass, stable ties",
        "token_readout": "Own-token attention (generation step j+1); mean over selected heads then description tokens",
        "threshold": 0.5, "spatial_mapping": "Qwen native post-merger raster, 32-pixel cell expansion, original TAG connected regions",
        "point_rule": "Center of highest-confidence region; largest-area tie break; map to original image",
        "fallback": "Original instruction-token prefill attention when generated description cannot be aligned",
        "max_new_tokens": 1024, "do_sample": False, "training": "none", "native_qwen_coordinate_baseline": "excluded by user",
        "source_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted(sources, key=str)},
        "model_sha256": {p.relative_to(ROOT).as_posix(): sha(p) for p in model_paths()}, "versions": versions(),
        "sample_order_sha256": {suite: identity([r["sample_id"] for r in rr]) for suite, rr in rows.items()},
        "label_sha256_from_prior_receipts": {
            "pointer": read(ROOT / "runs/paper_scale_ablation_v2/statistical_analysis_protocol.json")["pointer_labels_sha256_unopened"],
            "sspro": read(ROOT / "runs/paper_scale_ablation_v2/statistical_analysis_protocol.json")["full_labels_sha256_unopened"],
            "mmb": read(ROOT / "runs/mmbench_gui_new_source_v1/protocol.json")["source_annotation_sha256"],
        },
        "metric": "Pinned official inclusive point-in-box on Pointer/MMB; existing normalized point-in-target-geometry on SSP",
        "invalid_output": "Null/out-of-image point counts incorrect; no sample removal; a capped generation retains an available derived point and records truncation",
        "runtime_failure": "Stop and resume identical configuration; never silently drop an error case or lower pixel budget",
        "timing": "One excluded pilot warmup; per-case synchronized inference plus separate wall time including image hashing/cache overhead",
        "analysis": "All four bundles and per-case SHA must verify before a separate CPU process reads labels; descriptive accuracy/counts and timing",
        "pilot": "20 unlabelled timing queries and one warmup previously generated; rerun every formal query, no pilot cache reuse",
        "limitations": ["Qwen adaptation of TAG, not original MiniCPM publisher reproduction",
                        "Retrospective same-cohort comparison; previous project results and labels already analyzed",
                        "Pointer had at least seven known answer previews; full preview count unknown",
                        "SSP1581 has historical development exposure; descriptive comparison only",
                        "MMB1000 is the fixed project subset, not the complete official L2 leaderboard",
                        "Same per-view cap does not imply equal total computation or equal views across methods"],
        "labels_opened_by_inference": False,
    }
    write(PROTOCOL, spec, immutable=True)
    status(phase="registered", protocol_sha256=sha(PROTOCOL), total=3581, labels_opened=False)
    return spec


def verify_protocol(full_weights=False):
    spec = read(PROTOCOL)
    require(spec["method"] == METHOD and spec["cohort_n"] == COUNTS, "Protocol identity drift")
    require(spec["versions"] == versions(), "Package versions changed")
    for rel, digest in spec["source_sha256"].items():
        require(sha(ROOT / rel) == digest, "Frozen source changed: " + rel)
    for rel, digest in spec["model_sha256"].items():
        if full_weights or not rel.endswith(".safetensors"):
            require(sha(ROOT / rel) == digest, "Frozen model changed: " + rel)
    for suite in COUNTS:
        require(identity([r["sample_id"] for r in public(suite)]) == spec["sample_order_sha256"][suite], "Order changed")
    return spec


def case_path(suite, row):
    return RUN / "cases" / suite / (hashlib.sha256(row["sample_id"].encode()).hexdigest() + ".json")


def check_case(case, row, suite, protocol_sha):
    for key, expected in {"sample_id": row["sample_id"], "suite": suite, "method": METHOD,
                          "image_sha256": row["image_sha256"], "public_identity": identity(row),
                          "protocol_sha256": protocol_sha, "labels_opened": False}.items():
        require(case.get(key) == expected, "Case identity changed: " + key)
    payload = {k: v for k, v in case.items() if k != "payload_sha256"}
    require(case.get("payload_sha256") == identity(payload), "Case payload SHA changed")
    point = case["point"]
    require(point is None or (type(point) is list and len(point) == 2 and all(type(v) is int and 0 <= v < s for v, s in zip(point, row["image_size"]))), "Invalid native point")
    require(0 < case["processed_pixels"] <= 262144, "Pixel cap changed")
    require(math.isfinite(case["elapsed_seconds"]) and case["elapsed_seconds"] > 0, "Invalid timing")


def seal(suite):
    protocol_sha = sha(PROTOCOL)
    rows = public(suite)
    outputs, hashes = [], {}
    for row in rows:
        path = case_path(suite, row)
        case = read(path)
        check_case(case, row, suite, protocol_sha)
        outputs.append(case)
        hashes[row["sample_id"]] = sha(path)
    result = {"complete": True, "method": METHOD, "suite": suite, "n": len(rows), "rows": outputs,
              "case_hashes": hashes, "protocol_sha256": protocol_sha, "public_source_sha256": sha(SOURCES[suite]), "labels_opened": False}
    path = RUN / "predictions" / f"{suite}.json"
    write(path, result, immutable=True)
    return sha(path)


def gate():
    verify_protocol()
    hashes = {suite: seal(suite) for suite in COUNTS}
    receipt = {"complete": True, "method": METHOD, "n": sum(COUNTS.values()), "protocol_sha256": sha(PROTOCOL),
               "prediction_sha256": hashes, "labels_opened": False}
    write(RUN / "public_completion.json", receipt, immutable=True)
    return receipt


@contextmanager
def process_lock():
    import msvcrt
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / "worker.lock").open("a+b") as lock:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        try:
            yield
        finally:
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_UNLCK, 1)


def run():
    verify_protocol(full_weights=True)
    protocol_sha = sha(PROTOCOL)
    pools = {suite: public(suite) for suite in COUNTS}
    done = {}
    elapsed = []
    for suite, rows in pools.items():
        done[suite] = 0
        for row in rows:
            path = case_path(suite, row)
            if path.exists():
                result = read(path)
                check_case(result, row, suite, protocol_sha)
                done[suite] += 1
    if sum(done.values()) == 3581:
        gate()
        return
    status(phase="loading", completed_by_suite=done, total=3581, labels_opened=False)
    from clickagent.tag_qwen_runtime_v1 import TagQwenRuntime
    import torch
    import transformers
    start = time.perf_counter()
    runtime = TagQwenRuntime()
    runtime.sync()
    require(all(str(v) in ("0", "1", "cuda:0", "cuda:1") for v in runtime.model.hf_device_map.values()), "Unexpected CPU/disk model offload")
    write(RUN / "runtime.json", {"at": now(), "model_load_seconds": time.perf_counter() - start,
                               "device_map": {k: str(v) for k, v in runtime.model.hf_device_map.items()},
                               "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                               "torch": torch.__version__, "transformers": transformers.__version__})
    pilot = ROOT / "runs/tag_qwen_timing_pilot_v1"
    warm_row = read(pilot / "protocol.json")["warmup"]
    require(sha(warm_row["image"]) == warm_row["image_sha256"], "Warmup image changed")
    warm = runtime.infer(warm_row)
    previous = read(pilot / "warmup.json")
    for key in ("point", "generated_text", "selected_description_tokens", "instruction_fallback", "processed_pixels", "grid_hw"):
        require(warm[key] == previous[key], "Pilot parity failed: " + key)
    write(RUN / "warmup_parity.json", {"passed": True, "at": now(), "result": warm, "excluded_from_formal_predictions": True})
    status(phase="warmup_passed", completed_by_suite=done, labels_opened=False)
    calls = [0]
    handle = runtime.model.register_forward_pre_hook(lambda *_: calls.__setitem__(0, calls[0] + 1))
    try:
        for suite, rows in pools.items():
            for row in rows:
                path = case_path(suite, row)
                if path.exists():
                    continue
                current_mean = statistics.mean(elapsed[-100:]) if elapsed else 8.0
                current_remaining = (3581 - sum(done.values())) * current_mean
                status(phase="inference", current_suite=suite, current_sample_id=row["sample_id"],
                       completed=sum(done.values()), completed_by_suite=done, total=3581,
                       eta_remaining_hours=current_remaining / 3600,
                       eta_completion_utc=(datetime.now(timezone.utc) + timedelta(seconds=current_remaining)).isoformat(),
                       labels_opened=False)
                t0 = time.perf_counter()
                require(sha(row["image"]) == row["image_sha256"], "Input image SHA changed")
                before_calls = calls[0]
                result = runtime.infer({**row, "suite": suite})
                point = result["point"]
                if point is not None and not all(0 <= value < size for value, size in zip(point, row["image_size"])):
                    result["point"] = None
                    result["output_failure"] = "Point outside original image"
                result.update(method=METHOD, image_sha256=row["image_sha256"], pixel_sha256=row["pixel_sha256"],
                              public_identity=identity(row), protocol_sha256=protocol_sha, completed_at_utc=now(),
                              backbone_forward_calls=calls[0] - before_calls, view_count=1,
                              wall_seconds=time.perf_counter() - t0)
                result["payload_sha256"] = identity(result)
                check_case(result, row, suite, protocol_sha)
                write(path, result, immutable=True)
                elapsed.append(time.perf_counter() - t0)
                done[suite] += 1
                mean = statistics.mean(elapsed[-100:])
                remaining = (3581 - sum(done.values())) * mean
                status(phase="inference", current_suite=suite, completed=sum(done.values()), completed_by_suite=done,
                       total=3581, recent_mean_wall_seconds=mean, eta_remaining_hours=remaining / 3600,
                       eta_completion_utc=(datetime.now(timezone.utc) + timedelta(seconds=remaining)).isoformat(), labels_opened=False)
            seal(suite)
            status(phase="suite_sealed", suite=suite, completed_by_suite=done, total=3581, labels_opened=False)
        gate()
        status(phase="complete_public", completed=3581, completed_by_suite=done, total=3581, labels_opened=False)
    finally:
        handle.remove()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("register", "run", "verify-gate", "all"))
    args = parser.parse_args()
    with process_lock():
        try:
            if args.stage in ("register", "all"):
                register()
            if args.stage in ("run", "all"):
                run()
            if args.stage in ("verify-gate", "all"):
                gate()
            if args.stage == "all":
                # New CPU process; it verifies every public seal before reading labels.
                status(phase="analyzing_sealed_predictions", completed=3581, labels_opened=False)
                subprocess.run([sys.executable, "-B", "-X", "utf8", str(ANALYZER)], cwd=ROOT, check=True)
                status(phase="complete_analyzed", completed=3581, total=3581, labels_opened_by_analysis=True)
        except Exception as exc:
            status(phase="failed", error=repr(exc), traceback=traceback.format_exc())
            raise


if __name__ == "__main__":
    main()
