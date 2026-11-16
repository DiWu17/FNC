"""Sealed public-only MVP resolution/crop controls on PointerBench Pro500.

Preparation and registration are CPU-only. Inference refuses to start until the
ongoing ClickAgent 1M run has sealed all 2,581 predictions. Ground truth is
never opened here. Run `fixed` and `adaptive` in separate Python processes.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import run_pointer_mvp_same_cohort_v1 as old

OUT = ROOT / "runs/review_mvp_resolution_v1"
PROTOCOL = OUT / "protocol.json"
BUGFIX = OUT / "adaptive_bugfix_addendum_v1.json"
PUBLIC = ROOT / "runs/paper_scale_ablation_v2/pointerbench/public.jsonl"
VAL = ROOT / "data/expansion_v15_qc1_candidate/manifests/validation_reserve.jsonl"
HPR = ROOT / "runs/high_pixel_full_inference_v1"
MVP = ROOT / "third_party/MVP"
ADAPTER = ROOT / "src/clickagent/advantage_runtimes.py"
METHODS = ("fixed", "adaptive")
PIXELS = 1_048_576
N = 500
WARMUP_N = 5


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: Path) -> str:
    return old.sha(path)


def read(path: Path):
    return old.read(path)


def write(path: Path, value, *, immutable: bool = False) -> None:
    old.atomic_json(path, value, immutable=immutable)


def case_key(sample_id: str) -> str:
    return old.case_key(sample_id)


def pro_rows() -> list[dict]:
    rows = old.validate_parent()  # existing V2 public-only source and SHA audit
    pro = [row for row in rows if row["suite"] == "pro"]
    require(len(pro) == N and all(row["image_size"] == [1024, 768] for row in pro),
            "Pro500 cohort or native size drift")
    return pro


def reference_crop_defaults() -> tuple[int, int]:
    tree = ast.parse((MVP / "mvp_sspro_qwen3vl.py").read_text(encoding="utf-8"))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == "get_top_attention_regions")
    defaults = {arg.arg: ast.literal_eval(value)
                for arg, value in zip(node.args.args[-len(node.args.defaults):],
                                      node.args.defaults)}
    return int(defaults["subimage_width"]), int(defaults["subimage_height"])


def val_geometry() -> dict:
    unique: dict[str, dict] = {}
    for line in VAL.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        key = row["pixel_sha256"]
        if key in unique:
            require(unique[key]["image_size"] == row["image_size"],
                    "Validation image dimension inconsistent")
        else:
            unique[key] = {field: row[field] for field in
                           ("pixel_sha256", "image_size", "image", "image_sha256", "instruction")}
    require(len(unique) >= 100, "Independent validation geometry source too small")
    widths = [row["image_size"][0] for row in unique.values()]
    heights = [row["image_size"][1] for row in unique.values()]
    median_w, median_h = int(statistics.median(widths)), int(statistics.median(heights))
    require((median_w, median_h) == (1920, 1080), "Validation geometry changed")
    warmup = [row for row in unique.values() if row["image_size"] == [1920, 1080]][:WARMUP_N]
    require(len(warmup) == WARMUP_N, "Missing preselected validation warmup images")
    return {"unique_validation_images": len(unique), "median_image_size": [median_w, median_h],
            "warmup": warmup}


def adaptive_size(reference: tuple[int, int], median: tuple[int, int],
                  native: tuple[int, int]) -> tuple[int, int]:
    return (min(native[0], round(native[0] * reference[0] / median[0])),
            min(native[1], round(native[1] * reference[1] / median[1])))


def source_hashes() -> dict[str, str]:
    paths = [Path(__file__), ROOT / "scripts/analyze_review_mvp_resolution_v1.py",
             ROOT / "scripts/run_pointer_mvp_same_cohort_v1.py", ADAPTER,
             MVP / "mvp_sspro_qwen3vl.py", *sorted((MVP / "qwen3_vl").glob("*.py")),
             PUBLIC, VAL, HPR / "protocol.json", HPR / "public_manifest.json",
             ROOT / "runs/paper_scale_ablation_v2/pointerbench/source/pro/eval.py",
             ROOT / "runs/paper_scale_ablation_v2/pointerbench/evaluation_protocol_v2.json",
             old.PROTOCOL, old.COMPLETION]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path)
            for path in sorted(set(paths), key=str)}


def register() -> None:
    require(not PROTOCOL.exists(), "MVP 1M protocol already frozen")
    require(not (OUT / "predictions").exists(), "Predictions exist before protocol")
    pro = pro_rows()
    source = read(old.PROTOCOL)
    require(source["mvp"]["max_pixels_per_view"] == 262144 and
            source["mvp"]["max_subimages"] == 2 and
            source["public_source_sha256"] == sha(PUBLIC),
            "Prior local MVP provenance drift")
    high = read(HPR / "protocol.json")
    require(high["pixel_cap_per_view"] == PIXELS and high["denominators"]["pro"] == N and
            high["scientific_policy"].startswith("V2 frozen full aligned"),
            "High-pixel ClickAgent comparator protocol drift")
    ref = reference_crop_defaults()
    geometry = val_geometry()
    median = tuple(geometry["median_image_size"])
    adapted = adaptive_size(ref, median, (1024, 768))
    require(ref == (1288, 728) and adapted == (687, 518),
            "Pre-fixed MVP crop geometries changed")
    require(not ({row["pixel_sha256"] for row in pro} &
                 {row["pixel_sha256"] for row in geometry["warmup"]}),
            "Validation warmup overlaps test images")
    record = {
        "complete": True, "at_utc": now(), "run_id": "review_mvp_resolution_v1",
        "design": "Retrospective post-V2 same-Pro500 resolution and crop geometry controls",
        "public_suite": "PointerBench Pro500", "n": N,
        "public_source_sha256": sha(PUBLIC), "prior_mvp_completion_sha256": sha(old.COMPLETION),
        "highpixel_protocol_sha256": sha(HPR / "protocol.json"),
        "labels_expected_sha256": read(ROOT / "runs/paper_scale_ablation_v2/pointerbench/evaluation_protocol_v2.json")["labels_source_sha256_unopened"],
        "source_hashes": source_hashes(),
        "model_shards": source["model_shards"],
        "mvp_model": source["model"],
        "methods": list(METHODS), "pixel_cap_per_view": PIXELS,
        "fixed": {"crop_policy": "Unmodified official MVP 1288x728 crop constants",
                  "effective_attention_branch_crop_size_on_1024x768": [1024, 728]},
        "adaptive": {"crop_policy": "Geometric transfer of official crop dimensions relative to independent validation median screenshot; no label or accuracy tuning",
                     "validation_manifest_sha256": sha(VAL),
                     "unique_validation_images": geometry["unique_validation_images"],
                     "validation_median_image_size": list(median),
                     "reference_crop_size": list(ref),
                     "native_crop_size_on_1024x768": list(adapted),
                     "source_area_fraction": adapted[0] * adapted[1] / (1024 * 768)},
        "warmup": {"n_each_fresh_process": WARMUP_N,
                   "source": "Independent validation images; no PointerBench test sample",
                   "image_identity": [{"pixel_sha256": row["pixel_sha256"],
                                       "image_sha256": row["image_sha256"],
                                       "image_size": row["image_size"]}
                                      for row in geometry["warmup"]]},
        "runtime": {"model_frozen": True, "dtype": "float16", "attention": "sdpa",
                    "device_map": "balanced", "max_memory_per_gpu": "12GiB",
                    "max_subimages": 2, "consistency_threshold": 14,
                    "deterministic_decoding": True, "batch_size": 1,
                    "timer": "GPU-synchronized around one public inference, excluding PNG decode, image load, and file I/O; ClickAgent 1M pilot.one_call includes PNG decode, so raw latency is descriptive rather than a matched speed comparison"},
        "start_gate": "Both MVP modes refuse GPU model load until all 2581 high-pixel ClickAgent public predictions are SHA sealed and status is complete; the operator must also confirm the ClickAgent 1M GPU process has exited before invoking either mode",
        "prediction_gate": "All 500 fixed and 500 adaptive public-only predictions, per-case and bundle SHA, before opening any PointerBench label in this analysis",
        "primary_contrasts": ["1M full aligned 3-seed mean versus 1M MVP fixed",
                              "1M full aligned 3-seed mean versus 1M MVP adaptive"],
        "secondary_contrasts": ["MVP adaptive versus MVP fixed at 1M",
                                "MVP fixed 1M versus previously sealed local MVP fixed 262k"],
        "statistics": "Official native inclusive point-in-bbox; RGB screenshot-cluster paired bootstrap 10000 draws, seed 20261114; conditional on fixed fits",
        "strict_blind": False,
        "limitations": ["V2 labels and results previously analyzed; at least seven PointerBench answers were visible before earlier sealing",
                        "Adaptive crop is a separately declared geometric variant, not the publisher's official MVP setting",
                        "Equal per-view cap does not equalize total views, tokens, or runtime",
                        "MVP timer excludes PNG decode while ClickAgent 1M timer includes it; no matched end-to-end latency claim",
                        "Pro500 is a fixed public subset, not the full PointerBench leaderboard"],
        "labels_may_open_in_runner": False,
    }
    write(PROTOCOL, record, immutable=True)
    status("registered", protocol_sha256=sha(PROTOCOL), n=N, labels_opened=False)


def status(phase: str, **fields) -> None:
    value = {"phase": phase, "at_utc": now(), "pid": os.getpid(), **fields}
    write(OUT / "status.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def gate(*, deep_model: bool = False) -> tuple[dict, list[dict]]:
    require(PROTOCOL.exists(), "Register before inference")
    p = read(PROTOCOL)
    current_sources = source_hashes()
    if p["source_hashes"] != current_sources:
        # A versioned implementation-only repair is allowed without editing the
        # frozen scientific protocol. Every other registered source stays exact.
        key = "scripts/run_review_mvp_resolution_v1.py"
        require(BUGFIX.exists(), "MVP 1M source SHA drift without bugfix receipt")
        fix = read(BUGFIX)
        require(fix["kind"] == "adaptive_crop_closure_shadowing_only" and
                fix["protocol_sha256"] == sha(PROTOCOL) and
                fix["old_runner_sha256"] == p["source_hashes"][key] and
                fix["new_runner_sha256"] == current_sources[key] and
                fix["fixed_bundle_sha256"] == sha(OUT / "predictions/fixed.json") and
                fix["first_adaptive_case_sha256"] == sha(case_path("adaptive", "pbp_0000")) and
                fix["failure_receipt_sha256"] == sha(OUT / "failures" / fix["failure_receipt_name"]) and
                all(current_sources[name] == digest for name, digest in p["source_hashes"].items()
                    if name != key),
                "MVP 1M source drift exceeds registered adaptive bugfix")
    require(p["complete"] is True and p["methods"] == list(METHODS) and
            p["pixel_cap_per_view"] == PIXELS and p["n"] == N and
            p["public_source_sha256"] == sha(PUBLIC) and
            p["highpixel_protocol_sha256"] == sha(HPR / "protocol.json") and
            p["labels_may_open_in_runner"] is False,
            "MVP 1M protocol or source SHA drift")
    for name, record in p["model_shards"].items():
        path = old.MODEL / name
        require(path.stat().st_size == record["size_bytes"], "Model shard size drift")
        if deep_model:
            require(sha(path) == record["sha256"], "Model shard SHA drift: " + name)
    rows = pro_rows()
    return p, rows


def highpixel_sealed() -> None:
    """CPU-only gate; do not touch the ongoing worker's status or outputs."""
    require((HPR / "public_completion.json").exists(),
            "ClickAgent 1M inference is still running; MVP GPU inference forbidden")
    receipt = read(HPR / "public_completion.json")
    p = read(HPR / "protocol.json")
    require(receipt["complete"] is True and receipt["n"] == 2581 and
            receipt["labels_opened"] is False and
            receipt["protocol_sha256"] == sha(HPR / "protocol.json") and
            p["pixel_cap_per_view"] == PIXELS,
            "ClickAgent 1M seal incomplete or changed")
    require(read(HPR / "status.json").get("phase") == "complete",
            "ClickAgent 1M completion status has not been reached")
    for suite in ("ssp", "pro", "sheets"):
        path = HPR / suite / "completion.json"
        require(path.exists() and receipt["suite_receipt_sha256"][suite] == sha(path),
                "ClickAgent 1M suite SHA incomplete: " + suite)


def public_identity(row: dict) -> str:
    return old.public_identity(row)


def valid_point(point, size: list[int]) -> bool:
    return old.valid_point(point, size)


def check_case(record: dict, public: dict, method: str, protocol_sha: str) -> None:
    require(record["complete"] is True and record["labels_opened"] is False and
            record["sample_id"] == public["sample_id"] and record["suite"] == "pro" and
            record["method"] == method and record["public_identity"] == public_identity(public) and
            record["image_sha256"] == public["image_sha256"] and
            record["pixel_sha256"] == public["pixel_sha256"] and
            record["protocol_sha256"] == protocol_sha and
            record["pixel_cap_per_view"] == PIXELS and
            valid_point(record["point"], public["image_size"]),
            "MVP 1M per-case public identity drift: " + public["sample_id"])
    require(type(record["elapsed_seconds"]) in (int, float) and
            math.isfinite(record["elapsed_seconds"]) and record["elapsed_seconds"] >= 0 and
            type(record["backbone_forward_calls"]) is int and
            type(record["processed_pixels"]) is int and
            record["backbone_forward_calls"] >= 0 and record["processed_pixels"] >= 0 and
            type(record["inference_count"]) is int and 0 <= record["inference_count"] <= 3 and
            len(record["crop_boxes"]) <= 2 and
            all(len(box) == 4 and 0 <= box[0] < box[2] <= 1024 and
                0 <= box[1] < box[3] <= 768 for box in record["crop_boxes"]),
            "MVP 1M runtime or crop counter drift: " + public["sample_id"])


def case_path(method: str, sample_id: str) -> Path:
    return OUT / "predictions" / method / "cases" / (case_key(sample_id) + ".json")


def _run(method: str) -> None:
    p, rows = gate(deep_model=False)
    require(method in METHODS, "Unregistered MVP method")
    highpixel_sealed()  # before GPU import/model load
    bundle = OUT / "predictions" / (method + ".json")
    if bundle.exists():
        verify_method(method)
        status("already_complete", method=method, n=N)
        return
    if all(case_path(method, row["sample_id"]).exists() for row in rows):
        seal_method(method)
        return
    gate(deep_model=True)
    from PIL import Image
    import torch
    from clickagent.advantage_runtimes import configure_paths, create_runtime
    torch.set_num_threads(4)
    require(torch.cuda.device_count() == 2, "Expected two-GPU host")
    configure_paths("mvp")
    status("loading_model", method=method, labels_opened=False)
    load_started = time.perf_counter()
    runtime = create_runtime("mvp", OUT / ("runtime/" + method), PIXELS)
    load_elapsed = time.perf_counter() - load_started
    require(runtime.metadata["max_pixels"] == PIXELS and
            runtime.metadata["max_subimages"] == 2, "MVP runtime configuration drift")
    if method == "adaptive":
        original = runtime.official.get_top_attention_regions
        reference = tuple(p["adaptive"]["reference_crop_size"])
        median = tuple(p["adaptive"]["validation_median_image_size"])

        def scaled_regions(image, instruction, processor, model, device, **kwargs):
            require("subimage_width" not in kwargs and "subimage_height" not in kwargs,
                    "MVP caller changed crop policy")
            width, height = adaptive_size(reference, median, image.size)
            return original(image, instruction, processor, model, device,
                            subimage_width=width, subimage_height=height, **kwargs)

        runtime.official.get_top_attention_regions = scaled_regions
        runtime.metadata["crop_policy"] = "registered geometric transfer 687x518"
    runtime_info = {"metadata": runtime.metadata, "load_seconds": load_elapsed,
                    "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
                    "method": method, "protocol_sha256": sha(PROTOCOL)}
    metadata = OUT / "runtime" / method / "metadata.json"
    if metadata.exists():
        old_meta = read(metadata)
        require(old_meta["metadata"] == runtime_info["metadata"] and
                old_meta["protocol_sha256"] == runtime_info["protocol_sha256"],
                "MVP runtime metadata changed on resume")
    else:
        write(metadata, runtime_info, immutable=True)
    from PIL import Image
    warmup = val_geometry()["warmup"]
    for row in warmup:
        require(sha(Path(row["image"])) == row["image_sha256"], "Validation warmup image changed")
        with Image.open(row["image"]) as opened:
            image = opened.convert("RGB")
        try:
            with torch.inference_mode():
                runtime.infer(image, row["instruction"])
        finally:
            image.close()
    status("warmed", method=method, warmup=WARMUP_N, labels_opened=False)
    protocol_sha = sha(PROTOCOL)
    try:
        for index, public in enumerate(rows, 1):
            path = case_path(method, public["sample_id"])
            if path.exists():
                check_case(read(path), public, method, protocol_sha)
                continue
            image_path = Path(public["image"])
            require(sha(image_path) == public["image_sha256"],
                    "Public image SHA drift: " + public["sample_id"])
            with Image.open(image_path) as opened:
                require(list(opened.size) == public["image_size"], "Public image size drift")
                image = opened.convert("RGB")
            try:
                for device in range(torch.cuda.device_count()):
                    torch.cuda.synchronize(device)
                    torch.cuda.reset_peak_memory_stats(device)
                before = runtime.counter.snapshot()
                started = time.perf_counter()
                with torch.inference_mode():
                    output = runtime.infer(image, public["instruction"])
                for device in range(torch.cuda.device_count()):
                    torch.cuda.synchronize(device)
                elapsed = time.perf_counter() - started
                after = runtime.counter.snapshot()
                require(output.get("ground_truth_in_model_input") is False,
                        "MVP output label assurance failed")
                point = output.get("point")
                if point is not None:
                    point = [int(v) for v in point]
                    if not valid_point(point, public["image_size"]):
                        point = None
                reference_result = output.get("reference_result")
                crops = [list(map(int, x["region"])) for x in
                         reference_result.get("subimage_predictions", [])] if reference_result else []
                record = {"complete": True, "sample_id": public["sample_id"],
                          "suite": "pro", "method": method, "point": point,
                          "public_identity": public_identity(public),
                          "image_sha256": public["image_sha256"],
                          "pixel_sha256": public["pixel_sha256"],
                          "protocol_sha256": protocol_sha,
                          "pixel_cap_per_view": PIXELS,
                          "elapsed_seconds": elapsed,
                          "backbone_forward_calls": after["backbone_forward_calls"] -
                          before["backbone_forward_calls"],
                          "processed_pixels": after["processed_pixels"] -
                          before["processed_pixels"],
                          "peak_allocated_bytes": [torch.cuda.max_memory_allocated(device)
                                                   for device in range(torch.cuda.device_count())],
                          "inference_count": int(reference_result["inference_count"]) if reference_result else 0,
                          "crop_boxes": crops,
                          "output_failure": output.get("output_failure"),
                          "labels_opened": False}
                check_case(record, public, method, protocol_sha)
                write(path, record, immutable=True)
            finally:
                image.close()
            if index % 5 == 0 or index == N:
                status("inference", method=method, completed=index, total=N,
                       last_sample_id=public["sample_id"], labels_opened=False)
        seal_method(method)
    except Exception:
        raise


def run(method: str) -> None:
    """Preserve a failure receipt for gate, model load, warmup and inference."""
    try:
        _run(method)
    except Exception:
        prior = read(OUT / "status.json") if (OUT / "status.json").exists() else {}
        failure = {"phase": "failed", "at_utc": now(), "method": method,
                   "last_phase": prior.get("phase"),
                   "traceback": traceback.format_exc(), "labels_opened": False}
        write(OUT / "failures" / (now().replace(":", "-") + ".json"), failure,
              immutable=True)
        status("failed", method=method, labels_opened=False)
        raise


def seal_method(method: str) -> None:
    p, rows = gate()
    protocol_sha = sha(PROTOCOL)
    case_hashes, collected = {}, []
    for public in rows:
        path = case_path(method, public["sample_id"])
        require(path.exists(), "Missing MVP 1M case prediction")
        record = read(path)
        check_case(record, public, method, protocol_sha)
        case_hashes[public["sample_id"]] = sha(path)
        collected.append(record)
    value = {"complete": True, "method": method, "n": N,
             "rows": collected, "case_sha256": case_hashes,
             "protocol_sha256": protocol_sha, "public_source_sha256": p["public_source_sha256"],
             "labels_opened": False}
    bundle = OUT / "predictions" / (method + ".json")
    write(bundle, value, immutable=True)
    status("method_complete", method=method, n=N,
           prediction_sha256=sha(bundle), labels_opened=False)


def verify_method(method: str) -> str:
    p, rows = gate()
    path = OUT / "predictions" / (method + ".json")
    require(path.exists(), "MVP 1M method bundle missing")
    bundle = read(path)
    require(bundle["complete"] is True and bundle["labels_opened"] is False and
            bundle["method"] == method and bundle["n"] == N and
            bundle["protocol_sha256"] == sha(PROTOCOL) and
            bundle["public_source_sha256"] == p["public_source_sha256"] and
            len(bundle["rows"]) == N and len(bundle["case_sha256"]) == N,
            "MVP 1M method bundle drift")
    for public, record in zip(rows, bundle["rows"]):
        check_case(record, public, method, sha(PROTOCOL))
        per_case = case_path(method, public["sample_id"])
        require(sha(per_case) == bundle["case_sha256"][public["sample_id"]] and
                read(per_case) == record, "MVP 1M per-case SHA drift")
    return sha(path)


def verify_gate() -> dict:
    highpixel_sealed()
    hashes = {method: verify_method(method) for method in METHODS}
    value = {"complete": True, "n": N, "methods": list(METHODS),
             "prediction_sha256": hashes, "protocol_sha256": sha(PROTOCOL),
             "highpixel_completion_sha256": sha(HPR / "public_completion.json"),
             "labels_opened": False}
    completion = OUT / "public_completion.json"
    write(completion, value, immutable=True)
    status("gate_verified", n=N, methods=list(METHODS), labels_opened=False)
    return value


def self_test() -> None:
    assert reference_crop_defaults() == (1288, 728)
    geometry = val_geometry()
    assert adaptive_size((1288, 728), tuple(geometry["median_image_size"]),
                         (1024, 768)) == (687, 518)
    assert len(pro_rows()) == N
    print(json.dumps({"self_test": "passed", "validation_unique_images":
                      geometry["unique_validation_images"],
                      "adaptive_crop": [687, 518], "test_labels_opened": False}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=("self-test", "register", "fixed", "adaptive", "verify-gate"))
    args = parser.parse_args()
    if args.stage == "self-test":
        self_test()
    elif args.stage == "register":
        register()
    elif args.stage in METHODS:
        run(args.stage)
    else:
        print(json.dumps(verify_gate(), ensure_ascii=False))


if __name__ == "__main__":
    main()
