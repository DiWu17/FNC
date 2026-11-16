"""Fresh-inference, real-browser action benchmark; caches only grounder training.

Run with ``python -m clickagent.action_benchmark --help``. Test episode timings
always come from a new model forward and an actual browser event/verification.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import logging
from pathlib import Path
import random
import time
import traceback

import joblib
import numpy as np
from PIL import Image

from .dataset import load_manifest

LOGGER = logging.getLogger(__name__)
METHODS = ("prefill_hidden", "post_hidden", "text_decision_shared", "text_coordinates")
BENCHMARK_VERSION = 2


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    # A Windows reader (or antivirus scan) can briefly deny replacement of an
    # existing file. Keep the complete temporary JSON and retry the commit;
    # model outputs and per-case checkpoint selection are unaffected.
    for attempt in range(8):
        try:
            temporary.replace(path)
            return
        except PermissionError:
            if attempt == 7:
                raise
            time.sleep(min(.05 * 2 ** attempt, .5))


def stable_hash(value):
    normalized = json.loads(json.dumps(value, default=str))
    return hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()


def raw_path(output, sample_id):
    return output / "grounding_raw" / (hashlib.sha256(sample_id.encode()).hexdigest()[:24] + ".npz")


def compact_result(result):
    excluded = {"attention", "hidden", "raw_image_attention", "prefill_raw_map", "head_order", "system_prompt"}
    return {key: value for key, value in result.items() if key not in excluded}


def config_from_args(args):
    from .model_features import ExtractionConfig
    import torch
    return ExtractionConfig(model_path=str(args.model.resolve()), layers=(15, 28, 36),
                            device_map="balanced", dtype="float16", max_pixels=262144,
                            max_new_tokens=args.decision_token_cap, seed=43, task="visual_state",
                            conditions=("prefill", "post_decision"),
                            max_memory={**{i: args.gpu_memory for i in range(torch.cuda.device_count())}, "cpu": "24GiB"},
                            offload_folder=str((args.output / "offload").resolve()))


def feature_identity(config):
    from .model_features import _local_model_identity, _library_versions, system_prompt_for_task
    source = Path(__file__).parent
    return {"config": asdict(config), "model": _local_model_identity(config.model_path),
            "libraries": _library_versions(), "system_prompt": system_prompt_for_task("visual_state"),
            "source_sha256": {name: sha256(source / name) for name in
                              ("action_model.py", "model_features.py", "attention_features.py")}}


def collect_grounding(runtime, records, manifest_dir, output, identity, limit=None):
    selected = [row for row in records if row["split"] in {"train", "val"} and row["label"] == "click"]
    if limit is not None:
        # Smoke-only: retain at least one record from both partitions.
        selected = [row for split in ("train", "val") for row in
                    [item for item in selected if item["split"] == split][:limit]]
    fingerprint = stable_hash(identity)
    output.mkdir(parents=True, exist_ok=True)
    prepared = []
    for index, sample in enumerate(selected):
        image_path = manifest_dir / sample["image"]
        image_hash = sha256(image_path)
        target = raw_path(output, sample["sample_id"])
        if target.exists():
            with np.load(target, allow_pickle=False) as archive:
                info = json.loads(str(archive["metadata"]))
            if info["feature_fingerprint"] != fingerprint or info["image_sha256"] != image_hash:
                raise ValueError("Grounding cache belongs to a different model/input; select a new output")
        else:
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
            result = runtime.infer_readouts(image, stage="prefill")
            info = {"sample_id": sample["sample_id"], "feature_fingerprint": fingerprint,
                    "image_sha256": image_hash, "head_order": result["head_order"],
                    "generated_tokens": result["generated_tokens"], "elapsed_seconds": result["elapsed_seconds"]}
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".npz.tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, raw_attention=result["prefill_raw_map"],
                                    grid_shape=np.asarray(result["grid_shape"]),
                                    image_size=np.asarray(result["image_size"]),
                                    hidden=result["hidden"], attention=result["attention"],
                                    metadata=np.asarray(json.dumps(info)))
            temporary.replace(target)
        with np.load(target, allow_pickle=False) as archive:
            prepared.append({key: archive[key].copy() for key in ("raw_attention", "grid_shape", "image_size")})
        prepared[-1].update({key: sample[key] for key in ("sample_id", "group_id", "label", "split", "target_bbox")})
        prepared[-1]["head_order"] = info["head_order"]
        LOGGER.info("Grounding features %d/%d: %s", index + 1, len(selected), sample["sample_id"])
    write_json(output / "grounding_collection.json", {"samples": len(prepared), "feature_fingerprint": fingerprint,
                                                      "feature_identity": identity, "labels": ["click"],
                                                      "split_counts": dict(Counter(row["split"] for row in prepared))})
    return prepared


def fit_or_load_grounder(records, output):
    from .action_grounding import AttentionPointGrounder, fit_grounder
    identity = stable_hash([{**{key: row[key] for key in ("sample_id", "split", "target_bbox")},
                             "raw_sha256": hashlib.sha256(row["raw_attention"].tobytes()).hexdigest()}
                            for row in records])
    root = output / "grounder"
    source_hash = sha256(Path(__file__).with_name("action_grounding.py"))
    provenance = {"training_identity": identity, "implementation_sha256": source_hash}
    if (root / "grounder.joblib").exists():
        if json.loads((root / "provenance.json").read_text()) != provenance:
            raise ValueError("Existing localizer used different training data/code")
        return AttentionPointGrounder.load(root / "grounder.joblib")
    grounder = fit_grounder(records, root, seed=43)
    write_json(root / "provenance.json", provenance)
    return grounder


def load_gates(gate_root):
    metrics = json.loads((gate_root / "evaluation/metrics.json").read_text(encoding="utf-8"))
    gates, audit = {}, {}
    for stage in ("prefill", "post_decision"):
        result = next(row for row in metrics["runs"] if row["condition"] == stage and row["feature_set"] == "hidden")
        gate = next(row for row in result["click_gates"] if np.isclose(row["false_positive_budget"], .01))
        model_path = gate_root / "evaluation" / result["model_path"]
        gates[stage] = (joblib.load(model_path), float(gate["threshold"]))
        audit[stage] = {"model_path": str(model_path.resolve()), "model_sha256": sha256(model_path),
                        "threshold": float(gate["threshold"]), "val_false_positive_budget": .01,
                        "threshold_source": str((gate_root / "evaluation/metrics.json").resolve()),
                        "chosen_C": result["chosen_C"], "new_test_used_for_selection": False}
    return gates, audit


def propose_action(method, result, grounder, gates):
    valid = True
    probability = threshold = point_result = None
    if method == "text_coordinates":
        valid = bool(result["valid"])
        allowed = valid and result["action"] == "click"
        point = result.get("point")
    else:
        if method == "text_decision_shared":
            valid = result["generated_decision"] is not None
            allowed = result["generated_decision"] == "click"
        else:
            stage = "prefill" if method == "prefill_hidden" else "post_decision"
            valid = bool(result["readout_available"])
            allowed = False
            if valid:
                model, threshold = gates[stage]
                probability = float(model.predict_proba(result["hidden"][None, :])[0, list(model.classes_).index("click")])
                allowed = np.float64(probability) >= np.float64(threshold)
        point = None
        if allowed:
            point_result = grounder.predict_point(result["prefill_raw_map"], result["grid_shape"], result["image_size"])
            point = point_result["point"]
    action = {"type": "click", "x": point[0], "y": point[1]} if allowed and point is not None else {"type": "wait"}
    return action, {"parse_valid": valid, "click_probability": probability, "gate_threshold": threshold,
                    "point_readout": point_result}


def infer_method(runtime, method, image, text_token_cap):
    if method == "text_coordinates":
        return runtime.infer_text_action(image, max_new_tokens=text_token_cap)
    stage = {"prefill_hidden": "prefill", "post_hidden": "post_decision",
             "text_decision_shared": "decision_text"}[method]
    return runtime.infer_readouts(image, stage=stage)


def run_episode(env, runtime, grounder, gates, sample, manifest_dir, method, *, text_token_cap=32,
                warmup=False, cold_start=False):
    env.reset(sample, manifest_dir)
    before = env.inspect()
    started = time.perf_counter()
    image = env.observe()
    observed_at = time.perf_counter()
    result = infer_method(runtime, method, image, text_token_cap)
    inference_at = time.perf_counter()
    action, policy = propose_action(method, result, grounder, gates)
    policy_at = time.perf_counter()
    error = None
    response = None
    try:
        if action["type"] == "click":
            response = env.click(action["x"], action["y"])
    except (ValueError, RuntimeError) as exc:
        error = f"{type(exc).__name__}: {exc}"
    executed_at = time.perf_counter()
    # click() verifies the actual post-action screenshot. Waits use the same
    # verification explicitly; neither branch invents an expected side effect.
    verification = response["verification"] if response is not None else env.verify()
    after = env.inspect()
    finished = time.perf_counter()
    row = {"episode_id": f"{sample['sample_id']}__{method}" + ("__warmup" if warmup else ""),
           **{key: sample[key] for key in ("sample_id", "group_id", "label", "split")},
           "domain": sample["metadata"]["domain"], "layout": sample["metadata"]["layout"],
           "style_family": sample["metadata"]["style_family"], "method": method,
           "action": action, **policy, "model": compact_result(result),
           "n_generated_tokens": result["generated_tokens"], "env_before": before, "env_after": after,
           "verification": verification, **{key: verification[key] for key in ("pixel_changed", "changed_pixel_count")},
           "target_hit": after["target_hit"], "mapped_target": after["mapped_target"],
           "execution_error": error, "latency_measured": True, "model_load_included": False,
           "warmup": warmup, "cold_start": cold_start, "fallback": False,
           "timing_ms": {"observe": (observed_at - started) * 1000,
                         "inference": (inference_at - observed_at) * 1000,
                         "policy": (policy_at - inference_at) * 1000,
                         "execute": (executed_at - policy_at) * 1000,
                         "verify": (finished - executed_at) * 1000,
                         "total": (finished - started) * 1000},
           "execute_timing_includes_verification_for_clicks": True}
    # Numerical arrays are returned separately for diagnostics, never used to
    # repair the action or to replace its freshly measured timing.
    return row, image, result


def append_row(path, row):
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        stream.flush()


def checked_episode(output, env, runtime, grounder, gates, sample, manifest_dir, method, **kwargs):
    """Preserve infrastructure failures before aborting; never retry them silently."""
    started = time.perf_counter()
    try:
        return run_episode(env, runtime, grounder, gates, sample, manifest_dir, method, **kwargs)
    except Exception as exc:
        finished = time.perf_counter()
        partial = verification = None
        try:
            partial = env.inspect()
            verification = env.verify()
        except Exception:
            pass
        failure = {"sample_id": sample["sample_id"], "group_id": sample["group_id"],
                   "label": sample["label"], "method": method, "warmup": kwargs.get("warmup", False),
                   "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc(),
                   "elapsed_including_reset_ms": (finished - started) * 1000,
                   "partial_environment_after": partial, "verification_if_available": verification,
                   "status": "failed attempt retained; benchmark aborted; automatic retry prohibited"}
        append_row(output / "failed_attempts.jsonl", failure)
        run_file = output / "action_run.json"
        if run_file.exists():
            run = json.loads(run_file.read_text(encoding="utf-8"))
            run["status"] = "failed"
            run["last_failure"] = failure["error"]
            write_json(run_file, run)
        raise


def run_benchmark(runtime, records, manifest_dir, output, grounder, gates, provenance,
                  methods=METHODS, seed=144, test_limit=None, text_token_cap=32, browser_executable=None):
    from .action_env import ControlledSaveEnvironment
    test = [row for row in records if row["split"] == "test"]
    if test_limit is not None:
        test = test[:test_limit]
    if not test:
        raise ValueError("No held-out test scenes")
    identity = {**provenance, "methods": list(methods), "seed": seed, "text_token_cap": text_token_cap,
                "test_sample_ids": [row["sample_id"] for row in test],
                "actor_sha256": sha256(__file__), "environment_sha256": sha256(Path(__file__).with_name("action_env.py"))}
    fingerprint = stable_hash(identity)
    output.mkdir(parents=True, exist_ok=True)
    run_file = output / "action_run.json"
    log = output / "episodes.jsonl"
    failures = output / "failed_attempts.jsonl"
    if failures.exists() and failures.stat().st_size:
        raise ValueError("Failed-attempt ledger exists; refusing to hide failures with an automatic retry")
    if log.exists() and log.stat().st_size and not run_file.exists():
        raise ValueError("Episode log has no experiment identity; refusing to adopt an orphan log")
    if run_file.exists() and json.loads(run_file.read_text(encoding="utf-8"))["fingerprint"] != fingerprint:
        raise ValueError("Action output belongs to a different experiment; choose a new output path")
    rows = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines() if line.strip()] if log.exists() else []
    done = {(row["sample_id"], row["method"]) for row in rows if not row.get("warmup")}
    if len(done) != sum(not row.get("warmup") for row in rows):
        raise ValueError("Duplicate completed episode; do not silently select one")
    write_json(run_file, {"fingerprint": fingerprint, "identity": identity, "benchmark_version": BENCHMARK_VERSION,
                          "fresh_inference_for_each_episode": True, "model_runtime": runtime.runtime_metadata,
                          "load_seconds": runtime.load_seconds, "status": "running"})
    rng = random.Random(seed)
    first = True
    with ControlledSaveEnvironment(executable_path=browser_executable) as env:
        run = json.loads(run_file.read_text(encoding="utf-8"))
        run["browser"] = {"browser_version": env._browser.version,
                          "executable_path": str(env.executable_path), "headless": True,
                          "playwright_version": importlib.metadata.version("playwright"),
                          "gpu_disabled": True, "existing_user_session": False}
        write_json(run_file, run)
        warm_sample = next(row for row in records if row["split"] == "val" and row["label"] == "click")
        for method in methods:
            row, _, _ = checked_episode(output, env, runtime, grounder, gates, warm_sample, manifest_dir, method,
                                    text_token_cap=text_token_cap, warmup=True, cold_start=first)
            first = False
            append_row(log, row)
        for index, sample in enumerate(test):
            order = list(methods)
            rng.shuffle(order)
            for method in order:
                if (sample["sample_id"], method) in done:
                    continue
                row, image, result = checked_episode(output, env, runtime, grounder, gates, sample, manifest_dir, method,
                                                 text_token_cap=text_token_cap)
                append_row(log, row)
                done.add((sample["sample_id"], method))
                evidence_dir = output / "evidence" / method
                # Evidence I/O is deliberately outside the measured episode.
                if len(list(evidence_dir.glob("*_before.png"))) < 4:
                    evidence_dir.mkdir(parents=True, exist_ok=True)
                    image.save(evidence_dir / f"{sample['sample_id']}_before.png")
                    env.observe().save(evidence_dir / f"{sample['sample_id']}_after.png")
                if "prefill_raw_map" in result:
                    raw_file = output / "test_maps" / (sample["sample_id"] + ".npz")
                    if not raw_file.exists():
                        raw_file.parent.mkdir(parents=True, exist_ok=True)
                        np.savez_compressed(raw_file, raw_attention=result["prefill_raw_map"],
                                            grid_shape=result["grid_shape"], image_size=result["image_size"])
                LOGGER.info("Episode %d/%d %s %s: %s, %.1f ms", len(done), len(test) * len(methods),
                            sample["sample_id"], method, row["action"]["type"], row["timing_ms"]["total"])
    run = json.loads(run_file.read_text(encoding="utf-8"))
    run["status"] = "complete"
    run["episodes"] = len(done)
    write_json(run_file, run)
    return test


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("data/action_v3/manifest.jsonl"))
    parser.add_argument("--output", type=Path, default=Path("runs/action_v3"))
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--gate-run", type=Path, default=Path("runs/visual_state_v2_matched"))
    parser.add_argument("--phase", choices=("collect", "run", "all", "evaluate"), default="all")
    parser.add_argument("--gpu-memory", default="13GiB")
    parser.add_argument("--decision-token-cap", type=int, default=12)
    parser.add_argument("--text-token-cap", type=int, default=32)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--test-limit", type=int, help="Smoke only; uses a separate output directory")
    parser.add_argument("--grounding-limit-per-split", type=int, help="Smoke only")
    parser.add_argument("--browser-executable", type=Path)
    parser.add_argument("--seed", type=int, default=144)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    records = load_manifest(args.manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.phase == "evaluate":
        from .action_evaluation import evaluate_action_jsonl
        summary = evaluate_action_jsonl(args.output / "execution/episodes.jsonl", args.output / "evaluation",
                                        seed=args.seed, expected_methods=args.methods,
                                        expected_cases=[row for row in records if row["split"] == "test"][:args.test_limit])
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return
    from .action_model import ActionQwenRuntime
    from .action_grounding import AttentionPointGrounder
    config = config_from_args(args)
    feature_provenance = feature_identity(config)
    load_started = time.perf_counter()
    runtime = ActionQwenRuntime(config)
    runtime.load_seconds = time.perf_counter() - load_started
    if args.phase in {"collect", "all"}:
        prepared = collect_grounding(runtime, records, args.manifest.resolve().parent, args.output,
                                     feature_provenance, args.grounding_limit_per_split)
        grounder = fit_or_load_grounder(prepared, args.output)
    else:
        grounder = AttentionPointGrounder.load(args.output / "grounder/grounder.joblib")
    if args.phase in {"run", "all"}:
        gates, gate_audit = load_gates(args.gate_run)
        identity = {"feature_identity": feature_provenance, "manifest_sha256": sha256(args.manifest),
                    "grounder_sha256": sha256(args.output / "grounder/grounder.joblib"), "gates": gate_audit}
        test = run_benchmark(runtime, records, args.manifest.resolve().parent, args.output / "execution", grounder, gates, identity,
                             tuple(args.methods), args.seed, args.test_limit, args.text_token_cap, args.browser_executable)
        from .action_evaluation import evaluate_action_jsonl
        evaluate_action_jsonl(args.output / "execution/episodes.jsonl", args.output / "evaluation", seed=args.seed,
                              expected_methods=args.methods, expected_cases=test)
    print(json.dumps({"output": str(args.output.resolve()), "phase": args.phase}, ensure_ascii=False))


if __name__ == "__main__":
    main()
