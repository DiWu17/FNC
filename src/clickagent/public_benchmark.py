"""Zero-retraining public screenshot transfer of the existing frozen policies.

Select a reproducible cohort before model loading. All eight policies use that
same cohort. No public label is supplied to an inference or policy function.
This scores static screenshots, not browser execution or application success.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import logging
from pathlib import Path
import time
import traceback

import joblib
import numpy as np
from PIL import Image

from .action_benchmark import config_from_args, sha256, stable_hash, write_json
from .action_grounding import AttentionPointGrounder
from .public_evaluation import evaluate_public_rows


LOGGER = logging.getLogger(__name__)
METHODS = ("prefill_hidden", "post_hidden", "text_decision_shared", "text_coordinates",
           "prefill_attention", "prefill_combined", "post_attention", "post_combined")
SELECTION_VERSION = 1


def read_jsonl(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def select_cohort(records, per_dataset=200, seed=144):
    """Fixed hash order, balanced actionable/nonactionable where possible.

    All available negatives are retained if fewer than half the budget; the
    unused slots go to positives, and conversely. Click-compatible ConflictGUI
    is a prespecified subtask, not a reproduction of its multi-action scorer.
    """
    if per_dataset < 1:
        raise ValueError("per_dataset must be positive")
    datasets = sorted({r["dataset"] for r in records})
    selected, audit = [], {}
    for dataset in datasets:
        available = [r for r in records if r["dataset"] == dataset and r.get("split", "test") == "test"]
        eligible = available
        if "conflict" in dataset.lower():
            eligible = [r for r in available if str((r.get("metadata") or {}).get("reference_action_type", "")).lower()
                        in {"click", "tap"}]
            if not eligible:
                raise ValueError("ConflictGUI needs original click/tap reference_action_type metadata")
        def order(row):
            return hashlib.sha256(f"{seed}:{dataset}:{row['sample_id']}".encode()).hexdigest()
        positive = sorted((r for r in eligible if r["expected_actionable"]), key=order)
        negative = sorted((r for r in eligible if not r["expected_actionable"]), key=order)
        n = min(per_dataset, len(eligible))
        n_negative = min(len(negative), n // 2)
        n_positive = min(len(positive), n - n_negative)
        n_negative = min(len(negative), n - n_positive)
        chosen = sorted(positive[:n_positive] + negative[:n_negative], key=order)
        selected.extend(chosen)
        audit[dataset] = {"available_test": len(available), "eligible_test": len(eligible),
                          "selected": len(chosen), "positive": n_positive, "negative": n_negative,
                          "selected_task_kinds": dict(Counter(r["task_kind"] for r in chosen)),
                          "selection_uses_model_results": False,
                          "eligibility": "reference action click/tap only" if "conflict" in dataset.lower() else "all test records"}
    if len({r["sample_id"] for r in selected}) != len(selected):
        raise ValueError("duplicate selected sample IDs")
    return selected, {"version": SELECTION_VERSION, "seed": seed, "per_dataset_cap": per_dataset,
                      "order": "SHA256(seed:dataset:sample_id)",
                      "negative_budget": "up to half; fill remaining slots from other class",
                      "warning": "Refusal-enriched sample; aggregate accuracy is not the official full benchmark score.",
                      "datasets": audit}


def load_frozen_gates(root):
    root = Path(root)
    metric_path = root / "evaluation/metrics.json"
    metrics = json.loads(metric_path.read_text(encoding="utf-8"))
    gates, audit = {}, {}
    for stage in ("prefill", "post_decision"):
        for features in ("hidden", "attention", "combined"):
            entry = next(r for r in metrics["runs"] if r["condition"] == stage and r["feature_set"] == features)
            gate = next(r for r in entry["click_gates"] if np.isclose(r["false_positive_budget"], .01))
            path = root / "evaluation" / entry["model_path"]
            key = (stage, features)
            gates[key] = (joblib.load(path), float(gate["threshold"]))
            audit[f"{stage}__{features}"] = {"path": str(path.resolve()), "sha256": sha256(path),
                                           "threshold": float(gate["threshold"]), "chosen_C": entry["chosen_C"],
                                           "threshold_source": str(metric_path.resolve()),
                                           "old_validation_false_positive_budget": .01,
                                           "public_test_used_for_selection": False}
    return gates, audit


def policies_from_joint(result, grounder, gates):
    """Only model-generated vectors/maps/words enter this decision function."""
    if tuple(result["head_order"]) != tuple(grounder.head_order):
        raise ValueError("frozen localizer head order differs from recorder")
    point_info = grounder.predict_point(result["prefill_raw_map"], result["grid_shape"], result["image_size"])
    point = point_info["point"]
    rows = []
    for method in METHODS:
        if method == "text_coordinates":
            continue
        probability = threshold = None
        if method == "text_decision_shared":
            valid = result["generated_decision"] is not None
            allowed = valid and result["generated_decision"] == "click"
        else:
            prefix, feature_set = method.split("_", 1)
            stage = "prefill" if prefix == "prefill" else "post_decision"
            snapshot = result[stage]
            valid, allowed = snapshot is not None, False
            if valid:
                features = (np.concatenate([snapshot["attention"], snapshot["hidden"]])
                            if feature_set == "combined" else snapshot[feature_set])
                model, threshold = gates[stage, feature_set]
                probability = float(model.predict_proba(np.asarray(features)[None, :])[0, list(model.classes_).index("click")])
                allowed = np.float64(probability) >= np.float64(threshold)
        rows.append({"method": method,
                     "action": {"type": "click", "x": point[0], "y": point[1]} if allowed else {"type": "wait"},
                     "parse_valid": bool(valid), "click_probability": probability, "gate_threshold": threshold,
                     "localization_point": point, "point_readout": point_info,
                     "generated_decision": result["generated_decision"]})
    return rows


def cache_key(sample_id):
    return hashlib.sha256(sample_id.encode()).hexdigest()[:24]


def save_joint(path, result, fingerprint, image_hash):
    arrays = {"prefill_raw_map": result["prefill_raw_map"]}
    for stage in ("prefill", "post_decision"):
        if result[stage] is not None:
            for key in ("attention", "hidden"):
                arrays[f"{stage}__{key}"] = result[stage][key]
    metadata = {k: v for k, v in result.items() if k not in {"prefill", "post_decision", "prefill_raw_map"}}
    metadata.update({"fingerprint": fingerprint, "image_sha256": image_hash,
                     "snapshot_metadata": {stage: ({k: v for k, v in result[stage].items()
                                                   if k not in {"attention", "hidden", "raw_image_attention"}}
                                                  if result[stage] is not None else None)
                                           for stage in ("prefill", "post_decision")}})
    arrays["metadata"] = np.asarray(json.dumps(metadata, ensure_ascii=False, allow_nan=False))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(path)


def load_joint(path, fingerprint, image_hash):
    with np.load(path, allow_pickle=False) as archive:
        result = json.loads(str(archive["metadata"]))
        if result.pop("fingerprint") != fingerprint or result.pop("image_sha256") != image_hash:
            raise ValueError("joint cache model/input fingerprint mismatch")
        snapshots = result.pop("snapshot_metadata")
        for stage in ("prefill", "post_decision"):
            result[stage] = snapshots[stage]
            if result[stage] is not None:
                result[stage].update({key: archive[f"{stage}__{key}"].copy() for key in ("attention", "hidden")})
        result["prefill_raw_map"] = archive["prefill_raw_map"].copy()
    return result


def prepare_selection(args):
    args.output.mkdir(parents=True, exist_ok=True)
    records = read_jsonl(args.manifest)
    for row in records:
        path = Path(row["image"])
        if not path.is_absolute():
            row["image"] = str((args.manifest.parent / path).resolve())
    selected, audit = select_cohort(records, args.per_dataset, args.seed)
    if not selected:
        raise ValueError("empty selected test cohort")
    manifest_text = "".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in selected)
    path = args.output / "selected_manifest.jsonl"
    if path.exists() and path.read_text(encoding="utf-8") != manifest_text:
        raise ValueError("existing output has a different selected cohort; choose new output")
    path.write_text(manifest_text, encoding="utf-8")
    audit.update({"source_manifest": str(args.manifest.resolve()), "source_sha256": sha256(args.manifest),
                  "selected_manifest_sha256": sha256(path), "fixed_before_model_loading": True})
    old_audit = args.output / "selection.json"
    if old_audit.exists() and json.loads(old_audit.read_text(encoding="utf-8")) != audit:
        raise ValueError("selection audit mismatch")
    write_json(old_audit, audit)
    return selected


def make_identity(args, config, gates_audit):
    from .model_features import _local_model_identity, _library_versions
    from .public_model import PUBLIC_DECISION_SYSTEM_PROMPT, PUBLIC_TEXT_ACTION_SYSTEM_PROMPT, PUBLIC_PROTOCOL
    source = Path(__file__).parent
    return {"protocol": PUBLIC_PROTOCOL, "config": asdict(config),
            "model": _local_model_identity(config.model_path), "libraries": _library_versions(),
            "prompts": {"decision": PUBLIC_DECISION_SYSTEM_PROMPT, "coordinates": PUBLIC_TEXT_ACTION_SYSTEM_PROMPT},
            "selected_manifest_sha256": sha256(args.output / "selected_manifest.jsonl"),
            "gates": gates_audit,
            "grounder": {"path": str(args.grounder.resolve()), "sha256": sha256(args.grounder)},
            "methods": METHODS, "text_token_cap": args.text_token_cap,
            "source_sha256": {name: sha256(source / name) for name in
                              ("public_benchmark.py", "public_model.py", "action_model.py", "model_features.py",
                               "attention_features.py", "action_grounding.py")},
            "no_retraining": True, "no_public_threshold_selection": True,
            "static_screenshot_scoring": True, "independent_policy_latency_comparison": False}


def run(args):
    selected = prepare_selection(args)
    if args.prepare_only:
        LOGGER.info("Selection fixed: %d cases; model not loaded", len(selected))
        return
    config = config_from_args(args)
    gates, gate_audit = load_frozen_gates(args.gate_root)
    grounder = AttentionPointGrounder.load(args.grounder)
    identity = make_identity(args, config, gate_audit)
    fingerprint = stable_hash(identity)
    identity_path = args.output / "identity.json"
    if identity_path.exists() and stable_hash(json.loads(identity_path.read_text(encoding="utf-8"))) != fingerprint:
        raise ValueError("run implementation/weights/protocol changed; select new output directory")
    write_json(identity_path, identity)
    output_rows = args.output / "rows"
    output_rows.mkdir(exist_ok=True)
    runtime = None
    all_rows = []
    run_started = time.perf_counter()
    for index, sample in enumerate(selected, 1):
        key = cache_key(sample["sample_id"])
        row_path = output_rows / f"{key}.json"
        image_hash = sha256(sample["image"])
        if row_path.exists():
            saved = json.loads(row_path.read_text(encoding="utf-8"))
            if saved["fingerprint"] != fingerprint or saved["image_sha256"] != image_hash:
                raise ValueError("existing prediction belongs to a different run/input")
            rows = saved["rows"]
        else:
            with Image.open(sample["image"]) as opened:
                image = opened.convert("RGB")
            if runtime is None:
                from .public_model import PublicQwenRuntime
                runtime = PublicQwenRuntime(config)
                write_json(args.output / "runtime.json", {"load_seconds": runtime.load_seconds,
                                                         "runtime_metadata": runtime.runtime_metadata})
            joint_path = args.output / "features" / f"{key}.npz"
            rows = []
            try:
                if joint_path.exists():
                    joint = load_joint(joint_path, fingerprint, image_hash)
                else:
                    joint = runtime.infer_joint(image, sample["instruction"])
                    save_joint(joint_path, joint, fingerprint, image_hash)
                rows = policies_from_joint(joint, grounder, gates)
                for row in rows:
                    row.update({"model_image_size": joint["image_size"], "grid_shape": joint["grid_shape"],
                                "trajectory_id": f"{key}:joint", "trajectory_timing_ms":
                                {"total": joint["elapsed_seconds"] * 1000,
                                 **{k: v * 1000 if v is not None else None for k, v in joint["timing_seconds"].items()}},
                                "cap_reached": joint["truncated"], "generated_text": joint["generated_text"],
                                "generated_token_ids": joint["generated_token_ids"],
                                "shared_trajectory": True})
            except Exception as exc:
                LOGGER.exception("Joint inference failed: %s", sample["sample_id"])
                error = {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}
                rows = [{"method": m, "parse_valid": False, "action": {"type": "wait"},
                         "model_image_size": list(image.size), **error} for m in METHODS if m != "text_coordinates"]
            native_path = args.output / "native" / f"{key}.json"
            try:
                if native_path.exists():
                    native = json.loads(native_path.read_text(encoding="utf-8"))
                    if native.pop("fingerprint") != fingerprint or native.pop("image_sha256") != image_hash:
                        raise ValueError("coordinate cache input mismatch")
                else:
                    native = runtime.infer_text_action_public(image, sample["instruction"], args.text_token_cap)
                    write_json(native_path, {**native, "fingerprint": fingerprint, "image_sha256": image_hash})
                point = native["point"]
                rows.append({"method": "text_coordinates", "parse_valid": native["parse_valid"],
                             "action": {"type": "click", "x": point[0], "y": point[1]} if point is not None else {"type": "wait"},
                             "localization_point": point, "model_image_size": native["image_size"],
                             "grid_shape": native["grid_shape"], "generated_decision": native["generated_decision"],
                             "generated_text": native["generated_text"], "generated_token_ids": native["generated_token_ids"],
                             "cap_reached": native["truncated"], "trajectory_id": f"{key}:coordinates",
                             "trajectory_timing_ms": {"total": native["elapsed_seconds"] * 1000},
                             "shared_trajectory": False})
            except Exception as exc:
                LOGGER.exception("Coordinate inference failed: %s", sample["sample_id"])
                rows.append({"method": "text_coordinates", "parse_valid": False, "action": {"type": "wait"},
                             "model_image_size": list(image.size), "error": f"{type(exc).__name__}: {exc}",
                             "traceback": traceback.format_exc()})
            for row in rows:
                row.update({"sample_id": sample["sample_id"], "dataset": sample["dataset"],
                            "protocol": identity["protocol"], "prediction_ground_truth_used": False})
            write_json(row_path, {"fingerprint": fingerprint, "image_sha256": image_hash, "rows": rows})
        all_rows.extend(rows)
        LOGGER.info("Public %d/%d %s %s; %.1fs elapsed", index, len(selected), sample["dataset"],
                    sample["sample_id"], time.perf_counter() - run_started)
        write_json(args.output / "progress.json", {"completed": index, "expected": len(selected),
                                                   "sample_id": sample["sample_id"],
                                                   "elapsed_seconds": time.perf_counter() - run_started})
    (args.output / "predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in all_rows), encoding="utf-8")
    summary = evaluate_public_rows(all_rows, selected, args.output / "evaluation", args.seed,
                                   args.bootstrap_iterations, expected_methods=METHODS)
    write_json(args.output / "completion.json", {"complete": True, "n_cases": len(selected),
                                                "n_rows": len(all_rows), "fingerprint": fingerprint,
                                                "elapsed_seconds": time.perf_counter() - run_started,
                                                "error_rows": sum(bool(r.get("error")) for r in all_rows)})
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("runs/public_v1"))
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--gate-root", type=Path, default=Path("runs/visual_state_v2_matched"))
    parser.add_argument("--grounder", type=Path, default=Path("runs/action_v3/grounder/grounder.joblib"))
    parser.add_argument("--per-dataset", type=int, default=200)
    parser.add_argument("--seed", type=int, default=144)
    parser.add_argument("--decision-token-cap", type=int, default=12)
    parser.add_argument("--text-token-cap", type=int, default=32)
    parser.add_argument("--gpu-memory", default="13GiB")
    parser.add_argument("--bootstrap-iterations", type=int, default=1000)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
