"""One-component ablation: frozen attention proposal -> crop -> native coordinates.

The original public_v1 cohort, coarse proposal, gate actions and thresholds are
reused verbatim. Only the final coordinate is replaced. The refiner sees an RGB
crop and the instruction, never labels/geometry. WAIT, malformed, capped or failed
refinements fall back to the coarse point and do not add a new action veto.
This is a hybrid coordinate-token diagnostic, not a token-free action head.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from copy import deepcopy
from dataclasses import asdict
import json
import logging
import math
from pathlib import Path
import time

import numpy as np
from PIL import Image

from .action_benchmark import config_from_args, sha256, stable_hash, write_json
from .public_benchmark import cache_key, read_jsonl
from .public_diagnostics import parse_offline_lax_click
from .public_evaluation import evaluate_public_rows, point_in_geometry, quantization_oracle


LOGGER = logging.getLogger(__name__)
RUN_NAMES = ("screenspot_v2", "screenspot_pro", "osworld_g", "venusbench_gd")
GATE_METHODS = ("prefill_attention", "prefill_hidden", "prefill_combined",
                "post_attention", "post_hidden", "post_combined", "text_decision_shared")
PROTOCOL = "frozen_top1_attention_crop_native_refinement_v1"


def crop_box(patch_index, grid_shape, image_size, radius=2):
    """Integer PIL box for top-1 patch plus a fixed radius of neighboring cells."""
    gh, gw = grid_shape
    w, h = image_size
    values = (gh, gw, w, h, patch_index, radius)
    if any(isinstance(v, bool) or int(v) != v for v in values):
        raise ValueError("crop dimensions/index/radius must be integers")
    if min(gh, gw, w, h) <= 0 or not 0 <= patch_index < gh * gw or radius < 0:
        raise ValueError("invalid grid, image, index or radius")
    row, col = divmod(patch_index, gw)
    return [math.floor(max(0, col - radius) * w / gw),
            math.floor(max(0, row - radius) * h / gh),
            math.ceil(min(gw, col + radius + 1) * w / gw),
            math.ceil(min(gh, row + radius + 1) * h / gh)]


def resolve_point(native, box, fallback):
    """Parse a completed equivalent integer CLICK; map crop pixels to full image."""
    text = native.get("generated_text", "")
    if native.get("error") or native.get("inference_error"):
        return list(fallback), "fallback_error"
    if native.get("truncated") is not False or native.get("cap_reached") is True:
        return list(fallback), "fallback_incomplete_or_capped"
    if native.get("terminated_by_eos") is not True and not text.endswith(("\n", "\r")):
        return list(fallback), "fallback_no_completion"
    normalized = parse_offline_lax_click(text)
    if normalized is None:
        return list(fallback), "fallback_wait" if text.strip() in ("WAIT", "UNKNOWN") else "fallback_malformed"
    left, top, right, bottom = box
    point = [left + int(round(normalized[0] * (right - left - 1) / 1000)),
             top + int(round(normalized[1] * (bottom - top - 1) / 1000))]
    return point, "refined"


def policy_pair(original, point):
    """Replace coordinates only, preserving the old gate and parse/readout status."""
    before = deepcopy(original)
    after = deepcopy(original)
    for row, prefix in ((before, "center"), (after, "refined")):
        row["method"] = prefix + "__" + original["method"]
        # Cached old trajectories plus fresh crops are not new end-to-end timings.
        for key in ("trajectory_id", "trajectory_timing_ms", "shared_trajectory"):
            row.pop(key, None)
    after["localization_point"] = list(point)
    after["coarse_grid_shape"] = after.pop("grid_shape", None)
    after["coarse_point_readout"] = after.pop("point_readout", None)
    if after["action"]["type"] == "click":
        after["action"] = {"type": "click", "x": point[0], "y": point[1]}
    assert before["action"]["type"] == after["action"]["type"]
    assert before["parse_valid"] == after["parse_valid"]
    return before, after


def prepare(args, config):
    """Fix all 800 old cases, source hashes and model-input fields before inference."""
    from .model_features import _local_model_identity, _library_versions
    from .public_model import PUBLIC_TEXT_ACTION_SYSTEM_PROMPT
    cases, inputs, baseline, sources = [], [], {}, {}
    model = _local_model_identity(config.model_path)
    for name in RUN_NAMES:
        root = args.baseline / name
        identity = json.loads((root / "identity.json").read_text(encoding="utf-8"))
        if identity["model"] != model:
            raise ValueError("baseline checkpoint/processor differs from the selected model")
        source_cases = read_jsonl(root / "selected_manifest.jsonl")
        if len(source_cases) != 200:
            raise ValueError("this ablation requires every original 200-case cohort")
        by_id = defaultdict(dict)
        for row in read_jsonl(root / "predictions.jsonl"):
            if row["method"] in by_id[row["sample_id"]]:
                raise ValueError("duplicate baseline prediction")
            by_id[row["sample_id"]][row["method"]] = row
        sources[name] = {f: sha256(root / f) for f in ("identity.json", "selected_manifest.jsonl", "predictions.jsonl")}
        for case in source_cases:
            sid = case["sample_id"]
            original = by_id[sid]
            row = original["prefill_hidden"]
            for method in GATE_METHODS:
                other = original[method]
                if other["localization_point"] != row["localization_point"]:
                    raise ValueError("baseline policies do not share a coarse proposal")
            key = cache_key(sid)
            saved = json.loads((root / "rows" / (key + ".json")).read_text(encoding="utf-8"))
            if stable_hash(saved["rows"]) != stable_hash(list(original.values())):
                # JSONL may have a different order; compare by method.
                if {r["method"]: r for r in saved["rows"]} != original:
                    raise ValueError("baseline JSONL differs from committed case cache")
            box = crop_box(row["point_readout"]["patch_index"], row["grid_shape"], row["model_image_size"], args.crop_radius)
            inputs.append({"sample_id": sid, "dataset": case["dataset"], "image": case["image"],
                           "instruction": case["instruction"], "image_size": row["model_image_size"],
                           "coarse_point": row["localization_point"], "crop_box": box,
                           "image_sha256": saved["image_sha256"], "source_run": name})
            cases.append(case)
            baseline[sid] = original
    if len({r["sample_id"] for r in cases}) != len(cases):
        raise ValueError("duplicate case in the fixed cohort")
    module_root = Path(__file__).parent
    identity = {"protocol": PROTOCOL, "model": model, "libraries": _library_versions(),
                "config": asdict(config), "source_runs": sources,
                "coordinate_prompt_unchanged": PUBLIC_TEXT_ACTION_SYSTEM_PROMPT,
                "crop_radius": args.crop_radius, "crop_side_in_cells": 2 * args.crop_radius + 1,
                "text_token_cap": args.text_token_cap,
                "parser": "whole integer CLICK with comma/whitespace; uncapped completed output only",
                "fallback": "old point; never override old gate", "uses_explicit_coordinate_tokens": True,
                "no_retraining": True, "no_test_parameter_search": True,
                "input_digest": stable_hash(inputs), "n_cases": len(cases),
                "source_sha256": {name: sha256(module_root / name) for name in (
                    "public_refinement.py", "public_model.py", "action_model.py", "model_features.py",
                    "public_diagnostics.py", "public_evaluation.py")}}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "identity.json"
    if path.exists() and stable_hash(json.loads(path.read_text(encoding="utf-8"))) != stable_hash(identity):
        raise ValueError("ablation configuration changed; use a new output directory")
    write_json(path, identity)
    for name, rows in (("selected_manifest.jsonl", cases), ("model_inputs.jsonl", inputs)):
        (args.output / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    return cases, inputs, baseline, stable_hash(identity)


def point_hit(point, case):
    w, h = case["image_size"]
    return point_in_geometry((point[0] / w, point[1] / h), case["target_geometry"])


def paired_delta_ci(records, seed=144):
    """Paired screenshot-group bootstrap of localization accuracy difference."""
    groups = sorted({r["group"] for r in records})
    if len(groups) < 2:
        return None
    index = {g: i for i, g in enumerate(groups)}
    den, delta = np.zeros(len(groups)), np.zeros(len(groups))
    for r in records:
        den[index[r["group"]]] += 1
        delta[index[r["group"]]] += int(r["after_hit"]) - int(r["before_hit"])
    draws = np.random.default_rng(seed).multinomial(len(groups), np.full(len(groups), 1 / len(groups)), size=2000)
    values = draws @ delta / (draws @ den)
    return [float(v) for v in np.percentile(values, [2.5, 97.5])]


def summarize(cases, baseline, results, output):
    records, prediction_rows = [], []
    for case in cases:
        sid = case["sample_id"]
        result = results[sid]
        for method in GATE_METHODS:
            prediction_rows.extend(policy_pair(baseline[sid][method], result["point"]))
        if case["expected_actionable"] and case["target_geometry"] is not None:
            coarse = baseline[sid]["prefill_hidden"]
            records.append({"sample_id": sid, "dataset": case["dataset"],
                            "group": case.get("group_id") or case.get("image_id") or sid,
                            "before_hit": point_hit(coarse["localization_point"], case),
                            "after_hit": point_hit(result["point"], case),
                            "center_oracle": quantization_oracle(case["target_geometry"], coarse["grid_shape"], case["image_size"]),
                            "before_point": coarse["localization_point"], "after_point": result["point"],
                            "refinement_status": result["refinement_status"]})
    (output / "predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in prediction_rows), encoding="utf-8")
    expected = tuple(prefix + "__" + method for method in GATE_METHODS for prefix in ("center", "refined"))
    evaluate_public_rows(prediction_rows, cases, output / "evaluation", bootstrap_iterations=0, expected_methods=expected)
    datasets = {}
    for name in sorted({c["dataset"] for c in cases}):
        rows = [r for r in records if r["dataset"] == name]
        local_cases = [c for c in cases if c["dataset"] == name]
        before = sum(r["before_hit"] for r in rows)
        after = sum(r["after_hit"] for r in rows)
        gates = {}
        for method in GATE_METHODS:
            old_ok = new_ok = false_clicks = 0
            for c in local_cases:
                old = baseline[c["sample_id"]][method]
                allowed = old["action"]["type"] == "click" and old["parse_valid"] is True
                if c["expected_actionable"]:
                    old_ok += allowed and point_hit(old["localization_point"], c)
                    new_ok += allowed and point_hit(results[c["sample_id"]]["point"], c)
                else:
                    false_clicks += old["action"]["type"] == "click"
            gates[method] = {"before_correct_clicks": old_ok, "after_correct_clicks": new_ok,
                             "positive_denominator": len(rows), "negative_clicks_both": false_clicks,
                             "negative_denominator": sum(not c["expected_actionable"] for c in local_cases)}
        timings = [results[c["sample_id"]]["native"]["elapsed_seconds"] for c in local_cases if "elapsed_seconds" in results[c["sample_id"]]["native"]]
        datasets[name] = {"n_cases": len(local_cases), "positive_denominator": len(rows),
                          "before_hits": before, "after_hits": after,
                          "before_accuracy": before / len(rows), "after_accuracy": after / len(rows),
                          "paired_delta_group_bootstrap_95ci": paired_delta_ci(rows),
                          "fixed_errors": sum(not r["before_hit"] and r["after_hit"] for r in rows),
                          "regressions": sum(r["before_hit"] and not r["after_hit"] for r in rows),
                          "rescued_unreachable_targets": sum(not r["center_oracle"] and r["after_hit"] for r in rows),
                          "refinement_status_counts": dict(Counter(results[c["sample_id"]]["refinement_status"] for c in local_cases)),
                          "additional_native_inference_seconds": {"mean": float(np.mean(timings)), "median": float(np.median(timings))},
                          "gates": gates}
    summary = {"protocol": PROTOCOL, "complete": True, "n_cases": len(cases), "n_geometry_positives": len(records),
               "n_predictions": len(prediction_rows), "datasets": datasets,
               "n_inference_errors": sum(bool(r["native"].get("error")) for r in results.values()),
               "gate_actions_and_thresholds_unchanged": True, "static_screenshot_scoring": True,
               "timing_note": "Only additional native crop inference measured fresh; cached coarse/gate baseline is not an independent end-to-end latency measurement."}
    write_json(output / "summary.json", summary)
    (output / "paired_localization.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")
    lines = ["# 固定注意力候选的局部坐标精定位", "", "只替换坐标；复用原全部800题、注意力选格、模型与门控。5×5格裁剪，不用GT选区域。局部原生坐标解码失败/WAIT保留旧点。此为显式坐标token的混合诊断方案，静态截图判分。", "",
             "| 数据集 | 定位正例 | 原中心命中 | 精定位命中 | 修复错误 | 新增错误 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    for name, d in datasets.items():
        lines.append(f"| {name} | {d['positive_denominator']} | {d['before_hits']} ({d['before_accuracy']:.1%}) | {d['after_hits']} ({d['after_accuracy']:.1%}) | {d['fixed_errors']} | {d['regressions']} |")
    lines += ["", "全部门控动作类型与阈值不变，因此负例触发率按构造不变；这不证明误点控制有改善。门控后的正确点击、配对截图bootstrap区间、回退/错误及额外推理耗时见summary.json与evaluation/。旧基线沿用缓存，不能声称新端到端时延或提速。"]
    (output / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary


def run(args):
    config = config_from_args(args)
    cases, inputs, baseline, fingerprint = prepare(args, config)
    if args.prepare_only:
        LOGGER.info("Protocol and %d model inputs fixed before inference", len(inputs))
        return
    runtime, results = None, {}
    started = time.perf_counter()
    for index, model_input in enumerate(inputs, 1):
        sid = model_input["sample_id"]
        path = args.output / "rows" / (cache_key(sid) + ".json")
        if path.exists():
            result = json.loads(path.read_text(encoding="utf-8"))
            if result["fingerprint"] != fingerprint or result["model_input"] != model_input:
                raise ValueError("refinement cache input/config mismatch")
        else:
            if args.evaluate_only:
                raise ValueError(f"missing refinement cache: {sid}")
            if sha256(model_input["image"]) != model_input["image_sha256"]:
                raise ValueError("image changed since the original benchmark")
            if runtime is None:
                from .public_model import PublicQwenRuntime
                runtime = PublicQwenRuntime(config)
                write_json(args.output / "runtime.json", {"load_seconds": runtime.load_seconds, "runtime_metadata": runtime.runtime_metadata})
            with Image.open(model_input["image"]) as source:
                if list(source.size) != model_input["image_size"]:
                    raise ValueError("baseline image dimensions differ")
                crop = source.convert("RGB").crop(model_input["crop_box"])
            # Deliberately pass only pixels + instruction. Labels stay in evaluator.
            native = runtime.infer_text_action_public(crop, model_input["instruction"], args.text_token_cap)
            point, status = resolve_point(native, model_input["crop_box"], model_input["coarse_point"])
            result = {"fingerprint": fingerprint, "model_input": model_input,
                      "point": point, "refinement_status": status, "native": native}
            write_json(path, result)
        results[sid] = result
        LOGGER.info("Refine %d/%d %s %s %s %.1fs", index, len(inputs), model_input["dataset"], sid,
                    result["refinement_status"], time.perf_counter() - started)
        write_json(args.output / "progress.json", {"completed": index, "expected": len(inputs),
                                                   "sample_id": sid, "elapsed_seconds": time.perf_counter() - started,
                                                   "status_counts": dict(Counter(r["refinement_status"] for r in results.values()))})
    summary = summarize(cases, baseline, results, args.output)
    write_json(args.output / "completion.json", {"complete": True, "fingerprint": fingerprint,
                                                 "n_cases": len(cases), "elapsed_seconds": time.perf_counter() - started})
    LOGGER.info("Complete: %s", json.dumps(summary, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, default=Path("runs/public_v1"))
    parser.add_argument("--output", type=Path, default=Path("runs/public_refinement_v1"))
    parser.add_argument("--model", type=Path, default=Path("models/Qwen3-VL-8B-Instruct"))
    parser.add_argument("--crop-radius", type=int, default=2)
    parser.add_argument("--text-token-cap", type=int, default=32)
    parser.add_argument("--decision-token-cap", type=int, default=12)
    parser.add_argument("--gpu-memory", default="13GiB")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    args.baseline, args.output = args.baseline.resolve(), args.output.resolve()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args)


if __name__ == "__main__":
    main()
