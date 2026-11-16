"""Offline structural diagnosis of the completed public_v1/refinement outputs.

No VLM inference, fitting, calibration, test-time parameter selection, or changes
to formal predictions. Pair diagnostics use actual identical screenshot hashes.
"""
from collections import Counter, defaultdict
from itertools import combinations
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import roc_auc_score

from analyze_public_failures import distance_to_region
from summarize_refinement import intersects_crop, render_example
from clickagent.action_benchmark import write_json, sha256
from clickagent.public_benchmark import cache_key, read_jsonl, load_frozen_gates
from clickagent.public_evaluation import point_in_geometry


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/public_refinement_v1"
OUT = RUN / "structural_analysis"
NAMES = ("screenspot_v2", "screenspot_pro", "osworld_g", "venusbench_gd")
METHODS = ("prefill_attention", "prefill_hidden", "prefill_combined",
           "post_attention", "post_hidden", "post_combined", "text_decision_shared")


def distribution(values):
    a = np.asarray(list(values), dtype=float)
    return {"n": len(a), **({"median": float(np.median(a)), "p05": float(np.percentile(a, 5)),
                            "p95": float(np.percentile(a, 95)), "mean": float(a.mean())} if len(a) else {})}


def hit(point, case):
    return bool(case["expected_actionable"] and case["target_geometry"] is not None and
                point is not None and point_in_geometry((point[0]/case["image_size"][0], point[1]/case["image_size"][1]), case["target_geometry"]))


def envelope_boxes(geometry):
    if geometry["type"] == "bbox":
        return geometry["boxes"]
    return [[min(p[0] for p in poly), min(p[1] for p in poly),
             max(p[0] for p in poly), max(p[1] for p in poly)] for poly in geometry["polygons"]]


def definitely_disjoint(a, b):
    return all(x[2] < y[0] or y[2] < x[0] or x[3] < y[1] or y[3] < x[1]
               for x in envelope_boxes(a) for y in envelope_boxes(b))


def feature_drift(model, x):
    z = model.named_steps["scale"].transform(x)
    return {"mean_absolute_training_z": distribution(np.abs(z).mean(axis=1)),
            "fraction_features_abs_z_gt_5": distribution((np.abs(z) > 5).mean(axis=1))}


def main():
    import matplotlib
    matplotlib.use("Agg")
    OUT.mkdir(parents=True, exist_ok=True)
    cases = read_jsonl(RUN / "selected_manifest.jsonl")
    baseline, sources = {}, {}
    for name in NAMES:
        path = ROOT / "runs/public_v1" / name / "predictions.jsonl"
        sources[str(path)] = sha256(path)
        for r in read_jsonl(path):
            baseline[r["sample_id"], r["method"]] = r
    diagnosis_path = ROOT / "runs/public_v1/analysis/diagnostics.json"
    diagnosis = json.loads(diagnosis_path.read_text(encoding="utf-8"))
    full_native = {r["sample_id"]: r for d in diagnosis["datasets"].values() for r in d["offline_lax_coordinates"]["records"]}
    results, records, feature_rows = {}, [], defaultdict(list)
    for index, c in enumerate(cases):
        sid, key = c["sample_id"], cache_key(c["sample_id"])
        result = json.loads((RUN / "rows" / (key + ".json")).read_text(encoding="utf-8"))
        results[sid] = result
        old = baseline[sid, "prefill_hidden"]
        coarse_point = old["localization_point"]
        fine_point = result["point"]
        covered = bool(c["expected_actionable"] and (
            intersects_crop(c["target_geometry"], result["model_input"]["crop_box"], c["image_size"]) or hit(fine_point, c)))
        records.append({"sample_id": sid, "dataset": c["dataset"], "positive": c["expected_actionable"],
                        "instruction": c["instruction"], "image_sha256": result["model_input"]["image_sha256"],
                        "coarse_point": coarse_point, "fine_point": fine_point, "crop": result["model_input"]["crop_box"],
                        "coarse_score": old["point_readout"]["score"], "coarse_patch": old["point_readout"]["patch_index"],
                        "global_word": old["generated_decision"], "global_text": old["generated_text"],
                        "local_text": result["native"]["generated_text"], "local_status": result["refinement_status"],
                        "crop_contains_target": covered, "coarse_hit": hit(coarse_point, c), "fine_hit": hit(fine_point, c),
                        "full_native_hit": full_native[sid]["offline_grounding_hit"] is True,
                        "full_native_text": full_native[sid]["generated_text"],
                        "fine_distance_px": distance_to_region(fine_point, c["target_geometry"], c["image_size"]) if c["expected_actionable"] else None,
                        "gates": {m: {"allow": baseline[sid, m]["action"]["type"] == "click",
                                      "p": baseline[sid, m].get("click_probability"),
                                      "threshold": baseline[sid, m].get("gate_threshold")}
                                  for m in METHODS}})
        name = result["model_input"]["source_run"]
        with np.load(ROOT / "runs/public_v1" / name / "features" / (key + ".npz"), allow_pickle=False) as archive:
            for stage in ("prefill", "post_decision"):
                for feature in ("attention", "hidden"):
                    feature_rows[stage, feature].append(archive[f"{stage}__{feature}"].copy())
        if (index + 1) % 100 == 0:
            print("loaded", index + 1, flush=True)
    features = {k: np.stack(v) for k, v in feature_rows.items()}
    gates, gate_audit = load_frozen_gates(ROOT / "runs/visual_state_v2_matched")
    shift, max_replay_error = {}, 0.0
    for stage in ("prefill", "post_decision"):
        with np.load(ROOT / "runs/visual_state_v2_matched/features" / (stage + ".npz"), allow_pickle=False) as archive:
            old_features = {k: archive[k].copy() for k in ("attention", "hidden")}
            splits = archive["splits"].astype(str)
        for feature in ("attention", "hidden", "combined"):
            model, threshold = gates[stage, feature]
            method = ("prefill" if stage == "prefill" else "post") + "_" + feature
            x = np.concatenate([features[stage, "attention"], features[stage, "hidden"]], axis=1) if feature == "combined" else features[stage, feature]
            old_x = np.concatenate([old_features["attention"], old_features["hidden"]], axis=1) if feature == "combined" else old_features[feature]
            # Match the original one-row sklearn call, avoiding float32 BLAS
            # batch-vs-vector differences near saturated probabilities.
            probabilities = np.vstack([model.predict_proba(row[None, :])[0] for row in x])
            classes = list(model.classes_)
            predicted = np.asarray(classes)[probabilities.argmax(axis=1)]
            click_index = classes.index("click")
            conditional = defaultdict(Counter)
            for i, r in enumerate(records):
                cached = r["gates"][method]["p"]
                if cached is not None:
                    max_replay_error = max(max_replay_error, abs(float(probabilities[i, click_index]) - cached))
                r["gates"][method]["argmax_class"] = str(predicted[i])
                r["gates"][method]["all_class_probabilities"] = {str(k): float(v) for k, v in zip(classes, probabilities[i])}
                conditional[r["global_word"]][str(predicted[i])] += 1
            old_test = old_x[splits == "test"]
            shift[method] = {"old_train": feature_drift(model, old_x[splits == "train"]),
                             "old_test": feature_drift(model, old_test), "public": feature_drift(model, x),
                             "public_argmax_class_counts": dict(Counter(predicted)),
                             "old_test_argmax_class_counts": dict(Counter(model.predict(old_test))),
                             "public_class_given_generated_word": {str(k): dict(v) for k, v in conditional.items()}}
    assert max_replay_error < 1e-5, max_replay_error
    print("gate replay complete", max_replay_error, flush=True)
    by_id = {c["sample_id"]: c for c in cases}
    rec_by_id = {r["sample_id"]: r for r in records}
    groups = defaultdict(list)
    for r in records:
        if r["positive"]:
            groups[r["image_sha256"]].append(r)
    pairs = []
    for rows in groups.values():
        for a, b in combinations(rows, 2):
            if a["instruction"] == b["instruction"] or not definitely_disjoint(by_id[a["sample_id"]]["target_geometry"], by_id[b["sample_id"]]["target_geometry"]):
                continue
            pairs.append({"dataset": a["dataset"], "a": a["sample_id"], "b": b["sample_id"],
                          "same_coarse_patch": a["coarse_patch"] == b["coarse_patch"],
                          "same_crop": a["crop"] == b["crop"], "same_fine_point": a["fine_point"] == b["fine_point"],
                          "native_full_hits_both": a["full_native_hit"] and b["full_native_hit"],
                          "fine_hits_both": a["fine_hit"] and b["fine_hit"]})
    def stage_summary(rows):
        pos = [r for r in rows if r["positive"]]
        neg = [r for r in rows if not r["positive"]]
        covered = [r for r in pos if r["crop_contains_target"]]
        uncovered = [r for r in pos if not r["crop_contains_target"]]
        summary = {"n": len(rows), "positive": len(pos), "negative": len(neg),
                   "global_word_positive": dict(Counter(r["global_word"] for r in pos)),
                   "global_word_negative": dict(Counter(r["global_word"] for r in neg)),
                   "positive_crop_missing": len(uncovered), "positive_crop_present": len(covered),
                   "final_hits": sum(r["fine_hit"] for r in pos),
                   "covered_failures": sum(not r["fine_hit"] for r in covered),
                   "covered_local_status_vs_outcome": dict(Counter(r["local_status"] + ("_hit" if r["fine_hit"] else "_miss") for r in covered)),
                   "uncovered_local_status": dict(Counter(r["local_status"] for r in uncovered)),
                   "local_status_positive": dict(Counter(r["local_status"] for r in pos)),
                   "local_status_negative": dict(Counter(r["local_status"] for r in neg)),
                   "high_coarse_score_but_target_outside_crop": sum(r["coarse_score"] >= .9 for r in uncovered),
                   "gates": {}}
        for method in METHODS:
            p = [r["gates"][method]["p"] for r in pos]
            loc_auc = roc_auc_score([r["fine_hit"] for r in pos], p) if method != "text_decision_shared" and len({r["fine_hit"] for r in pos}) == 2 else None
            summary["gates"][method] = {
                "positive_allow": sum(r["gates"][method]["allow"] for r in pos),
                "negative_click": sum(r["gates"][method]["allow"] for r in neg),
                "correct_coordinate_allowed": sum(r["fine_hit"] and r["gates"][method]["allow"] for r in pos),
                "correct_coordinate_rejected": sum(r["fine_hit"] and not r["gates"][method]["allow"] for r in pos),
                "wrong_coordinate_allowed": sum(not r["fine_hit"] and r["gates"][method]["allow"] for r in pos),
                "local_WAIT_but_gate_click_positive": sum(r["local_status"] == "fallback_wait" and r["gates"][method]["allow"] for r in pos),
                "local_WAIT_but_gate_click_negative": sum(r["local_status"] == "fallback_wait" and r["gates"][method]["allow"] for r in neg),
                "global_CLICK_but_gate_reject_positive": sum(r["global_word"] == "click" and not r["gates"][method]["allow"] for r in pos),
                "candidate_correctness_auroc_positive_only": float(loc_auc) if loc_auc is not None else None}
        return summary
    summary = {"scope": "800 completed fixed-cohort cases, 646 geometry positives; descriptive read-only analysis",
               "max_gate_probability_replay_error": max_replay_error,
               "no_inference_fit_or_test_threshold_selection": True, "baseline_prediction_sources": sources,
               "pooled": stage_summary(records),
               "datasets": {name: stage_summary([r for r in records if r["dataset"] == name]) for name in sorted({r["dataset"] for r in records})},
               "feature_shift": shift,
               "same_screenshot_disjoint_target_pairs": {"n": len(pairs),
                   "same_coarse_patch": sum(p["same_coarse_patch"] for p in pairs),
                   "same_crop": sum(p["same_crop"] for p in pairs),
                   "same_fine_point": sum(p["same_fine_point"] for p in pairs),
                   "same_coarse_but_native_full_hits_both": sum(p["same_coarse_patch"] and p["native_full_hits_both"] for p in pairs),
                   "fine_hits_both": sum(p["fine_hits_both"] for p in pairs),
                   "unit_warning": "Pairs share screenshots/cases and are dependent; counts are descriptive, not independent trials or causal isolation."}}
    write_json(OUT / "summary.json", summary)
    write_json(OUT / "same_screenshot_pairs.json", pairs)
    (OUT / "per_sample.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False, allow_nan=False) + "\n" for r in records), encoding="utf-8")
    selectors = {
        "coarse_miss_native_full_correct": lambda r: r["positive"] and not r["crop_contains_target"] and r["full_native_hit"] and r["global_word"] == "click",
        "fine_wrong_with_target_in_crop": lambda r: r["positive"] and r["crop_contains_target"] and r["local_status"] == "refined" and not r["fine_hit"] and r["fine_distance_px"] > 20 and r["full_native_hit"],
        "fine_WAIT_with_target_in_crop": lambda r: r["positive"] and r["crop_contains_target"] and r["local_status"] == "fallback_wait" and not r["fine_hit"] and r["full_native_hit"],
        "correct_point_global_CLICK_gate_reject": lambda r: r["fine_hit"] and r["global_word"] == "click" and not r["gates"]["prefill_hidden"]["allow"],
        "local_WAIT_but_gate_click_negative": lambda r: not r["positive"] and r["local_status"] == "fallback_wait" and r["gates"]["prefill_attention"]["allow"],
        "local_malformed": lambda r: r["local_status"] == "fallback_malformed"}
    examples = {}
    for name, predicate in selectors.items():
        candidates = sorted((r for r in records if predicate(r)), key=lambda r: r["sample_id"])
        if candidates:
            chosen = candidates[0]
            examples[name] = chosen
            if chosen["positive"]:
                render_example(by_id[chosen["sample_id"]], results[chosen["sample_id"]], OUT / (name + ".png"))
    pair_options = sorted((p for p in pairs if p["same_coarse_patch"] and p["native_full_hits_both"]), key=lambda p: (p["a"], p["b"]))
    if pair_options:
        chosen = pair_options[0]
        examples["same_screenshot_pair"] = {"pair": chosen, "a": rec_by_id[chosen["a"]], "b": rec_by_id[chosen["b"]]}
        for key in ("a", "b"):
            sid = chosen[key]
            render_example(by_id[sid], results[sid], OUT / ("same_screenshot_pair_" + key + ".png"))
    write_json(OUT / "examples.json", examples)
    print(json.dumps({"pooled": summary["pooled"], "pairs": summary["same_screenshot_disjoint_target_pairs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
