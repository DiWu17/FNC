"""Read GUI labels only after all cached view-prefix predictions are SHA sealed."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import run_review_view_budget_cached_v1 as worker
from clickagent.public_evaluation import point_in_geometry
from run_mmbench_gui_l2_v1 import official_hit, source_reference_hit

OUT = worker.OUT
REG = OUT / "analysis_registration.json"
W3_REG = ROOT / "runs/review_controls_v1/w3/analysis_registration_v3.json"
POINTER_LABELS = ROOT / "runs/paper_scale_ablation_v2/pointerbench/labels.jsonl"
MM_LABELS = ROOT / "data/mmbench_gui_l2_v1/source/L2_annotations.json"
MM_SCORER = ROOT / "runs/mmbench_gui_new_source_v1/official_l2_evaluator.py"
POINTER_SCORER = {s: ROOT / f"runs/paper_scale_ablation_v2/pointerbench/source/{s}/eval.py"
                  for s in ("pro", "sheets")}
REPS = 10000
BOOT_SEED = 20261115


def register():
    p = worker.verify_protocol()
    worker.require(not worker.COMPLETE.exists(), "Analysis registration must precede prediction seal")
    old = worker.read(W3_REG)
    value = {
        "registered_before_prediction_seal": True,
        "analysis_script_sha256": worker.sha(__file__),
        "worker_protocol_sha256": worker.sha(worker.PROTOCOL),
        "label_sha256_unopened": old["label_sha256_unopened"],
        "source_label_hash_registration": str(W3_REG.relative_to(ROOT)),
        "pointer_official_scorer_sha256": {s: worker.sha(path) for s, path in POINTER_SCORER.items()},
        "mmbench_official_scorer_sha256": worker.sha(MM_SCORER),
        "prediction_gate": "All 2000 cases, 54 methods per case, SHA receipts and original m3 parity before any label-file hash or parse",
        "primary": "Pro500 View Budget Progression, m0 Top8 under frozen source quota; m0 Top16 sensitivity",
        "secondary": "Sheets500 and selected MMBench1000",
        "metric": "Original 262144-per-view cached source, K16 forced stable Top1, inclusive native point-in-target geometry",
        "statistics": "Three fixed seed hits/mean; oracle coverage; paired screenshot-cluster bootstrap 10000 draws, fixed seed 20261115, 95% descriptive intervals",
        "mvp_scope": "MVP only has Pro500 and Sheets500 in archived same-cohort result; per-query distinct view count absent; report maximum 3 and measured forwards/pixels, no guessed actual mean views",
        "label_access_at_registration": False,
        "retrospective_not_strict_blind": True,
    }
    worker.write_once(REG, value)
    print(json.dumps({"stage": "analysis_registered", "sha256": worker.sha(REG)}), flush=True)


def gate():
    worker.require(REG.exists(), "Register analysis before prediction")
    reg = worker.read(REG)
    worker.require(reg["analysis_script_sha256"] == worker.sha(__file__) and
                   reg["worker_protocol_sha256"] == worker.sha(worker.PROTOCOL),
                   "Analysis script/protocol drift")
    worker.verify_gate()
    for s, path in POINTER_SCORER.items():
        worker.require(reg["pointer_official_scorer_sha256"][s] == worker.sha(path), "Pointer scorer drift")
    worker.require(reg["mmbench_official_scorer_sha256"] == worker.sha(MM_SCORER), "MM scorer drift")
    return reg


def labels_after_gate(reg):
    expected = reg["label_sha256_unopened"]
    worker.require(worker.sha(POINTER_LABELS) == expected["pointer"] and
                   worker.sha(MM_LABELS) == expected["mmbench"], "Label source SHA drift")
    pointer_rows = [json.loads(line) for line in POINTER_LABELS.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
    worker.require(len(pointer_rows) == 1000, "Pointer label count drift")
    pointer = {r["sample_id"]: r for r in pointer_rows}
    mm_rows = worker.read(MM_LABELS)
    worker.require(len(mm_rows) == 3594, "MM label count drift")
    mm = {r["index"]: r for r in mm_rows}
    return pointer, mm


def native_scorer(suite):
    if suite == "mmbench":
        return source_reference_hit()
    spec = importlib.util.spec_from_file_location("pointer_view_budget_" + suite, POINTER_SCORER[suite])
    worker.require(spec is not None and spec.loader is not None, "Pointer official scorer unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._in_bbox


def score_point(suite, point, row, pointer, mm, official):
    if point is None:
        return False
    if suite == "mmbench":
        source = mm[row["source_index"]]
        worker.require(all(source[f] == row[f] for f in
                           ("instruction", "image_size", "data_type", "platform", "grounding_type")),
                       "MM public/annotation identity drift")
        local = official_hit(point, source["bbox"], row["image_size"])
        reference = bool(official(source["bbox"].copy(), tuple(row["image_size"]), tuple(point)))
    else:
        label = pointer[row["sample_id"]]
        w, h = row["image_size"]
        local = bool(point_in_geometry((point[0] / w, point[1] / h), label["target_geometry"]))
        reference = bool(official(tuple(point), label["bbox_pixels"]))
    worker.require(bool(local) == reference, "Official scorer parity failure: " + suite + "/" + row["sample_id"])
    return bool(local)


def bootstrap_delta(left, right, clusters):
    rng = np.random.default_rng(BOOT_SEED)
    group = list(clusters.values())
    draw = rng.multinomial(len(group), np.full(len(group), 1 / len(group)), size=REPS).astype(np.float32)
    den = draw @ np.asarray([len(ix) for ix in group], np.float32)
    difference = np.mean(left - right, axis=0)
    numer = np.asarray([difference[ix].sum() for ix in group], np.float32)
    sampled = draw @ numer / den
    return {"delta_pp": float(difference.mean() * 100),
            "ci95_pp": (np.quantile(sampled, [.025, .975]) * 100).astype(float).tolist(),
            "rgb_clusters": len(group), "bootstrap_replicates": REPS}


def summarize_suite(suite, rows, pointer, mm):
    scorer = native_scorer(suite)
    names = [f"{pool}_{method}" for pool in
             ("A_m0", "B_m0_top8", "B_m0_top16", "A_m1", "B_m1", "A_m2", "B_m2", "A_m3", "B_m3")
             for method in worker.METHODS]
    hits = {name: np.zeros(len(rows), np.float32) for name in names}
    pools = ("A_m0", "B_m0_top8", "B_m0_top16", "A_m1", "B_m1", "A_m2", "B_m2", "A_m3", "B_m3")
    coverage = {name: 0 for name in pools}
    pool_size = {name: [] for name in pools}
    views = []; forwards = []
    parity_points = 0
    clusters = defaultdict(list)
    for i, row in enumerate(rows):
        sid = row["sample_id"]
        key = hashlib.sha256(sid.encode()).hexdigest()[:24]
        pred = worker.read(OUT / "predictions" / suite / "cases" / f"{key}.json")
        worker.require(pred["sample_id"] == sid, "Sealed/public order drift")
        clusters[row["pixel_sha256"]].append(i)
        views.append(pred["actual_view_count"])
        forwards.append(pred["per_view_backbone_forwards_reconstructed"])
        for pool in pools:
            points = pred["pool_points"][pool]
            outcomes = [score_point(suite, point, row, pointer, mm, scorer) for point in points]
            parity_points += len(points)
            coverage[pool] += int(any(outcomes))
            pool_size[pool].append(len(points))
        for name in names:
            point = pred["predictions"][name]["point"]
            hits[name][i] = score_point(suite, point, row, pointer, mm, scorer)
            parity_points += int(point is not None)
    scores = {}
    for pool in pools:
        scores[pool] = {"coverage_hits": coverage[pool], "coverage_pct": coverage[pool] / len(rows) * 100,
                        "mean_pool_size": float(np.mean(pool_size[pool]))}
        for kind in ("aligned", "no_evidence"):
            values = np.stack([hits[f"{pool}_{kind}_s{s}"] for s in (1, 2, 3)])
            scores[pool][kind] = {"seed_hits": values.sum(axis=1).astype(int).tolist(),
                                   "mean_pct": float(values.mean() * 100)}
    def triple(pool, kind="aligned"):
        return np.stack([hits[f"{pool}_{kind}_s{s}"] for s in (1, 2, 3)])
    paired = {
        "A_m3_minus_A_m0": bootstrap_delta(triple("A_m3"), triple("A_m0"), clusters),
        "B_m3_minus_B_m0top8": bootstrap_delta(triple("B_m3"), triple("B_m0_top8"), clusters),
        "B_m3_minus_B_m0top16": bootstrap_delta(triple("B_m3"), triple("B_m0_top16"), clusters),
    }
    vp = {str(m): float(np.mean([min(v, m + 1) for v in views])) for m in worker.VIEWS}
    fp = {str(m): float(np.mean([sum(v[:m + 1]) for v in forwards])) for m in worker.VIEWS}
    return {"n": len(rows), "scores": scores, "paired": paired, "actual_archived_view_mean": float(np.mean(views)),
            "prefix_view_mean": vp, "prefix_backbone_forwards_reconstructed_mean": fp,
            "actual_view_histogram": {str(k): views.count(k) for k in sorted(set(views))},
            "parity_nonnull_points": parity_points, "official_scorer_parity_passed": True,
            "rgb_clusters": len(clusters)}


def mvp_stats(pointer, mm):
    result = worker.read(worker.MVP)
    worker.require(result["complete"] and result["n"] == 1000, "MVP prediction bundle drift")
    stats = {}
    for suite in ("pro", "sheets"):
        rows = [r for r in result["rows"] if r["suite"] == suite]
        worker.require(len(rows) == 500, "MVP suite denominator drift")
        public = {r["sample_id"]: r for r in worker.read(worker.PUBLIC[suite])}
        official = native_scorer(suite)
        worker.require(set(public) == {r["sample_id"] for r in rows}, "MVP/public cohort identity drift")
        correct = sum(score_point(suite, r["point"], public[r["sample_id"]],
                                  pointer, mm, official) for r in rows)
        stats[suite] = {"n": 500, "mean_backbone_forward_calls_logged": float(np.mean([r["backbone_forward_calls"] for r in rows])),
                        "mean_processed_pixels_logged": float(np.mean([r["processed_pixels"] for r in rows])),
                        "official_parity_correct": correct, "accuracy_pct": correct / 5,
                        "unique_processed_pixels": sorted({r["processed_pixels"] for r in rows}),
                        "max_distinct_views_by_protocol": 3,
                        "actual_distinct_views_per_query": None,
                        "reason_actual_views_unavailable": "Official reference_result.inference_count was discarded from sealed per-case output"}
    return stats


def analyze():
    reg = gate()
    pointer, mm = labels_after_gate(reg)
    results = {}
    for suite in worker.PUBLIC:
        rows = worker.read(worker.PUBLIC[suite])
        results[suite] = summarize_suite(suite, rows, pointer, mm)
    result = {"complete": True, "registration_sha256": worker.sha(REG),
              "prediction_completion_sha256": worker.sha(worker.COMPLETE),
              "cohorts": results, "mvp_archived": mvp_stats(pointer, mm),
              "limitations": ["Frozen full-view-trained scorers; no view-specific retraining",
                              "A keeps candidate coordinates from later views, so is not single-view end-to-end",
                              "B only reselects archived per-view proposals, not a newly executed truncated pipeline",
                              "MVP actual distinct view counts were not retained per query",
                              "Pointer labels previously analyzed; MMB1000 is fixed selected subset"]}
    output = OUT / "analysis/summary.json"
    worker.write_once(output, result)
    lines = ["Cached view-prefix diagnostic; original 262144 px/view; frozen full readouts; no new Qwen or candidate coordinates.",
             "A: archived K16 fixed, later evidence/base masked. B: prefix-only archived proposals; m0 Top8 primary, Top16 sensitivity.",
             "All 2000 cases/54 predictions each SHA sealed before this analysis opened labels."]
    for suite, item in results.items():
        lines += ["", f"{suite.upper()} n={item['n']} RGB clusters={item['rgb_clusters']}",
                  "Setting | oracle | aligned S1/S2/S3 | aligned mean | no evidence S1/S2/S3 | no evidence mean"]
        for pool, scores in item["scores"].items():
            a, n = scores["aligned"], scores["no_evidence"]
            lines.append(f"{pool} | {scores['coverage_hits']}/{item['n']} ({scores['coverage_pct']:.2f}%) | "
                         f"{a['seed_hits']} | {a['mean_pct']:.2f}% | {n['seed_hits']} | {n['mean_pct']:.2f}%")
        lines += ["mean actual archived views=" + f"{item['actual_archived_view_mean']:.3f}",
                  "mean prefix views=" + str(item["prefix_view_mean"]),
                  "mean reconstructed prefix backbone forwards=" + str(item["prefix_backbone_forwards_reconstructed_mean"])]
        for name, value in item["paired"].items():
            lines.append(f"{name}: {value['delta_pp']:+.2f} pp, 95% CI {value['ci95_pp']}")
    lines += ["", "MVP same-cohort archived resource counters (Pro/Sheets only): " + str(result["mvp_archived"]),
              "MVP distinct per-query view count missing from sealed archive; 3 is the configured maximum, not a measured mean.",
              *result["limitations"], ""]
    text = "\n".join(lines)
    out_text = OUT / "analysis/readout.txt"
    if out_text.exists():
        worker.require(out_text.read_text(encoding="utf-8") == text, "Analysis text drift")
    else:
        out_text.parent.mkdir(parents=True, exist_ok=True)
        out_text.write_text(text, encoding="utf-8")
    worker.write_once(OUT / "analysis/completion.json", {"complete": True, "summary_sha256": worker.sha(output),
                                                       "readout_sha256": worker.sha(out_text),
                                                       "prediction_completion_sha256": worker.sha(worker.COMPLETE)})
    print(json.dumps({"stage": "analyzed", "summary_sha256": worker.sha(output)}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("register", "verify-gate", "analyze"))
    args = parser.parse_args()
    worker.low_priority()
    if args.stage == "register": register()
    elif args.stage == "verify-gate":
        gate()
        print(json.dumps({"stage": "verify_gate_pass", "labels_opened": False}), flush=True)
    else: analyze()


if __name__ == "__main__":
    main()
