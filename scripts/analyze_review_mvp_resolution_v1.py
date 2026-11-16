"""Official-score retrospective MVP 1M controls after all predictions seal.

No PointerBench labels are opened until both new MVP modes and the complete
ClickAgent 1M run have passed their public prediction SHA gates.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import numpy as np
import run_review_mvp_resolution_v1 as runner
import run_high_pixel_full_inference_v1 as high
import run_pointer_mvp_same_cohort_v1 as old
from clickagent.public_evaluation import point_in_geometry

OUT = runner.OUT / "analysis"
LABELS = ROOT / "runs/paper_scale_ablation_v2/pointerbench/labels.jsonl"
SCORER = ROOT / "runs/paper_scale_ablation_v2/pointerbench/source/pro/eval.py"
REPLICATES = 10_000
BOOT_SEED = 20261114


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(path: Path) -> str:
    return old.sha(path)


def read(path: Path):
    return old.read(path)


def write_once(path: Path, data, *, text: bool = False) -> None:
    content = data if text else json.dumps(data, ensure_ascii=False, indent=2) + "\n"
    if path.exists():
        require(path.read_text(encoding="utf-8") == content,
                "Existing immutable analysis result differs: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def high_pro_gate(p: dict, public: list[dict]) -> list[dict]:
    high_protocol, manifest = high.gate()
    receipt = read(high.OUT / "public_completion.json")
    require(receipt["complete"] is True and receipt["n"] == 2581 and
            receipt["labels_opened"] is False and
            receipt["protocol_sha256"] == p["highpixel_protocol_sha256"],
            "ClickAgent 1M global seal drift")
    models = {name: high_protocol["models"][name]["sha256"]
              for name in high.MODEL_NAMES}
    overlap = high.pilot_rows()
    for suite in high.SUITES:
        completion = high.OUT / suite / "completion.json"
        require(receipt["suite_receipt_sha256"][suite] == sha(completion),
                "ClickAgent 1M suite completion SHA drift: " + suite)
        suite_receipt = read(completion)
        path = high.OUT / suite / "predictions.json"
        require(suite_receipt["predictions_sha256"] == sha(path),
                "ClickAgent 1M suite prediction SHA drift: " + suite)
    bundle = read(high.OUT / "pro/predictions.json")
    require(bundle["complete"] is True and bundle["labels_opened"] is False and
            bundle["n"] == runner.N and len(bundle["rows"]) == runner.N and
            len(bundle["case_hashes"]) == runner.N,
            "ClickAgent 1M Pro500 prediction bundle incomplete")
    pro_manifest = manifest["cases"]["pro"]
    require([x["sample_id"] for x in pro_manifest] ==
            [x["sample_id"] for x in public],
            "ClickAgent 1M and MVP Pro500 cohort differs")
    for source, row in zip(pro_manifest, bundle["rows"]):
        high.validate_case(row, source, models, overlap)
        path = high.OUT / "pro/cases" / (high.key(source["sample_id"]) + ".json")
        require(bundle["case_hashes"][source["sample_id"]] == sha(path) and
                read(path) == row, "ClickAgent 1M per-case SHA drift")
    return bundle["rows"]


def gate() -> tuple[dict, list[dict], dict[str, list[dict]], list[dict], list[dict]]:
    p, public = runner.gate()
    require(p["source_hashes"]["scripts/analyze_review_mvp_resolution_v1.py"] ==
            sha(Path(__file__)), "Analysis script SHA drift")
    completion = runner.verify_gate()  # no label access; seals both new methods
    require(completion["complete"] is True and completion["labels_opened"] is False and
            completion["prediction_sha256"] ==
            {method: sha(runner.OUT / "predictions" / (method + ".json"))
             for method in runner.METHODS},
            "MVP 1M completion SHA drift")
    new = {method: read(runner.OUT / "predictions" / (method + ".json"))["rows"]
           for method in runner.METHODS}
    high_rows = high_pro_gate(p, public)
    require(p["prior_mvp_completion_sha256"] == sha(old.COMPLETION),
            "Prior local MVP completion SHA drift")
    old.verify_bundle("mvp")
    previous = [r for r in read(old.PRED / "mvp.json")["rows"] if r["suite"] == "pro"]
    require(len(previous) == runner.N and
            [r["sample_id"] for r in previous] == [r["sample_id"] for r in public],
            "Prior 262k MVP Pro500 predictions differ")
    return p, public, new, high_rows, previous


def official_scorer():
    spec = importlib.util.spec_from_file_location("pointerbench_pro_eval_review_mvp1m", SCORER)
    require(spec is not None and spec.loader is not None,
            "Pinned official scorer unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(callable(getattr(module, "_in_bbox", None)), "Official scorer missing")
    return module._in_bbox


def hit(point, label, scorer, sample_id: str) -> float:
    native = bool(point is not None and scorer(tuple(point), label["bbox_pixels"]))
    normalized = bool(point is not None and point_in_geometry(
        [point[0] / 1024, point[1] / 768], label["target_geometry"]))
    require(native == normalized, "Official/V2 scorer parity mismatch: " + sample_id)
    return float(native)


def paired(left: np.ndarray, right: np.ndarray, public: list[dict], draws) -> dict:
    diff = left - right
    clusters: dict[str, list[int]] = defaultdict(list)
    for i, row in enumerate(public):
        clusters[row["pixel_sha256"]].append(i)
    groups = list(clusters.values())
    numerator = np.asarray([diff[ix].sum() for ix in groups], dtype=np.float64)
    denominator = np.asarray([len(ix) for ix in groups], dtype=np.float64)
    sampled = (draws @ numerator) / (draws @ denominator)
    lo, hi = np.quantile(sampled, [0.025, 0.975])
    return {"delta_pp": float(100 * diff.mean()),
            "ci95_pp": [float(100 * lo), float(100 * hi)],
            "wins": int(np.sum(left > right)),
            "losses": int(np.sum(left < right)),
            "ties": int(np.sum(left == right)),
            "rgb_clusters": len(groups)}


def cost(rows: list[dict], highpixel: bool = False) -> dict:
    if highpixel:
        timing = [row["measured"]["elapsed_seconds"] for row in rows]
        calls = [row["measured"]["counter"]["backbone_forward_calls"] for row in rows]
        pixels = [row["measured"]["counter"]["processed_pixels"] for row in rows]
        peaks = [row["measured"]["peak_allocated_bytes"] for row in rows]
    else:
        timing = [row["elapsed_seconds"] for row in rows]
        calls = [row["backbone_forward_calls"] for row in rows]
        pixels = [row["processed_pixels"] for row in rows]
        peaks = [row["peak_allocated_bytes"] for row in rows]
    return {"mean_seconds": float(np.mean(timing)),
            "median_seconds": float(np.median(timing)),
            "mean_forward_calls": float(np.mean(calls)),
            "mean_processed_pixels": float(np.mean(pixels)),
            "maximum_peak_allocated_bytes_per_gpu":
                [int(max(p[device] for p in peaks)) for device in range(len(peaks[0]))]}


def crop_diagnostics(rows: list[dict], expected: tuple[int, int]) -> dict:
    sizes = Counter()
    region_pairs = Counter()
    for row in rows:
        boxes = row["crop_boxes"]
        for x0, y0, x1, y1 in boxes:
            sizes[f"{x1 - x0}x{y1 - y0}"] += 1
        region_pairs[len({tuple(x) for x in boxes})] += 1
    expected_hits = sizes[f"{expected[0]}x{expected[1]}"]
    return {"actual_crop_sizes": dict(sizes),
            "number_distinct_subcrop_boxes_per_case": dict(region_pairs),
            "expected_crop_size": list(expected),
            "expected_size_crop_count": expected_hits,
            "total_crop_count": sum(sizes.values()),
            "failed_outputs": dict(Counter(str(row["output_failure"]) for row in rows))}


def analyze() -> dict:
    p, public, new, high_rows, previous = gate()
    require(sha(LABELS) == p["labels_expected_sha256"], "PointerBench labels SHA drift")
    labels = [json.loads(line) for line in LABELS.read_text(encoding="utf-8-sig").splitlines()
              if line.strip()]
    by_label = {row["sample_id"]: row for row in labels}
    require(len(by_label) == 1000 and all(row["sample_id"] in by_label for row in public),
            "PointerBench label denominator drift")
    scorer = official_scorer()
    vectors: dict[str, np.ndarray] = {}
    for method in runner.METHODS:
        vectors["mvp_1m_" + method] = np.asarray([
            hit(record["point"], by_label[row["sample_id"]], scorer, row["sample_id"])
            for row, record in zip(public, new[method])], dtype=np.float64)
    vectors["mvp_262k_fixed"] = np.asarray([
        hit(record["point"], by_label[row["sample_id"]], scorer, row["sample_id"])
        for row, record in zip(public, previous)], dtype=np.float64)
    for seed in (1, 2, 3):
        vectors[f"ours_1m_s{seed}"] = np.asarray([
            hit(record["seed_points"][seed - 1], by_label[row["sample_id"]],
                scorer, row["sample_id"])
            for row, record in zip(public, high_rows)], dtype=np.float64)
    vectors["ours_1m_mean"] = np.mean([vectors[f"ours_1m_s{seed}"]
                                       for seed in (1, 2, 3)], axis=0)
    clusters = len({row["pixel_sha256"] for row in public})
    rng = np.random.default_rng(BOOT_SEED)
    draws = rng.multinomial(clusters, np.full(clusters, 1 / clusters),
                            size=REPLICATES).astype(np.float32)
    pairs = [("ours_1m_mean", "mvp_1m_fixed"),
             ("ours_1m_mean", "mvp_1m_adaptive"),
             ("mvp_1m_adaptive", "mvp_1m_fixed"),
             ("mvp_1m_fixed", "mvp_262k_fixed")]
    contrasts = {f"{left}_minus_{right}": paired(vectors[left], vectors[right],
                                                public, draws)
                 for left, right in pairs}
    for seed in (1, 2, 3):
        for baseline in ("mvp_1m_fixed", "mvp_1m_adaptive"):
            left = f"ours_1m_s{seed}"
            contrasts[f"{left}_minus_{baseline}"] = paired(vectors[left],
                                                              vectors[baseline], public, draws)
    result = {
        "complete": True, "n": runner.N, "rgb_clusters": clusters,
        "cohort": "PointerBench Pro500",
        "metric": "Pinned official inclusive native point-in-bbox; every point checked against V2 normalized geometry scorer",
        "official_scorer_sha256": sha(SCORER),
        "point_scorer_parity_count": runner.N * (len(runner.METHODS) + 1 + 3),
        "hits": {name: int(vector.sum()) if name != "ours_1m_mean" else None
                 for name, vector in vectors.items()},
        "accuracy_percent": {name: float(100 * vector.mean())
                             for name, vector in vectors.items()},
        "contrasts": contrasts,
        "runtime": {"mvp_1m_fixed": cost(new["fixed"]),
                    "mvp_1m_adaptive": cost(new["adaptive"]),
                    "mvp_262k_fixed": cost(previous),
                    "ours_1m": cost(high_rows, highpixel=True)},
        "crop_diagnostics": {
            "mvp_1m_fixed": crop_diagnostics(new["fixed"], (1024, 728)),
            "mvp_1m_adaptive": crop_diagnostics(new["adaptive"], (687, 518))},
        "protocol_sha256": sha(runner.PROTOCOL),
        "new_public_completion_sha256": sha(runner.OUT / "public_completion.json"),
        "ours_1m_public_completion_sha256": sha(high.OUT / "public_completion.json"),
        "prior_262k_mvp_completion_sha256": sha(old.COMPLETION),
        "labels_sha256": sha(LABELS),
        "retrospective": True, "strict_blind": False,
        "latency_protocol_matched_to_ours": False,
        "latency_boundary": "MVP per-case timer excludes PNG decode/image load; ClickAgent 1M pilot.one_call timer includes PNG decode. Raw seconds are descriptive, not a matched speed comparison.",
    }
    write_once(OUT / "results.json", result)
    lines = ["MVP 1M resolution and adaptive-crop controls: PointerBench Pro500",
             "Retrospective same-cohort comparison; official native inclusive point-in-bbox.",
             "All 500 fixed and 500 adaptive predictions and all 2581 ClickAgent 1M predictions passed SHA seals before this program read labels.",
             "Adaptive crop 687x518 derived geometrically from official 1288x728 and median 1920x1080 on an independent validation reserve; no Pro labels used to choose crop size.",
             "Equal per-view max_pixels=1048576 does not equalize view count or total compute.", ""]
    for name in ("ours_1m_mean", "ours_1m_s1", "ours_1m_s2", "ours_1m_s3",
                 "mvp_1m_fixed", "mvp_1m_adaptive", "mvp_262k_fixed"):
        lines.append(f"{name}: {result['accuracy_percent'][name]:.2f}%")
    lines.append("")
    for left, right in pairs:
        c = contrasts[f"{left}_minus_{right}"]
        lines.append(f"{left} - {right}: {c['delta_pp']:+.2f} pp, 95% RGB-cluster paired CI [{c['ci95_pp'][0]:+.2f}, {c['ci95_pp'][1]:+.2f}], wins/losses/ties {c['wins']}/{c['losses']}/{c['ties']}")
    lines.append("")
    for name, values in result["runtime"].items():
        lines.append(f"{name} observed cost: mean/median {values['mean_seconds']:.2f}/{values['median_seconds']:.2f}s; mean model forwards {values['mean_forward_calls']:.2f}; mean processed pixels {values['mean_processed_pixels']:.0f}")
    lines.extend(["", "Cost is descriptive: MVP timing excludes PNG decode/image load while ClickAgent 1M timing includes PNG decode; warmup counts also differ. No matched-latency superiority claim.",
                  "V2 labels/results had been analyzed before this comparison; at least seven earlier PointerBench answers were visible. Adaptive crop is a declared variant, not official publisher MVP."])
    write_once(OUT / "readout.txt", "\n".join(lines) + "\n", text=True)
    completion = {"complete": True, "result_sha256": sha(OUT / "results.json"),
                  "readout_sha256": sha(OUT / "readout.txt"),
                  "protocol_sha256": sha(runner.PROTOCOL),
                  "labels_sha256": sha(LABELS)}
    write_once(OUT / "completion.json", completion)
    return completion


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("verify-gate", "analyze", "self-test"))
    args = parser.parse_args()
    if args.stage == "self-test":
        assert paired(np.array([1., 0.]), np.array([0., 0.]),
                      [{"pixel_sha256": "a"}, {"pixel_sha256": "b"}],
                      np.array([[1., 1.], [2., 0.]]))["wins"] == 1
        print(json.dumps({"self_test": "passed", "labels_opened": False}))
    elif args.stage == "verify-gate":
        p, public, new, high_rows, previous = gate()
        print(json.dumps({"gate": "passed", "n": len(public),
                          "mvp_methods": list(new), "highpixel_pro": len(high_rows),
                          "prior_mvp_pro": len(previous), "labels_opened": False}))
    else:
        print(json.dumps(analyze(), ensure_ascii=False))


if __name__ == "__main__":
    main()
