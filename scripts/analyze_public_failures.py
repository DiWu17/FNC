"""Read-only post-hoc analysis of public_v1 caches; no training or inference.

Ground-truth-dependent top-k/head oracles and expanded regions are diagnostics,
never deployable policies or replacements for official point-in-region scores.
Run from the repository root with the project's Python environment.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from clickagent.action_grounding import AttentionPointGrounder, _pixel_centers, patch_features
from clickagent.public_evaluation import point_in_geometry


ROOT = Path(__file__).resolve().parents[1]
RUN_NAMES = ("screenspot_v2", "screenspot_pro", "osworld_g", "venusbench_gd", "conflictgui")


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def distance_to_region(point, geometry, size):
    """Euclidean pixels to the nearest allowed region, zero inside/on boundary."""
    w, h = size
    if point_in_geometry((point[0] / w, point[1] / h), geometry):
        return 0.0
    p = np.asarray(point, dtype=float)
    if geometry["type"] == "bbox":
        boxes = np.asarray(geometry["boxes"]) * [w, h, w, h]
        nearest = np.minimum(np.maximum(p, boxes[:, :2]), boxes[:, 2:])
        return float(np.linalg.norm(nearest - p, axis=1).min())
    distances = []
    for polygon in geometry["polygons"]:
        vertices = np.asarray(polygon) * [w, h]
        ends = np.roll(vertices, -1, axis=0)
        vectors = ends - vertices
        den = np.sum(vectors * vectors, axis=1)
        t = np.divide(np.sum((p - vertices) * vectors, axis=1), den,
                      out=np.zeros_like(den), where=den > 0).clip(0, 1)
        distances.extend(np.linalg.norm(vertices + t[:, None] * vectors - p, axis=1))
    return float(min(distances))


def polygon_rect_area(polygon, rect):
    vertices = [np.asarray(p, dtype=float) for p in polygon]
    for axis, boundary, sign in ((0, rect[0], 1), (0, rect[2], -1),
                                 (1, rect[1], 1), (1, rect[3], -1)):
        previous, vertices = vertices, []
        if not previous:
            break
        for a, b in zip(previous, previous[1:] + previous[:1]):
            ain, bin_ = sign * (a[axis] - boundary) >= 0, sign * (b[axis] - boundary) >= 0
            if ain:
                vertices.append(a)
            if ain != bin_:
                t = (boundary - a[axis]) / (b[axis] - a[axis])
                vertices.append(a + t * (b - a))
    if len(vertices) < 3:
        return 0.0
    a = np.asarray(vertices)
    return abs(float(np.sum(a[:, 0] * np.roll(a[:, 1], -1) - a[:, 1] * np.roll(a[:, 0], -1)))) / 2


def target_masks(geometry, grid, size):
    gh, gw = grid
    x, y = _pixel_centers(grid, size)
    centers = np.asarray([point_in_geometry((xx / size[0], yy / size[1]), geometry)
                          for xx, yy in zip(x, y)])
    overlap = []
    for i in range(gh * gw):
        row, col = divmod(i, gw)
        rect = (col / gw, row / gh, (col + 1) / gw, (row + 1) / gh)
        if geometry["type"] == "bbox":
            area = any(min(rect[2], box[2]) > max(rect[0], box[0]) and
                       min(rect[3], box[3]) > max(rect[1], box[1]) for box in geometry["boxes"])
        else:
            area = any(polygon_rect_area(poly, rect) > 1e-16 for poly in geometry["polygons"])
        overlap.append(area)
    return centers, np.asarray(overlap)


def distribution(values):
    values = np.asarray(list(values), dtype=float)
    return {"n": len(values), **({"median": float(np.median(values)), "mean": float(values.mean()),
                                 "p25": float(np.percentile(values, 25)), "p75": float(np.percentile(values, 75))}
                                if len(values) else {})}


def aggregate(records):
    hits = [r for r in records if r["hit"]]
    misses = [r for r in records if not r["hit"]]
    reachable = [r for r in records if r["center_oracle"]]
    counts = Counter(r["failure_stage"] for r in records)
    result = {"n": len(records), "hits": len(hits), "misses": len(misses),
              "decomposition": dict(counts), "center_reachable": len(reachable),
              "chosen_cell_overlaps_target": sum(r["chosen_cell_overlaps_target"] for r in records),
              "misses_with_target_in_chosen_cell": sum(r["chosen_cell_overlaps_target"] for r in misses),
              "miss_distance_px": distribution(r["distance_px"] for r in misses),
              "miss_distance_fraction_short_screen": distribution(r["distance_fraction_short_screen"] for r in misses),
              "target_short_edge_px": distribution(r["target_short_edge_px"] for r in records),
              "diagnostic_expanded_region_hits": {str(px): sum(r["distance_px"] <= px for r in records)
                                                    for px in (0, 5, 10, 20)},
              "diagnostic_within_half_percent_short_screen": sum(r["distance_fraction_short_screen"] <= .005 for r in records),
              "center_topk_oracle_hits": {str(k): sum(r[f"center_top{k}"] for r in records) for k in (1, 5, 10)},
              "cell_overlap_topk_oracle_hits": {str(k): sum(r[f"overlap_top{k}"] for r in records) for k in (1, 5, 10)},
              "raw_mean_attention_center_hits": sum(r["raw_mean_center_hit"] for r in records),
              "raw_mean_attention_cell_overlap": sum(r["raw_mean_cell_overlap"] for r in records),
              "any_individual_head_center_oracle_hits": sum(r["individual_head_center_hits"] > 0 for r in records),
              "any_individual_head_cell_oracle_hits": sum(r["individual_head_cell_hits"] > 0 for r in records)}
    result["cross_quantization_and_selection"] = {
        "unreachable_and_selected_cell_overlaps": sum(not r["center_oracle"] and r["chosen_cell_overlaps_target"] for r in records),
        "unreachable_and_selected_cell_does_not_overlap": sum(not r["center_oracle"] and not r["chosen_cell_overlaps_target"] for r in records),
        "reachable_miss_and_selected_cell_overlaps": sum(r["center_oracle"] and not r["hit"] and r["chosen_cell_overlaps_target"] for r in records),
        "reachable_miss_and_selected_cell_does_not_overlap": sum(r["center_oracle"] and not r["hit"] and not r["chosen_cell_overlaps_target"] for r in records)}
    result["localization_confidence_diagnostics"] = {
        "concentration_negative_entropy_auroc": float(roc_auc_score([r["hit"] for r in records], [-r["mean_normalized_entropy"] for r in records])) if hits and misses else None,
        "point_head_score_auroc": float(roc_auc_score([r["hit"] for r in records], [r["point_head_score"] for r in records])) if hits and misses else None,
        "misses_with_point_head_score_at_least_0_9": sum(r["point_head_score"] >= .9 for r in misses),
        "note": "Descriptive existing-score diagnostics, no test-set threshold optimization; point head scores are not calibrated action confidence."}
    native = [r for r in records if r.get("native_lax_point") is not None]
    result["native_lax_coordinate_diagnostic"] = {
        "point_available": len(native), "denominator_all_geometry_positives": len(records),
        "hits": sum(r["native_lax_distance_px"] == 0 for r in native),
        "miss_distance_px_given_point": distribution(r["native_lax_distance_px"] for r in native if r["native_lax_distance_px"] > 0),
        "expanded_region_hits": {str(px): sum(r["native_lax_distance_px"] <= px for r in native) for px in (0, 5, 10, 20)},
        "note": "Missing/refused/malformed native actions remain in the fixed denominator; distance distribution itself is conditional on an available point."}
    result["signal_by_outcome"] = {}
    for name, rows in (("hit", hits), ("miss", misses),
                       ("reachable_miss", [r for r in reachable if not r["hit"]])):
        result["signal_by_outcome"][name] = {key: distribution(r[key] for r in rows) for key in (
            "mean_image_attention_mass", "mean_normalized_entropy", "target_cell_enrichment",
            "point_head_score", "individual_head_cell_hits")}
    result["gates"] = {}
    for method in (records[0]["gates"] if records else []):
        result["gates"][method] = dict(Counter(
            ("hit" if r["hit"] else "miss") + ("_allow" if r["gates"][method]["allow"] else "_reject")
            for r in records))
    return result


def analyze_run(name, grounder):
    run = ROOT / "runs/public_v1" / name
    cases = read_jsonl(run / "selected_manifest.jsonl")
    predictions = read_jsonl(run / "predictions.jsonl")
    by_id = defaultdict(dict)
    for row in predictions:
        assert row["method"] not in by_id[row["sample_id"]]
        by_id[row["sample_id"]][row["method"]] = row
    records = []
    for case in cases:
        if not case["expected_actionable"] or case["target_geometry"] is None:
            continue
        sid = case["sample_id"]
        stem = hashlib.sha256(sid.encode()).hexdigest()[:24]
        rows = by_id[sid]
        base = rows["prefill_hidden"]
        grid, size = base["grid_shape"], base["model_image_size"]
        point = base["localization_point"]
        for method, row in rows.items():
            if method != "text_coordinates":
                assert row["localization_point"] == point, (sid, method)
        with np.load(run / "features" / (stem + ".npz"), allow_pickle=False) as archive:
            raw = archive["prefill_raw_map"].astype(float)
        # Replay the unchanged CPU readout; verify against cached actual output.
        proposal = grounder.predict_point(raw, grid, size)
        assert proposal["point"] == point
        probabilities = grounder.pipeline.predict_proba(patch_features(raw, grid))[:, list(grounder.pipeline.classes_).index(1)]
        order = np.argsort(-probabilities, kind="stable")
        patch = int(order[0])
        centers, overlap = target_masks(case["target_geometry"], grid, size)
        assert centers[patch] == point_in_geometry((point[0] / size[0], point[1] / size[1]), case["target_geometry"])
        mass = raw.sum(axis=1)
        spatial = np.divide(raw, mass[:, None], out=np.zeros_like(raw), where=mass[:, None] > 0)
        entropy = -(spatial * np.log(np.maximum(spatial, 1e-300))).sum(axis=1) / math.log(raw.shape[1])
        head_peaks = np.argmax(raw, axis=1)
        raw_mean_peak = int(np.argmax(raw.mean(axis=0)))
        if case["target_geometry"]["type"] == "bbox":
            verts = [np.asarray([b[:2], b[2:]]) * size for b in case["target_geometry"]["boxes"]]
        else:
            verts = [np.asarray(p) * size for p in case["target_geometry"]["polygons"]]
        short_edge = max(float(np.ptp(v, axis=0).min()) for v in verts)
        hit, reachable = bool(centers[patch]), bool(centers.any())
        distance = distance_to_region(point, case["target_geometry"], size)
        record = {"sample_id": sid, "dataset": case["dataset"], "run_name": name,
                  "stem": stem, "image": case["image"], "instruction": case["instruction"],
                  "geometry": case["target_geometry"], "image_size": size, "grid": grid,
                  "domain": case["domain"], "data_type": case["metadata"].get("data_type", "unknown"),
                  "point": point, "patch_index": patch, "point_head_score": proposal["score"],
                  "hit": hit, "center_oracle": reachable, "chosen_cell_overlaps_target": bool(overlap[patch]),
                  "failure_stage": "hit" if hit else "no_center_inside_target" if not reachable else "reachable_but_wrong_patch",
                  "distance_px": distance, "distance_fraction_short_screen": distance / min(size),
                  "target_short_edge_px": short_edge,
                  "mean_image_attention_mass": float(mass.mean()),
                  "mean_normalized_entropy": float(entropy.mean()),
                  "target_cell_enrichment": float(spatial[:, overlap].sum(axis=1).mean() / (overlap.sum() / len(overlap))),
                  "raw_mean_center_hit": bool(centers[raw_mean_peak]),
                  "raw_mean_cell_overlap": bool(overlap[raw_mean_peak]),
                  "individual_head_center_hits": int(centers[head_peaks].sum()),
                  "individual_head_cell_hits": int(overlap[head_peaks].sum()),
                  "best_target_center_rank": next((i + 1 for i, p in enumerate(order) if centers[p]), None),
                  "best_target_cell_rank": next((i + 1 for i, p in enumerate(order) if overlap[p]), None),
                  "gates": {m: {"allow": r["action"]["type"] == "click", "probability": r.get("click_probability"),
                                 "threshold": r.get("gate_threshold"), "valid": r["parse_valid"]}
                            for m, r in rows.items() if m not in ("text_coordinates", "text_decision_shared")}}
        for k in (1, 5, 10):
            record[f"center_top{k}"] = bool(centers[order[:k]].any())
            record[f"overlap_top{k}"] = bool(overlap[order[:k]].any())
        records.append(record)
    return records


def render_example(record, grounder, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon
    from PIL import Image

    with np.load(ROOT / "runs/public_v1" / record["run_name"] / "features" / (record["stem"] + ".npz")) as archive:
        raw = archive["prefill_raw_map"].astype(float)
    w, h = record["image_size"]
    with Image.open(record["image"]) as source:
        image = source.convert("RGB")
        image.thumbnail((1600, 1100))
        pixels = np.asarray(image)
    gh, gw = record["grid"]
    scores = grounder.pipeline.predict_proba(patch_features(raw, record["grid"]))[:, list(grounder.pipeline.classes_).index(1)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    titles = ["Screenshot: green = GT region; red = proposal", "Mean raw image attention (relative colors)",
              "Learned point-head score (uncalibrated)"]
    for i, ax in enumerate(axes):
        ax.imshow(pixels, extent=[0, w, h, 0])
        if i:
            field = raw.mean(axis=0) if i == 1 else scores
            ax.imshow(field.reshape(gh, gw), extent=[0, w, h, 0], interpolation="nearest", cmap="magma", alpha=.64)
        for x in np.arange(gw + 1) * w / gw:
            ax.axvline(x, color="cyan", alpha=.16, lw=.5)
        for y in np.arange(gh + 1) * h / gh:
            ax.axhline(y, color="cyan", alpha=.16, lw=.5)
        geo = record["geometry"]
        if geo["type"] == "bbox":
            for x0, y0, x1, y1 in geo["boxes"]:
                ax.add_patch(Rectangle((x0*w, y0*h), (x1-x0)*w, (y1-y0)*h, fill=False, edgecolor="lime", lw=2))
        else:
            for poly in geo["polygons"]:
                ax.add_patch(Polygon(np.asarray(poly)*[w, h], fill=False, edgecolor="lime", lw=2))
        ax.plot(*record["point"], marker="x", color="red", mew=2, ms=9)
        ax.set_title(titles[i], fontsize=10)
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_axis_off()
    fig.suptitle(f"{record['sample_id']} | {record['instruction']}\n"
                 f"{record['failure_stage']} | distance to GT {record['distance_px']:.1f}px | "
                 f"target overlaps chosen cell: {record['chosen_cell_overlaps_target']} | "
                 f"head score {record['point_head_score']:.6f}", fontsize=11)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "runs/public_v1/failure_analysis")
    parser.add_argument("--from-records", action="store_true", help="Reaggregate existing per-sample diagnostics without replaying the frozen point head")
    args = parser.parse_args()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    grounder_path = ROOT / "runs/action_v3/grounder/grounder.joblib"
    grounder = AttentionPointGrounder.load(grounder_path)
    all_records = []
    datasets = {}
    existing = read_jsonl(output / "per_sample.jsonl") if args.from_records else None
    native_diagnostics = json.loads((ROOT / "runs/public_v1/analysis/diagnostics.json").read_text(encoding="utf-8"))
    native_records = {r["sample_id"]: r for d in native_diagnostics["datasets"].values()
                      for r in d["offline_lax_coordinates"]["records"]}
    for name in RUN_NAMES:
        records = [r for r in existing if r["run_name"] == name] if existing is not None else analyze_run(name, grounder)
        if not records:
            continue
        for record in records:
            native = native_records[record["sample_id"]]
            point = native["offline_pixel_point"] if native["eligible_offline_lax_click"] else None
            record["native_lax_point"] = point
            record["native_lax_distance_px"] = distance_to_region(point, record["geometry"], record["image_size"]) if point is not None else None
            if point is not None:
                assert (record["native_lax_distance_px"] == 0) == native["offline_grounding_hit"]
        all_records.extend(records)
        stats = aggregate(records)
        for field in ("domain", "data_type"):
            stats["by_" + field] = {key: aggregate([r for r in records if r[field] == key])
                                     for key in sorted({r[field] for r in records})}
        datasets[records[0]["dataset"]] = stats
        print(name, json.dumps({k: stats[k] for k in ("n", "hits", "decomposition", "miss_distance_px", "diagnostic_expanded_region_hits")}), flush=True)
    with (output / "per_sample.jsonl").open("w", encoding="utf-8") as stream:
        for record in all_records:
            stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
    summary = {"scope": f"{len(all_records)} selected positive geometry cases; post-hoc frozen-cache analysis",
               "changes_formal_scores": False, "uses_training_or_model_inference": False,
               "grounder_sha256": hashlib.sha256(grounder_path.read_bytes()).hexdigest(),
               "distance_definition": "Euclidean original-image pixels to nearest GT boundary; zero inside region",
               "topk_and_any_head_are_gt_dependent_diagnostic_oracles": True,
               "overlap_note": "Chosen grid cell overlaps GT does not prove accurate point or semantic recognition",
               "entropy_note": "Normalized per-head spatial entropy, averaged; 0=peak, 1=uniform. Relative colors do not show image attention mass.",
               "pooled": aggregate(all_records), "datasets": datasets}
    (output / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    # Deterministic illustrative cases, never used for choosing policies/thresholds.
    selectors = {
        "near_boundary": lambda r: not r["hit"] and r["distance_px"] <= 10 and r["chosen_cell_overlaps_target"],
        "far_wrong_patch": lambda r: r["failure_stage"] == "reachable_but_wrong_patch" and r["distance_fraction_short_screen"] > .15,
        "correct_but_gate_rejected": lambda r: r["hit"] and not r["gates"]["prefill_hidden"]["allow"],
        "small_unreachable": lambda r: r["run_name"] == "screenspot_pro" and not r["center_oracle"] and r["chosen_cell_overlaps_target"],
    }
    examples = {}
    for name, predicate in selectors.items():
        options = [r for r in all_records if predicate(r)]
        if options:
            chosen = sorted(options, key=lambda r: r["sample_id"])[0]
            render_example(chosen, grounder, output / (name + ".png"))
            examples[name] = chosen["sample_id"]
    (output / "examples.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
    print("pooled", json.dumps(summary["pooled"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
