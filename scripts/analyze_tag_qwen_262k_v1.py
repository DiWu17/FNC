"""Descriptive TAG-Qwen results, accessible only after all 3,581 predictions seal."""
from __future__ import annotations

from collections import defaultdict
import json
import statistics

import run_tag_qwen_262k_v1 as run


def main():
    gate = run.gate()  # No label access or even label-file hashing before this.
    spec = run.read(run.PROTOCOL)
    root = run.ROOT
    paths = {
        "pointer": root / "runs/paper_scale_ablation_v2/pointerbench/labels.jsonl",
        "sspro": root / "data/public_v1/manifests/screenspot_pro.jsonl",
        "mmb": root / "data/mmbench_gui_l2_v1/source/L2_annotations.json",
    }
    for name, path in paths.items():
        run.require(run.sha(path) == spec["label_sha256_from_prior_receipts"][name], "Label SHA changed: " + name)
    pointer_rows, ssp_rows, mmb_rows = run.jsonl(paths["pointer"]), run.jsonl(paths["sspro"]), run.read(paths["mmb"])
    pointer = {r["sample_id"]: r for r in pointer_rows}
    ssp = {r["sample_id"]: r for r in ssp_rows}
    mmb = {r["index"]: r for r in mmb_rows}
    run.require(len(pointer_rows) == len(pointer) == 1000 and len(ssp_rows) == len(ssp) == 1581 and len(mmb_rows) == len(mmb) == 3594, "Label count drift")
    import analyze_pointer_mvp_same_cohort_v1 as pointer_scoring
    import run_mmbench_gui_l2_v1 as mmb_scoring
    from clickagent.public_evaluation import point_in_geometry
    mmb_reference = mmb_scoring.source_reference_hit()
    results, details = {}, []
    for suite, n in run.COUNTS.items():
        public = run.public(suite)
        bundle = run.read(run.RUN / "predictions" / f"{suite}.json")
        scorer = pointer_scoring.load_official_scorer(suite)[0] if suite in ("pro", "sheets") else None
        if suite in ("pro", "sheets"):
            run.require({r["sample_id"] for r in public} <= set(pointer), "Pointer label IDs changed")
        elif suite == "sspro":
            run.require({r["sample_id"] for r in public} == set(ssp), "SSP label IDs changed")
        hits, by_group = [], defaultdict(list)
        for row, prediction in zip(public, bundle["rows"]):
            sid, point = row["sample_id"], prediction["point"]
            if suite in ("pro", "sheets"):
                hit = pointer_scoring.hit_with_parity(point, pointer[sid], sid, scorer)
            elif suite == "sspro":
                hit = bool(point is not None and point_in_geometry([v / size for v, size in zip(point, row["image_size"])], ssp[sid]["target_geometry"]))
            else:
                official = mmb[row["source_index"]]
                run.require(all(official[k] == row[k] for k in ("instruction", "image_size", "data_type", "platform", "grounding_type")), "MMB source identity changed")
                run.require(official["app_name"] == row["application"], "MMB application changed")
                hit = mmb_scoring.official_hit(point, official["bbox"], row["image_size"])
                if point is not None:
                    run.require(hit == bool(mmb_reference(official["bbox"].copy(), tuple(row["image_size"]), tuple(point))), "MMB official scorer parity failed")
                for key in ("grounding_type", "platform", "data_type"):
                    by_group[(key, row[key])].append(int(hit))
            hits.append(int(hit))
            details.append({"sample_id": sid, "suite": suite, "point": point, "hit": int(hit)})
        run.require(len(hits) == n, "Fixed denominator changed")
        times = [r["elapsed_seconds"] for r in bundle["rows"]]
        results[suite] = {
            "n": n, "correct": sum(hits), "accuracy_percent": 100 * sum(hits) / n,
            "mean_seconds": statistics.mean(times), "median_seconds": statistics.median(times),
            "mean_wall_seconds": statistics.mean(r["wall_seconds"] for r in bundle["rows"]),
            "mean_backbone_forward_calls": statistics.mean(r["backbone_forward_calls"] for r in bundle["rows"]),
            "mean_processed_pixels": statistics.mean(r["processed_pixels"] for r in bundle["rows"]),
            "views_per_query": 1,
            "peak_allocated_GiB_by_gpu": [max(r["peak_allocated_bytes"][i] for r in bundle["rows"]) / 2**30 for i in (0, 1)],
            "missing_points": sum(r["point"] is None for r in bundle["rows"]),
            "instruction_fallbacks": sum(r["instruction_fallback"] for r in bundle["rows"]),
            "generation_cap_reached": sum(not r["terminated_by_eos"] for r in bundle["rows"]),
            "strata": {f"{key}={value}": {"n": len(v), "correct": sum(v), "accuracy_percent": 100 * sum(v) / len(v)} for (key, value), v in by_group.items()},
        }
    out = run.RUN / "analysis"
    summary = {"complete": True, "method": run.METHOD, "protocol_sha256": run.sha(run.PROTOCOL),
               "public_completion_sha256": run.sha(run.RUN / "public_completion.json"), "results": results,
               "limitations": spec["limitations"], "official_pointer_mmb_parity": "passed on every non-null point",
               "label_reading_after_all_public_seals": True, "new_training_or_model_selection": False}
    run.write(out / "summary.json", summary, immutable=True)
    run.write(out / "per_query.json", details, immutable=True)
    lines = [run.METHOD, "262144 pixels per view; one view; no native Qwen baseline dispatched.",
             "All 3581 public predictions sealed and verified before this analysis opened labels.", "",
             "Cohort\tCorrect/N\tAccuracy (%)\tMean seconds\tMedian seconds"]
    for suite, r in results.items():
        lines.append(f"{suite}\t{r['correct']}/{r['n']}\t{r['accuracy_percent']:.4f}\t{r['mean_seconds']:.3f}\t{r['median_seconds']:.3f}")
    lines += ["", "Scope and limitations:", *spec["limitations"]]
    content = "\n".join(lines) + "\n"
    path = out / "readout.txt"
    if path.exists():
        run.require(path.read_text(encoding="utf-8") == content, "Readout drift")
    else:
        path.write_text(content, encoding="utf-8")
    current_path = root / "CURRENT_RESEARCH.json"
    current = run.read(current_path)
    current.setdefault("tag_qwen_262k_experiment", {}).update(
        run_id=run.RUN.name, status="complete_analyzed", completed_at_utc=run.now(),
        protocol_path=str(run.PROTOCOL), protocol_sha256=run.sha(run.PROTOCOL),
        readout_path=str(path), summary_path=str(out / "summary.json"),
        result_snapshot=results, limitations=spec["limitations"])
    run.write(current_path, current)
    print(content, flush=True)


if __name__ == "__main__":
    main()
