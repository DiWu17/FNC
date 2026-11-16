"""Fixed-decoder timing ablation: prefill, action word, and generated target.

Exploratory, paired static-screenshot experiment on previously inspected public
cohorts. No fitted parameters or deployment thresholds are selected here.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

import numpy as np

from clickagent.action_benchmark import sha256, write_json
from clickagent.action_grounding import AttentionPointGrounder
from clickagent.public_evaluation import point_in_geometry, _point_in_polygon
from clickagent.target_token import TARGET_NAME_PROMPT, spatial_scores


SEED = 20261030
STAGES = ("prefill", "action_word", "target_marker", "target_first", "target_last", "target_mean", "all_generated_mean")
READOUTS = ("legacy_96", "layer15_mean", "all_layers_mean")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_jsonl(path):
    return [json.loads(s) for s in Path(path).read_text(encoding="utf-8").splitlines() if s.strip()]


def key(value):
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def order(case):
    return key(f"{SEED}:{case['sample_id']}")


def select(root, output, per_dataset):
    if (output / "manifest.jsonl").exists():
        return read_jsonl(output / "manifest.jsonl")
    cases, pool = [], []
    for dataset in ("screenspot_v2", "screenspot_pro", "osworld_g", "venusbench_gd"):
        eligible = [dict(r, source_run=dataset) for r in read_jsonl(root / "runs/public_v1" / dataset / "selected_manifest.jsonl")
                    if r["expected_actionable"] and r.get("target_geometry")]
        pool.extend(eligible)
        cases.extend(sorted(eligible, key=order)[:per_dataset])
    groups = defaultdict(list)
    for case in pool:
        groups[(case["dataset"], case["image"])].append(case)
    pairs = []
    for dataset in sorted({r["dataset"] for r in pool}):
        available = [sorted(group, key=order) for (ds, _), group in groups.items()
                     if ds == dataset and len({r["instruction"] for r in group}) > 1]
        available.sort(key=lambda group: order(group[0]))
        if available:
            first = available[0][0]
            second = next(r for r in available[0] if r["instruction"] != first["instruction"])
            pairs.append([first["sample_id"], second["sample_id"]])
            cases.extend((first, second))
    cases = list({r["sample_id"]: r for r in cases}.values())
    for case in cases:
        case["image_sha256"] = sha256(Path(case["image"]))
    (output / "manifest.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cases), encoding="utf-8")
    write_json(output / "selection.json", {"seed": SEED, "core_per_dataset": per_dataset,
        "n": len(cases), "datasets": dict(Counter(r["dataset"] for r in cases)), "same_image_pairs": pairs,
        "selection_uses_model_outcomes": False, "all_requests_are_grounding_positives": True,
        "scope": "Exploratory timing diagnosis on already-inspected cohorts, not independent official benchmark scores."})
    return cases


def _segment_intersects(a, b, c, d):
    def cross(p, q, r):
        return (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
    # Bounding boxes avoid accepting disjoint collinear segments.
    if max(min(a[0], b[0]), min(c[0], d[0])) > min(max(a[0], b[0]), max(c[0], d[0])):
        return False
    if max(min(a[1], b[1]), min(c[1], d[1])) > min(max(a[1], b[1]), max(c[1], d[1])):
        return False
    return cross(a,b,c)*cross(a,b,d) <= 0 and cross(c,d,a)*cross(c,d,b) <= 0


def overlap_cells(geometry, grid):
    """Diagnostic peak-cell intersection, independent of center quantization.

    This does not change the click metric, nor claim a valid click within a
    merely overlapping coarse cell. Polygon boundary contact counts here.
    """
    gh, gw = grid
    result = []
    for row in range(gh):
        for col in range(gw):
            a, b, c, d = col/gw, row/gh, (col+1)/gw, (row+1)/gh
            if geometry["type"] == "bbox":
                result.append(any(min(c,x1)>max(a,x0) and min(d,y1)>max(b,y0)
                                  for x0,y0,x1,y1 in geometry["boxes"]))
                continue
            corners = [(a,b),(c,b),(c,d),(a,d)]
            good = False
            for poly in geometry["polygons"]:
                if (any(a <= x <= c and b <= y <= d for x,y in poly)
                    or any(_point_in_polygon(p, poly) for p in corners)
                    or any(_segment_intersects(p,q,r,s) for p,q in zip(poly,poly[1:]+poly[:1])
                           for r,s in zip(corners,corners[1:]+corners[:1]))):
                    good = True
                    break
            result.append(good)
    return np.asarray(result, bool)


def evaluate(root, output, cases):
    grounder = AttentionPointGrounder.load(root / "runs/action_v3/grounder/grounder.joblib")
    rows, metadata = [], {}
    for case in cases:
        path = output / "features" / f"{key(case['sample_id'])}.npz"
        if not path.exists():
            continue
        with np.load(path, allow_pickle=False) as z:
            raw = z["raw_attention"]
            meta = json.loads(str(z["metadata"]))
        metadata[case["sample_id"]] = meta
        grid, size = meta["grid_shape"], case["image_size"]
        mask = overlap_cells(case["target_geometry"], grid)
        for stage in STAGES:
            indices = meta["stages"].get(stage)
            attention = raw[indices].mean(0) if indices else None
            for readout in READOUTS:
                point, peak_overlap, mass = None, False, None
                if attention is not None:
                    point = spatial_scores(attention, readout, grounder, grid, size)
                    if readout == "legacy_96":
                        idx = grounder.predict_point(attention, grid, size)["patch_index"]
                        heat = None
                    else:
                        heat = attention[:32].mean(0) if readout == "layer15_mean" else attention.mean(0)
                        idx = int(heat.argmax())
                    peak_overlap = bool(mask[idx])
                    if heat is not None:
                        mass = float(heat[mask].sum() / heat.sum()) if heat.sum() > 0 else 0.
                correct = point is not None and point_in_geometry([point[0]/size[0],point[1]/size[1]],case["target_geometry"])
                rows.append({"sample_id": case["sample_id"], "dataset": case["dataset"], "stage": stage,
                    "readout": readout, "available": indices is not None, "point": point, "hit": bool(correct),
                    "peak_cell_overlaps_target": peak_overlap, "target_intersecting_cells_attention_mass": mass,
                    "chance_cell_overlap": float(mask.mean())})
    (output / "predictions.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    summary = {"n":len(metadata),"complete":len(metadata)==len(cases), "methods":{},
        "target_span_available":sum(m["target_available"] for m in metadata.values()),
        "generated_actions":dict(Counter(m["action"] for m in metadata.values())),
        "note":"Stage selection never uses annotations. Missing target spans count as failures. Point and cell-overlap metrics are distinct; no thresholds, head selection or training."}
    by_id = {r["sample_id"]:r for r in cases}
    for readout in READOUTS:
        summary["methods"][readout]={}
        for stage in STAGES:
            chosen=[r for r in rows if r["readout"]==readout and r["stage"]==stage]
            base={r["sample_id"]:r for r in rows if r["readout"]==readout and r["stage"]=="prefill"}
            entry={"n":len(chosen),"available":sum(r["available"] for r in chosen),
                "hits":sum(r["hit"] for r in chosen),"peak_cell_overlaps":sum(r["peak_cell_overlaps_target"] for r in chosen),
                "paired_gains_over_prefill":sum(r["hit"] and not base[r["sample_id"]]["hit"] for r in chosen),
                "paired_losses_over_prefill":sum(not r["hit"] and base[r["sample_id"]]["hit"] for r in chosen),
                "datasets":{ds:{"n":sum(r["dataset"]==ds for r in chosen),
                    "hits":sum(r["hit"] for r in chosen if r["dataset"]==ds),
                    "cell_overlap":sum(r["peak_cell_overlaps_target"] for r in chosen if r["dataset"]==ds)}
                    for ds in sorted({r["dataset"] for r in chosen})}}
            masses=[r["target_intersecting_cells_attention_mass"] for r in chosen if r["target_intersecting_cells_attention_mass"] is not None]
            entry["mean_target_cell_mass_when_available"] = float(np.mean(masses)) if masses else None
            summary["methods"][readout][stage]=entry
    pairs=read_json(output/"selection.json")["same_image_pairs"]
    summary["same_image_pairs"]={readout:{stage:sum(all(next(r for r in rows if r["sample_id"]==sid and r["stage"]==stage and r["readout"]==readout)["hit"]
        for sid in pair) for pair in pairs if all(sid in metadata for sid in pair)) for stage in STAGES} for readout in READOUTS}
    write_json(output / "summary.json",summary)
    lines=["# Generated-target own-token timing ablation","",f"Completed {len(metadata)}/{len(cases)}; generated target spans available: {summary['target_span_available']}.",
        "", "All spatial decoders, layers, resolution, grid centers and old weights are fixed. This is an exploratory localization experiment; no gate or OS actions.",
        "", "| Readout | Stage | Point hits | Peak cells intersect target | Available | Paired gains / losses |", "|---|---|---:|---:|---:|---:|"]
    for readout in READOUTS:
        for stage in STAGES:
            e=summary["methods"][readout][stage]
            lines.append(f"| {readout} | {stage} | {e['hits']}/{e['n']} | {e['peak_cell_overlaps']}/{e['n']} | {e['available']} | {e['paired_gains_over_prefill']} / {e['paired_losses_over_prefill']} |")
    (output/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path.cwd())
    parser.add_argument("--output",type=Path,default=Path("runs/target_token_v1"))
    parser.add_argument("--per-dataset",type=int,default=16)
    parser.add_argument("--evaluate-only",action="store_true")
    parser.add_argument("--limit",type=int)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    cases=select(args.root,args.output,args.per_dataset)
    if not args.evaluate_only:
        from clickagent.model_features import ExtractionConfig
        from clickagent.target_token import TargetTokenRuntime
        from PIL import Image
        cfg=read_json(args.root/"runs/candidate_v2/identity.json")["config"]
        cfg.update(layers=tuple(cfg["layers"]),conditions=("prefill",),max_new_tokens=64,seed=SEED,
            max_memory={int(k) if k.isdigit() else k:v for k,v in cfg["max_memory"].items()},
            offload_folder=str((args.output/"offload").resolve()))
        source=Path(__file__).resolve().parents[1]/"src/clickagent"
        identity={"config":cfg,"prompt":TARGET_NAME_PROMPT,"manifest_sha256":sha256(args.output/"manifest.jsonl"),
            "sources":{n:sha256(source/n) for n in ("target_token.py","public_model.py","action_model.py","model_features.py","action_grounding.py")},
            "driver_sha256":sha256(Path(__file__)),"grounder_sha256":sha256(args.root/"runs/action_v3/grounder/grounder.joblib")}
        identity=json.loads(json.dumps(identity))
        if (args.output/"identity.json").exists() and read_json(args.output/"identity.json")!=identity:
            raise ValueError("Source/config changed; use a new run directory.")
        write_json(args.output/"identity.json",identity)
        (args.output/"frozen_source").mkdir(exist_ok=True)
        for name in identity["sources"]:
            target=args.output/"frozen_source"/name
            if not target.exists():
                target.write_bytes((source/name).read_bytes())
        target=args.output/"frozen_source"/Path(__file__).name
        if not target.exists():
            target.write_bytes(Path(__file__).read_bytes())
        (args.output/"features").mkdir(exist_ok=True)
        pending=[r for r in cases if not (args.output/"features"/f"{key(r['sample_id'])}.npz").exists()]
        if args.limit:
            pending=pending[:args.limit]
        if pending:
            print(f"Loading frozen model for {len(pending)} cases / {len(cases)} planned",flush=True)
            runtime=TargetTokenRuntime(ExtractionConfig(**cfg))
            write_json(args.output/"runtime.json",runtime.runtime_metadata)
            for i,case in enumerate(pending,1):
                if sha256(Path(case["image"]))!=case["image_sha256"]:
                    raise ValueError("Image content changed")
                with Image.open(case["image"]) as opened:
                    image=opened.convert("RGB")
                arrays,meta=runtime.infer_target_name(image,case["instruction"],max_new_tokens=64)
                meta.update(sample_id=case["sample_id"],dataset=case["dataset"])
                arrays["metadata"]=np.asarray(json.dumps(meta,ensure_ascii=False))
                path=args.output/"features"/f"{key(case['sample_id'])}.npz"
                temporary=path.with_suffix(".tmp")
                with temporary.open("wb") as f:
                    np.savez_compressed(f,**arrays)
                temporary.replace(path)
                print(f"[{i}/{len(pending)}] {case['dataset']} target={meta['target_text']!r} action={meta['action']} "
                      f"tokens={len(meta['generated_token_ids'])} seconds={meta['elapsed_seconds']:.2f}",flush=True)
    summary=evaluate(args.root,args.output,cases)
    print(json.dumps({"n":summary["n"],"complete":summary["complete"],"target_span_available":summary["target_span_available"]}),flush=True)


if __name__=="__main__":
    main()
