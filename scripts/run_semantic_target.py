"""Development-selected target-word timing with a sealed fresh screenshot test."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from clickagent.action_benchmark import sha256, write_json
from clickagent.action_grounding import AttentionPointGrounder
from clickagent.public_evaluation import point_in_geometry
from clickagent.semantic_target import PROMPTS, POLICIES, temporal_map
from clickagent.target_token import spatial_scores
from run_target_token_experiment import overlap_cells, key, read_json, read_jsonl

SEED = 20261031
DATASETS = ("screenspot_v2", "screenspot_pro", "osworld_g", "venusbench_gd")


def prepare(root, output, dev_per_dataset=16, test_per_dataset=32):
    path=output/"manifest.jsonl"
    if path.exists():
        return read_jsonl(path)
    old_paths, old_hashes = set(), set()
    prior = list((root/"runs/public_v1").glob("*/selected_manifest.jsonl"))
    prior += [root/"runs/candidate_v2/manifest.jsonl",root/"runs/target_token_v1/manifest.jsonl"]
    for manifest in prior:
        for r in read_jsonl(manifest):
            old_paths.add(r["image"])
            if r.get("image_sha256"):
                old_hashes.add(r["image_sha256"])
    # Hash prior images too, to exclude alternate filenames/cross-dataset copies.
    for image in old_paths:
        old_hashes.add(sha256(Path(image)))
    rows, used = [], set(old_hashes)
    for ds in DATASETS:
        pool=[r for r in read_jsonl(root/"data/public_v1/manifests"/f"{ds}.jsonl")
              if r["expected_actionable"] and r.get("target_geometry") and r["image"] not in old_paths]
        pool.sort(key=lambda r:key(f"{SEED}:{r['sample_id']}"))
        chosen=[]
        for case in pool:
            digest=sha256(Path(case["image"]))
            if digest in used:
                continue
            used.add(digest)
            chosen.append(dict(case,image_sha256=digest,source_run=ds))
            if len(chosen)==dev_per_dataset+test_per_dataset:
                break
        if len(chosen)!=dev_per_dataset+test_per_dataset:
            raise ValueError(f"Insufficient fresh images: {ds}: {len(chosen)}")
        rows.extend(dict(r,experiment_split="dev" if i<dev_per_dataset else "test") for i,r in enumerate(chosen))
    path.write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    write_json(output/"selection.json",{"seed":SEED,"dev_per_dataset":dev_per_dataset,"test_per_dataset":test_per_dataset,
        "counts":dict(Counter(r["experiment_split"] for r in rows)),"excluded_prior_unique_images":len(old_hashes),
        "image_sha_unique_across_splits":len({r['image_sha256'] for r in rows})==len(rows),
        "prior_manifests":[str(p) for p in prior],"selection_uses_model_outputs":False,
        "note":"Fresh image-content disjoint sample, not full official benchmarks; applications may overlap."})
    return rows


def load_feature(output,case,variant):
    with np.load(output/"features"/variant/f"{key(case['sample_id'])}.npz",allow_pickle=False) as z:
        return z["raw_attention"],json.loads(str(z["metadata"]))


def score_map(raw,meta,case,grounder,readout="legacy_96"):
    if raw is None:
        return {"point":None,"hit":False,"cell_overlap":False}
    grid=meta["grid_shape"]
    point=spatial_scores(raw,readout,grounder,grid,case["image_size"])
    hit=point_in_geometry([point[0]/case["image_size"][0],point[1]/case["image_size"][1]],case["target_geometry"])
    mask=overlap_cells(case["target_geometry"],grid)
    if readout=="legacy_96":
        idx=grounder.predict_point(raw,grid,case["image_size"])["patch_index"]
    else:
        idx=int((raw[:32] if readout=="layer15_mean" else raw).mean(0).argmax())
    return {"point":point,"hit":bool(hit),"cell_overlap":bool(mask[idx])}


def freeze_policy(root,output,cases):
    path=output/"frozen_selection.json"
    if path.exists():
        return read_json(path)
    dev=[r for r in cases if r["experiment_split"]=="dev"]
    grounder=AttentionPointGrounder.load(root/"runs/action_v3/grounder/grounder.joblib")
    options=[]
    for variant in PROMPTS:  # original before compact is the fixed tie-break
        for policy in POLICIES:
            hits=0
            for case in dev:
                raw,meta=load_feature(output,case,variant)
                hits+=score_map(temporal_map(raw,meta,policy),meta,case,grounder)["hit"]
            options.append({"variant":variant,"policy":policy,"dev_hits":int(hits),"n_dev":len(dev)})
    selected=max(options,key=lambda r:r["dev_hits"])
    # Only dev feature paths may exist before this record is written.
    forbidden=[str(output/"features"/v/f"{key(c['sample_id'])}.npz") for v in PROMPTS for c in cases
               if c["experiment_split"]=="test" and (output/"features"/v/f"{key(c['sample_id'])}.npz").exists()]
    if forbidden:
        raise ValueError("Test features existed before policy freeze")
    record={"selected":selected,"all_development_options":options,"test_features_at_freeze":0,
        "identity_sha256":sha256(output/"identity.json"),"spatial_readout":"legacy_96 frozen Save head",
        "selection_rule":"max dev point hits; original then compact; fixed policy order breaks ties",
        "no_parameter_training":True}
    write_json(path,record)
    return record


def evaluate(root,output,cases):
    selected=read_json(output/"frozen_selection.json")["selected"]
    grounder=AttentionPointGrounder.load(root/"runs/action_v3/grounder/grounder.joblib")
    rows=[]
    for case in cases:
        if case["experiment_split"]!="test":
            continue
        observations={v:load_feature(output,case,v) for v in PROMPTS}
        for method,variant,policy,readout in (
            ("v1_original_target_mean","original","target_mean","legacy_96"),
            ("v1_original_target_first","original","target_first","legacy_96"),
            ("optimized",selected["variant"],selected["policy"],"legacy_96"),
            ("optimized_layer15_mean",selected["variant"],selected["policy"],"layer15_mean"),
            ("selected_prompt_prefill",selected["variant"],None,"legacy_96")):
            raw,meta=observations[variant]
            heat=raw[0] if policy is None else temporal_map(raw,meta,policy)
            rows.append(dict(sample_id=case["sample_id"],dataset=case["dataset"],method=method,
                variant=variant,policy=policy,**score_map(heat,meta,case,grounder,readout)))
        native=read_json(output/"native"/f"{key(case['sample_id'])}.json")
        for method,point in (("qwen_native_strict",native["point"]),("qwen_native_completed_lax",native["completed_lax_point"])):
            correct=point is not None and point_in_geometry([point[0]/case['image_size'][0],point[1]/case['image_size'][1]],case['target_geometry'])
            rows.append(dict(sample_id=case["sample_id"],dataset=case["dataset"],method=method,point=point,hit=bool(correct)))
    (output/"predictions.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8")
    methods={}
    for name in sorted({r["method"] for r in rows}):
        group=[r for r in rows if r["method"]==name]
        methods[name]={"n":len(group),"hits":sum(r["hit"] for r in group),"accuracy":sum(r["hit"] for r in group)/len(group),
            "datasets":{ds:{"n":sum(r["dataset"]==ds for r in group),"hits":sum(r["hit"] for r in group if r["dataset"]==ds)}
                for ds in sorted({r["dataset"] for r in group})}}
    summary={"selected_on_dev":selected,"methods":methods,"test_n":len(rows)//7,
        "scope":"Fresh screenshot-content-disjoint exploratory test; positive grounding only, no actual clicks."}
    write_json(output/"summary.json",summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,default=Path.cwd())
    parser.add_argument("--output",type=Path,default=Path("runs/semantic_target_v2"))
    parser.add_argument("--phase",choices=("prepare","dev","test","all","evaluate"),default="all")
    args=parser.parse_args()
    output=args.output;output.mkdir(parents=True,exist_ok=True)
    cases=prepare(args.root,output)
    if args.phase=="prepare":
        print(json.dumps(read_json(output/"selection.json")),flush=True);return
    if args.phase=="evaluate":
        print(json.dumps(evaluate(args.root,output,cases)),flush=True);return
    from clickagent.model_features import ExtractionConfig
    from clickagent.semantic_target import SemanticTargetRuntime
    from clickagent.public_diagnostics import parse_offline_lax_click
    from PIL import Image
    source=args.root/"src/clickagent"
    cfg=read_json(args.root/"runs/candidate_v2/identity.json")["config"]
    cfg.update(layers=tuple(cfg["layers"]),conditions=("prefill",),max_new_tokens=64,seed=SEED,
        max_memory={int(k) if k.isdigit() else k:v for k,v in cfg["max_memory"].items()},offload_folder=str((output/"offload").resolve()))
    identity=json.loads(json.dumps({"config":cfg,"prompts":PROMPTS,"policies":POLICIES,
        "manifest_sha256":sha256(output/"manifest.jsonl"),"driver_sha256":sha256(Path(__file__)),
        "source_sha256":{name:sha256(source/name) for name in ("semantic_target.py","target_token.py","public_model.py","action_model.py","model_features.py","action_grounding.py")},
        "grounder_sha256":sha256(args.root/"runs/action_v3/grounder/grounder.joblib")}))
    if (output/"identity.json").exists() and read_json(output/"identity.json")!=identity:
        raise ValueError("Experiment source/config changed; use a new output.")
    write_json(output/"identity.json",identity)
    (output/"frozen_source").mkdir(exist_ok=True)
    for name in identity["source_sha256"]:
        if not (output/"frozen_source"/name).exists():
            (output/"frozen_source"/name).write_bytes((source/name).read_bytes())
    if not (output/"frozen_source"/Path(__file__).name).exists():
        (output/"frozen_source"/Path(__file__).name).write_bytes(Path(__file__).read_bytes())
    for variant in PROMPTS:(output/"features"/variant).mkdir(parents=True,exist_ok=True)
    (output/"native").mkdir(exist_ok=True)
    runtime=SemanticTargetRuntime(ExtractionConfig(**cfg))
    write_json(output/"runtime.json",runtime.runtime_metadata)
    phases=("dev","test") if args.phase=="all" else (args.phase,)
    for phase in phases:
        if phase=="test" and not (output/"frozen_selection.json").exists():
            raise ValueError("Freeze the development choice before collecting test features")
        chosen=[r for r in cases if r["experiment_split"]==phase]
        for i,case in enumerate(chosen,1):
            if sha256(Path(case["image"]))!=case["image_sha256"]:raise ValueError("Image changed")
            with Image.open(case["image"]) as opened:image=opened.convert("RGB")
            texts={}
            for variant in PROMPTS:
                path=output/"features"/variant/f"{key(case['sample_id'])}.npz"
                if not path.exists():
                    arrays,meta=runtime.infer_variant(image,case["instruction"],variant)
                    meta.update(sample_id=case["sample_id"],experiment_split=phase,image_sha256=case["image_sha256"])
                    arrays["metadata"]=np.asarray(json.dumps(meta,ensure_ascii=False))
                    tmp=path.with_suffix(".tmp")
                    with tmp.open("wb") as stream:np.savez_compressed(stream,**arrays)
                    tmp.replace(path)
                    texts[variant]=meta["target_text"]
            if phase=="test":
                path=output/"native"/f"{key(case['sample_id'])}.json"
                if not path.exists():
                    result=runtime.infer_text_action_public(image,case["instruction"],max_new_tokens=32)
                    completed=result["terminated_by_eos"] or result["generated_text"].endswith(("\n","\r"))
                    parsed=parse_offline_lax_click(result["generated_text"]) if completed else None
                    result["completed_lax_point"]=[round(v*(side-1)/1000) for v,side in zip(parsed,image.size)] if parsed else None
                    result["sample_id"]=case["sample_id"]
                    write_json(path,result)
            print(f"{phase} [{i}/{len(chosen)}] {case['dataset']} names={texts}",flush=True)
        if phase=="dev":
            frozen=freeze_policy(args.root,output,cases)
            print("FROZEN "+json.dumps(frozen["selected"]),flush=True)
        else:
            summary=evaluate(args.root,output,cases)
            write_json(output/"completion.json",{"complete":True,"n_test":summary["test_n"],"policy_frozen_before_test":True})
            print("COMPLETE "+json.dumps(summary),flush=True)


if __name__=="__main__":main()
