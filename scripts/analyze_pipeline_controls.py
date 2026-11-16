"""Additional cached controls: visible target fraction, paired targets, hook parity."""
from collections import Counter, defaultdict
import json
from pathlib import Path

import joblib
import numpy as np

from analyze_public_failures import polygon_rect_area, target_masks
from summarize_refinement import render_example
from clickagent.action_benchmark import sha256, write_json
from clickagent.public_benchmark import read_jsonl, cache_key


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/public_refinement_v1"
OUT = RUN / "structural_analysis"


def max_visible_fraction(geometry, box, size):
    rect = np.asarray(box) / [size[0], size[1], size[0], size[1]]
    fractions = []
    if geometry["type"] == "bbox":
        for a,b,c,d in geometry["boxes"]:
            intersection = max(0, min(c,rect[2])-max(a,rect[0])) * max(0,min(d,rect[3])-max(b,rect[1]))
            fractions.append(intersection/((c-a)*(d-b)))
    else:
        for polygon in geometry["polygons"]:
            p = np.asarray(polygon)
            area = abs(float(np.sum(p[:,0]*np.roll(p[:,1],-1)-p[:,1]*np.roll(p[:,0],-1))))/2
            fractions.append(polygon_rect_area(polygon,rect)/area)
    return min(1.0,max(fractions))


def main():
    import matplotlib
    matplotlib.use("Agg")
    cases = {c["sample_id"]:c for c in read_jsonl(RUN/"selected_manifest.jsonl")}
    records = read_jsonl(OUT/"per_sample.jsonl")
    by_id={r["sample_id"]:r for r in records}
    visibility,visibility_rows=defaultdict(Counter),[]
    for r in records:
        if not r["positive"]:
            continue
        c=cases[r["sample_id"]]
        fraction=max_visible_fraction(c["target_geometry"],r["crop"],c["image_size"])
        category="no_area_overlap" if fraction==0 else "less_than_half" if fraction<.5 else "partial_at_least_half" if fraction<1-1e-10 else "full_component"
        d=visibility[category]
        d["n"]+=1;d["hits"]+=r["fine_hit"];d["misses"]+=not r["fine_hit"]
        d["local_WAIT"]+=r["local_status"]=="fallback_wait"
        visibility_rows.append({"sample_id":r["sample_id"],"fraction":fraction,"category":category,"fine_hit":r["fine_hit"]})
    pairs=json.loads((OUT/"same_screenshot_pairs.json").read_text(encoding="utf-8"))
    pair_counts=Counter()
    detailed=[]
    baseline={}
    for name in ("screenspot_v2","screenspot_pro","osworld_g","venusbench_gd"):
        for r in read_jsonl(ROOT/"runs/public_v1"/name/"predictions.jsonl"):
            if r["method"]=="prefill_hidden":baseline[r["sample_id"]]=r
    for p in pairs:
        a,b=by_id[p["a"]],by_id[p["b"]]
        if not p["same_coarse_patch"]:
            continue
        missing=not a["crop_contains_target"] or not b["crop_contains_target"]
        pair_counts["same_patch_pairs"]+=1
        pair_counts["same_patch_and_one_or_both_targets_outside_crop"]+=missing
        pair_counts["same_patch_native_both_correct_and_one_or_both_outside_crop"]+=missing and p["native_full_hits_both"]
        if missing:
            maps=[];masks=[]
            for r in (a,b):
                sid=r["sample_id"]
                with np.load(ROOT/"runs/public_v1"/sid.split(":")[0]/"features"/(cache_key(sid)+".npz"),allow_pickle=False) as z:
                    maps.append(z["prefill_raw_map"].copy())
                _,overlap=target_masks(cases[sid]["target_geometry"],baseline[sid]["grid_shape"],cases[sid]["image_size"])
                masks.append(overlap)
            pa,pb=maps[0].argmax(axis=1),maps[1].argmax(axis=1)
            tracking=(pa!=pb)&masks[0][pa]&masks[1][pb]
            detailed.append({**p,"head_peaks_changed":int((pa!=pb).sum()),
                             "any_head_tracks_both_different_target_cells":bool(tracking.any()),
                             "tracking_head_indices":np.flatnonzero(tracking).tolist()})
    # Compare the two extraction implementations on pixel-identical cached inputs.
    old_root=ROOT/"runs/visual_state_v2_matched/features"
    old_run=json.loads((old_root/"run.json").read_text(encoding="utf-8"))
    old_manifest=Path(old_run["manifest_path"])
    old_cases={c["sample_id"]:c for c in read_jsonl(old_manifest)}
    with np.load(old_root/"prefill.npz",allow_pickle=False) as z:
        ids=z["sample_ids"].astype(str);index={sid:i for i,sid in enumerate(ids)}
        old_hidden=z["hidden"].copy();old_attention=z["attention"].copy()
    matched=0;different_pixels=0;max_hd=max_ad=0.0
    for path in sorted((ROOT/"runs/action_v3/grounding_raw").glob("*.npz")):
        with np.load(path,allow_pickle=False) as z:
            meta=json.loads(str(z["metadata"]));sid=meta["sample_id"]
            if sid not in index:continue
            if sha256(old_manifest.parent/old_cases[sid]["image"])!=meta["image_sha256"]:
                different_pixels+=1;continue
            i=index[sid];matched+=1
            max_hd=max(max_hd,float(np.abs(z["hidden"]-old_hidden[i]).max()))
            max_ad=max(max_ad,float(np.abs(z["attention"]-old_attention[i]).max()))
    collection=json.loads((ROOT/"runs/action_v3/grounding_collection.json").read_text(encoding="utf-8"))
    parity={"matched_pixel_identical_cached_prefill_cases":matched,"different_image_cases_excluded":different_pixels,
            "same_system_prompt":collection["feature_identity"]["system_prompt"]==old_run["system_prompt"],
            "same_model_identity":collection["feature_identity"]["model"]==old_run["model_identity"],
            "max_hidden_abs_difference":max_hd,"max_attention_feature_abs_difference":max_ad,
            "scope":"Cached original SelectedLayerRecorder vs action RawSelectedLayerRecorder; prefill only; no new VLM forwards."}
    old_probe={}
    for stage in ("prefill","post_decision"):
        with np.load(old_root/(stage+".npz"),allow_pickle=False) as z:
            mask=z["splits"].astype(str)=="test"
            x=z["hidden"][mask];labels=z["labels"].astype(str)[mask]
        model=joblib.load(ROOT/"runs/visual_state_v2_matched/evaluation/models"/(stage+"__hidden.joblib"))
        old_probe[stage]={"old_test_n":len(labels),"old_test_accuracy":float(np.mean(model.predict(x)==labels)),
                          "class_names":list(model.classes_)}
    covered_miss=[r["fine_distance_px"] for r in records if r["positive"] and r["crop_contains_target"] and r["local_status"]=="refined" and not r["fine_hit"]]
    wait_veto={m:{"correct_clicks_lost":sum(r["fine_hit"] and r["local_status"]=="fallback_wait" and r["gates"][m]["allow"] for r in records),
                  "negative_clicks_prevented":sum(not r["positive"] and r["local_status"]=="fallback_wait" and r["gates"][m]["allow"] for r in records)}
               for m in ("prefill_attention","prefill_hidden","prefill_combined","text_decision_shared")}
    out={"analysis_source_sha256":sha256(Path(__file__)),
         "visibility_best_component_fraction":{k:dict(v) for k,v in visibility.items()},
         "visibility_note":"Fraction of annotated region retained is not a guarantee of semantic context or actual clickability. Multiple valid components use maximum fraction.",
         "same_screenshot_pair_controls":dict(pair_counts),"raw_head_pair_diagnostics":detailed,
         "raw_head_note":"GT-selected any-head tracking is a diagnostic oracle, not a deployable selected-head policy.",
         "cached_feature_implementation_parity":parity,"old_hidden_probe_control":old_probe,
         "covered_refined_wrong_coordinate_distance":{"n":len(covered_miss),"median_px":float(np.median(covered_miss)),
             "within_px_counts":{str(t):sum(d<=t for d in covered_miss) for t in (5,10,20)}},
         "WAIT_veto_counterfactual_exploratory_only":wait_veto,
         "WAIT_veto_note":"Post-hoc deterministic diagnosis on already-inspected test outputs, not independently validated new policy results. No formal actions or scores changed."}
    write_json(OUT/"controls.json",out)
    (OUT/"crop_visibility.jsonl").write_text("".join(json.dumps(r)+"\n" for r in visibility_rows),encoding="utf-8")
    selected=next((p for p in detailed if p["native_full_hits_both"]),None)
    if selected:
        write_json(OUT/"failing_pair_example.json",{"pair":selected,"a":by_id[selected["a"]],"b":by_id[selected["b"]]})
        for key in ("a","b"):
            sid=selected[key]
            result=json.loads((RUN/"rows"/(cache_key(sid)+".json")).read_text(encoding="utf-8"))
            render_example(cases[sid],result,OUT/("failing_pair_"+key+".png"))
    print(json.dumps(out,ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
