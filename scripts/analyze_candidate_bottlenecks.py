"""Read-only decomposition of proposal, ranking, null and threshold failures.

No inference, training, threshold selection, or original-result modification.
Low-FPR ROC values use test labels for descriptive diagnostics only.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from clickagent.action_benchmark import write_json
from clickagent.candidate_probes import load_features, evaluation_test_indices


def distance_to_region(point,case):
    w,h=case["image_size"]
    x,y=point
    return min(np.hypot(max(a*w-x,0,x-c*w),max(b*h-y,0,y-d*h))
               for a,b,c,d in case["target_geometry"]["boxes"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("runs/candidate_v2"))
    args=parser.parse_args()
    output=args.output
    destination=output/"bottleneck_analysis"
    destination.mkdir(exist_ok=True)
    cases,metadata,arrays=load_features(output)
    test,_=evaluation_test_indices(cases,output)
    labels=arrays["labels"].astype(bool)
    results=json.loads((output/"results.json").read_text(encoding="utf-8"))
    click=np.asarray([i for i in test if cases[i]["case_kind"]=="click"])
    report={"scope":"read-only diagnosis, no new fits or threshold selection", "strict_test":len(test),
            "click_requests":len(click),"proposal_distributions":{},"methods":{}}
    for split in ("train","dev","calibration","test"):
        ids=test if split=="test" else np.asarray([i for i,r in enumerate(cases) if r["split"]==split])
        requests=np.asarray([i for i in ids if cases[i]["case_kind"]=="click"])
        report["proposal_distributions"][split]={"queries":len(ids),"click_requests":len(requests),
            "click_requests_with_correct_proposal":int(labels[requests].any(-1).sum()),
            "click_proposal_coverage":float(labels[requests].any(-1).mean()),
            "null_label_queries":int((~labels[ids].any(-1)).sum()),
            "null_label_fraction":float((~labels[ids].any(-1)).mean()),
            "positive_points_per_candidate":labels[ids].sum(0).tolist()}
    report["dataset_proposal_coverage"]={}
    for dataset in sorted({cases[i]["dataset"] for i in click}):
        ids=np.asarray([i for i in click if cases[i]["dataset"]==dataset])
        report["dataset_proposal_coverage"][dataset]={"n":len(ids),
            "first_hits":int(labels[ids,0].sum()),"any_hits":int(labels[ids].any(-1).sum())}
    first_sources=Counter(metadata[i]["proposal"]["sources"][0] for i in click)
    first_hit_sources=Counter(metadata[i]["proposal"]["sources"][0] for i in click if labels[i,0])
    report["first_candidate_sources_click_requests"]=dict(first_sources)
    report["first_candidate_correct_sources"]=dict(first_hit_sources)
    model_root=Path("models/Qwen3-VL-8B-Instruct")
    vision=json.loads((model_root/"config.json").read_text())["vision_config"]
    cell=vision["patch_size"]*vision["spatial_merge_size"]
    sizes=[]
    for i in click:
        gh,gw=metadata[i]["grid_shape"]
        # Largest annotated component; these records use bbox geometry.
        a,b,c,d=max(cases[i]["target_geometry"]["boxes"],key=lambda v:(v[2]-v[0])*(v[3]-v[1]))
        sizes.append(min((c-a)*gw*cell,(d-b)*gh*cell))
    report["target_projected_min_side_pixels"]={"median":float(np.median(sizes)),"cell_side":cell,"buckets":{}}
    for lo,hi in ((0,4),(4,8),(8,16),(16,float("inf"))):
        ids=np.asarray([i for i,s in zip(click,sizes) if lo<=s<hi],dtype=int)
        report["target_projected_min_side_pixels"]["buckets"][f"{lo}-{hi}"]={"n":len(ids),
            "any_correct_proposal":int(labels[ids].any(-1).sum()),"first_correct":int(labels[ids,0].sum())}
    with np.load(output/"scores.npz",allow_pickle=False) as scores:
        for method in scores.files:
            probability=scores[method]
            p=probability[:,:4]
            entry={"policies":{}}
            fpr,tpr,_=roc_curve(labels[click,0],p[click,0])
            entry["raw_score_low_fpr_click_only_diagnostic"]={str(budget):{
                "max_recall_on_test_roc":float(tpr[fpr<=budget+1e-12].max()),
                "standardized_partial_auc":float(roc_auc_score(labels[click,0],p[click,0],max_fpr=budget))}
                for budget in (.01,.05,.10,.20)}
            for policy in ("verify_first","select_four"):
                chosen=np.zeros(len(cases),int) if policy=="verify_first" else p.argmax(-1)
                confidence=p[np.arange(len(cases)),chosen]
                selected_correct=labels[np.arange(len(cases)),chosen]
                eligible=confidence>probability[:,4] if probability.shape[1]==5 else np.ones(len(cases),bool)
                threshold=results["methods"][method]["policies"][policy]["0.05"]["test"]["threshold"]
                accepted=eligible&(confidence.astype(np.float64)>=threshold)
                correct_click=selected_correct&accepted
                valid_pool=labels.any(-1)
                good_selected=selected_correct[click]
                stages={"n_click_requests":len(click),"no_correct_proposal":int((~valid_pool[click]).sum()),
                    "correct_proposal_available":int(valid_pool[click].sum()),
                    "correct_proposal_available_but_wrong_selected":int((valid_pool[click]&~good_selected).sum()),
                    "correct_selected":int(good_selected.sum()),
                    "correct_selected_but_null_veto":int((good_selected&~eligible[click]).sum()),
                    "correct_selected_passes_null":int((good_selected&eligible[click]).sum()),
                    "correct_selected_but_numeric_threshold_reject":int((good_selected&eligible[click]&~accepted[click]).sum()),
                    "correct_executed":int(correct_click[click].sum()),"threshold":threshold}
                assert stages["correct_selected"]==stages["correct_selected_but_null_veto"]+stages["correct_selected_but_numeric_threshold_reject"]+stages["correct_executed"]
                assert len(click)==stages["no_correct_proposal"]+stages["correct_proposal_available_but_wrong_selected"]+stages["correct_selected"]
                wrong_clicks=[i for i in click if accepted[i] and not selected_correct[i]]
                distances=[distance_to_region(metadata[i]["points"][chosen[i]],cases[i]) for i in wrong_clicks]
                hist={"<=5px":int(sum(d<=5 for d in distances)),"5-20px":int(sum(5<d<=20 for d in distances)),
                      "20-100px":int(sum(20<d<=100 for d in distances)),">100px":int(sum(d>100 for d in distances))}
                stages["wrong_location_execution_distance"]={"n":len(distances),"median_pixels":float(np.median(distances)) if distances else None,"histogram":hist}
                stages["nonactionable_executions"]=int(sum(accepted[i] for i in test if cases[i]["case_kind"]!="click"))
                stages["wrong_location_executions"]=len(wrong_clicks)
                stages["no_correct_proposal_but_executed"]=int(sum(accepted[i] and not valid_pool[i] for i in click))
                entry["policies"][policy]=stages
            report["methods"][method]=entry
    write_json(destination/"summary.json",report)
    lines=["# Candidate bottleneck diagnosis", "", "Read-only decomposition; existing 5% calibration thresholds remain unchanged.","",
           "| Method | Policy | Correct available | Correct chosen | Null veto | Numeric threshold rejection | Correct executed |",
           "|---|---|---:|---:|---:|---:|---:|"]
    for method,entry in report["methods"].items():
        for policy,s in entry["policies"].items():
            lines.append(f"| {method} | {policy} | {s['correct_proposal_available']} | {s['correct_selected']} | {s['correct_selected_but_null_veto']} | {s['correct_selected_but_numeric_threshold_reject']} | {s['correct_executed']} |")
    (destination/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False),flush=True)


if __name__=="__main__":
    main()
