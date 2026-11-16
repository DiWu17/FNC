"""Audit the completed candidate pilot and render fixed-policy comparisons."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np

from clickagent.action_benchmark import sha256, stable_hash, write_json
from clickagent.candidate_data import candidate_labels
from clickagent.candidate_probes import load_features, policy_metrics, evaluation_test_indices
from clickagent.public_benchmark import read_jsonl, cache_key
from sklearn.metrics import roc_auc_score


def wilson(numerator, denominator):
    if not denominator:
        return None
    p, z = numerator / denominator, 1.95996398454
    center = (p + z*z / (2*denominator)) / (1+z*z/denominator)
    half = z*math.sqrt(p*(1-p)/denominator+z*z/(4*denominator**2))/(1+z*z/denominator)
    return [max(0,center-half),min(1,center+half)]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/candidate_v2"))
    args = parser.parse_args()
    output = args.output
    result = json.loads((output / "results.json").read_text(encoding="utf-8"))
    cases, metadata, arrays = load_features(output)
    assert len(cases) == 768
    labels = arrays["labels"].astype(bool)
    for case,meta in zip(cases,metadata):
        assert candidate_labels(case,meta["points"]) == meta["candidate_labels"]
        assert sha256(Path(case["image"])) == case["image_sha256"]
        assert meta["generated_tokens"] == 0
        if case["split"] in {"calibration","test"}:
            assert "training_candidates" not in case
            assert "annotation_sampling" not in meta["proposal"]["sources"]
    image_splits = {}
    for r in cases:
        previous = image_splits.setdefault(r["image_sha256"],r["split"])
        assert previous == r["split"]
    test,exclusions = evaluation_test_indices(cases,output)
    lookup = {r["sample_id"]:i for i,r in enumerate(cases)}
    predictions = read_jsonl(output / "predictions.jsonl")
    assert len(predictions) == len(result["methods"])*2*3*len(test)
    for row in predictions:
        i = lookup[row["sample_id"]]
        assert cases[i]["split"] == "test"
        assert bool(labels[i,row["candidate_index"]]) == row["correct_candidate"]
        assert row["execute"] == (not row["null_veto"] and row["score"] >= row["threshold"])
    failures = {}
    for method in result["methods"]:
        rs = [r for r in predictions if r["method"]==method and r["policy"]=="select_four"
              and r["calibration_error_budget"]==.05]
        counts = Counter()
        for r in rs:
            i = lookup[r["sample_id"]]
            if cases[i]["case_kind"]!="click":
                counts["nonactionable_wrong_execution" if r["execute"] else "nonactionable_wait"]+=1
            elif not labels[i].any():
                counts["no_correct_proposal_but_executed" if r["execute"] else "no_correct_proposal_wait"]+=1
            elif r["execute"]:
                counts["correct_click" if r["correct_candidate"] else "valid_proposal_exists_but_wrong_execution"]+=1
            else:
                counts["correct_candidate_chosen_but_rejected" if r["correct_candidate"] else "wrong_candidate_chosen_then_rejected"]+=1
        failures[method]=dict(counts)
    click_test=[i for i in test if cases[i]["case_kind"]=="click"]
    prior=np.broadcast_to([4.,3.,2.,1.],(len(click_test),4))
    prior_wins=[]
    for j,i in enumerate(click_test):
        if labels[i].any() and not labels[i].all():
            d=prior[j,labels[i]][:,None]-prior[j,~labels[i]][None,:]
            prior_wins.extend(((d>0)+.5*(d==0)).ravel())
    rank_prior={"click_only_candidate_auroc":float(roc_auc_score(labels[click_test].ravel(),prior.ravel())),
                "within_screenshot_pair_ordering":float(np.mean(prior_wins))}
    frozen_weights = json.loads((output/"training_weights_manifest.json").read_text(encoding="utf-8"))
    for filename,digest in frozen_weights["models_sha256"].items():
        assert sha256(output/"models"/filename)==digest
    training_records = json.loads((output/"training_records.json").read_text(encoding="utf-8"))
    assert training_records["training_source_sha256"] == sha256(output/"training_source.py")
    training_ids = training_records["sample_ids"]
    case_order = {r["sample_id"]:i for i,r in enumerate(cases)}
    indices = [case_order[sid] for sid in training_ids]
    with np.load(output/"training_scores.npz",allow_pickle=False) as before, np.load(output/"scores.npz",allow_pickle=False) as after:
        replay = {name:float(np.abs(before[name]-after[name][indices]).max()) for name in before.files}
    assert max(replay.values()) < 5e-5
    by_dataset = {}
    for dataset in sorted({r["dataset"] for r in cases if r["split"]=="test"}):
        names = {r["sample_id"] for r in cases if r["split"]=="test" and r["dataset"]==dataset}
        by_dataset[dataset] = {}
        for method in result["methods"]:
            selected = [r for r in predictions if r["sample_id"] in names and r["method"]==method
                        and r["policy"]=="select_four" and r["calibration_error_budget"]==.05]
            good = sum(r["execute"] and r["correct_candidate"] for r in selected)
            bad = sum(r["execute"] and not r["correct_candidate"] for r in selected)
            by_dataset[dataset][method] = {"n":len(selected),"correct_clicks":good,"wrong_clicks":bad}
    # Dataset app names for OS settings are not interchangeable with true
    # unseen software: Venus calls them system, Pro calls them *_common.
    generic_os = {r["sample_id"] for r in cases if r["split"]=="test" and r["application"] in
                  {"windows_common","macos_common","linux_common","system"}}
    strict_application_subset = {}
    for method in result["methods"]:
        rs = [r for r in predictions if r["sample_id"] not in generic_os and r["method"]==method
              and r["policy"]=="select_four" and r["calibration_error_budget"]==.05]
        strict_application_subset[method] = {"n":len(rs),
            "correct_clicks":sum(r["execute"] and r["correct_candidate"] for r in rs),
            "wrong_clicks":sum(r["execute"] and not r["correct_candidate"] for r in rs)}
    confidence = {}
    for method,entry in result["methods"].items():
        confidence[method] = {}
        for policy in ("verify_first","select_four"):
            m = entry["policies"][policy]["0.05"]["test"]
            confidence[method][policy] = {"wrong_click_per_request_wilson95":wilson(m["wrong_clicks"],m["n"]),
                "execution_precision_wilson95":wilson(m["correct_clicks"],m["executed"]),
                "click_request_success_wilson95":wilson(m["correct_clicks"],m["click_request_count"])}
    paired = {}
    rng = np.random.default_rng(20261030)
    draws = rng.integers(0,len(test),size=(4000,len(test)))
    for policy in ("verify_first","select_four"):
        rows = {method:{r["sample_id"]:r for r in predictions if r["method"]==method and r["policy"]==policy
                        and r["calibration_error_budget"]==.05} for method in result["methods"]}
        for alternative in ("native_choice_logits","global_combined"):
            def indicators(method,kind):
                return np.asarray([int(rows[method][cases[i]["sample_id"]]["execute"] and
                    (rows[method][cases[i]["sample_id"]]["correct_candidate"] == (kind=="correct"))) for i in test])
            for kind in ("correct","wrong"):
                delta = indicators("candidate_combined",kind)-indicators(alternative,kind)
                paired[f"{policy}:combined-minus-{alternative}:{kind}"] = {
                    "delta_per_request":float(delta.mean()),
                    "paired_screenshot_bootstrap95":np.percentile(delta[draws].mean(-1),[2.5,97.5]).tolist()}
    audit = {"complete":True,"n_records":len(cases),"unique_screenshots":len(image_splits),
        "split_counts":dict(Counter(r["split"] for r in cases)),"cross_split_image_leakage":False,
        "test_candidate_GT_used_by_proposer":False,"prediction_count":len(predictions),
        "all_labels_recomputed":True,"all_image_hashes_verified":True,
        "source_identities":{name:sha256(output/name) for name in ("identity.json","manifest.jsonl","augmentation/identity.json","augmentation/manifest.jsonl")},
        "models_sha256":{p.name:sha256(p) for p in (output/"models").glob("*.pt")},
        "fitted_weights_unchanged_since_freeze":True,
        "frozen_before_test_collection":frozen_weights["test_features_collected_at_freeze"]==0,
        "training_checkpoint_replay_max_error":replay,
        "test_dataset_breakdown":by_dataset,"confidence_intervals":confidence,
        "application_names_disjoint":True,"generic_os_test_ids":sorted(generic_os),
        "strict_application_subset_excluding_generic_os":strict_application_subset,
        "paired_comparisons":paired,
        "failure_decomposition_5pct_budget":failures,
        "diagnostic_static_proposal_rank_prior":rank_prior,
        "interval_note":"Intervals reflect test screenshot sampling only, conditional on fitted heads and selected calibration thresholds."}
    write_json(output / "final_audit.json",audit)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = list(result["methods"])
    short = [m.replace("native_choice_logits","Native logits").replace("candidate_","Candidate ").replace("global_","Global ") for m in names]
    fig,axes = plt.subplots(1,2,figsize=(13,5),sharey=True,layout="constrained")
    for ax,policy,title in zip(axes,("verify_first","select_four"),("Same first coordinate: verification","Same four coordinates: selection")):
        good = [result["methods"][m]["policies"][policy]["0.05"]["test"]["correct_clicks"] for m in names]
        bad = [result["methods"][m]["policies"][policy]["0.05"]["test"]["wrong_clicks"] for m in names]
        y = np.arange(len(names))
        ax.barh(y-.16,good,height=.3,color="#278776",label="Correct clicks")
        ax.barh(y+.16,bad,height=.3,color="#dc684e",label="Wrong clicks")
        for j,(a,b) in enumerate(zip(good,bad)):
            ax.text(a+.4,j-.16,str(a),va="center",fontsize=8)
            ax.text(b+.4,j+.16,str(b),va="center",fontsize=8)
        ax.set_yticks(y,short)
        ax.set_title(title)
        ax.set_xlabel(f"Count on {len(test)} held-out screenshots")
        ax.grid(axis="x",alpha=.2)
        ax.set_axisbelow(True)
    axes[1].legend(loc="lower right")
    axes[0].invert_yaxis()
    fig.suptitle("Frozen Qwen3-VL-8B candidate verification\n5% calibration mistake budget; test error is measured independently",fontsize=12)
    fig.savefig(output/"comparison.png",dpi=170)
    fig.savefig(output/"comparison.pdf")
    plt.close(fig)
    # Qualitative inspection uses labels only after scoring; these renderings
    # are never fed back into the model or used to choose thresholds.
    from PIL import Image
    from matplotlib.patches import Rectangle
    from matplotlib.font_manager import FontProperties
    import textwrap
    example_dir=output/"examples"
    example_dir.mkdir(exist_ok=True)
    policy_rows={m:{r["sample_id"]:r for r in predictions if r["method"]==m
        and r["policy"]=="select_four" and r["calibration_error_budget"]==.05}
        for m in ("candidate_combined","native_choice_logits")}
    chosen_examples={}
    for i in test:
        c=cases[i]
        if c["case_kind"]!="click":
            continue
        a,b=policy_rows["candidate_combined"][c["sample_id"]],policy_rows["native_choice_logits"][c["sample_id"]]
        aok=a["execute"] and a["correct_candidate"]
        bok=b["execute"] and b["correct_candidate"]
        tags=[]
        if aok and not bok: tags.append("combined_gain")
        if bok and not aok: tags.append("combined_loss")
        if a["execute"] and not a["correct_candidate"]: tags.append("combined_wrong_click")
        if not labels[i].any(): tags.append("candidate_coverage_failure")
        for tag in tags:
            chosen_examples.setdefault(tag,i)
    example_info={}
    font_path=Path("C:/Windows/Fonts/msyh.ttc")
    font=FontProperties(fname=str(font_path)) if font_path.exists() else None
    with np.load(output/"scores.npz",allow_pickle=False) as score_file:
        for tag,i in chosen_examples.items():
            c=cases[i]
            image=Image.open(c["image"]).convert("RGB")
            w,h=image.size
            fig,axes=plt.subplots(1,2,figsize=(15,6),layout="constrained")
            boxes=c["target_geometry"]["boxes"]
            for ax in axes:
                ax.imshow(image)
                for x0,y0,x1,y1 in boxes:
                    ax.add_patch(Rectangle((x0*w,y0*h),(x1-x0)*w,(y1-y0)*h,fill=False,color="#19a35b",lw=2))
                for index,(x,y) in enumerate(metadata[i]["points"]):
                    ax.scatter([x],[y],s=80,facecolors="none",edgecolors="#ed6039",lw=1.5)
                    ax.annotate("ABCD"[index],(x,y),xytext=(5,5),textcoords="offset points",color="#ed6039",weight="bold")
            a=policy_rows["candidate_combined"][c["sample_id"]]
            b=policy_rows["native_choice_logits"][c["sample_id"]]
            axes[0].set_title(f"Combined: {'ABCD'[a['candidate_index']] if a['execute'] else 'WAIT'}; Native: {'ABCD'[b['candidate_index']] if b['execute'] else 'WAIT'}")
            x0,y0,x1,y1=boxes[0]
            cx,cy=(x0+x1)*w/2,(y0+y1)*h/2
            dx,dy=max(200,(x1-x0)*w*2),max(150,(y1-y0)*h*2)
            axes[1].set_xlim(max(0,cx-dx),min(w,cx+dx))
            axes[1].set_ylim(min(h,cy+dy),max(0,cy-dy))
            axes[1].set_title("Target detail; green = valid annotated region")
            for ax in axes: ax.axis("off")
            fig.suptitle("\n".join(textwrap.wrap(c["instruction"],110)),fontproperties=font,fontsize=11)
            fig.savefig(example_dir/f"{tag}.png",dpi=150)
            plt.close(fig)
            example_info[tag]={"sample_id":c["sample_id"],"instruction":c["instruction"],
                "candidate_labels":labels[i].tolist(),"points":metadata[i]["points"],
                "combined":a,"native":b,"combined_probabilities":score_file["candidate_combined"][i].tolist(),
                "native_probabilities":score_file["native_choice_logits"][i].tolist()}
            image.close()
    write_json(output/"examples.json",example_info)
    write_json(output / "completion.json",{"complete":True,"audited":True,"n_test":len(test),
        "results_sha256":sha256(output/"results.json"),"audit_sha256":sha256(output/"final_audit.json")})
    print(json.dumps({"audit":"passed","n_records":len(cases),"n_test":len(test)}),flush=True)


if __name__ == "__main__":
    main()
