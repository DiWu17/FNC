"""CPU-only audit, crop coverage and figures for the completed refinement run."""
from collections import Counter
import json
from pathlib import Path

import numpy as np

from analyze_public_failures import polygon_rect_area
from clickagent.action_benchmark import sha256, stable_hash, write_json
from clickagent.public_benchmark import cache_key, read_jsonl
from clickagent.public_refinement import resolve_point


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "runs/public_refinement_v1"


def intersects_crop(geometry, box, size):
    w, h = size
    x0, y0, x1, y1 = np.asarray(box) / [w, h, w, h]
    if geometry["type"] == "bbox":
        return any(min(x1, b[2]) >= max(x0, b[0]) and min(y1, b[3]) >= max(y0, b[1])
                   for b in geometry["boxes"])
    return any(polygon_rect_area(p, [x0, y0, x1, y1]) > 1e-16 for p in geometry["polygons"])


def render_example(case, result, path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Polygon
    from PIL import Image
    w, h = case["image_size"]
    box = result["model_input"]["crop_box"]
    with Image.open(case["image"]) as opened:
        screenshot = opened.convert("RGB")
        crop = screenshot.crop(box)
        screenshot.thumbnail((1400, 900))
        before, after = result["model_input"]["coarse_point"], result["point"]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
        axes[0].imshow(np.asarray(screenshot), extent=[0, w, h, 0])
        axes[1].imshow(np.asarray(crop), extent=[box[0], box[2], box[3], box[1]])
        axes[0].add_patch(Rectangle(box[:2], box[2]-box[0], box[3]-box[1], fill=False, edgecolor="orange", lw=2))
        for ax in axes:
            geometry = case["target_geometry"]
            if geometry["type"] == "bbox":
                for x0,y0,x1,y1 in geometry["boxes"]:
                    ax.add_patch(Rectangle((x0*w,y0*h),(x1-x0)*w,(y1-y0)*h,fill=False,edgecolor="lime",lw=2))
            else:
                for poly in geometry["polygons"]:
                    ax.add_patch(Polygon(np.asarray(poly)*[w,h],fill=False,edgecolor="lime",lw=2))
            ax.plot(*before,marker="x",color="red",ms=10,mew=2,label="Old center")
            ax.plot(*after,marker="o",color="deepskyblue",ms=7,mfc="none",mew=2,label="Refined point")
        axes[0].set_xlim(0,w)
        axes[0].set_ylim(h,0)
        axes[1].set_xlim(box[0],box[2])
        axes[1].set_ylim(box[3],box[1])
        axes[0].set_title("Original screenshot and fixed attention crop")
        axes[1].set_title("Crop: green = GT; red = old; blue = new")
        for ax in axes:
            ax.set_axis_off()
        fig.suptitle(case["sample_id"] + "\n" + case["instruction"],fontsize=10)
        fig.savefig(path,dpi=150)
        plt.close(fig)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    summary = json.loads((OUT / "summary.json").read_text(encoding="utf-8"))
    identity = json.loads((OUT / "identity.json").read_text(encoding="utf-8"))
    assert summary["complete"] and summary["n_cases"] == 800
    cases = read_jsonl(OUT / "selected_manifest.jsonl")
    model_inputs=read_jsonl(OUT/"model_inputs.jsonl")
    assert stable_hash(model_inputs)==identity["input_digest"]
    inputs_by_id={r["sample_id"]:r for r in model_inputs}
    paired = {r["sample_id"]: r for r in read_jsonl(OUT / "paired_localization.jsonl")}
    coverage = {}
    entries, results = [], {}
    for case in cases:
        result = json.loads((OUT / "rows" / (cache_key(case["sample_id"]) + ".json")).read_text(encoding="utf-8"))
        results[case["sample_id"]] = result
        assert result["model_input"]==inputs_by_id[case["sample_id"]]
        assert result["fingerprint"]==stable_hash(identity)
        point,status=resolve_point(result["native"],result["model_input"]["crop_box"],result["model_input"]["coarse_point"])
        assert point==result["point"] and status==result["refinement_status"]
        assert result["model_input"]["image_sha256"] == sha256(case["image"])
        left,top,right,bottom=result["model_input"]["crop_box"]
        assert left <= result["point"][0] < right and top <= result["point"][1] < bottom
        if case["sample_id"] not in paired:
            continue
        row = paired[case["sample_id"]]
        # A valid point shared by crop and GT also proves boundary intersection.
        overlap = intersects_crop(case["target_geometry"], result["model_input"]["crop_box"], case["image_size"]) or row["after_hit"]
        assert not row["after_hit"] or overlap
        entries.append({**row, "target_intersects_crop": overlap})
    for name in sorted(summary["datasets"]):
        rows = [r for r in entries if r["dataset"] == name]
        coverage[name] = {"positive_denominator":len(rows), "target_intersects_crop":sum(r["target_intersects_crop"] for r in rows),
                          "hits_given_target_in_crop":sum(r["after_hit"] and r["target_intersects_crop"] for r in rows),
                          "target_missing_from_crop":sum(not r["target_intersects_crop"] for r in rows)}
    predictions = read_jsonl(OUT / "predictions.jsonl")
    cells = {(r["sample_id"], r["method"]):r for r in predictions}
    assert len(cells)==len(predictions)==11200
    action_mismatch = threshold_mismatch = probability_mismatch = 0
    for (sid,method), row in cells.items():
        if method.startswith("center__"):
            other=cells[sid,method.replace("center__","refined__",1)]
            action_mismatch += row["action"]["type"] != other["action"]["type"]
            threshold_mismatch += row.get("gate_threshold") != other.get("gate_threshold")
            probability_mismatch += row.get("click_probability") != other.get("click_probability")
    assert not (action_mismatch or threshold_mismatch or probability_mismatch)
    current = ROOT / "src/clickagent"
    assert all(sha256(current/name)==digest for name,digest in identity["source_sha256"].items())
    snapshot_matches=all(sha256(OUT/"frozen_source"/name)==digest for name,digest in identity["source_sha256"].items())
    assert snapshot_matches
    baseline_matches=all(sha256(ROOT/"runs/public_v1"/name/file)==digest
                         for name,files in identity["source_runs"].items() for file,digest in files.items())
    assert baseline_matches
    from clickagent.model_features import _local_model_identity
    assert _local_model_identity(identity["config"]["model_path"])==identity["model"]
    runtime=json.loads((OUT/"runtime.json").read_text(encoding="utf-8"))["runtime_metadata"]
    assert runtime["all_parameters_frozen"] and runtime["trainable_parameter_count"]==0 and not runtime["model_training"]
    audit = {"complete":True,"n_cases":len(cases),"n_geometry_positives":len(paired),"n_predictions":len(predictions),
             "image_hashes_checked":len(cases),"gate_action_mismatches":action_mismatch,
             "threshold_mismatches":threshold_mismatch,"probability_mismatches":probability_mismatch,
             "source_hashes_match":True,"frozen_source_snapshot_matches":snapshot_matches,
             "baseline_source_hashes_match":baseline_matches,"model_identity_matches":True,
             "all_model_parameters_frozen":True,"point_mapping_replayed":len(results),"crop_coverage":coverage,
             "native_generated_token_counts":dict(Counter(r["native"]["generated_tokens"] for r in results.values()))}
    write_json(OUT / "final_audit.json",audit)
    names = ["ScreenSpot-v2","ScreenSpot-Pro","OSWorld-G","VenusBench-GD"]
    old = [summary["datasets"][n]["before_accuracy"]*100 for n in names]
    new = [summary["datasets"][n]["after_accuracy"]*100 for n in names]
    x=np.arange(len(names))
    fig,ax=plt.subplots(figsize=(9,4.5),constrained_layout=True)
    for offset,values,label,color in [(-.19,old,"Old attention patch center","#8095a5"),(.19,new,"Same proposal + native crop refinement","#147c90")]:
        bars=ax.bar(x+offset,values,width=.36,label=label,color=color)
        ax.bar_label(bars,labels=[f"{v:.1f}%" for v in values],padding=4)
    ax.set_xticks(x,names)
    ax.set_ylabel("Point inside GT region (%)")
    ax.set_ylim(0,max(new+old)+17)
    ax.legend(loc="upper left",fontsize=9)
    ax.set_title("Single-component ablation on identical public_v1 samples")
    ax.spines[["top","right"]].set_visible(False)
    fig.savefig(OUT/"comparison.png",dpi=180)
    fig.savefig(OUT/"comparison.pdf")
    plt.close(fig)
    by_sid={c["sample_id"]:c for c in cases}
    for label,predicate in (("rescued_small_target",lambda r:not r["center_oracle"] and r["after_hit"]),
                            ("regression",lambda r:r["before_hit"] and not r["after_hit"])):
        options=sorted((r for r in entries if predicate(r)),key=lambda r:r["sample_id"])
        if options:
            chosen=options[0]
            sid=chosen["sample_id"]
            render_example(by_sid[sid],results[sid],OUT/(label+".png"))
    print(json.dumps({"datasets":summary["datasets"],"audit":audit},ensure_ascii=False,indent=2))


if __name__=="__main__":
    main()
