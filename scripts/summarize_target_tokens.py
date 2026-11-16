"""Inspect generated target phrases and visualize fixed-decoder timing checks."""
import argparse
import json
from pathlib import Path

import numpy as np

from run_target_token_experiment import read_json,read_jsonl,key,overlap_cells,STAGES
from clickagent.action_benchmark import write_json


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("runs/target_token_v1"))
    args=parser.parse_args()
    output=args.output
    cases=read_jsonl(output/"manifest.jsonl")
    by_id={c["sample_id"]:c for c in cases}
    predictions=read_jsonl(output/"predictions.jsonl")
    lookup={(r["sample_id"],r["stage"],r["readout"]):r for r in predictions}
    scored={r["sample_id"] for r in predictions}
    metadata,metrics={},{}
    for case in cases:
        if case["sample_id"] not in scored:continue
        path=output/"features"/f"{key(case['sample_id'])}.npz"
        if not path.exists():continue
        with np.load(path,allow_pickle=False) as z:
            raw=z["raw_attention"]
            meta=json.loads(str(z["metadata"]))
        assert len(raw)==len(meta["generated_token_ids"])+1
        assert meta["query_token_indices"]==list(range(meta["input_tokens"]-1,meta["input_tokens"]+len(meta["generated_token_ids"])))
        metadata[case["sample_id"]]=meta
        mask=overlap_cells(case["target_geometry"],meta["grid_shape"])
        for stage,indices in meta["stages"].items():
            heat=raw[indices,:32].mean((0,1))
            metrics[(case["sample_id"],stage)]={"image_mass":float(heat.sum()),
                "relative_target_mass":float(heat[mask].sum()/heat.sum()) if heat.sum()>0 else 0.}
    named=[sid for sid,m in metadata.items() if m["target_available"]]
    focus={}
    for stage in STAGES:
        valid=[sid for sid in named if (sid,stage) in metrics]
        if valid:
            before=np.asarray([metrics[(sid,"prefill")]["relative_target_mass"] for sid in valid])
            after=np.asarray([metrics[(sid,stage)]["relative_target_mass"] for sid in valid])
            focus[stage]={"n":len(valid),"mean_relative_target_mass":float(after.mean()),
                "paired_mean_change_from_prefill":float((after-before).mean()),
                "cases_with_increased_target_mass":int((after>before).sum()),
                "mean_total_image_mass":float(np.mean([metrics[(sid,stage)]["image_mass"] for sid in valid]))}
    examples=[]
    for kind in ("gain","loss","cell_gain"):
        for sid in named:
            a=lookup[(sid,"prefill","layer15_mean")]
            b=lookup[(sid,"target_mean","layer15_mean")]
            good=((b["hit"] and not a["hit"]) if kind=="gain" else
                  (a["hit"] and not b["hit"]) if kind=="loss" else
                  (b["peak_cell_overlaps_target"] and not a["peak_cell_overlaps_target"]))
            if good and sid not in [r["sample_id"] for r in examples]:
                examples.append({"kind":kind,"sample_id":sid})
                break
    if not examples and named:
        examples=[{"kind":"first_named_case","sample_id":named[0]}]
    write_json(output/"temporal_diagnostics.json",{"n":len(metadata),"named_cases":len(named),
        "layer15_relative_attention_on_same_named_cases":focus,"illustrative_examples":examples,
        "note":"Example selection is outcome-based for illustration only. Full aggregate results include all cases."})
    (output/"generated_targets.jsonl").write_text("".join(json.dumps({"sample_id":sid,"instruction":by_id[sid]["instruction"],
        "generated_text":m["generated_text"],"target_text":m["target_text"],"stages":m["stages"]},ensure_ascii=False)+"\n" for sid,m in metadata.items()),encoding="utf-8")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    from matplotlib.patches import Rectangle,Polygon
    plt.rcParams["font.family"]=["Microsoft YaHei","DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"]=False
    (output/"examples").mkdir(exist_ok=True)
    for example in examples:
        sid=example["sample_id"]
        case,meta=by_id[sid],metadata[sid]
        with np.load(output/"features"/f"{key(sid)}.npz",allow_pickle=False) as z:
            raw=z["raw_attention"]
        with Image.open(case["image"]) as opened:
            image=opened.convert("RGB")
        w,h=image.size
        fig,axes=plt.subplots(2,2,figsize=(12,8))
        shown=("prefill","action_word","target_marker","target_mean")
        relative={}
        for stage in shown:
            if stage in meta["stages"]:
                heat=raw[meta["stages"][stage],:32].mean((0,1)).reshape(meta["grid_shape"])
                relative[stage]=heat/heat.sum() if heat.sum()>0 else np.zeros_like(heat)
        vmax=max(float(v.max()) for v in relative.values())
        for ax,stage in zip(axes.ravel(),shown):
            ax.imshow(image)
            if stage in meta["stages"]:
                ax.imshow(relative[stage],extent=(0,w,h,0),cmap="magma",alpha=.60,
                    interpolation="nearest",vmin=0,vmax=vmax)
                r=lookup[(sid,stage,"layer15_mean")]
                ax.scatter(*r["point"],c="cyan",marker="+",s=90,linewidths=2)
                ax.set_title(f"{stage}: point_hit={r['hit']}, cell_overlap={r['peak_cell_overlaps_target']}")
            else:ax.set_title(stage+": unavailable")
            g=case["target_geometry"]
            if g["type"]=="bbox":
                for a,b,c,d in g["boxes"]:ax.add_patch(Rectangle((a*w,b*h),(c-a)*w,(d-b)*h,fill=False,edgecolor="lime",linewidth=2))
            else:
                for poly in g["polygons"]:ax.add_patch(Polygon(np.asarray(poly)*[w,h],fill=False,edgecolor="lime",linewidth=2))
            ax.set_xlim(0,w);ax.set_ylim(h,0);ax.axis("off")
        fig.suptitle(f"Instruction: {case['instruction']}\nGenerated target: {meta['target_text']} | layer 15 mean; green=GT, cyan=unchanged grid-center decoder",fontsize=10)
        fig.tight_layout()
        fig.savefig(output/"examples"/f"{example['kind']}.png",dpi=150)
        plt.close(fig)
        mask=overlap_cells(case["target_geometry"],meta["grid_shape"])
        heats=raw[:,:32].mean(1)
        values=np.divide(heats[:,mask].sum(-1),heats.sum(-1),out=np.zeros(len(heats)),where=heats.sum(-1)>0)
        fig,ax=plt.subplots(figsize=(12,4))
        ax.plot(np.arange(len(values)),values,"o-",label="relative image attention on target-intersecting cells")
        for j in meta["stages"].get("target_mean",[]):ax.axvspan(j-.45,j+.45,color="green",alpha=.12)
        labels=["prefill"]+[p[len(meta["generated_prefixes"][i-1]) if i else 0:] for i,p in enumerate(meta["generated_prefixes"])]
        ax.set_xticks(np.arange(len(values)),[s.replace("\n","\\n") for s in labels],rotation=65,ha="right",fontsize=8)
        ax.set_ylabel("Attention fraction within image");ax.set_ylim(bottom=0);ax.legend(fontsize=8)
        ax.set_title("Green shading: generated target-name own-token rows")
        fig.tight_layout();fig.savefig(output/"examples"/f"{example['kind']}_timeline.png",dpi=150);plt.close(fig)
    print(json.dumps({"n":len(metadata),"named":len(named),"focus":focus},ensure_ascii=False),flush=True)


if __name__=="__main__":main()
