"""Post-scoring diagnostics; do not select thresholds or alter predictions."""
import argparse
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score,roc_curve

from clickagent.action_benchmark import write_json
from clickagent.candidate_probes import load_features,evaluation_test_indices

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output",type=Path,default=Path("runs/candidate_v2"))
args=parser.parse_args()
cases,metadata,arrays=load_features(args.output)
test,_=evaluation_test_indices(cases,args.output)
click=np.asarray([i for i in test if cases[i]["case_kind"]=="click"])
y=arrays["labels"][click,0].astype(bool)
positive,negative=np.flatnonzero(y),np.flatnonzero(~y)
rng=np.random.default_rng(20261030)
wp=rng.multinomial(len(positive),np.full(len(positive),1/len(positive)),size=4000)
wn=rng.multinomial(len(negative),np.full(len(negative),1/len(negative)),size=4000)
result={"scope":"read-only post-scoring diagnostics; first proposal fixed; click requests only",
        "n_click_requests":len(click),"n_good_first_proposals":int(y.sum()),"n_bad_first_proposals":int((~y).sum()),
        "interval":"paired stratified screenshot bootstrap, conditional on positive/negative counts; 4000 draws",
        "methods":{}}
draws={}
with np.load(args.output/"scores.npz",allow_pickle=False) as scores:
    for method in scores.files:
        s=scores[method][click,0]
        d=s[positive,None]-s[negative][None,:]
        concordance=(d>0)+.5*(d==0)
        draws[method]=((wp@concordance)*wn).sum(-1)/(len(positive)*len(negative))
        result["methods"][method]={"first_proposal_click_only_auroc":float(roc_auc_score(y,s)),
            "bootstrap95":np.percentile(draws[method],[2.5,97.5]).tolist()}
    result["paired_auroc_differences"]={}
    for reference in ("native_choice_logits","query_only_selector"):
        delta=draws["candidate_hidden"]-draws[reference]
        result["paired_auroc_differences"][f"candidate_hidden-minus-{reference}"]={
            "delta":result["methods"]["candidate_hidden"]["first_proposal_click_only_auroc"]-
                    result["methods"][reference]["first_proposal_click_only_auroc"],
            "bootstrap95":np.percentile(delta,[2.5,97.5]).tolist()}
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6.5,5),layout="constrained")
    for method in ("native_choice_logits","candidate_attention","candidate_hidden","candidate_combined","query_only_selector"):
        fpr,tpr,_=roc_curve(y,scores[method][click,0])
        auc=result["methods"][method]["first_proposal_click_only_auroc"]
        ax.plot(fpr,tpr,label=f"{method.replace('_',' ')} ({auc:.3f})")
    ax.plot([0,1],[0,1],"--",color="gray",lw=.8)
    ax.set(xlabel="False acceptance rate on wrong first proposals",ylabel="Recall of correct first proposals",
           title="Fixed first proposal: verification discrimination\n111 click requests; curve is diagnostic, not a deployable threshold")
    ax.grid(alpha=.2)
    ax.legend(fontsize=8,loc="lower right")
    fig.savefig(args.output/"first_proposal_roc.png",dpi=170)
    fig.savefig(args.output/"first_proposal_roc.pdf")
    plt.close(fig)
write_json(args.output/"verification_diagnostics.json",result)
print(result["paired_auroc_differences"],flush=True)
