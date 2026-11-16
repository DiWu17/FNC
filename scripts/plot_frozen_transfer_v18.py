"""Static scientific readout of frozen-policy effects across separate cohorts."""
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import numpy as np
from run_frozen_alternatives_v18 import comparison

OUT=ROOT/'runs/frozen_mvp_diagnostic_v18/analysis'


def read(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    panels=[];inputs=[Path(__file__)]
    for cohort,title in [('old_val','Old development (172 queries)'),('new_development','New development (300 queries / 206 RGB groups)')]:
        paths=[ROOT/f'runs/frozen_geometry_v18/evaluation/{cohort}.json',
               ROOT/f'runs/coordinate_control_v18/evaluation/{cohort}.json',
               ROOT/f'runs/two_decoder_banks_v18/evaluation/{cohort}.json']
        inputs+=paths;geo,fixed,banks=map(read,paths)
        rows=[geo['comparisons']['coarse_center'],fixed['contrasts']['coarse_center']['coordinate_only_vs_original'],
              banks['comparisons']['vs_original']]
        panels.append(dict(title=title,baseline_percent=geo['methods']['original']['mean_accuracy_percent'],rows=rows,
                           scope='Reused development only'))
    paths=[ROOT/'runs/frozen_mvp_diagnostic_v18/results.json',OUT/'two_banks_control.json',
           ROOT/'runs/contribution_v16/independent/index.json']
    inputs+=paths;diagnostic,banks,index=map(read,paths)
    rows=[diagnostic['decoding_vs_full_union']['center_rescore'],diagnostic['decoding_vs_full_union']['coordinate_center'],
          comparison(banks['hit_vectors'],diagnostic['hit_vectors']['full_union'],index['cases'])]
    panels.append(dict(title='Reused benchmark diagnostic (300 screenshots)',
                       baseline_percent=diagnostic['methods']['full_union']['accuracy_percent'],rows=rows,
                       scope='Previously used V16/V17 cohort; not new independent confirmation'))
    labels=['Center + feature rebuild / rescore','Original selection, center coordinate','Two decoder banks, max score']
    OUT.mkdir(parents=True,exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR',str(OUT/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14,4.5),sharey=True,layout='constrained')
    colors=['#2878a6','#4c9266','#9673a6']
    for ax,panel in zip(axes,panels):
        for i,(row,color) in enumerate(zip(panel['rows'],colors)):
            delta=row['delta_pp'];lo,hi=row['ci95_pp']
            ax.errorbar(delta,i,xerr=[[delta-lo],[hi-delta]],fmt='o',color=color,capsize=4)
        ax.axvline(0,color='#777777',lw=1)
        ax.set_xlim(-6,10);ax.set_ylim(2.6,-.6);ax.set_xticks([-4,0,4,8]);ax.grid(axis='x',alpha=.2)
        ax.set_title(panel['title']+f'\nFull-union baseline: {panel["baseline_percent"]:.2f}%',fontsize=10)
        ax.set_xlabel('Change vs own full-union baseline (pp)')
    axes[0].set_yticks(range(3),labels)
    fig.suptitle('Frozen 3k scorers: decoding gains depend on the evaluated cohort\nFixed three-seed mean; descriptive paired 95% intervals; all cohorts previously used',fontsize=12)
    path=OUT/'transfer_diagnostic.png';fig.savefig(path,dpi=180);plt.close(fig)
    summary=dict(complete=True,method_labels=labels,panels=panels,all_cohorts_reused=True,
                 new_independent_280_used=False,no_policy_change=True)
    p=OUT/'transfer_summary.json';p.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    audit=dict(complete=True,input_sha256={str(p):sha(p) for p in inputs},
               artifacts_sha256={str(p):sha(p) for p in [path,OUT/'transfer_summary.json']})
    (OUT/'transfer_plot_verification.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(path,flush=True)


if __name__=='__main__':main()
