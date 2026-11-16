"""Training-only source/app-group held-out diagnostics; no policy selection."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'4')
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
from run_visual_feedback_v23 import read,write,sha
import run_grounded_composition_v28 as run
from clickagent.grounded_composition_v28 import fit,infer
OUT=run.OUT/'training_transfer'

def main():
    assert read(OUT/'plan.json')['source_sha256']==sha(Path(__file__))
    assert read(run.OUT/'models_stage2/frozen.json')['complete']
    cases=[c for c in read(run.OUT/'data/public.json') if c['cohort']=='train'];labels=read(run.OUT/'data/labels.json')
    public={c['sample_id']:c for c in cases};rows=[]
    sources=sorted({c['source'] for c in cases})
    # All same application/session source_group observations stay together.
    groups={src:sorted({str(c['source_group']) for c in cases if c['source']==src},key=lambda g:run.digest([g,'v28_app_fold'])) for src in sources}
    assignment={(src,g):j%3 for src,gg in groups.items() for j,g in enumerate(gg)}
    for arm in ['grounded','targeted','combined']:
        cc=[c for c in cases if arm=='combined' or c['training_group']==('targeted' if arm=='targeted' else 'random')]
        for name in run.NAMES:
            rr=[run.record(c,name,arm,labels) for c in cc]
            for mode in ['source','source_group']:
                held=[];optimizers=[]
                for exclusion in (sources if mode=='source' else range(3)):
                    def excluded(r):
                        c=public[r['sample_id']]
                        return c['source']==exclusion if mode=='source' else assignment[c['source'],str(c['source_group'])]==exclusion
                    fitting=[r for r in rr if not excluded(r)];testing=[r for r in rr if excluded(r)]
                    if not fitting or not testing:continue
                    model=fit(fitting,.3,True);optimizers.append(dict(exclusion=exclusion,rankable=model['rankable_queries'],success=model['success']))
                    for r in testing:
                        h=bool(r['y'][int(infer(model,r).argmax())]) if len(r['x']) else False
                        b=bool(r['y'][int(r['base'].argmax())]) if len(r['base']) else False
                        held.append(dict(sample_id=r['sample_id'],source=r['source'],hit=h,base_hit=b))
                row=dict(arm=arm,old_head=name,mode=mode,new_training_rgb=len(cc),held_out=held,optimizers=optimizers,hits=sum(x['hit'] for x in held),base_hits=sum(x['base_hit'] for x in held),n=len(held))
                rows.append(row);write(OUT/'progress.json',dict(completed=len(rows),total=36,last={k:row[k] for k in ['arm','old_head','mode','hits','base_hits','n']}));print({k:row[k] for k in ['arm','old_head','mode','hits','base_hits','n']},flush=True)
    summaries={}
    for family in ['original3k','full']:
        summaries[family]={}
        for arm in ['grounded','targeted','combined']:
            summaries[family][arm]={}
            for mode in ['source','source_group']:
                rr=[r for r in rows if r['arm']==arm and r['mode']==mode and r['old_head'].startswith(family+'_')]
                summaries[family][arm][mode]=dict(percent=float(np.mean([100*r['hits']/r['n'] for r in rr])),neutral_prior_percent=float(np.mean([100*r['base_hits']/r['n'] for r in rr])),seed_hits=[r['hits'] for r in rr],n=rr[0]['n'])
    write(OUT/'results.json',dict(rows=rows,summaries=summaries,selection_performed=False))
    (OUT/'readout.txt').write_text('训练内来源/应用分组留出：仅隔离新增残差拟合，旧主干和旧评分器已见训练集，不能当整个系统的跨来源独立泛化。\n未据此选择、重训或更换部署分支。\n'+json.dumps(summaries,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    write(OUT/'completion.json',dict(complete=True,source_sha256=sha(Path(__file__)),artifact_sha256={str(p):sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='completion.json'}))

if __name__=='__main__':
    if '--register' in sys.argv:
        write(OUT/'plan.json',dict(source_sha256=sha(Path(__file__)),description='Training112 only. Fixed ridge.3. No selection. Each6 old heads x random56/targeted56/combined112: sourceLOSO7 and deterministic source_group3fold. Report residual and neutral transported-prior top1 hits. Groups held together. This diagnoses new residual transfer only, not unseen-source whole-system generalization. Registered before fitting stage2 or any new development call.'))
    else:main()
