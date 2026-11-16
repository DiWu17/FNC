"""Predeclared descriptive readouts; never fits or selects a prediction policy."""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'4')
import sys,json,re
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
from run_visual_feedback_v23 import read,write,sha,hit,bootstrap
import run_grounded_composition_v28 as run
OUT=run.OUT/'mechanisms'


def analyze():
    assert read(run.OUT/'completion.json')['complete']
    cases=read(run.OUT/'data/public.json');labels=read(run.OUT/'data/labels.json')
    p1=read(run.OUT/'predictions_stage1.json');p2=read(run.OUT/'predictions_stage2.json')
    lookup={(r['sample_id'],r['old_head'],r['arm']):r for r in p1['rows']+p2['rows']}
    old={(r['sample_id'],r['model']):r for r in read(run.V23/'predictions.json')['rows'] if r['arm']=='baseline'}
    parity=0;repr_rows=[];train_rows=[];traces=[];native=[]
    relation_pattern=re.compile(r'\b(above|below|between|left of|right of|under|beneath|inside|within|next to)\b',re.I)
    for c in cases:
        sid=c['sample_id'];a,_=run.load(c,'original');rep,_=run.load(c,'representation');g,gm=run.load(c,'grounded')
        truth=np.asarray([hit(p.tolist(),c,labels[sid]) for p in a['points']],bool);gt=np.asarray([hit(p.tolist(),c,labels[sid]) for p in g['points']],bool)
        pos=np.flatnonzero(truth);neg=np.flatnonzero(~truth);aucs={}
        if len(pos) and len(neg):
            distance=np.linalg.norm(a['points'][neg,None]/c['image_size']-a['points'][pos][None]/c['image_size'],axis=-1).min(1)
            neg=neg[np.argsort(distance,kind='stable')[:16]]
            for mode in run.MODES:
                # Fixed equal-head all-token mean score, with view availability.
                vv=rep[mode];ss=[]
                for side in [0,1]:ss.append(vv[:,side*576+384:side*576+480].mean(1))
                s=(np.stack(ss,1)*rep['valid']).sum(1)/np.maximum(rep['valid'].sum(1),1)
                d=s[pos,None]-s[neg][None];aucs[mode]=float(np.mean((d>0)+.5*(d==0)))
        repr_rows.append(dict(sample_id=sid,source=c['source'],cohort=c['cohort'],rankable=bool(len(pos) and len(neg)),near_false_AUC=aucs))
        origin_a=len(a['points']);correct_native=[part for part,p in gm['native_points'].items() if hit(p,c,labels[sid])]
        row=dict(sample_id=sid,source=c['source'],cohort=c['cohort'],training_group=c['training_group'],instruction=c['instruction'],
            A_complete_covered=any(hit(p.tolist(),c,labels[sid]) for p in a['complete_points']),A32_covered=bool(truth.any()),augmented_covered=bool(gt.any()),
            supplemental_correct=bool(gt[origin_a:].any()),correct_native_parts=correct_native,grounded_anchors=sum(x['valid'] for x in gm['anchors']),explicit_relation_instruction=bool(relation_pattern.search(c['instruction'])))
        if c['cohort']=='train':train_rows.append(row)
        else:
            for name in run.NAMES:
                assert lookup[sid,name,'A']['point']==old[sid,name]['point'],(sid,name);parity+=1
            row['targets']={k:gm[k] for k in ['global_target','free_target','locked_target']}
            row['anchors']=gm['anchors'];row['outcomes']={name:{arm:lookup[sid,name,arm] for arm in ['A','grounded','no_identity','no_relation','combined','targeted','combined_nohard']} for name in run.NAMES}
            row['native_points']=gm['native_points'];traces.append(row)
    write(OUT/'case_traces.json',traces);write(OUT/'training_supervision.json',train_rows);write(OUT/'representation_diagnostics.json',repr_rows)
    summaries={}
    for cohort in ['train','new_development','old_val']:
        rr=[r for r in repr_rows if r['cohort']==cohort]
        summaries[cohort]={m:dict(rankable_rgb=sum(m in r['near_false_AUC'] for r in rr),mean_AUC=float(np.mean([r['near_false_AUC'][m] for r in rr if m in r['near_false_AUC']])) if any(m in r['near_false_AUC'] for r in rr) else None) for m in run.MODES}
    training_summary={}
    for group in ['random','targeted','combined']:
        rr=[r for r in train_rows if group=='combined' or r['training_group']==group]
        training_summary[group]=dict(n=len(rr),A_missing=sum(not r['A_complete_covered'] for r in rr),native_rescues=sum(not r['A_complete_covered'] and r['augmented_covered'] for r in rr),supplemental_correct_queries=sum(r['supplemental_correct'] for r in rr),grounded_relation_queries=sum(r['grounded_anchors']>0 for r in rr),explicit_relation_queries=sum(r['explicit_relation_instruction'] for r in rr),augmented_rankable=sum(r['augmented_covered'] for r in rr))
    decomposition={}
    for family in ['original3k','full']:
        cc=[r for r in traces if r['cohort']=='new_development'];stats=[]
        for s in [1,2,3]:
            name=f'{family}_s{s}';miss=sum(not r['augmented_covered'] for r in cc)
            ranking=sum(r['augmented_covered'] and not r['outcomes'][name]['combined']['hit'] for r in cc)
            repair=sum(not r['outcomes'][name]['A']['hit'] and r['outcomes'][name]['combined']['hit'] for r in cc)
            harm=sum(r['outcomes'][name]['A']['hit'] and not r['outcomes'][name]['combined']['hit'] for r in cc)
            stats.append(dict(old_head=name,pool_missing=miss,reachable_but_not_selected=ranking,repairs=repair,harms=harm))
        decomposition[family]=stats
    # Historical official-MVP comparator on exactly the same reused screenshots;
    # it is descriptive and has a different observation/computation budget.
    mvp=read(ROOT/'runs/mvp_repair_v21/results.json')['cohorts']
    comparator={co:dict(n=mvp[co]['n'],MVP_percent=mvp[co]['methods']['mvp']['accuracy_percent'],V28=read(run.OUT/'results.json')[co]['summary']) for co in ['new_development','old_val']}
    result=dict(registered_descriptive_only=True,original_prediction_parity=parity,representation=summaries,training=training_summary,decomposition=decomposition,cached_MVP_comparison=comparator)
    write(OUT/'summary.json',result)
    lines=['V28 机制补充（描述性，未据此重新训练或选择方法）','A旧预测逐点复放一致: '+str(parity),'', '固定等权头均值对正确点/最近错误点AUC：',json.dumps(summaries,ensure_ascii=False,indent=2),'','实际训练纠错监督：',json.dumps(training_summary,ensure_ascii=False,indent=2),'','新72错误分解：',json.dumps(decomposition,ensure_ascii=False,indent=2),'','同截图历史MVP（不同成本、非独立确认）：']
    for co in ['new_development','old_val']:
        lines.append(co+' MVP='+str(comparator[co]['MVP_percent'])+'; combined3k='+str(comparator[co]['V28']['original3k']['combined']['percent'])+'; combinedfull='+str(comparator[co]['V28']['full']['combined']['percent']))
    lines+=['','这些AUC是固定等权证据的机制读数，不是部署准确率或按标注挑头后的成绩。名称差异可能来自同义词，不自动视为身份错误。','参照框没有单独的人类锚点标注，关系分支的正负结果不能完全分解成框定位误差或关系执行误差。']
    (OUT/'readout.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(OUT/'completion.json',dict(complete=True,source_sha256=sha(Path(__file__)),artifact_sha256={str(p):sha(p) for p in OUT.iterdir() if p.is_file() and p.name!='completion.json'}))
    print('\n'.join(lines),flush=True)


if __name__=='__main__':
    if '--register' in sys.argv:
        write(OUT/'plan.json',dict(source_sha256=sha(Path(__file__)),description='Fixed equal-head all-token-mean near-negative AUC; actual training repair modes; exact old-prediction replay; new72 pool/ranking/repair/harm decomposition; historical same-screenshot MVP descriptive comparator. No fitting or policy selection.'))
    else:
        assert read(OUT/'plan.json')['source_sha256']==sha(Path(__file__));analyze()
