"""Posthoc explanatory audit of V27; no training, policy selection or VLM calls."""
import os
for k,v in {'PYTHONDONTWRITEBYTECODE':'1','OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'}.items():os.environ.setdefault(k,v)
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path
import json,sys,time
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import run_candidate_identity_v27 as run
from clickagent.candidate_identity_v27 import (retain,probe_features,infer,unit,PRIOR,VISUAL,PEER,CONTEXT)
from run_location_selection_v26 import geometry_bins
from run_visual_feedback_v23 import read,write,sha,hit

OUT=run.OUT/'why_analysis_20261109'
CONTROL=run.OUT/'scaling_control'


def now():return datetime.now(timezone.utc).isoformat()
def row_model(info):return run.prior.load_npz(info['path'])[0]
def outcome(scores,y):return bool(len(scores) and y[int(np.argmax(scores))])


def contexts(c,points):
    mask=np.zeros(len(points),np.int64);inverse={};descriptions=[]
    for vi,v in enumerate(c['views']):
        f=run.V25/f'raw/{run.prior.task_for(c,vi,"context")["task_id"]}.npz'
        with np.load(f,allow_pickle=False) as z:m=json.loads(str(z['metadata']))
        bounds=np.asarray([0,0,*c['image_size']] if v['transform'] is None else v['transform']['box'],float)
        if m['readout_stages']['container']:
            inside=((points>=bounds[:2])&(points<bounds[2:])).all(-1);mask[inside]|=1<<vi
        inverse[vi]=1/(bounds[2:]-bounds[:2])
        descriptions.append(dict(view=vi,text=m['generated_text'],fields=m['ground_fields'],grid=m['grid_shape']))
    return mask,inverse,descriptions


def raw_template_stats(c,prompt):
    if prompt=='A':f=Path(c['views'][0]['feature_path'])
    else:f=run.V25/f'raw/{run.prior.task_for(c,0,"context")["task_id"]}.npz'
    with np.load(f,allow_pickle=False) as z:
        m=json.loads(str(z['metadata']));vision=z['vision_hidden'].astype(np.float32)
        if prompt=='A':attention=z['target_attention'].astype(np.float32)
        else:attention=z['all_attention'][np.asarray(m['readout_stages']['target'],int)-1].astype(np.float32)
    if not len(attention):return dict(sample_id=c['sample_id'],prompt=prompt,available=False)
    gh,gw=m['grid_shape'];vf=unit(vision).reshape(2*gh,2*gw,-1)
    coarse=vf.reshape(gh,2,gw,2,-1).mean((1,3)).reshape(gh*gw,-1)
    a=attention/np.maximum(attention.sum(-1,keepdims=True),1e-12)
    proto=unit(a@coarse);uniform=unit(vf.reshape(-1,vision.shape[-1]).mean(0,keepdims=True))[0]
    cosine=proto@uniform;entropy=-(a*np.log(np.maximum(a,1e-30))).sum(-1)/np.log(gh*gw)
    return dict(sample_id=c['sample_id'],prompt=prompt,available=True,actual_tokens=len(attention),
        mean_template_cosine_to_uniform=float(cosine.mean()),head_template_cosine_quantiles=np.quantile(cosine,[.1,.5,.9]).tolist(),
        mean_normalized_attention_entropy=float(entropy.mean()),
        mean_fine_vector_cosine_to_image_mean=float((vf.reshape(-1,vision.shape[-1])@uniform).mean()))


def train_diagnosis(train,models,info,labels):
    rows=[];data={c['sample_id']:run.load(c) for c in train}
    for name in run.NAMES:
        for arm in run.ARMS:
            mm=models[arm+'__'+name];ii=info['models'][arm+'__'+name]
            rr=[run.record(c,data[c['sample_id']],name,arm,labels=labels) for c in train]
            rankable=[r for r in rr if r['y'].any() and (~r['y']).any()]
            pairs=sum(int(r['y'].sum()*(~r['y']).sum()) for r in rankable)
            base_hits=[outcome(r['base'],r['y']) for r in rr];fit_hits=[outcome(infer(mm,r),r['y']) for r in rr]
            selected=next(ch for ch in ii['choices'] if ch['ridge']==ii['ridge'])
            zero=next(ch for ch in ii['choices'] if ch['ridge'] is None)
            by_source={}
            for src in sorted({r['source'] for r in rr}):
                held=[r['hit'] for r in selected['held_out'] if r['source']==src]
                plain=[r['hit'] for r in zero['held_out'] if r['source']==src]
                by_source[src]=dict(n=len(held),OOF_hits=int(sum(held)),identity_hits=int(sum(plain)))
            rows.append(dict(model=name,arm=arm,chosen_ridge=ii['ridge'],queries=len(rr),rankable_queries=len(rankable),
                positive_negative_pairs=pairs,train_original_hits=int(sum(base_hits)),train_fit_hits=int(sum(fit_hits)),
                train_repairs=sum(a and not b for a,b in zip(fit_hits,base_hits)),train_harms=sum(b and not a for a,b in zip(fit_hits,base_hits)),
                OOF_identity_hits=sum(r['hit'] for r in zero['held_out']),OOF_selected_hits=sum(r['hit'] for r in selected['held_out']),
                by_source=by_source,weight_norms={k:float(np.linalg.norm(mm['weight'][sl])) for k,sl in [('prior',PRIOR),('visual',VISUAL),('peer',PEER),('container',CONTEXT)]}))
    return rows


def main():
    start=time.perf_counter();assert not (OUT/'plan.json').exists()
    parent_complete=read(run.OUT/'completion.json');control_complete=read(CONTROL/'completion.json')
    for complete in [parent_complete,control_complete]:
        for p,h in complete['artifact_sha256'].items():assert sha(p)==h,p
    write(OUT/'plan.json',dict(recorded_at_utc=now(),source_sha256=sha(Path(__file__)),
        purpose='Explain existing results with frozen models/caches; no training or selection of a new production method.',
        questions=['candidate construction vs retention vs ranking','two new-coverage case traces',
            'train fitting vs source-held-out transfer','linear anchor cancellation within shared views',
            'template similarity to uniform reference','posthoc K8/16/32/64 frozen-model sensitivity'],
        ground_truth='Annotations used for retrospective evaluation/attribution only. K sensitivity is diagnostic, not a newly validated method.',
        source_results_unchanged=True,new_vlm_calls=0))
    cases=read(run.V25/'data/public.json');dev=[c for c in cases if c['cohort']!='train'];train=[c for c in cases if c['cohort']=='train']
    labels=read(run.V25/'data/labels.json');info=read(CONTROL/'models/frozen.json')
    models={tag:row_model(row) for tag,row in info['models'].items()}
    old={(r['sample_id'],r['model'],r['arm']):r for r in read(run.OUT/'predictions.json')['rows']}
    pred={(r['sample_id'],r['model'],r['arm']):r for r in read(CONTROL/'predictions.json')['rows']}
    training=train_diagnosis(train,models,info,labels);write(OUT/'training.json',training)
    print('Training-fit/source-OOF diagnosis complete',flush=True)
    flow=[];budget=[];special=[];contributions=[];anchor_checks=[];templates=[];peers=[]
    for ci,c in enumerate(dev,1):
        a=run.load(c);sid=c['sample_id'];points=a['points'];truth=np.asarray([hit(p.tolist(),c,labels[sid]) for p in points],bool)
        A=a['ai']>=0;cmask,inverse,descriptions=contexts(c,points)
        np.testing.assert_array_equal(cmask>0,a['valid'][:,2])
        if c['cohort']=='new_development':templates.extend(raw_template_stats(c,p) for p in ['A','B'])
        is_special=c['cohort']=='new_development' and c['instruction'] in ['Latest','Sofascore']
        case=dict(sample_id=sid,instruction=c['instruction'],image=c['image'],image_size=c['image_size'],
            target_boxes_pixels=(np.asarray(labels[sid]['boxes'])*np.tile(c['image_size'],2)).tolist(),
            true_points=points[truth].tolist(),descriptions=descriptions,per_model=[])
        for name in run.NAMES:
            score=a['U__'+name];ix=retain(score,a['ai']);true_ix=ix[truth[ix]]
            q=pred[sid,name,'full'];Arow=old[sid,name,'A'];truerank={int(j):i+1 for i,j in enumerate(np.argsort(-score,kind='stable'))}
            bidx=np.flatnonzero(~A);border=bidx[np.argsort(-score[bidx],kind='stable')];brank={int(j):i+1 for i,j in enumerate(border)}
            ai=np.flatnonzero(A);aranks=ai[np.argsort(-score[ai],kind='stable')[:16]]
            flow.append(dict(sample_id=sid,cohort=c['cohort'],model=name,
                A_complete=bool(truth[A].any()),A16=bool(truth[aranks].any()),A_hit=Arow['hit'],
                union_complete=bool(truth.any()),retained=bool(len(true_ix)),correct=q['hit']))
            r=run.record(c,a,name,'full');mm=models['full__'+name]
            # A linear offset to a shared anchor cancels from pairwise ranking.
            for mask in sorted(set(cmask[ix].tolist())-{0}):
                group=ix[cmask[ix]==mask];used=[v for v in inverse if mask&(1<<v)]
                alpha=np.mean([inverse[v] for v in used],axis=0)
                for jj in range(1,len(group)):
                    i,j=group[0],group[jj];expected=(points[i]-points[j])*alpha
                    actual=a['geometry'][i,:48]-a['geometry'][j,:48]
                    error=max(float(np.abs(actual[:24]-expected[0]).max()),float(np.abs(actual[24:]-expected[1]).max()))
                    assert error<2e-6,error
                    anchor_checks.append(dict(sample_id=sid,model=name,view_mask=int(mask),max_error=error))
            for k in [8,16,32,64]:
                kk=retain(score,a['ai'],budget=k)
                x,b=probe_features(points,c['image_size'],score,a['ai'],a['bi'],a['role'],a['geometry'],a['valid'],kk,'full')
                ss=infer(mm,dict(x=x,base=b));j=int(kk[ss.argmax()]) if len(ss) else None
                if k==16:assert j==q['index']
                budget.append(dict(sample_id=sid,cohort=c['cohort'],model=name,K=k,
                    covered=bool(truth[kk].any()),hit=bool(j is not None and truth[j]),index=j))
            if is_special:
                selected=q['index'];row=dict(model=name,best_union_rank=min([truerank[int(j)] for j in np.flatnonzero(truth)],default=None),
                    best_B_only_rank=min([brank[int(j)] for j in np.flatnonzero(truth) if int(j) in brank],default=None),
                    positives_retained=len(true_ix),selected_point=q['point'],hit=q['hit'],K_results=[z for z in budget[-4:]])
                if selected is not None and len(true_ix):
                    j=int(true_ix[0]);li=int(np.flatnonzero(ix==selected)[0]);lj=int(np.flatnonzero(ix==j)[0])
                    d=(r['x'][li]-r['x'][lj])/mm['std']*mm['weight']
                    row['wrong_minus_true_score']=dict(frozen=float(r['base'][li]-r['base'][lj]),
                        **{k:float(d[sl].sum()) for k,sl in [('prior',PRIOR),('visual',VISUAL),('peer',PEER),
                           ('container_appearance',slice(202,250)),('container_dxdy',slice(250,298)),('container_distance',slice(298,322))]})
                    row['true_reference_point']=points[j].tolist()
                case['per_model'].append(row)
            if Arow['hit']!=q['hit'] and q['index'] is not None:
                reference=next((int(j) for j in ix if points[j].tolist()==Arow['point']),None)
                if reference is not None:
                    i=int(np.flatnonzero(ix==q['index'])[0]);j=int(np.flatnonzero(ix==reference)[0])
                    d=(r['x'][i]-r['x'][j])/mm['std']*mm['weight']
                    contributions.append(dict(sample_id=sid,cohort=c['cohort'],model=name,instruction=c['instruction'],
                        change='harm' if Arow['hit'] else 'repair',A_point=Arow['point'],new_point=q['point'],
                        frozen_delta=float(r['base'][i]-r['base'][j]),
                        groups={k:float(d[sl].sum()) for k,sl in [('prior',PRIOR),('visual',VISUAL),('peer',PEER),
                            ('container_appearance',slice(202,250)),('container_dxdy',slice(250,298)),('container_distance',slice(298,322))]}))
        if is_special:special.append(case)
        if ci%6==0:print(f'Deep trace {ci}/{len(dev)}',flush=True)
    write(OUT/'case_traces.json',special);write(OUT/'pipeline_rows.json',flow)
    write(OUT/'budget_sensitivity.json',budget);write(OUT/'score_contributions.json',contributions)
    write(OUT/'anchor_cancellation.json',dict(checks=len(anchor_checks),max_error=max((r['max_error'] for r in anchor_checks),default=0),rows=anchor_checks,
        scope='Only linear dx/dy components within equal active-container-view masks. Distance terms remain anchor-dependent.'))
    write(OUT/'template_stats.json',templates)
    summarize(flow,budget,templates,training,special,contributions,anchor_checks)
    print(f'Deep analysis finished in {time.perf_counter()-start:.1f}s',flush=True)


def summarize(flow,budget,templates,training,special,contributions,anchor_checks):
    summary={}
    for cohort in ['new_development','old_val']:
        summary[cohort]={}
        for family in ['original3k','full']:
            rr=[r for r in flow if r['cohort']==cohort and r['model'].startswith(family+'_')];n=len(rr)/3
            s={k:sum(r[k] for r in rr)/3 for k in ['A_complete','A16','A_hit','union_complete','retained','correct']}
            s.update(n=n,no_union_candidate=n-s['union_complete'],lost_to_retention=s['union_complete']-s['retained'],
                     ranking_misses=s['retained']-s['correct'])
            s['budget']={}
            for k in [8,16,32,64]:
                bb=[r for r in budget if r['cohort']==cohort and r['model'].startswith(family+'_') and r['K']==k]
                s['budget'][str(k)]=dict(coverage=100*float(np.mean([r['covered'] for r in bb])),accuracy=100*float(np.mean([r['hit'] for r in bb])))
            summary[cohort][family]=s
    ts={}
    for prompt in ['A','B']:
        rows=[r for r in templates if r['prompt']==prompt and r['available']]
        ts[prompt]={k:dict(mean=float(np.mean([r[k] for r in rows])),p10_p50_p90=np.quantile([r[k] for r in rows],[.1,.5,.9]).tolist())
            for k in ['mean_template_cosine_to_uniform','mean_normalized_attention_entropy','mean_fine_vector_cosine_to_image_mean']}
        ts[prompt]['n_rgb']=len(rows)
    write(OUT/'summary.json',dict(pipeline=summary,template_statistics=ts,
        linear_anchor_cancellation_checks=len(anchor_checks),linear_anchor_max_error=max((r['max_error'] for r in anchor_checks),default=0),
        trained_nonidentity_examples=[r for r in training if r['chosen_ridge'] is not None and r['arm'] in ['visual','full']],
        source_results_unchanged=True,new_vlm_calls=0,all_counterfactuals_are_posthoc_diagnostics=True))
    lines=['V27失败原因：事后机制核对；没有新训练/大模型调用，没有修改历史结果。',
        '管线分解（每种子均值）：',json.dumps(summary,ensure_ascii=False,indent=2),
        '全图注意力模板统计：',json.dumps(ts,ensure_ascii=False,indent=2),
        f'线性锚点dx/dy抵消数值验证：{len(anchor_checks)}对，最大误差{max((r["max_error"] for r in anchor_checks),default=0):.3g}。',
        '这只证明相同有效视图集合内的线性位移项退化为位置偏置；距离项仍含锚点信息。',
        '训练中非零视觉/完整模块：',json.dumps([r for r in training if r['chosen_ridge'] is not None and r['arm'] in ['visual','full']],ensure_ascii=False,indent=2)]
    (OUT/'raw_analysis.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    outputs=list(OUT.glob('*.json'))+[OUT/'raw_analysis.txt',Path(__file__)]
    write(OUT/'completion.json',dict(complete=True,completed_at_utc=now(),artifact_sha256={str(p):sha(p) for p in outputs},
        no_training=True,no_new_vlm_calls=True,original_results_unchanged=True))


if __name__=='__main__':main()
