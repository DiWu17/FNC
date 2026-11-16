"""Score fixed-pool verifier comparisons and independent candidate coverage."""
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np
import torch
from clickagent.action_benchmark import sha256,write_json
from clickagent.point_critic import FrozenPointEncoder,load_point_critic
from clickagent.point_aligned_readout import load_aligned
from clickagent.point_aligned_inference import map_crop_point
from evaluate_conditioned_readouts import score_point
from evaluate_phrase_readouts import paired_interval
from point_aligned_data import read,archive_record
from run_target_token_experiment import key,read_jsonl
from train_context_point_critic import load_arrays

OUT=Path('runs/context_verifier_v9')


def get_state(encoder,path,size):
    record=archive_record(path,dict(image_size=size),labels=False)
    return encoder.prepare(load_arrays(path),record['metadata']['grid_shape'],size)


def best_point(pool,scores):
    return pool[int(np.argmax(scores))]['point'] if pool else None


def critic_input(encoder,state,points):
    # Training/validation caches store FP16 features. Apply identical rounding
    # online before the FP32 head, rather than introducing a preprocessing shift.
    features=encoder.features(state,points).astype(np.float16).astype(np.float32)
    assert np.isfinite(features).all()
    return torch.from_numpy(features)


def cross_view_score(encoder,critic,global_state,local_states,pool):
    """Cross-view support, excluding each candidate's own generating view."""
    if not pool:return [],[]
    points=np.asarray([v['point'] for v in pool]);global_values=critic(critic_input(encoder,global_state,points)).detach().numpy()
    fallback=(global_values-global_values.mean())/max(float(global_values.std()),1e-6)
    evidence=[[] for _ in pool];view_records=[]
    views=[('global',global_state,None)]+local_states
    for name,state,transform in views:
        if state is None:continue
        if transform is None:inside=np.ones(len(pool),bool);local=points.copy()
        else:
            b=transform['box'];inside=(points[:,0]>=b[0])&(points[:,0]<b[2])&(points[:,1]>=b[1])&(points[:,1]<b[3])
            local=np.rint((points-np.asarray(b[:2]))*transform['scale']).astype(int)
        selected=np.flatnonzero(inside)
        if len(selected)<2:continue
        scores=critic(critic_input(encoder,state,local[selected])).detach().numpy()
        standardized=(scores-scores.mean())/max(float(scores.std()),1e-6)
        for i,value in zip(selected,standardized):
            if pool[i]['source']!=name:evidence[i].append(float(value))
        view_records.append(dict(view=name,indices=selected.tolist(),scores=scores.tolist()))
    return [float(np.mean(v)) if v else float(fallback[i]) for i,v in enumerate(evidence)],view_records


@torch.inference_mode()
def score(root,cases,with_ui):
    encoder=FrozenPointEncoder(load_aligned('runs/point_aligned_v8/models/joint_connected_s1.pt','cpu')[0])
    critics={s:load_point_critic(OUT/'models'/f'point_critic_{s}.pt')[0] for s in ['s1','s2']}
    rows=[];details=[];crop_details=[]
    for c in cases:
        sid=key(c['sample_id']);p=read(root/'pipeline'/f'{sid}.json');state=get_state(encoder,root/'features'/f'{sid}.npz',c['image_size'])
        predictions={'global_readout':p['global_result']['point'],
            'old_crop_readout':p['old_crop_candidates'][0]['point'] if p['old_crop_candidates'] else None}
        for i in range(2):
            local=p['context_locals'][i] if len(p['context_locals'])>i else None
            predictions[f'context_roi{i}_readout']=local['candidates'][0]['point'] if local and local['candidates'] else None
        native=read(root/'native'/f'{sid}.json');predictions['qwen_native_completed_lax']=native['completed_lax_point']
        predictions['qwen_native_strict']=native['point']
        if with_ui:predictions['ui_tars']=read(root/'ui_tars'/f'{sid}.json')['point']
        gui=read(root/'gui_verifier'/f'{sid}.json')
        view_states=[]
        for i,local in enumerate(p['context_locals']):
            size=local['transform']['crop_size'];ls=get_state(encoder,root/f'context_features_{i}'/f'{sid}.npz',size)
            view_states.append((f'context_{i}',ls,local['transform']))
        item=dict(sample_id=c['sample_id'],dataset=c['dataset'],pools={})
        for name,pool in p['pools'].items():
            points=[v['point'] for v in pool]
            labels=[bool(score_point(v,c)) for v in points]
            yn=[p['verification'][','.join(map(str,v))]['score'] for v in points]
            predictions[name+'__yes_no']=best_point(pool,yn)
            yn_odds=[p['verification'][','.join(map(str,v))]['yes_logit']-p['verification'][','.join(map(str,v))]['no_logit'] for v in points]
            predictions[name+'__yes_no_logodds']=best_point(pool,yn_odds)
            comp=p['comparisons'][name]
            predictions[name+'__comparative']=comp['symmetric']['point']
            predictions[name+'__comparative_forced']=comp['symmetric']['forced_point']
            predictions[name+'__comparative_forward']=comp['forward']['point']
            critic_scores={}
            for suffix,critic in critics.items():
                values=critic(critic_input(encoder,state,points)).numpy().tolist() if pool else []
                critic_scores[suffix]=values;predictions[name+'__point_critic_'+suffix]=best_point(pool,values)
            cross,view_records=cross_view_score(encoder,critics['s1'],state,view_states,pool)
            predictions[name+'__cross_view_critic_s1']=best_point(pool,cross)
            gui_scores={}
            for mode in ['crop','full']:
                lookup={tuple(v['point']):v['log_odds'] for v in gui['scores'][mode]}
                values=[lookup[tuple(v)] for v in points];gui_scores[mode]=values
                predictions[name+'__gui_actor_'+mode]=best_point(pool,values)
            item['pools'][name]=dict(points=points,point_hits=labels,point_oracle=any(labels),yes_no=yn,
                yes_no_logodds=yn_odds,critic_scores=critic_scores,cross_view_scores=cross,cross_view_details=view_records,
                gui_scores=gui_scores,comparative=comp['symmetric'],order_disagrees=comp['symmetric']['order_disagrees'])
        for method,point in predictions.items():
            pool=next((v for v in p['pools'] if method.startswith(v+'__')),None)
            rows.append(dict(sample_id=c['sample_id'],dataset=c['dataset'],method=method,point=point,
                hit=bool(score_point(point,c)),pool=pool,
                pool_oracle=item['pools'][pool]['point_oracle'] if pool else None))
        details.append(item)
        # Label-assisted coverage diagnostics ONLY, after every prediction above.
        from clickagent.spatial_geometry import shape_from_geometry
        import shapely
        shape=shape_from_geometry(c['target_geometry']);w,h=c['image_size'];area=shape.area
        def visible(t):
            b=t['box'];region=shapely.box(b[0]/w,b[1]/h,b[2]/w,b[3]/h)
            return float(shape.intersection(region).area/area)
        oldvis=visible(p['old_crop_transform']) if p['old_crop_transform'] else 0.
        boxes=[v['transform'] for v in p['context_locals']]
        regions=[shapely.box(t['box'][0]/w,t['box'][1]/h,t['box'][2]/w,t['box'][3]/h) for t in boxes]
        union=shapely.union_all(regions) if regions else shapely.GeometryCollection()
        frac=float(shape.intersection(union).area/area)
        crop_details.append(dict(sample_id=c['sample_id'],old_target_any_visible=oldvis>0,context_union_target_any_visible=frac>0,
            old_target_fully_visible=oldvis>=1-1e-7,context_union_target_fully_visible=frac>=1-1e-7,
            context_individual_visible=[visible(t) for t in boxes],old_visible_fraction=oldvis,context_union_visible_fraction=frac))
    return rows,details,crop_details


def summarize(rows,details,crop_details,cases):
    groups=defaultdict(list)
    for r in rows:groups[r['method']].append(r)
    methods={}
    for method,items in groups.items():
        pool=items[0]['pool'];n=len(items);hits=sum(v['hit'] for v in items);available=sum(v['pool_oracle'] for v in items) if pool else None
        methods[method]=dict(n=n,hits=hits,accuracy=hits/n,points=sum(v['point'] is not None for v in items),
            pool_available=available,conditional_selection_accuracy=hits/available if available else None,
            datasets={d:dict(n=sum(v['dataset']==d for v in items),hits=sum(v['hit'] for v in items if v['dataset']==d)) for d in sorted({v['dataset'] for v in items})})
        if pool:
            anchor={v['sample_id']:v['hit'] for v in groups[pool+'__yes_no']}
            crop={v['sample_id']:v['hit'] for v in groups['old_crop_readout']}
            methods[method].update(rescue_vs_yesno=sum(v['hit'] and not anchor[v['sample_id']] for v in items),
                harm_vs_yesno=sum(not v['hit'] and anchor[v['sample_id']] for v in items),
                rescue_vs_old_crop=sum(v['hit'] and not crop[v['sample_id']] for v in items),
                harm_vs_old_crop=sum(not v['hit'] and crop[v['sample_id']] for v in items))
    pool_summary={}
    for name in ['fixed_pool','context_pool']:
        d=[v['pools'][name] for v in details]
        pool_summary[name]=dict(available=sum(v['point_oracle'] for v in d),
            total_points=sum(len(v['points']) for v in d),order_disagreements=sum(v['order_disagrees'] for v in d),
            candidate_size_counts=dict(Counter(len(v['points']) for v in d)))
    compare=[]
    for name in ['fixed_pool','context_pool']:
        for variant in ['yes_no_logodds','comparative','comparative_forced','point_critic_s1','point_critic_s2','cross_view_critic_s1','gui_actor_crop','gui_actor_full']:
            compare.append(dict(factor=name+' verifier',**paired_interval(rows,name+'__'+variant,name+'__yes_no',cases)))
    for variant in ['yes_no','comparative','point_critic_s1','gui_actor_crop','gui_actor_full']:
        compare.append(dict(factor='candidate pool change',**paired_interval(rows,'context_pool__'+variant,'fixed_pool__'+variant,cases)))
    compare.append(dict(factor='same first ROI, full context',**paired_interval(rows,'context_roi0_readout','old_crop_readout',cases)))
    primary='context_pool__point_critic_s1'
    for baseline in ['qwen_native_completed_lax','ui_tars']:
        if baseline in groups:compare.append(dict(factor='primary versus baseline',**paired_interval(rows,primary,baseline,cases)))
    return dict(complete=True,n=len(cases),primary=primary,methods=methods,pools=pool_summary,
        crop_coverage={k:sum(v[k] for v in crop_details) for k in ['old_target_any_visible','context_union_target_any_visible','old_target_fully_visible','context_union_target_fully_visible']},
        paired_comparisons=compare,protocol_sha256=sha256(OUT/'protocol.json'),
        warning='New192 positive static localization. Main critic seed1 and policies fixed before test. Candidate oracles use GT; not deployable. Two critic seeds; one frozen VLM readout. Training remains873 queries. View budgets and forward counts differ; report separately.')


def compute_budgets(root,cases):
    """Policy-required calls/views, excluding calls used only by other controls.

    These are compute accounting, NOT independently measured end-to-end latency.
    A GUI-Actor candidate batch counts one forward and several image views.
    """
    costs=defaultdict(list)
    for c in cases:
        p=read(root/'pipeline'/f'{key(c["sample_id"])}.json');nlocal=len(p['context_locals']);old=int(p['old_crop_transform'] is not None)
        costs['qwen_native'].append((1,1));costs['global_readout'].append((1,1))
        for name,pool in p['pools'].items():
            n=len(pool);calls=1+(old if name=='fixed_pool' else nlocal);views=1+(old if name=='fixed_pool' else 2*nlocal)
            costs[name+'__point_critic'].append((calls,views))
            costs[name+'__yes_no'].append((calls+n,views+n))
            costs[name+'__comparative'].append((calls+2*bool(n),views+2*(n+1)*bool(n)))
            costs[name+'__gui_actor_batch'].append((calls+bool(n),views+n))
    return {k:dict(mean_vlm_forward_calls=float(np.mean([v[0] for v in values])),
        mean_image_views=float(np.mean([v[1] for v in values]))) for k,values in costs.items()}


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['validation','test'],default='test');args=p.parse_args();torch.set_num_threads(4)
    root=OUT/'validation_smoke' if args.stage=='validation' else OUT
    cases=[c for c in read_jsonl(OUT/'manifest.jsonl') if c['experiment_split']==('val' if args.stage=='validation' else 'test')]
    if args.stage=='validation':cases=sorted(cases,key=lambda c:key(c['sample_id']))[:12]
    else:assert read(root/'ui_tars/summary.json')['complete']
    assert read(root/'completion.json')['complete'] and read(root/'gui_verifier/completion.json')['complete']
    rows,details,crop_details=score(root,cases,args.stage=='test');summary=summarize(rows,details,crop_details,cases)
    summary['compute_accounting']=compute_budgets(root,cases)
    dest=root/'evaluation';dest.mkdir(exist_ok=True)
    (dest/'predictions.jsonl').write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in rows),encoding='utf-8')
    write_json(dest/'summary.json',summary);write_json(dest/'pool_details.json',details);write_json(dest/'crop_details.json',crop_details)
    print(json.dumps({m:v['hits'] for m,v in summary['methods'].items()}),flush=True)


if __name__=='__main__':main()
