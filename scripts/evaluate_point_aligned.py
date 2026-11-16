"""Score the sealed fresh cohort; all geometry labels enter here after prediction."""
from collections import Counter,defaultdict
import json
from pathlib import Path

import numpy as np
import shapely
from shapely.affinity import scale
from PIL import Image,ImageDraw
import torch

from clickagent.action_benchmark import sha256,write_json
from clickagent.spatial_geometry import shape_from_geometry
from clickagent.point_aligned_inference import map_crop_point,unique_points
from evaluate_conditioned_readouts import score_point
from evaluate_phrase_readouts import paired_interval
from run_target_token_experiment import key,read_jsonl
from point_aligned_data import OUT,read


def record(case,method,point,**extra):
    hit=bool(score_point(point,case));distance=None
    if point is not None:
        shape=scale(shape_from_geometry(case['target_geometry']),xfact=case['image_size'][0],yfact=case['image_size'][1],origin=(0,0))
        distance=float(shape.distance(shapely.Point(point)))
    return dict(sample_id=case['sample_id'],dataset=case['dataset'],method=method,point=point,
        hit=hit,distance_to_target_region_original_px=distance,**extra)


def diagnose(case,result):
    if not result['candidates']:
        return dict(failure_stage='no_name',parent_oracle=False,point_oracle=False,selected_parent_overlap=False)
    gh,gw=result['grid_shape'];indices=np.asarray(result['parent_indices']);rr,cc=np.divmod(indices,gw)
    shape=shape_from_geometry(case['target_geometry']);cells=shapely.box(cc/gw,rr/gh,(cc+1)/gw,(rr+1)/gh)
    covered=np.asarray(shapely.area(shapely.intersection(cells,shape)))>0
    # Use the common evaluator's inclusive edge tolerance for both the score
    # and the point oracle; geometric intersection is only a region diagnostic.
    hits=np.asarray([score_point(point,case) for point in result['all_points']],bool)
    ci=result['candidates'][0]['candidate_index'];hit=bool(hits[ci]);parent=bool(covered[ci])
    stage='hit' if hit else 'fine_miss' if parent else 'ranking_miss' if covered.any() else 'no_parent'
    return dict(failure_stage=stage,parent_oracle=bool(covered.any()),point_oracle=bool(hits.any()),
        selected_parent_overlap=parent,top4_point_oracle=bool(any(hits[c['candidate_index']] for c in result['candidates'][:4])),
        top4_parent_oracle=bool(any(covered[c['candidate_index']] for c in result['candidates'][:4])))


def main():
    assert read(OUT/'completion.json')['complete']
    assert read(OUT/'ui_tars'/'summary.json')['complete']
    declaration=OUT/'evaluation'/'posthoc_pool_fill_declaration.json'
    if not declaration.exists():
        write_json(declaration,dict(preregistered=False,primary_policy_unchanged=True,
            origin='Replay of the fixed exploratory fill rule introduced in the original v8 after its ScreenSpot-v2 partial scores.',
            rule='Unique(combined pool, globalTop4), retain first4, choose highest archived verifier score.',new_vlm_calls=0))
    cases=[c for c in read_jsonl(OUT/'manifest.jsonl') if c['experiment_split']=='test'];rows=[];pipeline_diagnostics=[];crop_diagnostics=[]
    timings=defaultdict(list);passes=Counter();method_passes=Counter()
    for c in cases:
        sid=key(c['sample_id']);global_results=read(OUT/'global'/f'{sid}.json')['predictions'];p=read(OUT/'pipeline'/f'{sid}.json')
        for method,result in global_results.items():
            rows.append(record(c,method,result['point'],**diagnose(c,result)))
        native=read(OUT/'native'/f'{sid}.json');ui=read(OUT/'ui_tars'/f'{sid}.json')
        rows.append(record(c,'qwen_native_completed_lax',native['completed_lax_point']))
        rows.append(record(c,'qwen_native_strict',native['point']))
        rows.append(record(c,'ui_tars',ui['point']))
        for name,point in [('crop_readout',p['crop_point']),('global_verified',p['global_verified_point']),
                           ('crop_verified',p['combined_verified_point']),('crop_native_reference',p['crop_native_point'])]:
            rows.append(record(c,'joint_connected_s1__'+name,point))
        filled=unique_points(p['combined_pool'],p['global_pool'],limit=4)
        filled_winner=max(filled,key=lambda v:p['verification'][','.join(map(str,v['point']))]['score']) if filled else None
        filled_point=filled_winner['point'] if filled_winner else None
        rows.append(record(c,'joint_connected_s1__crop_verified_fill_posthoc',filled_point,posthoc=True))
        global_pool_hit=any(score_point(v['point'],c) for v in p['global_pool'])
        combined_pool_hit=any(score_point(v['point'],c) for v in p['combined_pool'])
        if p['transform'] is not None:
            t=p['transform'];box=np.asarray(t['box'],float);w,h=c['image_size'];shape=shape_from_geometry(c['target_geometry'])
            area=float(shape.area);crop_shape=shapely.box(box[0]/w,box[1]/h,box[2]/w,box[3]/h)
            visible_fraction=float(shape.intersection(crop_shape).area/area)
            cr=p['crop_result'];parent_oracle=False;point_oracle=False;selected_parent=False
            if cr['candidates']:
                gh,gw=cr['grid_shape'];idx=np.asarray(cr['parent_indices']);rr,cc=np.divmod(idx,gw)
                cells=shapely.box((box[0]+cc/gw*(box[2]-box[0]))/w,(box[1]+rr/gh*(box[3]-box[1]))/h,
                    (box[0]+(cc+1)/gw*(box[2]-box[0]))/w,(box[1]+(rr+1)/gh*(box[3]-box[1]))/h)
                overlaps=np.asarray(shapely.area(shapely.intersection(cells,shape)))>0
                parent_oracle=bool(overlaps.any());selected_parent=bool(overlaps[cr['candidates'][0]['candidate_index']])
                point_oracle=any(score_point(map_crop_point(v,t),c) for v in cr['all_points'])
            crop_diagnostics.append(dict(sample_id=c['sample_id'],visible_fraction=visible_fraction,
                target_any_visible=visible_fraction>0,target_fully_visible=visible_fraction>=1-1e-7,
                named=bool(cr['candidates']),parent_oracle=parent_oracle,point_oracle=point_oracle,
                selected_parent_overlap=selected_parent,hit=bool(score_point(p['crop_point'],c)),
                native_hit=bool(score_point(p['crop_native_point'],c)),
                global_native_hit=bool(score_point(native['completed_lax_point'],c)),
                verifier_hit=bool(score_point(p['combined_verified_point'],c))))
        pipeline_diagnostics.append(dict(sample_id=c['sample_id'],dataset=c['dataset'],global_top4_oracle=global_pool_hit,
            combined_pool_oracle=combined_pool_hit,global_verified_hit=score_point(p['global_verified_point'],c),
            combined_verified_hit=score_point(p['combined_verified_point'],c),
            global_hit=score_point(global_results['joint_connected_s1']['point'],c),crop_hit=score_point(p['crop_point'],c),
            crop_native_hit=score_point(p['crop_native_point'],c),verifier_passes=p['verifier_unique_passes']))
        pipeline_diagnostics[-1].update(combined_pool_size=len(p['combined_pool']),filled_pool_size=len(filled),
            filled_pool_oracle=any(score_point(v['point'],c) for v in filled),filled_hit=bool(score_point(filled_point,c)))
        timings['native'].append(native['elapsed_seconds']);timings['ui_tars'].append(ui['elapsed_seconds'])
        with np.load(OUT/'features'/f'{sid}.npz',allow_pickle=False) as z:meta=json.loads(str(z['metadata']))
        timings['global_name_features'].append(meta['elapsed_seconds'])
        cp=OUT/'crop_features'/f'{sid}.npz'
        if cp.exists():
            with np.load(cp,allow_pickle=False) as z:cm=json.loads(str(z['metadata']))
            timings['crop_name_features'].append(cm['elapsed_seconds'])
            timings['crop_coordinate_reference'].append(read(OUT/'crop_native'/f'{sid}.json')['elapsed_seconds'])
        for item in p['verification'].values():timings['verification_pass'].append(item['elapsed_seconds'])
        passes['global_feature']+=1;passes['global_coordinate_reference']+=1
        passes['crop_feature']+=int(cp.exists());passes['verifier']+=p['verifier_unique_passes']
        passes['crop_coordinate_reference']+=int((OUT/'crop_native'/f'{sid}.json').exists())
        method_passes['global_only']+=1
        method_passes['crop_readout']+=1+int(cp.exists())
        method_passes['global_verified']+=1+len(p['global_pool'])
        method_passes['crop_verified']+=1+int(cp.exists())+len(p['combined_pool'])
        method_passes['crop_verified_fill_posthoc']+=1+int(cp.exists())+len(filled)
    groups=defaultdict(list)
    for r in rows:groups[r['method']].append(r)
    methods={}
    for method,items in sorted(groups.items()):
        assert len(items)==len(cases)==len({r['sample_id'] for r in items})
        info=dict(n=len(cases),hits=sum(r['hit'] for r in items),accuracy=np.mean([r['hit'] for r in items]),
            posthoc=any(r.get('posthoc',False) for r in items),
            points=sum(r['point'] is not None for r in items),datasets={ds:dict(n=sum(r['dataset']==ds for r in items),
                hits=sum(r['hit'] for r in items if r['dataset']==ds)) for ds in sorted({r['dataset'] for r in items})})
        if any('failure_stage' in r for r in items):
            info.update(failure_stages=dict(Counter(r['failure_stage'] for r in items)),
                **{k:sum(r.get(k,False) for r in items) for k in ('parent_oracle','point_oracle','selected_parent_overlap','top4_point_oracle','top4_parent_oracle')})
        distances=[r['distance_to_target_region_original_px'] for r in items if not r['hit'] and r['point'] is not None]
        info['miss_distance_diagnostic']=dict(failed_with_point=len(distances),
            median_original_px=float(np.median(distances)) if distances else None,
            within2px=sum(d<=2 for d in distances),within5px=sum(d<=5 for d in distances),within10px=sum(d<=10 for d in distances),
            note='Distance to nearest GT region boundary, not to one target center; diagnostic only, official hits unchanged.')
        methods[method]=info
    comparisons=[]
    def compare(factor,a,b):comparisons.append(dict(factor=factor,**paired_interval(rows,a,b,cases)))
    for suffix in ('s1','s2'):
        compare('point objective recipe',f'point_fixed_{suffix}',f'region_fixed_{suffix}')
        compare('direct field to fine gradient',f'joint_connected_{suffix}',f'joint_detached_{suffix}')
        compare('own decoder plus dynamic targets',f'joint_detached_{suffix}',f'point_fixed_{suffix}')
        compare('global versus warm-start v7',f'joint_connected_{suffix}',f'v7_point_{suffix}')
        compare('joint versus warm-start own decoder',f'joint_connected_{suffix}',f'v7_point_{suffix}_ownfine')
        compare('global versus Qwen coordinates',f'joint_connected_{suffix}','qwen_native_completed_lax')
    primary='joint_connected_s1'
    for method in ['crop_readout','global_verified','crop_verified','crop_native_reference']:
        compare(method,primary+'__'+method,primary)
    compare('pipeline versus Qwen',primary+'__crop_verified','qwen_native_completed_lax')
    compare('pipeline versus UI-TARS',primary+'__crop_verified','ui_tars')
    compare('secondary crop readout versus Qwen',primary+'__crop_readout','qwen_native_completed_lax')
    compare('secondary crop readout versus UI-TARS',primary+'__crop_readout','ui_tars')
    compare('POST-HOC fill unused pool slots',primary+'__crop_verified_fill_posthoc',primary+'__crop_verified')
    verify_summary=dict(n=len(cases),**{k:sum(bool(r[k]) for r in pipeline_diagnostics) for k in [
        'global_top4_oracle','combined_pool_oracle','global_verified_hit','combined_verified_hit','global_hit','crop_hit','crop_native_hit']})
    verify_summary['posthoc_fill']=dict(pool_extended_cases=sum(r['filled_pool_size']>r['combined_pool_size'] for r in pipeline_diagnostics),
        oracle=sum(r['filled_pool_oracle'] for r in pipeline_diagnostics),hits=sum(r['filled_hit'] for r in pipeline_diagnostics),
        rescue=sum(not r['combined_verified_hit'] and r['filled_hit'] for r in pipeline_diagnostics),
        hurt=sum(r['combined_verified_hit'] and not r['filled_hit'] for r in pipeline_diagnostics),
        preregistered=False,main_policy_unchanged=True)
    for name,oracle,chosen in [('global','global_top4_oracle','global_verified_hit'),('combined','combined_pool_oracle','combined_verified_hit')]:
        verify_summary[name+'_pool_available_but_verifier_missed']=sum(r[oracle] and not r[chosen] for r in pipeline_diagnostics)
        verify_summary[name+'_rescue']=sum(not r['global_hit'] and r[chosen] for r in pipeline_diagnostics)
        verify_summary[name+'_hurt']=sum(r['global_hit'] and not r[chosen] for r in pipeline_diagnostics)
    summary=dict(complete=True,n=len(cases),primary=primary+'__crop_verified',methods=methods,
        paired_comparisons=comparisons,pipeline=verify_summary,passes=dict(passes),
        method_required_visual_passes={k:dict(total=v,mean_per_case=v/len(cases)) for k,v in method_passes.items()},
        crop_cascade=dict(n=len(crop_diagnostics),**{k:sum(bool(r[k]) for r in crop_diagnostics) for k in [
            'target_any_visible','target_fully_visible','named','parent_oracle','point_oracle','selected_parent_overlap','hit','native_hit']},
            fully_visible_but_no_name=sum(r['target_fully_visible'] and not r['named'] for r in crop_diagnostics),
            fully_visible_native_hit_readout_miss=sum(r['target_fully_visible'] and r['native_hit'] and not r['hit'] for r in crop_diagnostics),
            fully_visible_global_native_hit_crop_native_miss=sum(r['target_fully_visible'] and r['global_native_hit'] and not r['native_hit'] for r in crop_diagnostics),
            crop_readout_hit_but_verifier_missed=sum(r['hit'] and not r['verifier_hit'] for r in crop_diagnostics)),
        timings={k:dict(n=len(v),mean=float(np.mean(v)),median=float(np.median(v)),p90=float(np.quantile(v,.9))) for k,v in timings.items()},
        protocol_sha256=sha256(OUT/'protocol.json'),inference_identity_sha256=sha256(OUT/'inference_identity.json'),
        evaluation_sources_sha256={p:sha256(Path(p)) for p in [str(Path(__file__)),
            'scripts/evaluate_conditioned_readouts.py','src/clickagent/public_evaluation.py','src/clickagent/spatial_geometry.py']},
        posthoc_pool_fill_declaration_sha256=sha256(OUT/'evaluation'/'posthoc_pool_fill_declaration.json'),
        pro_input_resolution_audit_sha256=sha256(OUT/'evaluation'/'label_visual_checks'/'pro_audit.json') if (OUT/'evaluation'/'label_visual_checks'/'pro_audit.json').exists() else None,
        verifier_precision_diagnostic_sha256=sha256(OUT/'evaluation'/'verifier_precision_diagnostic.json') if (OUT/'evaluation'/'verifier_precision_diagnostic.json').exists() else None,
        warning='Fresh192 positive static pointing only. Two global training seeds; crop and frozen zero-shot verifier use seed1 only. Crops/verifier add visual compute. Exploratory paired bootstrap, no multiplicity correction. Native crop is an explicit coordinate reference, excluded from attention pipeline.')
    dest=OUT/'evaluation';dest.mkdir(exist_ok=True)
    (dest/'predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    write_json(dest/'summary.json',summary);write_json(dest/'pipeline_diagnostics.json',pipeline_diagnostics)
    write_json(dest/'crop_diagnostics.json',crop_diagnostics)
    print(json.dumps({m:v['hits'] for m,v in methods.items()},indent=2),flush=True)


if __name__=='__main__':main()
