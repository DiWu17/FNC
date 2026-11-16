"""Fixed V18 decoding policies on the reused V16/V17 cohort; CPU caches only."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
import run_quality_readout_v15_3k as prior
from clickagent.evidence_retention import RetentionModel
from clickagent.frozen_geometry_v18 import pipeline_geometry
from run_coordinate_control_v18 import moved_point_map
from run_frozen_geometry_v18 import feature_record
from run_frozen_alternatives_v18 import comparison

OUT=ROOT/'runs/frozen_mvp_diagnostic_v18'
MAIN=ROOT/'runs/quality_readout_v15_3k'
OLD=ROOT/'runs/contribution_v16'
BENCH=ROOT/'runs/advantage_validation_v17'
read,write,sha=prior.read,prior.write,prior.sha


def status(phase,**fields):
    value=dict(phase=phase,complete=phase=='complete',updated_at_utc=datetime.now(timezone.utc).isoformat(),**fields)
    write(OUT/'status.json',value);print(json.dumps(value,ensure_ascii=False),flush=True)


def register():
    path=OUT/'protocol.json'
    if not path.exists():
        files=[Path(__file__),ROOT/'scripts/run_coordinate_control_v18.py',ROOT/'scripts/run_frozen_geometry_v18.py',
               ROOT/'scripts/run_frozen_alternatives_v18.py',ROOT/'src/clickagent/frozen_geometry_v18.py',
               OLD/'independent/index.json',OLD/'models/frozen.json',MAIN/'models/frozen.json',
               BENCH/'plan.json',BENCH/'data/cohorts.json',
               BENCH/'inference/mvp_p262144_all/completion.json',BENCH/'inference/ours_p262144_all/completion.json',
               ROOT/'runs/frozen_geometry_v18/completion.json',ROOT/'runs/coordinate_control_v18/completion.json',
               ROOT/'runs/full_scaling_v18/protocol.json',ROOT/'runs/mvp_confirmation_v18/protocol.json',
               ROOT/'runs/mvp_confirmation_v18/coordinate_addendum/protocol.json']
        files += [ROOT/name for name in read(MAIN/'protocol.json')['source_identity']]
        write(path,dict(registered_at_utc=datetime.now(timezone.utc).isoformat(),hashes={str(p):sha(p) for p in files},
                       scope='Reused300 V16/V17 diagnostic screenshots, not new independent confirmation. Same exact cached MVP predictions and original3k checkpoints.',
                       fixed_methods=['Original K16','Full union','Full union with center feature rebuild/rescore','Full-union selected point moved to canonical coarse center','Cached official MVP'],
                       rules='Center modes fixed using old/new development before this diagnostic. No thresholds, fitting, scorer changes, new views or backbone calls.',
                       controls='CPU original reconstruction must match archived base points, scores within1e-4, and original union argmax exactly for every seed. Verify archived K16 and MVP hits offline.',
                       uncertainty='2000 source-stratified RGB-cluster paired bootstrap, descriptive95% intervals only; fixed3 fits, no training-seed population claim',
                       fresh_test_guard='Do not retune, select, remove or add any of the three already registered fresh MVP policies based on this diagnostic. Fresh280 remains unused.',
                       no_mainline_replacement=True,training=False,new_backbone_calls=0,html=False))
    spec=read(path)
    for p,h in spec['hashes'].items():assert sha(p)==h,p
    return spec


@torch.inference_mode()
def main():
    torch.set_num_threads(2);spec=register()
    if (OUT/'completion.json').exists():
        for p,h in read(OUT/'completion.json')['artifact_sha256'].items():assert sha(p)==h,p
        print((OUT/'results.txt').read_text(encoding='utf-8'),flush=True);return
    cases=read(BENCH/'data/cohorts.json')['full_test'];entries=read(OLD/'independent/index.json')['cases']
    case_map={c['sample_id']:c for c in cases};assert len(cases)==len(entries)==300
    assert set(case_map)=={e['sample_id'] for e in entries}
    outputs={}
    provenance={}
    for method in ['ours','mvp']:
        folder=BENCH/f'inference/{method}_p262144_all'
        receipt=read(folder/'completion.json');assert receipt['complete']
        for p,h in receipt['hashes'].items():assert sha(p)==h,p
        path=folder/'accuracy/results.json';outputs[method]={r['sample_id']:r for r in read(path)['rows']}
        assert set(outputs[method])==set(case_map);provenance[str(path)]=sha(path)
    frozen=read(MAIN/'models/frozen.json')['models'];legacy=read(OLD/'models/frozen.json')['models'];models=[]
    for s in [1,2,3]:
        info=frozen[f'expanded_s{s}'];assert info['sha256']==legacy[f'aligned_s{s}']['sha256']
        assert sha(info['path'])==info['sha256']
        saved=torch.load(info['path'],map_location='cpu',weights_only=True)
        model=RetentionModel().eval().requires_grad_(False);model.load_state_dict(saved['state_dict'])
        models.append((model,saved))
    methods=['K16','full_union','center_rescore','coordinate_center','MVP']
    vectors={name:[[],[],[]] for name in methods};oracle={name:[] for name in ['K16','full_union','center_rescore','coordinate_center','MVP_native3']}
    rows=[];replay=dict(seed_predictions=0,point_agreement=0,max_abs_score_error=0.)
    status('replaying_used_300',completed=0,total=300)
    for i,entry in enumerate(entries):
        case=case_map[entry['sample_id']]
        for p,h in [('path','sha256'),('labels_path','labels_sha256'),('pipeline_path','pipeline_sha256')]:
            assert sha(entry[p])==entry[h];provenance[entry[p]]=entry[h]
        with np.load(entry['path'],allow_pickle=False) as archive:
            old_points=archive['points'].copy();baseline=archive['baseline'].copy()
        old_labels=np.load(entry['labels_path'],allow_pickle=False)>.5
        pipe=read(entry['pipeline_path']);mapping=moved_point_map(pipe,case['image_size'],'coarse_center')
        assert set(mapping)==set(map(tuple,old_points))
        raw=[]
        for view in [dict(feature_path=pipe['global_feature_path'],feature_sha256=pipe['global_feature_sha256'])]+pipe['local_views']:
            assert sha(view['feature_path'])==view['feature_sha256'];provenance[view['feature_path']]=view['feature_sha256']
            with np.load(view['feature_path'],allow_pickle=False) as z:
                raw.append(({k:z[k].copy() for k in ['target_attention','target_hidden','image_hidden']},json.loads(str(z['metadata']))))
        records=[];scores_by_mode=[];points_by_mode=[]
        for mode in ['original','coarse_center']:
            changed=pipeline_geometry(pipe,case['image_size'],mode,prior.base.build_c1_c3_pools)
            record=feature_record(changed,dict(image_size=case['image_size']),raw);records.append(record)
            scores=[];points=[]
            for seed,(model,saved) in enumerate(models):
                if record['n']:
                    inputs,_=prior.tensors([record],saved['mean'].numpy(),saved['std'].numpy(),'aligned','cpu')
                    values=model(**inputs).float().numpy();point=record['points'][int(values.argmax())].tolist()
                else:values=np.empty(0,np.float32);point=None
                assert np.isfinite(values).all()
                if mode=='original':
                    assert np.array_equal(record['points'],old_points)
                    path=OLD/f'predictions/independent/aligned_s{seed+1}/{Path(entry["path"]).stem}.npz'
                    with np.load(path,allow_pickle=False) as z:
                        assert str(z['checkpoint_sha256'])==frozen[f'expanded_s{seed+1}']['sha256']
                        assert str(z['base_sha256'])==entry['sha256'];cached=z['scores'].copy()
                    assert values.shape==cached.shape
                    error=float(np.max(np.abs(values-cached))) if len(values) else 0.
                    assert error<1e-4,(entry['sample_id'],error)
                    assert not len(values) or int(values.argmax())==int(cached.argmax())
                    replay['max_abs_score_error']=max(replay['max_abs_score_error'],error)
                    replay['seed_predictions']+=1;replay['point_agreement']+=1;provenance[str(path)]=sha(path)
                scores.append(values);points.append(point)
            scores_by_mode.append(np.asarray(scores));points_by_mode.append(points)
        old=outputs['ours'][entry['sample_id']];mvp=outputs['mvp'][entry['sample_id']]
        predicted=dict(K16=old['seed_points'],full_union=points_by_mode[0],center_rescore=points_by_mode[1],
                       coordinate_center=[mapping[tuple(p)] if p is not None else None for p in points_by_mode[0]],
                       MVP=[mvp['point']]*3)
        # No target geometry was used to generate or select any point above.
        def hit(point):
            return bool(point is not None and prior.base.point_in_geometry((np.asarray(point)/case['image_size']).tolist(),case['target_geometry']))
        assert np.array_equal(old_labels,[hit(p) for p in old_points])
        assert [hit(p) for p in old['seed_points']]==old['seed_hits']
        assert hit(mvp['point'])==bool(mvp['hit'])
        for name,points in predicted.items():
            for seed,p in enumerate(points):vectors[name][seed].append(hit(p))
        native=[v['point'] for v in mvp['output']['reference_result']['all_predictions']];assert len(native)==3
        sets=dict(K16=old_points[baseline],full_union=old_points,center_rescore=records[1]['points'],
                  coordinate_center=list(mapping.values()),MVP_native3=native)
        for name,points in sets.items():oracle[name].append(any(hit(p) for p in points))
        cache=OUT/f'scores/{Path(entry["path"]).stem}.npz'
        prior.base.npz(cache,original_points=records[0]['points'],center_points=records[1]['points'],
                       original_scores=scores_by_mode[0],center_scores=scores_by_mode[1],protocol_sha256=np.asarray(sha(OUT/'protocol.json')))
        provenance[str(cache)]=sha(cache)
        rows.append(dict(sample_id=entry['sample_id'],source=case['dataset'],points=predicted))
        if (i+1)%10==0 or i+1==300:status('replaying_used_300',completed=i+1,total=300)
    assert np.asarray(vectors['K16']).sum(1).tolist()==[90,81,92]
    assert np.asarray(vectors['full_union']).sum(1).tolist()==[93,84,98]
    assert np.asarray(vectors['MVP']).sum(1).tolist()==[116,116,116]
    assert sum(oracle['full_union'])==192 and sum(oracle['K16'])==152 and sum(oracle['MVP_native3'])==151
    metrics={name:dict(accuracy_percent=100*float(np.mean(v)),seed_hits=np.asarray(v).sum(1).tolist(),
                       by_source={source:100*float(np.asarray(v)[:,[j for j,e in enumerate(entries) if e['source']==source]].mean())
                                  for source in sorted({e['source'] for e in entries})}) for name,v in vectors.items()}
    comparisons={name:{k:r[k] for k in ['delta_pp','ci95_pp']} for name,v in vectors.items() if name!='MVP'
                 for r in [comparison(v,vectors['MVP'],entries)]}
    within={name:{k:r[k] for k in ['delta_pp','ci95_pp']} for name in ['center_rescore','coordinate_center']
            for r in [comparison(vectors[name],vectors['full_union'],entries)]}
    result=dict(complete=True,n=300,scope=spec['scope'],methods=metrics,oracle_hits={k:sum(v) for k,v in oracle.items()},
                comparisons_to_MVP=comparisons,decoding_vs_full_union=within,hit_vectors=vectors,predictions=rows,
                replay=replay,independent_confirmation=False,fresh_280_used=False,no_new_backbone_calls=True,no_mainline_replacement=True)
    write(OUT/'results.json',result)
    lines=['固定评分器的MVP同截图诊断：这300张已反复使用，不是新的独立确认。',
           '仅CPU重放缓存；没有训练或新增大模型前向。所有坐标规则在本次诊断前由开发集固定。',
           '每个本方法结果是固定三种子均值，不是集成；MVP为相同旧配置的缓存预测。']
    for name,row in metrics.items():
        line=f'{name}: {row["accuracy_percent"]:.2f}%; 三种子命中{row["seed_hits"]}'
        if name!='MVP':line+=f'; 相对MVP {comparisons[name]["delta_pp"]:+.2f}百分点'
        lines.append(line)
    lines += ['候选oracle：'+json.dumps(result['oracle_hits'],ensure_ascii=False),
              '坐标变化相对完整候选：'+json.dumps(within,ensure_ascii=False),
              '区间仅作已用诊断集的描述，不据此宣称独立优越性或修改既有三项新截图方案。',
              '相同8B和单视图像素上限不代表总计算量相同；本次不是速度实验。',
              '新280张测试与全量新模型仍由原协议执行，本诊断不会选新种子、改模型或替换主线。']
    (OUT/'results.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    register()
    for model in frozen.values():assert sha(model['path'])==model['sha256']
    files=[OUT/'protocol.json',OUT/'results.json',OUT/'results.txt']
    write(OUT/'completion.json',dict(complete=True,completed_at_utc=datetime.now(timezone.utc).isoformat(),inputs_sha256=provenance,
                                    artifact_sha256={str(p):sha(p) for p in files},mainline_unchanged=True,independent_confirmation=False))
    status('complete');print('\n'.join(lines),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        status('failed',error=traceback.format_exc());raise
