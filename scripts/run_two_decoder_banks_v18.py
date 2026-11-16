"""Development-only fusion of two public coordinate decodings, fixed scorer."""
from datetime import datetime, timezone
from pathlib import Path
import json
import sys
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import torch
import run_quality_readout_v15_3k as prior
from clickagent.evidence_retention import RetentionModel
from clickagent.frozen_geometry_v18 import pipeline_geometry
from run_frozen_geometry_v18 import feature_record
from run_frozen_alternatives_v18 import comparison

OUT=ROOT/'runs/two_decoder_banks_v18'
MAIN=ROOT/'runs/quality_readout_v15_3k'
GEO=ROOT/'runs/frozen_geometry_v18'
COORD=ROOT/'runs/coordinate_control_v18'
FULL=ROOT/'runs/full_scaling_v18'
read,write,sha=prior.read,prior.write,prior.sha


def status(phase,**fields):
    value=dict(phase=phase,complete=phase=='complete',updated_at_utc=datetime.now(timezone.utc).isoformat(),**fields)
    write(OUT/'status.json',value);print(json.dumps(value,ensure_ascii=False),flush=True)


def register():
    path=OUT/'protocol.json'
    if not path.exists():
        files=[Path(__file__),ROOT/'scripts/run_frozen_geometry_v18.py',ROOT/'scripts/run_frozen_alternatives_v18.py',
               ROOT/'src/clickagent/frozen_geometry_v18.py',FULL/'data/development_indices.json',
               MAIN/'models/frozen.json',GEO/'completion.json',COORD/'completion.json']
        files += [ROOT/name for name in read(MAIN/'protocol.json')['source_identity']]
        write(path,dict(registered_at_utc=datetime.now(timezone.utc).isoformat(),hashes={str(p):sha(p) for p in files},
                       hypothesis='Moving every point to a coarse center loses some previously reachable targets. Keep both fine and center coordinates and let the existing frozen scorer arbitrate.',
                       policy='Construct original and coarse_center feature banks separately from the same cached views. Select the largest raw scorer logit over both banks, preferring original on an exact tie. No calibration, thresholds, fitting, new views, or backbone calls.',
                       provenance='Each bank preserves its own public proposal ranks and coordinate deduplication; not a concatenated bank with changed proposal ranks.',
                       controls='All three seeds; exact original and coarse-center selected-point replay against completed geometry study. Ground truth only used after predictions.',
                       evaluation='Reused old172/new300 development only. Descriptive paired RGB-cluster intervals; not an independent superiority claim.',
                       prospective_confirmation_gate='Only consider a future independent addendum if the new-development mean is strictly higher than fixed-selection coarse_center AND old-development mean is at least that control. No heldout inference or primary-family change in this script.',
                       device='cpu',training=False,mainline_unchanged=True,html=False))
    spec=read(path)
    for p,h in spec['hashes'].items():assert sha(p)==h,p
    return spec


@torch.inference_mode()
def main():
    torch.set_num_threads(2);spec=register()
    frozen=read(MAIN/'models/frozen.json')['models'];models=[]
    hashes=[]
    for s in [1,2,3]:
        info=frozen[f'expanded_s{s}'];assert sha(info['path'])==info['sha256']
        saved=torch.load(info['path'],map_location='cpu',weights_only=True)
        model=RetentionModel().eval().requires_grad_(False);model.load_state_dict(saved['state_dict'])
        models.append((model,saved));hashes.append(info['sha256'])
    public={c['sample_id']:c for p in [ROOT/'runs/multiregion_v13/data/val.jsonl',MAIN/'data/new_reserve.jsonl']
            for c in prior.base.rows(p)}
    indices=read(FULL/'data/development_indices.json');results={};provenance={}
    for cohort,entries in indices.items():
        control=read(GEO/f'evaluation/{cohort}.json')
        coordinate=read(COORD/f'evaluation/{cohort}.json')
        vectors={name:[[],[],[]] for name in ['original','center_rescore','two_banks']}
        oracles={name:[] for name in ['original','center','two_banks']}
        outputs=[]
        status('evaluating_'+cohort,completed=0,total=len(entries))
        for i,entry in enumerate(entries):
            case=public[entry['sample_id']]
            assert sha(entry['pipeline_path'])==entry['pipeline_sha256']
            provenance[entry['pipeline_path']]=entry['pipeline_sha256']
            pipe=read(entry['pipeline_path']);cache=OUT/f'scores/{cohort}/{Path(entry["path"]).stem}.npz'
            if cache.exists():
                with np.load(cache,allow_pickle=False) as z:
                    assert str(z['protocol_sha256'])==sha(OUT/'protocol.json')
                    assert str(z['pipeline_sha256'])==entry['pipeline_sha256']
                    assert z['checkpoint_sha256'].tolist()==hashes
                    bank_points=[z['fine_points'].copy(),z['center_points'].copy()]
                    bank_scores=[z['fine_scores'].copy(),z['center_scores'].copy()]
            else:
                raw=[]
                views=[dict(feature_path=pipe['global_feature_path'],feature_sha256=pipe['global_feature_sha256'])]+pipe['local_views']
                for view in views:
                    assert sha(view['feature_path'])==view['feature_sha256']
                    provenance[view['feature_path']]=view['feature_sha256']
                    with np.load(view['feature_path'],allow_pickle=False) as z:
                        raw.append(({k:z[k].copy() for k in ['target_attention','target_hidden','image_hidden']},json.loads(str(z['metadata']))))
                bank_points=[];bank_scores=[]
                for mode in ['original','coarse_center']:
                    changed=pipeline_geometry(pipe,case['image_size'],mode,prior.base.build_c1_c3_pools)
                    record=feature_record(changed,dict(image_size=case['image_size']),raw)
                    scores=[]
                    for model,saved in models:
                        if record['n']:
                            inputs,_=prior.tensors([record],saved['mean'].numpy(),saved['std'].numpy(),'aligned','cpu')
                            scores.append(model(**inputs).float().numpy())
                        else:scores.append(np.empty(0,np.float32))
                    bank_points.append(record['points']);bank_scores.append(np.asarray(scores))
                prior.base.npz(cache,fine_points=bank_points[0],center_points=bank_points[1],
                               fine_scores=bank_scores[0],center_scores=bank_scores[1],
                               protocol_sha256=np.asarray(sha(OUT/'protocol.json')),
                               pipeline_sha256=np.asarray(entry['pipeline_sha256']),checkpoint_sha256=np.asarray(hashes))
            predictions={name:[] for name in vectors};selected_banks=[];best_logits=[]
            for seed in range(3):
                points=[];maxima=[]
                for p,scores in zip(bank_points,bank_scores):
                    if len(p):
                        pick=int(scores[seed].argmax());points.append(p[pick].tolist());maxima.append(float(scores[seed,pick]))
                    else:points.append(None);maxima.append(float('-inf'))
                choose=int(maxima[1]>maxima[0])
                predictions['original'].append(points[0]);predictions['center_rescore'].append(points[1])
                predictions['two_banks'].append(points[choose]);selected_banks.append(['original','center'][choose])
                best_logits.append([v if np.isfinite(v) else None for v in maxima])
            assert predictions['original']==control['predictions'][i]['points']['original']
            assert predictions['center_rescore']==control['predictions'][i]['points']['coarse_center']
            # All decisions above use public features and frozen weights only.
            def hit(point):
                return bool(point is not None and prior.base.point_in_geometry((np.asarray(point)/case['image_size']).tolist(),case['target_geometry']))
            for name,points in predictions.items():
                for seed,p in enumerate(points):vectors[name][seed].append(hit(p))
            oracle=[any(hit(p) for p in points) for points in bank_points]
            for name,value in zip(oracles,[*oracle,any(oracle)]):oracles[name].append(bool(value))
            outputs.append(dict(sample_id=entry['sample_id'],points=predictions,selected_banks=selected_banks,best_logits=best_logits))
            provenance[str(cache)]=sha(cache)
            if (i+1)%10==0 or i+1==len(entries):status('evaluating_'+cohort,completed=i+1,total=len(entries))
        assert vectors['original']==control['hit_vectors']['original']
        assert vectors['center_rescore']==control['hit_vectors']['coarse_center']
        methods={name:dict(mean_accuracy_percent=100*float(np.mean(v)),seed_hits=np.asarray(v).sum(1).tolist()) for name,v in vectors.items()}
        result=dict(complete=True,n=len(entries),methods=methods,hit_vectors=vectors,predictions=outputs,
                    oracle_hits={name:sum(v) for name,v in oracles.items()},
                    comparisons=dict(vs_original=comparison(vectors['two_banks'],vectors['original'],entries),
                                     vs_center_rescore=comparison(vectors['two_banks'],vectors['center_rescore'],entries),
                                     vs_coordinate_only=comparison(vectors['two_banks'],coordinate['hit_vectors']['coarse_center'],entries)),
                    descriptive_only=True)
        write(OUT/f'evaluation/{cohort}.json',result);results[cohort]=result
    gate=all([results['new_development']['methods']['two_banks']['mean_accuracy_percent']>
              read(COORD/'evaluation/new_development.json')['fixed_selected_proposal_methods']['coarse_center']['mean_accuracy_percent'],
              results['old_val']['methods']['two_banks']['mean_accuracy_percent']>=
              read(COORD/'evaluation/old_val.json')['fixed_selected_proposal_methods']['coarse_center']['mean_accuracy_percent']])
    lines=['双坐标候选库：已有评分器冻结、仅CPU复用缓存的开发实验。',
           '两个库各自构建特征并用同一模型评分；跨库取最高原始logit，完全相同时优先原坐标。',
           '无训练、无新增8B前向；只涉及已经使用过的开发集。']
    for cohort,result in results.items():
        lines.append('\n'+cohort)
        for name,row in result['methods'].items():lines.append(f'{name}: {row["mean_accuracy_percent"]:.2f}%; 命中{row["seed_hits"]}')
        lines.append('候选oracle命中数：'+json.dumps(result['oracle_hits'],ensure_ascii=False))
        lines.append('相对仅移动坐标：'+json.dumps(result['comparisons']['vs_coordinate_only'],ensure_ascii=False))
    lines.append('预先登记的追加独立确认门槛是否满足：'+str(gate))
    lines.append('此脚本不会执行独立测试、改动既有三项主比较或替换主线。')
    (OUT/'results.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    register()
    paths=[OUT/'protocol.json',OUT/'results.txt']+[OUT/f'evaluation/{cohort}.json' for cohort in results]
    write(OUT/'completion.json',dict(complete=True,completed_at_utc=datetime.now(timezone.utc).isoformat(),
                                    input_sha256=provenance,artifact_sha256={str(p):sha(p) for p in paths},
                                    confirmation_gate_passed=bool(gate),training=False,new_backbone_calls=0,mainline_unchanged=True))
    status('complete',confirmation_gate_passed=bool(gate));print('\n'.join(lines),flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        status('failed',error=traceback.format_exc());raise
