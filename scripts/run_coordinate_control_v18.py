"""Hold original selected proposal fixed to separate coordinate vs re-ranking effects."""
from datetime import datetime,timezone
from pathlib import Path
import json
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
from clickagent.evidence_retention import view_rows
from clickagent.frozen_geometry_v18 import MODES,relocate
from run_frozen_alternatives_v18 import read,write,sha,comparison
from clickagent.public_evaluation import point_in_geometry

OUT=ROOT/'runs/coordinate_control_v18'
MAIN=ROOT/'runs/quality_readout_v15_3k'


def moved_point_map(pipe,size,mode):
    old_views=[view_rows(pipe['global_result'],0)]
    new_views=[view_rows(relocate(pipe['global_result'],size,mode),0)]
    for j,v in enumerate(pipe['local_views'],1):
        old_views.append(view_rows(v['result'],j,v['transform']))
        new_views.append(view_rows(relocate(v['result'],v['transform']['crop_size'],mode),j,v['transform']))
    old_order=old_views[0]+[v[k] for k in range(max(map(len,old_views[1:]),default=0)) for v in old_views[1:] if k<len(v)]
    new_order=new_views[0]+[v[k] for k in range(max(map(len,new_views[1:]),default=0)) for v in new_views[1:] if k<len(v)]
    assert len(old_order)==len(new_order)
    mapped={}
    for old,new in zip(old_order,new_order):
        assert (old['view'],old['rank'])==(new['view'],new['rank'])
        mapped.setdefault(tuple(old['point']),new['point'])
    return mapped


def main():
    inputs=[Path(__file__),ROOT/'src/clickagent/frozen_geometry_v18.py',ROOT/'src/clickagent/evidence_retention.py',
        ROOT/'runs/full_scaling_v18/data/development_indices.json',MAIN/'models/frozen.json',
        ROOT/'runs/frozen_geometry_v18/completion.json']
    protocol=OUT/'protocol.json'
    if not protocol.exists():
        write(protocol,dict(registered_at_utc=datetime.now(timezone.utc).isoformat(),
            hashes={str(p):sha(p) for p in inputs},
            purpose='Follow-up development control after geometry experiment: keep the old argmax/proposal source fixed, only relocate its final coordinate. Compare with full feature rebuild and re-ranking.',
            scope='Exploratory used-development-only diagnosis. No fitting, scorer/backbone inference, held-out inference, policy change or adoption.',
            canonical_mapping='First global source then round-robin local rank order, identical to build_base coordinate provenance.'))
    for p,h in read(protocol)['hashes'].items():assert sha(p)==h
    indices=read(ROOT/'runs/full_scaling_v18/data/development_indices.json')
    public={c['sample_id']:c for p in [ROOT/'runs/multiregion_v13/data/val.jsonl',MAIN/'data/new_reserve.jsonl'] for c in (json.loads(s) for s in p.read_text(encoding='utf-8').splitlines())}
    frozen=read(MAIN/'models/frozen.json')['models'];results={};provenance={}
    for cohort,entries in indices.items():
        vectors={mode:[[],[],[]] for mode in MODES};points_out=[]
        for e in entries:
            case=public[e['sample_id']]
            for field,h in [('path','sha256'),('pipeline_path','pipeline_sha256')]:
                assert sha(e[field])==e[h];provenance[e[field]]=e[h]
            pipe=read(e['pipeline_path'])
            maps={mode:moved_point_map(pipe,case['image_size'],mode) for mode in MODES}
            with np.load(e['path'],allow_pickle=False) as z:points=z['points'].copy()
            assert set(maps['original'])==set(map(tuple,points))
            predictions={mode:[] for mode in MODES}
            for s in [1,2,3]:
                legacy='old_val' if cohort=='old_val' else 'new_reserve'
                p=MAIN/f'predictions/{legacy}/expanded_s{s}/{Path(e["path"]).stem}.npz'
                with np.load(p,allow_pickle=False) as z:
                    assert str(z['checkpoint_sha256'])==frozen[f'expanded_s{s}']['sha256']
                    assert str(z['base_sha256'])==e['sha256'];scores=z['scores'].copy()
                provenance[str(p)]=sha(p)
                selected=tuple(points[int(scores.argmax())]) if len(points) else None
                for mode in MODES:
                    point=maps[mode][selected] if selected is not None else None
                    predictions[mode].append(point)
            for mode,seed_points in predictions.items():
                for s,p in enumerate(seed_points):
                    vectors[mode][s].append(bool(p is not None and point_in_geometry((np.asarray(p)/case['image_size']).tolist(),case['target_geometry'])))
            points_out.append(dict(sample_id=e['sample_id'],points=predictions))
        geometry=read(ROOT/f'runs/frozen_geometry_v18/evaluation/{cohort}.json')
        assert vectors['original']==geometry['hit_vectors']['original']
        methods={m:dict(seed_hits=np.asarray(v).sum(1).tolist(),mean_accuracy_percent=100*float(np.mean(v))) for m,v in vectors.items()}
        contrasts={}
        for mode in MODES[1:]:
            contrasts[mode]=dict(coordinate_only_vs_original=comparison(vectors[mode],vectors['original'],entries),
                additional_rebuild_and_rerank_vs_coordinate_only=comparison(geometry['hit_vectors'][mode],vectors[mode],entries))
        result=dict(complete=True,n=len(entries),fixed_selected_proposal_methods=methods,contrasts=contrasts,
            hit_vectors=vectors,predictions=points_out,scope='Descriptive development controls, no new independent test')
        write(OUT/f'evaluation/{cohort}.json',result);results[cohort]=result
    lines=['保持原评分器与已选候选来源不变，仅移动最后坐标的开发对照。']
    for cohort,result in results.items():
        lines.append('\n'+cohort)
        for mode,row in result['fixed_selected_proposal_methods'].items():
            lines.append(f'{mode}: {row["mean_accuracy_percent"]:.2f}%; 命中{row["seed_hits"]}')
        lines.append(json.dumps(result['contrasts'],ensure_ascii=False))
    path=OUT/'results.txt';path.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for p,h in read(protocol)['hashes'].items():assert sha(p)==h
    write(OUT/'completion.json',dict(complete=True,inputs_sha256=provenance,
        artifact_sha256={str(p):sha(p) for p in [protocol,path]+[OUT/f'evaluation/{c}.json' for c in results]},
        no_model_inference=True,no_training=True,mainline_unchanged=True))
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
