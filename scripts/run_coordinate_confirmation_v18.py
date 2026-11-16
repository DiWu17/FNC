"""Prospective additive coordinate-only control, with combined family3 analysis.

Registered before any held-out prediction. Original confirmation sources and
protocol remain untouched; this adds a simpler output rule and recomputes the
combined three-primary-test intervals after the parent experiment completes.
"""
from datetime import datetime,timezone
from pathlib import Path
import json
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
sys.path.insert(0,str(ROOT/'src'))
from run_mvp_confirmation_worker_v18 import read,write,sha
PARENT=ROOT/'runs/mvp_confirmation_v18'
OUT=PARENT/'coordinate_addendum'


def status(phase,**fields):
    value=dict(phase=phase,complete=phase=='complete',updated_at_utc=datetime.now(timezone.utc).isoformat(),**fields)
    write(OUT/'status.json',value);print(json.dumps(value,ensure_ascii=False),flush=True)


def register():
    path=OUT/'protocol.json'
    if path.exists():
        spec=read(path)
        for p,h in spec['hashes'].items():assert sha(p)==h,p
        return spec
    assert not list((PARENT/'predictions').rglob('*.json')), 'Register before any held-out predictions'
    files=[Path(__file__),ROOT/'scripts/run_coordinate_control_v18.py',
        ROOT/'src/clickagent/frozen_geometry_v18.py',ROOT/'src/clickagent/evidence_retention.py',
        PARENT/'protocol.json',PARENT/'data/selection.json',PARENT/'data/heldout.jsonl',
        ROOT/'runs/coordinate_control_v18/completion.json']
    spec=dict(registered_at_utc=datetime.now(timezone.utc).isoformat(),hashes={str(p):sha(p) for p in files},
        extension='Coordinate-only coarse-center output from the original frozen-scorer-selected proposal. Canonical first global then round-robin local provenance. No feature rebuild or re-ranking.',
        primary_family=['full_union vs MVP','full_selected_geometry vs MVP','full_coordinate_center vs MVP'],
        uncertainty='All three primary tests use98.333% paired source-stratified screenshot bootstrap2000 intervals in the combined result. Parent97.5% intervals describe only its original2-family; use this combined table for any3-policy superiority claim.',
        secondary='Original3k coordinate-only control; descriptive95% interval.',
        reason='Development-only follow-up found coordinate-only movement accounts for the gain. This additive control is fixed before any held-out output, not selected after test results.',
        no_new_training_or_model_inference=True,mainline_unchanged=True,html=False)
    write(path,spec);return spec


def analyze():
    import numpy as np
    from run_coordinate_control_v18 import moved_point_map
    from clickagent.public_evaluation import point_in_geometry
    from run_mvp_confirmation_worker_v18 import geometry_hit
    import run_quality_readout_v15 as base
    cases=[json.loads(s) for s in (PARENT/'data/heldout.jsonl').read_text(encoding='utf-8').splitlines()]
    own={r['sample_id']:r for r in read(PARENT/'predictions/ours.json')['rows']}
    mvp={r['sample_id']:r for r in read(PARENT/'predictions/mvp.json')['rows']}
    assert len(cases)==280 and {c['sample_id'] for c in cases}==set(own)==set(mvp)
    added={'full_coordinate_center':[],'original3k_coordinate_center':[]};predictions=[]
    for case in cases:
        sid=case['sample_id'];pipe=read(PARENT/f'features/pipeline/{base.key(sid)}.json')
        mapping=moved_point_map(pipe,case['image_size'],'coarse_center')
        outputs={}
        for arm in ['full','original3k']:
            outputs[arm+'_coordinate_center']=[mapping[tuple(p)] if p is not None else None for p in own[sid]['points'][arm+'_union']]
        for name,points in outputs.items():added[name].append([geometry_hit(p,case) for p in points])
        predictions.append(dict(sample_id=sid,points=outputs))
    ids=[c['sample_id'] for c in cases]
    values={name:np.array([own[sid]['hits'][name] for sid in ids],float).mean(1) for name in own[ids[0]]['hits']}
    values.update({name:np.asarray(v,float).mean(1) for name,v in added.items()})
    values['MVP']=np.array([mvp[sid]['hit'] for sid in ids],float)
    groups=[np.array([i for i,c in enumerate(cases) if c['dataset']==source]) for source in sorted({c['dataset'] for c in cases})]
    rng=np.random.default_rng(20261105)
    draws=np.stack([np.concatenate([g[rng.integers(0,len(g),len(g))] for g in groups]) for _ in range(2000)])
    comparisons={};primary_names=['full_union','full_geometry','full_coordinate_center']
    for name,v in values.items():
        if name=='MVP':continue
        diff=v-values['MVP'];boot=diff[draws].mean(1);primary=name in primary_names
        alpha=.05/3 if primary else .05;bounds=np.quantile(boot,[alpha/2,1-alpha/2])
        comparisons[name]=dict(delta_pp=100*float(diff.mean()),ci_pp=(100*bounds).tolist(),
            primary=primary,interval_level_percent=100*(1-alpha),
            status='higher_than_MVP' if bounds[0]>0 else 'lower_than_MVP' if bounds[1]<0 else 'inconclusive')
    result=dict(complete=True,n=280,methods={name:dict(accuracy_percent=100*float(v.mean())) for name,v in values.items()},
        comparisons=comparisons,coordinate_predictions=predictions,
        scope='Prospectively fixed fresh image subset; combined3-primary family analysis. No full benchmark, general speed or automatic adoption claim.',
        parent_protocol_sha256=sha(PARENT/'protocol.json'),addendum_protocol_sha256=sha(OUT/'protocol.json'))
    write(OUT/'combined_results.json',result)
    lines=['新280张：完整候选、重评分坐标调整、仅输出坐标调整的联合结果。',
        '本表三项主比较使用98.333%区间；原父实验97.5%区间只对应它的两项比较，三策略结论以本表为准。']
    for name,row in result['methods'].items():
        line=f'{name}: {row["accuracy_percent"]:.2f}%'
        if name in comparisons:
            c=comparisons[name];line+=f'; 相对MVP{c["delta_pp"]:+.2f}百分点, {c["interval_level_percent"]:.3f}%区间{c["ci_pp"]}; {c["status"]}'
        lines.append(line)
    lines.append('所有策略及坐标规则在新截图预测前登记；原3000例评分器保持不变。')
    p=OUT/'combined_results.txt';p.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    register()
    write(OUT/'completion.json',dict(complete=True,completed_at_utc=datetime.now(timezone.utc).isoformat(),
        parent_completion_sha256=sha(PARENT/'completion.json'),
        artifact_sha256={str(p):sha(p) for p in [OUT/'protocol.json',OUT/'combined_results.json',OUT/'combined_results.txt']},
        mainline_unchanged=True,no_new_model_inference=True))
    status('complete');print('\n'.join(lines),flush=True)


def main():
    register()
    while not (PARENT/'completion.json').exists():
        if (PARENT/'status.json').exists() and read(PARENT/'status.json').get('phase')=='failed':
            raise RuntimeError('Parent confirmation failed; do not assign missing accuracy')
        status('waiting_for_parent_confirmation');time.sleep(50)
    for p,h in read(PARENT/'completion.json')['artifact_sha256'].items():assert sha(p)==h
    register();analyze()


if __name__=='__main__':
    try:main()
    except BaseException:
        status('failed',error=traceback.format_exc());raise
