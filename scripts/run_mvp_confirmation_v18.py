"""Wait for full-data training, freeze development choice, then fresh MVP test."""
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from run_mvp_confirmation_worker_v18 import OUT,read,write,sha
FULL=ROOT/'runs/full_scaling_v18'
GEO=ROOT/'runs/frozen_geometry_v18'


def status(phase,**fields):
    row=dict(phase=phase,complete=phase=='complete',updated_at_utc=datetime.now(timezone.utc).isoformat(),**fields)
    write(OUT/'status.json',row);print(json.dumps(row,ensure_ascii=False),flush=True)


def register():
    path=OUT/'protocol.json'
    if path.exists():
        spec=read(path)
        for p,expected in spec['hashes'].items():assert sha(p)==expected,p
        return spec
    files=[Path(__file__),ROOT/'scripts/run_mvp_confirmation_worker_v18.py',
        ROOT/'scripts/run_full_scaling_v18.py',ROOT/'scripts/run_frozen_geometry_v18.py',
        ROOT/'scripts/run_frozen_alternatives_v18.py',ROOT/'src/clickagent/frozen_geometry_v18.py',
        ROOT/'src/clickagent/frozen_selection_v18.py',ROOT/'src/clickagent/advantage_runtimes.py',
        OUT/'data/selection.json',OUT/'data/heldout.jsonl',FULL/'protocol.json',GEO/'protocol.json',
        ROOT/'runs/advantage_validation_v17/setup/sources_and_dependencies.json']
    hashes={str(p):sha(p) for p in files}
    hashes.update(read(FULL/'protocol.json')['hashes'])
    spec=dict(registered_at_utc=datetime.now(timezone.utc).isoformat(),hashes=hashes,
        cohort='280 fresh RGB/dHash-isolated screenshots:140 ScreenSpot-Pro plus140 VenusBench-GD; fixed before model results.',
        checkpoints='All3 seeds: full13373 data, matched3k with identical full update budget, original2973-data2976-update references. No seed selection or ensemble.',
        geometry_selection='From3 fixed candidates select highest new-development seed-mean accuracy only if strictly above original and old-development mean at least original. Stable listed-mode tie order. Otherwise original. Freeze choice before first held-out inference.',
        primary_family=['full_union vs MVP','full_selected_geometry vs MVP'],
        secondary='Original3k, matched3k and original3k-selected geometry; full-vs-matched3k data effect, all reported descriptively.',
        uncertainty='Paired source-stratified screenshot bootstrap2000;97.5% intervals for2 primary tests,95% descriptive intervals. Conditional on fixed3 scorer fits.',
        upstream='Same Qwen3-VL-8B weights, FP16 and262144 pixels per view. MVP pinned official2-subview adapter, layer24, consistency14.',
        budget='Shared Qwen features across our accuracy policies only; not a latency test. Per-view cap does not match total compute; view counts and model/data differences declared.',
        guards='All six new fits and geometry development must complete; every checkpoint and policy frozen before held-out inference. Technical failures stop; never assigned as accuracy misses.',
        independent_scope='Fresh images within two pinned benchmarks; not full official benchmark, unseen-app or pretraining-independence guarantee.',
        mainline_unchanged=True,html=False)
    write(path,spec);return spec


def wait_and_freeze():
    while not (FULL/'completion.json').exists() or not (GEO/'completion.json').exists():
        for folder in [FULL,GEO]:
            p=folder/'status.json'
            if p.exists() and read(p).get('phase')=='failed':raise RuntimeError('Dependency failed: '+str(folder))
        status('waiting_for_full_training_and_geometry');time.sleep(50)
    complete=read(FULL/'completion.json')
    for p,expected in complete['artifact_sha256'].items():assert sha(p)==expected,p
    gcomplete=read(GEO/'completion.json')
    for p,expected in gcomplete['artifact_sha256'].items():assert sha(p)==expected,p
    new=read(GEO/'evaluation/new_development.json')['methods']
    old=read(GEO/'evaluation/old_val.json')['methods']
    candidates=[m for m in ['coarse_center','half_to_center','child_center']
        if new[m]['mean_accuracy_percent']>new['original']['mean_accuracy_percent']
        and old[m]['mean_accuracy_percent']>=old['original']['mean_accuracy_percent']]
    chosen=max(candidates,key=lambda m:new[m]['mean_accuracy_percent']) if candidates else 'original'
    frozen=read(FULL/'models/frozen.json')
    assert frozen['complete'] and len(frozen['models'])==9
    value=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(OUT/'protocol.json'),
        models=frozen['models'],geometry_mode=chosen,geometry_development_sha256={
            str(GEO/f'evaluation/{c}.json'):sha(GEO/f'evaluation/{c}.json') for c in ['old_val','new_development']},
        model_freeze_sha256=sha(FULL/'models/frozen.json'),full_completion_sha256=sha(FULL/'completion.json'),
        policy_selected_using_development_only=True)
    p=OUT/'freeze.json'
    if p.exists():
        previous=read(p)
        for field in ['models','geometry_mode','model_freeze_sha256','protocol_sha256']:
            assert previous[field]==value[field]
    else:
        assert not list((OUT/'predictions').rglob('*.json'))
        for model in value['models'].values():assert sha(model['path'])==model['sha256']
        write(p,value)
    return read(p)


def worker(method):
    if (OUT/f'predictions/{method}.json').exists():
        assert read(OUT/f'predictions/{method}.json')['complete'];return
    with (OUT/f'{method}.stdout.log').open('a',encoding='utf-8') as out,(OUT/f'{method}.stderr.log').open('a',encoding='utf-8') as err:
        proc=subprocess.Popen([sys.executable,'-B','-X','utf8','-u',str(ROOT/'scripts/run_mvp_confirmation_worker_v18.py'),'--method',method],cwd=ROOT,stdout=out,stderr=err)
        while proc.poll() is None:
            p=OUT/'worker_status.json';child=read(p) if p.exists() else {}
            status('inference',method=method,child_pid=proc.pid,child_phase=child.get('phase'),completed=child.get('completed'),total=child.get('total'))
            time.sleep(30)
        if proc.returncode:raise RuntimeError(f'{method} failed with exit{proc.returncode}; inspect '+str(OUT/f'{method}.stderr.log'))


def analyze():
    import numpy as np
    cases=[json.loads(s) for s in (OUT/'data/heldout.jsonl').read_text(encoding='utf-8').splitlines()]
    own={r['sample_id']:r for r in read(OUT/'predictions/ours.json')['rows']}
    mvp={r['sample_id']:r for r in read(OUT/'predictions/mvp.json')['rows']}
    ids=[c['sample_id'] for c in cases]
    assert len(ids)==280 and set(own)==set(mvp)==set(ids)
    values={name:np.array([own[sid]['hits'][name] for sid in ids],float).mean(1) for name in own[ids[0]]['hits']}
    values['MVP']=np.array([mvp[sid]['hit'] for sid in ids],float)
    rng=np.random.default_rng(20261105);groups=[np.array([i for i,c in enumerate(cases) if c['dataset']==source]) for source in sorted({c['dataset'] for c in cases})]
    draws=np.stack([np.concatenate([g[rng.integers(0,len(g),len(g))] for g in groups]) for _ in range(2000)])
    comparisons={}
    for name,v in values.items():
        if name=='MVP':continue
        diff=v-values['MVP'];boot=diff[draws].mean(1)
        primary=name in ['full_union','full_geometry']
        bounds=np.quantile(boot,[.0125,.9875] if primary else [.025,.975])
        comparisons[name]=dict(delta_pp=100*float(diff.mean()),ci_pp=(100*bounds).tolist(),
            primary=primary,interval_level_percent=97.5 if primary else 95,
            status='higher_than_MVP' if bounds[0]>0 else 'lower_than_MVP' if bounds[1]<0 else 'inconclusive')
    metrics={name:dict(accuracy_percent=100*float(v.mean()),by_source={source:100*float(v[[i for i,c in enumerate(cases) if c['dataset']==source]].mean()) for source in sorted({c['dataset'] for c in cases})}) for name,v in values.items()}
    result=dict(complete=True,n=280,geometry_mode=read(OUT/'freeze.json')['geometry_mode'],methods=metrics,comparisons=comparisons,
        scope=read(OUT/'protocol.json')['independent_scope'],no_latency_claim=True,mainline_unchanged=True)
    write(OUT/'results.json',result)
    lines=['新280张图像隔离确认：相同8B骨干与单视图像素上限，实际总计算量并未完全匹配。',
        '本方法准确率为固定三种子均值，不是集成；MVP为固定公开检查点。',f'冻结的坐标规则：{result["geometry_mode"]}']
    for name,metric in metrics.items():
        line=f'{name}: {metric["accuracy_percent"]:.2f}%'
        if name in comparisons:
            c=comparisons[name];line+=f'; 相对MVP {c["delta_pp"]:+.2f}百分点, {c["interval_level_percent"]}%区间{c["ci_pp"]}; {c["status"]}'
        lines.append(line)
    lines.append('该结果只涉及两个基准子集的静态定位，不能外推为完整榜单、执行成功率或速度优势。')
    (OUT/'results.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    files=[OUT/p for p in ['protocol.json','freeze.json','predictions/ours.json','predictions/mvp.json','results.json','results.txt']]
    write(OUT/'completion.json',dict(complete=True,completed_at_utc=datetime.now(timezone.utc).isoformat(),
        artifact_sha256={str(p):sha(p) for p in files},mainline_unchanged=True))
    status('complete',n=280);print('\n'.join(lines),flush=True)


def main():
    register();wait_and_freeze()
    for method in ['ours','mvp']:worker(method)
    register();analyze()


if __name__=='__main__':
    try:main()
    except BaseException:
        status('failed',error=traceback.format_exc());raise
