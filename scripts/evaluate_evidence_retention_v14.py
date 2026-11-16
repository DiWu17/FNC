"""Experiment 3: frozen downstream heads/verifier on matched V14 pools.

Registration and execution require experiment 2's completed result. GUI workers
receive only public images/instructions/points; no target labels or geometry.
"""
from __future__ import annotations
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
import numpy as np
import torch
from PIL import Image
from clickagent.internal_candidate_scorer import subset_internal_inputs,internal_inputs_to_torch
from clickagent.multiregion_scorer import build_multiregion_inputs,load_multiregion_scorer
from clickagent.public_evaluation import point_in_geometry
from run_evidence_retention_v14 import (OUT,LEGACY,read,write as original_write,sha,key,cases,paired,require,completion_hashes,render_report)

STAGE=OUT/'experiment3'
POLICIES=('baseline','rank_s1','rank_s2','aligned_s1','aligned_s2')
HEADS=('c3_top1_s1_loso','c3_top1_s2_loso')


def write(path,value):
    """Retry transient Windows sharing locks without changing output bytes."""
    for attempt in range(20):
        try:
            return original_write(path,value)
        except PermissionError:
            if attempt==19:raise
            time.sleep(min(.05*(attempt+1),.5))


def verify_execution_sources():
    expected=read(STAGE/'protocol.json')['source_identity'];actual=identities()
    if expected==actual:return
    receipt=read(STAGE/'execution_io_compat.json')
    require(receipt['original_source_identity']==expected and receipt['corrected_source_identity']==actual,'I/O repair identity mismatch')
    changed=[p for p in expected if expected[p]!=actual.get(p)]
    require(set(actual)==set(expected) and len(changed)==1 and Path(changed[0]).as_posix()=='scripts/evaluate_evidence_retention_v14.py','Only the new experiment3 I/O driver may change')
    original=STAGE/'source_archive/evaluate_evidence_retention_v14.py'
    require(sha(original)==expected[changed[0]],'Original registered source was not preserved')
    require(receipt['changes_models_candidates_metrics_or_protocol'] is False,'Non-I/O compatibility change')


def public_job(case,points):
    return dict(sample_id=case['sample_id'],image=case['image'],image_sha256=case['image_sha256'],
        instruction=case['instruction'],points=np.asarray(points,np.int64).reshape(-1,2).tolist(),split='val')


def choose_gui(points,scores):
    if not points:return None
    require(all(tuple(p) in scores and np.isfinite(scores[tuple(p)]) for p in points),'Missing/nonfinite shared GUI score')
    return max(points,key=lambda p:scores[tuple(p)])


def policy_points(row,policy):
    return np.asarray(row['points'],np.int64).reshape(-1,2)[sorted(row['pools'][policy])].tolist()


def render_final_overview(result):
    import html
    retention=read(OUT/'experiment2/results.json')['val']['matched_coverage']
    parts=['<section><h2>第三轮：固定选择器的最终命中 / 172</h2>',
        '<p>下表均为同一开发验证集、逐题相同实际候选数。S1为主，S2固定稳健性；GUI读取同一份逐点分数。</p>',
        '<table><tr><th>保留器 / 冻结头</th><th>K16可达</th><th>内部Top1</th><th>Top4可达</th><th>Top4→GUI</th><th>完整池GUI</th></tr>']
    details=read(STAGE/'details.json')
    for policy,head in [('baseline',HEADS[0]),('baseline',HEADS[1]),('rank_s1',HEADS[0]),('rank_s2',HEADS[1]),('aligned_s1',HEADS[0]),('aligned_s2',HEADS[1])]:
        stem=policy+'__'+head
        values=[retention[policy],result['summary'][stem+'__top1']['hits'],
            sum(d['methods'][stem+'__top4_gui']['top4_coverage'] for d in details),
            result['summary'][stem+'__top4_gui']['hits'],result['summary'][policy+'__gui']['hits']]
        parts.append('<tr><td>'+html.escape(policy+' / '+('S1' if head==HEADS[0] else 'S2'))+'</td>'+''.join('<td>'+str(v)+'</td>' for v in values)+'</tr>')
    parts.append('</table><p>以上为静态点命中；没有真实点击、权限/禁用/目标缺失评估。候选可达性、Top4保留、最终选择分别统计。</p>')
    parts.append('<table><tr><th>配对比较</th><th>赢 / 输</th><th>差值（百分点）</th><th>95%区间（百分点）</th></tr>')
    for name,c in result['comparisons'].items():
        parts.append('<tr><td>'+html.escape(name)+'</td><td>'+f"{c['wins']} / {c['losses']}"+'</td><td>'+f"{100*c['delta']:+.2f}"+'</td><td>'+f"[{100*c['ci95'][0]:.2f}, {100*c['ci95'][1]:.2f}]"+'</td></tr>')
    parts.append('</table><p>区间为固定已训练模型下按来源分层、RGB截图簇重采样；不包含训练随机性，也不消除验证集历史复用。多个次要比较未作多重校正。</p></section>')
    path=OUT/'report.html';text=path.read_text(encoding='utf-8')
    position=text.find('<h2>问题三：')
    text=text[:position]+''.join(parts)+text[position:] if position>=0 else text.replace('</body>',''.join(parts)+'</body>')
    path.write_text(text,encoding='utf-8')


def identities():
    paths=[Path(__file__),ROOT/'scripts/run_evidence_retention_v14.py',ROOT/'src/clickagent/evidence_retention.py',
        ROOT/'src/clickagent/multiregion_scorer.py',ROOT/'src/clickagent/internal_candidate_scorer.py',
        ROOT/'scripts/run_gui_verifier.py',ROOT/'src/clickagent/point_aligned_inference.py']
    return {str(p.relative_to(ROOT)):sha(p) for p in paths}


def pools():
    return [json.loads(l) for l in (OUT/'experiment2/pools.jsonl').read_text(encoding='utf-8').splitlines()
            if l.strip() and json.loads(l)['split']=='val']


def register():
    completed=read(OUT/'experiment2/completion.json')
    require(completed['complete'],'Need completed experiment2 first')
    require(sha(OUT/'experiment2/results.json')==completed['results_sha256'] and
            sha(OUT/'experiment2/pools.jsonl')==completed['pools_sha256'],'Experiment2 result changed')
    path=STAGE/'protocol.json'
    if path.exists():verify_execution_sources();return
    spec=dict(question='保留机制是否在相同实际候选数下，经固定下游选择转化为最终点命中？',
        data='Original172 historically reused validation queries,128 RGB groups. Development result, not fresh independent test.',
        policies=list(POLICIES),seeds='S1 primary, S2 robustness; all preselected arms evaluated regardless of prior result sign.',
        frozen_heads=list(HEADS),head_design='Both heads evaluate every pool; primary pairs aligned_s1 vs rank_s1/baseline with headS1, aligned_s2 vs rank_s2/baseline with headS2. Crossed combinations also saved.',
        head_device='Canonical CPU float32 inference, torch4 threads, consistent with V13 audit reference.',
        outcomes=['forced internal Top1','GUI verifier over whole matched pool','fixed internal Top4 followed by same GUI point scores'],
        control='Identical frozen weights, candidate count per query, source screenshots, instructions, marker, pixels, pool canonical order and geometry scorer across policies.',
        GUI=dict(model='GUI-Actor-Verifier-2B',mode='crop',marker='existing marked_crop, side768, radius12 width3',max_pixels=262144,
            dtype='float16',attention='sdpa',max_batch=6,physical_devices=[0,1],parallel_anchors='same first3 hash-ordered nonempty jobs, first6 points; require max abs difference<=1e-5 before case work'),
        cost='Score each case/coordinate once over union of fixed pools; all policies read common scores. This is controlled counterfactual selection, not separately timed end-to-end execution.',
        learning='No retraining, epoch selection, policy selection or GUI teacher training in this experiment.',
        frozen_head_selection_sha256=sha(LEGACY/'models/selection.json'),
        verifier_asset_manifest_sha256=sha(ROOT/'models/GUI-Actor-Verifier-2B/clickagent_download_verified.json'),
        retention_model_sha256={n:sha(OUT/f'models/{n}.pt') for n in POLICIES if n!='baseline'},
        second_experiment_results_sha256=sha(OUT/'experiment2/results.json'),
        second_experiment_pools_sha256=sha(OUT/'experiment2/pools.jsonl'),
        source_identity=identities(),legacy_completion_hashes=completion_hashes(),created_at_utc=datetime.now(timezone.utc).isoformat())
    STAGE.mkdir(parents=True,exist_ok=True)
    for p in (Path(__file__),ROOT/'scripts/run_evidence_retention_v14.py',ROOT/'src/clickagent/evidence_retention.py'):
        folder=STAGE/'source_archive';folder.mkdir(exist_ok=True);(folder/p.name).write_bytes(p.read_bytes())
    write(path,spec)
    write(OUT/'work_state.json',dict(phase='experiment3_registered',complete=False))


def raw_view(view,ledger):
    path=Path(view['feature_path']);canonical=os.path.normcase(os.path.abspath(path))
    data=path.read_bytes();actual=hashlib.sha256(data).hexdigest()
    require(actual==view['feature_sha256']==ledger[canonical],'Raw archive changed')
    with np.load(io.BytesIO(data),allow_pickle=False) as z:
        meta=json.loads(str(z['metadata']))
        require(meta['ground_truth_in_model_input'] is False,'GT in feature extraction')
        arrays={k:z[k].copy() for k in ('target_attention','target_hidden','image_hidden','vision_hidden')}
    return arrays,meta


@torch.inference_mode()
def head_predictions(device):
    require(device=='cpu','Use canonical CPU head inference for this protocol')
    register();selection=read(LEGACY/'models/selection.json')['selections']
    models={}
    for name in HEADS:
        e=selection[name];require(sha(e['checkpoint_path'])==e['sha256'],'Frozen head changed')
        models[name]=load_multiregion_scorer(e['checkpoint_path'],device)[0]
    cohort={c['sample_id']:c for c in cases('val')}
    ledger=read(OUT/'states/index.json')['raw_sha256']
    jobs=[];base_entries={e['sample_id']:e for e in read(OUT/'base/index.json')['cases'] if e['split']=='val'}
    started=time.perf_counter();checked=0
    for i,row in enumerate(pools(),1):
        case=cohort[row['sample_id']];sid=key(row['sample_id']);dest=STAGE/f'heads/{sid}.json'
        union_indices=sorted({int(ix) for name in POLICIES for ix in row['pools'][name]})
        all_points=np.asarray(row['points'],np.int64).reshape(-1,2)
        points=all_points[union_indices];lookup={ix:j for j,ix in enumerate(union_indices)}
        input_sha=sha(OUT/'experiment2/pools.jsonl')
        if not dest.exists():
            entry=base_entries[row['sample_id']];pipe=read(entry['pipeline_path'])
            require(sha(entry['pipeline_path'])==entry['pipeline_sha256'],'Pipeline changed')
            global_view=dict(feature_path=pipe['global_feature_path'],feature_sha256=pipe['global_feature_sha256'])
            ga,gm=raw_view(global_view,ledger)
            locals_=[]
            for view in pipe['local_views']:
                a,m=raw_view(view,ledger);locals_.append((a,m,view['transform']))
            inputs=build_multiregion_inputs(ga,gm,locals_,points,case['image_size']) if len(points) else None
            # Compare fresh input reconstruction with the legacy cache on the
            # first12 nonempty cases; this checks all arrays, not only scores.
            if inputs is not None and checked<12:
                old_index=read(LEGACY/'cache/val_c3_index.json')
                old=next(e for e in old_index['cases'] if e['sample_id']==row['sample_id'])
                with np.load(old['path'],allow_pickle=False) as z:
                    old_points=z['points'].tolist();old_lookup={tuple(p):j for j,p in enumerate(old_points)}
                    canonical_indices=sorted(row['pools']['baseline'])
                    canonical_points=points[[lookup[ix] for ix in canonical_indices]].tolist()
                    ix_old=np.asarray([old_lookup[tuple(p)] for p in canonical_points],np.int64)
                    require(set(ix_old.tolist())==set(z['pool_indices_16'].tolist()),'Baseline coordinates changed')
                    current=subset_internal_inputs(inputs,np.asarray([lookup[ix] for ix in canonical_indices]))
                    from clickagent.internal_candidate_scorer import CANDIDATE_KEYS
                    for field,value in current.items():
                        archived=z['i__'+field]
                        expected=archived[ix_old] if field in CANDIDATE_KEYS else archived
                        require(np.array_equal(value,expected),'Legacy input reconstruction mismatch: '+field)
                checked+=1
            predictions={}
            for policy in POLICIES:
                ix=np.asarray([lookup[j] for j in sorted(row['pools'][policy])],np.int64)
                predictions[policy]={}
                for name,model in models.items():
                    if len(ix):
                        public=internal_inputs_to_torch(subset_internal_inputs(inputs,ix),device)
                        before=time.perf_counter();out=model(**public)
                        if device.startswith('cuda'):torch.cuda.synchronize()
                        scores=out['scores'].float().cpu().numpy();require(np.isfinite(scores).all(),'Nonfinite head score')
                        predictions[policy][name]=dict(points=points[ix].tolist(),scores=scores.tolist(),
                            chosen_point=points[ix[int(scores.argmax())]].tolist(),elapsed_cached_head_seconds=time.perf_counter()-before,
                            checkpoint_sha256=selection[name]['sha256'])
                    else:predictions[policy][name]=dict(points=[],scores=[],chosen_point=None,checkpoint_sha256=selection[name]['sha256'])
            write(dest,dict(sample_id=row['sample_id'],predictions=predictions,pools_sha256=input_sha,
                protocol_sha256=sha(STAGE/'protocol.json'),ground_truth_in_model_inputs=False))
        else:
            saved=read(dest);require(saved['pools_sha256']==input_sha and saved['protocol_sha256']==sha(STAGE/'protocol.json'),'Stale head output')
        jobs.append(public_job(case,points))
        if i%10==0 or i==172:print(json.dumps(dict(stage='frozen_heads',i=i,n=172,seconds=time.perf_counter()-started)),flush=True)
    write(STAGE/'gui_jobs.json',dict(complete=True,cases=jobs,contains_gt_geometry_or_labels=False,
        point_count=sum(len(j['points']) for j in jobs),protocol_sha256=sha(STAGE/'protocol.json')))
    write(STAGE/'head_completion.json',dict(complete=True,n=172,legacy_input_exact_reconstructions=checked,
        frozen_head_sha256={n:selection[n]['sha256'] for n in HEADS},source_identity=identities(),
        unique_case_points=sum(len(j['points']) for j in jobs)))


def canonical_runtime(meta):
    value=json.loads(json.dumps(meta))
    value['loading']['unexpected_keys']=sorted(value['loading']['unexpected_keys'])
    return value


def gui_worker(part):
    from run_gui_verifier import VerifierRuntime
    verify_execution_sources()
    jobs=read(STAGE/'gui_jobs.json')['cases'];runtime=VerifierRuntime()
    write(STAGE/f'gui/runtime_{part}.json',canonical_runtime(runtime.metadata))
    anchor=[]
    for job in sorted(jobs,key=lambda r:key(r['sample_id'])):
        if not job['points']:continue
        require(sha(job['image'])==job['image_sha256'],'GUI image identity changed')
        with Image.open(job['image']) as im:
            values=runtime.score(im.convert('RGB'),job['instruction'],job['points'][:6],'crop')
        anchor.extend(dict(sample_id=job['sample_id'],point=r['point'],log_odds=r['log_odds']) for r in values)
        if len({r['sample_id'] for r in anchor})==3:break
    write(STAGE/f'gui/anchor_{part}.json',anchor)
    while not (STAGE/'gui/acceptance.json').exists():
        if (STAGE/'gui/stop.json').exists():raise RuntimeError('Parent stopped anchor acceptance')
        time.sleep(1)
    acceptance=read(STAGE/'gui/acceptance.json')
    require(acceptance['protocol_sha256']==sha(STAGE/'protocol.json'),'Anchor acceptance identity changed')
    for i,job in enumerate(jobs):
        if i%2!=part:continue
        dest=STAGE/f'gui/cases/{key(job["sample_id"])}.json'
        identity=hashlib.sha256(json.dumps(job,sort_keys=True,ensure_ascii=False).encode()).hexdigest()
        if dest.exists():
            previous=read(dest);require(previous['input_sha256']==identity,'GUI input changed')
            if previous['complete']:
                write(STAGE/f'gui/done/{key(job["sample_id"])}.json',dict(complete=True,sample_id=job['sample_id']))
                continue
            rows=previous['scores']
        else:rows=[]
        done={tuple(r['point']) for r in rows}
        pending=[p for p in job['points'] if tuple(p) not in done]
        if pending:
            require(sha(job['image'])==job['image_sha256'],'GUI image SHA changed')
            with Image.open(job['image']) as im:image=im.convert('RGB')
            for start in range(0,len(pending),6):
                batch=pending[start:start+6]
                rows.extend(runtime.score(image,job['instruction'],batch,'crop'))
                write(dest,dict(complete=len(rows)==len(job['points']),sample_id=job['sample_id'],scores=rows,
                    input_sha256=identity,protocol_sha256=sha(STAGE/'protocol.json'),physical_device=part,
                    ground_truth_in_model_inputs=False))
            image.close()
        else:
            write(dest,dict(complete=True,sample_id=job['sample_id'],scores=rows,input_sha256=identity,
                protocol_sha256=sha(STAGE/'protocol.json'),physical_device=part,ground_truth_in_model_inputs=False))
        write(STAGE/f'gui/done/{key(job["sample_id"])}.json',dict(complete=True,sample_id=job['sample_id']))
        print(json.dumps(dict(stage='gui_case',part=part,i=i+1,n=len(jobs),points=len(rows))),flush=True)
    write(STAGE/f'gui/worker_{part}_complete.json',dict(complete=True,part=part,protocol_sha256=sha(STAGE/'protocol.json')))


def run_gui():
    require(read(STAGE/'head_completion.json')['complete'],'Need heads and public GUI jobs')
    folder=STAGE/'gui';folder.mkdir(parents=True,exist_ok=True)
    # Existing acceptance may be reused only after checking the same contract.
    if (folder/'acceptance.json').exists():
        require(read(folder/'acceptance.json')['protocol_sha256']==sha(STAGE/'protocol.json'),'Stale acceptance')
    processes=[];streams=[];started=time.perf_counter();last_progress=-1
    try:
        for part in (0,1):
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(part),PYTHONDONTWRITEBYTECODE='1',PYTHONUTF8='1')
            stream=(folder/f'worker_{part}.log').open('a',encoding='utf-8');streams.append(stream)
            cmd=[sys.executable,'-B','-X','utf8','-u',str(Path(__file__).resolve()),'--stage','gui-worker','--part',str(part)]
            flags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0
            processes.append(subprocess.Popen(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,creationflags=flags))
        while not all((folder/f'anchor_{p}.json').exists() for p in (0,1)):
            require(all(p.poll() is None for p in processes),'GUI worker failed before anchors; inspect worker logs')
            time.sleep(1)
        anchors=[read(folder/f'anchor_{p}.json') for p in (0,1)]
        require([(r['sample_id'],r['point']) for r in anchors[0]]==[(r['sample_id'],r['point']) for r in anchors[1]],'Anchor coordinate mismatch')
        diffs=[abs(a['log_odds']-b['log_odds']) for a,b in zip(*anchors)]
        require(max(diffs,default=0)<=1e-5,'GPU numerical anchor disagreement')
        require(read(folder/'runtime_0.json')==read(folder/'runtime_1.json'),'GPU runtime mismatch')
        write(folder/'acceptance.json',dict(complete=True,protocol_sha256=sha(STAGE/'protocol.json'),
            same_label_free_anchors=len(diffs),max_abs_log_odds_difference=max(diffs,default=0),atol=1e-5,
            runtime_sha256=sha(folder/'runtime_0.json'),physical_devices=[0,1]))
        print(json.dumps(dict(stage='gui_anchors_accepted',points=len(diffs),max_difference=max(diffs,default=0))),flush=True)
        while any(p.poll() is None for p in processes):
            require(all(p.poll() in (None,0) for p in processes),'GUI worker failed; inspect worker logs')
            # Immutable done markers avoid opening a case file while a worker
            # atomically replaces it. The former progress scan caused WinError5.
            complete=len(list((folder/'done').glob('*.json')))
            if complete!=last_progress:
                last_progress=complete
                print(json.dumps(dict(stage='gui_progress',complete_cases=complete,n=172,seconds=time.perf_counter()-started)),flush=True)
            time.sleep(10)
        require(all(p.returncode==0 for p in processes),'GUI worker exit failure')
    except BaseException:
        write(folder/'stop.json',dict(stop=True))
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            try:p.wait(timeout=10)
            except subprocess.TimeoutExpired:p.kill();p.wait()
        raise
    finally:
        for stream in streams:stream.close()
    write(folder/'completion.json',dict(complete=True,n=172,elapsed_union_wall_seconds=time.perf_counter()-started,
        shared_union_point_count=read(STAGE/'gui_jobs.json')['point_count'],protocol_sha256=sha(STAGE/'protocol.json')))


def hit(point,case):
    if point is None:return False
    return bool(point_in_geometry([point[0]/case['image_size'][0],point[1]/case['image_size'][1]],case['target_geometry']))


def summarize():
    require(read(STAGE/'gui/completion.json')['complete'],'GUI incomplete')
    cohort={c['sample_id']:c for c in cases('val')};entries=[];details=[];vectors={}
    for row in pools():
        case=cohort[row['sample_id']];sid=key(row['sample_id'])
        head=read(STAGE/f'heads/{sid}.json');gui=read(STAGE/f'gui/cases/{sid}.json')
        require(gui['complete'],'Incomplete GUI case')
        gs={tuple(r['point']):r['log_odds'] for r in gui['scores']}
        require(len(gs)==len(gui['scores']),'Repeated GUI coordinate')
        methods={};entries.append(dict(source=case['dataset'],group=case['pixel_sha256']))
        for policy in POLICIES:
            pool=policy_points(row,policy)
            coverage=any(hit(p,case) for p in pool)
            picked=choose_gui(pool,gs)
            name=policy+'__gui';methods[name]=dict(point=picked,hit=hit(picked,case),coverage=coverage)
            for hn in HEADS:
                pred=head['predictions'][policy][hn];require(pred['points']==pool,'Head pool changed')
                own=policy+'__'+hn+'__top1';methods[own]=dict(point=pred['chosen_point'],hit=hit(pred['chosen_point'],case),coverage=coverage)
                order=np.argsort(-np.asarray(pred['scores']),kind='stable')[:4]
                shortlist=[pool[j] for j in sorted(order.tolist())]
                p=choose_gui(shortlist,gs)
                name=policy+'__'+hn+'__top4_gui'
                methods[name]=dict(point=p,hit=hit(p,case),coverage=coverage,top4_coverage=any(hit(p,case) for p in shortlist))
            for value in methods.values():require(not value['hit'] or value['coverage'],'Hit outside reachable pool')
        for name,v in methods.items():vectors.setdefault(name,[]).append(v['hit'])
        details.append(dict(sample_id=case['sample_id'],source=case['dataset'],methods=methods))
    summary={name:dict(hits=sum(v),n=172,accuracy=sum(v)/172) for name,v in vectors.items()}
    comparisons={}
    def compare(a,b):
        value=paired(vectors[a],vectors[b],entries)
        ca=np.asarray([d['methods'][a]['coverage'] for d in details],bool)
        cb=np.asarray([d['methods'][b]['coverage'] for d in details],bool)
        ha=np.asarray(vectors[a],bool);hb=np.asarray(vectors[b],bool)
        value['attribution']=dict(pool_coverage_gained=int((ca&~cb).sum()),pool_coverage_lost=int((cb&~ca).sum()),
            wins_newly_reachable=int((ha&~hb&~cb).sum()),wins_both_pools_reachable=int((ha&~hb&cb).sum()),
            losses_no_longer_reachable=int((hb&~ha&~ca).sum()),losses_both_pools_reachable=int((hb&~ha&ca).sum()))
        require(value['attribution']['wins_newly_reachable']+value['attribution']['wins_both_pools_reachable']==value['wins'],'Win attribution mismatch')
        require(value['attribution']['losses_no_longer_reachable']+value['attribution']['losses_both_pools_reachable']==value['losses'],'Loss attribution mismatch')
        return value
    for s in (1,2):
        hn=HEADS[s-1]
        for ending in ('__gui','__'+hn+'__top1','__'+hn+'__top4_gui'):
            for ref in ('baseline',f'rank_s{s}'):
                a=f'aligned_s{s}'+ending;b=ref+ending
                comparisons[a+' vs '+b]=compare(a,b)
        for ending in ('__gui','__'+hn+'__top1','__'+hn+'__top4_gui'):
            a=f'rank_s{s}'+ending;b='baseline'+ending
            comparisons[a+' vs '+b]=compare(a,b)
    result=dict(complete=True,n=172,summary=summary,comparisons=comparisons,
        by_source={source:{name:sum(hit_ for hit_,d in zip(v,details) if d['source']==source) for name,v in vectors.items()} for source in sorted({d['source'] for d in details})},
        scope='Historically reused development validation; fixed downstream weights and matched per-case point count. No fresh independent test and no action execution.')
    write(STAGE/'results.json',result);write(STAGE/'details.json',details)
    verify_execution_sources()
    legacy=completion_hashes()
    # Authenticate all pretest-sealed source/model files again, without running
    # the write-back V13 audit CLI or reading the54GB evidence ledger anew.
    old_frozen=read(LEGACY/'frozen.json')
    for name,expected in old_frozen['files'].items():require(sha(name)==expected,'V13 sealed file changed: '+name)
    write(STAGE/'completion.json',dict(complete=True,protocol_sha256=sha(STAGE/'protocol.json'),
        results_sha256=sha(STAGE/'results.json'),details_sha256=sha(STAGE/'details.json'),
        legacy_completion_hashes=legacy,sealed_files_rehashed=len(old_frozen['files']),source_identity=identities()))
    write(OUT/'work_state.json',dict(complete=True,phase='three_development_experiments_complete',
        experiments={str(i):sha(OUT/f'experiment{i}/results.json') for i in (1,2,3)},fresh_independent_test=False))
    render_report()
    render_final_overview(result)
    print(json.dumps(dict(stage='experiment3',complete=True,summary={k:v for k,v in summary.items() if '__gui' in k or k.endswith('__top1')})),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=('register','heads','gui','gui-worker','summarize','all'),required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--part',type=int,choices=(0,1));args=p.parse_args()
    os.chdir(ROOT);torch.set_num_threads(4)
    if args.stage=='gui-worker':require(args.part is not None,'Need worker part');gui_worker(args.part);return
    register()
    if args.stage in ('heads','all'):head_predictions(args.device)
    if args.stage in ('gui','all'):run_gui()
    if args.stage in ('summarize','all'):summarize()


if __name__=='__main__':main()
