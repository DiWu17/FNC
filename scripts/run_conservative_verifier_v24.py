"""V24: expanded candidate views and conservative replacement, local Python.

All visual decisions are saved before labels are opened. A small replacement
gate is subsequently assessed by application-group out-of-fold development
predictions; this is explicitly not an independent test or VLM fine-tuning.
"""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('HF_HUB_OFFLINE', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import numpy as np
from PIL import Image, ImageDraw
from clickagent.conservative_verifier_v24 import (K, MARGIN, FEATURE_NAMES, make_pool,
    order_for, visual_inputs, combine, features, application_folds, fit_gate, predict_utility)
from clickagent.visual_feedback_v23 import digest
from run_visual_feedback_v23 import read, write, sha, hit, bootstrap

OUT = ROOT/'runs/conservative_verifier_v24'
PRIOR = ROOT/'runs/visual_feedback_v23'
V21 = ROOT/'runs/mvp_repair_v21'
SOURCES = [Path(__file__), ROOT/'src/clickagent/conservative_verifier_v24.py',
           ROOT/'tests/test_conservative_verifier_v24.py']
FAMILIES = ['original3k', 'full']
NAMES = [f'{f}_s{s}' for f in FAMILIES for s in [1, 2, 3]]
ARMS = ['baseline', 'v23_verify', 'expanded_ungated', 'expanded_fixed', 'expanded_learned_oof']


def now(): return datetime.now(timezone.utc).isoformat()
def source_identity(): return {str(p): sha(p) for p in SOURCES}


def status(phase, **kwargs):
    row = dict(phase=phase, complete=phase=='complete', updated_at_utc=now(), pid=os.getpid(), **kwargs)
    write(OUT/'status.json', row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def verify():
    p = read(OUT/'protocol.json')
    assert p['source_sha256'] == source_identity(), 'Registered V24 source changed'
    for path, value in p['sealed_sha256'].items(): assert sha(path) == value, path
    for path, value in p['backbone_identity'].items():
        st = Path(path).stat()
        assert st.st_size == value['bytes'] and st.st_mtime_ns == value['mtime_ns'], path
    return p


def prepare():
    if (OUT/'protocol.json').exists(): return verify()
    test = read(OUT/'unit_tests.json')
    assert test['exit_code'] == 0 and test['source_sha256'] == source_identity()
    smoke = read(OUT/'smoke/receipt.json')
    assert smoke['complete'] and smoke['source_sha256'] == source_identity()
    prior = read(PRIOR/'protocol.json')
    # All sealed V23 data/source/heads remain intact; no historical result edits.
    for path, value in prior['sealed_sha256'].items(): assert sha(path) == value, path
    for path, value in prior['backbone_identity'].items():
        st = Path(path).stat()
        assert st.st_size == value['bytes'] and st.st_mtime_ns == value['mtime_ns']
    cases = read(PRIOR/'data/public.json')
    proposals = read(PRIOR/'data/proposals.json')
    assert len(cases) == 100 and len({c['group'] for c in cases}) == 100
    enlarged = {}; unique = {}; old_subset = 0
    for c in cases:
        assert sha(c['image']) == c['image_sha256']
        p = proposals[c['sample_id']]
        row = dict(points=p['points'], models={})
        for name in NAMES:
            old = p['models'][name]
            pool = make_pool(p['points'], old['scores'], c['image_size'])
            assert set(map(tuple, old['pool'])) <= set(map(tuple, pool))
            assert not pool or pool[0] == old['anchor']
            old_subset += 1
            canonical = sorted(pool)
            signature = digest([c['sample_id'], canonical, 'v24_symmetric'])[:24] if pool else None
            row['models'][name] = dict(pool=pool, anchor=old['anchor'], scores=old['scores'], signature=signature,
                                      previous_shortlist=old['pool'])
            if pool:
                for reverse in [False, True]:
                    task = dict(sample_id=c['sample_id'], pool=canonical, reverse=reverse,
                                task_id=signature+'_'+str(int(reverse)))
                    unique[task['task_id']] = task
        enlarged[c['sample_id']] = row
    tasks = sorted(unique.values(), key=lambda x: digest([x['task_id'], 'v24_execution']))
    folds = application_folds(cases)
    # Copy already frozen inputs; no target geometry is inspected in preparation.
    for name in ['public', 'labels']:
        path = OUT/f'data/{name}.json'; path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(PRIOR/f'data/{name}.json', path)
    write(OUT/'data/proposals.json', enlarged)
    write(OUT/'data/tasks.json', tasks)
    write(OUT/'data/folds.json', folds)
    write(OUT/'registry_before.json', read(ROOT/'CURRENT_RESEARCH.json'))
    sealed = SOURCES + [OUT/f'data/{name}.json' for name in ['public', 'labels', 'proposals', 'tasks', 'folds']]
    sealed += [OUT/'unit_tests.json', OUT/'smoke/receipt.json', PRIOR/'protocol.json', PRIOR/'predictions.json',
        PRIOR/'results.json', V21/'protocol.json', V21/'results.json', V21/'data/public.json',
        ROOT/'src/clickagent/visual_feedback_v23.py', ROOT/'scripts/run_visual_feedback_v23.py',
        ROOT/'src/clickagent/advantage_runtimes.py', ROOT/'src/clickagent/public_evaluation.py']
    for c in cases:
        key = hashlib.sha256(c['sample_id'].encode()).hexdigest()[:24]
        sealed += [V21/f'mvp/{key}.json', V21/f'cases/{key}.json']
    spec = dict(registered_at_utc=now(), source_sha256=source_identity(),
        sealed_sha256={str(p): sha(p) for p in sealed}, backbone_identity=prior['backbone_identity'],
        cohort='Exactly V23/V21 reused development100: new72 and old28, one query/RGB, zero train RGB overlap inherited and sealed. No fresh280 or new independent test.',
        models=prior['models'], model_names=NAMES, actual_calls=len(tasks), old_shortlist_inclusion_checks=old_subset,
        inference='Frozen Qwen3-VL-8B-Instruct, FP16, two P5000, first-token A..I/N constrained logits; all visual decisions saved before labels are opened.',
        proposal='Keep complete-pool argmax; expand same greedy14px-diverse shortlist rule from3 to9 distinct points. Original3-point sets must be subsets. Canonical pool sets deduplicate calls even when scorer anchors differ.',
        observation='Overview512x512 plus one256-source-pixel crop rendered256x256 per candidate; boundary shifted, no target-derived box. Every candidate has its own local view, no initial-point-only crop. Initial answer is not shown to the VLM. Two deterministic reversed presentation orders. More candidates/context and more compute than V23; joint design pilot, not an isolated component attribution.',
        arms=dict(baseline='Own frozen complete-pool argmax', v23_verify='Read-only previous one-order three-candidate verification',
            expanded_ungated='Mean aligned log probabilities across2 orders, N keeps own anchor',
            expanded_fixed=f'Replace only if both orders choose exactly the same alternative and each winner-vs-all-other-labels margin >= log(3)={MARGIN}. Otherwise keep own anchor.',
            expanded_learned_oof='Same order consensus; replace only if fixed-strength ridge expected replacement utility >0. Five application/episode-group folds, all6 scorer instances of a screenshot in one fold. Train only on proposed changes in other folds, per-RGB total weight1. No hyperparameter or threshold search. Development OOF evaluation, not independent testing.'),
        gate_training=dict(features=list(FEATURE_NAMES), ridge=20., folds=5, targets='new_hit minus original_hit in {-1,0,1}', backbone_training=False, scorer_training=False,
                           limitation='Small reused development data; grouped cross-fitting reduces within-run label leakage but does not erase historical design reuse. Learned gate is supplementary and needs new held-out confirmation.'),
        comparisons='Primary4: expanded_fixed minus own baseline, and minus fixed official MVP, separately3k/full onnew72. 5000 source-stratified RGB bootstrap over3 fixed-fit means,98.75% intervals. Secondary ungated/OOF/V23/native-MVP, source/application sensitivity, old28/all100, repair/harm retention, full/3/9-pool oracles.',
        advancement='No automatic adoption. Candidate signal requires fixed gate vs baseline >=3pp and corrected lower>0, old28 delta>=-3pp. Exceeding MVP additionally requires corrected lower>0 vsMVP. Reused-dev positive still requires a new independent test.',
        compute='Logical2-prefill extra cost per individual scorer, not a6-seed ensemble. Identical canonical pools cached acrossfits. Original MVP and native-verified MVP are previously pinned real outputs on exactly the same screenshots. Extra verification compute differs; no equal-compute or speed-superiority claim.',
        scope='Joint candidate retention/symmetric observation/strict replacement experiment plus small learned utility gate. No 8B fine-tuning, coordinate generation, real action execution, HTML or automatic production replacement.')
    write(OUT/'protocol.json', spec)
    for p in SOURCES:
        dest = OUT/'source_archive'/p.relative_to(ROOT); dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dest)
    reg = read(ROOT/'CURRENT_RESEARCH.json')
    reg['active_experiment'] = dict(run_id=OUT.name, status='registered', phase='prepared', n=100,
        protocol_path=str(OUT/'protocol.json'), status_path=str(OUT/'status.json'), mainline_unchanged=True,
        training='Small application-group OOF utility gate only; original scorers/backbone frozen', actual_feedback_calls=len(tasks))
    write(ROOT/'CURRENT_RESEARCH.json', reg)
    write(ROOT/'research/decisions/20261109_conservative_verifier_v24.json', dict(
        user_authorization='按照你说的来改吧，你觉得现在加上复核器后，我们方法的上限能超过MVP吗',
        registered_at_utc=now(), protocol_sha256=sha(OUT/'protocol.json'), mainline_unchanged=True))
    status('registered', actual_feedback_calls=len(tasks), candidate_budget=K)
    return spec


def sync(torch):
    for device in range(torch.cuda.device_count()): torch.cuda.synchronize(device)


def load_runtime():
    from clickagent.advantage_runtimes import configure_paths, QwenCoordinateAdapter
    configure_paths('coordinate')
    import torch
    torch.set_num_threads(4); torch.manual_seed(20262401)
    runtime = QwenCoordinateAdapter(OUT/'runtime', max_pixels=262144)
    runtime.device = runtime.model.get_input_embeddings().weight.device
    return runtime, torch


def infer(runtime, torch, image, instruction, sid, pool, reverse, preview=None):
    order = order_for(sid, pool, reverse)
    images, messages, geometry = visual_inputs(image, instruction, pool, order)
    if preview:
        canvas = Image.new('RGB', (512+256*3, 768), 'white')
        canvas.paste(images[0], (0, 0))
        for i, im in enumerate(images[1:]): canvas.paste(im, (512+(i%3)*256, (i//3)*256))
        Path(preview).parent.mkdir(parents=True, exist_ok=True); canvas.save(preview); canvas.close()
    sync(torch); before=runtime.counter.snapshot(); started=time.perf_counter()
    text = runtime.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = runtime.processor(text=[text], images=images, return_tensors='pt', padding=False).to(runtime.device)
    grids = inputs['image_grid_thw'].detach().cpu().tolist()
    assert grids == [[1,32,32]]+[[1,16,16]]*len(pool), grids
    with torch.inference_mode():
        output = runtime.model(**inputs, use_cache=False, logits_to_keep=1, return_dict=True)
        logits = output.logits[0,-1].float().cpu(); del output
    groups=[]
    for label in geometry['labels']:
        group=set()
        for prefix in ['', ' ']:
            ids=runtime.processor.tokenizer.encode(prefix+label, add_special_tokens=False)
            if len(ids)==1: group.add(ids[0])
        assert group; groups.append(sorted(group))
    assert len({i for group in groups for i in group}) == sum(map(len, groups))
    scores = torch.stack([torch.logsumexp(logits[group], 0) for group in groups])
    assert torch.isfinite(scores).all()
    mass=float((torch.logsumexp(scores,0)-torch.logsumexp(logits,0)).exp())
    sync(torch); after=runtime.counter.snapshot()
    result=dict(order=order, scores=scores.tolist(), geometry=geometry, valid_label_mass=mass,
        input_tokens=int(inputs['input_ids'].shape[-1]), image_grids=grids, elapsed_seconds=time.perf_counter()-started,
        counter={key:after[key]-before[key] for key in before}, ground_truth_in_model_inputs=False)
    assert result['counter']['backbone_forward_calls']==1
    assert result['counter']['processed_pixels']==geometry['input_pixels']
    for im in images: im.close()
    return result


def smoke():
    status('synthetic_interface_check_loading')
    runtime, torch = load_runtime()
    image=Image.new('RGB', (1200,800), 'white'); draw=ImageDraw.Draw(image); pool=[]
    for i in range(9):
        x,y=180+(i%3)*400,130+(i//3)*260; pool.append([x,y])
        draw.rectangle((x-55,y-40,x+55,y+40),fill=(20,70,220) if i==7 else (220,20,20))
    outputs=[infer(runtime,torch,image,'Click the only blue rectangle.', 'synthetic9',pool,rev,
                   OUT/f'smoke/order_{int(rev)}.png') for rev in [False,True]]
    result=combine(pool,outputs)
    write(OUT/'smoke/receipt.json',dict(complete=True,source_sha256=source_identity(),at_utc=now(),
        synthetic_only=True, outputs=outputs, combined=result, semantic_success=result['unconditional_point']==pool[7],
        runtime_metadata=runtime.metadata))
    image.close(); status('synthetic_interface_check_complete', semantic_success=result['unconditional_point']==pool[7])


def worker():
    spec=verify(); tasks=read(OUT/'data/tasks.json'); cases={c['sample_id']:c for c in read(OUT/'data/public.json')}
    protocol_sha=sha(OUT/'protocol.json'); runtime=torch=None; started=time.perf_counter(); completed=0
    for task in tasks:
        path=OUT/f'calls/{task["task_id"]}.json'
        if path.exists():
            saved=read(path); assert saved['task']==task and saved['protocol_sha256']==protocol_sha
            completed+=1; continue
        if runtime is None:
            status('loading_frozen_visual_verifier', completed=completed, total=len(tasks))
            runtime,torch=load_runtime();write(OUT/'runtime_metadata.json',runtime.metadata)
        c=cases[task['sample_id']]; assert sha(c['image'])==c['image_sha256']
        with Image.open(c['image']) as opened: image=opened.convert('RGB')
        assert list(image.size)==c['image_size']
        preview=OUT/f'preview/{task["task_id"]}.png' if completed<2 else None
        result=infer(runtime,torch,image,c['instruction'],c['sample_id'],task['pool'],task['reverse'],preview)
        image.close()
        write(path,dict(task=task,output=result,protocol_sha256=protocol_sha,image_sha256=c['image_sha256']))
        completed+=1
        status('visual_verification',completed=completed,total=len(tasks),elapsed_seconds=time.perf_counter()-started,
               last_call_seconds=result['elapsed_seconds'],last_candidates=len(task['pool']))
    write(OUT/'inference_complete.json',dict(complete=True,at_utc=now(),calls=len(tasks),protocol_sha256=protocol_sha,
        label_geometry_not_loaded_in_worker=True))
    status('visual_inference_complete',completed=completed,total=len(tasks))


def analyze():
    verify(); assert read(OUT/'inference_complete.json')['complete']
    cases=read(OUT/'data/public.json'); labels=read(OUT/'data/labels.json')
    proposals=read(OUT/'data/proposals.json'); folds=read(OUT/'data/folds.json'); tasks=read(OUT/'data/tasks.json')
    protocol_sha=sha(OUT/'protocol.json'); calls={}; receipts=[]
    for task in tasks:
        path=OUT/f'calls/{task["task_id"]}.json'; saved=read(path)
        assert saved['task']==task and saved['protocol_sha256']==protocol_sha
        out=saved['output']; assert not out['ground_truth_in_model_inputs']
        assert out['order']==order_for(task['sample_id'],task['pool'],task['reverse'])
        assert out['counter']['processed_pixels']==512**2+len(task['pool'])*256**2
        calls[task['task_id']]=out; receipts.append(dict(path=str(path),sha256=sha(path)))
    old_rows=read(PRIOR/'predictions.json')['rows']
    old={(r['sample_id'],r['model'],r['arm']):r for r in old_rows}
    records=[]; mvp={}; native={}
    for c in cases:
        sid=c['sample_id']; p=proposals[sid]
        key=hashlib.sha256(sid.encode()).hexdigest()[:24]
        saved_mvp=read(V21/f'mvp/{key}.json'); saved_case=read(V21/f'cases/{key}.json')
        assert saved_mvp['image_sha256']==c['image_sha256']
        mvp[sid]=saved_mvp['output']['point']; native[sid]=saved_case['native_result']['point']
        for name in NAMES:
            model=p['models'][name]; pool=model['pool']; anchor=model['anchor']
            observations=[]
            if pool:
                canonical=sorted(pool)
                for reverse in [False,True]:
                    out=calls[model['signature']+'_'+str(int(reverse))]
                    aligned_order=[pool.index(canonical[i]) for i in out['order']]
                    observations.append(dict(order=aligned_order,scores=out['scores']))
                combined=combine(pool,observations)
                feat=features(combined,pool,p['points'],model['scores'],c['image_size'])
                unconditional=combined['unconditional_point']; fixed=combined['fixed_point']
                logical_seconds=sum(calls[model['signature']+'_'+str(i)]['elapsed_seconds'] for i in [0,1])
                logical_pixels=2*(512**2+len(pool)*256**2)
            else:
                combined=dict(proposed_change=False,order_agrees=False,fixed_accept=False)
                feat=[0.]*len(FEATURE_NAMES); unconditional=fixed=anchor; logical_seconds=0.;logical_pixels=0
            base_hit=hit(anchor,c,labels[sid]); un_hit=hit(unconditional,c,labels[sid])
            assert base_hit==old[sid,name,'baseline']['hit']
            records.append(dict(sample_id=sid,model=name,fold=folds[sid],group=[c['source'],c['source_group']],
                features=feat,proposed_change=combined['proposed_change'],utility=int(un_hit)-int(base_hit),
                combined=combined,points=dict(baseline=anchor,v23_verify=old[sid,name,'marked_verify']['point'],
                    expanded_ungated=unconditional,expanded_fixed=fixed),
                extra_seconds=logical_seconds,extra_pixels=logical_pixels,
                full_oracle=any(hit(q,c,labels[sid]) for q in p['points']),
                old3_oracle=any(hit(q,c,labels[sid]) for q in model['previous_shortlist']),
                new9_oracle=any(hit(q,c,labels[sid]) for q in pool)))
    gate_models=[]
    for fold in range(5):
        training=[r for r in records if r['fold']!=fold]; testing=[r for r in records if r['fold']==fold]
        train_groups={tuple(r['group']) for r in training}; test_groups={tuple(r['group']) for r in testing}
        assert not train_groups & test_groups
        model=fit_gate(training)
        gate_models.append(dict(fold=fold,model=model,training_rgb=sorted({r['sample_id'] for r in training}),
            heldout_rgb=sorted({r['sample_id'] for r in testing}),application_overlap=0))
        for r in testing:
            value=predict_utility(model,r['features'])
            accept=bool(r['proposed_change'] and r['combined']['order_agrees'] and value>0)
            r['predicted_utility_oof']=value; r['learned_accept_oof']=accept
            r['points']['expanded_learned_oof']=r['points']['expanded_ungated'] if accept else r['points']['baseline']
    write(OUT/'gate_models.json',dict(feature_names=list(FEATURE_NAMES),ridge=20.,fold_models=gate_models,
        all_development_fit_for_future_test_only=fit_gate(records),adopted=False))
    lookup={(r['sample_id'],r['model']):r for r in records}; results={}
    lines=['V24：扩大候选、每候选独立局部图、双顺序一致性与保守替换。',
        '100张历史复用开发图（新72/旧28），两版各3固定种子均值，不是集成或新独立测试。',
        '原评分器与8B骨干冻结；OOF是5折应用/episode分组的小型效用门，不是视觉模型微调。',
        '联合设计改变候选数/局部视野/顺序/计算量，不能把整体差异归因给其中某一项。']
    for co in ['new_development','old_val','all']:
        selected=[c for c in cases if co=='all' or c['cohort']==co]
        summaries={}; comparisons={}; oracles={}
        external={name:np.asarray([[int(hit(mapping[c['sample_id']],c,labels[c['sample_id']])) for c in selected]]*3,float)
                  for name,mapping in [('MVP',mvp),('MVP_native_verify',native)]}
        lines+=['',f'{co}, n={len(selected)}',f'MVP={100*external["MVP"].mean():.2f}%, MVP+既有native复核={100*external["MVP_native_verify"].mean():.2f}%（额外预算不同）']
        for family in FAMILIES:
            arrays={a:np.asarray([[int(hit(lookup[c['sample_id'],f'{family}_s{s}']['points'][a],c,labels[c['sample_id']]))
                for c in selected] for s in [1,2,3]],float) for a in ARMS}
            baseline=arrays['baseline']; summaries[family]={};oracles[family]={}
            for field in ['full_oracle','old3_oracle','new9_oracle']:
                v=np.asarray([[lookup[c['sample_id'],f'{family}_s{s}'][field] for c in selected] for s in [1,2,3]],float)
                oracles[family][field]=dict(percent=100*float(v.mean()),seed_counts=v.sum(1).astype(int).tolist())
            lines.append(f'{family} 覆盖上限 完整池/旧3点/新9点='+str([round(oracles[family][v]['percent'],2) for v in ['full_oracle','old3_oracle','new9_oracle']]))
            for arm,v in arrays.items():
                rows=[lookup[c['sample_id'],f'{family}_s{s}'] for s in [1,2,3] for c in selected]
                repairs=((v==1)&(baseline==0)).sum(1).astype(int).tolist()
                harms=((v==0)&(baseline==1)).sum(1).astype(int).tolist()
                summary=dict(accuracy_percent=100*float(v.mean()),seed_hits=v.sum(1).astype(int).tolist(),
                    repaired_seed_counts=repairs,harmed_seed_counts=harms,
                    mean_extra_seconds=float(np.mean([r['extra_seconds'] for r in rows])) if arm.startswith('expanded') else None,
                    mean_extra_pixels=float(np.mean([r['extra_pixels'] for r in rows])) if arm.startswith('expanded') else None,
                    by_source={src:100*float(v[:,[i for i,c in enumerate(selected) if c['source']==src]].mean()) for src in sorted({c['source'] for c in selected})})
                summaries[family][arm]=summary
                lines.append(f'{family}/{arm}: {summary["accuracy_percent"]:.2f}%, 命中{summary["seed_hits"]},修复{repairs},误改{harms}')
                if arm!='baseline':
                    level=1-.05/4 if co=='new_development' and arm=='expanded_fixed' else .95
                    comparisons[f'{family}/{arm}-baseline']=bootstrap(v-baseline,selected,level)
                    comparisons[f'{family}/{arm}-MVP']=bootstrap(v-external['MVP'],selected,level)
            for arm in ['expanded_fixed','expanded_learned_oof']:
                comparisons[f'{family}/{arm}-ungated']=bootstrap(arrays[arm]-arrays['expanded_ungated'],selected,.95)
                comparisons[f'{family}/{arm}-v23']=bootstrap(arrays[arm]-arrays['v23_verify'],selected,.95)
                if co=='new_development':
                    comparisons[f'{family}/{arm}-baseline_app_sensitivity']=bootstrap(arrays[arm]-baseline,selected,.95,'source_group')
            lines.append(f'{family}/固定保守门-原流程：'+str(comparisons[f'{family}/expanded_fixed-baseline']))
            lines.append(f'{family}/固定保守门-MVP：'+str(comparisons[f'{family}/expanded_fixed-MVP']))
        results[co]=dict(n=len(selected),methods=summaries,comparisons=comparisons,oracles=oracles,
            external={k:100*float(v.mean()) for k,v in external.items()})
    gates={}
    for f in FAMILIES:
        a=results['new_development']['comparisons'][f'{f}/expanded_fixed-baseline']
        m=results['new_development']['comparisons'][f'{f}/expanded_fixed-MVP']
        o=results['old_val']['comparisons'][f'{f}/expanded_fixed-baseline']
        gates[f'{f}/improvement_signal']=a['delta_pp']>=3 and a['ci_pp'][0]>0 and o['delta_pp']>=-3
        gates[f'{f}/MVP_signal']=gates[f'{f}/improvement_signal'] and m['ci_pp'][0]>0
    lines+=['','固定门预登记信号：'+str(gates),'全部结果是复用开发比较，OOF也不是新独立测试。无自动采用、无原fresh280读取。',
        f'完成{len(tasks)}次实际视觉调用，600条原流程命中一致，5折应用组隔离检查通过。',
        '覆盖上限使用真实标签事后计算，仅诊断候选限制，不是可部署模型准确率。',
        '固定门与学习门共享完全相同的视觉输出，可直接观察门控的修复保留/误改消除代价。',
        '原MVP、native复核与本轮的额外候选/像素/调用预算不同，不宣称等计算或速度优势。']
    write(OUT/'predictions.json',dict(records=records,mvp=mvp,mvp_native_verify=native))
    write(OUT/'results.json',dict(complete=True,cohorts=results,positive_gates=gates,independent_test=False,mainline_unchanged=True))
    (OUT/'results.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    write(OUT/'verification.json',dict(complete=True,protocol_sha256=protocol_sha,call_count=len(tasks),
        call_receipts=receipts,baseline_parity=600,application_group_overlap=0,source_sha256=source_identity(),
        decisions_saved_before_label_loading=True))
    plot(results)
    verify()
    paths=[OUT/name for name in ['protocol.json','results.json','results.txt','predictions.json','gate_models.json','verification.json','summary_zh.png']]
    write(OUT/'completion.json',dict(complete=True,completed_at_utc=now(),artifact_sha256={str(p):sha(p) for p in paths},mainline_unchanged=True))
    reg=read(ROOT/'CURRENT_RESEARCH.json');before=read(OUT/'registry_before.json')
    assert reg['models']==before['models'] and reg['run_id']==before['run_id']
    reg['active_experiment'].update(status='complete',phase='inference_OOF_analysis_complete',results_path=str(OUT/'results.txt'),completion_path=str(OUT/'completion.json'),positive_gates=gates)
    reg['completed_experiments'].append(dict(run_id=OUT.name,status='complete',analysis_path=str(OUT/'results.txt'),mainline_unchanged=True))
    reg['next_research_objective']['status']='v24_conservative_verification_complete_MVP_target_pending'
    write(ROOT/'CURRENT_RESEARCH.json',reg)
    write(ROOT/'research/decisions/20261109_conservative_verifier_v24_closeout.json',dict(completed_at_utc=now(),positive_gates=gates,mainline_unchanged=True,readout=str(OUT/'results.txt')))
    status('complete',actual_calls=len(tasks),positive_gates=gates)
    print('\n'.join(lines),flush=True)


def plot(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams['font.sans-serif']=['Microsoft YaHei','SimHei','DejaVu Sans']
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(13,4.8))
    arms=['baseline','v23_verify','expanded_ungated','expanded_fixed','expanded_learned_oof']
    for ax,co,title in zip(axes,['new_development','old_val'],['复用新开发72张','旧开发28张']):
        x=np.arange(len(arms))
        for f,offset,color,label in [('original3k',-.18,'#2878B5','3k'),('full',.18,'#E78A32','全量')]:
            vals=[results[co]['methods'][f][a]['accuracy_percent'] for a in arms]
            bars=ax.bar(x+offset,vals,.34,color=color,label=label);ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=3)
        ax.axhline(results[co]['external']['MVP'],ls='--',color='black',label='MVP（预算不同）')
        ax.set_xticks(x,['原流程','旧复核','扩展无门控','固定保守门','学习门OOF'],rotation=12)
        ax.set_ylim(0,100);ax.set_ylabel('准确率（%）');ax.set_title(title);ax.legend(fontsize=8)
    fig.suptitle('V24：候选扩展与保守替换（3种子均值，学习门为分组OOF，非独立测试）')
    fig.tight_layout();fig.savefig(OUT/'summary_zh.png',dpi=170);plt.close(fig)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['smoke','prepare','worker','analyze','all'],default='all')
    args=parser.parse_args()
    try:
        if args.stage=='smoke':smoke();return
        if args.stage in ['prepare','all']:prepare()
        if args.stage in ['worker','all']:worker()
        if args.stage in ['analyze','all']:analyze()
    except Exception as exc:
        status('failed',error=str(exc),traceback=traceback.format_exc());raise


if __name__=='__main__':main()
