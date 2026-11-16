"""Independent replay and plain-language interpretation of the registered V31 pilot."""
from collections import Counter
from datetime import datetime
from pathlib import Path
import numpy as np
import run_reinspection_v31 as run
from clickagent.reinspection_v31 import (MODES,plan_diverse,scores,decide,judge_canvas)


def main():
    spec=run.verify();out=run.OUT;cases=run.read(out/'data/public.json');by={c['sample_id']:c for c in cases}
    results=run.read(out/'results.json');rows=run.read(out/'predictions.json')['rows'];assert len(rows)==100
    frozen=run.read(out/'frozen.json');freeze_time=datetime.fromisoformat(frozen['frozen_at_utc']).timestamp()
    checks=Counter();raw_calls=Counter();raw_seconds=Counter();judge_seconds=0.
    duplicate_top64=[]
    for c in cases:
        for part in ['global','local0','refine','diverse']:
            _,m=run.load(run.path(c,'raw/'+part))
            assert m['sample_id']==c['sample_id'] and m['instruction']==c['instruction']
            assert m['image_sha256']==c['image_sha256'] and not m['ground_truth_in_model_input']
            assert m['own_token_forward'] and m['all_backbone_parameters_frozen']
            raw_calls[part]+=1;raw_seconds[part]+=m['elapsed_seconds'];checks['generated_observations']+=1
            if c['cohort']!='train':assert run.path(c,'raw/'+part).stat().st_mtime>=freeze_time
            if part!='global' and m['point'] is not None:
                tr=m['transform'];p=np.asarray(m['point'])/1000*(np.asarray(m['image_size'])-1)
                expected=np.clip(np.rint(p/tr['scale']+tr['box'][:2]),[0,0],np.asarray(c['image_size'])-1).astype(int).tolist()
                assert m['world_point']==expected;checks['coordinate_transform_replays']+=1
        _,g=run.load(run.path(c,'raw/global'))
        pp=np.asarray(g['proposal']['all_points']).reshape(-1,2);ss=np.asarray(g['proposal']['all_scores'])
        top=pp[np.argsort(-ss,kind='stable')[:64]]
        duplicate_top64.append(len(top)-len({tuple(p) for p in top.tolist()}))
        tr,plan=plan_diverse(g['proposal']['all_points'],g['proposal']['all_scores'],c['image_size'],g['first_crop']['box'])
        assert tr==g['diverse_crop'] and plan==g['diverse_plan'];checks['ROI_plan_replays']+=1
        for blur in ['clear','blur']:
            for reverse in [0,1]:
                a,m=run.load(run.path(c,f'judge/{blur}_{reverse}'))
                assert not m['ground_truth_in_model_input'] and a['hidden'].shape==(4096,)
                if not m['skipped']:
                    assert m['full_instruction']==c['instruction'] and m['generated_tokens']==0
                    assert m['geometry']['blurred']==(blur=='blur') and not m['geometry']['ground_truth_used']
                    judge_seconds+=m['elapsed_seconds'];checks['real_joint_prefills']+=1
                else:checks['unnecessary_judges_skipped']+=1
    dev=[by[r['sample_id']] for r in rows];rr=run.records(dev);eligible=[r for r in rr if r['eligible']]
    for mode in MODES:
        model=run.load_model(frozen['models'][mode]);prediction={r['sample_id']:p for r,p in zip(eligible,scores(model,eligible))}
        for r,row in zip(rr,rows):
            expected,_=decide(r['base'],r['alternative'],prediction.get(r['sample_id'],0.),.1)
            assert expected==row['methods'][mode];checks['learned_decision_replays']+=1
            assert row['methods'][mode] in [r['base'],r['alternative']]
    for p,h in frozen['training_feature_sha256'].items():assert run.sha(p)==h
    assert run.read(out/'completion.json')['complete']
    # Two training examples selected by ID only, not outcomes, for input rendering QA.
    import PIL.Image
    preview=[]
    for c in sorted([c for c in cases if c['cohort']=='train'],key=lambda c:run.key(c['sample_id'])):
        _,m=run.load(run.path(c,'features'))
        if not m['eligible']:continue
        with PIL.Image.open(c['image']) as source:
            image=source.convert('RGB')
        for blur in [False,True]:
            canvas,g=judge_canvas(image,m['base'],m['alternative'],blur)
            p=out/'preview'/f'train_{len(preview):02d}_{int(blur)}.png';p.parent.mkdir(parents=True,exist_ok=True);canvas.save(p);canvas.close()
        image.close();preview.append(c['sample_id'])
        if len(preview)==2:break
    labeled=run.records(dev,run.read(out/'data/labels.json'))
    pair_ceiling=dict(hits=sum(r['base_hit'] or r['alternative_hit'] for r in labeled),n=len(labeled),
        base_hits=sum(r['base_hit'] for r in labeled),alternative_hits=sum(r['alternative_hit'] for r in labeled))
    summary=results['all'];gates={k:v['positive'] for k,v in summary['comparisons'].items()}
    n=summary['n'];m=summary['methods'];replacements=summary['replacements'];timing=results['prompt_timing']['means']
    prompt_check=[]
    for sid in run.read(out/'data/prompt_timing.json'):
        c=by[sid];_,long=run.load(run.path(c,'timing/full_global'));_,short=run.load(run.path(c,'raw/global'))
        _,old=run.load(run.PRIOR/'raw/global'/(run.key(sid)+'.npz'))
        prompt_check.append(dict(sample_id=sid,long_valid=long['point'] is not None,short_valid=short['point'] is not None,
            historical_long_text_exact=long['generated_text']==old['generated_text'],historical_long_point_exact=long['point']==old['point']))
    run.write(out/'prompt_timing_validity.json',prompt_check)
    decision_breakdown={mode:{reason:dict(count=sum(r['switches'][mode]['reason']==reason for r in rows),
        repairs=sum(r['switches'][mode]['repair'] and r['switches'][mode]['reason']==reason for r in rows),
        harms=sum(r['switches'][mode]['harm'] and r['switches'][mode]['reason']==reason for r in rows))
        for reason in ['learned_switch','missing_base_fallback','keep_base']} for mode in MODES}
    run.write(out/'decision_breakdown.json',decision_breakdown)
    lines=[
        'V31：精简全图输出、互补局部重看、真实内容选择器',
        '本轮112张训练、100张复用开发图；保留V15_3k采用状态。所有新增选择器在开发集新观测前冻结。',
        '新选择器读取完整指令和真实候选图块经冻结8B共同编码的4096维表示；换序取反对称差，训练预测替换的净效用，包含两者都对/都错的零效用样本。',
        '几何、图像模糊、联合表示打乱对照全部保留。模糊对照先模糊GUI像素，再画相同的候选标记和标题。', '',
        f'全部{n}张：',
        '  实际二选一点集的可达上限：'+str(pair_ceiling),
    ]
    names={'original_global':'旧长格式全图','short_global':'新短格式全图','original_local':'旧长格式后的自由局部点',
        'short_local':'新短格式后的自由局部点','short_single':'新短格式单局部加缺失回退',
        'refine_consensus':'同预算原点附近重看共识','diverse_consensus':'同预算互补区域重看共识',
        'geometry':'仅几何替换器','content':'仅真实内容替换器','combined':'真实内容＋几何替换器',
        'blurred':'模糊内容＋几何对照','shuffled':'打乱联合表示＋几何对照','zero_shot':'冻结模型字母分数对照',
        'refine_union':'附近重看原生点并集覆盖','diverse_union':'互补重看原生点并集覆盖','MVP_historical':'同图历史MVP参考'}
    for name,label in names.items():lines.append(f'  {label}: {m[name]["percent"]:.2f}% ({m[name]["hits"]}/{n})')
    lines += ['', '新增选择器替换的修复/破坏（不是把所有改点都算成破坏）：']
    for mode,r in replacements.items():lines.append(f'  {mode}: 改点{r["switches"]}，修复{r["repairs"]}，破坏{r["harms"]}')
    lines += ['  学习替换与缺失回退分开：'+str(decision_breakdown)]
    lines += ['', '预登记主比较（全部100图，五比较校正99%区间）：']
    for name,r in summary['comparisons'].items():lines.append(f'  {name}: {r["delta_pp"]:+.2f}个百分点，区间{r["ci_pp"]}，通过={r["positive"]}')
    lines += ['', '训练条件有效截图：'+str(results['training_pairs']),
        '训练内分组与来源留出：'+str(results['training_cv']),
        f'6张预定训练图上的配对全图阶段计时：长格式{timing["full_seconds"]:.2f}秒/{timing["full_tokens"]:.1f}token，短格式{timing["short_seconds"]:.2f}秒/{timing["short_tokens"]:.1f}token。',
        '配对计时输出有效性与历史长格式重放：'+str(prompt_check),
        '这是全图阶段配对计时；共享视图的耗时相加只代表工作量，不等于单独运行整套策略的端到端计时。',
        '各策略模型调用工作量（秒/题，不是独立端到端计时）：'+str(summary['workload_seconds']),
        '', '主比较门槛：'+str(gates),
        '这些仍是已多轮观察的开发图。候选并集覆盖是诊断上限，不是部署准确率；MVP为历史同图参考且观测预算不同。',
        '尚未进行正式1581题独立确认，不因开发集最高分自动采用某个模型、阈值或分支。',
        '256px补充在相同输出面板和像素预算下扩大原图范围，同时降低局部放大倍率；这是上下文与细节的权衡，不是单独只增加上下文的因果干预。',
    ]
    (out/'interpretation.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    audit=dict(complete=True,at_utc=run.now(),checks=dict(checks),raw_calls=dict(raw_calls),raw_model_seconds=dict(raw_seconds),
        joint_prefill_seconds=judge_seconds,source_sha256=run.sha(__file__),preview_train_ids=preview,pair_ceiling=pair_ceiling,
        top64_duplicate_coordinates=dict(cases_with_duplicates=sum(v>0 for v in duplicate_top64),maximum=max(duplicate_top64)),
        development_generated_after_model_freeze=True,all_primary_gates=gates,mainline_unchanged=True)
    run.write(out/'closeout_verification.json',audit)
    reg=run.read(run.ROOT/'CURRENT_RESEARCH.json');assert reg['run_id']=='quality_readout_v15_3k'
    reg['active_experiment'].update(interpretation_path=str(out/'interpretation.txt'),verification_path=str(out/'closeout_verification.json'))
    reg.setdefault('completed_experiments',[]).append(dict(run_id=out.name,status='complete',results_path=str(out/'results.json'),
        interpretation_path=str(out/'interpretation.txt'),verification_path=str(out/'closeout_verification.json'),mainline_unchanged=True))
    run.write(run.ROOT/'CURRENT_RESEARCH.json',reg)
    run.write(run.ROOT/'research/decisions/20261110_reinspection_v31_closeout.json',dict(at_utc=run.now(),
        primary_gates=gates,mainline_unchanged=True,interpretation_path=str(out/'interpretation.txt'),
        source_sha256=run.sha(__file__),verification_sha256=run.sha(out/'closeout_verification.json')))
    run.write(out/'interpretation_manifest.json',dict(complete=True,at_utc=run.now(),
        artifacts={str(p):run.sha(p) for p in [out/'protocol.json',out/'frozen.json',out/'results.json',out/'predictions.json',out/'decision_breakdown.json',out/'prompt_timing_validity.json',
        out/'completion.json',out/'verification.json',out/'interpretation.txt',out/'closeout_verification.json',Path(__file__)]}))
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
