"""Independent endpoint replay and scoped error interpretation; no inference."""
import csv
from collections import Counter,defaultdict
import math
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
import run_independent_diagnostic_v32 as r

def direct_hit(point,case,label):
    if point is None or len(point)!=2 or not all(math.isfinite(v) for v in point):return False
    x,y=point;w,h=case['image_size']
    if not (0<=x<w and 0<=y<h):return False
    return any(x0-1e-12<=x/w<=x1+1e-12 and y0-1e-12<=y/h<=y1+1e-12 for x0,y0,x1,y1 in label['boxes'])

def main():
    p=r.OUT;spec=r.verify();assert r.base.read(p/'completion.json')['complete']
    public=r.base.read(p/'data/public.json');reserve=r.base.read(p/'data/reserve_public.json');labels=r.base.read(p/'data/labels.json')
    histories=r.base.read(p/'historical_exclusion.json');pixels=set(histories['historical_RGB']);hh=[int(q,16) for q in histories['historical_dhash256']];seen=[]
    for c in public+reserve:
        assert 'target_geometry' not in c and c['pixel_sha256'] not in pixels
        dh=int(c['dhash256'],16);assert all((dh^h).bit_count()>4 for h in hh+seen);seen.append(dh)
    reserve_ids={q['sample_id'] for q in reserve};cases={q['sample_id']:q for q in public};assert set(labels)==set(cases)
    outputs={m:r.base.read(p/m/'results.json')['rows'] for m in ['ours','mvp']}
    raw={m:{q['sample_id']:q for q in qs} for m,qs in outputs.items()}
    for m,qs in outputs.items():
        assert len(qs)==len(cases) and set(raw[m])==set(cases)
        assert len(list((p/m/'cases').glob('*.json')))==len(cases)
        for q in qs:
            assert not q['warmup'] and q['image_sha256']==cases[q['sample_id']]['image_sha256'] and q['output']['ground_truth_in_model_input'] is False
        warm=[r.base.read(q)['sample_id'] for q in (p/m/'warmup').glob('*.json')]
        assert not reserve_ids&(set(raw[m])|set(warm))
    predictions=r.base.read(p/'predictions.json')['rows'];result=r.base.read(p/'results.json');replays=0
    gaps=defaultdict(list);visibility=defaultdict(Counter);fallback=Counter();abstentions=Counter();mvp_missing_reference=0;csv_rows=[]
    for q in predictions:
        sid=q['sample_id'];c=cases[sid];gt=labels[sid];o=raw['ours'][sid]['output'];m=raw['mvp'][sid]['output']
        oh=[direct_hit(x,c,gt) for x in o['seed_points']];mh=direct_hit(m['point'],c,gt)
        assert oh==q['head_hits'] and mh==q['mvp_hit'];replays+=7
        assert len(oh)==6
        oa=any(direct_hit(x['point'],c,gt) for x in o['observed_candidates']);assert oa==q['ours_diagnosis']['any_observed_candidate_correct']
        ref=m.get('reference_result')
        if ref:assert ref['target_bbox']==[0,0,0,0]
        else:mvp_missing_reference+=1
        group='both_correct' if oh[0] and mh else 'ours_only' if oh[0] else 'mvp_only' if mh else 'both_wrong'
        gaps[group].append(sid)
        if o['observed_fallback'] is not None:fallback['ours']+=1
        if o['reason']!='keep_exact_free':fallback['free_missing_selection']+=1
        abstentions['ours']+=o['point'] is None;abstentions['mvp']+=m['point'] is None
        fraction=q['ours_diagnosis']['crop_GT_fraction']
        bucket='full_ge99pct' if fraction>=.99 else 'partial_ge25pct' if fraction>=.25 else 'tiny_lt25pct' if fraction>0 else 'none'
        if not oh[0]:visibility['ours_all_errors'][bucket]+=1
        if group=='mvp_only':visibility['ours_errors_MVP_correct'][bucket]+=1
        csv_rows.append(dict(sample_id=sid,application=c['application'],application_group=c['application_group'],ui_type=c['ui_type'],pair_group=group,ours_hit=oh[0],mvp_hit=mh,ours_error=q['ours_diagnosis']['category'],mvp_error=q['mvp_diagnosis']['category'],primary_crop_target_fraction=fraction,ours_reason=o['reason'],ours_seconds=q['ours_seconds'],mvp_seconds=q['mvp_seconds'],image=c['image']))
    assert len(predictions)==len(cases) and replays==7*len(cases)
    win=len(gaps['ours_only']);loss=len(gaps['mvp_only']);assert win==result['primary']['wins'] and loss==result['primary']['losses']
    for i,name in enumerate(r.base.NAMES):assert sum(q['head_hits'][i] for q in predictions)==result['head_hits'][name]
    assert sum(q['mvp_hit'] for q in predictions)==result['mvp_hits']
    with (p/'paired_cases.csv').open('w',encoding='utf-8-sig',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(csv_rows[0]));writer.writeheader();writer.writerows(csv_rows)
    workload={}
    for method,qs in outputs.items():
        times=[q['elapsed_seconds'] for q in qs]
        workload[method]=dict(mean_seconds=float(np.mean(times)),median_seconds=float(np.median(times)),p90_seconds=float(np.quantile(times,.9)),mean_generation_calls=float(np.mean([q['output']['generation_calls'] for q in qs])),mean_forward_calls=float(np.mean([q['counter']['backbone_forward_calls'] for q in qs])),mean_processed_pixels=float(np.mean([q['counter']['processed_pixels'] for q in qs])))
    detail=dict(complete=True,at_utc=r.base.now(),paired_groups={k:dict(n=len(v),sample_ids=v,ours_error_categories=dict(Counter(next(q for q in predictions if q['sample_id']==sid)['ours_diagnosis']['category'] for sid in v))) for k,v in gaps.items()},
        primary_crop_visibility_on_errors={k:dict(v) for k,v in visibility.items()},ours_conditional_legacy_fallbacks=fallback['ours'],ours_free_missing_selection_fallbacks=fallback['free_missing_selection'],abstentions=dict(abstentions),mvp_missing_reference=mvp_missing_reference,workload=workload,
        scope_note=r.base.read(p/'diagnostic_scope_note.json'),frozen_analysis_source_sha256=r.base.sha(__file__))
    r.write(p/'gap_diagnosis.json',detail)
    ratio=workload['ours']['mean_seconds']/workload['mvp']['mean_seconds'];primary=result['primary']
    lines=['V32最终结论：固定提速方法与MVP的严格未见截图诊断',
        f'正式ScreenSpot-Pro来源，排除本项目所有已发现历史训练/开发/测试及已准备截图，另做dHash<=4近似排除。剩173张，预先按115诊断+58保留分配；24个应用、六类软件。不是1581题整榜，不能与论文整榜分数直接排名。',
        f'预定原3k主头：{result["head_hits"]["original3k_s1"]}/{len(cases)}，{result["head_accuracy_percent"]["original3k_s1"]:.2f}%；MVP：{result["mvp_hits"]}/{len(cases)}，{result["mvp_accuracy_percent"]:.2f}%。',
        '全部六头：'+str(result['head_hits'])+'；系列均值%：'+str(result['family_mean_percent']),
        f'逐题我们独对{win}、MVP独对{loss}；差{primary["delta_pp"]:+.2f}个百分点，95%截图分层区间{primary["ci95_pp"]}，配对检验p={primary["exact_two_sided_p"]:.6g}。',
        '观察到的错误分类：'+str(result['diagnosis']),
        '仅MVP正确的样本中，我们的错误分类：'+str(detail['paired_groups'].get('mvp_only',{}).get('ours_error_categories',{})),
        '我们失败时主局部裁剪的目标面积覆盖分布：'+str(detail['primary_crop_visibility_on_errors']),
        f'单题平均我们{workload["ours"]["mean_seconds"]:.2f}秒，MVP{workload["mvp"]["mean_seconds"]:.2f}秒，耗时比{ratio:.3f}。此次新样本的回退比例与此前六图不同，不能直接外推此前19.10秒。',
        '实际工作量：'+str(workload),f'我们实际执行旧候选流程{fallback["ours"]}题，其中局部点缺失后的选择回退{fallback["free_missing_selection"]}题；无有效最终点：'+str(dict(abstentions))+f'；MVP缺少可审计视图轨迹{mvp_missing_reference}题。',
        '解释边界：我们的crop_miss只指主自由局部裁剪；条件回退还有其他未暴露几何的视图，不能解释为所有视图漏目标。正面积交集不等于目标完整可辨，已另列覆盖比例。正确候选只统计实际观察到的点。缺少MVP轨迹时，其crop_miss不能作为确定漏目标证据。',
        f'推理不读取真实目标标签，双方全部输出后才评分；独立逐点回放{replays}个最终判断一致。58张保留图没有前向，诊断图在本次查看后作为开发证据，不再用于未来独立确认。',
        '保持V15_3k采用状态；未训练、未调阈值、未挑选最高种子，未因本轮分数自动替换模型。']
    final=p/'final_readout.txt';final.write_text('\n'.join(lines)+'\n',encoding='utf-8')
    audit=dict(complete=True,at_utc=r.base.now(),final_point_replays=replays,observed_candidate_replays=len(predictions),all_frozen_protocol_hashes_verified=True,historical_exact_and_near_exclusion_verified=True,reserve_no_forward=True,original_MVP_bbox_dummy_verified=True,all_six_heads_reported=True,no_training_or_seed_selection=True,mainline_unchanged=True)
    r.write(p/'independent_final_audit.json',audit)
    reg=r.base.read(r.ROOT/'CURRENT_RESEARCH.json');assert reg['run_id']=='quality_readout_v15_3k'
    for model in reg['models'].values():assert r.base.sha(model['path'])==model['sha256']
    reg['active_experiment'].update(status='complete',phase='independent_closeout_verified',results_path=str(final),gap_diagnosis_path=str(p/'gap_diagnosis.json'),audit_path=str(p/'independent_final_audit.json'));r.write(r.ROOT/'CURRENT_RESEARCH.json',reg)
    r.write(p/'final_completion.json',dict(complete=True,at_utc=r.base.now(),original_completion_sha256=r.base.sha(p/'completion.json'),final_readout_sha256=r.base.sha(final),gap_diagnosis_sha256=r.base.sha(p/'gap_diagnosis.json'),independent_audit_sha256=r.base.sha(p/'independent_final_audit.json')))
    print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
