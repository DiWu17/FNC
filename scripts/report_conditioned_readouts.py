"""Matched results, paired uncertainty, and stage-by-stage failure accounting."""
import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np

from clickagent.action_benchmark import sha256,write_json
from clickagent.spatial_geometry import shape_from_geometry
from run_target_token_experiment import read_json,read_jsonl,key


LABELS={'full':'新版：完整条件化空间读出','static_96':'同数据重训：静态96头线性读出','attention_only':'消融：仅注意力空间读出',
    'full_generated_action_policy':'新版：同时服从生成的CLICK/WAIT',
    'no_image':'消融：去图像内容特征','no_mass':'消融：去总量及绝对注意力','no_query':'消融：去显式目标隐藏状态',
    'no_attention':'消融：去显式注意力特征',
    'legacy_prefill':'旧96头：回答前prefill','legacy_first':'旧96头：目标首token','legacy_mean':'旧96头：目标片段平均','layer15_mean':'第15层：首token头平均',
    'full_fine_centers':'新版同权重：细网格中心','full_coarse_centers':'新版同权重：粗网格中心',
    'qwen_native_strict':'Qwen直接坐标：严格格式','qwen_native_completed_lax':'Qwen直接坐标：完整宽松格式',
    'ui_tars':'UI-TARS-1.5-7B'}
ORDER=list(LABELS)


def wilson(hits,n):
    z=1.95996398454;p=hits/n;denom=1+z*z/n
    center=(p+z*z/(2*n))/denom
    span=z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/denom
    return [float(center-span),float(center+span)]


def paired(a,b,ds,seed=20261101):
    rng=np.random.default_rng(seed);delta=np.asarray(a,int)-np.asarray(b,int)
    strata=[np.flatnonzero(np.asarray(ds)==name) for name in sorted(set(ds))]
    differences=np.zeros(5000)
    for indices in strata:
        sampled=rng.choice(indices,size=(5000,len(indices)),replace=True)
        differences+=delta[sampled].sum(1)
    differences/=len(delta)
    return {'delta':float(delta.mean()),'paired_bootstrap_95ci':np.quantile(differences,[.025,.975]).tolist(),
        'wins':int((delta>0).sum()),'losses':int((delta<0).sum()),'bootstrap_samples':5000,
        'resampling':'paired screenshots, stratified by dataset; exploratory, no multiplicity adjustment'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,default=Path('runs/conditioned_v3'))
    args=p.parse_args();out=args.output;dest=out/'evaluation'
    manifest=read_jsonl(out/'manifest.jsonl');cases=[c for c in manifest if c['experiment_split']=='test']
    summary=read_json(dest/'summary.json');freeze=read_json(out/'frozen_readouts.json');selection=read_json(out/'selection.json')
    rows=read_jsonl(dest/'predictions.jsonl')
    ui=read_json(out/'ui_tars/summary.json');assert ui['complete'] and ui['n']==len(cases)
    rows.extend(dict(r,method='ui_tars') for r in read_jsonl(out/'ui_tars/predictions.jsonl'))
    ids=[c['sample_id'] for c in cases];ds=[c['dataset'] for c in cases]
    methods={};hits={}
    for method in ORDER:
        group=[r for r in rows if r['method']==method];indexed={r['sample_id']:r for r in group}
        assert len(indexed)==len(group)==len(cases) and set(indexed)==set(ids)
        values=[bool(indexed[i]['hit']) for i in ids];hits[method]=values
        proposals=sum(indexed[i]['point'] is not None for i in ids)
        methods[method]={'n':len(cases),'hits':int(sum(values)),'accuracy':float(np.mean(values)),
            'coordinate_proposals':proposals,'wrong_coordinate_proposals':proposals-int(sum(values)),
            'no_coordinate_proposal':len(cases)-proposals,
            'precision_among_coordinate_proposals':sum(values)/proposals if proposals else None,
            'wilson_95ci':wilson(sum(values),len(cases)),
            'datasets':{name:{'n':sum(v==name for v in ds),'hits':sum(v and d==name for v,d in zip(values,ds))} for name in sorted(set(ds))}}
    deltas={name:paired(hits['full'],hits[name],ds) for name in ORDER if name!='full'}
    primary=[r for r in rows if r['method']=='full'];failure=Counter()
    from shapely.affinity import scale
    from shapely.geometry import Point
    pixel_targets={c['sample_id']:scale(shape_from_geometry(c['target_geometry']),
        xfact=c['image_size'][0],yfact=c['image_size'][1],origin=(0,0)) for c in cases}
    for r in primary:
        if r['hit']:stage='hit'
        elif r['point'] is None:stage='no_generated_target_name'
        elif not r['selected_coarse_cell_overlap']:stage='wrong_coarse_region'
        elif not r['selected_fine_cell_overlap']:stage='correct_coarse_region_wrong_fine_cell'
        else:stage='selected_fine_cell_overlaps_but_offset_misses'
        failure[stage]+=1
        r['failure_stage']=stage
        r['pixel_distance_to_target']=float(pixel_targets[r['sample_id']].distance(Point(*r['point']))) if r['point'] is not None else None
    near_miss={str(tolerance):sum(r['hit'] or (tolerance>0 and r['pixel_distance_to_target'] is not None
        and r['pixel_distance_to_target']<=tolerance) for r in primary) for tolerance in (0,2,5,10,20)}
    distances={stage:[r['pixel_distance_to_target'] for r in primary if r['failure_stage']==stage and r['pixel_distance_to_target'] is not None]
        for stage in failure if stage!='hit'}
    distance_summary={stage:{'n':len(values),'median_original_pixels':float(np.median(values)) if values else None,
        'p90_original_pixels':float(np.quantile(values,.9)) if values else None} for stage,values in distances.items()}
    oracles={name:sum(bool(r[name]) for r in primary) for name in
        ('candidate_point_oracle_hit','candidate_region_window_oracle_overlap','selected_coarse_cell_overlap','selected_fine_cell_overlap')}
    native=[read_json(out/'native'/f"{key(c['sample_id'])}.json") for c in cases]
    format_stats={'qwen_strict_coordinate_outputs':sum(r['point'] is not None for r in native),
        'qwen_completed_lax_coordinate_outputs':sum(r['completed_lax_point'] is not None for r in native),
        'qwen_outputs_recovered_by_format_only':sum(r['point'] is None and r['completed_lax_point'] is not None for r in native),
        'qwen_generated_wait':sum(r['generated_decision']=='wait' for r in native),
        'qwen_token_cap':sum(bool(r['truncated']) for r in native),
        'ui_tars_coordinate_outputs':sum(r['point'] is not None for r in rows if r['method']=='ui_tars')}
    actions=Counter()
    for c in cases:
        with np.load(out/'features'/f"{key(c['sample_id'])}.npz",allow_pickle=False) as z:meta=json.loads(str(z['metadata']))
        actions[str(meta['action'])]+=1
    format_stats['name_prompt_generated_actions']=dict(actions)
    source_errors=[]
    for c,ours,native_hit in zip(cases,hits['full'],hits['qwen_native_completed_lax']):
        if native_hit and not ours:
            r=next(r for r in primary if r['sample_id']==c['sample_id'])
            source_errors.append(r)
    native_advantage={'n_native_hit_full_miss':len(source_errors),
        'no_name':sum(r['point'] is None for r in source_errors),
        'selected_coarse_region_overlaps':sum(r['selected_coarse_cell_overlap'] for r in source_errors),
        'selected_fine_cell_overlaps':sum(r['selected_fine_cell_overlap'] for r in source_errors)}
    named=[r for r in primary if r['point'] is not None]
    temporal={'named_samples':len(named),
        'highest_weight_on_first_name_token':sum(int(np.argmax(r['target_token_weights']))==0 for r in named),
        'mean_effective_name_tokens':float(np.mean([np.exp(-np.sum(np.asarray(r['target_token_weights'])*np.log(np.maximum(r['target_token_weights'],1e-12)))) for r in named])),
        'dynamic_head_weight_std_l2':float(np.linalg.norm(np.std([r['dynamic_head_weights'] for r in named],axis=0)))}
    fusion=read_json(dest/'fusion_diagnostics.json') if (dest/'fusion_diagnostics.json').exists() else None
    result={'primary_method':'full','n':len(cases),'methods':methods,'paired_full_minus_comparator':deltas,
        'failure_stages':dict(failure),'diagnostic_oracles':oracles,'quantization':{k:summary[k] for k in
            ('coarse_center_geometric_ceiling','fine_center_geometric_ceiling')},
        'coordinate_format_stats':format_stats,'native_advantage_diagnosis':native_advantage,
        'learned_temporal_readout_diagnostics':temporal,
        'full_diagnostic_near_miss_hits_by_original_pixel_tolerance':near_miss,
        'miss_distance_by_stage':distance_summary,
        'posthoc_fusion_diagnostics':fusion,
        'available_names':summary['available_names'],'test_used_for_model_selection':False,
        'manifest_sha256':sha256(out/'manifest.jsonl'),'frozen_readouts_sha256':sha256(out/'frozen_readouts.json')}
    write_json(dest/'comparison.json',result)
    (dest/'all_predictions.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')
    datasets=sorted(set(ds));text=['# 目标语义条件化空间读出：独立测试结果','',
        '主方案在测试前固定为full；以下各方法均使用同一批192张独立新截图及262144像素预算。命中指输出点落入标注框/多边形区域，接受其中任意点。full主指标诊断定位提案，不因生成的WAIT隐藏点；另单列服从CLICK/WAIT的正例策略成绩，便于区分定位与动作拒绝。','',
        f"训练/验证/测试截图数：{selection['unique_images']}；条数：{selection['counts']}。训练来自官方train split，每图最多两个不同目标。",'',
        '| 方法 | 总体命中 | '+ ' | '.join(datasets)+' |','|---|---:|'+ '|'.join(['---:']*len(datasets))+'|']
    for method in ORDER:
        r=methods[method];cells=[f"{v['hits']}/{v['n']}" for name,v in r['datasets'].items()]
        text.append(f"| {LABELS[method]} | {r['hits']}/{r['n']}（{r['accuracy']:.1%}） | "+' | '.join(cells)+' |')
    text+=['','## 主要差异及不确定性','',
        '配对差异按数据集分层、以截图为单位bootstrap 5000次。置信区间是探索性诊断，未作多重比较校正。','',
        '| full 相对比较方法 | 净变化（百分点） | 配对95%区间 | full胜/负题数 |','|---|---:|---:|---:|']
    for name in ('legacy_first','static_96','attention_only','no_image','no_mass','no_query','no_attention','full_fine_centers','full_coarse_centers','qwen_native_completed_lax','ui_tars'):
        r=deltas[name];lo,hi=r['paired_bootstrap_95ci']
        text.append(f"| {LABELS[name]} | {r['delta']*100:+.1f} | [{lo*100:+.1f}, {hi*100:+.1f}] | {r['wins']}/{r['losses']} |")
    text+=['','## 问题仍发生在哪一步','',*['- '+name+'：'+str(count) for name,count in failure.items()], '',
        '点到最近有效标注区域的像素距离（不是到单个标准点的距离）：','',
        *['- '+name+'：'+json.dumps(value,ensure_ascii=False) for name,value in distance_summary.items()], '',
        '仅作诊断的原图像素容差命中数（不替代严格主指标）：'+json.dumps(near_miss,ensure_ascii=False)+'。','',
        f"真实生成目标名称可用：{summary['available_names']}/{len(cases)}。粗网格中心几何上限：{summary['coarse_center_geometric_ceiling']}/{len(cases)}；细网格中心上限：{summary['fine_center_geometric_ceiling']}/{len(cases)}。连续偏移不受这两个离散中心上限约束。",'',
        '候选诊断（均不是实际单次点击命中率）：','',*['- '+name+'：'+str(value) for name,value in oracles.items()], '',
        'Qwen完整宽松坐标命中、新版未命中的题目：','',*['- '+name+'：'+str(value) for name,value in native_advantage.items()], '',
        '坐标输出及格式诊断：','',*['- '+name+'：'+str(value) for name,value in format_stats.items()], '',
        '名称token与动态头诊断（仅描述，不替代消融结论）：','',*['- '+name+'：'+str(value) for name,value in temporal.items()], '',
        '## 实现与训练','',
        '冻结Qwen3-VL-8B；真实生成目标名称，各token送回KV cache后读取自己的第15/28/36层注意力及隐藏状态。另取第15层prefill的P×4096空间图像状态和视觉merger前4P×1152特征。读出学习目标token权重、动态96头组合和目标条件化图像特征，在2h×2w细网格预测候选分数与连续偏移。', '',
        '所有和目标有正面积交集的细网格均获监督；区域标签按交集面积分配，偏移标签取交集内有效代表点。原来的中心不可达小目标不被丢弃。', '',
        '另在完全相同训练/验证数据上重训静态96头首token线性读出，使用粗网格区域交叉熵和原来的中心坐标，区分扩大训练数据与新结构的收益。它有96个参数，共同偏置会在图内softmax中抵消；同样按内部验证命中选checkpoint，不使用测试数据。', '',
        '| 版本 | 参数数 | 选中epoch | 训练命中 | 内部验证命中 |','|---|---:|---:|---:|---:|']
    for name,r in freeze['models'].items():
        v=r['validation'];t=read_json(out/'models'/f'{name}_history.json')['train_metrics_at_selected_checkpoint']
        text.append(f"| {name} | {r['head_parameters']:,} | {r['best_epoch']} | {t['hits']}/{t['n']} | {v['hits']}/{v['n']} |")
    if fusion:
        text+=['','## 联合读出的事后分解','',
            '将同一个full网络拆看为条件化注意力先验及CNN空间修正；没有重训、替换主预测或选择另一个模型。组件共同训练，下面的区域相交数不能充当独立模型点击成绩。','',
            *['- '+name+'：'+str(value) for name,value in fusion['counts'].items()], '',
            *['- '+name+'：'+str(value) for name,value in fusion['prior_vs_final_coarse_matrix'].items()], '',
            '先验错误、最终正确表示CNN救回；先验正确、最终错误表示CNN移离目标。CPU重放与已缓存GPU点预测逐例比较，作为实现检查。']
    text+=['','## 训练来源及验证泛化','',
        '| 来源 | 训练截图/指令 | 验证截图/指令 | full训练命中 | full验证命中 |',
        '|---|---:|---:|---:|---:|']
    full_history=read_json(out/'models/full_history.json');train_metrics=full_history['train_metrics_at_selected_checkpoint']
    for ds in sorted({c['dataset'] for c in manifest if c['experiment_split']=='train'}):
        tr=[c for c in manifest if c['experiment_split']=='train' and c['dataset']==ds]
        va=[c for c in manifest if c['experiment_split']=='val' and c['dataset']==ds]
        t=train_metrics['datasets'][ds];v=freeze['models']['full']['validation']['datasets'][ds]
        text.append(f"| {ds} | {len({c['pixel_sha256'] for c in tr})}/{len(tr)} | {len({c['pixel_sha256'] for c in va})}/{len(va)} | {t['hits']}/{t['n']} | {v['hits']}/{v['n']} |")
    text+=['','## 正例中的点提案与拒绝','',
        '| 方法 | 出点 | 区域内命中 | 区域外点提案 | 无坐标 |','|---|---:|---:|---:|---:|']
    for method in ('full','full_generated_action_policy','qwen_native_strict','qwen_native_completed_lax','ui_tars'):
        r=methods[method]
        text.append(f"| {LABELS[method]} | {r['coordinate_proposals']} | {r['hits']} | {r['wrong_coordinate_proposals']} | {r['no_coordinate_proposal']} |")
    text+=['','全部是有目标的正例。区域外点提案不是阴性误点率；无坐标同时包含拒绝和格式失败。这里没有实际执行鼠标操作。', '',
        '## 解读边界','',
        'Qwen严格坐标与完整宽松格式来自同一生成文本，宽松诊断只接受完整输出中的逗号/空格差异，不利用标准答案修复坐标。UI-TARS使用已验证官方权重及同像素预算，这些分数不是完整官方高分辨率榜单。', '',
        '测试每张截图一题；没有用测试结果选头、选token、选checkpoint或替换主方案。训练图与全部公开截图做精确及保守近似去重，但没有宣称应用级完全独立。','',
        '本轮是正例静态定位，不证明负例误点率或完整电脑任务成功率。候选区域窗口不是控件边界，top-k oracle和窗口覆盖也不能当成实际点击成功。', '',
        '特征采集包含Python hooks、GPU同步和CPU拷贝；如另报该时间，只用于实现诊断，不能据此宣称动作加速。']
    (dest/'report.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
