"""Post-prediction V23 interpretation and an independent accounting audit."""
import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src')]
from clickagent.public_evaluation import point_in_geometry
OUT=ROOT/'runs/visual_feedback_v23'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')


def coordinate_grid_reachable(case,geometry):
    """GT-only diagnosis of the fresh CLICK channel, not the KEEP channel."""
    if geometry is None:return False
    w,h=case['image_size']
    xs=np.unique(np.rint(np.arange(1001)*(w-1)/1000).astype(int))/w
    ys=np.unique(np.rint(np.arange(1001)*(h-1)/1000).astype(int))/h
    bounds=geometry['boxes'] if geometry['type']=='bbox' else [
        [min(p[0] for p in polygon),min(p[1] for p in polygon),
         max(p[0] for p in polygon),max(p[1] for p in polygon)] for polygon in geometry['polygons']]
    for x0,y0,x1,y1 in bounds:
        xx=xs[(xs>=x0-1e-12)&(xs<=x1+1e-12)]
        yy=ys[(ys>=y0-1e-12)&(ys<=y1+1e-12)]
        if not len(xx) or not len(yy):continue
        if geometry['type']=='bbox':return True
        if any(point_in_geometry([x,y],geometry) for x in xx for y in yy):return True
    return False


def main():
    completed=read(OUT/'completion.json');assert completed['complete']
    for p,h in completed['artifact_sha256'].items():assert sha(p)==h,p
    protocol=read(OUT/'protocol.json')
    for p,h in protocol['sealed_sha256'].items():assert sha(p)==h,p
    before=read(OUT/'registry_before.json');current=read(ROOT/'CURRENT_RESEARCH.json')
    assert current['run_id']==before['run_id'] and current['models']==before['models']
    cases=read(OUT/'data/public.json');public={c['sample_id']:c for c in cases};labels=read(OUT/'data/labels.json')
    proposals=read(OUT/'data/proposals.json');rows=read(OUT/'predictions.json')['rows'];results=read(OUT/'results.json')
    lookup={(r['model'],r['arm'],r['sample_id']):r for r in rows}
    tasks=read(OUT/'data/tasks.json');calls={};pixels=forwards=0;seconds=0.;kinds=Counter()
    for t in tasks:
        row=read(OUT/f'calls/{t["task_id"]}.json');assert row['task']==t
        c=public[t['sample_id']];o=row['output'];g=o['geometry']
        assert row['image_sha256']==c['image_sha256'] and g['pool']==t['pool']
        assert g['original_size']==c['image_size'] and g['input_pixels']==524288
        assert g['marked']==(t['arm']!='local_relook')
        assert sorted(g['order'])==list(range(len(t['pool'])))
        assert g['shown']==[t['pool'][i] for i in g['order']]
        assert g['labels'][g['order'].index(0)]==g['initial_label']
        x,y,xx,yy=g['crop_box'];w,h=c['image_size']
        assert xx-x==min(512,w) and yy-y==min(512,h) and 0<=x<xx<=w and 0<=y<yy<=h
        assert x<=t['pool'][0][0]<xx and y<=t['pool'][0][1]<yy
        if t['arm']=='marked_verify':
            assert o['point'] in t['pool'] and o['counter']['backbone_forward_calls']==1
        elif o['kind']=='coordinate':
            assert o['terminated_by_eos'] and o['point']==[round(v*(s-1)/1000) for v,s in zip(o['normalized_point'],c['image_size'])]
        if o['fallback']:assert o['point']==t['pool'][0]
        calls[t['task_id']]=o;pixels+=o['counter']['processed_pixels'];forwards+=o['counter']['backbone_forward_calls'];seconds+=o['elapsed_seconds'];kinds[o['kind']]+=1
    for t in tasks:
        if t['arm']!='local_relook':continue
        from clickagent.visual_feedback_v23 import digest
        gs=[calls[digest([t['sample_id'],t['pool'],a])[:24]]['geometry'] for a in ['local_relook','marked_verify','marked_refine']]
        for field in ['pool','shown','order','crop_box','original_size','normalized']:
            assert gs[0][field]==gs[1][field]==gs[2][field]
        assert gs[1]==gs[2]
    independently_scored=0
    for r in rows:
        c=public[r['sample_id']];p=r['point'];geometry=labels[r['sample_id']]
        norm=[v/s for v,s in zip(p,c['image_size'])] if p is not None else None
        actual=bool(norm is not None and point_in_geometry(norm,geometry))
        assert actual==r['hit'];independently_scored+=1
    details={};examples=[]
    for cohort in ['new_development','old_val','all']:
        selected=[c for c in cases if cohort=='all' or c['cohort']==cohort]
        details[cohort]={}
        for family in ['original3k','full']:
            details[cohort][family]={}
            for arm in ['local_relook','marked_verify','marked_refine']:
                per_seed=[]
                for s in [1,2,3]:
                    name=f'{family}_s{s}';counts=Counter()
                    for c in selected:
                        sid=c['sample_id'];b=lookup[(name,'baseline',sid)];r=lookup[(name,arm,sid)]
                        changed=r['point']!=b['point'];repair=r['hit'] and not b['hit'];harm=b['hit'] and not r['hit']
                        counts['changed']+=changed;counts['repaired']+=repair;counts['harmed']+=harm
                        counts['initial_full_reachable']+=b['original_pool_oracle']
                        counts['shortlist_reachable']+=b['shortlist_oracle']
                        counts['repaired_outside_shortlist']+=repair and not b['shortlist_oracle']
                        counts['repaired_outside_original_pool']+=repair and not b['original_pool_oracle']
                        counts['novel_coordinate_hit']+=r['hit'] and not r['point_in_original_pool']
                        counts['novel_coordinate_attempt']+=r['point'] is not None and not r['point_in_original_pool']
                        counts['changed_but_wrong']+=changed and not r['hit']
                        if s==1 and cohort=='new_development' and (repair or harm):
                            examples.append(dict(sample_id=sid,model=name,arm=arm,category='repair' if repair else 'harm',
                                original_point=b['point'],final_point=r['point'],instruction=c['instruction']))
                    per_seed.append(dict(counts))
                names=set().union(*(r.keys() for r in per_seed))
                details[cohort][family][arm]=dict(seed_counts=per_seed,mean_counts={k:float(np.mean([r.get(k,0) for r in per_seed])) for k in sorted(names)})
    write(OUT/'error_decomposition.json',details)
    write(OUT/'example_index.json',dict(posthoc_descriptive=True,examples=examples))
    grid={c['sample_id']:coordinate_grid_reachable(c,labels[c['sample_id']]) for c in cases}
    grid_summary={co:dict(n=sum(co=='all' or c['cohort']==co for c in cases),
        unreachable_sample_ids=[c['sample_id'] for c in cases if (co=='all' or c['cohort']==co) and not grid[c['sample_id']]])
        for co in ['new_development','old_val','all']}
    write(OUT/'coordinate_quantization_diagnostic.json',dict(scope='Posthoc GT-only coordinate-family diagnosis. KEEP can retain arbitrary original pixel points; this is not an upper bound on the entire feedback policy.',cohorts=grid_summary,reachable=grid))
    write(OUT/'runtime/call_contract.json',dict(note='Adapter metadata describes its unused default infer() method. V23 calls run_visual_feedback_v23.infer() with the sealed visual_feedback_v23 prompts.',
        coordinate_max_new_tokens=32,coordinate_sampling='greedy',coordinate_grid='Full original-image normalized0..1000,rounded to integer pixels; KEEP preserves exact original point',
        verification='Constrained A/B/C/N first-token score',images_per_call=2,processed_pixels_per_call=524288))
    lines=['V23 最终解读：预测后视觉反馈，3k与全量评分器。',
        '样本为V21固定抽取的100张复用开发截图，新72/旧28；不是新独立测试。',
        '两版均为冻结22,050参数评分器，Qwen3-VL-8B底座冻结；各3个固定训练种子，不集成、不挑种子。',
        '相同完整池初始点、最多3点短名单、同一概览和初始点周围局部。所有干预仅一轮。',
        '', '新开发72张（三种子平均）：']
    for f in ['original3k','full']:
        for a,m in results['cohorts']['new_development']['methods'][f].items():
            line=f'{f}/{a}: {m["accuracy_percent"]:.2f}%，命中{m["seed_hits"]}，额外均时{m["mean_extra_seconds"]:.2f}秒。'
            if a!='baseline':
                v=results['cohorts']['new_development']['comparisons'][f'{f}/{a}-baseline']
                d=details['new_development'][f][a]['mean_counts']
                line+=f' 相对自身基线{v["delta_pp"]:+.2f}pp，校正99.1667%区间{v["ci_pp"]}；平均修复{d["repaired"]:.2f}、误改{d["harmed"]:.2f}。'
            lines.append(line)
        for label in ['marker_effect','coordinate_protocol']:
            v=results['cohorts']['new_development']['comparisons'][f'{f}/{label}']
            lines.append(f'{f}/{label}: {v["delta_pp"]:+.2f}pp，描述性95%区间{v["ci_pp"]}。')
    lines+=['','解释边界：',
        '1. 局部重看与标记后修正使用完全相同的提示词、图片几何、候选文本和解码，差别只有标记像素；二者差可以检验此处画点的增量作用。',
        '2. 标记后复核只允许选择已有短名单；标记后修正还允许生成新坐标。二者输出协议、token成本也不同，差异不能全部归因于一个抽象的推理能力。',
        '3. 原方法本来已有多局部提议。本轮额外裁剪围绕最终选点，负结果不能否定历史V8相对全图的局部重看收益。',
        '4. 短名单最多3点，其他完整候选不直接提供给验证器；坐标组仍可根据全图重定位。它不能代表无限候选验证或经过专门训练的反馈策略。',
        '5. WAIT/N/格式无效一律回退原点，因此只评定位，不宣称拒绝策略或实际动作成功。',
        '6. 三种子的区间对固定模型条件化；模型使用同一个冻结VLM、固定哈希显示顺序，不能当作独立模型投票。',
        '7. 时间为反馈阶段实测额外耗时，包含图像处理和解码、不包含原候选生成或模型加载；原流程表中0秒指额外反馈为0，不是原模型运行免费。',
        '8. 100张截图已用于历史开发；新72的多重比较区间用于本轮比较，不能把它升级为新的独立泛化证明。',
        '9. 新CLICK采用原图0..1000坐标，KEEP可精确保留任意原候选像素；归一化网格诊断见coordinate_quantization_diagnostic.json，不能把它当作整个反馈流程的上限。',
        '新开发无法由新CLICK坐标网格命中的样本数：'+str(len(grid_summary['new_development']['unreachable_sample_ids'])),
        '', '登记开发正向门槛：'+str(results['positive_gates']),
        '保持现有采用记录；全部实验完成后根据证据讨论后续训练或独立确认，不自动替换。']
    (OUT/'interpretation.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    audit=dict(complete=True,at_utc=datetime.now(timezone.utc).isoformat(),source_seals_intact=True,
        previous_completion_intact=True,mainline_models_intact=True,formal_unique_calls=len(tasks),
        rendered_geometry_triplets_verified=len(tasks)//3,independent_geometry_scores=independently_scored,
        measured_feedback_seconds=seconds,total_formal_processed_pixels=pixels,total_formal_backbone_forwards=forwards,
        unique_call_output_kinds=dict(kinds),files_sha256={str(OUT/p):sha(OUT/p) for p in ['error_decomposition.json','example_index.json','interpretation.txt','coordinate_quantization_diagnostic.json','runtime/call_contract.json']},
        closeout_source_sha256=sha(__file__))
    write(OUT/'final_audit.json',audit)
    print(json.dumps(audit,ensure_ascii=False),flush=True)
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
