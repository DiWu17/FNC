"""Self-contained v9 result tables, mechanism counts, and inspectable failures."""
from collections import defaultdict
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image,ImageDraw,ImageFont

from clickagent.action_benchmark import sha256,write_json
from clickagent.spatial_geometry import shape_from_geometry
from clickagent.point_aligned_inference import map_crop_point
from evaluate_conditioned_readouts import score_point
from point_aligned_data import read
from run_target_token_experiment import key,read_jsonl
from evaluate_context_verifier import paired_interval

OUT=Path('runs/context_verifier_v9');DEST=OUT/'evaluation'


def e(value):return html.escape(str(value))


def annotation_font(size):
    try:return ImageFont.truetype('C:/Windows/Fonts/arial.ttf',size=size)
    except OSError:return ImageFont.load_default(size=size)


def overlay(case,predictions,pipeline,path):
    image=Image.open(case['image']).convert('RGB');draw=ImageDraw.Draw(image);w,h=image.size
    candidate_font=annotation_font(max(14,round(max(w/1400,h/1000)*16)))
    shape=shape_from_geometry(case['target_geometry'])
    shapes=list(shape.geoms) if hasattr(shape,'geoms') else [shape]
    for region in shapes:
        draw.line([(round(x*w),round(y*h)) for x,y in region.exterior.coords],fill='#00d040',width=max(2,round(w/500)))
    for i,item in enumerate(pipeline['pools']['context_pool'],1):
        x,y=item['point'];draw.ellipse((x-4,y-4,x+4,y+4),outline='#303030',width=2)
        draw.text((x+7,y+7),f'C{i}',font=candidate_font,fill='#202020',stroke_width=1,stroke_fill='white')
    colors={'global_readout':'#007aff','old_crop_readout':'#ff9500','context_roi0_readout':'#a000d0',
        'context_pool__point_critic_s1':'#d50025','context_pool__comparative':'#00b5b0','qwen_native_completed_lax':'#805500',
        'fixed_pool__gui_actor_crop':'#101010'}
    radius=max(5,round(max(w,h)/120))
    for name,color in colors.items():
        point=predictions[name]['point']
        if point is None:continue
        x,y=point
        if name=='fixed_pool__gui_actor_crop':
            for stroke,width in [('white',max(4,radius//2+2)),(color,max(2,radius//3))]:
                draw.line((x-radius,y,x+radius,y),fill=stroke,width=width);draw.line((x,y-radius,x,y+radius),fill=stroke,width=width)
        else:draw.ellipse((x-radius,y-radius,x+radius,y+radius),outline=color,width=max(2,radius//3))
    image.thumbnail((1400,1000));image.save(path)
    # GT zoom is display-only, never an inference input.
    x0,y0,x1,y1=shape.bounds;pad=max(50,.3*max((x1-x0)*w,(y1-y0)*h))
    box=(max(0,round(x0*w-pad)),max(0,round(y0*h-pad)),min(w,round(x1*w+pad)),min(h,round(y1*h+pad)))
    original=Image.open(case['image']).convert('RGB');d=ImageDraw.Draw(original)
    for region in shapes:d.line([(round(x*w),round(y*h)) for x,y in region.exterior.coords],fill='#00d040',width=3)
    for i,item in enumerate(pipeline['pools']['context_pool'],1):
        x,y=item['point'];d.ellipse((x-4,y-4,x+4,y+4),outline='#303030',width=2)
        d.text((x+7,y+7),f'C{i}',font=annotation_font(14),fill='#202020',stroke_width=1,stroke_fill='white')
    for name,color in colors.items():
        point=predictions[name]['point']
        if point is None:continue
        x,y=point
        if name=='fixed_pool__gui_actor_crop':
            for stroke,width in [('white',4),(color,2)]:
                d.line((x-6,y,x+6,y),fill=stroke,width=width);d.line((x,y-6,x,y+6),fill=stroke,width=width)
        else:d.ellipse((x-6,y-6,x+6,y+6),outline=color,width=2)
    zoom=original.crop(box);zoom=zoom.resize((zoom.width*3,zoom.height*3),Image.Resampling.NEAREST);zoom.thumbnail((1400,850))
    zoom.save(path.with_name(path.stem+'_zoom.png'))


def mechanism_analysis(cases,predictions,details,crop_details):
    """Retrospective diagnostics, never used for a prediction or policy choice."""
    import shapely
    from shapely.affinity import scale
    primary='context_pool__point_critic_s1';failures=defaultdict(int);native_gap=defaultdict(int);distances=[]
    pools=defaultdict(int);rejection={};all_candidates=defaultdict(int);view_validity=defaultdict(int)
    selector_failures=defaultdict(lambda:defaultdict(int))
    crop_byid={r['sample_id']:r for r in crop_details}
    for c in cases:
        sid=c['sample_id'];p=predictions[sid];d=details[sid]
        fixed=d['pools']['fixed_pool']['point_oracle'];context=d['pools']['context_pool']['point_oracle']
        pools['gained_correct_candidate']+=context and not fixed;pools['lost_correct_candidate']+=fixed and not context
        pools['both_available']+=fixed and context;pools['both_unavailable']+=not fixed and not context
        pipeline=read(OUT/'pipeline'/f'{key(sid)}.json')
        crop=crop_byid[sid]
        for i,local in enumerate(pipeline['context_locals']):
            named=bool(local['candidates']);visible=crop['context_individual_visible'][i]>0
            view_validity['context_views']+=1;view_validity['context_named_without_visible_target']+=named and not visible
            view_validity['context_unnamed_with_visible_target']+=not named and visible
        if pipeline['old_crop_transform']:
            named=bool(pipeline['old_crop_candidates']);visible=crop['old_target_any_visible']
            view_validity['old_views']+=1;view_validity['old_named_without_visible_target']+=named and not visible
            view_validity['old_unnamed_with_visible_target']+=not named and visible
        global_available=any(score_point(v,c) for v in pipeline['global_result'].get('all_points',[]))
        old_result=pipeline.get('old_crop_result') or {}
        old_available=any(score_point(map_crop_point(point,pipeline['old_crop_transform']),c)
            for point in old_result.get('all_points',[])) if pipeline['old_crop_transform'] else False
        context_available=any(score_point(map_crop_point(point,local['transform']),c)
            for local in pipeline['context_locals'] for point in local['result'].get('all_points',[]))
        all_candidates['all_global_points_oracle']+=global_available
        all_candidates['all_old_crop_points_oracle']+=old_available
        all_candidates['all_global_or_old_crop_points_oracle']+=global_available or old_available
        all_candidates['correct_global_or_old_point_not_in_fixed_pool']+=(global_available or old_available) and not fixed
        all_candidates['all_context_points_oracle']+=context_available
        all_candidates['all_global_or_context_points_oracle']+=global_available or context_available
        all_candidates['correct_point_exists_but_not_in_four_point_pool']+=(global_available or context_available) and not context
        for method in ['fixed_pool__yes_no','fixed_pool__comparative','fixed_pool__point_critic_s1',
                'fixed_pool__gui_actor_crop','context_pool__point_critic_s1','context_pool__comparative','context_pool__gui_actor_crop']:
            pool_name=method.split('__')[0];pd=d['pools'][pool_name];r=p[method]
            if r['hit']:selector_failures[method]['hits']+=1;continue
            stage='empty_pool_no_name' if not pd['points'] else 'pool_has_no_correct_point' if not pd['point_oracle'] else 'selector_missed_available_correct_point'
            selector_failures[method][stage]+=1
            if p['qwen_native_completed_lax']['hit']:selector_failures[method]['native_correct_'+stage]+=1
        if not p[primary]['hit']:
            empty=not d['pools']['context_pool']['points']
            stage='empty_pool_no_name' if empty else 'pool_has_no_correct_point' if not context else 'critic_missed_available_correct_point'
            failures[stage]+=1
            if p['qwen_native_completed_lax']['hit']:native_gap[stage]+=1
            if p[primary]['point'] is not None:
                shape=scale(shape_from_geometry(c['target_geometry']),xfact=c['image_size'][0],yfact=c['image_size'][1],origin=(0,0))
                distances.append(float(shape.distance(shapely.Point(p[primary]['point']))))
    for name in ['fixed_pool','context_pool']:
        counts=defaultdict(int)
        for c in cases:
            sid=c['sample_id'];d=details[sid]['pools'][name];r=predictions[sid][name+'__comparative']
            available=d['point_oracle'];empty=not d['points'];abstain=r['point'] is None
            counts['nonempty_pools']+=not empty;counts['empty_pools']+=empty
            if empty:continue
            counts['available_nonempty']+=available;counts['unavailable_nonempty']+=not available
            counts['reject_bad_pool']+=abstain and not available
            counts['reject_good_pool']+=abstain and available
            counts['click_bad_pool']+=not abstain and not available
            counts['correct_click']+=r['hit'];counts['clicks']+=not abstain
            counts['wrong_choice_despite_available_target']+=available and not abstain and not r['hit']
        counts['rejection_rate_bad_nonempty']=counts['reject_bad_pool']/max(1,counts['unavailable_nonempty'])
        counts['false_rejection_rate_good']=counts['reject_good_pool']/max(1,counts['available_nonempty'])
        counts['precision_among_outputs']=counts['correct_click']/max(1,counts['clicks'])
        rejection[name]=dict(counts)
    return dict(primary_failure_partition=dict(failures),native_correct_primary_wrong_partition=dict(native_gap),
        selector_failure_partitions={name:dict(v) for name,v in selector_failures.items()},
        candidate_pool_change=dict(pools),comparative_abstention=rejection,
        view_name_availability_diagnostic=dict(view_validity),
        unrestricted_candidate_oracles=dict(all_candidates),
        primary_miss_distance=dict(n=len(distances),median_original_px=float(np.median(distances)) if distances else None,
            within2px=sum(v<=2 for v in distances),within5px=sum(v<=5 for v in distances),within10px=sum(v<=10 for v in distances)),
        context_any_individual_roi_fully_visible=sum(any(v>=1-1e-7 for v in r['context_individual_visible']) for r in crop_details),
        note='Post-evaluation diagnosis only. Refusing a bad candidate pool is distinct from forbidden/disabled-action detection; official point accuracy counts abstention as a miss.')


def main():
    s=read(DEST/'summary.json');assert s['complete'] and read(DEST/'audit.json')['passed']
    cases=[c for c in read_jsonl(OUT/'manifest.jsonl') if c['experiment_split']=='test']
    predictions=defaultdict(dict)
    for r in read_jsonl(DEST/'predictions.jsonl'):predictions[r['sample_id']][r['method']]=r
    details={v['sample_id']:v for v in read(DEST/'pool_details.json')}
    mechanism=mechanism_analysis(cases,predictions,details,read(DEST/'crop_details.json'))
    flat=[r for methods in predictions.values() for r in methods.values()]
    mechanism['additional_descriptive_pairs']=[paired_interval(flat,a,b,cases) for a,b in [
        ('context_pool__point_critic_s1','fixed_pool__yes_no'),
        ('fixed_pool__gui_actor_crop','qwen_native_completed_lax'),('fixed_pool__gui_actor_crop','ui_tars')]]
    mechanism['additional_pairs_note']='Retrospective descriptive analysis; no change to the prespecified primary or frozen selection policies.'
    write_json(DEST/'mechanism_analysis.json',mechanism)
    primary=s['primary'];rows=[]
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    variants=['yes_no','comparative','comparative_forced','point_critic_s1','point_critic_s2','gui_actor_crop','gui_actor_full','cross_view_critic_s1']
    labels=['Zero-shot Yes/No','Joint choice + None','Joint forced choice','Point critic seed1','Point critic seed2','GUI-Actor crop','GUI-Actor full','Cross-view critic']
    fig,axes=plt.subplots(1,2,figsize=(14,6),sharex=True)
    limit=max(s['pools'][p]['available'] for p in ['fixed_pool','context_pool'])
    limit=max(limit,s['methods']['qwen_native_completed_lax']['hits'])+12
    for ax,pool in zip(axes,['fixed_pool','context_pool']):
        counts=[s['methods'][pool+'__'+v]['hits'] for v in variants]
        bars=ax.barh(np.arange(len(variants)),counts,color=['#7187a4','#36a6a0','#80c5bf','#d34b65','#d995a4','#65955c','#99b67d','#a28ac6'])
        ax.set_yticks(np.arange(len(variants)),labels);ax.invert_yaxis();ax.set_xlim(0,limit)
        ax.axvline(s['pools'][pool]['available'],color='#299447',linestyle='--',label='Candidate oracle (GT)')
        ax.axvline(s['methods']['qwen_native_completed_lax']['hits'],color='#303844',linestyle=':',label='Qwen native coordinates')
        ax.set_title(pool.replace('_',' ').title());ax.set_xlabel('Hits /192');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
        for bar,count in zip(bars,counts):ax.text(count+.6,bar.get_y()+bar.get_height()/2,str(count),va='center',fontsize=10)
    fig.legend(*axes[0].get_legend_handles_labels(),fontsize=9,loc='lower center',ncol=2)
    fig.suptitle('V9: same candidate points, different selectors');fig.tight_layout(rect=(0,.07,1,.95));fig.savefig(DEST/'verifier_comparison.png',dpi=150);plt.close(fig)
    main_names=['global_readout','old_crop_readout','context_roi0_readout','context_roi1_readout',
        'fixed_pool__yes_no','fixed_pool__yes_no_logodds','fixed_pool__comparative','fixed_pool__comparative_forced','fixed_pool__point_critic_s1',
        'fixed_pool__point_critic_s2','fixed_pool__gui_actor_crop','fixed_pool__gui_actor_full','fixed_pool__cross_view_critic_s1',
        'context_pool__yes_no','context_pool__yes_no_logodds','context_pool__comparative','context_pool__comparative_forced','context_pool__point_critic_s1',
        'context_pool__point_critic_s2','context_pool__gui_actor_crop','context_pool__gui_actor_full','context_pool__cross_view_critic_s1',
        'qwen_native_completed_lax','qwen_native_strict','ui_tars']
    for name in main_names:
        v=s['methods'][name];conditional=f'{v["conditional_selection_accuracy"]:.1%}' if v['conditional_selection_accuracy'] is not None else '—'
        cells=[name,f'{v["hits"]}/{v["n"]}',f'{v["accuracy"]:.2%}',v['points'],conditional,
            v['datasets']['ScreenSpot-v2']['hits'],v['datasets']['ScreenSpot-Pro']['hits'],v['datasets']['VenusBench-GD']['hits']]
        rows.append('<tr'+(' class="primary"' if name==primary else '')+'>'+''.join('<td>'+e(x)+'</td>' for x in cells)+'</tr>')
    cirows=[]
    for r in s['paired_comparisons']:
        cirows.append('<tr>'+''.join('<td>'+e(v)+'</td>' for v in [r['factor'],r['first'],r['second'],f'{r["delta_pp"]:+.2f}',
            f'[{r["ci95_pp"][0]:.2f}, {r["ci95_pp"][1]:.2f}]',f'{r["wins"]}/{r["losses"]}'])+'</tr>')
    examples=DEST/'examples';examples.mkdir(exist_ok=True);cards=[];picked=[]
    categories={
        '同一固定池：专用局部验证救回':lambda p,d:p['fixed_pool__gui_actor_crop']['hit'] and not p['fixed_pool__yes_no']['hit'],
        '主方案相对旧方案救回':lambda p,d:p[primary]['hit'] and not p['fixed_pool__yes_no']['hit'],
        '候选池有正确点，评分头仍选错':lambda p,d:d['pools']['context_pool']['point_oracle'] and not p[primary]['hit'],
        '候选池没有正确点':lambda p,d:not d['pools']['context_pool']['point_oracle'],
        '同屏比较与逐点验证不同':lambda p,d:p['fixed_pool__comparative']['hit']!=p['fixed_pool__yes_no']['hit'],
        '局部本来正确，最终改错':lambda p,d:p['context_roi0_readout']['hit'] and not p[primary]['hit'],
        '直接坐标正确，内部方法失败':lambda p,d:p['qwen_native_completed_lax']['hit'] and not p[primary]['hit']}
    for title,predicate in categories.items():
        selected=[c for c in sorted(cases,key=lambda c:key(c['sample_id'])) if predicate(predictions[c['sample_id']],details[c['sample_id']])][:4]
        for c in selected:
            sid=key(c['sample_id']);p=predictions[c['sample_id']];pipeline=read(OUT/'pipeline'/f'{sid}.json');path=examples/f'{sid}.png'
            overlay(c,p,pipeline,path)
            selected_points={name:p[name] for name in ['global_readout','old_crop_readout','context_roi0_readout',primary,
                'context_pool__comparative','fixed_pool__yes_no','fixed_pool__gui_actor_crop','qwen_native_completed_lax']}
            cards.append(f'<article><h3>{e(title)} · {e(c["dataset"])}</h3><p>{e(c["instruction"])}</p><p>{e(c["sample_id"])}</p>'
                f'<p>全图名称：{e(pipeline["global_result"].get("target_name"))}；局部名称：{e([v["result"].get("target_name") for v in pipeline["context_locals"]])}</p>'
                f'<img loading="lazy" src="examples/{sid}.png"><p>下图使用GT裁剪，仅供诊断展示：</p><img loading="lazy" src="examples/{sid}_zoom.png">'
                f'<details><summary>预测与候选分数</summary><pre>{e(json.dumps(dict(predictions=selected_points,pools=details[c["sample_id"]]["pools"]),ensure_ascii=False,indent=2))}</pre></details></article>')
            picked.append(dict(category=title,sample_id=c['sample_id'],image=f'examples/{sid}.png'))
    training={suffix:read(OUT/'models'/f'point_critic_{suffix}.json')['selected'] for suffix in ['s1','s2']}
    sensitivity=read(DEST/'point_critic_validation_sensitivity.json') if (DEST/'point_critic_validation_sensitivity.json').exists() else None
    sensitivity_summary={suffix:dict(baseline=v['baseline_hits'],interventions={name:dict(hits=r['hits'],choice_changes=r['choice_changes'])
        for name,r in v['interventions'].items()}) for suffix,v in sensitivity['models'].items()} if sensitivity else {}
    alignment=read(DEST/'context_validation_alignment.json')['summary'] if (DEST/'context_validation_alignment.json').exists() else {}
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V9：候选验证与全图上下文重看</title><style>body{{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:1500px;margin:30px auto;padding:0 24px;color:#172238;background:#f4f6fa;line-height:1.65}}table{{border-collapse:collapse;background:white;width:100%;font-size:13px}}td,th{{padding:9px;border:1px solid #dce2ec;text-align:left}}.scroll{{overflow-x:auto}}.primary{{background:#e2f2e7;font-weight:bold}}article{{background:white;padding:20px;margin:22px 0;border-radius:12px}}img{{display:block;max-width:100%;height:auto;margin:12px auto}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}.note{{background:#e8eef9;padding:18px;border-radius:10px}}h1,h2,h3{{line-height:1.35}}</style>
<h1>V9：候选验证与全图上下文重看</h1><p>新192题，ScreenSpot-v2、ScreenSpot-Pro、VenusBench-GD各64题；历史截图及dHash≤4近似图排除。Qwen与v8空间读出器冻结。主方案预先指定：{e(primary)}。</p>
<p class="note">预设主方案命中{s['methods'][primary]['hits']}/192，未改善旧策略{s['methods']['fixed_pool__yes_no']['hits']}/192。固定候选池上的GUI-Actor局部验证为{s['methods']['fixed_pool__gui_actor_crop']['hits']}/192，表现较好，但不替换主方案，也不代表完整GUI-Actor定位系统。Qwen原生完整坐标{s['methods']['qwen_native_completed_lax']['hits']}/192，UI-TARS{s['methods']['ui_tars']['hits']}/192。固定池全部已解码点覆盖{mechanism['unrestricted_candidate_oracles']['all_global_or_old_crop_points_oracle']}题，四点池覆盖{s['pools']['fixed_pool']['available']}题，再验证命中{s['methods']['fixed_pool__gui_actor_crop']['hits']}题；候选筛减是后续具体可检验的损失。</p>
<p class="note">固定候选池：完全沿用v8的全图Top2＋独立裁剪Top2。上下文候选池：全图和最多两处不同区域联合提供点。所有验证器使用同一池内的相同坐标。分数是点落在任意有效目标框/多边形内；不要求等于中心点。所有无输出都保留在192分母。</p>
<h2>同题结果</h2><img src="verifier_comparison.png" alt="同一候选池内各验证方法的命中数"><div class="scroll"><table><thead><tr><th>方法</th><th>命中</th><th>准确率</th><th>有输出</th><th>有正确候选时选对率</th><th>v2/64</th><th>Pro/64</th><th>Venus/64</th></tr></thead><tbody>{''.join(rows)}</tbody></table></div>
<h2>覆盖与验证分开看</h2><pre>{e(json.dumps(dict(pools=s['pools'],crop_coverage=s['crop_coverage']),ensure_ascii=False,indent=2))}</pre>
<p>候选覆盖是使用GT的事后oracle，不能部署，也不是冻结VLM的信息上限。条件选对率以同一候选池中存在正确点的题目为分母，不能替代总准确率。</p>
<p>下方all_points oracle仅枚举归档中已构造的解码点，取消四点池限制；它既不枚举图像中全部像素，也不等于原始注意力信息的覆盖上限。</p>
<h2>失败环节与拒选行为</h2><pre>{e(json.dumps(mechanism,ensure_ascii=False,indent=2))}</pre><p>“拒绝坏候选池”说明当前候选都不适合点击，不能等同于识别目标不存在、禁用或用户禁止点击。N与强制选点分别报告；总准确率始终把拒选计为未命中。</p>
<h2>方法与预算</h2><p>评分头只训练约20万参数，输入冻结全图名称token状态、图像空间特征、每头注意力统计及实际落点。裁剪来源的点也统一在全图状态中打分。训练873条、验证172条；训练点包括预测器真实候选、GT内部点、边界外点、同图其他目标和随机点；验证只用全图Top4。两个种子独立选轮次，主种子在新测试前指定。</p>
<p>主评分头复用既有冻结32维语义/视觉投影，不能视为对原始高维状态的无损解码。高清局部状态只影响主方案的候选提议，主评分仍读取原全图状态；只有单列跨视图方案会读取其他局部状态，该评分头并未在多图局部状态上重新训练。</p>
<p>同屏比较同时看全图缩略图和各候选的768像素原图局部；每个局部与单点Yes/No使用相同262144像素处理上限，缩略图约65536像素。对候选顺序正序/倒序各做一次，平均归一化log分数。允许N拒选；forced一行强制选点，仅为单列对照。</p>
<p>同屏比较的全图缩略图没有标候选ID，各局部用文字A/B/C/D对应。这可能限制重复控件的全局位置关联；本实现的负结果不能推广为所有相对比较方法均无效。</p>
<p>上下文重看在固定的v8裁剪几何上，同时输入全图缩略图＋高清局部，原目标名作为可修正假设；第二处ROI与第一处IoU≤0.3。ROI0对照与旧独立裁剪共享同一裁剪框。GPU计算随视图和次数增加，不声称加速或相同总计算。</p>
<p>上下文方案同时加入了全图和名称假设，其差异不单独识别纯全图上下文的因果效果。定位读出器仍冻结于原输入分布；现有归档只保留局部图注意力，没有保留全图缩略图的空间注意力。</p>
<p>GUI-Actor为官方训练过的2B验证器，从ModelScope下载并核验微软官方哈希。crop行使用与旧验证器相同的裁剪和标记；full行使用官方提示与按图高缩放的标记。两者每图上限262144像素，区别于其模型卡默认更高图像预算。跨视图评分头在能包含候选的位置上取其他视图的标准化支持，排除产生候选的视图，无其他支持则回退全图分数；属于预设次要实验。</p>
<h2>旧验证集选模记录</h2><pre>{e(json.dumps(training,ensure_ascii=False,indent=2))}</pre>
<h2>评分头输入敏感性（原验证集）</h2><pre>{e(json.dumps(sensitivity_summary,ensure_ascii=False,indent=2))}</pre><p>这是固定权重的均值替换诊断，不是匹配重新训练。所有变体仍通过实际名称token注意力生成候选；替换显式名称状态也没有移除注意力中已有的目标信息，不能解释成完全无目标或无视觉输入实验。</p>
<p>相对头内最大值的连续通道在峰值处也等于1，因此移除显式峰值标记没有移除全部峰值信息；该诊断不能单独证明评分头主要利用非峰值证据。</p>
<h2>多图提取对齐检查（原验证集）</h2><pre>{e(json.dumps(alignment,ensure_ascii=False,indent=2))}</pre><p>同一局部图像的合并前视觉向量一致，支持图像块切片与顺序正确。语言层图像状态与局部注意力总量会随全图上下文变化；现有定位头保持冻结，可能存在读出分布变化。注意力总量下降本身不等于定位变差，也不能据此断言剩余注意力都去了全图。</p>
<h2>计算量</h2><pre>{e(json.dumps(s['compute_accounting'],ensure_ascii=False,indent=2))}</pre><p>上述legacy字段mean_vlm_forward_calls按“带图前填充批次”计数，逐token的KV语言解码forward不在计数中；image_views统计各批次内的图像数。专用验证器候选可以批处理。这些是推理构成统计，不能当作独立实测的端到端延迟。</p>
<h2>配对差异与95%区间</h2><p>按数据集分层、5000次bootstrap；探索性对照，未作多重比较校正。</p><div class="scroll"><table><tr><th>因素</th><th>方法A</th><th>方法B</th><th>差值pp</th><th>95%区间</th><th>救回/损失</th></tr>{''.join(cirows)}</table></div>
<h2>失败与成功样例</h2><p>绿框为GT；蓝圈全图，橙圈旧局部，紫圈上下文ROI0，红圈主评分头，青圈上下文池同屏比较，棕圈原生坐标；黑色十字为固定池GUI-Actor局部验证。C1～C4标出完整上下文候选池，可能不包含固定池所选点。示例按样本ID哈希固定顺序，每类最多四例。</p>{''.join(cards)}
<h2>验证与范围</h2><p>新增单测、验证集接口检查及全量CPU预测重放审计见下方链接。大模型视觉状态和logits使用归档证据，审计未重跑大模型。实验为正目标静态定位，不能推出禁用、目标不存在时的误点率或多步执行成功率。已经分析过的v8测试题没有进入评分头训练或新策略选择。</p>
<p><a href="summary.json">全部统计</a> · <a href="audit.json">审计</a> · <a href="pool_details.json">候选详情</a> · <a href="../protocol.json">封存协议</a> · <a href="../inference_identity.json">模型及源码身份</a></p></html>'''
    (DEST/'report.html').write_text(body,encoding='utf-8');write_json(DEST/'report_examples.json',picked)
    print(str(DEST/'report.html'),flush=True)


if __name__=='__main__':main()
