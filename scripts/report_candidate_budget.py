"""Inspectable report for the reused-image, matched-score candidate-budget curve."""
from collections import Counter, defaultdict
import html
from pathlib import Path
import json

import numpy as np
from PIL import Image, ImageDraw

from clickagent.action_benchmark import write_json
from clickagent.candidate_budget import BUDGETS
from clickagent.spatial_geometry import shape_from_geometry
from point_aligned_data import read
from run_candidate_budget import BASE, OUT, cases
from run_target_token_experiment import key


def e(value):return html.escape(str(value))


def overlay(case, detail, path):
    original = Image.open(case['image']).convert('RGB');w,h = original.size
    shape = shape_from_geometry(case['target_geometry'])
    shapes = list(shape.geoms) if hasattr(shape,'geoms') else [shape]
    def draw_on(image, zoom=False):
        draw = ImageDraw.Draw(image)
        for region in shapes:
            draw.line([(round(x*w),round(y*h)) for x,y in region.exterior.coords],fill='#00ce40',width=3 if zoom else max(2,round(w/500)))
        radius = 6 if zoom else max(5,round(max(w,h)/120))
        for item in detail['budgets']['32']['pool']:
            x,y=item['point'];draw.ellipse((x-2,y-2,x+2,y+2),fill='#606060',outline='white')
        for budget,color in [(4,'#007aff'),(8,'#ee8d00'),(16,'#a000d0'),(32,'#d50025')]:
            point=detail['budgets'][str(budget)]['point']
            if point is None:continue
            x,y=point
            if budget==32:
                for stroke,width in [('white',5),(color,2)]:
                    draw.line((x-radius,y,x+radius,y),fill=stroke,width=width)
                    draw.line((x,y-radius,x,y+radius),fill=stroke,width=width)
            else:draw.ellipse((x-radius,y-radius,x+radius,y+radius),outline=color,width=max(2,radius//3))
    full=original.copy();draw_on(full);full.thumbnail((1400,1000));full.save(path)
    annotated=original.copy();draw_on(annotated,True)
    x0,y0,x1,y1=shape.bounds;pad=max(50,.3*max((x1-x0)*w,(y1-y0)*h))
    box=(max(0,round(x0*w-pad)),max(0,round(y0*h-pad)),min(w,round(x1*w+pad)),min(h,round(y1*h+pad)))
    zoom=annotated.crop(box);zoom=zoom.resize((zoom.width*3,zoom.height*3),Image.Resampling.NEAREST)
    zoom.thumbnail((1400,850));zoom.save(path.with_name(path.stem+'_zoom.png'))


def main():
    dest=OUT/'evaluation';s=read(dest/'summary.json');audit=read(dest/'audit.json');assert audit['passed']
    details={v['sample_id']:v for v in read(dest/'details.json')};by_id={c['sample_id']:c for c in cases()}
    mechanism={}
    for k in BUDGETS[1:]:
        counts=Counter();sources=Counter();ranks=Counter()
        for d in details.values():
            a=d['budgets']['4'];b=d['budgets'][str(k)]
            if b['hit'] and not a['hit']:
                counts['rescues']+=1;counts['rescues_from_new_coverage']+=not a['oracle']
                counts['rescues_with_old_correct_candidate']+=a['oracle']
                sources[b['source']]+=1;ranks[str(b['source_rank'])]+=1
            counts['harms']+=a['hit'] and not b['hit']
            counts['new_coverage']+=b['oracle'] and not a['oracle']
        mechanism[f'K{k}_vs_K4']=dict(**counts,rescued_point_sources=dict(sources),rescued_point_source_ranks=dict(ranks))
    write_json(dest/'report_mechanism.json',dict(exploratory=True,all_budget_rescue_diagnostics=mechanism,
        note='Post-inference description only; no policy or model selection changed.'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x=np.arange(len(BUDGETS));m=[s['methods'][f'K{k}'] for k in BUDGETS]
    fig,axes=plt.subplots(1,2,figsize=(13,5.5))
    axes[0].plot(x,[v['available'] for v in m],marker='o',color='#2d9a56',label='Candidate oracle (GT)')
    axes[0].plot(x,[v['hits'] for v in m],marker='o',color='#c73e62',label='GUI-Actor selected hits')
    axes[0].axhline(s['full_decoded_point_oracle'],color='#7a8390',linestyle='--',label='All decoded-point oracle (GT)')
    axes[0].axhline(s['methods']['qwen_native_completed_lax']['hits'],color='#333d4a',linestyle=':',label='Qwen native coordinates')
    axes[0].axhline(s['methods']['ui_tars']['hits'],color='#8e78b0',linestyle='-.',label='UI-TARS')
    for i,v in enumerate(m):
        axes[0].annotate(str(v['available']),(i,v['available']),xytext=(0,7),textcoords='offset points',ha='center',color='#208a49')
        axes[0].annotate(str(v['hits']),(i,v['hits']),xytext=(0,-15),textcoords='offset points',ha='center',color='#bc3155')
    axes[0].set_ylim(0,max(140,s['full_decoded_point_oracle']+15));axes[0].set_ylabel('Cases /192')
    axes[0].set_title('Coverage and actual selected hits')
    bottom=np.zeros(4)
    for field,color,label in [('hits','#409669','Hit'),('selector_missed_correct_point','#e5b64e','Correct point available, selection missed'),
            ('nonempty_pool_without_correct_point','#c07777','No correct point in nonempty pool'),('empty_pools','#a3adbb','Empty pool')]:
        values=np.array([v[field] for v in m]);axes[1].bar(x,values,bottom=bottom,color=color,label=label)
        for i,(v,b) in enumerate(zip(values,bottom)):
            if v>=7:axes[1].text(i,b+v/2,str(v),ha='center',va='center',fontsize=10)
        bottom+=values
    axes[1].set_ylim(0,192);axes[1].set_title('Where the remaining errors occur');axes[1].set_ylabel('Cases /192')
    for ax in axes:
        ax.set_xticks(x,[str(v) for v in BUDGETS]);ax.set_xlabel('Candidate budget K (half from each view)')
        ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
    fig.legend(*axes[0].get_legend_handles_labels(),loc='lower left',bbox_to_anchor=(.01,0),fontsize=8,ncol=2)
    fig.legend(*axes[1].get_legend_handles_labels(),loc='lower right',bbox_to_anchor=(1,0),fontsize=8,ncol=1)
    fig.suptitle('V10 candidate-budget diagnostic: identical per-point scores for every K')
    fig.tight_layout(rect=(0,.19,1,.95));fig.savefig(dest/'budget_curve.png',dpi=160);plt.close(fig)
    table=[];cost=[]
    for k,v in zip(BUDGETS,m):
        transition=s['transitions'].get(f'4_to_{k}')
        if k not in (4,32):
            a={d['sample_id']:d['budgets']['4']['hit'] for d in details.values()}
            transition=dict(rescues=sum(d['budgets'][str(k)]['hit'] and not a[d['sample_id']] for d in details.values()),
                harms=sum(a[d['sample_id']] and not d['budgets'][str(k)]['hit'] for d in details.values()))
        cells=[k,f'{v["mean_candidate_views"]:.2f}',v['available'],v['hits'],f'{v["accuracy"]:.2%}',
            f'{v["conditional_selection_accuracy"]:.2%}',
            f'{transition["rescues"]}/{transition["harms"]}' if transition else '—',
            v['datasets']['ScreenSpot-v2']['hits'],v['datasets']['ScreenSpot-Pro']['hits'],v['datasets']['VenusBench-GD']['hits']]
        table.append('<tr>'+''.join('<td>'+e(c)+'</td>' for c in cells)+'</tr>')
        cells=[k,v['total_candidate_views'],f'{v["mean_independent_batch_call_budget"]:.2f}',
            f'{v["mean_unpadded_processed_pixels"]:,.0f}',f'{v["mean_unpadded_processed_pixels"]/m[0]["mean_unpadded_processed_pixels"]:.2f}×']
        cost.append('<tr>'+''.join('<td>'+e(c)+'</td>' for c in cells)+'</tr>')
    comparisons=[]
    for v in s['paired_comparisons']:
        cells=[v['first'],v['second'],f'{v["delta_pp"]:+.2f}',f'[{v["ci95_pp"][0]:.2f}, {v["ci95_pp"][1]:.2f}]',f'{v["wins"]}/{v["losses"]}']
        comparisons.append('<tr>'+''.join('<td>'+e(c)+'</td>' for c in cells)+'</tr>')
    example_dir=dest/'examples';example_dir.mkdir(exist_ok=True);cards=[];picked=[]
    categories={
        '扩大到16点后救回':lambda b:not b['4']['hit'] and b['16']['hit'],
        '扩大到16点后改错':lambda b:b['4']['hit'] and not b['16']['hit'],
        '16到32点，新候选将正确结果改错':lambda b:b['16']['hit'] and not b['32']['hit'],
        '32点已有正确点，验证仍选错':lambda b:b['32']['oracle'] and not b['32']['hit'],
        '32点仍没有正确点':lambda b:not b['32']['oracle']}
    for title,predicate in categories.items():
        selected=[d for d in sorted(details.values(),key=lambda d:key(d['sample_id'])) if predicate(d['budgets'])][:4]
        for d in selected:
            c=by_id[d['sample_id']];sid=key(c['sample_id']);path=example_dir/f'{sid}.png';overlay(c,d,path)
            description={k:{name:value for name,value in v.items() if name not in ('pool','point_hits','candidate_log_odds')}
                         for k,v in d['budgets'].items() if k in ('4','8','16','32')}
            cards.append(f'<article><h3>{e(title)} · {e(c["dataset"])}</h3><p>{e(c["instruction"])}</p><p>{e(c["sample_id"])}</p>'
                f'<img loading="lazy" src="examples/{sid}.png"><p>下图按GT裁剪，仅供展示，未进入模型：</p><img loading="lazy" src="examples/{sid}_zoom.png">'
                f'<details><summary>四种预算的选择</summary><pre>{e(json.dumps(description,ensure_ascii=False,indent=2))}</pre></details>'
                f'<details><summary>完整候选与分数</summary><pre>{e(json.dumps(d,ensure_ascii=False,indent=2))}</pre></details></article>')
            picked.append(dict(category=title,sample_id=c['sample_id'],image=f'examples/{sid}.png'))
    headline='；'.join(f'K{k}：{s["methods"][f"K{k}"]["hits"]}/192' for k in BUDGETS)
    native_pair=next(v for v in s['paired_comparisons'] if v['first']=='K16' and v['second']=='qwen_native_completed_lax')
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>V10：候选数量与实际命中</title><style>body{{font-family:system-ui,"Microsoft YaHei",sans-serif;max-width:1450px;margin:30px auto;padding:0 24px;background:#f4f6fa;color:#172238;line-height:1.65}}table{{border-collapse:collapse;width:100%;background:white;font-size:14px}}td,th{{border:1px solid #dbe2ec;padding:9px;text-align:left}}.scroll{{overflow-x:auto}}img{{max-width:100%;height:auto;display:block;margin:14px auto}}article,.note{{background:white;padding:20px;border-radius:12px;margin:20px 0}}.note{{background:#e5edf9}}pre{{white-space:pre-wrap;overflow-wrap:anywhere;font-size:12px}}h1,h2,h3{{line-height:1.35}}</style>
<h1>V10：扩大候选池能否恢复实际定位？</h1><p class="note">{e(headline)}。这批192张是已分析过的v9截图，属于探索性诊断，没有新独立测试结论。所有K共享同一份逐点验证分数，没有按本次结果选择新主策略。</p>
<p>固定全图＋旧独立局部的生成结果，按每个视图前K/2名组成池，再做整数坐标精确去重；没有补齐名额，所以实际点数可能少于K。Qwen及原空间读出器均未重跑或重训。唯一变化是候选数量预算。</p>
<h2>覆盖与最终命中</h2><img src="budget_curve.png"><div class="scroll"><table><thead><tr><th>预算K</th><th>平均实际点数</th><th>候选覆盖/192</th><th>命中/192</th><th>准确率</th><th>有正确点时选对率</th><th>相对K4救回/损失</th><th>v2/64</th><th>Pro/64</th><th>Venus/64</th></tr></thead><tbody>{''.join(table)}</tbody></table></div>
<p>候选覆盖使用GT辅助选择当前构造点，不能部署。全部已解码点覆盖{s['full_decoded_point_oracle']}/192，尚未代表这些点能被自动正确筛选。目标框或多边形内任意点均算命中，所有无输出保留在192分母。</p>
<p>同题历史基线：Qwen直接坐标完整宽松解析{s['methods']['qwen_native_completed_lax']['hits']}/192，UI-TARS{s['methods']['ui_tars']['hits']}/192。它们从相同截图的v9归档引用，没有重跑。GUI-Actor这里只验证我们的候选点，不是完整GUI-Actor定位系统跑分。</p>
<p class="note">4到16点：候选覆盖{s['methods']['K4']['available']}→{s['methods']['K16']['available']}，实际命中{s['methods']['K4']['hits']}→{s['methods']['K16']['hits']}；有正确点时的选对率{s['methods']['K4']['conditional_selection_accuracy']:.1%}→{s['methods']['K16']['conditional_selection_accuracy']:.1%}。新增候选恢复了实际成功，同时也增加选择难度。16到32点新增覆盖{s['transitions']['16_to_32']['newly_covered']}题，救回{s['transitions']['16_to_32']['rescues']}题、损失{s['transitions']['16_to_32']['harms']}题。K16只是本批曲线的最高观测值，没有据此发布新默认策略。</p>
<p>K16相对Qwen原生的探索性95%差异区间为[{native_pair['ci95_pp'][0]:.2f},{native_pair['ci95_pp'][1]:.2f}]个百分点，仍包含零；多视图加外部2B验证器的组合，也不能归因为内部评分头已经超过原生解码。还需要在策略固定后用新截图独立确认。</p>
<h2>四点旧结果与数值控制</h2><p>沿用旧缓存的四点结果为{s['methods']['legacy_K4']['hits']}/192；本次统一缓存的四点结果为{s['methods']['K4']['hits']}/192。最大池的每个不同坐标在固定最多6点批次内重新打分一次，所有K读取完全相同的逐点log-odds。相比旧批次的数值漂移单列如下，未用旧分数拼接新分数。</p><pre>{e(json.dumps(s['legacy_score_drift'],ensure_ascii=False,indent=2))}</pre>
<h2>验证开销</h2><table><tr><th>K</th><th>总候选图像数</th><th>每题独立运行批次数预算</th><th>每题未padding图像像素总量</th><th>相对K4像素预算</th></tr>{''.join(cost)}</table>
<p>模型为冻结GUI-Actor-Verifier-2B，FP16、SDPA，每图上限262144像素，输入采用同一768像素标记局部及官方True/False提示。本实验统一缓存实际开销：</p><pre>{e(json.dumps(s['actual_common_cache_compute'],ensure_ascii=False,indent=2))}</pre>
<p>每个K的批次数/像素数是工作量核算；本轮只实际执行一次最大池打分，不能把分摊时间称为独立测得的端到端延迟。像素统计未计批内padding与文字开销，也未计已缓存的原Qwen候选生成。不同候选和批次会产生细小FP16差异，所有K内共享分数隔离了该因素。</p>
<h2>新增覆盖是否转化为成功</h2><pre>{e(json.dumps(s['transitions'],ensure_ascii=False,indent=2))}</pre>
<p>所有预算相对K4的救回来源和原始排名：</p><pre>{e(json.dumps(mechanism,ensure_ascii=False,indent=2))}</pre>
<p>所有非空池都选最高分，没有增加拒选阈值。本轮不能评价目标不存在、禁用或用户禁止点击时的误点率。更多候选可能引入更高分错误点，覆盖率上升不保证实际命中上升。</p>
<h2>探索性配对差异</h2><p>按数据集分层，5000次bootstrap，未做多重比较校正；复用截图，不能据曲线选择后声称独立泛化。</p><div class="scroll"><table><tr><th>A</th><th>B</th><th>差值pp</th><th>95%区间</th><th>救回/损失</th></tr>{''.join(comparisons)}</table></div>
<h2>成功与失败示例</h2><p>绿框为GT，灰点为最大池全部候选；蓝圈K4、橙圈K8、紫圈K16、红色十字K32。示例按样本ID哈希固定排序，每类最多四例。</p>{''.join(cards)}
<h2>复现与审计</h2><p>11项相关单元测试通过；全量CPU选择/指标重放审计通过。核对封存候选、嵌套关系、图像来源、标记几何、逐点logits、全部{audit['prediction_rows_replayed']}条结果及工作量。审计复用归档VLM输出，没有独立重跑大模型。</p>
<p><a href="summary.json">统计</a> · <a href="details.json">逐题候选</a> · <a href="audit.json">审计</a> · <a href="../protocol.json">封存协议</a> · <a href="../identity.json">源码与输入身份</a> · <a href="../../context_verifier_v9/evaluation/report.html">原v9报告</a></p></html>'''
    (dest/'report.html').write_text(body,encoding='utf-8');write_json(dest/'report_examples.json',picked)
    print(str(dest/'report.html'),flush=True)


if __name__=='__main__':main()
