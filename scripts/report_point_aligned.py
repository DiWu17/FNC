"""Chinese static report with plots and inspectable rescue/hurt screenshots."""
from collections import defaultdict
import html
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR',str(Path('runs/point_aligned_v8/matplotlib_cache').absolute()))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image,ImageDraw,ImageFont

from clickagent.action_benchmark import write_json
from clickagent.spatial_geometry import shape_from_geometry
from clickagent.point_aligned_inference import bounded_box
from point_aligned_data import OUT,read
from run_target_token_experiment import key,read_jsonl


LABELS={
    'region_fixed_s1':'区域监督 / 固定点 s1','region_fixed_s2':'区域监督 / 固定点 s2',
    'point_fixed_s1':'点命中监督 / 固定点 s1','point_fixed_s2':'点命中监督 / 固定点 s2',
    'joint_detached_s1':'联合细定位 / 截断梯度 s1','joint_detached_s2':'联合细定位 / 截断梯度 s2',
    'joint_connected_s1':'联合细定位 / 接通梯度 s1','joint_connected_s2':'联合细定位 / 接通梯度 s2',
    'joint_connected_s1__crop_readout':'局部放大 / 注意力读出',
    'joint_connected_s1__global_verified':'全图 Top4 / 视觉验证',
    'joint_connected_s1__crop_verified':'全图+局部候选 / 视觉验证',
    'joint_connected_s1__crop_native_reference':'局部放大 / Qwen 坐标参考',
    'joint_connected_s1__crop_verified_fill_posthoc':'补齐空候选位 / 事后诊断',
    'qwen_native_completed_lax':'Qwen3-VL-8B / 完整宽松解析',
    'qwen_native_strict':'Qwen3-VL-8B / 严格解析','ui_tars':'UI-TARS-1.5-7B',
}


def escaped(v):return html.escape(str(v))


def overlay(case,global_point,crop_point,verified,dest,native_point=None):
    with Image.open(case['image']) as im:image=im.convert('RGB')
    draw=ImageDraw.Draw(image);w,h=image.size;geometry=case['target_geometry']
    if geometry['type']=='bbox':
        for box in geometry['boxes']:draw.rectangle([box[0]*w,box[1]*h,box[2]*w,box[3]*h],outline=(30,220,40),width=max(3,int(w/700)))
    else:
        for polygon in geometry['polygons']:draw.line([(x*w,y*h) for x,y in polygon]+[(polygon[0][0]*w,polygon[0][1]*h)],fill=(30,220,40),width=max(3,int(w/700)))
    radius=max(10,round(w/160))
    for label,point,color in [('G',global_point,(20,120,255)),('C',crop_point,(255,170,0)),('V',verified,(220,30,220)),('N',native_point,(0,160,170))]:
        if point is None:continue
        x,y=point;draw.ellipse((x-radius,y-radius,x+radius,y+radius),outline=color,width=max(3,int(w/600)))
        draw.text((x+radius,y),label,fill=color,stroke_width=1)
    bounds=shape_from_geometry(geometry).bounds
    center=((bounds[0]+bounds[2])*w/2,(bounds[1]+bounds[3])*h/2)
    span=(max(160,(bounds[2]-bounds[0])*w*2),max(160,(bounds[3]-bounds[1])*h*2))
    zoom=image.crop(bounded_box(image.size,center,span));zoom=zoom.resize((zoom.width*3,zoom.height*3),Image.Resampling.NEAREST);zoom.thumbnail((700,500))
    zoom.save(dest.with_name(dest.stem+'_zoom.png'))
    image.thumbnail((1280,900));image.save(dest)


def main():
    root=OUT/'evaluation';s=read(root/'summary.json');audit=read(root/'audit.json');assert audit['passed']
    methods=s['methods'];primary=s['primary'];cases=[c for c in read_jsonl(OUT/'manifest.jsonl') if c['experiment_split']=='test']
    datasets=sorted({c['dataset'] for c in cases});rows=read_jsonl(root/'predictions.jsonl');by={(r['method'],r['sample_id']):r for r in rows}
    pairs=['region_fixed','point_fixed','joint_detached','joint_connected']
    fig,ax=plt.subplots(figsize=(10,4.2));x=np.arange(4)
    for i,suffix in enumerate(['s1','s2']):
        values=[methods[v+'_'+suffix]['accuracy']*100 for v in pairs]
        ax.bar(x+(i-.5)*.32,values,width=.32,label=suffix)
    max_control=max(methods[v+'_'+suffix]['accuracy']*100 for v in pairs for suffix in ('s1','s2'))
    ax.set_xticks(x,['Region / fixed','Point / fixed','Joint / detached','Joint / connected']);ax.set_ylim(0,max(65,max_control+10));ax.set_ylabel('Point inside target (%)');ax.legend();ax.grid(axis='y',alpha=.2)
    fig.tight_layout();fig.savefig(root/'training_controls.png',dpi=160);plt.close(fig)
    names=['joint_connected_s1','joint_connected_s1__crop_readout','joint_connected_s1__global_verified',primary,'qwen_native_completed_lax','ui_tars']
    fig,ax=plt.subplots(figsize=(10,4.2));values=[methods[m]['accuracy']*100 for m in names]
    bars=ax.bar(np.arange(len(names)),values,color=['#6376b6','#40a3a0','#a382b6','#d87a50','#607d55','#7d848d'])
    ax.set_xticks(np.arange(len(names)),['Global','Crop readout','Global + verify','Global/crop + verify','Qwen coordinates','UI-TARS']);ax.set_ylim(0,max(65,max(values)+10));ax.set_ylabel('Point inside target (%)')
    for bar,value,m in zip(bars,values,names):ax.text(bar.get_x()+bar.get_width()/2,value+1,f'{methods[m]["hits"]}/192\n{value:.1f}%',ha='center',fontsize=9)
    ax.grid(axis='y',alpha=.2);fig.tight_layout();fig.savefig(root/'inference_comparison.png',dpi=160);plt.close(fig)
    controls=[]
    for m in [p+'_'+suffix for p in pairs for suffix in ['s1','s2']]:
        h=read(OUT/'models'/f'{m}_history.json');controls.append(dict(method=m,epoch=h['selected']['epoch'],val=h['selected']['validation']['hits']))
    selected={r['method']:r for r in controls}
    table=[]
    order=[p+'_'+suffix for p in pairs for suffix in ['s1','s2']]+[m for m in methods if m.startswith('v')]+names[1:]+['joint_connected_s1__crop_native_reference','qwen_native_strict','joint_connected_s1__crop_verified_fill_posthoc']
    for m in dict.fromkeys(order):
        v=methods[m];cells=[LABELS.get(m,m),f'{v["hits"]}/{v["n"]}',f'{100*v["accuracy"]:.2f}%',str(v['points'])]
        cells+=[f'{v["datasets"][ds]["hits"]}/{v["datasets"][ds]["n"]}' for ds in datasets]
        if m in selected:cells+=[f'{selected[m]["val"]}/172（epoch {selected[m]["epoch"]}）']
        else:cells+=['—']
        table.append('<tr>'+''.join('<td>'+escaped(v)+'</td>' for v in cells)+'</tr>')
    comparisons=[]
    for c in s['paired_comparisons']:
        low,high=c['ci95_pp'];comparisons.append(f'<tr><td>{escaped(c["factor"])}</td><td>{escaped(c["first"])}</td><td>{escaped(c["second"])}</td><td>{c["delta_pp"]:+.2f}</td><td>[{low:+.2f}, {high:+.2f}]</td><td>{c["wins"]} / {c["losses"]}</td></tr>')
    example_dir=root/'examples';example_dir.mkdir(exist_ok=True);categories=defaultdict(list)
    for c in cases:
        g=by['joint_connected_s1',c['sample_id']];v=by[primary,c['sample_id']];crop=by['joint_connected_s1__crop_readout',c['sample_id']]
        category='验证后救回' if not g['hit'] and v['hit'] else '验证后损失' if g['hit'] and not v['hit'] else '全图与验证均失败' if not g['hit'] else '全图与验证均成功'
        categories[category].append(c)
        native=by['joint_connected_s1__crop_native_reference',c['sample_id']]
        if crop['hit'] and not v['hit']:categories['局部点本已正确，却被验证改错'].append(c)
        if native['hit'] and not crop['hit']:categories['局部原生坐标正确，内部读出错误'].append(c)
    cards=[]
    for category,group in categories.items():
        for c in sorted(group,key=lambda c:key(c['sample_id']))[:5]:
            sid=key(c['sample_id']);g=by['joint_connected_s1',c['sample_id']];v=by[primary,c['sample_id']];crop=by['joint_connected_s1__crop_readout',c['sample_id']]
            native=by['joint_connected_s1__crop_native_reference',c['sample_id']]
            overlay(c,g['point'],crop['point'],v['point'],example_dir/f'{sid}.png',native_point=native['point'])
            p=read(OUT/'pipeline'/f'{sid}.json');scores=[dict(point=list(map(int,k.split(','))),score=round(x['score'],4),yes_no_mass=round(x['yes_no_mass'],4)) for k,x in p['verification'].items()]
            crop_name=p['crop_result'].get('target_name','') if p['crop_result'] else ''
            cards.append(f'<article><h3>{escaped(category)} · {escaped(c["dataset"])}</h3><p>{escaped(c["instruction"])}</p><p class="muted">{escaped(c["sample_id"])} · 全图名称：{escaped(read(OUT/"global"/f"{sid}.json")["predictions"]["joint_connected_s1"].get("target_name"))} · 局部名称：{escaped(crop_name)}</p><img src="examples/{sid}.png"><p>目标局部诊断图（使用标注裁剪，仅展示，不进入模型）：</p><img src="examples/{sid}_zoom.png"><p>全图 G {g["point"]} / 局部 C {crop["point"]} / 验证 V {v["point"]} / 局部原生参考 N {native["point"]}</p><pre>{escaped(json.dumps(scores,ensure_ascii=False))}</pre></article>')
    pipeline=s['pipeline'];stages=methods['joint_connected_s1']['failure_stages']
    routing=read(root/'routing_diagnostic.json') if (root/'routing_diagnostic.json').exists() else None
    routing_compact={} if routing is None else {name:dict(outcomes=value['outcomes'],
        effective_heads=value['case_equal_statistics']['effective_heads']['mean'],
        head_entropy_normalized=value['case_equal_statistics']['head_entropy_normalized']['mean']) for name,value in routing['models'].items()}
    training_note=('主全图模型最终选中 epoch0：这意味着完整训练后没有在验证集超越起点，不能把后续推理变化归为新增空间适配器学到的收益。裁剪与视觉验证仍按原协议单独检验。'
        if selected['joint_connected_s1']['epoch']==0 else f'主全图模型由验证集选择 epoch {selected["joint_connected_s1"]["epoch"]}；是否有稳定结构收益须同时看匹配对照和第二种子。')
    resolution_path=root/'label_visual_checks'/'pro_audit.json';resolution_html=''
    if resolution_path.exists():
        resolution=read(resolution_path)
        resolution_html=f'<h2>Pro 的标注与输入分辨率核对</h2><p>64条原始框转换及图片尺寸均核对通过，另对3例作视觉抽查。当前262144像素预算下，目标短边的缩放后中位数约5.9像素，46/64小于8像素、62/64小于16像素；示例图标从20×25像素缩到约3.5×4.4像素。这支持共同输入分辨率是瓶颈之一，尚未单独完成全图高分辨率对照，不能把所有失败归给缩图。</p><pre>{escaped(json.dumps(resolution,ensure_ascii=False,indent=2))}</pre><img src="label_visual_checks/pro_0.png"><img src="label_visual_checks/pro_8.png">'
    precision_path=root/'verifier_precision_diagnostic.json';precision_html=''
    if precision_path.exists():
        precision=read(precision_path);v=precision['stats']['combined']
        precision_html=f'<h2>验证分数的数值诊断</h2><p>原始float32概率在很大正logit处会有舍入和饱和。主组合有{v["maximum_probability_tied"]}条最高概率并列；用同一归档logit的Yes减No直接排序，改变{v["changed_points"]}个落点，但命中仍为{v["logodds_hits"]}/192。这项事后精度诊断不改写主结果，未显示浮点并列是主瓶颈。正式审计按原float32的logaddexp和exp逐项重放。</p><pre>{escaped(json.dumps(precision["stats"],ensure_ascii=False,indent=2))}</pre>'
    headers=['方法','命中数','准确率','有效坐标数']+datasets+['验证集选择']
    text=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>V8 点命中、局部重看与视觉验证</title>
<style>body{{font:16px/1.65 system-ui;margin:32px auto;max-width:1280px;padding:0 24px;color:#203047;background:#f6f8fb}}h1,h2,h3{{line-height:1.3}}table{{border-collapse:collapse;width:100%;font-size:14px;background:white}}td,th{{border:1px solid #dde3eb;padding:8px;text-align:left}}th{{background:#e6ecf5}}img{{max-width:100%;height:auto}}article{{background:white;padding:20px;border-radius:10px;margin:20px 0}}pre{{white-space:pre-wrap;word-break:break-word;background:#edf1f7;padding:12px;font-size:13px}}.muted{{color:#6c7785;font-size:13px}}.lead{{padding:20px;border-left:5px solid #5477ac;background:#e8f0fb}}.scroll{{overflow:auto}}</style>
<h1>V8：点命中监督、空间细定位、局部重看与视觉验证</h1>
<p class="lead">预先指定主方案：全图与局部候选合并后视觉验证，{methods[primary]['hits']}/192（{100*methods[primary]['accuracy']:.2f}%）。全图接通模型 {methods['joint_connected_s1']['hits']}/192；同集 Qwen 直接坐标 {methods['qwen_native_completed_lax']['hits']}/192；UI-TARS {methods['ui_tars']['hits']}/192。是否构成可靠改进请结合下方双种子与配对区间判断。</p>
<p>{escaped(training_note)}</p>
<h2>实验口径</h2><p>冻结 Qwen3-VL-8B，读取模型实际生成的目标名称 token。873 条训练、172 条验证；新 192 条测试由三个公开集各 64 条组成，排除全部历史截图及 dHash 距离 ≤4 的近似图。模型、策略和源码均在测试前封存。预测点落在标注框或多边形内部即算正确，边界有效，不要求等于某个标准点。</p>
<p>四组同参数量、同起点与预算，各两个种子：区域监督固定点、点命中监督固定点、联合细定位截断梯度、联合细定位接通梯度。均从对应种子 v7 点选择模型继续训练；每组 20 epoch ×600 条/epoch，包含 epoch0，由验证集 CPU 整数点命中选取。点目标使用当前候选中的全部正确点；没有正确点时监督 null，不代表图中没有目标。动态命中标签不穿过离散取点反向传播；细定位梯度来自共享子格和偏移辅助损失。</p>
<p>区域→点对照还将区域辅助权重降为 0.25，因此检验的是预设点监督配方。截断与接通组仅改变注意力路由特征到细定位的直接梯度；两组共享图像/语义投影仍可获得梯度。固定点→联合组同时改变细定位解码与动态标签，不能归因于单一因素。</p>
<img src="training_controls.png"><h2>同一批截图的分数</h2><div class="scroll"><table><tr>{''.join('<th>'+escaped(h)+'</th>' for h in headers)}</tr>{''.join(table)}</table></div>
<img src="inference_comparison.png"><h2>局部重看与验证器到底做了什么</h2><p>只对预先指定的 seed1 运行完整推理矩阵。以全图 Top1 父格为中心，在原图裁剪至少 256 像素、跨度约 4 个父格的区域，放大 2 倍后重新生成目标名称并读取内部状态；用同一个封存读出器预测，再映射回原图整数坐标。每次 Qwen 图像预算仍为 262144 像素。</p>
<p>视觉验证使用每个候选周围最多 768×768 的原图区域和空心红圆，只向冻结 Qwen 询问圆心是否位于指令目标内，以首 token 的 Yes/No 相对似然排序。全图 Top4 与“全图 Top2＋局部 Top2”分别对比，整数重复点合并；不是训练过的 GUI-Actor 验证器，也没有概率校准或误点阈值。局部 Qwen 直接坐标单列为参考，不进入我们的注意力候选池。</p>
<pre>{escaped(json.dumps(pipeline,ensure_ascii=False,indent=2))}</pre><p>局部重看漏斗（先检查裁剪是否包含目标，再看名称、候选和最终点）：</p><pre>{escaped(json.dumps(s['crop_cascade'],ensure_ascii=False,indent=2))}</pre><h2>错误分解</h2><p>全图模型的失败阶段为：{escaped(stages)}。候选父格覆盖 {methods['joint_connected_s1']['parent_oracle']}/192，任一已解码候选点命中上限 {methods['joint_connected_s1']['point_oracle']}/192。这是本读出器和候选集合的诊断上限，不能当作 Qwen 内部信息上限。</p>
<p>事后诊断额外考察了“补齐空候选位”：在原合并池不足四点时，从已有全图Top4补齐，再按同一已缓存验证分数选点。没有新增模型推理或改动封存主方案。这条规则在看到ScreenSpot-v2部分结果后提出，不能作为独立预注册验证的改进结论。</p>
<p>主组合方案失败点到标注区域最近边界的原图像素距离：{escaped(methods[primary]['miss_distance_diagnostic'])}。该诊断用来区分近边缘误差与远处误选，不放宽正式命中标准。</p>
{resolution_html}
{precision_html}
<h2>裁剪的语义边界：原验证集输入核对</h2><p>固定验证样例的指令为 Enter the price per gallon.，全图名称为 Gallon price field。局部裁剪切掉左侧字段标签后，名称变为 gal，可能转向燃油体积字段。回映坐标正常不代表裁剪保留了目标语义，候选验证窗口也可能截断标签。这张图仅作原验证集输入诊断，没有据此修改封存策略。</p><img src="../validation_smoke/visual_interface.png">
<h2>配对变化与不确定性</h2><p>同截图按数据集分层、5000 次配对 bootstrap；探索性 95% 区间，未做多重比较校正。区间跨 0 时，不应宣称已证实优越。</p><div class="scroll"><table><tr><th>因素</th><th>方案 A</th><th>方案 B</th><th>百分点 A−B</th><th>95% 区间</th><th>救回 / 损失</th></tr>{''.join(comparisons)}</table></div>
<h2>实际推理预算</h2><p>名称生成带注意力记录，坐标基线和验证预填充走不同路径，阶段延迟只作描述，不等于严格独立计时实验。本轮关注定位效果；裁剪与验证增加了视觉计算。passes 为整个实验的实际调用数，包含共享验证缓存和诊断坐标参考；method_required_visual_passes 单列各方案独立运行所需次数，主方案每题最多六次视觉推理。</p><pre>{escaped(json.dumps(dict(passes=s['passes'],method_required_visual_passes=s['method_required_visual_passes'],timings=s['timings']),ensure_ascii=False,indent=2))}</pre>
<h2>验证集上的分支诊断</h2><p>仅使用原验证集固定64条名称有效样本，事后关闭新增精定位残差或新路由的显式目标语义投影；不据此选择检查点或改推理策略。关闭残差是在最终权重固定时做干预，不等于重新训练对照。</p><pre>{escaped(json.dumps(routing_compact,ensure_ascii=False,indent=2))}</pre>
<h2>可检查的样例</h2><p>绿框/轮廓为标注；蓝 G 为全图点，橙 C 为局部点，紫 V 为最终验证选择，青 N 为局部原生坐标参考（不进入我们的候选池）。样例按样本 ID 哈希固定排序，每类最多 5 条。</p>{''.join(cards)}
<h2>验证与边界</h2><p>审计重放 {audit['global_predictions_replayed']} 个全图预测，核对 {audit['marked_point_records_checked']} 个验证记录，重新计算全部 {audit['scored_predictions']} 个评分；读出器使用 CPU 作为数值基准。Qwen 的视觉状态与 logits 使用归档证据，没有在审计中重新运行大模型。</p><p>这是正目标静态定位测试，不能推断禁用/不存在时误点率或多步任务成功率。两个种子只覆盖全图训练；局部与验证仅覆盖预设 seed1。公开论文全量榜单与本 192 条子集、图像预算及训练量不具直接可比性。</p>
<p><a href="summary.json">完整统计 JSON</a> · <a href="predictions.jsonl">逐条预测</a> · <a href="audit.json">审计</a> · <a href="../protocol.json">训练与推理协议</a> · <a href="../inference_identity.json">推理封存身份</a></p></html>'''
    (root/'report.html').write_text(text,encoding='utf-8');print(root/'report.html',flush=True)


if __name__=='__main__':main()
