"""Self-contained fresh-test report with ranking/fine-decoding decomposition."""
import base64
from collections import Counter
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from shapely.geometry import box

from clickagent.action_benchmark import sha256, write_json
from clickagent.spatial_geometry import shape_from_geometry
from run_target_token_experiment import key, read_jsonl


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def esc(value): return html.escape(str(value))
def embed(path): return 'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()


def example_plot(case, full, old, native, path):
    image = Image.open(case['image']).convert('RGB'); w, h = image.size
    target = shape_from_geometry(case['target_geometry']); x0, y0, x1, y1 = target.bounds
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1.6, 1]})
    for ax in axes:
        ax.imshow(image)
        for geom in getattr(target, 'geoms', [target]):
            coords = np.asarray(geom.exterior.coords)*[w, h]
            ax.plot(coords[:, 0], coords[:, 1], color='lime', lw=2)
        for i, proposal in enumerate(full.get('candidates', [])[:3]):
            a, b, c, d = proposal['region']
            ax.add_patch(plt.Rectangle((a*w, b*h), (c-a)*w, (d-b)*h, fill=False, ec='#bb66ee', lw=1.3))
            ax.text((a+c)*w/2, (b+d)*h/2, str(i+1), color='white', fontsize=8,
                    ha='center', clip_on=True, bbox=dict(facecolor='#704080', edgecolor='none', alpha=.8))
        for label, row, color, marker in [('New ranker', full, '#ff7733', 'x'), ('Previous readout', old, '#e72d69', '+'),
                                         ('Qwen native', native, '#208bff', 'o')]:
            if row['point'] is not None:
                options = dict(facecolors='none', edgecolors=color) if marker == 'o' else dict(color=color)
                ax.scatter(*row['point'], marker=marker, s=95, linewidths=2, label=label, **options)
        ax.set_axis_off()
    px = max((x1-x0)*1.5, .035); py = max((y1-y0)*1.5, .035)
    axes[1].set_xlim(max(0, x0-px)*w, min(1, x1+px)*w)
    axes[1].set_ylim(min(1, y1+py)*h, max(0, y0-py)*h)
    axes[0].set_xlim(0, w); axes[0].set_ylim(h, 0)
    axes[0].set_title('Screenshot; purple boxes = ranked candidate parents')
    axes[1].set_title('Annotated target zoom (green)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False)
    fig.subplots_adjust(left=.015, right=.99, top=.91, bottom=.13, wspace=.035)
    fig.savefig(path, dpi=135); plt.close(fig)


def main():
    out = Path('runs/candidate_ranker_v5'); dest = out/'evaluation'
    audit = read(out/'final_audit.json'); assert audit['complete']
    summary = read(dest/'summary.json'); frozen = read(out/'frozen_readouts.json'); methods = summary['methods']
    rows = read_jsonl(dest/'predictions.jsonl'); cases = {c['sample_id']: c for c in read_jsonl(out/'manifest.jsonl') if c['experiment_split'] == 'test'}
    by = {(r['method'], r['sample_id']): r for r in rows}; plots = []
    # Region-level comparison for native output uses the identical Qwen grid.
    native_parent_hits = 0; previous_region_hits = 0
    for sid, c in cases.items():
        with np.load(out/'features'/f'{key(sid)}.npz', allow_pickle=False) as z: meta = json.loads(str(z['metadata']))
        gh, gw = meta['grid_shape']; w, h = c['image_size']; target = shape_from_geometry(c['target_geometry'])
        point = by['qwen_native_completed_lax', sid]['point']
        if point is not None:
            rr, cc = min(gh-1, int(point[1]*gh/h)), min(gw-1, int(point[0]*gw/w))
            native_parent_hits += target.intersection(box(cc/gw, rr/gh, (cc+1)/gw, (rr+1)/gh)).area > 0
        old = by['phrase_local', sid]
        if old.get('candidates'):
            previous_region_hits += target.intersection(box(*old['candidates'][0]['region'])).area > 0
    full_rows = [r for r in rows if r['method'] == 'full']
    named_rows = [r for r in full_rows if 'null_rejected' in r]
    policy = methods['full_null_policy']
    null_diagnostics = dict(proposed_clicks=policy['points'], correct_clicks=policy['hits'],
        wrong_location_clicks=policy['points']-policy['hits'],
        rejected_with_target_candidate=sum(r['null_rejected'] and r['candidate_region_oracle'] for r in named_rows),
        rejected_without_target_candidate=sum(r['null_rejected'] and not r['candidate_region_oracle'] for r in named_rows),
        total_without_target_candidate=sum(not r['candidate_region_oracle'] for r in named_rows))
    extra = dict(native_same_grid_parent_overlap=int(native_parent_hits), previous_selected_parent_overlap=int(previous_region_hits),
                 positive_task_null_diagnostics=null_diagnostics)
    gap = [r for r in full_rows if not r['hit'] and by['qwen_native_completed_lax', r['sample_id']]['hit']]
    extra['native_correct_full_wrong'] = dict(n=len(gap), failure_stages=dict(Counter(r['failure_stage'] for r in gap)))
    extra['new_fine_vs_center_same_ranking'] = dict(
        rescues=sum(r['hit'] and not by['full_center', r['sample_id']]['hit'] for r in full_rows),
        destroys=sum(not r['hit'] and by['full_center', r['sample_id']]['hit'] for r in full_rows))
    extra['new_fine_vs_frozen_fine_same_ranking'] = dict(
        rescues=sum(r['hit'] and not by['full_frozen_fine', r['sample_id']]['hit'] for r in full_rows),
        destroys=sum(not r['hit'] and by['full_frozen_fine', r['sample_id']]['hit'] for r in full_rows))
    names = ['full', 'full_replica', 'no_visual', 'no_query', 'stats_only', 'phrase_local', 'phrase_global',
             'train_weighted_peak_votes', 'qwen_native_completed_lax', 'ui_tars']
    fig, ax = plt.subplots(figsize=(11, 6))
    colors = ['#24947b' if n == 'full' else '#4d83bd' if n in frozen['models'] else '#9da8b7' for n in names]
    vals = [methods[n]['accuracy']*100 for n in names]
    bars = ax.barh(range(len(names)), vals, color=colors)
    ax.set_yticks(range(len(names)), names); ax.invert_yaxis(); ax.set_xlim(0, 100)
    for bar, name in zip(bars, names):
        ax.text(bar.get_width()+1, bar.get_y()+bar.get_height()/2, f'{methods[name]["hits"]}/192', va='center')
    ax.set_xlabel('Integer-pixel point inside any annotated target (%)')
    ax.set_title('Same fresh 192 screenshots; forced pointing, all missing outputs retained')
    ax.grid(axis='x', alpha=.2); fig.tight_layout(); path = dest/'same_image_scores.png'; fig.savefig(path, dpi=145); plt.close(fig); plots.append(path)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for name in frozen['models']:
        history = read(out/'models'/f'{name}_history.json')
        for ax, field, title in zip(axes, ('region', 'point'), ('Validation Top1 parent-region hits', 'Validation pixel-point hits')):
            values = [h['val']['region_hits']['1'] if field == 'region' else h['val']['hits'] for h in history['history']]
            epochs = [h['epoch'] for h in history['history']]
            line, = ax.plot(epochs, values, label=name)
            ep = history['best_epoch']; ax.scatter(ep, values[ep], color=line.get_color(), s=45)
            ax.set_title(title); ax.set_xlabel('Epoch'); ax.set_ylabel('Hits / 172'); ax.grid(alpha=.2)
    axes[0].legend(); fig.tight_layout(); path = dest/'learning_curves.png'; fig.savefig(path, dpi=145); plt.close(fig); plots.append(path)
    primary = methods['full']; n = summary['n']
    fig, ax = plt.subplots(figsize=(10, 5))
    labels = ['Pool contains target parent', 'Top5 contains target parent', 'Top1 correct parent', 'New fine point', 'Frozen old fine point', 'Parent center point']
    values = [primary['candidate_region_coverage'], primary['region_top5'], primary['region_top1'], primary['hits'], methods['full_frozen_fine']['hits'], methods['full_center']['hits']]
    bars = ax.barh(range(len(labels)), values, color=['#7f94b3', '#668cbd', '#367fba', '#24947b', '#59a58b', '#8db9a4'])
    for bar, v in zip(bars, values): ax.text(v+2, bar.get_y()+bar.get_height()/2, str(v), va='center')
    ax.set_yticks(range(len(labels)), labels); ax.invert_yaxis(); ax.set_xlim(0, n+10)
    ax.set_title('Separate candidate coverage, ranking and pixel refinement'); ax.set_xlabel('Screenshots / 192')
    ax.grid(axis='x', alpha=.2); fig.tight_layout(); path = dest/'stage_breakdown.png'; fig.savefig(path, dpi=145); plt.close(fig); plots.append(path)
    table = ''.join(f'<tr><td>{esc(name)}</td><td>{methods[name]["hits"]}/192 ({100*methods[name]["accuracy"]:.1f}%)</td><td>{methods[name].get("region_top1", "—")}</td><td>{methods[name].get("region_top5", "—")}</td><td>{methods[name].get("null_rejections", "—")}</td></tr>' for name in names)
    ds_table = ''.join('<tr><td>'+esc(ds)+'</td>'+''.join(f'<td>{methods[name]["datasets"][ds]["hits"]}/64</td>' for name in ['full', 'full_replica', 'phrase_local', 'qwen_native_completed_lax', 'ui_tars'])+'</tr>' for ds in sorted({c['dataset'] for c in cases.values()}))
    comparisons = ''.join(f'<tr><td>full − {esc(c["second"])}</td><td>{c["delta_pp"]:+.2f}</td><td>[{c["ci95_pp"][0]:+.2f}, {c["ci95_pp"][1]:+.2f}]</td><td>{c["wins"]}/{c["losses"]}</td></tr>' for c in summary['paired_comparisons'])
    selected = []; used = set()
    rules = [('rescue', '新方法救回旧读出', lambda r, sid: r['hit'] and not by['phrase_local', sid]['hit']),
             ('fine_miss', '区域选对，精定位失败', lambda r, sid: not r['hit'] and r.get('selected_region_overlap')),
             ('ranking_miss', '候选包含目标，但排序选错', lambda r, sid: r.get('failure_stage') == 'ranking_miss_with_target_candidate'),
             ('native_only', '原生坐标正确，新方法仍失败', lambda r, sid: not r['hit'] and by['qwen_native_completed_lax', sid]['hit'])]
    for slug, title, rule in rules:
        pool = [sid for sid in sorted(cases) if sid not in used and rule(by['full', sid], sid)]
        if pool: selected.append((slug, title, pool[0])); used.add(pool[0])
    examples = []
    for slug, title, sid in selected:
        row = by['full', sid]; c = cases[sid]; path = dest/f'example_{slug}.png'
        example_plot(c, row, by['phrase_local', sid], by['qwen_native_completed_lax', sid], path); plots.append(path)
        with np.load(out/'features'/f'{key(sid)}.npz', allow_pickle=False) as z: meta = json.loads(str(z['metadata']))
        prefixes = meta['generated_prefixes']; indices = meta['target_raw_indices']
        pieces = [prefixes[i-1][len(prefixes[i-2]) if i > 1 else 0:] for i in indices]
        sources = ''.join(f'<tr><td>{s["token_index"]}</td><td><code>{esc(repr(pieces[s["token_index"]]))}</code></td><td>L{s["layer"]}/H{s["head_zero_based"]}</td><td>{s["within_candidate_weight"]:.3f}</td></tr>' for s in row.get('selected_sources', []))
        examples.append(f'<section><h2>{esc(title)}</h2><p>{esc(sid)} · {esc(c["instruction"])}<br>生成名称：{esc(row.get("target_name"))}<br>失败阶段：{esc(row.get("failure_stage"))}</p><img src="{embed(path)}"><p>以下为首选区域中权重最高的前8个来源，权重在该区域来源内部归一化；它不是点击正确概率，也不是因果解释。</p><table><tr><th>token索引</th><th>真实片段</th><th>层/头（头从0编号）</th><th>候选内权重</th></tr>'+sources+'</table></section>')
    image_html = ''.join(f'<img src="{embed(p)}">' for p in plots[:3])
    history_rows = ''
    for name, entry in frozen['models'].items():
        history = read(out/'models'/f'{name}_history.json')
        history_rows += f'<tr><td>{name}</td><td>{entry["parameters"]}</td><td>{entry["best_epoch"]}</td><td>{history["training"]["region_hits"]["1"]}/618</td><td>{history["training"]["hits"]}/618</td><td>{entry["validation"]["region_hits"]["1"]}/172</td><td>{entry["validation"]["hits"]}/172</td></tr>'
    fine_comparison = ''.join(f'<tr><td>{label}</td><td>{methods[name]["hits"]}/192</td></tr>' for name, label in [('full', '新训练的条件化四子格与偏移'), ('full_frozen_fine', '复用旧模型的细定位，父格由新排序器选择'), ('full_center', '新排序器父格的中心'), ('full_null_policy', '新精定位并应用候选全错拒绝分数')])
    document = '<!doctype html><meta charset="utf-8"><title>候选语义排序实验 v5</title><style>body{font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;max-width:1200px;margin:35px auto;padding:0 24px;color:#172033;background:#f8fafc}table{border-collapse:collapse;width:100%;background:white}th,td{border:1px solid #d6deea;padding:7px;text-align:left}th{background:#e9eff8}img{max-width:100%;margin-top:18px}.note{padding:14px;background:#fff3d7}section{margin-top:35px;border-top:2px solid #d6deea}code{white-space:pre-wrap}h1{font-size:29px}</style>'
    document += '<h1>名称token × head空间候选：语义排序实验</h1><p>冻结Qwen3-VL-8B，保留每个名称token每个原始头的峰值来源；相同父格合并成候选，再用目标隐藏状态、候选图像内容和来源质量统计直接排序。没有先将头平均或压成单张学习图。</p>'
    document += f'<p><strong>本版未取得提升：</strong>预先固定的full命中{primary["hits"]}/192，旧local为{methods["phrase_local"]["hits"]}/192，原生坐标为{methods["qwen_native_completed_lax"]["hits"]}/192。候选覆盖{primary["candidate_region_coverage"]}/192与实际可选中、可点击之间仍有很大差距。</p>'
    document += '<p class="note">同一批192张本项目此前未使用截图：ScreenSpot-v2、ScreenSpot-Pro、VenusBench-GD各64。OSWorld-G已无未使用截图，本轮不复用。排除旧像素及256位dHash距离≤2的近似图。沿用618训练、172验证；5个读出全部冻结后才采集新测试特征。各VLM图像上限均262144像素。主方案预先固定full，不能用较好的复现或消融替换。静态正例定位，不代表真实执行或阴性误点率。</p>'
    document += f'<h2>同题像素落点结果</h2><p>命中任一标注框/多边形的任意点均正确，含边界；无名称、拒绝、格式失败保留在分母。主分数强制提出最佳候选点，另列null拒绝策略。原生坐标接受完整结束的宽松格式，没有GT修复。</p><table><tr><th>方法</th><th>实际像素命中</th><th>Top1目标父格/192</th><th>Top5目标父格/192</th><th>拟应用null拒绝数</th></tr>'+table+'</table>'
    document += '<table><tr><th>数据集</th><th>full</th><th>full复现</th><th>旧local</th><th>原生Qwen</th><th>UI-TARS</th></tr>'+ds_table+'</table>'
    document += f'<h2>瓶颈分解</h2><p>原始峰值候选池包含目标父格：{primary["candidate_region_coverage"]}/192；新排序器Top1选对：{primary["region_top1"]}/192；候选存在时选对率：{100*primary["selection_accuracy_given_covered"]:.1f}%。新细定位实际点命中：{primary["hits"]}/192。旧local实际选择父格与目标重叠{previous_region_hits}/192；原生坐标映射至同一Qwen父格后覆盖{native_parent_hits}/192，区域覆盖与最终点指标不能混同。</p>'
    document += f'<p>标注辅助遍历所有新细定位候选点的上界：{primary["candidate_new_fine_point_oracle"]}/192；遍历所有父格中心的上界：{primary["candidate_center_point_oracle"]}/192。这是诊断上界，不能作为可部署准确率。失败阶段：<code>{esc(primary["failure_stages"])}</code>。</p><table><tr><th>同一新排序器的坐标解码对照</th><th>实际命中</th></tr>'+fine_comparison+'</table>'
    document += f'<p>原生坐标正确而full失败共{len(gap)}张，按阶段分解：<code>{esc(extra["native_correct_full_wrong"]["failure_stages"])}</code>。同一排序下，新细定位相对父格中心救回{extra["new_fine_vs_center_same_ranking"]["rescues"]}张、破坏{extra["new_fine_vs_center_same_ranking"]["destroys"]}张；相对冻结旧细化救回{extra["new_fine_vs_frozen_fine_same_ranking"]["rescues"]}张、破坏{extra["new_fine_vs_frozen_fine_same_ranking"]["destroys"]}张。</p>'
    document += f'<p>应用null后：实际提出点击{policy["points"]}次，其中命中{policy["hits"]}次、定位错误{null_diagnostics["wrong_location_clicks"]}次。有正确候选却拒绝{null_diagnostics["rejected_with_target_candidate"]}次，无正确候选时拒绝{null_diagnostics["rejected_without_target_candidate"]}/{null_diagnostics["total_without_target_candidate"]}次。这里全部是合法定位正例；这些数字不能代替真实“禁止点击/不可操作”任务上的误点率。null学习的是候选列表是否含目标，当前没有作真实阴性阈值校准。</p>'
    document += image_html
    document += '<h2>完整、去显式图像、去目标隐藏状态与统计对照</h2><p>no_visual仅屏蔽排序器的显式区域图像/视觉隐藏特征，no_query屏蔽目标隐藏状态，stats_only同时屏蔽两者；VLM仍看到了图像与指令，注意力仍包含目标条件信息。启用分支不同，激活参数容量也不同，不能将它们解释成完全没有图像或语义。所有模型宽度32，优化器及训练预算一致。</p><table><tr><th>模型</th><th>物理参数</th><th>验证选择epoch</th><th>训练区域Top1</th><th>训练像素点</th><th>验证区域Top1</th><th>验证像素点</th></tr>'+history_rows+'</table>'
    document += '<h2>配对比较</h2><p>同一截图配对、按数据集分层bootstrap 5000次；探索性95%区间，没有多重比较校正。单位为像素命中率百分点。</p><table><tr><th>比较</th><th>差值</th><th>95%区间</th><th>新增正确/丢失正确</th></tr>'+comparisons+'</table>'
    document += '<h2>确定性选择的案例</h2><p>每类按sample_id排序取首例并避免重复；仅用于解释输出，不代表该类普遍行为。</p>'+''.join(examples)
    document += '<section><h2>可复现性与边界</h2><p>候选由实际名称token自身forward后的第15/28/36层96头生成，保留所有不同峰值父格及来源。源统计包含图像注意力总量、相对/绝对峰值、熵、局部3×3分布和跨头/词支持；区域内容可以直接参与候选分数。训练25%候选列表增强会移除全部正确候选，学习拒绝列表，但未在真实阴性任务检验误点率。</p><p><strong>本版仍有信息压缩：</strong>候选的来源集合只包含把该父格作为argmax的token/head；保留的是这些来源的统计和峰值周围3×3。没有把全部token×head在每个候选位置上的完整注意力值交给排序器，其他头在该处的次高峰支持也未被完整读取。候选之间没有显式关系建模。这些是可从代码确认的结构限制，但本轮没有单独消融证明它们各自造成了多少损失。</p><p>完整模型训练时在510张有候选目标的样本中选对455张，验证为146张中的78张，新测试为152张中的53张。这个泛化差距说明不能把训练拟合能力当作稳定的语义匹配能力。新增显式视觉与目标隐藏状态的收益也未被消融稳定确认。</p><p>下一轮可先保持候选池与数据不变，补齐每个候选的全token×head注意力证据，检验当前峰值化压缩是否影响选择；同时扩大真实同图多目标监督。局部高分辨率细化应另做变量。本轮没有增加原图裁剪重编码。</p><p>审计核验新测试像素、近重复排除、候选输入无GT重建、验证epoch选择、主干冻结、所有点计分和原生格式转换。来源见同目录summary.json、predictions.jsonl及上级final_audit.json。运行<code>powershell -File scripts/run_candidate_ranker.ps1</code>可从已冻结状态续跑；缓存预测CLI为<code>.venv/python.exe scripts/predict_candidate_ranker.py --features &lt;raw-feature.npz&gt;</code>。</p></section>'
    (dest/'report.html').write_text(document, encoding='utf-8')
    write_json(dest/'additional_comparisons.json', extra)
    write_json(dest/'report_integrity.json', dict(source_sha256=sha256(Path(__file__)), prediction_sha256=sha256(dest/'predictions.jsonl'),
        plot_sha256={p.name: sha256(p) for p in plots}, selected_example_ids=[sid for _, _, sid in selected],
        all_pixels_from_existing_evaluated_images=True, no_predictions_modified=True))
    print(json.dumps(dict(report=str(dest/'report.html'), primary=primary, additional=extra), ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__': main()
