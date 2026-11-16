"""Self-contained v6 diagnostic report; no training or model selection."""
import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from clickagent.action_benchmark import sha256, write_json
from clickagent.spatial_geometry import shape_from_geometry
from run_target_token_experiment import read_jsonl


def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def esc(v): return html.escape(str(v))
def embed(path): return 'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()


def example(case, full, masked, path):
    image = Image.open(case['image']).convert('RGB'); w, h = image.size
    target = shape_from_geometry(case['target_geometry']); x0, y0, x1, y1 = target.bounds
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={'width_ratios': [1.5, 1]})
    for ax in axes:
        ax.imshow(image)
        for geom in getattr(target, 'geoms', [target]):
            xy = np.asarray(geom.exterior.coords)*[w, h]; ax.plot(xy[:, 0], xy[:, 1], color='lime', lw=2)
        for label, pred, color, marker in [('All fields', full, '#ff7733', 'x'), ('Peak-only fields', masked, '#207dff', '+')]:
            if pred.get('candidates'):
                a, b, c, d = pred['candidates'][0]['region']
                ax.add_patch(plt.Rectangle((a*w, b*h), (c-a)*w, (d-b)*h, fill=False, ec=color, lw=1.7))
            if pred['point'] is not None:
                ax.scatter(*pred['point'], color=color, marker=marker, s=90, linewidths=2, label=label)
        ax.set_axis_off()
    axes[0].set_xlim(0, w); axes[0].set_ylim(h, 0); axes[0].set_title('Selected parent and learned point')
    px = max((x1-x0)*1.5, .04); py = max((y1-y0)*1.5, .04)
    axes[1].set_xlim(max(0, x0-px)*w, min(1, x1+px)*w)
    axes[1].set_ylim(min(1, y1+py)*h, max(0, y0-py)*h); axes[1].set_title('Annotated target zoom (green)')
    handles, labels = axes[0].get_legend_handles_labels(); fig.legend(handles, labels, loc='lower center', ncol=2, frameon=False)
    fig.subplots_adjust(left=.015, right=.99, top=.91, bottom=.13, wspace=.035)
    fig.savefig(path, dpi=135); plt.close(fig)


def main():
    out = Path('runs/attention_field_v6'); dest = out/'evaluation'; summary = read(dest/'summary.json')
    audit = read(out/'final_audit.json'); assert audit['complete']
    protocol = read(out/'protocol.json'); frozen = read(out/'frozen_readouts.json'); m = summary['methods']
    rows = read_jsonl(dest/'predictions.jsonl'); by = {(r['method'], r['sample_id']): r for r in rows}
    cases = [c for c in read_jsonl(out/'manifest.jsonl') if c['experiment_split'] == 'test']
    names = list(frozen['models']); histories = {n: read(out/'models'/f'{n}_history.json') for n in names}
    plots = []
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    bars = [('all', 'All heads S1'), ('peaks_only', 'Peak-only S1'), ('all_replica', 'All heads S2'),
            ('peaks_only_replica', 'Peak-only S2'), ('v5_full', 'Previous v5')]
    positions = np.arange(len(bars)); width = .27
    for j, (label, values, color) in enumerate([
        ('Parent intersection', [m[n]['region_top1'] for n, _ in bars], '#7797be'),
        ('Same frozen fine decoder', [m[n+'_fixed_fine']['hits'] if n != 'v5_full' else m[n]['hits'] for n, _ in bars], '#319a86'),
        ('Own learned fine decoder', [m[n]['hits'] for n, _ in bars], '#dc9254')]):
        x = positions+(j-1)*width; result = axes[0].bar(x, values, width=width, label=label, color=color)
        for b, v in zip(result, values): axes[0].text(b.get_x()+b.get_width()/2, v+1, str(v), ha='center', fontsize=8)
    axes[0].set_xticks(positions, [v for _, v in bars], rotation=25, ha='right'); axes[0].set_ylim(0, 192)
    axes[0].set_ylabel('Hits / 192 screenshots'); axes[0].set_title('Matched ranking and coordinate controls'); axes[0].legend(fontsize=8)
    point_names = ['all', 'peaks_only', 'all_replica', 'peaks_only_replica', 'v5_full', 'phrase_local', 'qwen_native_completed_lax', 'ui_tars']
    values = [m[n]['hits'] for n in point_names]
    result = axes[1].barh(point_names, values, color=['#319a86' if n in names else '#9da8b7' for n in point_names])
    for b, v in zip(result, values): axes[1].text(v+1, b.get_y()+b.get_height()/2, f'{v}/192', va='center', fontsize=9)
    axes[1].invert_yaxis(); axes[1].set_xlim(0, 192); axes[1].set_xlabel('Point inside any target region'); axes[1].set_title('Same screenshots and cached backbone outputs')
    fig.tight_layout(); path = dest/'scores.png'; fig.savefig(path, dpi=145); plt.close(fig); plots.append(path)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for name, h in histories.items():
        xs = [r['epoch'] for r in h['history']]
        for ax, field, title in zip(axes, ['region', 'point'], ['Validation parent intersections / 172', 'Validation learned point hits / 172']):
            ys = [r['val']['region_hits']['1'] if field == 'region' else r['val']['hits'] for r in h['history']]
            line, = ax.plot(xs, ys, label=name); epoch = h['best_epoch']
            ax.scatter([epoch], [ys[epoch]], color=line.get_color()); ax.set_title(title); ax.set_xlabel('Epoch'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8); fig.tight_layout(); path = dest/'learning_curves.png'; fig.savefig(path, dpi=145); plt.close(fig); plots.append(path)
    example_html = []
    for label, condition in [('region_rescue', lambda a, b: a.get('selected_region_overlap') and not b.get('selected_region_overlap')),
                             ('region_loss', lambda a, b: not a.get('selected_region_overlap') and b.get('selected_region_overlap'))]:
        chosen = next((c for c in cases if condition(by['all', c['sample_id']], by['peaks_only', c['sample_id']])), None)
        if chosen:
            a = by['all', chosen['sample_id']]; b = by['peaks_only', chosen['sample_id']]
            path = dest/f'example_{label}.png'; example(chosen, a, b, path); plots.append(path)
            example_html.append(f'<p>{esc(label)} · {esc(chosen["sample_id"])} · {esc(chosen.get("instruction", ""))}</p><img src="{embed(path)}">')
    table = []
    for name in names+['v5_full', 'phrase_local', 'qwen_native_completed_lax', 'ui_tars']:
        r = m[name]; fixed = m.get(name+'_fixed_fine', {}).get('hits') if name in names else r['hits'] if name == 'v5_full' else None
        table.append(f'<tr><td>{esc(name)}</td><td>{r.get("region_top1", "—")}</td><td>{r.get("region_top3", "—")}</td>'
            f'<td>{fixed if fixed is not None else "—"}</td><td>{r["hits"]}/192 ({r["accuracy"]:.1%})</td></tr>')
    train_table = []
    for name, h in histories.items():
        t = h['training']; v = h['validation']; test = m[name]
        train_table.append(f'<tr><td>{name}</td><td>{h["best_epoch"]}</td><td>{t["region_hits"]["1"]}/{t["candidate_region_coverage"]}</td>'
            f'<td>{v["region_hits"]["1"]}/{v["candidate_region_coverage"]}</td><td>{test["region_top1"]}/{test["candidate_region_coverage"]}</td></tr>')
    datasets = sorted(m['all']['datasets'])
    dataset_table = ''.join('<tr><td>'+name+'</td>'+''.join(f'<td>{m[name]["datasets"][ds]["hits"]}/64</td>' for ds in datasets)+'</tr>'
        for name in names+['v5_full', 'phrase_local', 'qwen_native_completed_lax', 'ui_tars'])
    comparison_html = []
    for c in summary['paired_comparisons']:
        lo, hi = c['ci95_pp']; comparison_html.append(f'<li>{esc(c["first"])} − {esc(c["second"])}：{c["delta_pp"]:+.2f} pp，95% CI [{lo:+.2f}, {hi:+.2f}]；赢 {c["wins"]} / 输 {c["losses"]}。</li>')
    a = m['all']; b = m['peaks_only']; delta = a['region_top1']-b['region_top1']
    lead = f'固定主种子：全注意力选中相交父格 {a["region_top1"]}/192，遮掉非峰值对照 {b["region_top1"]}/192，差 {delta:+d} 张。自己的落点分别命中 {a["hits"]} 和 {b["hits"]} 张。'
    report = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>全部候选注意力诊断 v6</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;max-width:1250px;margin:35px auto;padding:0 24px;color:#23303e;background:#f5f7fa}}table{{border-collapse:collapse;background:white;width:100%}}td,th{{padding:9px 12px;border-bottom:1px solid #ddd;text-align:left}}img{{max-width:100%;background:white}}pre{{white-space:pre-wrap;font-size:13px;background:#edf0f5;padding:15px}}.note{{background:#fff0c9;padding:15px;border-radius:8px}}</style>
<h1>每个候选读取全部 token × head：受控诊断 v6</h1><p>{lead}</p>
<p>本轮两个种子的方向一致，且接同一个冻结精定位器后仍有收益，支持非峰值证据有助于候选选择。主种子的配对区间仍包含零；第二种子的幅度更大，说明种子敏感性仍明显。当前实际点命中仍低于原生 Qwen 的 72/192；不能据此宣布读出问题已解决，也不能把区域候选覆盖当作可实现的点击准确率。</p>
<p class="note">这 192 张截图已经用于上一轮 v5 分析，并启发了本轮改动。本轮是复用测试集的探索性诊断，不是新的独立验证。检查点只由原验证集选择；不按本轮测试结果改模型、阈值或挑种子。</p>
<h2>唯一受控差别</h2><p>候选池、原始目标名称 token、冻结 Qwen 状态、618 训练 / 172 验证 / 192 测试划分不变。保留 v5 原分支，添加每候选每 token 的 96 头 × 6 通道输入，经过可学习投影和语义条件化 token 池化，给候选分数加残差。两组结构与参数数目完全相同（{audit['identical_parameter_counts']:,}）；对照将非峰值 token/head 的六通道全部置零。两个种子各有一对模型。</p>
<p>通道为：相对注意力的 log1p 缩放、相对头内最大值、绝对注意力的 log1p 缩放、图片总注意力、空间熵、是否为该头自身峰值。头顺序保留到可学习投影，未先做头平均。模型仍只看固定候选格，且学习投影到 32 维；本实验不等于保留完整空间注意力图。</p>
<h2>相同截图上的结果</h2><p>点命中指整数像素位于任一有效目标框/多边形内，不要求等于标注中心。父格命中指网格与目标有正面积交集，属于更宽松的区域指标。所有缺失名称保留在 192 分母。</p>
<table><tr><th>方法</th><th>父格 Top1 /192</th><th>父格 Top3 /192</th><th>同一个冻结 v5 精定位 /192</th><th>各自实际落点命中</th></tr>{''.join(table)}</table>
<img src="{embed(plots[0])}"><p>固定解码器一列对所有新模型使用完全相同的已冻结 v5 full 精定位输出，仅让排序决定选哪个格，从而单独观察选择的效果。原生 Qwen 和 UI-TARS 直接复用上一轮同图输出及解析，不重新生成，也不替它们选择新的点。</p>
<h2>各数据集的实际点命中</h2><table><tr><th>方法</th>{''.join('<th>'+esc(ds)+'</th>' for ds in datasets)}</tr>{dataset_table}</table>
<h2>泛化与误差环节</h2><table><tr><th>方法</th><th>验证集选中 epoch</th><th>训练：选中 / 池中存在</th><th>验证：选中 / 池中存在</th><th>诊断测试：选中 / 池中存在</th></tr>{''.join(train_table)}</table>
<img src="{embed(plots[1])}"><p>主模型失败分解：{esc(json.dumps(a['failure_stages'], ensure_ascii=False))}。候选池的区域覆盖率保持 {a['candidate_region_coverage']}/192。</p>
<h2>配对差异的不确定性</h2><p>按数据集分层的 5,000 次配对 bootstrap；探索性区间，未校正多重比较。parent 表示父格交集，fixed_fine 表示同一冻结精定位。</p><ul>{''.join(comparison_html)}</ul>
<h2>事后案例</h2><p>使用标签挑选一例区域选择改善和一例退步，仅用于展示机制，不用于训练或选模型。框是选中的父格；叉号是各自训练得到的落点；绿色为标注。</p>{''.join(example_html)}
<h2>复现与审计</h2><p>无需重载 VLM：重用已验证的逐 token 特征。已从原始注意力重建全部 {audit['raw_test_fields_reconstructed']} 个有效测试输入，核对候选不变，逐条重算 {audit['all_point_scores_replayed']} 个点命中，独立 CPU 回放与 GPU 的新模型落点完全一致。原 v5 文件及预测哈希未改变。</p>
<pre>.venv/python.exe scripts/attention_field_data.py --split trainval
.venv/python.exe scripts/train_attention_field_ranker.py
.venv/python.exe scripts/attention_field_data.py --split test
.venv/python.exe scripts/evaluate_attention_field_ranker.py
.venv/python.exe scripts/audit_attention_field_ranker.py
.venv/python.exe scripts/report_attention_field_ranker.py</pre>
<p>训练脚本拒绝覆盖已冻结模型。预测指纹：{audit['predictions_sha256']}。本实验是静态正样本定位，不测禁用按钮、禁止点击或实际电脑执行。</p></html>'''
    (dest/'report.html').write_text(report, encoding='utf-8')
    write_json(dest/'report_integrity.json', dict(predictions_sha256=sha256(dest/'predictions.jsonl'),
        source_sha256=sha256(Path(__file__)), plots={p.name: sha256(p) for p in plots}, plot_count=len(plots)))
    print('REPORT '+str(dest/'report.html'), flush=True)


if __name__ == '__main__': main()
