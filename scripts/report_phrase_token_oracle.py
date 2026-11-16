"""Audit exact candidate scores and render a token-existence supplement."""
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
import torch

from clickagent.action_benchmark import sha256, write_json
from clickagent.conditioned_grounder import make_batch
from clickagent.phrase_grounder import load_phrase
from evaluate_conditioned_readouts import score_point
from run_target_token_experiment import key, read_jsonl
from train_conditioned_readouts import load_records


def text(value):
    return html.escape(str(value))


def embedded(path):
    return 'data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()


def image_plot(case, row, path):
    image = Image.open(case['image']).convert('RGB'); w, h = image.size
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), gridspec_kw={'width_ratios': [1.6, 1]})
    target = case['target_geometry']
    boxes = target.get('boxes')
    if boxes is None:
        boxes = [[min(x[0] for x in poly), min(x[1] for x in poly),
                  max(x[0] for x in poly), max(x[1] for x in poly)] for poly in target['polygons']]
    for ax in axes:
        ax.imshow(image)
        if target['type'] == 'bbox':
            for x0, y0, x1, y1 in boxes:
                ax.add_patch(plt.Rectangle((x0*w, y0*h), (x1-x0)*w, (y1-y0)*h,
                                          fill=False, edgecolor='lime', linewidth=2))
        else:
            for poly in target['polygons']:
                ax.add_patch(plt.Polygon(np.asarray(poly)*[w, h], closed=True, fill=False, edgecolor='lime', linewidth=2))
        for name, point, color, marker in [('Current', row['current_point'], '#ff4157', 'x'),
                                          ('Qwen native', row['native_point'], '#208bff', 'o')]:
            if point is not None:
                options = dict(facecolors='none', edgecolors=color) if marker == 'o' else dict(color=color)
                ax.scatter(*point, s=90, marker=marker, linewidths=2, label=name, **options)
        selected = row['families']['learned_token_forced_route']
        for i in selected['hit_candidate_indices'][:1]:
            ax.scatter(*selected['points'][i], s=80, marker='+', color='orange', linewidths=2, label='Correct token candidate')
        raw = row['families']['raw_head_any_token_coarse_center']
        for i in raw['hit_candidate_indices'][:1]:
            ax.scatter(*raw['points'][i], s=80, marker='*', color='cyan', label='Correct raw head center')
        ax.set_axis_off()
    x0, y0, x1, y1 = boxes[0]; pad_x = max((x1-x0)*2, .04); pad_y = max((y1-y0)*2, .04)
    axes[1].set_xlim(max(0, x0-pad_x)*w, min(1, x1+pad_x)*w)
    axes[1].set_ylim(min(1, y1+pad_y)*h, max(0, y0-pad_y)*h)
    axes[0].set_title('Full screenshot'); axes[1].set_title('Target zoom (green outline)')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=4, frameon=False)
    fig.tight_layout(rect=[0, .08, 1, 1]); fig.savefig(path, dpi=130); plt.close(fig)


@torch.inference_mode()
def heatmap_plot(out, case, row, model, path):
    records, _ = load_records(out, [case], with_labels=False); record = records[0]
    inputs, _ = make_batch(records, 'cpu', labels=False); result = model(**inputs)
    a = inputs['attention'][0].numpy(); p = a.shape[-1]
    relative = a/np.maximum(a.sum(-1, keepdims=True), 1e-12)
    evidence = np.einsum('th,thp->tp', result['head_weights'][0].numpy(), np.log1p(p*relative))
    gt = case['target_geometry']; gh, gw = row['grid_shape']
    raw_family = row['families']['raw_head_any_token_coarse_center']
    candidates = raw_family['hit_candidate_indices']
    chosen = candidates[0] if candidates else 0
    selected_token, selected_head = divmod(chosen, 96)
    panels = [('Final fused map', result['coarse_logits'][0].numpy(), None)]
    panels.extend((f'Token {i} | rank {token["best_target_cell_rank"]} | '+('HIT' if token['learned_forced_hit'] else 'miss'),
                   evidence[i].reshape(gh, gw), token) for i, token in enumerate(row['tokens']))
    layer = [15, 28, 36][selected_head//32]; head = selected_head % 32
    panels.append((f'Raw L{layer} head {head}, token {selected_token}', a[selected_token, selected_head].reshape(gh, gw), None))
    cols = min(4, len(panels)); nrows = (len(panels)+cols-1)//cols
    fig, axes = plt.subplots(nrows, cols, figsize=(4.2*cols, 3.5*nrows), squeeze=False)
    for ax, (title, values, _) in zip(axes.flat, panels):
        ax.imshow(values, extent=[0, 1, 1, 0], interpolation='nearest', cmap='magma', aspect='auto')
        if gt['type'] == 'bbox':
            for x0, y0, x1, y1 in gt['boxes']:
                ax.add_patch(plt.Rectangle((x0, y0), x1-x0, y1-y0, fill=False, edgecolor='lime', linewidth=2))
        else:
            for poly in gt['polygons']:
                ax.add_patch(plt.Polygon(poly, closed=True, fill=False, edgecolor='lime', linewidth=2))
        rr, cc = divmod(int(values.argmax()), gw)
        ax.scatter((cc+.5)/gw, (rr+.5)/gh, color='cyan', marker='+', s=65)
        ax.set_title(title, fontsize=9); ax.set_xticks([]); ax.set_yticks([])
    for ax in list(axes.flat)[len(panels):]:
        ax.set_axis_off()
    fig.suptitle('Each map uses its own color range. Green = target, cyan = map peak.\nToken indices refer to the exact generated pieces in the table.', fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, .92]); fig.savefig(path, dpi=135); plt.close(fig)
    return dict(raw_example_token=selected_token, raw_example_layer=layer, raw_example_head_zero_based=head)


def main():
    out = Path('runs/phrase_v4'); dest = out/'evaluation/token_oracle'
    rows = read_jsonl(dest/'cases.jsonl'); summary = json.loads((dest/'summary.json').read_text(encoding='utf-8'))
    cases = {c['sample_id']: c for c in read_jsonl(out/'manifest.jsonl') if c['experiment_split'] == 'test'}
    # Replay every stored candidate with the SAME scorer used by primary eval.
    tested = 0
    for r in rows:
        if not r['available']:
            continue
        for family in r['families'].values():
            hits = [i for i, point in enumerate(family['points']) if score_point(point, cases[r['sample_id']])]
            assert hits == family['hit_candidate_indices'], (r['sample_id'], hits, family)
            assert bool(hits) == family['hit']
            tested += len(family['points'])
    for path, digest in summary['integrity_before'].items():
        assert sha256(Path(path)) == digest
    extra = dict(n=len(rows), exact_candidates_replayed=tested, all_scores_match_primary_scorer=True,
        fixed_token_choices={}, error_partition_native_gap=dict(Counter(r.get('diagnostic_stage', 'no_name')
            for r in rows if r['native_hit'] and not r['current_hit'])))
    available = [r for r in rows if r['available']]
    extra['raw_head_center_vs_fine'] = dict(
        fine_rescues_center=sum(r['families']['raw_head_any_token']['hit'] and not r['families']['raw_head_any_token_coarse_center']['hit'] for r in available),
        center_rescues_fine=sum(r['families']['raw_head_any_token_coarse_center']['hit'] and not r['families']['raw_head_any_token']['hit'] for r in available))
    for group_name, group in [('all', rows), ('native_correct_current_wrong', [r for r in rows if r['native_hit'] and not r['current_hit']])]:
        values = {}
        for fam in ['learned_token_forced_route', 'single_token_entire_state', 'raw_l15_token_forced_route']:
            values[fam] = dict(first_hits=sum(0 in r['families'][fam]['hit_candidate_indices'] for r in group if r['available']),
                               last_hits=sum(r['n_tokens']-1 in r['families'][fam]['hit_candidate_indices'] for r in group if r['available']))
        extra['fixed_token_choices'][group_name] = values
    native_gap = [r for r in rows if r['native_hit'] and not r['current_hit'] and r['available']]
    selection_rules = [
        ('token_rescue', '换token能恢复正确点击', lambda r: r['families']['learned_token_forced_route']['hit'] and not r['tokens'][0]['learned_forced_hit']),
        ('head_rescue', 'token融合失败，但原始头的峰值中心能命中', lambda r: not r['families']['learned_token_forced_route']['hit'] and r['families']['raw_head_any_token_coarse_center']['hit']),
        ('fine_miss', 'token图包含正确父格，但当前细定位都未命中', lambda r: r['families']['learned_token_forced_route']['coarse_overlap'] and not r['families']['learned_token_forced_route']['hit']),
        ('no_peak', '原生坐标正确，缓存各token各头均无目标峰值', lambda r: not r['families']['raw_head_any_token']['coarse_overlap'] and not r['families']['raw_head_any_token']['hit']),
    ]
    selected = []; used = set()
    for slug, title, predicate in selection_rules:
        pool = sorted([r for r in native_gap if predicate(r) and r['sample_id'] not in used], key=lambda r: r['sample_id'])
        if pool:
            selected.append((slug, title, pool[0])); used.add(pool[0]['sample_id'])
    torch.set_num_threads(4); model, _ = load_phrase(out/'models/local.pt', 'cpu')
    sections = []
    for slug, title, row in selected:
        case = cases[row['sample_id']]; image_path = dest/f'{slug}_screenshot.png'; map_path = dest/f'{slug}_maps.png'
        image_plot(case, row, image_path); example = heatmap_plot(out, case, row, model, map_path)
        extra.setdefault('examples', []).append(dict(kind=slug, sample_id=row['sample_id'], **example,
            selection='First sample_id in the native-correct/current-wrong subset satisfying the stated rule; no visual-quality sorting.'))
        trs = ''.join('<tr>'+''.join(f'<td>{value}</td>' for value in [tok['index'], '<code>'+text(repr(tok['piece']))+'</code>',
                     tok['learned_peak_overlap'], text(tok['learned_forced_point']), tok['learned_forced_hit'],
                     tok['best_target_cell_rank'], f"{tok['weight_at_final_parent']:.3f}"] )+'</tr>' for tok in row['tokens'])
        comparison = ''
        raw_centers = row['families']['raw_head_any_token_coarse_center']
        if raw_centers['hit_candidate_indices']:
            raw_index = raw_centers['hit_candidate_indices'][0]
            comparison = f'<p>图示原始头峰值中心：{text(raw_centers["points"][raw_index])}；同一个峰值经过冻结细定位后：{text(row["families"]["raw_head_any_token"]["points"][raw_index])}。两种落点分别计分，不能将父格正确自动视作细定位正确。</p>'
        sections.append(f'<section><h2>{text(title)}</h2><p>{text(row["sample_id"])} · {text(row["instruction"])}<br>生成目标名称：{text(row["target_name"])}</p>'
            f'<img src="{embedded(image_path)}"><img src="{embedded(map_path)}">'
            '<p>热图颜色分别归一化；颜色鲜艳不代表绝对图像注意力总量高。图中父格命中与最终像素点击命中分开统计。</p>'
            +comparison+'<table><tr><th>token序号</th><th>真实生成片段</th><th>峰值父格重叠目标</th><th>强制路由后坐标</th><th>像素命中</th><th>最佳目标父格名次</th><th>原最终父格上的token权重</th></tr>'+trs+'</table></section>')
    all_f = summary['groups']['all']['families']; gap_f = summary['groups']['native_correct_current_wrong']['families']
    labels = [('learned_token_original_fine', '逐token选图，细定位不变'),
              ('learned_token_forced_route', '逐token选图，同时路由该token的细定位'),
              ('single_token_entire_state', '每次只输入一个token的全部已存状态'),
              ('raw_l15_token_coarse_center', '逐token，第15层平均头峰值中心'),
              ('raw_head_any_token_coarse_center', '逐token × 各原始头，直接峰值格中心'),
              ('raw_head_any_token', '逐token × 各原始头，加冻结细定位')]
    table = ''.join(f'<tr><td>{text(label)}</td><td>{all_f[name]["hits"]}/192 ({all_f[name]["accuracy"]*100:.2f}%)</td>'
        f'<td>{all_f[name]["coarse_overlap"]}/192</td><td>{gap_f[name]["hits"]}/46</td>'
        f'<td>{all_f[name]["shift_null_mean_hits"]:.1f}/192</td></tr>' for name, label in labels)
    document = '<!doctype html><meta charset="utf-8"><title>逐token定位信息存在性诊断</title><style>'
    document += 'body{font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;max-width:1200px;margin:35px auto;padding:0 25px;background:#f8fafc;color:#172033}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #d6deea;padding:8px;text-align:left}th{background:#e9eff8}img{max-width:100%;background:white;margin-top:15px}section{padding-top:25px;margin-top:30px;border-top:2px solid #d6deea}code{white-space:pre-wrap}.note{background:#fff4d9;padding:14px}h1{font-size:28px}h2{font-size:22px}</style>'
    document += '<h1>答案是否已经存在于某个目标名称token中？</h1><p>结论：部分样本确实存在可恢复的正确落点；但只选对token不足以补齐原生坐标输出的差距。头融合与细定位仍会丢失信号。没有找到峰值的样本，不能据此判定整个VLM内部不存在位置信息。</p>'
    document += '<p class="note">这是同一批192张已评分截图的事后诊断，没有训练、调参或修改主结果。下表除现有方法和原生Qwen外，都由真实标注事后挑选候选，是诊断上界，不是可部署准确率。实际指标为像素落点进入任一正确框/多边形，含边界；并非要求与单一label坐标完全相等。</p>'
    document += '<p>现有方法：48/192（25.00%）；原生Qwen完整宽松格式坐标：88/192（45.83%）。每张有名称截图平均5.53个token；每token保留96个头，遍历全部时平均531个候选。</p>'
    document += '<table><tr><th>标注辅助诊断</th><th>实际落点命中</th><th>仅父格与目标重叠</th><th>原生对而当前错的46张中恢复</th><th>整体平移对照平均命中</th></tr>'+table+'</table>'
    document += '<p>只替换token路由：20张当前失败可恢复；若允许保留当前结果并在标注帮助下二选一，上界为68/192，仍低于原生88/192。原生正确但当前失败的46张中，仅12张通过token路由恢复；原始头峰值格中心可恢复26张，加冻结细定位可恢复31张。这是提取/融合会漏掉现有信号的证据。</p>'
    fixed = extra['fixed_token_choices']['all']['learned_token_forced_route']
    document += f'<p>同一个冻结模型强制使用第一个token：{fixed["first_hits"]}/192；最后一个token：{fixed["last_hits"]}/192；标注辅助挑选任意token：64/192。没有证据支持简单固定某个词位置就能解决问题。</p>'
    document += '<p>对照将整套峰值共同作随机环形空间平移，保留候选数量、重复位置和相对形状，再用目的位置的原图细定位特征解码。原始头加细定位99次命中，对照平均65.3次，不能把99次全算成有效信号。此对照不是现实GUI先验；更多头带来的覆盖与原图细定位贡献仍需区分。</p>'
    document += '<p>我们检查的是第15/28/36层的已存注意力与隐藏状态。强制token路由仍使用最后一个token的完整短语上下文；“仅一个token全部状态”会连上下文一起替换，属于训练分布外干预。直接坐标提示的后续计算轨迹并未包含在这些缓存中。因而不能把当前读出器找不到答案解释为信息根本不存在。</p>'
    document += '<p>候选扩展诊断：学习头融合图的最佳目标父格，在某个token中排名第1的有107张，前3名131张，前5名144张。这只说明区域候选生成还有空间，不等于同等数量的正确点击。</p>'
    document += ''.join(sections)
    document += '<section><h2>复核与复现</h2><p>所有逐token实际像素候选均用主评测器重新计分；主模型参数、模型源文件、主预测JSONL哈希核对不变。三个诊断单元测试覆盖逐父格精确解码、整套平移保留重复与可逆性、框/多边形边界计分。</p><p>运行：<code>.venv/python.exe scripts/diagnose_phrase_token_oracle.py</code>，再运行<code>.venv/python.exe scripts/report_phrase_token_oracle.py</code>。数据见同目录summary.json、cases.jsonl、audit.json。</p></section>'
    (dest/'report.html').write_text(document, encoding='utf-8')
    extra['source_sha256'] = sha256(Path(__file__))
    extra['primary_files_unchanged'] = all(sha256(Path(path)) == digest for path, digest in summary['integrity_before'].items())
    write_json(dest/'audit.json', extra)
    print(json.dumps(extra, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
