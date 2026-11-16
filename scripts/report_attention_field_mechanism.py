"""Render the post-hoc frozen-readout diagnostic, without altering v6 results."""
import base64
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from clickagent.action_benchmark import sha256, write_json
from run_target_token_experiment import key, read_jsonl
from report_attention_field_ranker import example


OUT = Path('runs/attention_field_v6'); DEST = OUT/'evaluation/mechanism_analysis'
def read(path): return json.loads(Path(path).read_text(encoding='utf-8'))
def esc(x): return html.escape(str(x))
def embed(p): return 'data:image/png;base64,'+base64.b64encode(p.read_bytes()).decode()


def main():
    summary = read(DEST/'summary.json'); assert summary['complete']
    details = read(DEST/'rescue_attributions.json'); official = read_jsonl(OUT/'evaluation/predictions.jsonl')
    by = {(r['method'], r['sample_id']): r for r in official}
    rows = read_jsonl(DEST/'predictions.jsonl'); dy = {(r['model'], r['intervention'], r['sample_id']): r for r in rows}
    cases = {c['sample_id']: c for c in read_jsonl(OUT/'manifest.jsonl') if c['experiment_split'] == 'test'}
    derived = {}; table = []; labels = ['full', 'nonpeak_all_zero', 'nonpeak_spatial_zero', 'nonpeak_context_zero', 'spatial_shuffle_mean', 'no_field_residual']
    for name in ['all', 'all_replica']:
        m = summary['methods'][name]; wins = summary['cohorts'][name]['region_wins']
        shuffle = [m[f'spatial_shuffle_{s}'] for s in range(5)]
        m = dict(m, spatial_shuffle_mean={k: float(np.mean([v[k] for v in shuffle])) for k in ['hits', 'region_hits']})
        stats = {}
        for label in labels:
            metric = m[label]
            table.append(f'<tr><td>{name}</td><td>{label}</td><td>{metric["region_hits"]:g}</td><td>{metric["hits"]:g}</td></tr>')
        ds = [d for d in details if d['model'] == name]
        positive = [d for d in ds if d['margin_gain_from_spatial_nonpeaks'] > 0]
        shares = []
        for d in positive:
            h = np.maximum(d['per_head_margin_contribution'], 0); shares.append(h/max(h.sum(), 1e-12))
        share = np.mean(shares, axis=0) if shares else np.zeros(96)
        top = np.argsort(-share)[:8]
        head_name = lambda h: f'layer_{[15,28,36][h//32]}.head_{h%32}'
        top_tokens = CounterToken(ds)
        stats.update(spatial_margin_positive_cases=len(positive), rescue_n=len(ds),
            spatial_erasure_margin_flips=m['nonpeak_spatial_zero']['rescue_margin_flips'],
            spatial_erasure_rescue_region_hits=m['nonpeak_spatial_zero']['rescue_cohort_region_hits'],
            context_erasure_margin_flips=m['nonpeak_context_zero']['rescue_margin_flips'],
            context_erasure_rescue_region_hits=m['nonpeak_context_zero']['rescue_cohort_region_hits'],
            shuffle_point_hits=[s['hits'] for s in shuffle], shuffle_region_hits=[s['region_hits'] for s in shuffle],
            top_positive_attribution_heads=[dict(head=head_name(int(h)), mean_positive_share=float(share[h])) for h in top],
            top_token_positions=top_tokens,
            top3_head_erasure_margin_flips=sum(d['top3_head_erased_margin'] <= 0 for d in ds),
            correct_has_more_halfmax_secondary_support=sum(d['correct_support']['nonpeak_ratio_ge_half'] > d['wrong_support']['nonpeak_ratio_ge_half'] for d in ds),
            correct_has_more_peak_votes=sum(d['correct_support']['peak_votes'] > d['wrong_support']['peak_votes'] for d in ds))
        derived[name] = dict(metrics=m, rescue_analysis=stats)
    plots = []
    names = ['all', 'all_replica']; positions = np.arange(len(labels)); fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    short = ['Unchanged', 'Erase all nonpeak', 'Erase spatial only', 'Erase mass/entropy', 'Shuffle spatial (5 avg)', 'Base branch only']
    for ax, field, title in zip(axes, ['region_hits', 'hits'], ['Parent intersection / 192', 'Actual point hits / 192']):
        for j, name in enumerate(names):
            vals = [derived[name]['metrics'][label][field] for label in labels]
            bars = ax.barh(positions+(j-.5)*.34, vals, height=.34, label=name)
            for b, v in zip(bars, vals): ax.text(v+.6, b.get_y()+b.get_height()/2, f'{v:g}', va='center', fontsize=9)
        ax.set_yticks(positions, short); ax.invert_yaxis(); ax.set_title(title); ax.legend(); ax.grid(axis='x', alpha=.15)
        ax.set_xlim(0, max(derived[n]['metrics'][l][field] for n in names for l in labels)*1.2+1)
    fig.tight_layout(); path = DEST/'frozen_interventions.png'; fig.savefig(path, dpi=140); plt.close(fig); plots.append(path)
    # Case selection is explicitly diagnostic: prefer primary model, pixel hit,
    # spatial erasure changing the selected-vs-control score ordering.
    pool = [d for d in details if d['model'] == 'all' and d['spatial_erasure_margin'] <= 0 and by['all', d['sample_id']]['hit']]
    if not pool: pool = [d for d in details if d['model'] == 'all']
    chosen = sorted(pool, key=lambda d: -d['margin_gain_from_spatial_nonpeaks'])[0]
    sid = chosen['sample_id']; case = cases[sid]
    path = DEST/'case_screenshot.png'; example(case, by['all', sid], by['peaks_only', sid], path); plots.append(path)
    with np.load(OUT/'field_cache'/f'{key(sid)}.npz', allow_pickle=False) as z: field = z['attention_field']
    t = len(chosen['token_pieces']); fig, axes = plt.subplots(2, 1, figsize=(14, max(5.5, t*.5+2)), sharex=True)
    for ax, ci, title in zip(axes, [chosen['selected_candidate'], chosen['control_candidate']], ['Correct parent selected by full field', 'Distractor parent selected by trained masked control']):
        f = field[ci]; im = ax.imshow(f[..., 1], vmin=0, vmax=1, cmap='viridis', aspect='auto')
        rr, cc = np.nonzero(f[..., 5]); ax.scatter(cc, rr, marker='.', c='white', s=20)
        ax.axvline(31.5, color='white', alpha=.5); ax.axvline(63.5, color='white', alpha=.5)
        safe = [s.encode('ascii', errors='backslashreplace').decode() for s in chosen['token_pieces']]
        ax.set_yticks(np.arange(t), safe); ax.set_title(title)
    ticks = list(range(0, 96, 8)); axes[1].set_xticks(ticks, [f'L{[15,28,36][h//32]} H{h%32}' for h in ticks], rotation=30, ha='right')
    axes[1].set_xlabel('Head identity; white dots = own argmax (other cells are secondary support)')
    fig.subplots_adjust(left=.16, right=.89, bottom=.18, top=.92, hspace=.45)
    cax = fig.add_axes([.91, .23, .015, .6]); fig.colorbar(im, cax=cax, label='Attention / this head maximum')
    path = DEST/'case_head_fields.png'; fig.savefig(path, dpi=140); plt.close(fig); plots.append(path)
    chosen['top_head_relative_responses'] = [dict(head=h['head'],
        correct=field[chosen['selected_candidate'], :, h['index'], 1].astype(float).tolist(),
        distractor=field[chosen['control_candidate'], :, h['index'], 1].astype(float).tolist()) for h in chosen['top_heads']]
    head_table = ''.join(f'<tr><td>{esc(h["head"])}</td><td>{h["contribution"]:+.3f}</td>'
        f'<td>{" / ".join(f"{v:.3f}" for v in responses["correct"])}</td>'
        f'<td>{" / ".join(f"{v:.3f}" for v in responses["distractor"])}</td></tr>'
        for h, responses in zip(chosen['top_heads'], chosen['top_head_relative_responses']))
    token_table = ''.join(f'<tr><td>{i}</td><td>{esc(piece)}</td><td>{value:+.3f}</td></tr>' for i, (piece, value) in enumerate(zip(chosen['token_pieces'], chosen['per_token_margin_contribution'])))
    rescue_stats = ''.join(f'<h3>{n}</h3><pre>{esc(json.dumps(derived[n]["rescue_analysis"], ensure_ascii=False, indent=2))}</pre>' for n in names)
    write_json(DEST/'derived_findings.json', dict(models=derived, chosen_case=chosen))
    report = f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>v6为什么有正向提升</title>
<style>body{{font:16px/1.7 system-ui,sans-serif;max-width:1200px;margin:35px auto;padding:0 24px;background:#f5f7fa;color:#253142}}table{{border-collapse:collapse;background:white;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd;text-align:left}}img{{max-width:100%;background:white}}pre{{white-space:pre-wrap;background:#e9eef4;padding:15px;font-size:13px}}.note{{background:#fff0cb;padding:15px}}</style>
<h1>正向提升来自哪里：冻结权重后的机制诊断</h1><p class="note">所有权重、候选、图片、token轨迹和官方v6预测不变。本页是事后诊断，未训练、选模型或修改阈值。GT只用于计分、选择“完整组区域正确而独立训练的遮挡组错误”的案例，以及定义解释的分数对比。不是新的独立验证。</p>
<p>目前的解释是：读出器利用了每个候选上的连续头响应模式，并同时依赖图片总注意力与熵。它没有简单选择“最多头投票”的位置。空间对齐被打乱后两个种子都明显降分，但只擦除空间通道时主种子点命中33→34、区域63→54，说明空间证据也会造成干扰，且区域交集改善不保证落点改善。现有证据支持部分机制，不足以把训练后的净提升全部归因于次高注意力，更不能把测试时擦除损失当作训练消融收益。</p>
<h2>直接检查模型是否依赖空间对应关系</h2><p>对同一个完整模型分别：保留原输入；去掉全部非峰值输入；仅去掉三通道位置注意力；仅去掉图片总量/熵；在每个token/head内部对非峰值候选的位置注意力进行5次固定随机置换；关闭新增分数残差。置换保持每个token/head的三通道联合数值分布、峰值位置和全部总量/熵不变，只破坏非峰值数值与候选位置的对应。所有干预中每个候选的精定位输出都由同一权重决定，分支改动只影响候选排序。</p>
<table><tr><th>冻结模型</th><th>干预</th><th>区域命中 /192</th><th>实际点命中 /192</th></tr>{''.join(table)}</table><img src="{embed(plots[0])}">
<p>擦除会带来分布变化，单次降分不能独立证明训练收益由该通道造成。置换比直接置零更接近原数值分布，但仍是事后干预。不同通道存在交互，擦除损失不能相加。关闭残差后使用的是与完整分支共同训练的基础分支，不是原始v5检查点。</p>
<h2>收益与退步都存在</h2><p>独立训练组比较：主种子区域纠正17张、退步13张；第二种子纠正17张、退步5张。两组共同纠正6张。区域纠正的17张中，实际点分别只命中9、7张，说明精定位仍限制收益。下面的归因只解释34个区域纠正案例，不代表全部样本。</p>{rescue_stats}
<h2>一个可检查的案例</h2><p>{esc(sid)}；指令：{esc(case['instruction'])}；生成目标：{esc(chosen['target_name'])}。</p><img src="{embed(plots[1])}"><img src="{embed(plots[2])}">
<p>本例橙色落点靠近标注下边缘，主要用于说明候选筛选，精定位仍有偏差。</p>
<p>正确候选与干扰候选的峰值票数分别为{chosen['correct_support']['peak_votes']}、{chosen['wrong_support']['peak_votes']}；非峰值且达到该头最大值一半的响应数分别为{chosen['correct_support']['nonpeak_ratio_ge_half']}、{chosen['wrong_support']['nonpeak_ratio_ge_half']}。这些统计用于描述，不是新的阈值或部署规则。</p>
<p>对“正确候选分数−遮挡组选择的干扰候选分数”做积分梯度，起点只将非峰值位置注意力三通道置零，保留总量/熵与峰值。完整分数差{chosen['full_margin']:+.3f}，空间通道贡献的分数差变化{chosen['margin_gain_from_spatial_nonpeaks']:+.3f}；擦除贡献最高的三个头后分数差{chosen['top3_head_erased_margin']:+.3f}。数值解释的是这个小读出器的分数，不是Qwen内部注意力的因果重要性。</p>
<table><tr><th>头（零起始）</th><th>对分数差的归因</th><th>正确格各token相对峰值</th><th>干扰格各token相对峰值</th></tr>{head_table}</table><table><tr><th>名称token序号</th><th>token片段</th><th>对分数差的归因</th></tr>{token_table}</table>
<p>正贡献既可能来自提高正确候选，也可能来自压低干扰候选。头和token归因是条件化结果，不能当作全局固定选择规则。积分梯度的数值完整性误差最大为{summary['attribution_completeness_max_error']:.5f}；每例的步数、残差与原始贡献在rescue_attributions.json中保留。</p>
<h2>完整性</h2><p>原始预测指纹{read(DEST/'identity.json')['official_predictions_sha256']}保持不变；重新核对{summary['diagnostic_prediction_count']}个干预点的命中。扰动函数有3项测试，确认只改指定通道、峰值不变、置换保留各头边际和联合三通道对应、随机种子可复现，以及token标签使用一基索引。</p></html>'''
    (DEST/'report.html').write_text(report, encoding='utf-8')
    write_json(DEST/'report_integrity.json', dict(source_sha256=sha256(Path(__file__)),
        official_predictions_sha256=sha256(OUT/'evaluation/predictions.jsonl'),
        plots={p.name: sha256(p) for p in plots}, plot_count=len(plots)))
    print('REPORT '+str(DEST/'report.html'), flush=True)


def CounterToken(ds):
    from collections import Counter
    result = Counter()
    for d in ds:
        idx = int(np.argmax(d['per_token_margin_contribution'])); n = len(d['token_pieces'])
        result['single' if n == 1 else 'first' if idx == 0 else 'last' if idx == n-1 else 'middle'] += 1
    return dict(result)


if __name__ == '__main__': main()
