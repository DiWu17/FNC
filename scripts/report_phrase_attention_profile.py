"""Render descriptive head/token anatomy and audit sparse selector diagnostics."""
import base64
from collections import Counter
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from clickagent.action_benchmark import sha256, write_json
from diagnose_target_information import random_set_overlap_probability
from evaluate_conditioned_readouts import score_point
from run_target_token_experiment import read_jsonl


METHOD_NAMES = {
    'max_image_mass': 'Largest image mass', 'max_relative_peak': 'Sharpest relative peak',
    'min_entropy': 'Lowest entropy', 'max_absolute_peak': 'Largest absolute peak',
    'train_best_head_last_token': 'Train-best head, last token', 'uniform_peak_votes': 'Equal peak votes',
    'distinct_head_votes': 'Distinct head votes', 'train_weighted_peak_votes': 'Train-weighted peak votes'}
ROLE_NAMES = {'content_word_start': 'Content word starts', 'content_word_continuation': 'Content subword continuations',
              'function_or_relation_word': 'Function / relation words', 'number': 'Number pieces',
              'punctuation_or_space': 'Punctuation / spaces', 'cjk_piece': 'CJK pieces'}


def main():
    out = Path('runs/phrase_v4'); dest = out/'evaluation/head_token_profile'
    summary = json.loads((dest/'summary.json').read_text(encoding='utf-8'))
    prior = json.loads((dest/'train_only_head_prior.json').read_text(encoding='utf-8'))
    assert summary['source_sha256'] == sha256(Path('scripts/profile_phrase_attention_candidates.py')) == prior['source_sha256']
    test = summary['splits']['test']; figures = []
    assert test['region_hits'] == 170 and test['head_token_trials'] == 101472
    assert summary['primary_predictions_sha256'] == sha256(out/'evaluation/predictions.jsonl')
    frozen = json.loads((out/'frozen_readouts.json').read_text(encoding='utf-8'))
    for entry in frozen['models'].values():
        assert sha256(out/'models'/entry['file']) == entry['sha256']
    for source, digest in frozen['head_sources'].items():
        assert sha256(Path(source)) == digest
    cases = {c['sample_id']: c for c in read_jsonl(out/'manifest.jsonl')}
    scored = 0; selector_rows = read_jsonl(dest/'test_selector_predictions.jsonl')
    for row in selector_rows:
        c = cases[row['sample_id']]; assert not row['candidate_features_use_gt']
        for k in (1, 3, 5):
            actual = any(score_point(point, c) for point in row['points'][:k])
            assert actual == row['center_hits'][str(k)]
            scored += len(row['points'][:k])
    for method in METHOD_NAMES:
        group = [r for r in selector_rows if r['method'] == method]
        assert len(group) == test['named']
        for metric in ('region_hits', 'center_hits'):
            for k in ('1', '3', '5'):
                assert sum(r[metric][k] for r in group) == test['label_free_selectors'][method][metric][k]
    labels = read_jsonl(dest/'candidate_cases.jsonl'); labels = [r for r in labels if r['split'] == 'test']
    layer_masks = []
    for r in labels:
        heads = {h for values in r['correct_head_indices_by_token'] for h in values}
        layer_masks.append([any(h//32 == li for h in heads) for li in range(3)])
    masks = np.asarray(layer_masks)
    unique = {str(layer): int((masks[:, li] & (masks.sum(-1) == 1)).sum()) for li, layer in enumerate((15, 28, 36))}
    pair_union = int(masks[:, :2].any(-1).sum())
    existing = {r['sample_id']: r for r in read_jsonl(out/'evaluation/token_oracle/cases.jsonl')}
    shortlist_chance = {}
    for k in (1, 4, 8, 16, 32, 96):
        expected = 0.
        for r in labels:
            old = existing[r['sample_id']]; gh, gw = old['grid_shape']
            peaks = np.asarray(r['parent_indices'])[:, prior['ranking'][:k]]
            expected += random_set_overlap_probability(gh*gw, old['target_parent_count'], len(np.unique(peaks)))
        shortlist_chance[str(k)] = expected
    # Same scale across training, validation, and test; no test head selection.
    fig, axes = plt.subplots(3, 1, figsize=(15, 7.5), constrained_layout=True)
    for ax, split in zip(axes, ('train', 'val', 'test')):
        data = np.array([r['region_recall'] for r in summary['splits'][split]['heads']]).reshape(3, 32)
        im = ax.imshow(data, cmap='viridis', vmin=0, vmax=.55, aspect='auto')
        ax.set_yticks(range(3), ['Layer 15', 'Layer 28', 'Layer 36']); ax.set_xticks(range(32))
        ax.set_title(f'{split}: any-name-token target-parent coverage per head (N={summary["splits"][split]["n"]})')
        for index in prior['ranking'][:8]:
            ax.add_patch(plt.Rectangle((index % 32-.5, index//32-.5), 1, 1, fill=False, ec='orange', lw=1.8))
    axes[-1].set_xlabel('Head index (zero-based); orange cells = training-selected top 8')
    fig.colorbar(im, ax=axes, label='GT-assisted candidate coverage', shrink=.8)
    path = dest/'head_anatomy.png'; fig.savefig(path, dpi=145); plt.close(fig); figures.append(path)
    # Fixed head shortlist, not a deployed selector.
    fig, ax = plt.subplots(figsize=(9, 5.5))
    for split in ('train', 'val', 'test'):
        d = summary['splits'][split]; budget = d['fixed_train_head_candidate_recall']
        x = list(map(int, budget)); y = [100*v/d['n'] for v in budget.values()]
        ax.plot(x, y, marker='o', label=f'{split} (N={d["n"]})')
    ax.plot(list(map(int, shortlist_chance)), [100*v/test['n'] for v in shortlist_chance.values()],
            '--', color='gray', marker='.', label='test: uniform spatial-assignment expectation')
    ax.set_xscale('log', base=2); ax.set_xticks([1, 4, 8, 16, 32, 96], [1, 4, 8, 16, 32, 96])
    ax.set_xlabel('Number of heads retained, fixed using training labels only')
    ax.set_ylabel('Target-parent candidate coverage (%)'); ax.set_ylim(0, 100)
    ax.set_title('All generated name tokens retained; GT selects a matching candidate')
    ax.grid(alpha=.2); ax.legend(); fig.tight_layout()
    path = dest/'fixed_head_budget.png'; fig.savefig(path, dpi=145); plt.close(fig); figures.append(path)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={'width_ratios': [1, 1.6]})
    positions = test['positions']; keys = ['single', 'first', 'middle', 'last']
    axes[0].bar(range(4), [positions[k]['map_hit_rate']*100 for k in keys], color='#4577ba')
    axes[0].set_xticks(range(4), [f'{k}\n(n={positions[k]["token_instances"]})' for k in keys])
    axes[0].set_title('Name-token position'); axes[0].set_ylabel('Individual token/head peak overlap (%)')
    roles = test['lexical_roles']; keys = list(ROLE_NAMES)
    axes[1].barh(range(len(keys)), [roles[k]['map_hit_rate']*100 for k in keys], color='#409c83')
    axes[1].set_yticks(range(len(keys)), [f'{ROLE_NAMES[k]} (n={roles[k]["token_instances"]})' for k in keys])
    axes[1].set_xlabel('Individual token/head peak overlap (%)'); axes[1].set_title('Heuristic lexical categories; not POS tags')
    for ax in axes:
        ax.grid(axis='y' if ax == axes[0] else 'x', alpha=.2)
    fig.suptitle('Descriptive rates; token categories differ in images, targets, and name length')
    fig.tight_layout(); path = dest/'token_anatomy.png'; fig.savefig(path, dpi=145); plt.close(fig); figures.append(path)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.2), sharey=True)
    methods = list(METHOD_NAMES); ypos = np.arange(len(methods))
    for ax, metric, title in zip(axes, ['region_hits', 'center_hits'], ['Parent-region overlap', 'Actual integer-pixel center hit']):
        values = test['label_free_selectors']
        one = [100*values[m][metric]['1']/test['n'] for m in methods]
        five = [100*values[m][metric]['5']/test['n'] for m in methods]
        ax.barh(ypos-.17, one, height=.32, label='Top 1')
        ax.barh(ypos+.17, five, height=.32, label='Top 5 / all available if fewer')
        ax.set_yticks(ypos, [METHOD_NAMES[m] for m in methods]); ax.set_xlim(0, 100)
        ax.set_title(title); ax.set_xlabel('Success (%) of all 192 test screenshots'); ax.grid(axis='x', alpha=.2)
    axes[0].invert_yaxis(); axes[1].legend(loc='lower right')
    fig.suptitle('Fixed simple selectors, without test GT; already-examined test, exploratory only')
    fig.tight_layout(); path = dest/'simple_selectors.png'; fig.savefig(path, dpi=145); plt.close(fig); figures.append(path)
    fig, axes = plt.subplots(1, 3, figsize=(11, 4.1))
    for ax, label in zip(axes, ('all', 'correct', 'incorrect')):
        data = np.asarray(test['peak_spatial_thirds'][label])*100
        im = ax.imshow(data, vmin=0, vmax=30, cmap='magma')
        for rr in range(3):
            for cc in range(3):
                ax.text(cc, rr, f'{data[rr, cc]:.1f}%', ha='center', va='center', color='white')
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(label+' map peaks')
    fig.suptitle('Peak location in a 3x3 screen partition; per-map frequencies, GUI layout confounded')
    fig.colorbar(im, ax=axes, shrink=.7, label='Share of peaks (%)')
    path = dest/'spatial_bias.png'; fig.savefig(path, dpi=145); plt.close(fig); figures.append(path)
    # Preserve source examples for every head/token instead of only top heads.
    top = sorted(test['heads'], key=lambda x: -x['coverage_excess'])[:12]
    head_rows = ''.join(f'<tr><td>{h["label"]}</td><td>{h["region_hits"]}/192</td><td>{h["center_point_hits"]}/192</td><td>{h["random_assignment_expected"]:.1f}/192</td><td>{h["case_balanced_map_hit_rate"]*100:.1f}%</td></tr>' for h in top)
    word_rows = ''.join(f'<tr><td>{html.escape(word)}</td><td>{v["images_available"]}</td><td>{v["images_with_any_hit"]}</td><td>{v["map_hit_rate"]*100:.1f}%</td></tr>'
        for word, v in sorted(test['recurring_words'].items(), key=lambda x: -x[1]['images_available']))
    selector_rows_html = ''.join(f'<tr><td>{html.escape(METHOD_NAMES[m])}</td><td>{test["label_free_selectors"][m]["region_hits"]["1"]}</td><td>{test["label_free_selectors"][m]["region_hits"]["5"]}</td><td>{test["label_free_selectors"][m]["center_hits"]["1"]}</td><td>{test["label_free_selectors"][m]["first_parent_border"]}/{test["named"]}</td></tr>' for m in methods)
    images = ''.join('<img alt="'+path.stem+'" src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'">' for path in figures)
    document = '<!doctype html><meta charset="utf-8"><title>名称token和注意力头的空间候选分布</title><style>body{font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;max-width:1200px;margin:35px auto;padding:0 24px;color:#172033;background:#f8fafc}table{border-collapse:collapse;width:100%;background:white}td,th{border:1px solid #d6deea;padding:7px;text-align:left}th{background:#e9eff8}img{max-width:100%;margin:20px 0;background:white}.note{background:#fff3d6;padding:15px}h1{font-size:28px}h2{margin-top:35px}</style>'
    document += '<h1>目标名称token × 注意力头：信号在哪里，怎样筛选？</h1><p>部分已有空间候选未被当前读出利用。需要学习带目标语义的候选选择，并保留细定位与拒绝能力；仅选择固定token、最亮头或多数投票均不足。</p>'
    document += '<p class="note">训练618题、验证172题、已看测试192题，全部沿用现有缓存。头先验只从训练标签统计，在读取验证与测试前固定。测试分析仍属于事后诊断；GT辅助候选覆盖、无需GT的实际选择、最终像素点击严格分开。未执行新VLM推理、未训练主干或修改正式预测。</p>'
    document += '<h2>1. 区域候选确实存在，但正确峰值稀疏</h2><p>170/192张至少有一个目标区域峰值；全部1057个名称token × 96头产生101472张注意力图，只有5782个峰值与GT父区域重叠（5.70%）。这些峰值来自不同功能的头，不应全按同等定位证据处理。找到目标的170张中，正确头数量中位数11，16张仅一个头有目标峰值，44张最多四个；38张仅一个token有目标峰值。</p>'
    document += '<h2>2. 头身份有稳定结构</h2><p>第15层单独覆盖153/192，第28层138/192，第36层99/192。第15层head12在训练、验证、测试上都是强头。层号从1编号，头号从0编号。</p>'
    document += f'<p>第15+28层并集覆盖{pair_union}/192；第36层独有覆盖{unique["36"]}张。训练固定的前8头覆盖训练412/618、验证129/172、测试144/192；前32头在测试覆盖161/192，仍会漏掉全部96头可覆盖的9张，因此头先验适合软加权，不能直接当作无损硬过滤。</p>'
    document += f'<p>前8头在保持不同峰值格数量的均匀随机空间分配下，预期覆盖{shortlist_chance["8"]:.1f}/192；实际144/192。最强单头L15/H12为96/192，对照预期{shortlist_chance["1"]:.1f}/192。减少候选后依然有明显空间信号；该对照仍不等于真实GUI先验。</p>'
    document += '<table><tr><th>头</th><th>任意名称token区域覆盖</th><th>峰值中心实际点覆盖</th><th>相同不同峰值数的随机空间期望</th><th>逐图均衡的单图峰值命中率</th></tr>'+head_rows+'</table>'
    document += '<h2>3. 哪种token有信号？</h2><p>目标名称的实词起始片段、后续子词、数字、中文片段都可能有信号；完整词尚未结束的子词也不能丢弃。172个多token名称中，首token某头命中121张，末token命中108张，二者不是同一个集合。有145张至少3个token，遍历其中间token覆盖121张，候选更多，不能据此宣称中间位置天然更好。功能词与标点也可能继承前文目标信息，不能只按“名词”硬筛。</p>'
    document += '<p>以下为至少4张图出现的还原词片段；数值受目标大小、数据集和分词长度影响。词法标签仅启发式，不是人工验证的词性。</p><table><tr><th>还原词/片段</th><th>出现图数</th><th>该词某token某头能覆盖的图数</th><th>token×head峰值重叠率</th></tr>'+word_rows+'</table>'
    document += '<h2>4. 简单筛选规则不足，强背景共识会干扰</h2><p>正确峰值的图像总注意力中位数0.120，错误峰值0.044；正确峰值平均更尖锐。但单独选最大总量仅21张区域正确，最低熵22张，最大绝对注意力47张。趋势不能直接当成可靠选择规则。等权峰值投票只选中5张，训练头先验加权投票提升到53张，仍未达到当前正式粗区域选择的76张。需要目标语义和候选区域内容来区分有用头与稳定背景峰值。</p>'
    document += '<table><tr><th>无需测试GT的固定规则</th><th>Top1区域命中/192</th><th>最多Top5区域命中/192</th><th>Top1像素中心命中/192</th><th>首选峰值在网格最外圈</th></tr>'+selector_rows_html+'</table>'
    document += images
    document += '<h2>下一轮结构建议</h2><ol><li>保留每个token、每个头的候选与来源，按空间位置聚合候选，但保留来源列表；头先验作为软特征，保留长尾头。</li><li>训练目标语义条件化的候选排序器：融合头/层身份、目标token隐藏状态、完整短语状态、候选位置的图像特征、局部注意力形状、总量和跨头/跨token支持。每个候选独立打分，避免先压成单张平均图。</li><li>用同图其他相似按钮、同名控件和稳定背景峰值作为难负例，增加同图多目标指令，学习选区域及“候选均不正确”。</li><li>分别报告候选Recall@K、候选存在时的Top1选中率、选对区域后的细定位命中，以及阴性误点率。细化应单独核验，必要时放大原图候选裁剪后再次编码。</li><li>所有模型与超参数仅用训练/验证选择；最终重新保留未看过的截图测试，当前192张只作诊断。</li></ol>'
    document += '<p>机器数据：summary.json（全96头、分词法与位置统计）、head_stats.csv、token_role_stats.csv、candidate_cases.jsonl（每个token哪些头命中）、candidate_features.npz（14项无需GT的输入与分开存放的监督标签）、train_only_head_prior.json及各split的selector_predictions.jsonl。</p>'
    (dest/'report.html').write_text(document, encoding='utf-8')
    audit = dict(n_test=test['n'], primary_predictions_unchanged=True,
        frozen_checkpoints_verified=len(frozen['models']), frozen_sources_verified=len(frozen['head_sources']),
        selector_candidate_points_replayed=scored, selector_counts_match=True,
        test_layer_exclusive_region_coverage=unique, test_layer15_28_union=pair_union,
        test_train_shortlist_random_assignment_expected=shortlist_chance,
        source_sha256=sha256(Path(__file__)), profile_source_sha256=summary['source_sha256'],
        primary_predictions_sha256=summary['primary_predictions_sha256'], tests_passed=3,
        figure_sha256={p.name: sha256(p) for p in figures})
    write_json(dest/'audit.json', audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
