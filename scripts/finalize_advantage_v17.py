"""Audit frozen results and export plain-text interpretation and static figures.

This does not run inference, train, select a model, or modify registered sources.
The joint-proposal breakdown is explicitly a posthoc descriptive diagnostic.
"""
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/advantage_validation_v17'
ANALYSIS = OUT / 'analysis'
NAMES = {'ours': 'Ours', 'qwen': 'Qwen direct', 'ui_tars': 'UI-TARS',
         'aima': 'GUI-AIMA', 'mvp_single': 'MVP single-view', 'mvp': 'MVP',
         'tag': 'TAG', 'ours_gui': 'Candidates + GUI verifier'}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def validate_hashes(mapping):
    for path, expected in mapping.items():
        assert sha(path) == expected, path


def audit():
    completion = read(OUT / 'completion.json')
    assert completion['complete'] and completion['mainline_unchanged']
    validate_hashes(completion['artifact_sha256'])
    protocol = read(OUT / 'experiments_protocol.json')
    validate_hashes(protocol['source_and_input_hashes'])
    data = read(OUT / 'data/cohorts.json')
    ids = {r['sample_id'] for r in data['full_test']}
    selected = {r['sample_id'] for r in data['speed_robustness']}
    assert len(ids) == len({r['pixel_sha256'] for r in data['full_test']}) == 300
    assert len(selected) == len({r['pixel_sha256'] for r in data['speed_robustness']}) == 60
    for case in data['full_test']:
        assert sha(case['image']) == case['image_sha256']
    unavailable = {r['job']: r for r in completion['resource_failures']}
    verified, parity = [], {}
    for method, pixels, stage in protocol['controlled_jobs'] + protocol['stronger_jobs']:
        name = f'{method}_p{pixels}_{stage}'
        folder = OUT / 'inference' / name
        if not (folder / 'completion.json').exists():
            assert stage == 'stronger' and name in unavailable, name
            assert unavailable[name]['resource_limit'] and unavailable[name]['no_accuracy_assigned']
            continue
        done = read(folder / 'completion.json')
        assert done['complete']
        validate_hashes(done['hashes'])
        validate_hashes(read(folder / 'protocol.json')['hashes'])
        accuracy = None
        if stage != 'speed':
            rows = read(folder / 'accuracy/results.json')['rows']
            accuracy = {r['sample_id']: r for r in rows}
            assert len(rows) == len(accuracy)
            assert set(accuracy) == (selected if stage == 'stronger' else ids)
        if stage == 'speed' or stage == 'all' and method != 'ours':
            rows = read(folder / 'speed/results.json')['rows']
            grouped = defaultdict(list)
            for row in rows:
                grouped[row['sample_id']].append(row)
                assert np.isfinite(row['measured']['elapsed_seconds']) and row['measured']['elapsed_seconds'] > 0
            assert len(rows) == 120 and set(grouped) == selected
            assert all(Counter(r['repetition'] for r in v) == {0: 1, 1: 1} for v in grouped.values())
            parity[name] = dict(repeated_point_agreement=sum(v[0]['point'] == v[1]['point'] for v in grouped.values()), n=60)
            if method in ['ours_s1', 'ours_gui', 'ui_tars']:
                cached = accuracy if method != 'ours_s1' else {r['sample_id']: r for r in read(OUT / 'inference/ours_p262144_all/accuracy/results.json')['rows']}
                count = sum(v[0]['point'] == (cached[sid]['seed_points'][0] if method == 'ours_s1' else cached[sid]['point']) for sid, v in grouped.items())
                parity[name]['authenticated_cache_point_agreement'] = count
                assert count == 60, ('Reuse consistency requires investigation', name, count)
        if stage == 'all':
            robustness = read(folder / 'robustness/results.json')
            assert len(robustness['rows']) == 120
            assert {r['sample_id'] for r in robustness['clean_rows']} == selected
            for condition in ['jpeg70', 'border24']:
                rows = [r for r in robustness['rows'] if r['condition'] == condition]
                assert len(rows) == 60 and {r['sample_id'] for r in rows} == selected
        verified.append(name)
    frozen = read(OUT / 'data_efficiency/models/frozen.json')
    assert frozen['trained_fits'] == 36 and len(frozen['models']) == 48
    checked = set()
    for model in frozen['models'].values():
        if model['path'] not in checked:
            assert sha(model['path']) == model['sha256']
            checked.add(model['path'])
    assert len(checked) == 42
    return dict(complete=True, verified_jobs=verified, resource_limited_jobs=list(unavailable),
                new_fits=36, model_slots=48, distinct_checkpoints=42, screenshot_ids=300,
                paired_timing_screenshots=60, prediction_parity=parity,
                input_hash_note='Measurement input_image_sha256 identifies the original source file. Perturbations are determined by the frozen transform and condition.',
                no_new_training_inference_or_model_selection=True)


def figures(result):
    os.environ.setdefault('MPLCONFIGDIR', str(OUT / 'cache/matplotlib_final'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11})
    refs = ['tag', 'aima', 'mvp']
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.7), layout='constrained')
    for i, ref in enumerate(refs):
        row = result['accuracy_comparisons'][ref]
        value, bounds = row['estimate'] * 100, np.array(row['adjusted_ci']) * 100
        axes[0].errorbar(value, i, xerr=[[value - bounds[0]], [bounds[1] - value]], fmt='o', capsize=4,
                         color='#b44444' if value < 0 else '#2369a0')
        row = result['paired_speed'][ref]
        value, bounds = row['estimate'], row['adjusted_ci']
        axes[1].errorbar(value, i, xerr=[[value - bounds[0]], [bounds[1] - value]], fmt='o', capsize=4, color='#b44444')
    for ax in axes:
        ax.set_yticks(range(3), [NAMES[r] for r in refs]); ax.invert_yaxis(); ax.grid(axis='x', alpha=.2)
    axes[0].axvline(0, color='gray', lw=1); axes[0].set_xlabel('Ours minus reference (percentage points)')
    axes[0].set_title('Accuracy: 300 reused screenshots')
    axes[1].axvline(1, color='gray', lw=1); axes[1].set_xlabel('Mean latency ratio: ours / reference')
    axes[1].set_title('Latency: 60 paired screenshots, 2 repetitions')
    fig.suptitle('Controlled per-view image cap: 262,144 pixels\nOurs: 3-seed mean accuracy / S1 latency; 98.33% intervals; total computation differs', fontsize=12)
    fig.savefig(ANALYSIS / 'primary_comparisons.png', dpi=200)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharey=True, layout='constrained')
    for ax, budget in zip(axes, ['fixed_updates', 'nominal8']):
        curve = result['data_efficiency']['reused_independent'][budget]['curve']
        x = [r['queries'] for r in curve]
        ax.plot(x, [r['mean_accuracy_percent'] for r in curve], 'o-', color='#2369a0', label='Aligned evidence')
        ax.plot(x, [r['no_evidence_accuracy_percent'] for r in curve], 'o--', color='#777777', label='No-evidence control')
        ax.axhline(result['accuracy']['mvp_p262144_all']['accuracy_percent'], color='#a45632', ls=':', label='MVP (no project adaptation)')
        ax.set_xticks(x); ax.grid(alpha=.2); ax.set_xlabel('Readout training queries'); ax.set_title('2,976 updates' if budget == 'fixed_updates' else '8 nominal data passes')
        ax.legend(fontsize=9)
    axes[0].set_ylabel('Top-1 hit rate (%) on 300 reused screenshots')
    fig.suptitle('Readout-only data efficiency; proposer already trained on 873 labels\nTotal supervision union at these points: 1,093 / 1,396 / 1,932 / 2,973', fontsize=12)
    fig.savefig(ANALYSIS / 'readout_data_efficiency.png', dpi=200)
    plt.close(fig)


def interpretation(result, verification):
    primary = {k: v for k, v in result['accuracy'].items() if k.endswith('_all') and v['n'] == 300}
    rank = sorted(primary, key=lambda k: primary[k]['accuracy_percent'], reverse=True).index('ours_p262144_all') + 1
    lines = ['V17结果解读（探索性复用测试；没有更换主模型）',
             f'当前固定版本在{len(primary)}种相同单视图像素上限的实测配置中排名第{rank}。',
             '该设置没有统一视图数、总计算量、骨干模型或既有训练数据；排名仅限本轮配置。', '', '准确率与延时：']
    lines.append('主对比为TAG、GUI-AIMA、MVP，采用3项调整区间；其他对比为补充描述。')
    for ref in ['tag', 'aima', 'mvp', 'qwen', 'ui_tars', 'ours_gui']:
        quality = result['accuracy_comparisons'][ref]; timing = result['paired_speed'][ref]
        bounds = ', '.join(f'{100*x:+.2f}' for x in quality['adjusted_ci'])
        lines.append(f'相对{NAMES[ref]}：命中率差{100*quality["estimate"]:+.2f}百分点，调整区间[{bounds}]；耗时比{timing["estimate"]:.3f}（大于1表示本方法更慢）。')
    lines.extend(['', '解释边界：受限配置下的准确率收益不能外推为相同总计算量或论文默认预算下的全面优势。',
                  '相同骨干MVP的结果必须保留；小读出参数量不代表整个8B多视图流程的推理成本低。', '', '数据效率：'])
    for budget, data in result['data_efficiency']['reused_independent'].items():
        lines.append(budget + ': ' + '; '.join(f'{r["queries"]}条={r["mean_accuracy_percent"]:.2f}%' for r in data['curve']))
    lines.append('这些是冻结873条监督上游之后的读出学习曲线；不是相对无需新增定位训练的TAG/MVP更省数据的证明。')
    lines.extend(['', '鲁棒性（同时考察扰动准确率和相对干净条件的变化）：', json.dumps({ref: r['robustness'] for ref, r in result['claims'].items()}, ensure_ascii=False),
                  '未建立优势不等于证明二者完全相同；60张截图的区间较宽。', '', '高预算参考（同一60张；与300张表分开）：'])
    for name, row in result['accuracy'].items():
        if name.endswith('_stronger'):
            lines.append(f'{name}: {row["accuracy_percent"]:.2f}%')
    lines.append('资源不足而未赋准确率的配置：' + (', '.join(verification['resource_limited_jobs']) or '无'))
    diagnostic = read(ANALYSIS / 'joint_proposal_diagnostic.json')
    assert sum(row['n'] for row in diagnostic['groups'].values()) == 300
    assert np.isclose(sum(row['ours_mean_hits'] for row in diagnostic['groups'].values()), 3 * result['accuracy']['ours_p262144_all']['accuracy_percent'])
    assert np.isclose(sum(row['mvp_hits'] for row in diagnostic['groups'].values()), 3 * result['accuracy']['mvp_p262144_all']['accuracy_percent'])
    lines.extend(['', '事后诊断（仅解释固定预测，没有新推理或策略选择）：', json.dumps(diagnostic, ensure_ascii=False, indent=2),
                  '双方候选总体覆盖接近。共同可达子集中的差距说明候选组织与最终选择也需要检查，不能仅把总差距归因于候选覆盖。',
                  '该分组诊断并不单独识别评分器的因果效应，因为两方候选内容与数量不同。',
                  '', '可保留的是冻结骨干下的小读出适配与内部空间证据增益；外部整体优越性、速度优势和通用鲁棒性需要按实测结果限定。'])
    return '\n'.join(lines) + '\n'


def main():
    verification = audit()
    result = read(ANALYSIS / 'final_evidence.json')
    figures(result)
    (ANALYSIS / 'interpretation.txt').write_text(interpretation(result, verification), encoding='utf-8')
    files = [ANALYSIS / name for name in ['final_evidence.json', 'results.txt', 'interpretation.txt',
             'coverage_diagnostic.json', 'joint_proposal_diagnostic.json', 'primary_comparisons.png', 'readout_data_efficiency.png']]
    verification['analysis_artifact_sha256'] = {str(path): sha(path) for path in files}
    verification['finalizer_sha256'] = sha(__file__)
    (ANALYSIS / 'verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(verification, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
