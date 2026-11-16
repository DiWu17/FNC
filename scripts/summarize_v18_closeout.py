"""Audit completed V18 artifacts and produce a plain-text scientific readout.

This retrospective summary neither fits models nor selects test policies.
Registered primary intervals are copied from the combined three-test analysis.
Any added within-method contrasts are explicitly exploratory and descriptive.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from clickagent.public_evaluation import point_in_geometry

FULL = ROOT / 'runs/full_scaling_v18'
FRESH = ROOT / 'runs/mvp_confirmation_v18'
JOINT = FRESH / 'coordinate_addendum'
SCALING = ROOT / 'runs/scaling_effects_v18'
OUT = FULL / 'analysis/closeout'
ORDER = ['original3k_union', 'matched3k_union', 'full_union',
         'original3k_geometry', 'full_geometry',
         'original3k_coordinate_center', 'full_coordinate_center', 'MVP']
LABELS = ['3k / union', '3k / longer training', 'Full / union',
          '3k / center + rescore', 'Full / center + rescore',
          '3k / coordinate only', 'Full / coordinate only', 'MVP']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def hit(point, case):
    return bool(point is not None and point_in_geometry(
        [point[i] / case['image_size'][i] for i in [0, 1]], case['target_geometry']))


def plot(methods, joint, secondary):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'savefig.dpi': 180})
    fig, axes = plt.subplots(1, 3, figsize=(18, 6.6),
                             gridspec_kw={'width_ratios': [1.3, 1, 1]}, layout='constrained')
    ax = axes[0]
    colors = ['#8796a5', '#bbc3cc', '#286c9c', '#8796a5', '#286c9c', '#8796a5', '#286c9c', '#ca7b26']
    for i, name in enumerate(ORDER):
        row = methods[name]
        ax.barh(i, row['accuracy_percent'], color=colors[i], height=.62)
        if name != 'MVP':
            ax.plot(row['seed_accuracy_percent'], [i] * 3, '|', color='black', markersize=9)
        ax.text(row['accuracy_percent'] + .7, i, f"{row['accuracy_percent']:.2f}", va='center', fontsize=9)
    ax.set(yticks=np.arange(8), yticklabels=LABELS, xlim=(0, 54),
           xlabel='Point accuracy (%)', title='Fresh 280: fixed policies')
    ax.invert_yaxis()
    ax.text(0, -.19, 'Our bars: mean of 3 fixed fits; ticks: individual fits.\nMVP: one fixed checkpoint. No ensemble.',
            transform=ax.transAxes, fontsize=9, va='top')
    ax = axes[1]
    primary = ['full_union', 'full_geometry', 'full_coordinate_center']
    names = ['Full / union', 'Full / center + rescore', 'Full / coordinate only']
    for i, name in enumerate(primary):
        row = joint['comparisons'][name]
        v = row['delta_pp']; lo, hi = row['ci_pp']
        ax.errorbar(v, i, xerr=[[v-lo], [hi-v]], fmt='o', color='#286c9c', capsize=4)
        ax.annotate(f'{v:+.2f}', (v, i), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9)
    ax.axvline(0, color='black', ls='--', lw=1)
    ax.set(yticks=range(3), yticklabels=names, ylim=(2.7, -.7), xlim=(-22, 2),
           xlabel='Accuracy difference vs MVP (pp)', title='Registered primary comparisons')
    ax.text(0, -.19, '98.333% paired source-stratified bootstrap CIs.\nThree-comparison Bonferroni family.',
            transform=ax.transAxes, fontsize=9, va='top')
    ax = axes[2]
    pairs = [('full_vs_matched3k', 'Full vs longer-trained 3k'),
             ('matched3k_vs_original3k', 'Longer-trained vs original 3k'),
             ('full_vs_original3k', 'Full vs original 3k')]
    for i, (name, _) in enumerate(pairs):
        row = secondary['cohorts']['fresh_280']['contrasts'][name]
        v = row['delta_pp']; lo, hi = row['ci95_pp']
        ax.errorbar(v, i, xerr=[[v-lo], [hi-v]], fmt='o', color='#527c42', capsize=4)
        ax.annotate(f'{v:+.2f}', (v, i), xytext=(0, 10), textcoords='offset points', ha='center', fontsize=9)
    ax.axvline(0, color='black', ls='--', lw=1)
    ax.set(yticks=range(3), yticklabels=[v for _, v in pairs], ylim=(2.7, -.7), xlim=(-12, 12),
           xlabel='Accuracy difference (pp)', title='Registered secondary scaling contrasts')
    ax.text(0, -.19, 'Descriptive 95% CIs; fixed-fit uncertainty only.\nFull data also changes source mixture.',
            transform=ax.transAxes, fontsize=9, va='top')
    fig.suptitle('V18 completed experiments | 140 ScreenSpot-Pro + 140 VenusBench-GD screenshots', fontsize=14)
    fig.savefig(OUT / 'fresh_comparison.png', bbox_inches='tight')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for folder in [FULL, FRESH, JOINT, SCALING]:
        assert read(folder / 'status.json')['complete'], str(folder)
        path = folder / 'completion.json'; receipt = read(path)
        assert receipt['complete']
        provenance[str(path)] = sha(path)
        for name, digest in receipt['artifact_sha256'].items():
            assert sha(name) == digest, name
            provenance[name] = digest
    assert read(JOINT / 'completion.json')['parent_completion_sha256'] == sha(FRESH / 'completion.json')
    for name, digest in read(SCALING / 'completion.json')['input_sha256'].items():
        assert sha(name) == digest, name
    frozen = read(FULL / 'models/frozen.json')['models']
    for name, model in frozen.items():
        assert sha(model['path']) == model['sha256'], name
    adopted = read(ROOT / 'CURRENT_RESEARCH.json')
    for model in adopted['models'].values():
        assert sha(model['path']) == model['sha256']
    assert adopted['run_id'] == 'quality_readout_v15_3k'
    cases = [json.loads(line) for line in (FRESH / 'data/heldout.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(cases) == len({c['pixel_sha256'] for c in cases}) == 280
    ours = {r['sample_id']: r for r in read(FRESH / 'predictions/ours.json')['rows']}
    mvp = {r['sample_id']: r for r in read(FRESH / 'predictions/mvp.json')['rows']}
    joint = read(JOINT / 'combined_results.json')
    added = {r['sample_id']: r for r in joint['coordinate_predictions']}
    ids = [c['sample_id'] for c in cases]
    assert set(ids) == set(ours) == set(mvp) == set(added)
    values = {name: [] for name in ORDER}
    verified_hits = 0
    freeze_hash = sha(FRESH / 'freeze.json')
    for case in cases:
        sid = case['sample_id']; a = ours[sid]; b = mvp[sid]
        for row in [a, b]:
            assert row['freeze_sha256'] == freeze_hash
            assert row['pixel_sha256'] == case['pixel_sha256']
            assert row['image_sha256'] == case['image_sha256']
            assert not row['ground_truth_in_model_inputs']
        for name, points in a['points'].items():
            hits = [hit(p, case) for p in points]
            assert hits == a['hits'][name], (sid, name)
            values[name].append(hits); verified_hits += len(hits)
        for name, points in added[sid]['points'].items():
            hits = [hit(p, case) for p in points]
            values[name].append(hits); verified_hits += len(hits)
        assert hit(b['point'], case) == b['hit'], sid
        values['MVP'].append([b['hit']]); verified_hits += 1
    values = {name: np.asarray(v, float).T for name, v in values.items()}
    source_names = sorted({c['dataset'] for c in cases})
    groups = {s: [i for i, c in enumerate(cases) if c['dataset'] == s] for s in source_names}
    assert list(map(len, groups.values())) == [140, 140]
    methods = {}
    for name, v in values.items():
        accuracy = 100 * float(v.mean())
        assert abs(accuracy - joint['methods'][name]['accuracy_percent']) < 1e-9
        methods[name] = dict(accuracy_percent=accuracy, seed_hits=v.sum(1).astype(int).tolist(),
                             seed_accuracy_percent=(100*v.mean(1)).tolist(),
                             by_source={s: 100*float(v[:, ix].mean()) for s, ix in groups.items()},
                             number_of_fits=1 if name == 'MVP' else 3, ensemble=False)
    rng = np.random.default_rng(20261105)
    draws = np.stack([np.concatenate([np.asarray(ix)[rng.integers(0, len(ix), len(ix))]
                                      for ix in groups.values()]) for _ in range(2000)])
    extra = {}
    for name in ['original3k_geometry', 'original3k_coordinate_center', 'full_geometry', 'full_coordinate_center']:
        reference = 'original3k_union' if name.startswith('original3k') else 'full_union'
        diff = (values[name] - values[reference]).mean(0)
        extra[name] = dict(reference=reference, delta_pp=100*float(diff.mean()),
                           ci95_pp=(100*np.quantile(diff[draws].mean(1), [.025, .975])).tolist(),
                           scope='Retrospective exploratory descriptive contrast; not a registered primary superiority test')
    scaling = read(SCALING / 'results.json')
    result = dict(complete=True, generated_at_utc=datetime.now(timezone.utc).isoformat(), n=280,
                  sources={s: len(ix) for s, ix in groups.items()}, methods=methods,
                  primary_MVP_comparisons=joint['comparisons'],
                  registered_secondary_scaling=scaling['cohorts'], exploratory_coordinate_contrasts=extra,
                  mainline_unchanged=True, no_automatic_adoption=True,
                  limits=['Fixed three-fit means; no seed selection or ensemble.',
                          'Intervals condition on the fitted models; not a training-seed population interval.',
                          'Same backbone and per-view pixel cap; total inference compute is not matched.',
                          'Fresh screenshot subset only; not a full benchmark or interaction execution test.',
                          'Full-data arm changes both amount and source mixture.',
                          'Dataset labels are not fully human-recertified; prior semantic review remains pending.'])
    write(OUT / 'results.json', result)
    lines = ['V18 完整收尾：全量训练、冻结评分器替代策略、280张独立截图与MVP比较',
             '所有训练、双方推理、坐标对照和预先登记的统计分析均已完成。',
             '原3k权重保持不变；以下准确率是固定三个种子的均值，不是集成，不挑选最佳种子。MVP只有一个固定检查点。',
             '', '独立截图：ScreenSpot-Pro 140 + VenusBench-GD 140，每张一个查询，280个独立RGB。',
             '方法 | 平均准确率% | 三种子命中数（MVP为单次） | Pro% | Venus%']
    for name in ORDER:
        row = methods[name]
        lines.append(f"{name} | {row['accuracy_percent']:.2f} | {row['seed_hits']} | "
                     f"{row['by_source']['ScreenSpot-Pro']:.2f} | {row['by_source']['VenusBench-GD']:.2f}")
    lines += ['', '三项主比较，相对MVP的98.333%配对区间（Bonferroni三比较族）：']
    for name in ['full_union', 'full_geometry', 'full_coordinate_center']:
        row = joint['comparisons'][name]; lo, hi = row['ci_pp']
        assert row['primary'] and abs(row['interval_level_percent'] - 98.33333333333333) < 1e-8
        lines.append(f"{name}: {row['delta_pp']:+.2f}百分点；[{lo:+.2f}, {hi:+.2f}]；{row['status']}")
    lines += ['', '预先登记的数据量/步数次要描述性比较（独立280张，95%区间）：']
    for name, row in scaling['cohorts']['fresh_280']['contrasts'].items():
        lo, hi = row['ci95_pp']
        lines.append(f"{name}: {row['delta_pp']:+.2f}百分点；[{lo:+.2f}, {hi:+.2f}]")
    lines += ['', '追加的事后描述性坐标归因（95%区间；不承担预注册主结论）：']
    for name, row in extra.items():
        lo, hi = row['ci95_pp']
        lines.append(f"{name} vs {row['reference']}: {row['delta_pp']:+.2f}百分点；[{lo:+.2f}, {hi:+.2f}]")
    lines += ['', '冻结评分器的其他开发试验（旧验证172/新开发300，不能与独立280混作一张排名表）：']
    old = read(ROOT / 'runs/frozen_alternatives_v18/evaluation/old_val.json')['methods']
    new = read(ROOT / 'runs/frozen_alternatives_v18/evaluation/new_development.json')['methods']
    for name in old:
        lines.append(f"{name}: {old[name]['mean_accuracy_percent']:.2f}% / {new[name]['mean_accuracy_percent']:.2f}%")
    lines += ['双坐标银行的固定最大logit规则：旧52.71%/新61.00%，未通过开发门槛，未加入独立确认。',
              '中心坐标在新开发集带来的收益，未在旧300张已使用基准上复现；旧结果只作诊断。',
              '', '解释与决定：',
              '1. 全量完整候选33.93%对原3k完整候选34.40%，尚未确认泛化提升。不能承诺扩量就改善。',
              '2. 同步数下全量优于长训练3k；长训练3k又低于原3k。训练更久并不能替代有效数据。',
              '3. 坐标规则出现小幅点估计改善；仅此仍未超过MVP。三项注册主比较均低于MVP。',
              '4. 本轮保留原主线，不根据独立测试挑种子、规则或自动替换评分器。',
              '5. 下一轮应把候选几何/覆盖与候选选择分开研究，保留细粒度候选，并增强局部视觉判别；这仍需新开发与独立验证。',
              '', '训练口径：manifest13373，实际有候选拟合12886，487条按原规则跳过；评测保留完整分母。',
              '同结构22050参数，三全量拟合+三同13376步3k对照；原3k为2976步。数据来源比例也变了，不是纯条数实验。',
              '区间对固定模型条件化；同8B骨干和单视图像素上限，不等于总计算量匹配。',
              '不能由本子集宣称完整榜单SOTA、速度优势、鲁棒性优势或执行任务成功率。',
              '数据标签尚未全面人工复核；此前语义抽检待完成。',
              '', '分析阶段运行包装兼容性错误已修复并归档；科学脚本、协议、模型及已完成预测未修改。']
    (OUT / 'readout.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    plot(methods, joint, scaling)
    write(OUT / 'verification.json', dict(complete=True, verified_at_utc=datetime.now(timezone.utc).isoformat(),
        completed_chains=4, authenticated_artifact_sha256=provenance, checkpoint_hashes_verified=len(frozen),
        adopted_checkpoints_unchanged=3, unique_fresh_RGB=280, recomputed_point_hit_decisions=verified_hits,
        combined_table_agreement=True, primary_family_size=3, primary_intervals_copied_without_change=True,
        no_new_model_inference=True, no_policy_or_checkpoint_selection=True,
        retrospective_summary_source_sha256=sha(__file__)))
    paths = [OUT / name for name in ['results.json', 'readout.txt', 'fresh_comparison.png', 'verification.json']]
    write(OUT / 'completion.json', dict(complete=True, completed_at_utc=datetime.now(timezone.utc).isoformat(),
                                       artifact_sha256={str(p): sha(p) for p in paths}))
    print(json.dumps(dict(complete=True, output=str(OUT), verified_point_hit_decisions=verified_hits,
                          accuracy_percent={n: methods[n]['accuracy_percent'] for n in ORDER}), ensure_ascii=False))


if __name__ == '__main__':
    main()
