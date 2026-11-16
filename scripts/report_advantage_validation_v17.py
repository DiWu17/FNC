"""Read-only synthesis of V17 evidence; display files are regenerated safely."""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/advantage_validation_v17'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def interval(a, b, source, family=1, ratio=False):
    a, b, source = np.asarray(a, float), np.asarray(b, float), np.asarray(source)
    rng = np.random.default_rng(20261722)
    draws = np.concatenate([rng.choice(np.flatnonzero(source == s),
        size=(2000, int((source == s).sum())), replace=True) for s in sorted(set(source))], axis=1)
    samples = a[draws].mean(1) / b[draws].mean(1) if ratio else (a - b)[draws].mean(1)
    alpha = .05 / family
    return dict(n=len(a), estimate=float(a.mean() / b.mean()) if ratio else float((a - b).mean()),
                ci95=np.quantile(samples, [.025, .975]).tolist(),
                adjusted_ci=np.quantile(samples, [alpha / 2, 1 - alpha / 2]).tolist(), family=family)


def collect():
    results, states = {}, {}
    for folder in (OUT / 'inference').glob('*'):
        if not folder.is_dir():
            continue
        if (folder / 'status.json').exists():
            states[folder.name] = read(folder / 'status.json')
        current = {}
        for endpoint in ['accuracy', 'speed', 'robustness']:
            path = folder / endpoint / 'results.json'
            if path.exists():
                current[endpoint] = read(path)
        if current:
            results[folder.name] = current
    accuracy = {}
    for name, data in results.items():
        if 'accuracy' not in data:
            continue
        rows = data['accuracy']['rows']
        accuracy[name] = dict(n=len(rows), accuracy_percent=float(np.mean([r['hit'] for r in rows]) * 100),
            by_source_percent={s: float(np.mean([r['hit'] for r in rows if r['source'] == s]) * 100) for s in sorted({r['source'] for r in rows})})
    speed = {}
    speed_vectors = {}
    for name, data in results.items():
        if 'speed' not in data:
            continue
        rows = data['speed']['rows']
        grouped = defaultdict(list)
        for r in rows:
            grouped[r['sample_id']].append(r)
        assert len(grouped) == 60 and all(len(v) == 2 for v in grouped.values())
        values = [r['measured']['elapsed_seconds'] for r in rows]
        speed[name] = dict(unique_screenshots=60, timed_calls=len(rows),
            p50_seconds=float(np.median(values)), p95_seconds=float(np.quantile(values, .95)),
            mean_seconds=float(np.mean(values)),
            timing_cohort_accuracy_percent=float(np.mean([r['hit'] for r in rows]) * 100),
            per_gpu_peak_allocated_bytes=[max(r['measured']['peak_allocated_bytes'][d] for r in rows) for d in range(2)])
        speed_vectors[name] = {k: dict(seconds=float(np.mean([r['measured']['elapsed_seconds'] for r in v])),
             hit=float(np.mean([r['hit'] for r in v])), source=v[0]['source']) for k, v in grouped.items()}
    comparisons, stronger_comparisons = {}, {}
    full_key = 'ours_p262144_all'
    if full_key in results and 'accuracy' in results[full_key]:
        ours = {r['sample_id']: r for r in results[full_key]['accuracy']['rows']}
        for ref in ['tag', 'aima', 'mvp', 'mvp_single', 'qwen', 'ui_tars', 'ours_gui']:
            key = f'{ref}_p262144_all'
            if key not in results or 'accuracy' not in results[key]:
                continue
            baseline = {r['sample_id']: r for r in results[key]['accuracy']['rows']}
            ids = sorted(ours)
            assert set(ids) == set(baseline) and len(ids) == 300
            comparisons[ref] = interval([ours[i]['hit'] for i in ids], [baseline[i]['hit'] for i in ids],
                                        [ours[i]['source'] for i in ids], family=3 if ref in ['tag', 'aima', 'mvp'] else 1)
        for key, data in results.items():
            if not key.endswith('_stronger') or 'accuracy' not in data:
                continue
            baseline = {r['sample_id']: r for r in data['accuracy']['rows']}
            ids = sorted(baseline)
            assert len(ids) == 60 and set(ids) <= set(ours)
            stronger_comparisons[key] = interval([ours[i]['hit'] for i in ids], [baseline[i]['hit'] for i in ids],
                                                [ours[i]['source'] for i in ids], family=6)
    paired_speed = {}
    own_speed = speed_vectors.get('ours_s1_p262144_speed')
    if own_speed:
        for ref in ['tag', 'aima', 'mvp', 'mvp_single', 'qwen', 'ui_tars', 'ours_gui']:
            key = f'{ref}_p262144_all'
            if key not in speed_vectors:
                continue
            baseline = speed_vectors[key]
            ids = sorted(own_speed)
            assert set(ids) == set(baseline)
            value = interval([own_speed[i]['seconds'] for i in ids], [baseline[i]['seconds'] for i in ids],
                             [own_speed[i]['source'] for i in ids], family=3 if ref in ['tag', 'aima', 'mvp'] else 1, ratio=True)
            value['s1_accuracy_difference'] = interval([own_speed[i]['hit'] for i in ids], [baseline[i]['hit'] for i in ids], [own_speed[i]['source'] for i in ids], family=3 if ref in ['tag', 'aima', 'mvp'] else 1)
            paired_speed[ref] = value
    robustness, robust_comparisons = {}, {}
    for name, data in results.items():
        if 'robustness' not in data:
            continue
        endpoint = data['robustness']
        clean = {r['sample_id']: r for r in endpoint['clean_rows']}
        robustness[name] = {}
        for condition in ['jpeg70', 'border24']:
            rows = [r for r in endpoint['rows'] if r['condition'] == condition]
            assert len(rows) == 60
            new, old = np.asarray([r['hit'] for r in rows]), np.asarray([clean[r['sample_id']]['hit'] for r in rows])
            displacement = [float(np.linalg.norm(np.asarray(r['point']) - clean[r['sample_id']]['point']))
                            for r in rows if r['point'] is not None and clean[r['sample_id']]['point'] is not None]
            robustness[name][condition] = dict(clean_accuracy_percent=float(old.mean() * 100),
                perturbed_accuracy_percent=float(new.mean() * 100), change_pp=float((new - old).mean() * 100),
                valid_point_pairs=len(displacement), median_displacement_pixels=float(np.median(displacement)) if displacement else None,
                clean_correct_retention=float(sum(np.mean(np.asarray(r['seed_hits'], bool) & np.asarray(clean[r['sample_id']]['seed_hits'], bool)) for r in rows) / old.sum()) if old.sum() else None)
    own_robust = results.get(full_key, {}).get('robustness')
    if own_robust:
        oc = {r['sample_id']: r for r in own_robust['clean_rows']}
        for ref in ['tag', 'aima', 'mvp']:
            other = results.get(f'{ref}_p262144_all', {}).get('robustness')
            if not other:
                continue
            bc = {r['sample_id']: r for r in other['clean_rows']}
            for condition in ['jpeg70', 'border24']:
                a = {r['sample_id']: r for r in own_robust['rows'] if r['condition'] == condition}
                b = {r['sample_id']: r for r in other['rows'] if r['condition'] == condition}
                ids = sorted(a)
                assert set(ids) == set(b) == set(oc) == set(bc)
                source = [a[i]['source'] for i in ids]
                robust_comparisons[f'{ref}_{condition}'] = dict(
                    absolute_accuracy=interval([a[i]['hit'] for i in ids], [b[i]['hit'] for i in ids], source, family=12),
                    difference_of_changes=interval([a[i]['hit'] - oc[i]['hit'] for i in ids], [b[i]['hit'] - bc[i]['hit'] for i in ids], source, family=12))
    claims = {}
    for ref in ['tag', 'aima', 'mvp']:
        accuracy_ci = comparisons.get(ref, {}).get('adjusted_ci')
        speed_ci = paired_speed.get(ref, {}).get('adjusted_ci')
        claims[ref] = dict(accuracy='pending' if accuracy_ci is None else 'supported_in_controlled_scope' if accuracy_ci[0] > 0 else 'lower_than_reference' if accuracy_ci[1] < 0 else 'inconclusive',
            speed='pending' if speed_ci is None else 'faster' if speed_ci[1] < 1 else 'slower' if speed_ci[0] > 1 else 'inconclusive')
        paired = paired_speed.get(ref)
        claims[ref]['joint_timed_accuracy_and_speed'] = ('pending' if paired is None else
            'supported_on_timed60_S1' if paired['adjusted_ci'][1] < 1 and paired['s1_accuracy_difference']['adjusted_ci'][0] > 0 else 'not_established')
        claims[ref]['higher_budget_accuracy'] = {
            name: ('supported_on_fixed60' if value['adjusted_ci'][0] > 0 else
                   'lower_than_reference' if value['adjusted_ci'][1] < 0 else 'inconclusive')
            for name, value in stronger_comparisons.items()
            if name.startswith(ref + '_')
        }
        claims[ref]['robustness'] = {}
        for condition in ['jpeg70', 'border24']:
            c = robust_comparisons.get(f'{ref}_{condition}')
            claims[ref]['robustness'][condition] = ('pending' if c is None else
                'supported_in_controlled_scope' if c['absolute_accuracy']['adjusted_ci'][0] > 0 and c['difference_of_changes']['adjusted_ci'][0] > 0 else 'not_established')
    efficiency = {}
    curves_path = OUT / 'data_efficiency/results.json'
    if curves_path.exists():
        curves = read(curves_path)
        plan = read(OUT / 'plan.json')
        for cohort, data in curves['cohorts'].items():
            legacy_cohort = {'reused_independent': 'independent'}.get(cohort, cohort)
            legacy = read(ROOT / f'runs/contribution_v16/evaluation/{legacy_cohort}/results.json')
            assert data['n'] == legacy['n']
            efficiency[cohort] = {}
            for budget in ['fixed_updates', 'nominal8']:
                rows = []
                for subset in ['n010', 'n025', 'n050', 'n100']:
                    full = [data['methods'][f'{budget}_{subset}_aligned_s{i}'] for i in [1, 2, 3]]
                    no = [data['methods'][f'{budget}_{subset}_no_evidence_s{i}'] for i in [1, 2, 3]]
                    if subset == 'n100':
                        for i in [1, 2, 3]:
                            assert full[i-1]['hits'] == legacy['methods'][f'aligned_s{i}']['top1_hits']
                            assert no[i-1]['hits'] == legacy['methods'][f'no_evidence_s{i}']['top1_hits']
                    rows.append(dict(subset=subset, queries=full[0]['queries'],
                        upstream_plus_selected_queries=plan['training_subsets'][subset]['upstream873_plus_selected_new_queries'],
                        mean_accuracy_percent=float(np.mean([v['accuracy'] for v in full])*100),
                        no_evidence_accuracy_percent=float(np.mean([v['accuracy'] for v in no])*100)))
                target = .9 * rows[-1]['mean_accuracy_percent']
                efficiency[cohort][budget] = dict(curve=rows, target90_percent=target,
                    smallest_observed_queries_at90_percent=min(r['queries'] for r in rows if r['mean_accuracy_percent'] >= target),
                    interpretation='Descriptive threshold on reused cohorts; not proof of noninferiority or superiority to zero-label TAG')
    return dict(updated_at_utc=datetime.now(timezone.utc).isoformat(), states=states, accuracy=accuracy,
        accuracy_comparisons=comparisons, speed=speed, paired_speed=paired_speed,
        robustness=robustness, robustness_comparisons=robust_comparisons, claims=claims,
        stronger_reference_comparisons=stronger_comparisons, data_efficiency=efficiency,
        scope='Controlled per-view cap and reused screenshot cohorts. No general SOTA, publisher-default or entire-system data-efficiency claim.')


def text_summary(result):
    lines = ['ClickAgent V17 方法优势验证', '更新时间（UTC）：' + result['updated_at_utc'],
             '范围：复用300张截图；固定60张用于速度、扰动和更高预算参考。不是新的独立测试。',
             '主模型固定；本方法2973条标注含上游873条。TAG/MVP不额外训练定位模块；AIMA监督预算未匹配。',
             '', '统一每视图预算准确率（300张；各方法实际总计算量仍可能不同）']
    controlled = {name: row for name, row in result['accuracy'].items() if not name.endswith('_stronger')}
    for rank, (name, row) in enumerate(sorted(controlled.items(), key=lambda kv: kv[1]['accuracy_percent'], reverse=True), 1):
        source = ', '.join(f'{key}={value:.2f}%' for key, value in row['by_source_percent'].items())
        lines.append(f'{rank}. {name}: {row["accuracy_percent"]:.2f}% ({row["n"]}张); {source}')
    for ref, row in result['accuracy_comparisons'].items():
        interval = ', '.join(f'{100*x:+.2f}' for x in row['adjusted_ci'])
        lines.append(f'本方法减{ref}: {100*row["estimate"]:+.2f}百分点; 调整区间=[{interval}]; 比较族={row["family"]}')
    lines.extend(['', '端到端速度（60张各重复2次；图像为统计单位；本方法固定S1）'])
    for name, row in result['speed'].items():
        memory = ', '.join(f'{x/1024**3:.2f}' for x in row['per_gpu_peak_allocated_bytes'])
        lines.append(f'{name}: p50={row["p50_seconds"]:.3f}秒; p95={row["p95_seconds"]:.3f}秒; 均值={row["mean_seconds"]:.3f}秒; 命中率={row["timing_cohort_accuracy_percent"]:.2f}%; 各卡峰值分配GiB=[{memory}]')
    for ref, row in result['paired_speed'].items():
        bounds = ', '.join(f'{x:.3f}' for x in row['adjusted_ci'])
        accuracy = row['s1_accuracy_difference']
        accuracy_bounds = ', '.join(f'{100*x:+.2f}' for x in accuracy['adjusted_ci'])
        lines.append(f'本方法/{ref}耗时比={row["estimate"]:.3f}, 调整区间=[{bounds}]; S1准确率差={100*accuracy["estimate"]:+.2f}百分点, 调整区间=[{accuracy_bounds}]')
    lines.extend(['', '图像扰动鲁棒性（60张；均重新运行完整推理）'])
    for name, conditions in result['robustness'].items():
        for condition, row in conditions.items():
            retention = '无干净命中，无法定义' if row['clean_correct_retention'] is None else f'{100*row["clean_correct_retention"]:.2f}%'
            lines.append(f'{name} / {condition}: 干净={row["clean_accuracy_percent"]:.2f}%; 扰动={row["perturbed_accuracy_percent"]:.2f}%; 变化={row["change_pp"]:+.2f}百分点; 干净命中保留率={retention}')
    for name, comparisons in result['robustness_comparisons'].items():
        for metric, row in comparisons.items():
            bounds = ', '.join(f'{100*x:+.2f}' for x in row['adjusted_ci'])
            lines.append(f'{name} / {metric}: {100*row["estimate"]:+.2f}百分点; 调整区间=[{bounds}]')
    lines.extend(['', '更高预算参考（同一固定60张；与300张表分开比较）'])
    for name, row in sorted(result['accuracy'].items()):
        if name.endswith('_stronger'):
            lines.append(f'{name}: {row["accuracy_percent"]:.2f}% / {row["n"]}张')
            paired = result['stronger_reference_comparisons'].get(name)
            if paired:
                bounds = ', '.join(f'{100*x:+.2f}' for x in paired['adjusted_ci'])
                lines.append(f'  本方法减参考={100*paired["estimate"]:+.2f}百分点; 调整区间=[{bounds}]')
    for path in sorted((OUT / 'failures').glob('*.json')):
        failure = read(path)
        if failure.get('resource_limit'):
            lines.append('资源不足，未赋准确率：' + failure['job'])
    lines.extend(['', '读出数据效率（固定已用873条标注的上游；不是全系统从零训练）'])
    for cohort, budgets in result['data_efficiency'].items():
        for budget, data in budgets.items():
            lines.append(cohort + ' / ' + budget)
            for row in data['curve']:
                lines.append(f'  {row["queries"]}条读出查询 / 全系统监督并集{row["upstream_plus_selected_queries"]}条: 完整证据={row["mean_accuracy_percent"]:.2f}%; 无证据={row["no_evidence_accuracy_percent"]:.2f}%')
            lines.append(f'  达到全量准确率90%的最小已测读出数据量：{data["smallest_observed_queries_at90_percent"]}条（描述性阈值，不构成非劣效证明）')
    lines.extend(['', '按预定规则判定的主张', json.dumps(result['claims'], ensure_ascii=False, indent=2), '', result['scope']])
    return '\n'.join(lines) + '\n'


def analyze():
    result = collect()
    write(OUT / 'analysis/current_evidence.json', result)
    (OUT / 'analysis/results.txt').write_text(text_summary(result), encoding='utf-8')
    return result


if __name__ == '__main__':
    print(text_summary(analyze()))
