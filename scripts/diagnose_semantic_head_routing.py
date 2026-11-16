"""Post-hoc, validation-only CPU diagnosis of the six fixed v7 point checkpoints.

This script never loads fresh-test features/predictions, changes a checkpoint,
or chooses a model. Its condition-off intervention zeros *only* the new router
query/phrase projections; the target-conditioned field and base remain intact.
"""
import argparse
from collections import Counter
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from clickagent.action_benchmark import sha256
from clickagent.semantic_head_experiment import load_readout
from run_target_token_experiment import key, read_jsonl
from semantic_head_data import checked_digest, load_input_record, make_batch


JOBS = tuple(f'{family}_s{seed}' for family in
             ('static_base', 'semantic_base', 'semantic_hard') for seed in (1, 2))
EPS = 1e-30


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def entropy(probabilities, axis=-1):
    p = np.asarray(probabilities, np.float64)
    return -(p * np.log(np.maximum(p, EPS))).sum(axis=axis)


def cosine(a, b):
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    divisor = np.linalg.norm(a, axis=-1) * np.linalg.norm(b, axis=-1)
    return np.divide((a*b).sum(-1), divisor, out=np.zeros_like(divisor), where=divisor > 1e-20)


def js_divergence(a, b):
    """Pairwise Jensen-Shannon divergence, normalized by log(2), in [0,1]."""
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return np.maximum(0, entropy((a+b)*.5) - (entropy(a)+entropy(b))*.5) / np.log(2)


def describe(values):
    a = np.asarray(values, np.float64).reshape(-1)
    require(bool(np.isfinite(a).all()), 'Nonfinite diagnostic statistic')
    if not len(a):
        return dict(n=0, mean=None, std=None, p05=None, p50=None, p95=None, min=None, max=None)
    return dict(n=len(a), mean=float(a.mean()), std=float(a.std()),
                p05=float(np.quantile(a, .05)), p50=float(np.quantile(a, .5)),
                p95=float(np.quantile(a, .95)), min=float(a.min()), max=float(a.max()))


def softmax(scores):
    value = np.asarray(scores, np.float64)
    value = np.exp(value-value.max())
    return value/value.sum()


def group_pair_matrices(probabilities):
    """Input [...,groups,distribution]; retain each observation's 4x4 values."""
    groups = probabilities.shape[-2]
    similarities, divergences = [], []
    for first in range(groups):
        similarities.append(np.stack([cosine(probabilities[..., first, :], probabilities[..., second, :])
                                      for second in range(groups)], -1))
        divergences.append(np.stack([js_divergence(probabilities[..., first, :], probabilities[..., second, :])
                                     for second in range(groups)], -1))
    return np.stack(similarities, -2), np.stack(divergences, -2)


def routing_observations(head_weights, token_weights, candidates, tokens):
    """Return raw scalar observations; the caller first averages within a case.

    Every candidate-token and all 96 head identities remain in these diagnostics.
    Candidate diversity compares candidates at the *same token and group*.
    """
    weights = np.asarray(head_weights, np.float64)
    token = np.asarray(token_weights, np.float64)
    require(weights.ndim == 3 and weights.shape[:2] == (candidates*tokens, 96),
            'Head weights must retain every candidate/token/head')
    groups = weights.shape[-1]
    require(token.shape == (candidates*tokens, groups), 'Misaligned token routing')
    require(bool(np.isfinite(weights).all()) and bool((weights >= 0).all()), 'Invalid head probabilities')
    require(bool(np.isfinite(token).all()) and bool((token >= 0).all()), 'Invalid token probabilities')
    np.testing.assert_allclose(weights.sum(1), 1., atol=2e-6)
    p = weights.reshape(candidates, tokens, 96, groups).transpose(0, 1, 3, 2)
    token_p = token.reshape(candidates, tokens, groups).transpose(0, 2, 1)
    np.testing.assert_allclose(token_p.sum(-1), 1., atol=2e-6)
    head_entropy = entropy(p)
    head_cosine, head_js = group_pair_matrices(p)
    upper = np.triu_indices(groups, 1)
    # Generalized JS over candidates, with equal candidate mixture. Its maximum
    # is bounded by log(96); unlike a flattened diversity measure it cannot be
    # inflated by differences between tokens/cases.
    candidate_mean = p.mean(0, keepdims=True)
    candidate_js = np.maximum(0, entropy(candidate_mean[0])-head_entropy.mean(0))/np.log(96)
    observations = dict(
        head_entropy_normalized=(head_entropy/np.log(96)).ravel(),
        effective_heads=np.exp(head_entropy).ravel(),
        head_group_cosine=head_cosine[..., upper[0], upper[1]].ravel(),
        head_group_js_normalized=head_js[..., upper[0], upper[1]].ravel(),
        head_candidate_js_normalized=candidate_js.ravel(),
        head_candidate_cosine_to_mean=cosine(p, candidate_mean).ravel(),
        token_effective_count=np.exp(entropy(token_p)).ravel(),
    )
    if tokens > 1:
        token_cosine, token_js = group_pair_matrices(token_p)
        token_candidate_mean = token_p.mean(0, keepdims=True)
        observations.update(
            token_entropy_normalized=(entropy(token_p)/np.log(tokens)).ravel(),
            token_group_cosine=token_cosine[..., upper[0], upper[1]].ravel(),
            token_group_js_normalized=token_js[..., upper[0], upper[1]].ravel(),
            token_candidate_js_normalized=(np.maximum(0, entropy(token_candidate_mean[0])-entropy(token_p).mean(0))/np.log(tokens)).ravel(),
        )
    matrices = dict(head_group_cosine=head_cosine.mean((0, 1)),
                    head_group_js_normalized=head_js.mean((0, 1)),
                    head_weights=p.mean((0, 1)))
    return observations, matrices


def centered_ratio(values):
    values = np.asarray(values, np.float64)
    total = np.square(values).mean()
    centered = np.square(values-values.mean(0, keepdims=True)).mean()
    return float(np.sqrt(centered/max(total, 1e-30)))


def score_observations(base, residual, total, point_hits, region_hits, grouped, pooled):
    base, residual, total = (np.asarray(x, np.float64) for x in (base, residual, total))
    np.testing.assert_allclose(base+residual, total, rtol=3e-6, atol=3e-6)
    before, after = int(base.argmax()), int(total.argmax())
    base_std, residual_std = float(base.std()), float(residual.std())
    base_hit, rank_hit = bool(point_hits[before]), bool(point_hits[after])
    groups = grouped.shape[1]
    pairs = np.triu_indices(groups, 1)
    group_cos = np.stack([cosine(grouped[:, a], grouped[:, b]) for a, b in zip(*pairs)], -1)
    statistics = dict(
        base_score_std=base_std, residual_score_std=residual_std,
        residual_std_over_base_std=residual_std/base_std if base_std > 1e-12 else None,
        residual_abs_mean=float(np.abs(residual).mean()),
        residual_common_mean=float(residual.mean()),
        residual_range=float(np.ptp(residual)), base_range=float(np.ptp(base)),
        base_residual_correlation=float(np.corrcoef(base, residual)[0, 1]) if min(base_std, residual_std) > 1e-12 else None,
        score_distribution_js=float(js_divergence(softmax(base), softmax(total))),
        grouped_representation_group_cosine=float(group_cos.mean()),
        grouped_representation_candidate_centered_rms_ratio=centered_ratio(grouped.reshape(len(grouped), -1)),
        projected_representation_candidate_centered_rms_ratio=centered_ratio(pooled),
    )
    outcomes = dict(base_candidate_index=before, rank_candidate_index=after,
                    top1_changed=before != after, base_point_hit=base_hit, rank_point_hit=rank_hit,
                    point_rescued=not base_hit and rank_hit, point_lost=base_hit and not rank_hit,
                    base_parent_hit=bool(region_hits[before]), rank_parent_hit=bool(region_hits[after]))
    return statistics, outcomes


@contextmanager
def zero_new_query_condition(core):
    handles = [module.register_forward_hook(lambda _m, _a, output: torch.zeros_like(output))
               for module in (core.route_query, core.route_phrase)]
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()


def select_validation(out, limit):
    """Read only validation caches, sorted before testing target availability."""
    protocol = read(out/'protocol.json')
    selected, skipped = [], []
    cases = sorted((c for c in read_jsonl(out/'manifest.jsonl') if c['experiment_split'] == 'val'),
                   key=lambda c: c['sample_id'])
    for case in cases:
        record = load_input_record(out, case)
        if not record['metadata']['target_available']:
            skipped.append(case['sample_id'])
            continue
        point_path = out/'point_cache'/f'{key(case["sample_id"])}.npz'
        with np.load(point_path, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata']))
            require(meta['sample_id'] == case['sample_id'], 'Point-cache sample mismatch')
            require(meta['manifest_sha256'] == protocol['manifest_sha256'], 'Point-cache manifest mismatch')
            require(meta['fine_sha256'] == protocol['fixed_fine_sha256'], 'Fine decoder mismatch')
            require(meta['input_identity'] == record['cache_identity'], 'Point-cache input mismatch')
            np.testing.assert_array_equal(z['parent_indices'], record['parent_indices'])
            for name in ('frozen_point_features', 'label_fixed_point_hit'):
                record[name] = z[name].copy()
        record['point_cache_sha256'] = checked_digest(point_path)
        selected.append(record)
        if len(selected) == limit:
            break
    require(len(selected) == limit, f'Need {limit} named validation records, got {len(selected)}')
    return selected, skipped


def aggregate_cases(rows):
    keys = sorted({name for row in rows for name in row['metrics']})
    case_equal = {name: describe([r['metrics'][name] for r in rows
                                if r['metrics'].get(name) is not None]) for name in keys}
    keys = sorted({name for row in rows for name in row['outcomes']
                   if isinstance(row['outcomes'][name], bool)})
    outcomes = {name: sum(r['outcomes'].get(name, False) for r in rows) for name in keys}
    matrices = {name: np.mean([row['matrices'][name] for row in rows], 0).tolist()
                for name in rows[0]['matrices']}
    return dict(case_equal_statistics=case_equal, outcomes=outcomes, case_equal_matrices=matrices)


@torch.inference_mode()
def diagnose_checkpoint(out, name, entry, records):
    path = out/'models'/entry['file']
    require(sha256(path) == entry['sha256'], f'Checkpoint identity mismatch: {name}')
    model, metadata = load_readout(path, 'cpu')
    require(metadata['epoch'] == entry['epoch'] and metadata['selector'] == 'point', 'Wrong checkpoint selector')
    require(model.core.semantic_config.groups == 4, 'Expected four head pools')
    model.eval()
    captures = {}
    def projection_hook(_module, args, result):
        captures['grouped'] = args[0].detach().cpu().numpy().reshape(-1, 4, model.core.semantic_config.head_width).copy()
        captures['projected'] = result.detach().cpu().numpy().copy()
    handle = model.core.field_projection.register_forward_hook(projection_hook)
    rows = []; started = time.perf_counter()
    try:
        for index, record in enumerate(records):
            inputs, labels = make_batch([record], 'cpu', labels=False)
            require(not labels, 'Labels must not enter diagnostic model inputs')
            require(not any(k.startswith('label') for k in inputs), 'Supervision leaked into inputs')
            original = model(return_routing=True, **inputs)
            grouped, projected = captures['grouped'], captures['projected']
            with zero_new_query_condition(model.core):
                counterfactual = model(return_routing=True, **inputs)
            n, t = len(record['parent_indices']), len(record['query_hidden'])
            head, token = original['head_routing_weights'].numpy(), original['group_token_weights'].numpy()
            zero_head = counterfactual['head_routing_weights'].numpy()
            zero_token = counterfactual['group_token_weights'].numpy()
            obs, matrices = routing_observations(head, token, n, t)
            metrics = {k: float(v.mean()) for k, v in obs.items()}
            metrics.update(head_fraction_entropy_above_099=float((obs['head_entropy_normalized'] >= .99).mean()),
                           group_fraction_cosine_above_099=float((obs['head_group_cosine'] >= .99).mean()),
                           group_fraction_js_below_0001=float((obs['head_group_js_normalized'] <= .001).mean()))
            scores, outcomes = score_observations(original['base_scores'].numpy(), original['field_residual'].numpy(),
                original['scores'].numpy(), record['label_fixed_point_hit'] > 0, record['label_area'].sum(-1) > 0,
                grouped, projected)
            metrics.update(scores)
            # Head distributions are over dimension 1; transpose to last axis.
            metrics.update(
                explicit_condition_head_js_normalized=float(js_divergence(head.transpose(0, 2, 1), zero_head.transpose(0, 2, 1)).mean()),
                explicit_condition_head_max_abs_change=float(np.abs(head-zero_head).max()),
                explicit_condition_token_js_normalized=float(js_divergence(token.reshape(n, t, 4).transpose(0, 2, 1), zero_token.reshape(n, t, 4).transpose(0, 2, 1)).mean()),
                explicit_condition_score_js=float(js_divergence(softmax(original['scores'].numpy()), softmax(counterfactual['scores'].numpy()))),
                explicit_condition_residual_rms_change=float(np.square(original['field_residual'].numpy()-counterfactual['field_residual'].numpy()).mean()**.5),
            )
            zero_ci = int(counterfactual['scores'].argmax())
            outcomes.update(condition_off_rank_point_hit=bool(record['label_fixed_point_hit'][zero_ci] > 0),
                            explicit_condition_top1_changed=zero_ci != outcomes['rank_candidate_index'])
            if model.core.semantic_config.routing == 'static':
                for field in ('scores', 'head_routing_weights', 'group_token_weights'):
                    require(torch.equal(original[field], counterfactual[field]), 'Static intervention changed outputs')
            rows.append(dict(sample_id=record['case']['sample_id'], dataset=record['case']['dataset'],
                candidates=n, tokens=t, metrics=metrics, outcomes=outcomes,
                matrices={k: v.tolist() for k, v in matrices.items()}))
            if (index+1) % 16 == 0:
                print(f'ROUTING {name} {index+1}/{len(records)} CPU', flush=True)
    finally:
        handle.remove()
    full = dict(training=entry['training'], validation=entry['validation'])
    gap = {}
    for split in ('training', 'validation'):
        m = full[split]; hits = m['modes']['rank']['hits']
        gap[split] = dict(n=m['n'], named=m['named'], hits=hits, accuracy=hits/m['n'],
            hit_given_fixed_point_oracle=hits/m['frozen_fine_point_oracle'],
            candidate_region_coverage=m['candidate_region_coverage']/m['n'],
            fixed_point_oracle=m['frozen_fine_point_oracle']/m['n'], rank_loss=m['rank_loss'])
    gap['accuracy_gap_pp'] = 100*(gap['training']['accuracy']-gap['validation']['accuracy'])
    gap['hit_given_oracle_gap_pp'] = 100*(gap['training']['hit_given_fixed_point_oracle']-gap['validation']['hit_given_fixed_point_oracle'])
    aggregate = aggregate_cases(rows)
    mean_heads = np.asarray(aggregate['case_equal_matrices']['head_weights'])
    aggregate['mean_layer_weights'] = mean_heads.reshape(4, 3, 32).sum(-1).tolist()
    aggregate['largest_mean_head_weights'] = [
        [dict(layer=(15, 28, 36)[int(h)//32], head_zero_based=int(h)%32, weight=float(group[h]))
         for h in np.argsort(-group, kind='stable')[:5]] for group in mean_heads]
    return dict(checkpoint=str(path), checkpoint_sha256=entry['sha256'], selected_epoch=entry['epoch'],
        job=entry['job'], selector='point', routing=model.core.semantic_config.routing,
        elapsed_cpu_seconds=time.perf_counter()-started, full_metrics=full, train_validation_gap=gap,
        summary=aggregate, cases=rows)


def metric(checkpoint, name):
    return checkpoint['summary']['case_equal_statistics'][name]['mean']


def write_notes(summary, path):
    lines = ['# v7 验证集逐头路由机制诊断', '',
        '这是封存检查点的事后机制分析，不用于改模型、阈值、训练或检查点选择。没有读取新测试特征或正式预测。', '',
        f"固定选择规则：验证集按 sample_id 排序后，保留前 {summary['selection']['count']} 条有效名称样本；同一批样本用于全部六个 point 检查点。",
        '所有统计先在每条样本内平均，再对样本等权；表中“±”是64条样本均值的标准差，不是训练种子或置信区间。', '',
        '## 头路由与四组的差异', '',
        '| 检查点 | epoch | head H/log96 | 有效头数 exp(H) | 组间 cosine | 组间 JS/log2 | 跨候选 JS/log96 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def fmt(cp, name, digits=4):
        stat = cp['summary']['case_equal_statistics'][name]
        return f"{stat['mean']:.{digits}f} ± {stat['std']:.{digits}f}" if stat['mean'] is not None else 'N/A'
    for name, cp in summary['checkpoints'].items():
        lines.append(f"| {name} | {cp['selected_epoch']} | {fmt(cp,'head_entropy_normalized')} | {fmt(cp,'effective_heads',2)} | {fmt(cp,'head_group_cosine')} | {fmt(cp,'head_group_js_normalized')} | {fmt(cp,'head_candidate_js_normalized',6)} |")
    lines += ['', '均匀96头分布的 H/log96=1、有效头数=96；完全相同四组的 cosine=1、JS=0。有效头数是分布弥散程度，不是“真正使用了几个头”的因果证明。组间统计在同一 candidate-token 内计算；跨候选统计只比较同一 token、同一组的候选分布，避免混入目标名称或样本差异。', '',
              '## token 池与候选表示', '',
              '| 检查点 | token H/logT (T>1) | token 组间 cosine | token 组间 JS/log2 | 池后组向量 cosine | 候选表示中心化 RMS/总 RMS |',
              '|---|---:|---:|---:|---:|---:|']
    for name, cp in summary['checkpoints'].items():
        lines.append(f"| {name} | {fmt(cp,'token_entropy_normalized')} | {fmt(cp,'token_group_cosine')} | {fmt(cp,'token_group_js_normalized')} | {fmt(cp,'grouped_representation_group_cosine')} | {fmt(cp,'projected_representation_candidate_centered_rms_ratio')} |")
    lines += ['', 'T=1 的样本不进入归一 token 熵和组间 token 差异；每个统计的有效样本数见 JSON。头权重相似仍可能池化出有用值；候选图像分支也可贡献区分，不能只凭熵判定模块失效。', '',
              '## 残差强度和实际排序作用', '',
              '| 检查点 | 残差std/base std | 残差绝对值均值 | score JS/log2 | top1改变 /64 | base命中→rank命中 | 救回 /损失 |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name, cp in summary['checkpoints'].items():
        o = cp['summary']['outcomes']
        lines.append(f"| {name} | {fmt(cp,'residual_std_over_base_std',3)} | {fmt(cp,'residual_abs_mean',3)} | {fmt(cp,'score_distribution_js')} | {o['top1_changed']} | {o['base_point_hit']}→{o['rank_point_hit']} | {o['point_rescued']} / {o['point_lost']} |")
    lines += ['', '共同常数偏置不改变候选排序，所以主要比较候选内标准差，同时单列绝对值。base 是同一已训练模型内部的 base_scores；它可能因共同训练适应残差，不能把这一事后去残差当成独立训练基线。', '',
              '## 只移除新路由的显式 query/phrase 条件', '',
              '| 检查点 | head JS/log2 | token JS/log2 | 残差 RMS 改变 | top1改变 /64 | 条件保留命中 /置零命中 |',
              '|---|---:|---:|---:|---:|---:|']
    for name, cp in summary['checkpoints'].items():
        o = cp['summary']['outcomes']
        lines.append(f"| {name} | {fmt(cp,'explicit_condition_head_js_normalized',6)} | {fmt(cp,'explicit_condition_token_js_normalized',6)} | {fmt(cp,'explicit_condition_residual_rms_change',4)} | {o['explicit_condition_top1_changed']} | {o['rank_point_hit']} / {o['condition_off_rank_point_hit']} |")
    lines += ['', '置零通过临时 forward hook 操作 route_query/route_phrase 的输出，随后移除；没有改任何权重。目标条件化 attention field、candidate image、原 base 语义分支保持不变。static 三类输出（score/head/token）逐元素全等是强制检查。这个干预衡量已训练模型对显式条件的依赖，不等于重新训练一个无条件模型，也不是公开测试的消融结果。', '',
              '## 全量训练/验证差距（来自同一封存检查点）', '',
              '| 检查点 | train hit/n | val hit/n | train-val pp | train hit/oracle | val hit/oracle | rank loss train /val |',
              '|---|---:|---:|---:|---:|---:|---:|']
    for name, cp in summary['checkpoints'].items():
        g = cp['train_validation_gap']; a, b = g['training'], g['validation']
        lines.append(f"| {name} | {a['hits']}/{a['n']} ({100*a['accuracy']:.2f}%) | {b['hits']}/{b['n']} ({100*b['accuracy']:.2f}%) | {g['accuracy_gap_pp']:.2f} | {100*a['hit_given_fixed_point_oracle']:.2f}% | {100*b['hit_given_fixed_point_oracle']:.2f}% | {a['rank_loss']:.3f} / {b['rank_loss']:.3f} |")
    lines += ['', '训练表按该模型实际训练池的全量记录计算：base618、expanded873，并非每epoch600抽样的那一批。验证172含无名称失败。fixed-point oracle 的训练/验证比例不同，因此同时报告 hit/oracle；这只是本轮缓存+固定fine+候选集合的上限，不是VLM上限。较大的训练/验证差距支持泛化不足的解释，但不能单独区分过拟合、分布偏移或候选表达差异。', '',
              '## 观察范围与限制', '',
              '- 64条按名称排序的验证子集不是随机抽样；来源、token数、候选数和逐条统计完整保存在JSON。',
              '- 这是结果已封存后的描述性分析；没有显著性检验、未为“均匀/塌缩”设结论阈值。JSON中的0.99等比例只是固定参考描述。',
              '- 四组的相似性不能证明存在唯一有效头；反之非均匀权重也不能证明定位正确。',
              '- 所有标签只用于事后统计，不进入 make_batch(labels=False) 的模型输入。点命中引用既有固定fine标签，没有改变实际预测。',
              '- 各检查点选择epoch不同，hard训练池和目标函数也不同；六模型的表示差异不是单因素因果试验。',
              '- 不读取 fresh test 或旧192正式预测；不更新报告脚本，不改封存源/模型/manifest/特征/选择或预测。', '',
              '来源SHA、实际sample IDs、逐候选-token聚合口径、4×4矩阵、各组平均96头权重及每组前三层权重见 summary.json。']
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/semantic_head_v7'))
    parser.add_argument('--limit', type=int, default=64)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    require(args.limit == 64, 'Prespecified diagnostic fixes the first 64 named validation samples')
    require(args.threads in (1, 2, 4), 'Use 1, 2 or 4 CPU threads')
    torch.set_num_threads(args.threads)
    out = args.output
    protocol, frozen = read(out/'protocol.json'), read(out/'frozen_readouts.json')
    require(sha256(out/'manifest.jsonl') == protocol['manifest_sha256'] == frozen['manifest_sha256'], 'Manifest identity mismatch')
    for source, digest in frozen['sources'].items():
        require(sha256(Path(source)) == digest, f'Sealed source changed: {source}')
    require(all(name+'__point' in frozen['models'] for name in JOBS), 'Missing prespecified point checkpoints')
    records, skipped = select_validation(out, args.limit)
    summary = dict(schema_version=1, scope='validation_only_post_hoc_mechanism_diagnostic',
        source=dict(script_sha256=sha256(Path(__file__)), protocol_sha256=sha256(out/'protocol.json'),
                    frozen_sha256=sha256(out/'frozen_readouts.json'), manifest_sha256=protocol['manifest_sha256'],
                    sealed_sources=frozen['sources']),
        selection=dict(rule='sort validation sample_id ascending, take first64 target_available=True',
                       count=len(records), sample_ids=[r['case']['sample_id'] for r in records],
                       skipped_unavailable=skipped, datasets=dict(Counter(r['case']['dataset'] for r in records)),
                       token_lengths=describe([len(r['query_hidden']) for r in records]),
                       candidate_lengths=describe([len(r['parent_indices']) for r in records])),
        provenance=dict(cpu_only=True, threads=args.threads, fresh_features_or_predictions_read=False,
                        any_predictions_read=False, labels_in_forward=False,
                        model_or_selection_changed=False, checkpoint_selector='point',
                        aggregation='within case means followed by equally weighted case summaries; T=1 excluded only from normalized token measures',
                        head_order='layers15,28,36; within each layer heads0..31',
                        divergence_definitions=dict(pairwise_JS='JS/log(2)', candidate_JS='H(mean_candidate_distribution)-mean_candidate_H, divided by log(distribution size)',
                            effective_heads='exp(Shannon entropy), not a causal count',
                            condition_off='zero route_query/route_phrase outputs only; all other inputs/base fixed')),
        input_identities={r['case']['sample_id']:dict(input_identity=r['cache_identity'], point_cache_sha256=r['point_cache_sha256']) for r in records},
        checkpoints={})
    for job in JOBS:
        name = job+'__point'
        summary['checkpoints'][name] = diagnose_checkpoint(out, name, frozen['models'][name], records)
    # Verify sources/checkpoints again so diagnostics cannot conceal a concurrent change.
    for source, digest in frozen['sources'].items():
        require(sha256(Path(source)) == digest, f'Sealed source changed during diagnosis: {source}')
    for name, cp in summary['checkpoints'].items():
        require(sha256(Path(cp['checkpoint'])) == cp['checkpoint_sha256'], f'Checkpoint changed during diagnosis: {name}')
    dest = out/'evaluation'/'routing_diagnostic'; dest.mkdir(parents=True, exist_ok=True)
    (dest/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    write_notes(summary, dest/'notes.md')
    print('ROUTING COMPLETE '+str(dest), flush=True)


if __name__ == '__main__':
    main()
