"""Fixed, test-before-inspection secondary K16-to-Top4 compression diagnostic.

This is an addition during training, after training/validation progress had
already been inspected. It is NOT part of the original primary protocol.
No model is loaded. Formal analysis requires the complete two-cohort audit.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import time

import numpy as np

from clickagent.public_evaluation import point_in_geometry


OUT = Path('runs/candidate_scorer_v11')
ROOT = Path(__file__).resolve().parents[1]
HEADS = ('internal_full_s1', 'internal_full_s2', 'visual_s1', 'visual_s2')
SOURCE_K = 16
KEEP = 4
PRIMARY = 'internal_full_s1_K16_top4_gui'
N = 192
BOOTSTRAP_REPS = 5000
BOOTSTRAP_SEED = 20261131
CONSTANTS = dict(
    version=1, source_budget=SOURCE_K, retained_candidates=KEEP, expected_queries_per_cohort=N,
    heads=list(HEADS), primary=PRIMARY, secondary_heads=list(HEADS[1:]), cohorts=['test', 'diagnostic'],
    head_order='descending stored K16 head score, ties by original K16 pool order',
    gui_order='descending stored per-point GUI log-odds, ties by original K16 pool order, not head shortlist order',
    short_pool_policy='retain min(4, actual unique K16 points); empty pools fail',
    references=['gui_actor_K4', 'gui_actor_K16', 'each same head K16 Top1'],
    metric='inclusive point-in-any-region at fixed integer pixels; all192 denominator',
    bootstrap=dict(reps=BOOTSTRAP_REPS, seed=BOOTSTRAP_SEED, unit='RGB screenshot cluster', strata='dataset', interval='paired percentile 95%'),
    permitted_inference='none; reuse existing stored K16 head scores and GUI per-point scores',
    compute='actual retained point count as candidate-verification image-view proxy; no measured latency or speedup',
    actual_equal_view_count_slice='descriptive n/oracle/hits only, including empty cases; overall192 remains primary',
    model_or_budget_selection=False,
)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def write(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def key(sample_id): return hashlib.sha256(sample_id.encode()).hexdigest()[:24]


def dependencies():
    return {str(ROOT / 'src/clickagent/public_evaluation.py'): sha(ROOT / 'src/clickagent/public_evaluation.py')}


def preregister():
    """Record the addition now, without reading a test manifest/result/feature."""
    path = OUT / 'compression_preregistration.json'
    if path.exists():
        existing = read(path); validate_preregistration(existing)
        print('EXISTING COMPRESSION REGISTRATION VERIFIED', flush=True)
        return existing
    artifact_roots = [OUT / 'test', OUT / 'predictions/test', OUT / 'gui_reference/test', OUT / 'ui_tars',
                      OUT / 'cache/internal/test', OUT / 'cache/visual/test']
    files = [p for root in artifact_roots if root.exists() for p in root.rglob('*') if p.is_file()]
    assert not files and not (OUT / 'final_audit.json').exists(), 'Registration must precede independent test inference'
    stamp = time.time()
    record = dict(complete=True, classification='secondary addition during training, fixed before independent test',
        original_primary_protocol=False, training_validation_progress_previously_inspected=True,
        created_at_unix=stamp, created_at_utc=datetime.fromtimestamp(stamp, timezone.utc).isoformat(),
        constants=CONSTANTS, script=str(Path(__file__).resolve()), script_sha256=sha(Path(__file__)),
        dependency_sha256=dependencies(), observed_fresh_test_artifact_files=0,
        observed_existing_model_checkpoints=len(list((OUT / 'models').glob('*.pt'))),
        formal_run_requires='complete matching final_audit.json, frozen identities and all evaluation/input hashes before any test labels are loaded',
        change_policy='No rule/model/seed/K changes after this record. A changed script or constants is rejected.')
    write(path, record)
    document = '<!doctype html><meta charset="utf-8"><title>V11候选压缩补充登记</title><h1>训练进行中追加、测试前固定的次要诊断</h1>'
    document += '<p>此诊断提出前已查看训练/验证进度，不属于原始主协议。独立测试尚未提取，正式分析须在两组正式推理、评估及完整审计完成后执行。</p>'
    document += '<p>固定K16→Top4。internal_full_s1为主，internal_full_s2、visual_s1/s2为次；全部保留，不选模型、种子或预算。先按head分数保留4点，再按既有GUI分数选择。两阶段同分均按原K16池顺序。</p>'
    document += '<p>与GUI quota K4、GUI K16、相同head K16 Top1比较命中、实际点oracle与配对区间。192题全部保留，空池失败；旧诊断单列。数量仅作验证图像数代理，不是实测延迟。旧quota去重后可能不足4点，将明确记录实际数量差异。</p>'
    document += '<pre>' + html.escape(json.dumps(record, ensure_ascii=False, indent=2)) + '</pre>'
    (OUT / 'compression_preregistration.html').write_text(document, encoding='utf-8')
    print('COMPRESSION SECONDARY DIAGNOSTIC REGISTERED ' + str(path), flush=True)
    return record


def validate_preregistration(record):
    assert record['complete'] and record['original_primary_protocol'] is False
    assert record['training_validation_progress_previously_inspected'] is True
    assert record['constants'] == CONSTANTS and record['script_sha256'] == sha(Path(__file__))
    assert record['dependency_sha256'] == dependencies()
    assert math.isfinite(record['created_at_unix']) and record['observed_fresh_test_artifact_files'] == 0


def _normalized_mapping(mapping): return {str(Path(p).resolve()): digest for p, digest in mapping.items()}


def verified_evaluations():
    """Fail closed before any annotation or inference feature/result is loaded."""
    registration = read(OUT / 'compression_preregistration.json'); validate_preregistration(registration)
    audit = read(OUT / 'final_audit.json')
    assert audit['complete'] and set(audit['splits']) == {'test', 'diagnostic'}
    assert audit['frozen_sha256'] == sha(OUT / 'frozen.json') and audit['protocol_sha256'] == sha(OUT / 'protocol.json')
    frozen = read(OUT / 'frozen.json')
    assert frozen['complete'] and frozen['primary'] == 'internal_full_s1' and frozen['primary_budget'] == SOURCE_K
    assert frozen['fresh_test_features_at_freeze'] == 0
    sealed = _normalized_mapping(frozen['files'])
    for name in ('protocol.json', 'manifest.jsonl', 'data_selection.json', 'diagnostic_manifest.jsonl'):
        path = OUT / name
        assert sealed[str(path.resolve())] == sha(path)
    for head in HEADS:
        entry = frozen['models'][head]
        assert sealed[str(Path(entry['file']).resolve())] == entry['sha256'] == sha(entry['file'])
    source_evidence = _normalized_mapping(audit['source_sha256'])
    for name in ('evaluate_candidate_scorer.py', 'audit_candidate_scorer.py'):
        path = ROOT / 'scripts' / name
        assert source_evidence[str(path.resolve())] == sha(path)
    for path, digest in {**frozen['files'], **audit['source_sha256']}.items(): assert sha(path) == digest, path
    identities = dict(registration_sha256=sha(OUT / 'compression_preregistration.json'),
        final_audit_sha256=sha(OUT / 'final_audit.json'), frozen_sha256=sha(OUT / 'frozen.json'),
        protocol_sha256=sha(OUT / 'protocol.json'), script_sha256=sha(Path(__file__)), evaluation_sha256={}, input_sha256={})
    evidence = _normalized_mapping(audit['evaluation_sha256']); summaries, details = {}, {}
    for split in ('test', 'diagnostic'):
        folder = OUT / 'evaluation' / split
        for name in ('summary.json', 'details.json', 'predictions.jsonl'):
            path = folder / name
            assert evidence[str(path.resolve())] == sha(path), ('Audited evaluation changed', str(path))
            identities['evaluation_sha256'][str(path)] = sha(path)
        summary = read(folder / 'summary.json')
        assert summary['complete'] and summary['split'] == split and summary['n'] == N
        assert summary['primary'] == 'internal_full_s1_K16' and summary['selection_used_this_cohort'] is False
        assert summary['exploratory_reused_images'] == (split == 'diagnostic') and len(summary['methods']) == 38
        assert summary['frozen_sha256'] == identities['frozen_sha256'] and summary['protocol_sha256'] == identities['protocol_sha256']
        checked = audit['splits'][split]
        assert checked['n'] == N and checked['methods'] == 38 and checked['predictions_checked'] == N * 38
        assert checked['replay']['complete'] and checked['replay']['integer_points_exact']
        for path, digest in summary['input_sha256'].items():
            assert sha(path) == digest, ('Audited prediction/pool changed', path)
            assert identities['input_sha256'].setdefault(str(Path(path).resolve()), digest) == digest
        values = read(folder / 'details.json')
        assert len(values) == len({d['sample_id'] for d in values}) == N
        summaries[split] = summary; details[split] = {d['sample_id']: d for d in values}
    return summaries, details, identities


def hit(point, case):
    if point is None: return False
    if not isinstance(point, (list, tuple)) or len(point) != 2: return False
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or int(v) != v for v in point): return False
    w, h = case['image_size']
    return bool(0 <= point[0] < w and 0 <= point[1] < h and
                point_in_geometry([point[0] / w, point[1] / h], case['target_geometry']))


def finite_scores(values, n):
    values = np.asarray(values, np.float64)
    assert values.shape == (n,) and np.isfinite(values).all()
    return values


def shortlist_choice(head_scores, gui_scores):
    """No labels/points accepted: Top4 then GUI, both ties by original index."""
    head = finite_scores(head_scores, len(head_scores)); gui = finite_scores(gui_scores, len(head))
    retained = sorted(range(len(head)), key=lambda i: (-head[i], i))[:KEEP]
    winner = min(retained, key=lambda i: (-gui[i], i)) if retained else None
    return retained, winner


def _point_key(point): return ','.join(map(str, point))


def case_analysis(case, pipeline, prediction, gui, audited):
    from clickagent.candidate_budget import build_pools
    sid = case['sample_id']
    assert pipeline['sample_id'] == prediction['sample_id'] == gui['sample_id'] == audited['sample_id'] == sid
    assert pipeline['ground_truth_in_model_inputs'] is prediction['ground_truth_in_model_inputs'] is gui['ground_truth_in_model_inputs'] is False
    assert pipeline['pools'] == build_pools(pipeline)
    pool = pipeline['pools']['16']; points = [v['point'] for v in pool]
    assert len(points) <= SOURCE_K and len(set(map(tuple, points))) == len(points)
    quota = pipeline['pools']['4']; quota_points = [v['point'] for v in quota]
    assert {tuple(p) for p in quota_points} <= {tuple(p) for p in points}
    labels = [hit(p, case) for p in points]
    assert audited['pools']['16']['points'] == points and audited['pools']['16']['point_hits'] == labels
    assert audited['pools']['4']['points'] == quota_points
    assert audited['pools']['4']['point_hits'] == [hit(p, case) for p in quota_points]
    gui_scores = []
    for point in points:
        record = gui['records'][_point_key(point)]
        assert record['point'] == point
        gui_scores.append(record['log_odds'])
    gui_scores = finite_scores(gui_scores, len(points))
    methods, shortlists = {}, {}
    for budget in (4, 16):
        actual = pipeline['pools'][str(budget)]; selected = gui['choices'][str(budget)]
        values = finite_scores([gui['records'][_point_key(v['point'])]['log_odds'] for v in actual], len(actual))
        expected = actual[int(values.argmax())]['point'] if actual else None
        assert selected['point'] == expected and selected['score'] == (float(values.max()) if actual else None)
        name = 'gui_actor_K' + str(budget)
        assert audited['methods'][name]['point'] == expected and audited['methods'][name]['hit'] == hit(expected, case)
        methods[name] = dict(point=expected, hit=hit(expected, case), n_verification_views=len(actual),
            pool_oracle=any(hit(v['point'], case) for v in actual))
    for head in HEADS:
        output = prediction['methods'][head + '_K16']; scores = finite_scores(output['scores'], len(points))
        assert output['n_points'] == len(points)
        top1 = int(scores.argmax()) if len(points) else None
        expected = points[top1] if top1 is not None else None
        assert output['point'] == expected
        if top1 is not None: assert output['selected_index'] == top1
        assert audited['methods'][head + '_K16']['point'] == expected and audited['methods'][head + '_K16']['hit'] == hit(expected, case)
        assert audited['methods'][head + '_K16']['scores'] == output['scores']
        methods[head + '_K16'] = dict(point=expected, hit=hit(expected, case), pool_oracle=any(labels), n_verification_views=0)
        retained, winner = shortlist_choice(scores, gui_scores)
        selected_points = [points[i] for i in retained]
        chosen = points[winner] if winner is not None else None
        oracle = any(labels[i] for i in retained)
        assert not hit(expected, case) or oracle
        name = head + '_K16_top4_gui'
        methods[name] = dict(point=chosen, hit=hit(chosen, case), pool_oracle=oracle, n_verification_views=len(retained))
        shortlists[head] = dict(original_pool_indices=retained, points=selected_points,
            point_hits=[labels[i] for i in retained], point_oracle=oracle, selected_original_pool_index=winner,
            head_scores=[float(scores[i]) for i in retained], gui_scores=[float(gui_scores[i]) for i in retained],
            actual_verification_views=len(retained), quota4_views=len(quota_points))
    return dict(sample_id=sid, dataset=case['dataset'], source_k16_points=points, source_k16_point_hits=labels,
        source_k16_oracle=any(labels), quota4_points=quota_points, quota4_oracle=any(hit(p, case) for p in quota_points),
        shortlists=shortlists, methods=methods)


def bootstrap(matrix, cohort):
    rng = np.random.default_rng(BOOTSTRAP_SEED); matrix = np.asarray(matrix, np.float64)
    numerator = np.zeros((BOOTSTRAP_REPS, matrix.shape[1])); denominator = np.zeros(BOOTSTRAP_REPS)
    for dataset in sorted({c['dataset'] for c in cohort}):
        groups = {}
        for i, case in enumerate(cohort):
            if case['dataset'] == dataset:
                identity = case.get('pixel_sha256') or case.get('group_id') or case.get('image_sha256') or case['sample_id']
                groups.setdefault(identity, []).append(i)
        ordered = [groups[k] for k in sorted(groups)]
        hits = np.stack([matrix[ix].sum(0) for ix in ordered]); sizes = np.asarray([len(ix) for ix in ordered])
        draws = rng.integers(0, len(ordered), (BOOTSTRAP_REPS, len(ordered)))
        numerator += hits[draws].sum(1); denominator += sizes[draws].sum(1)
    return numerator / denominator[:, None]


def summarize(split, cohort, details):
    assert len(cohort) == len(details) == N
    assert [d['sample_id'] for d in details] == [c['sample_id'] for c in cohort]
    names = ['gui_actor_K4', 'gui_actor_K16'] + [h + '_K16' for h in HEADS] + [h + '_K16_top4_gui' for h in HEADS]
    oracle_names = ['quota4_oracle', 'source_k16_oracle'] + [h + '_top4_oracle' for h in HEADS]
    matrix = np.asarray([[d['methods'][m]['hit'] for m in names] +
        [d['quota4_oracle'], d['source_k16_oracle']] + [d['shortlists'][h]['point_oracle'] for h in HEADS] for d in details], bool)
    draws = bootstrap(matrix, cohort); methods = {}; source_views = sum(len(d['source_k16_points']) for d in details)
    for i, name in enumerate(names):
        rows = [d['methods'][name] for d in details]; hits = int(matrix[:, i].sum()); available = sum(r['pool_oracle'] for r in rows)
        views = [r['n_verification_views'] for r in rows]
        methods[name] = dict(n=N, hits=hits, accuracy=hits / N, ci95=np.quantile(draws[:, i], [.025, .975]).tolist(),
            point_oracle=available, point_oracle_rate=available / N, conditional_gui_or_head_hit=hits / available if available else None,
            verification_view_proxy=dict(total=sum(views), mean=float(np.mean(views)), maximum=max(views),
                relative_to_gui_k16=sum(views) / source_views if source_views else None),
            datasets={dataset: dict(n=sum(d['dataset'] == dataset for d in details),
                hits=sum(d['methods'][name]['hit'] for d in details if d['dataset'] == dataset),
                point_oracle=sum(d['methods'][name]['pool_oracle'] for d in details if d['dataset'] == dataset))
                for dataset in sorted({c['dataset'] for c in cohort})})
    def paired(a, b):
        i = (names + oracle_names).index(a); j = (names + oracle_names).index(b)
        difference = matrix[:, i].astype(int) - matrix[:, j].astype(int)
        return dict(method=a, reference=b, n=N, wins=int((difference > 0).sum()), losses=int((difference < 0).sum()),
            rescues=int((difference > 0).sum()), harms=int((difference < 0).sum()), ties=int((difference == 0).sum()),
            delta=float(difference.mean()), ci95=np.quantile(draws[:, i] - draws[:, j], [.025, .975]).tolist())
    hit_pairs = [paired(h + '_K16_top4_gui', r) for h in HEADS for r in ('gui_actor_K4', 'gui_actor_K16', h + '_K16')]
    oracle_pairs = [paired(h + '_top4_oracle', r) for h in HEADS for r in ('quota4_oracle', 'source_k16_oracle')]
    stages = {}
    for head in HEADS:
        name = head + '_K16_top4_gui'
        values = dict(hits=methods[name]['hits'], source_empty=sum(not d['source_k16_points'] for d in details),
            source_nonempty_without_correct=sum(bool(d['source_k16_points']) and not d['source_k16_oracle'] for d in details),
            compression_dropped_all_correct=sum(d['source_k16_oracle'] and not d['shortlists'][head]['point_oracle'] for d in details),
            gui_rerank_missed_correct=sum(d['shortlists'][head]['point_oracle'] and not d['methods'][name]['hit'] for d in details))
        assert sum(values.values()) == N
        stages[head] = dict(**values, equal_actual_views_to_quota4=sum(d['shortlists'][head]['actual_verification_views'] == len(d['quota4_points']) for d in details),
            more_actual_views_than_quota4=sum(d['shortlists'][head]['actual_verification_views'] > len(d['quota4_points']) for d in details),
            candidate_size_counts=dict(Counter(d['shortlists'][head]['actual_verification_views'] for d in details)))
        same = [d for d in details if d['shortlists'][head]['actual_verification_views'] == len(d['quota4_points'])]
        stages[head]['equal_actual_views_descriptive'] = dict(n=len(same), empty=sum(not d['source_k16_points'] for d in same),
            quota4_oracle=sum(d['quota4_oracle'] for d in same), top4_oracle=sum(d['shortlists'][head]['point_oracle'] for d in same),
            quota4_hits=sum(d['methods']['gui_actor_K4']['hit'] for d in same), top4_gui_hits=sum(d['methods'][name]['hit'] for d in same))
        assert stages[head]['equal_actual_views_to_quota4'] + stages[head]['more_actual_views_than_quota4'] == N
    return dict(complete=True, split=split, n=N, primary=PRIMARY, exploratory_reused_images=split == 'diagnostic',
        classification='secondary addition during training, fixed before independent test; not original primary protocol',
        constants=CONSTANTS, methods=methods, paired_comparisons=hit_pairs, oracle_paired_comparisons=oracle_pairs,
        stages=stages, selection_used_this_cohort=False,
        warning='Counterfactual reuse of already-computed GUI per-point scores, not a new Top4 batch run. Actual view counts may differ from post-dedup quota K4. No measured time saving, seed selection or noninferiority claim.')


def render_report(summaries, identities):
    def table(headers, rows):
        return '<table><tr>' + ''.join('<th>' + html.escape(str(x)) + '</th>' for x in headers) + '</tr>' + ''.join('<tr>' + ''.join('<td>' + html.escape(str(x)) + '</td>' for x in row) + '</tr>' for row in rows) + '</table>'
    document = '<!doctype html><html lang="zh"><meta charset="utf-8"><title>固定K16→Top4候选压缩诊断</title><style>body{max-width:1250px;margin:32px auto;font:16px/1.7 system-ui}table{border-collapse:collapse;width:100%}td,th{padding:8px;border:1px solid #ddd;text-align:left}h2{margin-top:36px}p{overflow-wrap:anywhere}</style><h1>固定K16→Top4候选压缩诊断</h1>'
    document += '<p>本项在查看训练/验证进度后、独立测试提取前追加并固定，是次要诊断，不是原始主协议。internal_full_s1固定为主，其余三头均保留。它检查评分器压缩候选的效用，不改原始Top1主结果。</p>'
    document += '<p>先用原K16 head分数降序保留最多4个点，再用已存GUI分数选择。两阶段同分均按原K16池顺序，null不参与。没有新模型推理。旧quota K4先限额再去重，实际可能不足4点；数量差异明确单列。</p>'
    for split in ('test', 'diagnostic'):
        summary = summaries[split]
        document += '<h2>' + ('新独立192题' if split == 'test' else '旧诊断192题：已分析图像，仅次要探索') + '</h2>'
        document += table(['方法', '命中/192', '命中率95%区间', '点oracle/192', '有正确点时命中率', '验证图像数量代理'],
            [[m, f"{v['hits']}/192", f"{100*v['accuracy']:.1f}% [{100*v['ci95'][0]:.1f}, {100*v['ci95'][1]:.1f}]", v['point_oracle'],
              f"{100*v['conditional_gui_or_head_hit']:.1f}%" if v['conditional_gui_or_head_hit'] is not None else '—', v['verification_view_proxy']['total']]
             for m, v in summary['methods'].items()])
        document += table(['配对：方法−参考', '救回/损失/平', '差值pp', '95%区间pp'],
            [[p['method'] + ' − ' + p['reference'], f"{p['rescues']}/{p['harms']}/{p['ties']}", f"{100*p['delta']:.2f}",
              f"[{100*p['ci95'][0]:.2f}, {100*p['ci95'][1]:.2f}]"] for p in summary['paired_comparisons']])
        document += table(['oracle配对：Top4−参考', '增加/丢失/平', '差值pp', '95%区间pp'],
            [[p['method'] + ' − ' + p['reference'], f"{p['wins']}/{p['losses']}/{p['ties']}", f"{100*p['delta']:.2f}",
              f"[{100*p['ci95'][0]:.2f}, {100*p['ci95'][1]:.2f}]"] for p in summary['oracle_paired_comparisons']])
        document += table(['head', '空源池', '源池无正确点', '压缩丢掉全部正确点', '保留正确点但GUI选错', '实际点数等于quota4的题数', '实际点数更多的题数'],
            [[head, v['source_empty'], v['source_nonempty_without_correct'], v['compression_dropped_all_correct'], v['gui_rerank_missed_correct'],
              v['equal_actual_views_to_quota4'], v['more_actual_views_than_quota4']] for head, v in summary['stages'].items()])
        document += table(['实际同数量切片（仅描述）', '题数/其中空池', 'quota4 oracle / Top4 oracle', 'quota4命中 / Top4+GUI命中'],
            [[head, f"{v['equal_actual_views_descriptive']['n']} / {v['equal_actual_views_descriptive']['empty']}",
              f"{v['equal_actual_views_descriptive']['quota4_oracle']} / {v['equal_actual_views_descriptive']['top4_oracle']}",
              f"{v['equal_actual_views_descriptive']['quota4_hits']} / {v['equal_actual_views_descriptive']['top4_gui_hits']}"] for head, v in summary['stages'].items()])
    document += '<p>区间按数据集分层、截图聚类，配对bootstrap5000次；多项次要比较未校正多重性，不据此选择模型、种子或K。192题均计入，空池失败。静态正例几何命中不等于点击许可或真实执行成功。</p>'
    document += '<p>数量只是候选验证图像数代理。GUI K16分数已经全部计算，本分析没有节省已发生的计算；没有重跑Top4 batch，batch数值差异亦未测量。新增评分器与特征提取成本未计入该代理，不能宣称端到端提速。</p>'
    document += '<p>登记SHA256：' + identities['registration_sha256'] + '<br>主审计SHA256：' + identities['final_audit_sha256'] + '</p></html>'
    return document


def analyze():
    summaries, audited_details, identities = verified_evaluations()
    # Only after the complete matching primary audit may annotations be read.
    rows = read_rows(OUT / 'manifest.jsonl')
    cohorts = {'test': [c for c in rows if c['experiment_split'] == 'test'],
               'diagnostic': read_rows(OUT / 'diagnostic_manifest.jsonl')}
    result = {}; destination = OUT / 'evaluation/compression'
    for split in ('test', 'diagnostic'):
        cohort = cohorts[split]
        assert len(cohort) == len({c['sample_id'] for c in cohort}) == N
        assert {c['sample_id'] for c in cohort} == set(audited_details[split])
        if split == 'diagnostic': assert all(c['diagnostic_reuse'] for c in cohort)
        details = []
        for case in cohort:
            stem = key(case['sample_id'])
            paths = [OUT / split / 'pipeline' / (stem + '.json'), OUT / 'predictions' / split / (stem + '.json'),
                     OUT / 'gui_reference' / split / (stem + '.json')]
            for path in paths: assert identities['input_sha256'][str(path.resolve())] == sha(path)
            details.append(case_analysis(case, *map(read, paths), audited_details[split][case['sample_id']]))
        summary = summarize(split, cohort, details); summary['identities'] = identities
        for name in ['gui_actor_K4', 'gui_actor_K16'] + [head + '_K16' for head in HEADS]:
            assert summary['methods'][name]['hits'] == summaries[split]['methods'][name]['hits'], ('Audited baseline changed', split, name)
        result[split] = summary; folder = destination / split
        write(folder / 'summary.json', summary); write(folder / 'details.json', details)
        predictions = [dict(sample_id=d['sample_id'], dataset=d['dataset'], method=m, **v) for d in details for m, v in d['methods'].items()]
        (folder / 'predictions.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in predictions), encoding='utf-8')
    assert not {c['sample_id'] for c in cohorts['test']} & {c['sample_id'] for c in cohorts['diagnostic']}
    write(destination / 'analysis.json', dict(complete=True, primary=PRIMARY, constants=CONSTANTS,
          classification='secondary addition during training, fixed before independent test', identities=identities, cohorts=result))
    (destination / 'report.html').write_text(render_report(result, identities), encoding='utf-8')
    print('FIXED COMPRESSION DIAGNOSTIC COMPLETE ' + str(destination), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', required=True, choices=['preregister', 'analyze'])
    args = parser.parse_args()
    preregister() if args.stage == 'preregister' else analyze()


if __name__ == '__main__': main()
