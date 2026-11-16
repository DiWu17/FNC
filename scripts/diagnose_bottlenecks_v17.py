"""Posthoc diagnosis of sealed predictions; CPU only, no model or policy changes."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from clickagent.public_evaluation import point_in_geometry

RUN = ROOT / 'runs/advantage_validation_v17'
OUT = RUN / 'diagnostics/bottlenecks_20261105'
SOURCES = {}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    path = Path(path)
    SOURCES[str(path)] = sha(path)
    return json.loads(path.read_text(encoding='utf-8'))


def authenticate(path, expected):
    actual = sha(path)
    assert actual == expected, path
    SOURCES[str(path)] = actual


def hits(points, case):
    size = np.asarray(case['image_size'])
    return [bool(p is not None and point_in_geometry((np.asarray(p) / size).tolist(), case['target_geometry'])) for p in points]


def main():
    verification = read(RUN / 'analysis/verification.json')
    assert verification['complete']
    for path, expected in verification['analysis_artifact_sha256'].items():
        authenticate(path, expected)
    plan = read(RUN / 'plan.json')
    idx_path = ROOT / 'runs/contribution_v16/independent/index.json'
    authenticate(idx_path, plan['input_hashes'][str(idx_path)])
    cases = {c['sample_id']: c for c in read(RUN / 'data/cohorts.json')['full_test']}
    entries = {e['sample_id']: e for e in read(idx_path)['cases']}
    outputs = {}
    for method in ['ours', 'mvp', 'ours_gui']:
        folder = RUN / f'inference/{method}_p262144_all'
        for path, expected in read(folder / 'completion.json')['hashes'].items():
            authenticate(path, expected)
        outputs[method] = {r['sample_id']: r for r in read(folder / 'accuracy/results.json')['rows']}
        assert set(outputs[method]) == set(cases) == set(entries)
    rows = []
    for i, (sid, case) in enumerate(cases.items(), 1):
        entry = entries[sid]
        for p, h in [('path', 'sha256'), ('labels_path', 'labels_sha256'), ('pipeline_path', 'pipeline_sha256')]:
            authenticate(entry[p], entry[h])
        with np.load(entry['path'], allow_pickle=False) as arrays:
            points, baseline = arrays['points'].copy(), arrays['baseline'].copy()
        labels = np.load(entry['labels_path'], allow_pickle=False) > .5
        assert np.array_equal(labels, hits(points.tolist(), case))
        pipe = read(entry['pipeline_path'])
        pool = {tuple(p) for p in points[baseline]}
        assert pool == {tuple(v['point']) for v in pipe['pools']['c3']['16']}
        ours = outputs['ours'][sid]
        own_hits = hits(ours['seed_points'], case)
        assert own_hits == ours['seed_hits']
        assert all((tuple(p) in pool if p is not None else not pool) for p in ours['seed_points'])
        gui = outputs['ours_gui'][sid]
        gui_hit = hits([gui['point']], case)[0]
        assert gui_hit == bool(gui['hit'])
        assert tuple(gui['point']) in pool if gui['point'] is not None else not pool
        mvp = outputs['mvp'][sid]
        mvp_hit = hits([mvp['point']], case)[0]
        assert mvp_hit == bool(mvp['hit'])
        proposals = [p['point'] for p in mvp['output']['reference_result']['all_predictions']]
        assert len(proposals) == 3
        k_reachable = bool(labels[baseline].any())
        union_reachable = bool(labels.any())
        m_reachable = any(hits(proposals, case))
        assert not any(own_hits) or k_reachable
        assert not gui_hit or k_reachable
        assert not mvp_hit or m_reachable
        rows.append(dict(sample_id=sid, source=case['dataset'],
            k16_reachable=k_reachable, union_reachable=union_reachable,
            mvp_reachable=m_reachable, ours_mean_hit=float(np.mean(own_hits)),
            ours_seed_hits=own_hits, gui_hit=gui_hit, mvp_hit=mvp_hit,
            union_candidates=len(points), k16_candidates=len(baseline),
            target_name=pipe['global_result']['target_name'],
            global_grid_shape=pipe['global_result']['grid_shape']))
        if i % 100 == 0:
            print(f'Authenticated and re-scored fixed predictions: {i}/300', flush=True)
    n = len(rows)
    own = sum(r['ours_mean_hit'] for r in rows)
    k = sum(r['k16_reachable'] for r in rows)
    union = sum(r['union_reachable'] for r in rows)
    m = sum(r['mvp_reachable'] for r in rows)
    mvphits = sum(r['mvp_hit'] for r in rows)
    guihits = sum(r['gui_hit'] for r in rows)
    groups = {}
    for key, predicate in [
        ('both_reachable', lambda r: r['k16_reachable'] and r['mvp_reachable']),
        ('K16_only', lambda r: r['k16_reachable'] and not r['mvp_reachable']),
        ('MVP_only', lambda r: not r['k16_reachable'] and r['mvp_reachable']),
        ('neither_reachable', lambda r: not r['k16_reachable'] and not r['mvp_reachable'])]:
        selected = [r for r in rows if predicate(r)]
        groups[key] = dict(n=len(selected), ours_mean_hits=sum(r['ours_mean_hit'] for r in selected),
                           mvp_hits=sum(r['mvp_hit'] for r in selected))
    assert (k, union, m, mvphits, guihits) == (152, 192, 151, 116, 103)
    assert np.isclose(own, 263 / 3)
    old_groups = read(RUN / 'analysis/joint_proposal_diagnostic.json')['groups']
    for name, group in groups.items():
        for key, value in group.items():
            assert np.isclose(value, old_groups[name][key])
    gap = mvphits - own
    shared_gap = groups['both_reachable']['mvp_hits'] - groups['both_reachable']['ours_mean_hits']
    exclusive_gap = groups['MVP_only']['mvp_hits'] - groups['K16_only']['ours_mean_hits']
    assert np.isclose(shared_gap + exclusive_gap, gap)
    by_source = {}
    for source in sorted({r['source'] for r in rows}):
        chosen = [r for r in rows if r['source'] == source]
        reachable = sum(r['k16_reachable'] for r in chosen)
        correct = sum(r['ours_mean_hit'] for r in chosen)
        by_source[source] = dict(n=len(chosen), K16_reachable=reachable,
            union_reachable=sum(r['union_reachable'] for r in chosen),
            ours_mean_hits=correct, conditional_selection_percent=100 * correct / reachable)
    timing = {}
    for method, stage in [('ours_s1', 'speed'), ('mvp', 'all'), ('ours_gui', 'all')]:
        data = read(RUN / f'inference/{method}_p262144_{stage}/speed/results.json')['rows']
        times = [r['measured']['elapsed_seconds'] for r in data]
        timing[method] = dict(n_calls=len(data), n_screenshots=len({r['sample_id'] for r in data}),
            mean_seconds=float(np.mean(times)),
            mean_backbone_forward_calls=float(np.mean([r['measured']['counter']['backbone_forward_calls'] for r in data])),
            mean_processed_pixels=float(np.mean([r['measured']['counter']['processed_pixels'] for r in data])))
    result = dict(complete=True, created_at_utc=datetime.now(timezone.utc).isoformat(),
        scope='Posthoc descriptive diagnosis of reused V16/V17 fixed outputs. No training, model inference, new policy, seed selection or independent confirmation.',
        n=n, ours_seed_hits=[sum(r['ours_seed_hits'][i] for r in rows) for i in range(3)],
        ours_mean_hits=own,
        failure_partition=dict(K16_unreachable=n-k, reachable_but_wrong_mean=k-own,
            total_misses_mean=n-own, unreachable_share_of_misses_percent=100*(n-k)/(n-own)),
        candidate_limits=dict(K16_oracle_percent=100*k/n, union_oracle_percent=100*union/n,
            union_correct_removed_by_K16=union-k, no_union_correct=n-union,
            ours_selection_given_K16_reachable_percent=100*own/k,
            gui_selection_given_K16_reachable_percent=100*guihits/k,
            mean_union_candidates=float(np.mean([r['union_candidates'] for r in rows])),
            K16_size_counts=dict(Counter(r['k16_candidates'] for r in rows))),
        mvp_gap=dict(mvp_minus_ours_hits=gap, common_reachable_gap_hits=shared_gap,
            exclusive_reachable_net_gap_hits=exclusive_gap,
            common_reachable_share_percent=100*shared_gap/gap,
            candidate_oracles=dict(ours=k, mvp=m), groups=groups,
            caveat='Accounting of fixed predictions, not causal effects of replacing a ranker or proposal policy; candidate counts, contents and crops differ.'),
        same_K16_verifier=dict(gui_hits=guihits, ours_mean_hits=own,
            net_gain_hits=guihits-own, net_gain_percentage_points=100*(guihits-own)/n,
            caveat='Same eligible coordinates; verifier also changes model capacity, training, image representation, prompt, crop resolution and compute. These causes are not isolated.'),
        by_source=by_source, timing=timing,
        structural_findings=[
            'Adopted inference restricts argmax to b[baseline] (original K16); scorer cannot add or refine coordinates.',
            'Global and local views use frozen Qwen3-VL-8B and a frozen 611919-parameter proposer; the 2973-query adaptation trains only the 22050-parameter readout.',
            'State input is 294 engineered channels per target token/view: 288 attention statistics, 3 query/image cosines, 3 query norms; plus 32 base channels. It is not full hidden vectors or a fresh candidate-image verifier.',
            'Training ranks the complete proposal union, while reported inference selects only original K16. This is a verified objective/pool difference, not a measured causal explanation of the accuracy gap.',
            'Fresh timing includes serial target-name generation, state extraction and multiple Qwen views. Existing traces do not isolate exact stage time; the small scorer alone is not an end-to-end cost proxy.'
        ],
        unresolved=['Readout parameter capacity vs state-information loss vs supervision/domain shift have not been causally separated.',
                    'MVP changes the prompt, attention readout token/layer, crop geometry, prediction representation and fusion together; these mechanisms have not been isolated.',
                    'Higher-capacity external verifier is not a controlled test that increasing our own head size will fix the gap.'],
        official_sources=dict(MVP='https://github.com/ZJUSCL/MVP',
            verifier='https://huggingface.co/microsoft/GUI-Actor-Verifier-2B',
            verifier_training='https://arxiv.org/html/2506.03143#S4'))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / 'diagnosis.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    (OUT / 'case_breakdown.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines = ['V17瓶颈诊断：仅复核固定预测，无新训练/推理/模型选择',
        f'300题，三种子命中数：{result["ours_seed_hits"]}；均值{own:.2f}题。',
        f'原K16没有正确候选：{n-k}题，占全部错误的{100*(n-k)/(n-own):.2f}%。',
        f'K16有正确候选但选错：平均{k-own:.2f}题；条件选对率{100*own/k:.2f}%。',
        f'完整已生成候选并集覆盖{union}题，其中{union-k}题的正确候选未进入原K16；{n-union}题连并集都没有正确点。',
        f'同K16外部验证器选对{guihits}题，条件选对率{100*guihits/k:.2f}%，净提升{100*(guihits-own)/n:.2f}百分点。',
        '外部验证器同时改变了容量、训练、图像输入和计算量，不能归因于参数量单一因素。',
        f'MVP候选覆盖{m}题，与我们{k}题接近。双方共同可达{groups["both_reachable"]["n"]}题，我们平均{groups["both_reachable"]["ours_mean_hits"]:.2f}题，MVP为{groups["both_reachable"]["mvp_hits"]}题。',
        f'MVP总领先{gap:.2f}题，共同可达子集贡献{shared_gap:.2f}题，双方独有可达子集净贡献{exclusive_gap:.2f}题；此为描述性记账，不是因果分解。',
        'MVP与我们加载同一本地Qwen3-VL-8B-Instruct权重，本轮均冻结骨干；MVP无额外定位训练。',
        'GUI-Actor-Verifier-2B是已专门训练的额外2B验证器，并非我们在本轮重新训练的模型。',
        '训练优化完整候选并集，评估强制在原K16选点；这是需检查的目标/候选池差异，尚无干预证据证明其影响大小。',
        '状态压缩和评分器容量都是待验证原因，当前不能断言扩大评分器或微调骨干即可解决。',
        '', '来源、耗时及细分统计见 diagnosis.json；逐题分组见 case_breakdown.json。']
    (OUT / 'diagnosis.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    SOURCES[str(Path(__file__))] = sha(__file__)
    receipt = dict(complete=True, inputs_sha256=SOURCES,
        outputs_sha256={str(OUT/name): sha(OUT/name) for name in ['diagnosis.json', 'diagnosis.txt', 'case_breakdown.json']},
        no_model_or_registered_source_changes=True)
    (OUT / 'provenance.json').write_text(json.dumps(receipt, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print('\n'.join(lines), flush=True)
    print(json.dumps(timing, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
