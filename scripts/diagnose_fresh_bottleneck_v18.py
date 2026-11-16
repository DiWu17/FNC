"""Retrospective error decomposition of completed fresh280 cached predictions.

No fitting, model inference, policy selection or modification of registered
results. Oracle means at least one candidate point lies in the annotated target;
it is a diagnostic upper bound, not an executable method.
"""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from clickagent.evidence_retention import view_rows
from clickagent.frozen_geometry_v18 import relocate
from clickagent.public_evaluation import point_in_geometry
from run_coordinate_control_v18 import moved_point_map

RUN = ROOT / 'runs/mvp_confirmation_v18'
OUT = ROOT / 'runs/full_scaling_v18/analysis/bottleneck_fresh280'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def hit(p, case):
    return bool(p is not None and point_in_geometry(
        [p[i] / case['image_size'][i] for i in [0, 1]], case['target_geometry']))


def points(pipe, case, mode):
    result = view_rows(relocate(pipe['global_result'], case['image_size'], mode), 0)
    for j, v in enumerate(pipe['local_views'], 1):
        result += view_rows(relocate(v['result'], v['transform']['crop_size'], mode), j, v['transform'])
    return {tuple(r['point']) for r in result}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    provenance = {}
    for folder in [RUN, RUN / 'coordinate_addendum']:
        receipt = read(folder / 'completion.json')
        assert receipt['complete']
        provenance[str(folder / 'completion.json')] = sha(folder / 'completion.json')
        for p, digest in receipt['artifact_sha256'].items():
            assert sha(p) == digest, p
    cases = [json.loads(s) for s in (RUN / 'data/heldout.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(cases) == len({c['pixel_sha256'] for c in cases}) == 280
    ours = {r['sample_id']: r for r in read(RUN / 'predictions/ours.json')['rows']}
    mvp = {r['sample_id']: r for r in read(RUN / 'predictions/mvp.json')['rows']}
    coordinate = {r['sample_id']: r for r in read(RUN / 'coordinate_addendum/combined_results.json')['coordinate_predictions']}
    rows = []
    for case in cases:
        sid = case['sample_id']; key = hashlib.sha256(sid.encode()).hexdigest()[:24]
        path = RUN / f'features/pipeline/{key}.json'
        pipe = read(path); assert pipe['sample_id'] == sid and pipe['complete']
        provenance[str(path)] = sha(path)
        fine = points(pipe, case, 'original')
        center = points(pipe, case, 'coarse_center')
        output_only = set(map(tuple, moved_point_map(pipe, case['image_size'], 'coarse_center').values()))
        native = {tuple(p['point']) for p in mvp[sid]['output']['reference_result']['all_predictions']}
        pools = dict(fine=fine, center=center, output_only=output_only, MVP=native)
        assert not ours[sid]['ground_truth_in_model_inputs'] and not mvp[sid]['ground_truth_in_model_inputs']
        hits = {}
        selected = dict(ours[sid]['points']); selected.update(coordinate[sid]['points']); selected['MVP'] = [mvp[sid]['point']]
        for name, predicted in selected.items():
            pool = native if name == 'MVP' else output_only if name.endswith('coordinate_center') else center if name.endswith('geometry') else fine
            assert all(p is None or tuple(p) in pool for p in predicted), (sid, name)
            hits[name] = [hit(p, case) for p in predicted]
            if name in ours[sid]['hits']:
                assert hits[name] == ours[sid]['hits'][name], (sid, name)
        assert hits['MVP'] == [mvp[sid]['hit']]
        rows.append(dict(sample_id=sid, source=case['dataset'],
                         oracle={name: any(hit(p, case) for p in pool) for name, pool in pools.items()},
                         candidate_counts={name: len(pool) for name, pool in pools.items()}, hits=hits))
    oracle = {name: sum(r['oracle'][name] for r in rows) for name in rows[0]['oracle']}
    methods = {}
    for name in rows[0]['hits']:
        pool = 'MVP' if name == 'MVP' else 'output_only' if name.endswith('coordinate_center') else 'center' if name.endswith('geometry') else 'fine'
        seed_hits = np.asarray([r['hits'][name] for r in rows], int).sum(0)
        methods[name] = dict(candidate_pool=pool, oracle_hits=oracle[pool],
                             seed_hits=seed_hits.tolist(), mean_hits=float(seed_hits.mean()),
                             mean_selection_errors=oracle[pool] - float(seed_hits.mean()),
                             candidate_miss_errors=280 - oracle[pool],
                             accuracy_percent=100*float(seed_hits.mean())/280,
                             conditional_selection_accuracy_percent=100*float(seed_hits.mean())/oracle[pool])
    groups = {}
    for a in [True, False]:
        for b in [True, False]:
            subset = [r for r in rows if r['oracle']['fine'] == a and r['oracle']['MVP'] == b]
            groups[f'ours_reachable_{a}_MVP_reachable_{b}'] = dict(n=len(subset),
                original3k_mean_hits=float(sum(np.mean(r['hits']['original3k_union']) for r in subset)),
                full_mean_hits=float(sum(np.mean(r['hits']['full_union']) for r in subset)),
                MVP_hits=sum(r['hits']['MVP'][0] for r in subset))
    result = dict(complete=True, analyzed_at_utc=datetime.now(timezone.utc).isoformat(), n=280,
                  scope='Retrospective descriptive diagnosis of the completed fresh280 test. No causal identification, new model inference, policy selection or new confirmatory claim.',
                  oracle_hits=oracle, methods=methods, joint_reachability=groups,
                  fine_to_center=dict(lost=sum(r['oracle']['fine'] and not r['oracle']['center'] for r in rows),
                                      gained=sum(not r['oracle']['fine'] and r['oracle']['center'] for r in rows)),
                  rows=rows, input_sha256=provenance, source_sha256=sha(__file__))
    (OUT / 'results.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    lines = ['V18 新280张缓存的事后错误分解（非新的确认性实验）。',
             'Oracle=候选中至少有一个点命中标注。它只是诊断上限，不是可执行准确率。',
             '没有新训练/推理/调参；没有改动已注册结果。MVP只有一个固定预测，其他方法为三种子均值。',
             '候选池可达数：'+json.dumps(oracle, ensure_ascii=False)]
    for name, r in methods.items():
        lines.append(f"{name}: 命中{r['mean_hits']:.2f}/280；候选可达{r['oracle_hits']}；有正确候选却选错{r['mean_selection_errors']:.2f}；没有正确候选{r['candidate_miss_errors']}；可达条件命中率{r['conditional_selection_accuracy_percent']:.2f}%")
    lines += ['共同可达分组：'+json.dumps(groups, ensure_ascii=False),
              '候选从细点改为中心的覆盖变化：'+json.dumps(result['fine_to_center'], ensure_ascii=False),
              '不同方法的候选内容、数量、上下文和计算量不同；分解不能把差距单独归因为评分器、局部视觉特征或参数量。',
              '候选缺失也不能单独诊断为坐标量化：视图、目标理解、分辨率、提议器及标注都可能影响。']
    (OUT / 'readout.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
