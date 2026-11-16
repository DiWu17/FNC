"""Post-hoc development diagnosis; never selects a policy or runs a model."""
from datetime import datetime, timezone
from pathlib import Path
import json
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import numpy as np
from clickagent.evidence_retention import view_rows
from clickagent.public_evaluation import point_in_geometry
from run_coordinate_control_v18 import moved_point_map
from run_frozen_alternatives_v18 import read, write, sha

OUT = ROOT / 'runs/coordinate_control_v18/analysis'


def canonical_sources(pipe):
    views = [view_rows(pipe['global_result'], 0)]
    views += [view_rows(v['result'], j, v['transform']) for j, v in enumerate(pipe['local_views'], 1)]
    ordered = views[0] + [v[k] for k in range(max(map(len, views[1:]), default=0)) for v in views[1:] if k < len(v)]
    sources = {}
    for row in ordered:
        sources.setdefault(tuple(row['point']), row['view'])
    return sources


def summarize(rows):
    before = np.asarray([r['original_hits'] for r in rows], bool)
    after = np.asarray([r['center_hits'] for r in rows], bool)
    return dict(n_queries=len(rows),
                original_percent=100 * float(before.mean()), center_percent=100 * float(after.mean()),
                seed_gains=np.sum(~before & after, axis=0).tolist(),
                seed_losses=np.sum(before & ~after, axis=0).tolist(),
                seed_misses_with_center_oracle=np.sum(~after & np.asarray([r['center_oracle'] for r in rows])[:, None], axis=0).tolist())


def main():
    inputs = [Path(__file__), ROOT / 'runs/coordinate_control_v18/completion.json',
              ROOT / 'runs/full_scaling_v18/data/development_indices.json',
              ROOT / 'src/clickagent/frozen_geometry_v18.py', ROOT / 'src/clickagent/evidence_retention.py',
              ROOT / 'scripts/run_coordinate_control_v18.py', ROOT / 'src/clickagent/public_evaluation.py',
              ROOT / 'runs/multiregion_v13/data/val.jsonl',
              ROOT / 'runs/quality_readout_v15_3k/data/new_reserve.jsonl']
    receipt = read(inputs[1])
    assert receipt['complete']
    for path, digest in receipt['artifact_sha256'].items():
        assert sha(path) == digest, path
    indices = read(inputs[2])
    public = {c['sample_id']: c for path in inputs[-2:] for c in
              (json.loads(line) for line in path.read_text(encoding='utf-8').splitlines())}
    results = {}
    provenance = {str(p): sha(p) for p in inputs}
    lines = ['V18 坐标变化机制：仅复用开发预测的事后描述性分析。',
             '不训练、不推理、不改变已登记策略；真值仅用于离线诊断。',
             '候选 oracle 表示候选池中至少一个点命中，并不是可执行方法的成绩。',
             '三个种子分别报告收益/损失；同一截图的查询及多个种子不当作独立样本。']
    for cohort, entries in indices.items():
        path = ROOT / f'runs/coordinate_control_v18/evaluation/{cohort}.json'
        predictions = read(path)
        provenance[str(path)] = sha(path)
        assert len(predictions['predictions']) == len(entries)
        rows = []
        for i, (entry, prediction) in enumerate(zip(entries, predictions['predictions'])):
            assert entry['sample_id'] == prediction['sample_id']
            case = public[entry['sample_id']]
            assert sha(entry['pipeline_path']) == entry['pipeline_sha256']
            assert sha(entry['path']) == entry['sha256']
            provenance[entry['pipeline_path']] = entry['pipeline_sha256']
            provenance[entry['path']] = entry['sha256']
            pipe = read(entry['pipeline_path'])
            mapping = moved_point_map(pipe, case['image_size'], 'coarse_center')
            sources = canonical_sources(pipe)
            with np.load(entry['path'], allow_pickle=False) as archive:
                points = archive['points'].copy()
            assert set(mapping) == set(map(tuple, points)) == set(sources)
            size = np.asarray(case['image_size'])
            def hit(point):
                return point is not None and point_in_geometry((np.asarray(point) / size).tolist(), case['target_geometry'])
            original_hits = [bool(hit(p)) for p in prediction['points']['original']]
            center_hits = [bool(hit(p)) for p in prediction['points']['coarse_center']]
            assert original_hits == [v[i] for v in predictions['hit_vectors']['original']]
            assert center_hits == [v[i] for v in predictions['hit_vectors']['coarse_center']]
            oracle_original = any(hit(p) for p in points)
            oracle_center = any(hit(mapping[tuple(p)]) for p in points)
            selected_views = [sources[tuple(p)] if p is not None else None for p in prediction['points']['original']]
            geometry = case['target_geometry']
            minside = None
            if geometry['type'] == 'bbox':
                # Largest acceptable rectangle, chosen for a stable descriptive size bin.
                box = max(geometry['boxes'], key=lambda b: (b[2]-b[0])*(b[3]-b[1]))
                minside = min((box[2]-box[0])*size[0], (box[3]-box[1])*size[1])
            size_bin = 'polygon' if minside is None else '<16 px' if minside < 16 else '16-32 px' if minside < 32 else '>=32 px'
            rows.append(dict(sample_id=entry['sample_id'], source=entry['source'], size_bin=size_bin,
                             original_hits=original_hits, center_hits=center_hits,
                             original_oracle=bool(oracle_original), center_oracle=bool(oracle_center),
                             selected_views=selected_views))
        original_oracles = np.asarray([r['original_oracle'] for r in rows])
        center_oracles = np.asarray([r['center_oracle'] for r in rows])
        result = dict(overall=summarize(rows),
                      oracle=dict(original_hits=int(original_oracles.sum()), center_hits=int(center_oracles.sum()),
                                  gained=int((~original_oracles & center_oracles).sum()),
                                  lost=int((original_oracles & ~center_oracles).sum())),
                      by_size={name:summarize([r for r in rows if r['size_bin']==name]) for name in sorted({r['size_bin'] for r in rows})},
                      by_source={name:summarize([r for r in rows if r['source']==name]) for name in sorted({r['source'] for r in rows})},
                      rows=rows)
        result['by_selected_view'] = {}
        for name in ['global', 'local']:
            counts = []
            for seed in range(3):
                selected = [r for r in rows if r['selected_views'][seed] is not None and (r['selected_views'][seed] == 0) == (name == 'global')]
                counts.append(dict(n=len(selected), gains=sum(not r['original_hits'][seed] and r['center_hits'][seed] for r in selected),
                                   losses=sum(r['original_hits'][seed] and not r['center_hits'][seed] for r in selected)))
            result['by_selected_view'][name] = counts
        results[cohort] = result
        overall = result['overall']; oracle = result['oracle']
        lines += ['', cohort,
                  f"准确率 {overall['original_percent']:.2f}% → {overall['center_percent']:.2f}%；种子收益{overall['seed_gains']}，损失{overall['seed_losses']}。",
                  f"全候选 oracle {oracle['original_hits']}/{len(rows)} → {oracle['center_hits']}/{len(rows)}；可达性新增{oracle['gained']}条、丢失{oracle['lost']}条。",
                  f"移动后仍未命中、但移动后候选池内存在正确点：每种子{overall['seed_misses_with_center_oracle']}条。",
                  '分目标尺寸（只是离线分层，不能按真值尺寸选择策略）：',
                  json.dumps(result['by_size'], ensure_ascii=False, indent=2),
                  '分原选中候选的视图：', json.dumps(result['by_selected_view'], ensure_ascii=False, indent=2)]
    write(OUT / 'mechanism.json', dict(complete=True, analysis_type='post_hoc_development_only', cohorts=results))
    (OUT / 'mechanism.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write(OUT / 'mechanism_verification.json', dict(complete=True, created_at_utc=datetime.now(timezone.utc).isoformat(),
          input_sha256=provenance, artifacts_sha256={str(p):sha(p) for p in [OUT/'mechanism.json', OUT/'mechanism.txt']},
          no_inference=True, no_training=True, no_policy_change=True))
    print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    main()
