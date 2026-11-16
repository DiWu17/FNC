"""Reuse raw VLM caches to test coordinate decoding; existing scorer frozen."""
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import numpy as np
import torch
import run_quality_readout_v15_3k as prior
from clickagent.evidence_retention import RetentionModel, STATE_DIM
from clickagent.frozen_geometry_v18 import MODES, pipeline_geometry
from run_frozen_alternatives_v18 import read, write, sha, comparison

OUT = ROOT / 'runs/frozen_geometry_v18'
MAIN = ROOT / 'runs/quality_readout_v15_3k'
FULL = ROOT / 'runs/full_scaling_v18'


def status(phase, **fields):
    row = dict(phase=phase, complete=phase == 'complete', pid=__import__('os').getpid(),
               updated_at_utc=datetime.now(timezone.utc).isoformat(), **fields)
    write(OUT / 'status.json', row)
    print(json.dumps(row, ensure_ascii=False), flush=True)


def register():
    path = OUT / 'protocol.json'
    if path.exists():
        value = read(path)
        for p, expected in value['hashes'].items(): assert sha(p) == expected, p
        return value
    files = [Path(__file__), ROOT/'src/clickagent/frozen_geometry_v18.py',
             ROOT/'tests/test_frozen_geometry_v18.py', ROOT/'scripts/run_frozen_alternatives_v18.py',
             FULL/'data/development_indices.json', MAIN/'models/frozen.json']
    files += [ROOT/name for name in read(MAIN/'protocol.json')['source_identity']]
    value = dict(registered_at_utc=datetime.now(timezone.utc).isoformat(),
        hashes={str(p):sha(p) for p in files}, modes=list(MODES),
        hypothesis='Fine decoding may incur coordinate bias; compare existing learned point with coarse center, half displacement to center and archived-child center.',
        inputs='Original cached four views and frozen readout; recompute public-only point/state/base features for changed coordinates. No new view or Qwen inference.',
        selection='Union argmax for every fixed scorer seed; no labels enter relocation, features or scoring.',
        controls='Original CPU replay against authenticated cached GPU scores; report point agreement and score tolerance separately.',
        evaluation='Only already-used old172/new300 development; do not tune on V16/V17 benchmark outcomes.',
        uncertainty='Descriptive paired source-stratified RGB-cluster intervals; no independent confirmation or automatic adoption.',
        model_weights_unchanged=True, training=False, device='cpu', html=False)
    write(path, value); status('registered', modes=list(MODES))
    return value


def feature_record(pipe, public, raw):
    b = prior.base.build_base(pipe, public['image_size']); features = []; masks = []
    views = [dict(transform=None)] + pipe['local_views']
    for (arrays, meta), view in zip(raw, views):
        f, m = prior.base.state_features(arrays, meta, b['points'], public['image_size'], view['transform'])
        features.append(f); masks.append(m)
    tmax = max(1, *(f.shape[1] for f in features))
    state = np.zeros((len(b['points']), 4, tmax, STATE_DIM), np.float16)
    mask = np.zeros((len(b['points']), 4, tmax), bool)
    for j, (f, m) in enumerate(zip(features, masks)):
        state[:, j, :f.shape[1]] = f; mask[:, j, :f.shape[1]] = m[:, None]
    return dict(b, n=len(b['points']), state=state, mask=mask)


@torch.inference_mode()
def main():
    torch.set_num_threads(2); spec = register()
    indices = read(FULL/'data/development_indices.json')
    public = {}
    for path in [ROOT/'runs/multiregion_v13/data/val.jsonl', MAIN/'data/new_reserve.jsonl']:
        public.update({c['sample_id']:c for c in prior.base.rows(path)})
    frozen = read(MAIN/'models/frozen.json')['models']; models = []
    for s in [1, 2, 3]:
        info = frozen[f'expanded_s{s}']; assert sha(info['path']) == info['sha256']
        saved = torch.load(info['path'], map_location='cpu', weights_only=True)
        model = RetentionModel().eval().requires_grad_(False); model.load_state_dict(saved['state_dict'])
        models.append((model, saved))
    results = {}; provenance = {}; replay = dict(n_seed_predictions=0, point_agreement=0, max_abs_score_error=0.)
    for cohort, entries in indices.items():
        status('evaluating_'+cohort, completed=0, total=len(entries))
        vectors = {mode:[[], [], []] for mode in MODES}
        cases_out = []
        for i, entry in enumerate(entries, 1):
            case = public[entry['sample_id']]
            assert sha(entry['pipeline_path']) == entry['pipeline_sha256']
            provenance[entry['pipeline_path']] = entry['pipeline_sha256']
            pipe = read(entry['pipeline_path'])
            views = [dict(feature_path=pipe['global_feature_path'], feature_sha256=pipe['global_feature_sha256'])] + pipe['local_views']
            raw = []
            for v in views:
                assert sha(v['feature_path']) == v['feature_sha256']
                provenance[v['feature_path']] = v['feature_sha256']
                with np.load(v['feature_path'], allow_pickle=False) as z:
                    raw.append(({k:z[k].copy() for k in ['target_attention', 'target_hidden', 'image_hidden']}, json.loads(str(z['metadata']))))
            public_case = dict(image_size=case['image_size'])
            predicted = {}
            for mode in MODES:
                changed = pipeline_geometry(pipe, case['image_size'], mode, prior.base.build_c1_c3_pools)
                r = feature_record(changed, public_case, raw)
                points = []
                for s, (model, saved) in enumerate(models):
                    if r['n']:
                        inputs, _ = prior.tensors([r], saved['mean'].numpy(), saved['std'].numpy(), 'aligned', 'cpu')
                        scores = model(**inputs).float().numpy()
                        point = r['points'][int(scores.argmax())].tolist()
                    else:
                        scores = np.empty(0, np.float32); point = None
                    points.append(point)
                    if mode == 'original':
                        legacy = 'old_val' if cohort == 'old_val' else 'new_reserve'
                        path = MAIN/f'predictions/{legacy}/expanded_s{s+1}/{Path(entry["path"]).stem}.npz'
                        with np.load(path, allow_pickle=False) as z:
                            cached = z['scores'].copy()
                            assert str(z['base_sha256']) == entry['sha256']
                            assert str(z['checkpoint_sha256']) == frozen[f'expanded_s{s+1}']['sha256']
                        assert cached.shape == scores.shape
                        error = float(np.max(np.abs(cached-scores))) if len(scores) else 0.
                        assert error < 1e-4, ('Original feature replay failed', entry['sample_id'], error)
                        replay['max_abs_score_error'] = max(replay['max_abs_score_error'], error)
                        replay['n_seed_predictions'] += 1
                        replay['point_agreement'] += int(not len(scores) or int(cached.argmax()) == int(scores.argmax()))
                predicted[mode] = points
            # Offline geometry scoring only after all public-only predictions.
            for mode, points in predicted.items():
                for s, point in enumerate(points):
                    vectors[mode][s].append(bool(point is not None and prior.base.point_in_geometry((np.asarray(point)/case['image_size']).tolist(), case['target_geometry'])))
            cases_out.append(dict(sample_id=entry['sample_id'], points=predicted))
            if i % 10 == 0 or i == len(entries): status('evaluating_'+cohort, completed=i, total=len(entries))
        methods = {mode:dict(seed_hits=np.asarray(v).sum(1).tolist(), mean_accuracy_percent=100*float(np.mean(v))) for mode,v in vectors.items()}
        comparisons = {mode:comparison(v, vectors['original'], entries) for mode,v in vectors.items() if mode != 'original'}
        result = dict(complete=True, n=len(entries), methods=methods, comparisons=comparisons,
                      hit_vectors=vectors, predictions=cases_out, scope='Development only; descriptive exploration')
        write(OUT/f'evaluation/{cohort}.json', result); results[cohort] = result
    for path, expected in spec['hashes'].items(): assert sha(path) == expected
    lines = ['冻结评分器的坐标解码开发实验：没有新大模型前向、训练或独立测试。', json.dumps(replay, ensure_ascii=False)]
    for cohort, result in results.items():
        lines.append('\n'+cohort)
        for mode, values in result['methods'].items():
            lines.append(f'{mode}: {values["mean_accuracy_percent"]:.2f}%; 三种子命中{values["seed_hits"]}')
    path = OUT/'results.txt'; path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write(OUT/'completion.json', dict(complete=True, inputs_sha256=provenance, replay=replay,
        artifact_sha256={str(p):sha(p) for p in [OUT/'protocol.json',path]+[OUT/f'evaluation/{c}.json' for c in results]},
        adopted_scorer_unchanged=True, new_backbone_inference=False, training=False))
    status('complete'); print('\n'.join(lines), flush=True)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        status('failed', error=traceback.format_exc()); raise
