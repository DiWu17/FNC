"""Fixed A candidates; cached B-name evidence and train-only location calibration.

No VLM calls, no original weight/source changes, no independent-test claims.
"""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import numpy as np
import torch
from clickagent.location_selection_v26 import (LOCATIONS, location_index, cell_ids,
    location_residual, choose, shuffled_locations)
from clickagent.evidence_retention import RetentionModel, STATE_DIM, state_features
from clickagent.prompt_readout_v25 import select_arrays
from run_evidence_retention_v14 import tensors
from run_visual_feedback_v23 import read, write, sha, hit, bootstrap
import run_prompt_readout_v25 as prior

OUT = ROOT / 'runs/location_selection_v26'
V25 = prior.OUT
SOURCES = [Path(__file__), ROOT / 'src/clickagent/location_selection_v26.py',
           ROOT / 'tests/test_location_selection_v26.py']
GRID = [0., .25, .5, 1., 2., 4., 8., 16.]
NAMES = prior.NAMES
SHUFFLES = 100
CONTRASTS = [('A_position', 'A'), ('B', 'A'), ('B_position', 'B'),
             ('B_position', 'A'), ('A_position', 'A_shuffled'),
             ('B_position', 'B_shuffled'), ('A_position', 'A_center'),
             ('B_position', 'B_center')]


def now(): return datetime.now(timezone.utc).isoformat()
def identity(): return {str(p): sha(p) for p in SOURCES}
def status(phase, **kw):
    value = dict(phase=phase, complete=phase == 'complete', updated_at_utc=now(), pid=os.getpid(), **kw)
    write(OUT / 'status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def feature_path(c): return OUT / 'features' / (prior.key(c['sample_id']) + '.npz')
def score_path(c): return OUT / 'scores' / (prior.key(c['sample_id']) + '.npz')


def prepare():
    assert not (OUT / 'protocol.json').exists(), 'Use a new run rather than overwrite a registered run'
    cases = read(V25 / 'data/public.json')
    train = [c for c in cases if c['cohort'] == 'train']
    dev = [c for c in cases if c['cohort'] != 'train']
    assert len(train) == 56 and len(dev) == 36
    assert not {c['group'] for c in train} & {c['group'] for c in dev}
    assert len({c['image_sha256'] for c in cases}) == 92
    ps = read(V25 / 'protocol.json')
    assert read(V25 / 'final_audit.json')['complete']
    sealed = {str(p): sha(p) for p in SOURCES + [
        V25 / ('data/' + n + '.json') for n in ['public', 'labels', 'tasks']] + [
        V25 / n for n in ['protocol.json', 'predictions.json', 'cache_audit_index.json', 'final_audit.json']]}
    for p, h in ps['source_sha256'].items():
        assert sha(p) == h; sealed[p] = h
    for name, model in ps['models'].items():
        assert sha(model['path']) == model['sha256']; sealed[model['path']] = model['sha256']
    for p in ['src/clickagent/evidence_retention.py', 'src/clickagent/public_evaluation.py',
              'scripts/run_evidence_retention_v14.py', 'scripts/run_visual_feedback_v23.py']:
        sealed[str(ROOT / p)] = sha(ROOT / p)
    tests = read(OUT / 'unit_tests.json')
    assert tests['exit_code'] == 0 and tests['source_sha256'] == identity()
    spec = dict(registered_at_utc=now(), source_sha256=identity(), sealed_sha256=sealed,
        samples='Exactly V25: original-training56 unique RGB; historically reused new24+old12 development. One query/RGB. Not fresh independent test.',
        observations='Cached original A and framed B trajectories only; same old global/local view geometry. Zero new VLM calls. No new candidate proposal or crop planning.',
        candidate_contract='Every production and diagnostic arm uses the complete original A candidate points, base features, active masks and canonical tie order. B-name states sampled at these SAME points. Candidate geometry/coverage never changes.',
        base_arms={'A':'Original A target-name state with original frozen head',
                   'B':'All B target-name subtokens, same A points/base/active and same original frozen head'},
        location='Primary global-view B frame ONLY, action CLICK, complete frame, nonempty entity, valid9-bin required. Otherwise zero residual. Local views diagnosed separately but cannot override global position.',
        residual='zscore(original candidate logits) + lambda * negative squared distance from normalized point to named nine-bin rectangle / (2*.35^2). Residual zero inside bin. lambda=0 exactly returns original argmax. No hard filtering in deployed arms.',
        calibration=dict(grid=GRID, selection='Fit one scalar per original model and base A/B using original-training labels only. Max equal-source mean top1 accuracy, then min lambda; no development hyperparameter selection.',
                         effective_train='Retain all56 cases, including empty pools; missing location is identity. Same sample IDs across bases/controls.',
                         cross_validation='Seven leave-one-source-out TRAIN-only folds report transfer; final lambda fitted to all original-training56 before development scoring/analysis.',
                         static_center='Separate training-only lambda, center bin on every image with a valid predicted location, same missingness.',
                         shuffled='100 fixed label-free Sattolo donor derangements of eligible global locations within cohort/source, preserving location marginals and missingness. Apply the real-position-fitted lambda; average null performance across100 draws. No development labels used in shuffle.',
                         original_checkpoints='All6 original3k/full checkpoints frozen; no adapters, new MLP or backbone fitting.'),
        diagnostics='Global and local location: center-bin match and boundary-aware target/binnine overlap, visibility counted separately; view observations correlated. Oracle target-bbox-intersecting bins hard-mask original logits FOR ANALYSIS ONLY; all positive A candidates retained by construction. This oracle uses annotation, not an inference method or full theoretical ceiling. Predicted hard-bin filter also descriptive only.',
        primary=dict(cohort='new_development24', contrasts=CONTRASTS,
                     inference='3 fixed-fit means, 5000 source-stratified RGB bootstrap,16 contrasts Bonferroni99.6875% intervals; tiny reused development evidence only.'),
        advancement='For each A/B x3k/full: real-position vs same base >=3pp with corrected lower>0; real-position vs shuffled and vs independently calibrated static-center lower>0; B also vs A lower>0; old12 vs A delta>=-5pp. No adoption. Only a positive gate can justify later independent confirmation.',
        user_authorization='试试吧：固定候选、生成位置可靠性、真实位置诊断上限及打乱位置对照；训练内轻量软约束校准。',
        mainline_unchanged=True, new_vlm_calls=0, outputs='Python/JSON/TXT/NPZ/static PNG; no HTML')
    write(OUT / 'protocol.json', spec)
    write(OUT / 'registry_before.json', read(ROOT / 'CURRENT_RESEARCH.json'))
    for p in SOURCES:
        dest = OUT / 'source_archive' / p.relative_to(ROOT)
        dest.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(p, dest)
    reg = read(ROOT / 'CURRENT_RESEARCH.json')
    reg['active_experiment'] = dict(run_id=OUT.name, status='running', phase='registered',
        protocol_path=str(OUT / 'protocol.json'), status_path=str(OUT / 'status.json'), mainline_unchanged=True)
    write(ROOT / 'CURRENT_RESEARCH.json', reg)
    write(ROOT / 'research/decisions/20261109_location_selection_v26.json',
          dict(user_authorization=spec['user_authorization'], protocol_sha256=sha(OUT / 'protocol.json'), registered_at_utc=now()))
    status('registered', training_rgb=56, development_rgb=36, new_vlm_calls=0)


def verify():
    spec = read(OUT / 'protocol.json')
    assert spec['source_sha256'] == identity(), 'Registered V26 source drift'
    for p, h in spec['sealed_sha256'].items(): assert sha(p) == h, p
    return spec


def load_meta(c, vi):
    task = prior.task_for(c, vi, 'context'); p = V25 / f'raw/{task["task_id"]}.npz'
    with np.load(p, allow_pickle=False) as z: m = json.loads(str(z['metadata']))
    assert m['task'] == task and m['transform'] == c['views'][vi]['transform']
    return m, p


def build(cases):
    expected = read(V25 / 'cache_audit_index.json')['raw_sha256']
    reads = {}; locations = {}; descriptions = {}; old = read(V25 / 'protocol.json')
    for ci, c in enumerate(cases, 1):
        f = prior.feature_path(c, 'name'); reads[str(f)] = sha(f)
        if c['cohort'] != 'train':
            assert reads[str(f)] == read(V25 / 'cache_audit_index.json')['development_feature_sha256'][str(f)]
        a, fm = prior.load_npz(f)
        assert fm['protocol_sha256'] == sha(V25 / 'protocol.json') and fm['sample_id'] == c['sample_id'] and fm['arm'] == 'name'
        n = len(a['points']); states = []; masks = []; descriptions[c['sample_id']] = []
        for vi, v in enumerate(c['views']):
            m, p = load_meta(c, vi)
            h = sha(p); assert expected[str(p)] == h, str(p); reads[str(p)] = h
            assert m['protocol_sha256'] == sha(V25 / 'protocol.json')
            descriptions[c['sample_id']].append(dict(view_index=vi, location=location_index(m),
                fields=m['ground_fields'], target_available=m['target_available'], action=m['action'],
                transform=v['transform'], generated_text=m['generated_text']))
            if vi == 0: locations[c['sample_id']] = location_index(m)
            # Only the target-name token slice is decompressed; other output roles
            # stay in their raw archive and cannot enter B-name feature extraction.
            with np.load(p, allow_pickle=False) as z:
                ix = np.asarray(m['readout_stages']['target'], np.int64) - 1
                arrays = dict(target_attention=z['all_attention'][ix], target_hidden=z['all_hidden'][ix],
                              image_hidden=z['image_hidden'])
            state, valid = state_features(arrays, m, a['points'], c['image_size'], v['transform'])
            states.append(state); masks.append(valid)
        tmax = max(1, *(s.shape[1] for s in states))
        state = np.zeros((n, 4, tmax, STATE_DIM), np.float16); mask = np.zeros((n, 4, tmax), bool)
        for vi, (s, valid) in enumerate(zip(states, masks)):
            state[:, vi, :s.shape[1]] = s; mask[:, vi, :s.shape[1]] = valid[:, None]
        prior.npz(feature_path(c), points=a['points'], base=a['base'], active=a['active'],
                  baseline=a['baseline'], state=state, mask=mask,
                  metadata=np.asarray(json.dumps(dict(sample_id=c['sample_id'], fixed_pool_sha256=sha(f),
                      original_point_count=n, ground_truth_in_features=False))))
        if ci % 8 == 0 or ci == len(cases): status('aligning_cached_B_states_to_A_points', completed=ci, total=len(cases))
    write(OUT / 'input_cache_sha256.json', reads)
    write(OUT / 'locations.json', locations); write(OUT / 'descriptions.json', descriptions)
    verify()


def score(cases, role):
    assert all((c['cohort'] == 'train') == (role == 'train') for c in cases)
    if role != 'train': assert read(OUT / 'calibration.json')['frozen_before_development_scoring']
    models = {}
    for name, row in read(V25 / 'protocol.json')['models'].items():
        s = torch.load(row['path'], map_location='cpu', weights_only=True)
        model = RetentionModel().eval().requires_grad_(False); model.load_state_dict(s['state_dict'])
        models[name] = (model, s)
    parity = 0
    prev = {(r['sample_id'], r['model']): r for r in read(V25 / 'predictions.json')['rows']
            if r['regime'] == 'frozen' and r['arm'] == 'name'}
    for ci, c in enumerate(cases, 1):
        values = {}
        for arm in ['A', 'B']:
            p = prior.feature_path(c, 'name') if arm == 'A' else feature_path(c)
            a, m = prior.load_npz(p); r = dict(n=len(a['points']), sample_id=c['sample_id'], **a)
            for name, (model, saved) in models.items():
                if r['n']:
                    inputs, _ = tensors([r], saved['mean'].numpy(), saved['std'].numpy(), 'aligned', 'cpu')
                    with torch.inference_mode(): s = model(**inputs).numpy()
                else: s = np.empty(0, np.float32)
                values[f'{arm}__{name}'] = s
                if arm == 'A' and role != 'train':
                    index = choose(s); point = a['points'][index].tolist() if index is not None else None
                    assert point == prev[c['sample_id'], name]['point']; parity += 1
        prior.npz(score_path(c), metadata=np.asarray(json.dumps(dict(sample_id=c['sample_id'], role=role))), **values)
        if ci % 8 == 0 or ci == len(cases): status('scoring_' + role, completed=ci, total=len(cases), baseline_parity=parity)
    write(OUT / f'{role}_scoring.json', dict(complete=True, at_utc=now(), cases=len(cases), baseline_parity=parity))


def records(cases):
    out = {}
    for c in cases:
        a, _ = prior.load_npz(feature_path(c))
        s, _ = prior.load_npz(score_path(c))
        out[c['sample_id']] = dict(points=a['points'], scores=s)
    return out


def labels_at(c, points, labels):
    return np.asarray([hit(p.tolist(), c, labels[c['sample_id']]) for p in points], bool)


def fit_strength(cases, recs, labels, loc, model, base):
    curves = []
    for strength in GRID:
        by_source = defaultdict(list)
        for c in cases:
            r = recs[c['sample_id']]; points = r['points']
            s = r['scores'][f'{base}__{model}']; g = location_residual(points, c['image_size'], loc[c['sample_id']])
            j = choose(s, g, strength)
            correct = bool(j is not None and labels_at(c, points[j:j+1], labels)[0])
            by_source[c['source']].append(correct)
        macro = float(np.mean([np.mean(v) for v in by_source.values()]))
        curves.append(dict(strength=strength, source_macro_accuracy=macro))
    best = max(curves, key=lambda r: (round(r['source_macro_accuracy'], 12), -r['strength']))
    return best['strength'], curves


def calibrate(cases):
    assert all(c['cohort'] == 'train' for c in cases)
    recs = records(cases); labels = read(V25 / 'data/labels.json'); locations = read(OUT / 'locations.json')
    loc = {c['sample_id']: locations[c['sample_id']] for c in cases}
    center = {sid: 4 if v is not None else None for sid, v in loc.items()}
    models = {}; cv = []
    for model in NAMES:
        for base in ['A', 'B']:
            tag = base + '__' + model
            real, curve = fit_strength(cases, recs, labels, loc, model, base)
            cen, ccurve = fit_strength(cases, recs, labels, center, model, base)
            models[tag] = dict(position_strength=real, center_strength=cen,
                               position_train_curve=curve, center_train_curve=ccurve)
            for src in sorted({c['source'] for c in cases}):
                fit = [c for c in cases if c['source'] != src]; val = [c for c in cases if c['source'] == src]
                weight, _ = fit_strength(fit, recs, labels, loc, model, base)
                for c in val:
                    r = recs[c['sample_id']]; s = r['scores'][tag]; p = r['points']
                    g = location_residual(p, c['image_size'], loc[c['sample_id']]); truth = labels_at(c, p, labels)
                    a, b = choose(s), choose(s, g, weight)
                    cv.append(dict(sample_id=c['sample_id'], source=src, model=model, base=base, strength=weight,
                        original_hit=bool(a is not None and truth[a]), position_hit=bool(b is not None and truth[b]),
                        excluded_source=src))
            status('calibrating_on_training_only', model=model, base=base, position_strength=real, center_strength=cen)
    write(OUT / 'calibration.json', dict(frozen_before_development_scoring=True, frozen_at_utc=now(),
        training_sample_ids=[c['sample_id'] for c in cases], training_rgb=len(cases),
        development_labels_used=False, original_models_frozen=True, models=models, train_leave_source_out=cv))


def geometry_bins(c, geometry, transform=None):
    """Diagnostic ONLY: GT boxes clipped into the displayed view, normalized."""
    assert geometry['type'] == 'bbox'
    size = np.asarray(c['image_size'], float)
    bounds = np.asarray([0., 0., *size] if transform is None else transform['box'], float)
    boxes = []
    for box in geometry['boxes']:
        b = np.asarray(box, float) * np.tile(size, 2)
        lo, hi = np.maximum(b[:2], bounds[:2]), np.minimum(b[2:], bounds[2:])
        if (hi > lo).all(): boxes.append(np.r_[(lo-bounds[:2])/(bounds[2:]-bounds[:2]), (hi-bounds[:2])/(bounds[2:]-bounds[:2])])
    bins = set(); center_bins = set()
    for box in boxes:
        center_bins.add(int(cell_ids([(box[:2] + box[2:]) / 2], [1, 1])[0]))
        for i in range(9):
            lo = np.asarray([i % 3, i // 3]) / 3; hi = lo + 1/3
            if (box[2:] >= lo - 1e-12).all() and (box[:2] <= hi + 1e-12).all(): bins.add(i)
    return bins, center_bins, boxes


def evaluate(cases):
    assert all(c['cohort'] != 'train' for c in cases)
    labels = read(V25 / 'data/labels.json'); recs = records(cases); calib = read(OUT / 'calibration.json')
    all_loc = read(OUT / 'locations.json'); loc = {c['sample_id']: all_loc[c['sample_id']] for c in cases}
    shuffles = [shuffled_locations(cases, loc, i) for i in range(SHUFFLES)]
    write(OUT / 'shuffle_donors.json', [donors for _, donors in shuffles])
    rows = []; oracle = []; spatial = []; coverage = []; descriptions = read(OUT / 'descriptions.json')
    for c in cases:
        sid = c['sample_id']; r = recs[sid]; p = r['points']; truth = labels_at(c, p, labels); cells = cell_ids(p, c['image_size'])
        oracle_bins, _, _ = geometry_bins(c, labels[sid]); oracle_allowed = np.isin(cells, list(oracle_bins))
        assert np.all(oracle_allowed[truth]), 'Diagnostic GT-bin mask must retain every true point'
        coverage.append(dict(sample_id=sid, candidate_count=len(p), covered=bool(truth.any()),
            true_candidates=int(truth.sum()), candidate_sha256=hashlib.sha256(p.tobytes()).hexdigest()))
        for d in descriptions[sid]:
            bins, centers, boxes = geometry_bins(c, labels[sid], d['transform'])
            valid = d['location'] is not None
            spatial.append(dict(sample_id=sid, source=c['source'], cohort=c['cohort'],
                view_index=d['view_index'], global_view=d['view_index']==0, target_visible=bool(boxes),
                valid_description=valid, predicted_bin=d['location'], compatible=bool(valid and d['location'] in bins),
                center_match=bool(valid and d['location'] in centers),
                target_bins=sorted(bins), target_center_bins=sorted(centers), fields=d['fields']))
        for model in NAMES:
            for base in ['A', 'B']:
                tag = base + '__' + model; s = r['scores'][tag]; info = calib['models'][tag]
                g = location_residual(p, c['image_size'], loc[sid])
                center_g = location_residual(p, c['image_size'], 4 if loc[sid] is not None else None)
                variants = [(base, choose(s)), (base+'_position', choose(s, g, info['position_strength'])),
                            (base+'_center', choose(s, center_g, info['center_strength']))]
                for arm, j in variants:
                    rows.append(dict(sample_id=sid, model=model, arm=arm, index=j,
                        point=p[j].tolist() if j is not None else None, hit=bool(j is not None and truth[j])))
                shuffled_hits = []; shuffled_indices = []
                for shuffled, _ in shuffles:
                    gg = location_residual(p, c['image_size'], shuffled[sid]); j = choose(s, gg, info['position_strength'])
                    shuffled_hits.append(bool(j is not None and truth[j])); shuffled_indices.append(j)
                rows.append(dict(sample_id=sid, model=model, arm=base+'_shuffled',
                                 hit=float(np.mean(shuffled_hits)), draw_hits=shuffled_hits, draw_indices=shuffled_indices))
                for arm, allowed in [('oracle_bbox_bins', oracle_allowed), ('predicted_hard_bin', cells == loc[sid] if loc[sid] is not None else np.ones(len(p), bool))]:
                    ix = np.flatnonzero(allowed); j = int(ix[s[ix].argmax()]) if len(ix) else choose(s)
                    oracle.append(dict(sample_id=sid, model=model, base=base, arm=arm, index=j,
                        hit=bool(j is not None and truth[j]), allowed_candidates=int(len(ix)),
                        retained_true=int(truth[allowed].sum()), label_informed=arm.startswith('oracle')))
    write(OUT / 'predictions.json', dict(rows=rows, diagnostic_oracles=oracle, spatial=spatial, coverage=coverage))
    status('development_predictions_complete', rows=len(rows), label_informed_diagnostic_rows=len(oracle))


def analyze(cases):
    pred = read(OUT / 'predictions.json'); rows = pred['rows']
    lookup = {(r['sample_id'],r['model'],r['arm']):r for r in rows}; results = {}
    arms = ['A','B','A_position','B_position','A_center','B_center','A_shuffled','B_shuffled']
    for cohort in ['new_development', 'old_val', 'all']:
        cc = [c for c in cases if cohort == 'all' or c['cohort'] == cohort]
        summary = {}; comparisons = {}; diagnostics = {}
        for family in ['original3k', 'full']:
            arrays = {}
            summary[family] = {}
            for arm in arms:
                v = np.asarray([[lookup[c['sample_id'],f'{family}_s{s}',arm]['hit'] for c in cc] for s in [1,2,3]], float)
                arrays[arm] = v
                summary[family][arm] = dict(accuracy_percent=100*float(v.mean()), seed_hits=v.sum(1).tolist(),
                    by_source={src:100*float(v[:,[i for i,c in enumerate(cc) if c['source']==src]].mean()) for src in sorted({c['source'] for c in cc})})
            for a,b in CONTRASTS:
                comparisons[family+'/'+a+'-'+b] = bootstrap(arrays[a]-arrays[b],cc,1-.05/16 if cohort=='new_development' else .95)
            for base in ['A','B']:
                for arm in ['oracle_bbox_bins','predicted_hard_bin']:
                    rr = [r for r in pred['diagnostic_oracles'] if r['base']==base and r['arm']==arm and r['model'].startswith(family+'_') and r['sample_id'] in {c['sample_id'] for c in cc}]
                    diagnostics[family+'/'+base+'/'+arm] = 100*float(np.mean([r['hit'] for r in rr]))
        spatial = {}
        for mode in ['global','local']:
            rr = [r for r in pred['spatial'] if r['sample_id'] in {c['sample_id'] for c in cc} and r['global_view']==(mode=='global')]
            valid = [r for r in rr if r['valid_description']]; visible = [r for r in valid if r['target_visible']]
            spatial[mode] = dict(total_views=len(rr), valid=len(valid), valid_target_visible=len(visible),
                valid_target_absent=len(valid)-len(visible), compatible=sum(r['compatible'] for r in visible),
                center_match=sum(r['center_match'] for r in visible), independent_unit='RGB, not view')
        cvg = [r for r in pred['coverage'] if r['sample_id'] in {c['sample_id'] for c in cc}]
        results[cohort] = dict(n=len(cc),methods=summary,comparisons=comparisons,diagnostics=diagnostics,
            spatial=spatial,covered=sum(r['covered'] for r in cvg),mean_candidates=float(np.mean([r['candidate_count'] for r in cvg])))
    gates = {}
    for family in ['original3k','full']:
        for base in ['A','B']:
            cmp = results['new_development']['comparisons']; tag=family+'/'+base+'_position'
            required=[tag+'-'+base,tag+'-'+base+'_shuffled',tag+'-'+base+'_center']
            if base=='B': required.append(tag+'-A')
            old = results['old_val']['methods'][family]
            gates[family+'/'+base] = (cmp[tag+'-'+base]['delta_pp']>=3 and all(cmp[k]['ci_pp'][0]>0 for k in required)
                and old[base+'_position']['accuracy_percent']-old['A']['accuracy_percent']>=-5)
    result=dict(complete=True,cohorts=results,positive_gates=gates,new_vlm_calls=0,
                independent_test=False,mainline_unchanged=True)
    write(OUT/'results.json',result)
    lines=['V26 固定原候选：结构化名称与显式位置软约束。',
        '56原训练RGB拟合单一位置权重；36历史复用开发RGB（新24/旧12），不是新独立测试。',
        'A/B及所有软约束组候选完全一致。原6评分器冻结，0次新VLM调用。',
        'B只读新描述的名称token；位置从完整全图描述解析，未知/WAIT回退原分数。',
        '打乱对照为100次来源/队列内置换均值，同真实位置权重；中心对照单独在训练集校准。']
    for cohort,r in results.items():
        lines+=['',f'{cohort} n={r["n"]},候选覆盖{r["covered"]}/{r["n"]},平均候选{r["mean_candidates"]:.1f}',str(r['spatial'])]
        for family,mm in r['methods'].items():
            for arm,m in mm.items(): lines.append(f'{family}/{arm}: {m["accuracy_percent"]:.2f}%,命中{m["seed_hits"]}')
        for key,v in r['comparisons'].items(): lines.append(f'{key}: {v["delta_pp"]:+.2f}pp, CI{v["interval_percent"]:.4f}% {v["ci_pp"]}')
        lines.append('诊断GT九宫格/生成位置硬筛选（非可部署方法）：'+str(r['diagnostics']))
    lines+=['','预登记进阶门槛：'+str(gates),'GT约束仅诊断，未参与部署预测或位置权重校准。无独立泛化/MVP优势结论。']
    (OUT/'results.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    status('analysis_complete',positive_gates=gates)


def audit(cases):
    verify(); hashes=read(OUT/'input_cache_sha256.json')
    for p,h in hashes.items(): assert sha(p)==h,p
    pred=read(OUT/'predictions.json'); labels=read(V25/'data/labels.json'); cal=read(OUT/'calibration.json')
    devscore=read(OUT/'development_scoring.json')
    assert datetime.fromisoformat(cal['frozen_at_utc']) < datetime.fromisoformat(devscore['at_utc'])
    assert set(cal['training_sample_ids']) == {c['sample_id'] for c in cases if c['cohort']=='train'}
    assert not set(cal['training_sample_ids']) & {r['sample_id'] for r in pred['rows']}
    baseline=read(V25/'predictions.json')['rows']; previous={(r['sample_id'],r['model']):r for r in baseline if r['regime']=='frozen' and r['arm']=='name'}
    parity=0; checks=0
    dev=[c for c in cases if c['cohort']!='train']; loc=read(OUT/'locations.json')
    loc={c['sample_id']:loc[c['sample_id']] for c in dev}; recs=records(dev)
    lookup={(r['sample_id'],r['model'],r['arm']):r for r in pred['rows']}
    for c in cases:
        a,_=prior.load_npz(prior.feature_path(c,'name')); b,_=prior.load_npz(feature_path(c))
        for k in ['points','base','active','baseline']: assert np.array_equal(a[k],b[k]),(c['sample_id'],k)
    for c in dev:
        sid=c['sample_id']; r=recs[sid];truth=labels_at(c,r['points'],labels)
        for model in NAMES:
            assert lookup[sid,model,'A']['point']==previous[sid,model]['point']
            assert lookup[sid,model,'A']['hit']==previous[sid,model]['hit'];parity+=1
            for base in ['A','B']:
                tag=base+'__'+model;s=r['scores'][tag];p=r['points'];weights=cal['models'][tag]
                for suffix,idx,w in [('',None,0),('_position',loc[sid],weights['position_strength']),
                    ('_center',4 if loc[sid] is not None else None,weights['center_strength'])]:
                    j=choose(s,location_residual(p,c['image_size'],idx),w)
                    row=lookup[sid,model,base+suffix]
                    assert j==row['index'] and bool(j is not None and truth[j])==row['hit'];checks+=1
                row=lookup[sid,model,base+'_shuffled']
                for i in range(SHUFFLES):
                    sl,_=shuffled_locations(dev,loc,i);j=choose(s,location_residual(p,c['image_size'],sl[sid]),weights['position_strength'])
                    assert j==row['draw_indices'][i] and bool(j is not None and truth[j])==row['draw_hits'][i];checks+=1
    before=read(OUT/'registry_before.json');reg=read(ROOT/'CURRENT_RESEARCH.json')
    assert reg['run_id']==before['run_id'] and reg['models']==before['models']
    write(OUT/'final_audit.json',dict(complete=True,completed_at_utc=now(),fixed_pool_cases=len(cases),
        baseline_parity=parity,decision_replays=checks,historical_cache_hashes_verified=len(hashes),
        original_models_unchanged=True,calibration_training_only=True,new_vlm_calls=0))
    paths=[OUT/n for n in ['protocol.json','results.json','results.txt','predictions.json','calibration.json',
        'final_audit.json','input_cache_sha256.json','locations.json','descriptions.json','shuffle_donors.json']]
    artifacts={str(p):sha(p) for p in paths}
    artifacts.update({str(p):sha(p) for sub in ['features','scores'] for p in (OUT/sub).glob('*.npz')})
    write(OUT/'completion.json',dict(complete=True,completed_at_utc=now(),artifact_sha256=artifacts,
        original_mainline_unchanged=True,new_vlm_calls=0))
    for p,h in artifacts.items(): assert sha(p)==h
    reg['active_experiment'].update(status='complete',phase='calibration_analysis_and_audit_complete',
        results_path=str(OUT/'results.txt'),completion_path=str(OUT/'completion.json'),
        audit_path=str(OUT/'final_audit.json'),positive_gates=read(OUT/'results.json')['positive_gates'])
    reg['completed_experiments'].append(dict(run_id=OUT.name,status='complete',analysis_path=str(OUT/'results.txt'),mainline_unchanged=True))
    write(ROOT/'CURRENT_RESEARCH.json',reg)
    (OUT/'SESSION_STATE.txt').write_text('V26全部执行、分析、固定候选与决策重放审计完成。0次新VLM调用。原权重与主线不变。\n',encoding='utf-8')
    status('complete',baseline_parity=parity,decision_replays=checks,positive_gates=read(OUT/'results.json')['positive_gates'])


def main():
    torch.set_num_threads(4); started=time.perf_counter();prepare();verify()
    cases=read(V25/'data/public.json');train=[c for c in cases if c['cohort']=='train'];dev=[c for c in cases if c['cohort']!='train']
    build(cases);score(train,'train');calibrate(train);score(dev,'development');evaluate(dev);analyze(dev);audit(cases)
    print(f'V26 finished, seconds={time.perf_counter()-started:.1f}',flush=True)


if __name__=='__main__':
    try:main()
    except BaseException:
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/'failure.txt').write_text(traceback.format_exc(),encoding='utf-8');status('failed');raise
