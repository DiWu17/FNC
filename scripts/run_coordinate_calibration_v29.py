"""Sequential cached V29 experiment. No HTML, VLM calls, or OS actions."""
import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']:
    os.environ.setdefault(k, '4')
import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import numpy as np
from clickagent.coordinate_calibration_v29 import features, fit_ranker, rank_scores, fit_gate, choose, X_NAMES, Q_NAMES, GATE_NAMES, MODES
from clickagent.public_evaluation import point_in_geometry

OUT = ROOT/'runs/coordinate_calibration_v29'
PRIOR = ROOT/'runs/grounded_composition_v28'
NAMES = [f'{family}_s{s}' for family in ['original3k', 'full'] for s in [1, 2, 3]]
SOURCES = [Path(__file__), ROOT/'src/clickagent/coordinate_calibration_v29.py', ROOT/'tests/test_coordinate_calibration_v29.py', ROOT/'src/clickagent/public_evaluation.py']
CONTRASTS = [('temperature', 'fixed_prior'), ('source_calibrated', 'temperature'),
             ('gate_conservative', 'source_calibrated'), ('gate_conservative', 'free_then_A'),
             ('gate_conservative', 'consensus'), ('gate_conservative', 'v28_geometry'), ('gate_conservative', 'A')]
ARMS = ['A', 'native_free', 'free_then_A', 'consensus', 'fixed_prior', 'temperature',
        'source_calibrated', 'keep_all_free', 'gate_equal', 'gate_conservative', 'v28_combined', 'v28_geometry']


def now(): return datetime.now(timezone.utc).isoformat()
def key(s): return hashlib.sha256(s.encode()).hexdigest()[:24]
def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8*1024*1024), b''): h.update(b)
    return h.hexdigest()
def write(p, x):
    p = Path(p);p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(x, ensure_ascii=False, indent=2)+'\n', encoding='utf-8');os.replace(tmp, p)
def status(phase, **kw):
    d = dict(phase=phase, updated_at_utc=now(), complete=phase == 'complete', **kw)
    write(OUT/'status.json', d);print(json.dumps(d, ensure_ascii=False), flush=True)
def npz(p, **arrays):
    p = Path(p);p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('wb') as f: np.savez_compressed(f, **arrays)
def input_path(c, part): return PRIOR/part/(key(c['sample_id'])+'.npz')
def feature_path(c): return OUT/'features'/(key(c['sample_id'])+'.npz')
def verify():
    p = read(OUT/'protocol.json')
    for f, h in p['source_sha256'].items(): assert sha(f) == h, f
    for f, h in p['sealed_sha256'].items(): assert sha(f) == h, f
    return p
def hit(point, c, geometry):
    return bool(point is not None and point_in_geometry([p/s for p, s in zip(point, c['image_size'])], geometry))


def prepare():
    assert not (OUT/'protocol.json').exists(), 'Do not overwrite a registered run'
    OUT.mkdir(parents=True, exist_ok=True)
    check = subprocess.run([sys.executable, '-B', '-X', 'utf8', '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_coordinate_calibration_v29.py', '-v'], cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    write(OUT/'unit_tests.json', dict(exit_code=check.returncode, output=check.stdout+check.stderr));assert check.returncode == 0
    cases = read(PRIOR/'data/public.json');train = [c for c in cases if c['cohort'] == 'train']
    assert len(train) == 112 and len({c['group'] for c in cases}) == 212
    delivery = read(PRIOR/'delivery_manifest.json')['artifact_sha256']
    inputs = {}
    for c in cases:
        for part in ['original', 'grounded', 'raw/global']:
            f = input_path(c, part);inputs[str(f)] = delivery[str(f)]
    write(OUT/'input_index.json', inputs);write(OUT/'data/public.json', cases)
    old_models = read(PRIOR/'protocol.json')['models']
    sealed = [PRIOR/'delivery_manifest.json', PRIOR/'interpretation_manifest.json', PRIOR/'data/public.json', PRIOR/'data/labels.json',
              OUT/'input_index.json', OUT/'data/public.json']
    sealed += [Path(r['path']) for r in old_models.values()]
    spec = dict(registered_at_utc=now(), source_sha256={str(f): sha(f) for f in SOURCES}, sealed_sha256={str(f): sha(f) for f in sealed},
                training_ids=[c['sample_id'] for c in train], old_models=old_models,
                data='ExactlyV28 training112(uniqueRGB,7sources), reuseddevelopment100(new72,old28). SameA32+3pool, samecoordinates/evidence. No fresh/independent test; 3k/full denotes oldscoringhead only and sharedA32dependsall6.',
                features=dict(candidate=X_NAMES, query=Q_NAMES, gate=GATE_NAMES,
                    description='Relative native-point geometry, provenance, valid-output/name-overlap/crop-edge evidence, A-onlymargin/entropy. Origin-times-query interactions. No absolute coordinates, source/image IDs, full lexical names/instruction embeddings, GT, new VLM scores, or new VLM calls.'),
                calibration='All112. Same66features/pairloss/native-positive-priority/top4positive/nearest8+highest8negative/query-sourcebalance/ridge.3. fixed_prior: oldZ coefficient1 and legacy native-medianfill. temperature: same but alpha learned[0,10], ridge(alpha-1)^2/2. source_calibrated: oldZ computedONLYonA, missingnewscorezero, alpha learned; originbiases explicit. This tests ranking-score calibration, not calibrated per-candidate correctness probabilities.',
                retention='Exact free-native coordinate is default. Rival=maxsourcecalibratedcandidate EXCLUDING exactfree. Gate predicts rival-better conditional exactlyonecorrect, fitONLYon training discordant free/rival pairs whose rival was selected byRGB3foldOOFrankers. Same ridge.3 proper weightedBCE, no class rebalance. Switch strictlyP>.75 primary, P>.5 secondary. Missingfree fallsback tosourcecalibratedbest; keep_all_free controls thatfallback. No coordinate averaging/snapping/movement.',
                cross_validation='Nested source_group3fold and leave-source-out7. Each outer gate trains on inner3fold predictions restricted to outer-trainingONLY. Feature scaling and all fitted weights exclude held-out groups. Old6heads already saw originaltraining; CV isolates NEWcalibration/gate only, not whole-system unseen-source generalization. source_group granularity varies app/site/session/image.',
                hyperparameters='Fixedridge.3, no dev hyperparameter/threshold selection. Gate threshold.75 primary,.5 exploratory; all heads reported, no seed selection/ensemble. Full models/gates frozen before ANYV29developmentfeature/evaluationpass.',
                baselines='A,V28combined,V28geometry_combined; nativefree; free_then_A; sameV25complete-link14pxconsensus(global/free/locked,firstviewtie,Afallback); keep_all_free(sourcefallback). Originalparity600points; consensusparity600points.',
                primary_contrasts=CONTRASTS,
                statistics='14primarycontrasts across2headfamilies,5000source-stratifiedRGBclusterbootstrap,99.642857%intervals(new72); originalsource_groupcluster sensitivity;old28descriptive95%.3fixed-fitmean,notindependentseeds. Extra gate_equal/sourcecontrols descriptive95%,notadditional confirmatoryclaims. Gate logs preserve every switch/repair/harm and available-free subset as descriptive only.',
                user_authorization='记录一下现在的方法到文档里，然后解决旧先验与新候选的分数校准，以及何时保留原始局部坐标', new_vlm_calls=0, mainline_unchanged=True)
    write(OUT/'registry_before.json', read(ROOT/'CURRENT_RESEARCH.json'));write(OUT/'protocol.json', spec)
    for f in SOURCES+[ROOT/'docs/CURRENT_METHOD.md']:
        dest = OUT/'source_archive'/f.relative_to(ROOT);dest.parent.mkdir(parents=True, exist_ok=True);shutil.copyfile(f, dest)
    reg = read(ROOT/'CURRENT_RESEARCH.json');reg['active_experiment'] = dict(run_id=OUT.name, status='running', phase='registered', protocol_path=str(OUT/'protocol.json'), status_path=str(OUT/'status.json'), mainline_unchanged=True)
    write(ROOT/'CURRENT_RESEARCH.json', reg)
    write(ROOT/'research/decisions/20261110_coordinate_calibration_v29.json', dict(registered_at_utc=now(), protocol_sha256=sha(OUT/'protocol.json'), user_authorization=spec['user_authorization'], mainline_unchanged=True))
    status('registered', train_rgb=112, development_rgb=100, new_vlm_calls=0)


def build(cases):
    index = read(OUT/'input_index.json');used = {}
    for i, c in enumerate(cases, 1):
        parts = {}
        for part in ['original', 'grounded', 'raw/global']:
            p = input_path(c, part);assert sha(p) == index[str(p)], p;used[str(p)] = index[str(p)]
            with np.load(p, allow_pickle=False) as z:
                parts[part] = dict(metadata=json.loads(str(z['metadata'])))
                for field in (['points', 'scores'] if part in ['original', 'grounded'] else []): parts[part][field] = z[field].copy()
        a, g = parts['original'], parts['grounded'];m = g['metadata']
        assert m['sample_id'] == c['sample_id'] and not m['labels_used']
        crop = parts['raw/global']['metadata']['planned_local_transform']['box']
        names = {p: m[p+'_target'] for p in ['global', 'free', 'locked']}
        rr = [features(g['points'], len(a['points']), score, m['native_points'], names, crop, c['image_size']) for score in g['scores']]
        native_index = np.asarray([rr[0]['native_index'][p] if rr[0]['native_index'][p] is not None else -1 for p in ['global', 'free', 'locked']])
        npz(feature_path(c), x=np.stack([r['x'] for r in rr]), q=np.stack([r['q'] for r in rr]),
            base=np.stack([r['base'] for r in rr]), source_base=np.stack([r['source_base'] for r in rr]),
            points=g['points'], origin=rr[0]['origin'], native_index=native_index,
            metadata=np.asarray(json.dumps(dict(sample_id=c['sample_id'], old_count=len(a['points']), labels_used=False, input_sha256={str(input_path(c, p)): index[str(input_path(c, p))] for p in parts}))))
        if i % 28 == 0 or i == len(cases): status('building_'+c['cohort'], completed=i, total=len(cases))
    write(OUT/('training_inputs_verified.json' if cases[0]['cohort'] == 'train' else 'development_inputs_verified.json'), used)


def records(cases, name, labels=None):
    si = NAMES.index(name);rr = []
    for c in cases:
        with np.load(feature_path(c), allow_pickle=False) as z:
            m = json.loads(str(z['metadata']));assert not m['labels_used'] and m['sample_id'] == c['sample_id']
            r = {k: z[k][si].copy() for k in ['x', 'q', 'base', 'source_base']}
            r.update(points=z['points'].copy(), origin=z['origin'].copy(), old_count=m['old_count'],
                     native_index={p: (int(z['native_index'][i]) if z['native_index'][i] >= 0 else None) for i, p in enumerate(['global', 'free', 'locked'])})
        r.update(sample_id=c['sample_id'], source=c['source'], group=c['group'], source_group=str(c['source_group']), size=np.asarray(c['image_size']))
        if labels is not None: r['y'] = np.asarray([hit(p.tolist(), c, labels[c['sample_id']]) for p in r['points']], bool)
        rr.append(r)
    return rr


def folds(rr, salt):
    assignments = {}
    for source in sorted({r['source'] for r in rr}):
        gg = sorted({r['source_group'] for r in rr if r['source'] == source}, key=lambda g: key(source+'|'+g+'|'+salt))
        assignments.update({(source, g): j % 3 for j, g in enumerate(gg)})
    return {r['sample_id']: assignments[r['source'], r['source_group']] for r in rr}


def consensus(r):
    groups = []
    for part in ['global', 'free', 'locked']:
        j = r['native_index'][part]
        if j is None: continue
        p = r['points'][j]
        group = next((g for g in groups if all(np.max(np.abs(p-r['points'][q])) <= 14 for q in g)), None)
        if group is None: groups.append([j])
        else: group.append(j)
    return max(groups, key=len)[0] if groups else old_top(r)


def old_top(r): return int(np.argmax(r['base'][:r['old_count']])) if r['old_count'] else None
def best(scores): return int(np.argmax(scores)) if len(scores) else None


def policies(r, rankers, gate):
    ss = {mode: rank_scores(rankers[mode], r) for mode in MODES};f = r['native_index']['free']
    answer = dict(A=(old_top(r), {}), native_free=(f, {}), free_then_A=(f if f is not None else old_top(r), {}),
                  consensus=(consensus(r), {}), fixed_prior=(best(ss['fixed']), {}), temperature=(best(ss['temperature']), {}),
                  source_calibrated=(best(ss['source']), {}), keep_all_free=choose(r, ss['source']))
    for arm, threshold in [('gate_equal', .5), ('gate_conservative', .75)]: answer[arm] = choose(r, ss['source'], gate, threshold)
    return answer


def save_model(path, model):
    arrays = {k: v for k, v in model.items() if isinstance(v, np.ndarray)}
    meta = {k: v for k, v in model.items() if not isinstance(v, np.ndarray)}
    npz(path, **arrays, metadata=np.asarray(json.dumps(meta)))
    return dict(path=str(path), sha256=sha(path), optimizer=meta)
def load_model(row):
    assert sha(row['path']) == row['sha256']
    with np.load(row['path'], allow_pickle=False) as z:
        m = json.loads(str(z['metadata']));m.update({k: z[k].copy() for k in z.files if k != 'metadata'})
    return m


def train(cases):
    verify();assert all(c['cohort'] == 'train' for c in cases)
    labels = read(PRIOR/'data/labels.json');frozen = {};diagnostics = [];fit_audit = [];steps = 0
    for name in NAMES:
        rr = records(cases, name, labels);rgb_fold = folds(rr, 'v29_outer');full_gate_oof = {}
        for mode in ['RGB', 'source']:
            exclusions = range(3) if mode == 'RGB' else sorted({r['source'] for r in rr})
            for exclusion in exclusions:
                held = lambda r: rgb_fold[r['sample_id']] == exclusion if mode == 'RGB' else r['source'] == exclusion
                fitting, testing = [r for r in rr if not held(r)], [r for r in rr if held(r)]
                assert not {r['sample_id'] for r in fitting} & {r['sample_id'] for r in testing}
                assert not {(r['source'], r['source_group']) for r in fitting} & {(r['source'], r['source_group']) for r in testing}
                models = {m: fit_ranker(fitting, m) for m in MODES};inner_fold = folds(fitting, 'v29_inner');inner_oof = {}
                inner_audits = []
                for inner in range(3):
                    tr = [r for r in fitting if inner_fold[r['sample_id']] != inner];va = [r for r in fitting if inner_fold[r['sample_id']] == inner]
                    if not va: continue
                    assert tr
                    model = fit_ranker(tr, 'source')
                    inner_oof.update({r['sample_id']: rank_scores(model, r) for r in va})
                    inner_audits.append(dict(training_ids=[r['sample_id'] for r in tr], validation_ids=[r['sample_id'] for r in va]))
                assert set(inner_oof) == {r['sample_id'] for r in fitting}
                gate = fit_gate(fitting, inner_oof)
                for r in testing:
                    if mode == 'RGB': full_gate_oof[r['sample_id']] = rank_scores(models['source'], r)
                    for arm, (j, detail) in policies(r, models, gate).items():
                        diagnostics.append(dict(sample_id=r['sample_id'], old_head=name, cv_mode=mode, fold=str(exclusion), arm=arm,
                                                hit=bool(j is not None and r['y'][j]), chosen=j, detail=detail))
                fit_audit.append(dict(old_head=name, mode=mode, exclusion=str(exclusion), training_ids=[r['sample_id'] for r in fitting],
                                      validation_ids=[r['sample_id'] for r in testing], inner_splits=inner_audits,
                                      prior_coefficients={m: float(v['prior_coef']) for m, v in models.items()}, gate_discordant_queries=gate['discordant_queries']))
                steps += 1;status('nested_training_validation', old_head=name, mode=mode, completed=steps, total=60)
        assert set(full_gate_oof) == {r['sample_id'] for r in rr}
        for mode in MODES:
            model = fit_ranker(rr, mode);frozen[name+'__'+mode] = save_model(OUT/'models'/f'{name}__{mode}.npz', model)
        gate = fit_gate(rr, full_gate_oof);frozen[name+'__gate'] = save_model(OUT/'models'/f'{name}__gate.npz', gate)
    write(OUT/'training_nested_predictions.json', diagnostics);write(OUT/'training_fold_audit.json', fit_audit)
    write(OUT/'frozen.json', dict(complete=True, frozen_at_utc=now(), models=frozen, training_ids=[c['sample_id'] for c in cases],
                                training_feature_sha256={str(feature_path(c)): sha(feature_path(c)) for c in cases}, development_used=False,
                                source_sha256=verify()['source_sha256']))
    status('models_frozen', rankers=18, retention_gates=6)


def evaluate(cases):
    spec = verify();frozen = read(OUT/'frozen.json');assert frozen['complete'] and not frozen['development_used']
    models = {tag: load_model(row) for tag, row in frozen['models'].items()}
    labels = read(PRIOR/'data/labels.json');rows = [];public = {c['sample_id']: c for c in cases}
    v28 = {(r['sample_id'], r['old_head'], r['arm']): r for r in read(PRIOR/'predictions_stage2.json')['rows']}
    geometry = {(r['sample_id'], r['old_head'], r['arm']): r for r in read(PRIOR/'component_controls/predictions.json')['rows']}
    old_consensus = {(r['sample_id'], r['old_head']): r for r in read(PRIOR/'nonlearned_references.json')['rows'] if r['arm'] == 'native_consensus_then_A'}
    parity = 0
    for name in NAMES:
        rankers = {m: models[name+'__'+m] for m in MODES};gate = models[name+'__gate']
        for r in records(cases, name):
            assert 'y' not in r
            c = public[r['sample_id']]
            for arm, (j, detail) in policies(r, rankers, gate).items():
                p = r['points'][j].tolist() if j is not None else None
                if arm == 'A': assert p == v28[c['sample_id'], name, 'A']['point'];parity += 1
                if arm == 'consensus': assert p == old_consensus[c['sample_id'], name]['point'];parity += 1
                if detail.get('reason') == 'keep_exact_free': assert j == r['native_index']['free']
                rows.append(dict(sample_id=c['sample_id'], old_head=name, arm=arm, point=p, hit=hit(p, c, labels[c['sample_id']]), detail=detail))
            for arm, prior in [('v28_combined', v28[c['sample_id'], name, 'combined']), ('v28_geometry', geometry[c['sample_id'], name, 'geometry_combined'])]:
                assert hit(prior['point'], c, labels[c['sample_id']]) == prior['hit']
                rows.append(dict(sample_id=c['sample_id'], old_head=name, arm=arm, point=prior['point'], hit=prior['hit'], detail={}))
    assert parity == 1200
    write(OUT/'predictions.json', dict(rows=rows, baseline_and_consensus_point_parity=parity, labels_used_in_inference=False))
    status('predictions_complete', rows=len(rows), parity=parity, models_frozen_at_utc=frozen['frozen_at_utc'])


def bootstrap_weights(cases, field='group'):
    strata = defaultdict(lambda: defaultdict(list))
    for i, c in enumerate(cases): strata[c['source']][str(c[field])].append(i)
    rng = np.random.default_rng(20262910);counts = np.zeros((5000, len(cases)))
    for groups in strata.values():
        gg = list(groups.values());draw = rng.integers(0, len(gg), size=(5000, len(gg)))
        for j, ix in enumerate(gg): counts[:, ix] = (draw == j).sum(1)[:, None]
    return counts/counts.sum(1, keepdims=True)


def interval(diff, matrix, level):
    v = np.asarray(diff).mean(0);draws = matrix@v;alpha = (1-level)/2
    return dict(delta_pp=100*float(v.mean()), ci_pp=(100*np.quantile(draws, [alpha, 1-alpha])).tolist(),
                interval_percent=100*level, seed_delta_pp=(100*np.asarray(diff).mean(1)).tolist())


def analyze():
    verify();pred = read(OUT/'predictions.json');cases = [c for c in read(OUT/'data/public.json') if c['cohort'] != 'train']
    lookup = {(r['sample_id'], r['old_head'], r['arm']): r for r in pred['rows']};results = {};sensitivities = {};switches = {}
    for co in ['new_development', 'old_val', 'all']:
        cc = [c for c in cases if co == 'all' or c['cohort'] == co];w = bootstrap_weights(cc);wg = bootstrap_weights(cc, 'source_group');summary = {};contrasts = {};secondary = {};sensitivity = {}
        for family in ['original3k', 'full']:
            arrays = {a: np.asarray([[lookup[c['sample_id'], f'{family}_s{s}', a]['hit'] for c in cc] for s in [1, 2, 3]], float) for a in ARMS}
            summary[family] = {a: dict(percent=100*float(v.mean()), seed_hits=v.sum(1).astype(int).tolist(), by_source={src: 100*float(v[:, [i for i,c in enumerate(cc) if c['source'] == src]].mean()) for src in sorted({c['source'] for c in cc})}) for a,v in arrays.items()}
            for a, b in CONTRASTS:
                tag = family+'/'+a+'-'+b;contrasts[tag] = interval(arrays[a]-arrays[b], w, 1-.05/14 if co == 'new_development' else .95)
                sensitivity[tag] = interval(arrays[a]-arrays[b], wg, 1-.05/14 if co == 'new_development' else .95)
            for a,b in [('source_calibrated', 'v28_geometry'), ('source_calibrated', 'free_then_A'), ('gate_equal', 'source_calibrated'), ('gate_equal', 'free_then_A'), ('gate_conservative', 'keep_all_free')]:
                secondary[family+'/'+a+'-'+b] = interval(arrays[a]-arrays[b], w, .95)
        results[co] = dict(n=len(cc), summary=summary, primary_contrasts=contrasts, exploratory_contrasts=secondary)
        sensitivities[co] = sensitivity
    for co in ['new_development', 'old_val']:
        cc = [c for c in cases if c['cohort'] == co];switches[co] = {}
        for name in NAMES:
            switches[co][name] = {}
            for arm in ['source_calibrated', 'gate_equal', 'gate_conservative']:
                counts = defaultdict(int)
                for c in cc:
                    free = lookup[c['sample_id'], name, 'native_free'];out = lookup[c['sample_id'], name, arm]
                    counts['n'] += 1
                    if free['point'] is None:
                        counts['free_absent'] += 1;counts['fallback_hits'] += out['hit'];continue
                    counts['free_available'] += 1
                    changed = out['point'] != free['point'];counts['changed_exact_coordinate'] += changed
                    counts['repairs_vs_free'] += not free['hit'] and out['hit'];counts['harms_vs_free'] += free['hit'] and not out['hit']
                    counts['free_available_final_hits'] += out['hit']
                switches[co][name][arm] = dict(counts)
    train_rows = read(OUT/'training_nested_predictions.json');training_summary = {}
    for cv in ['RGB', 'source']:
        training_summary[cv] = {family: {a: 100*float(np.mean([r['hit'] for r in train_rows if r['cv_mode'] == cv and r['old_head'].startswith(family) and r['arm'] == a])) for a in ARMS if not a.startswith('v28_')} for family in ['original3k', 'full']}
    write(OUT/'results.json', results);write(OUT/'source_group_sensitivity.json', sensitivities);write(OUT/'retention_diagnostics.json', switches);write(OUT/'training_cv_summary.json', training_summary)
    gates = {f: {a+'-'+b: results['new_development']['primary_contrasts'][f+'/'+a+'-'+b]['ci_pp'][0] > 0 for a,b in CONTRASTS} for f in ['original3k', 'full']}
    lines = ['V29：旧先验/新候选分数校准与原生局部坐标保留', '固定V28候选和观察；112训练、100复用开发；新增VLM调用0。所有模型/规则冻结后才做V29开发推理；没有独立测试或自动采用。', '', '准确率%，三个固定拟合均值；3k/full只指旧头，共用A32使用六个头。']
    for co in ['new_development', 'old_val']:
        lines += ['', co, 'arm | original3k | full']
        for a in ARMS: lines.append(a+' | '+' | '.join(f'{results[co]["summary"][f][a]["percent"]:.2f}' for f in ['original3k', 'full']))
    lines += ['', '原生局部坐标保留：有效自由点严格原坐标返回；只有OOF训练门控预测替换收益概率>.75才换点，>.5为单列探索对照。缺失自由点时走来源校准排序；keep_all_free单独隔离此回退收益。',
              '门控概率以free/rival恰一正确为条件，是小样本正则模型估计，不是每次点击正确率保证。', '', '主比较：14项校正区间，新72按RGB/来源分层。', json.dumps(results['new_development']['primary_contrasts'], ensure_ascii=False, indent=2),
              '', '各主比较正证据门槛：', json.dumps(gates, ensure_ascii=False, indent=2), '', '训练内嵌套分组/来源留出，仅隔离新增模块：', json.dumps(training_summary, ensure_ascii=False, indent=2),
              '', '逐次替换救回与损失见retention_diagnostics.json；旧先验系数及门控参数见frozen.json/models。source_group粒度依来源而异，不能统一称跨应用测试。']
    (OUT/'readout.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    write(OUT/'positive_gates.json', gates)
    status('analysis_complete')


def audit():
    spec = verify();frozen = read(OUT/'frozen.json');assert not frozen['development_used']
    for row in frozen['models'].values(): assert sha(row['path']) == row['sha256']
    for p,h in frozen['training_feature_sha256'].items(): assert sha(p) == h
    for cv in read(OUT/'training_fold_audit.json'):
        assert not set(cv['training_ids']) & set(cv['validation_ids'])
        for inner in cv['inner_splits']:
            assert not set(inner['training_ids']) & set(inner['validation_ids'])
            assert (set(inner['training_ids']) | set(inner['validation_ids'])) == set(cv['training_ids'])
    reg = read(ROOT/'CURRENT_RESEARCH.json');reg['active_experiment'].update(status='complete', phase='analyzed_and_audited',
            results_path=str(OUT/'readout.txt'), frozen_path=str(OUT/'frozen.json'), mainline_unchanged=True)
    write(ROOT/'CURRENT_RESEARCH.json', reg)
    write(OUT/'audit.json', dict(passed=True, at_utc=now(), nested_outer_splits=60, baseline_and_consensus_coordinate_parity=1200,
                               new_vlm_calls=0, old_model_sha256_unchanged=True, inference_has_no_labels=True,
                               original_native_coordinates_not_moved=True, mainline_unchanged=True))
    status('complete', stages=['calibration', 'exact_coordinate_retention'], mainline_unchanged=True)
    files = [p for p in OUT.rglob('*') if p.is_file() and p.name not in ['completion.json'] and not p.name.endswith('.log')]
    write(OUT/'completion.json', dict(complete=True, at_utc=now(), artifact_sha256={str(p): sha(p) for p in files}, source_sha256=spec['source_sha256']))
    print('ALL_V29_EXPERIMENTS_COMPLETE', flush=True)


def main():
    parser = argparse.ArgumentParser();parser.add_argument('phase', choices=['prepare', 'train', 'evaluate', 'all']);arg = parser.parse_args().phase
    if arg in ['prepare', 'all']: prepare()
    verify();cases = read(OUT/'data/public.json');tr = [c for c in cases if c['cohort'] == 'train'];dev = [c for c in cases if c['cohort'] != 'train']
    if arg in ['train', 'all']: build(tr);train(tr)
    if arg in ['evaluate', 'all']:
        assert read(OUT/'frozen.json')['complete'];build(dev);evaluate(dev);analyze();audit()


if __name__ == '__main__':
    try: main()
    except BaseException as e:
        status('failed', error=repr(e), traceback=traceback.format_exc());raise
