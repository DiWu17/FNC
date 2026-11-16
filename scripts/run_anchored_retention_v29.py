"""Separately registered all-rival comparator after train-only gate scarcity."""
import os
for k in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS']: os.environ.setdefault(k, '4')
import sys
import json
import shutil
import subprocess
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import numpy as np
import run_coordinate_calibration_v29 as run
from clickagent.anchored_retention_v29 import fit, choose
from clickagent.coordinate_calibration_v29 import rank_scores
OUT = run.OUT/'anchored_comparator'
ARMS = ['anchored_equal', 'anchored_conservative']
FILES = [Path(__file__), ROOT/'src/clickagent/anchored_retention_v29.py', ROOT/'tests/test_anchored_retention_v29.py']


def main():
    run.verify();assert run.read(run.OUT/'completion.json')['complete']
    assert not (OUT/'plan.json').exists(), 'Never overwrite a registered secondary'
    check = subprocess.run([sys.executable, '-B', '-X', 'utf8', '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_anchored_retention_v29.py', '-v'], cwd=ROOT, capture_output=True, text=True, encoding='utf-8')
    run.write(OUT/'unit_tests.json', dict(exit_code=check.returncode, output=check.stdout+check.stderr));assert check.returncode == 0
    plan = dict(registered_at_utc=run.now(), source_sha256={str(f): run.sha(f) for f in FILES},
                primary_completion_sha256=run.sha(run.OUT/'completion.json'), primary_frozen_sha256=run.sha(run.OUT/'frozen.json'),
                trigger='Full gate frozen.json reported only12-14TRAINING discordant top-rival queries per oldhead. PrimaryV29newdevelopment results/predictions were NOT inspected to design this secondary. Main experiment and all outputs remain unchanged.',
                training='Same112 trainingRGB. For validfree, compare it with ALL other candidates whose TRAIN correctness differs; query/source-balanced conditional logistic, no class rebalance. Relative66features + A-onlyoldZdifference, alpha[0,10], fixedridge.3 shrinkingalpha towards1, nativebias learned. No added points, no lexical/image/sourceID features. Labels never enter inference.',
                deployment='Score each rival against exactfree. Keep exactfree unless maximal predicted pairwise preference strictly>.75 (conservative) or>.5(equal). This maximized conditional probability is NOT a calibrated post-selection risk/correctness guarantee. Missingfree uses originalV29frozen source-calibratedranker.',
                validation='Same source_group3fold andsourceLOSO7. Comparator fit onlyoutertraining(no learned first-stage input); missingfree fallback uses primary nested-OOF sourcechoice from EXACTsameouter split. Full6comparators frozen before any secondary development prediction.',
                evaluation='Same reused100developmentRGB, mainnew72. Four contrasts x2headfamilies=>99.375%source-stratifiedRGBbootstrap,old28descriptive95%. Separately registered exploratory follow-up after train statistics, not independent/primary confirmation. No threshold/hyperparameter search or automatic adoption.',
                contrasts=[['anchored_equal', 'gate_equal'], ['anchored_conservative', 'gate_conservative'], ['anchored_conservative', 'free_then_A'], ['anchored_conservative', 'consensus']],
                new_vlm_calls=0, mainline_unchanged=True)
    run.write(OUT/'plan.json', plan)
    for f in FILES:
        p = OUT/'source_archive'/f.relative_to(ROOT);p.parent.mkdir(parents=True, exist_ok=True);shutil.copyfile(f, p)
    cases = run.read(run.OUT/'data/public.json');tr = [c for c in cases if c['cohort'] == 'train'];dev = [c for c in cases if c['cohort'] != 'train']
    labels = run.read(run.PRIOR/'data/labels.json');primary_oof = run.read(run.OUT/'training_nested_predictions.json')
    fallback = {(r['sample_id'], r['old_head'], r['cv_mode']): r for r in primary_oof if r['arm'] == 'source_calibrated'}
    frozen = {};diagnostics = [];audits = []
    for name in run.NAMES:
        rr = run.records(tr, name, labels);folds = run.folds(rr, 'v29_outer')
        for mode in ['RGB', 'source']:
            for exclusion in (range(3) if mode == 'RGB' else sorted({r['source'] for r in rr})):
                held = lambda r: folds[r['sample_id']] == exclusion if mode == 'RGB' else r['source'] == exclusion
                fitting, testing = [r for r in rr if not held(r)], [r for r in rr if held(r)]
                model = fit(fitting)
                for r in testing:
                    old = fallback[r['sample_id'], name, mode];assert old['fold'] == str(exclusion)
                    ss = np.zeros(len(r['points']))
                    if old['chosen'] is not None: ss[old['chosen']] = 1
                    for arm, threshold in zip(ARMS, [.5, .75]):
                        j, detail = choose(model, r, ss, threshold)
                        diagnostics.append(dict(sample_id=r['sample_id'], old_head=name, cv_mode=mode, arm=arm,
                                                hit=bool(j is not None and r['y'][j]), detail=detail))
                audits.append(dict(old_head=name, mode=mode, exclusion=str(exclusion), training_ids=[r['sample_id'] for r in fitting],
                                   validation_ids=[r['sample_id'] for r in testing], eligible_queries=model['eligible_queries'], pairs=model['pairs']))
        model = fit(rr);frozen[name] = run.save_model(OUT/'models'/f'{name}.npz', model)
        print('TRAINED', name, 'eligible_queries', model['eligible_queries'], 'pairs', model['pairs'], flush=True)
        run.write(OUT/'status.json', dict(phase='training', completed=len(frozen), total=6, at_utc=run.now()))
    run.write(OUT/'frozen.json', dict(complete=True, frozen_at_utc=run.now(), models=frozen, training_ids=[c['sample_id'] for c in tr], development_used=False))
    run.write(OUT/'training_predictions.json', diagnostics);run.write(OUT/'fold_audit.json', audits)
    primary_frozen = run.read(run.OUT/'frozen.json');rows = [];public = {c['sample_id']: c for c in dev}
    for name in run.NAMES:
        model = run.load_model(frozen[name]);source = run.load_model(primary_frozen['models'][name+'__source'])
        for r in run.records(dev, name):
            assert 'y' not in r
            scores = rank_scores(source, r)
            for arm, threshold in zip(ARMS, [.5, .75]):
                j, detail = choose(model, r, scores, threshold)
                if detail['reason'] == 'keep_exact_free': assert j == r['native_index']['free']
                p = r['points'][j].tolist() if j is not None else None;c = public[r['sample_id']]
                rows.append(dict(sample_id=r['sample_id'], old_head=name, arm=arm, point=p, hit=run.hit(p, c, labels[c['sample_id']]), detail=detail))
    run.write(OUT/'predictions.json', dict(rows=rows, inference_labels_used=False))
    primary = run.read(run.OUT/'predictions.json')['rows'];lookup = {(r['sample_id'],r['old_head'],r['arm']):r for r in rows+primary};results = {}
    for co in ['new_development', 'old_val']:
        cc = [c for c in dev if c['cohort'] == co];weights = run.bootstrap_weights(cc);summary = {};contrasts = {}
        for family in ['original3k', 'full']:
            arms = ARMS+['source_calibrated', 'keep_all_free', 'gate_equal', 'gate_conservative', 'free_then_A', 'consensus', 'v28_geometry']
            arrays = {a: np.asarray([[lookup[c['sample_id'],f'{family}_s{s}',a]['hit'] for c in cc] for s in [1,2,3]],float) for a in arms}
            summary[family] = {a:dict(percent=100*float(v.mean()),seed_hits=v.sum(1).astype(int).tolist()) for a,v in arrays.items()}
            for a,b in plan['contrasts']: contrasts[family+'/'+a+'-'+b] = run.interval(arrays[a]-arrays[b],weights,.99375 if co == 'new_development' else .95)
        results[co] = dict(summary=summary,contrasts=contrasts)
    training_summary = {mode:{family:{a:100*float(np.mean([r['hit'] for r in diagnostics if r['cv_mode']==mode and r['old_head'].startswith(family) and r['arm']==a])) for a in ARMS} for family in ['original3k','full']} for mode in ['RGB','source']}
    run.write(OUT/'results.json',results);run.write(OUT/'training_summary.json',training_summary)
    (OUT/'readout.txt').write_text('V29单独登记补充：以原生自由局部点为锚的所有候选比较。训练有效查询比原top-rival门控更多；不按开发成绩重选阈值或分支。\n'+json.dumps(dict(results=results,training=training_summary),ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    for a in audits: assert not set(a['training_ids']) & set(a['validation_ids'])
    for f,h in plan['source_sha256'].items(): assert run.sha(f)==h
    assert run.sha(run.OUT/'completion.json')==plan['primary_completion_sha256'] and run.sha(run.OUT/'frozen.json')==plan['primary_frozen_sha256']
    run.write(OUT/'status.json',dict(phase='complete',complete=True,at_utc=run.now()))
    run.write(OUT/'completion.json',dict(complete=True,artifact_sha256={str(p):run.sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='completion.json'}))
    print('ANCHORED_COMPARATOR_COMPLETE',flush=True)


if __name__=='__main__': main()
