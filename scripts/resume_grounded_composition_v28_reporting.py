"""Resume only reporting after a duplicate sample_id keyword error.

Keep the sealed experiment sources, observations, pools, and weights unchanged.
"""
import hashlib
import inspect
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
import run_grounded_composition_v28 as run

OUT = run.OUT
AMENDMENT = OUT / 'amendments/03_reporting_duplicate_sample_id'


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    run.verify()
    AMENDMENT.mkdir(parents=True, exist_ok=False)
    for name in ['status.json', 'execution.json', 'finish.stderr.log']:
        shutil.copyfile(OUT / name, AMENDMENT / name)
    before = run.identity()
    original = inspect.getsource(run.evaluate)
    old = "diagnostics.append(dict(sample_id=c['sample_id'],**m))"
    new = "assert m['sample_id']==c['sample_id'];diagnostics.append(dict(m))"
    assert original.count(old) == 1
    repaired = original.replace(old, new)
    (AMENDMENT / 'evaluate_original.py').write_text(original, encoding='utf-8')
    (AMENDMENT / 'evaluate_reporting_repair.py').write_text(repaired, encoding='utf-8')
    run.write(AMENDMENT / 'record.json', dict(
        at_utc=now(), source_sha256=run.sha(__file__),
        reason='Grounded metadata already contains sample_id; dict(sample_id=..., **m) raises before output.',
        correction='Assert the two sample IDs agree, then copy metadata once. No scoring or labels change.',
        frozen_sources_unchanged=before, replacement_before=old, replacement_after=new,
        new_vlm_calls=0, retraining=False, scope='Evaluation diagnostic serialization only.'))
    exec(compile(repaired, str(AMENDMENT / 'evaluate_reporting_repair.py'), 'exec'), run.__dict__)
    run.status('resuming_reporting', amendment_path=str(AMENDMENT / 'record.json'))
    cases = run.read(OUT / 'data/public.json')
    dev = [c for c in cases if c['cohort'] != 'train']
    execution = run.read(OUT / 'execution.json')
    previous = execution['phases'][-1]
    assert previous['phase'] == 'finish' and previous['exit_code'] != 0
    execution['phases'] = execution['phases'][:-1]
    execution['recovery_record'] = str(AMENDMENT / 'record.json')
    run.write(OUT / 'execution.json', execution)
    started = now()
    print('RESUME_EVALUATION', started, flush=True)
    run.evaluate(dev, 2)
    run.analyze()
    assert run.identity() == before
    run.audit()
    execution['phases'].append(dict(phase='finish', started_at_utc=started,
                                    finished_at_utc=now(), exit_code=0,
                                    reporting_recovery=str(AMENDMENT / 'record.json')))
    execution.update(complete=True, finished_at_utc=now())
    run.write(OUT / 'execution.json', execution)
    receipt = ROOT / 'research/executions/v28_followups.json'
    followups = run.read(receipt)
    assert followups['phases'][0] == dict(part='training_transfer', exit_code=0)
    for script, part in [('analyze_grounded_composition_v28_mechanisms.py', 'mechanisms'),
                         ('finalize_grounded_composition_v28.py', 'finalize')]:
        print('START', part, flush=True)
        with (OUT / (part + '.stdout.log')).open('wb') as stdout, (OUT / (part + '.stderr.log')).open('wb') as stderr:
            result = subprocess.run([sys.executable, '-B', '-X', 'utf8', '-u', str(ROOT / 'scripts' / script)],
                                    cwd=ROOT, stdout=stdout, stderr=stderr)
        followups['phases'].append(dict(part=part, exit_code=result.returncode))
        run.write(receipt, followups)
        print('EXIT', part, result.returncode, flush=True)
        if result.returncode:
            sys.exit(result.returncode)
    followups['complete'] = True
    run.write(receipt, followups)
    print('ALL_THREE_STAGES_AND_FOLLOWUPS_COMPLETE', now(), flush=True)


if __name__ == '__main__':
    main()
