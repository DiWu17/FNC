"""Serial GPU queue for the registered V17 follow-up experiments."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/advantage_validation_v17'
from report_advantage_validation_v17 import analyze

CONTROLLED = [('ours', 262144, 'all'), ('ours_s1', 262144, 'speed'),
              ('qwen', 262144, 'all'), ('ui_tars', 262144, 'all'),
              ('aima', 262144, 'all'), ('mvp_single', 262144, 'all'),
              ('mvp', 262144, 'all'), ('tag', 262144, 'all'), ('ours_gui', 262144, 'all')]
STRONGER = [('qwen', 1048576, 'stronger'), ('ui_tars', 1048576, 'stronger'),
            ('aima', 1048576, 'stronger'), ('aima_zoom', 1048576, 'stronger'),
            ('mvp', 8847360, 'stronger'), ('tag', 1806336, 'stronger')]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, path)


def status(phase, **fields):
    s = dict(phase=phase, complete=phase in ['complete', 'complete_with_resource_limits'],
             pid=os.getpid(), updated_at_utc=datetime.now(timezone.utc).isoformat(), **fields)
    write(OUT / 'status.json', s)
    print(json.dumps(s, ensure_ascii=False), flush=True)


def wait_for(path, phase):
    while not ready(path):
        data = read(OUT / 'data_efficiency/status.json')
        if data.get('phase') == 'failed':
            raise RuntimeError('Data-efficiency run failed; preserve its failure and repair before continuation')
        status(phase, waiting_for=str(path), data_efficiency_phase=data.get('phase'),
               data_efficiency_model=data.get('model'), data_efficiency_step=data.get('step'))
        time.sleep(30)


def ready(path):
    path = Path(path)
    try:
        return path.exists() and time.time() - path.stat().st_mtime >= 2 and read(path).get('complete') is True
    except (FileNotFoundError, json.JSONDecodeError, PermissionError):
        return False


def worker(method, pixels, stage):
    label = f'{method}_p{pixels}_{stage}'
    if stage == 'smoke':
        label += '_' + sha(ROOT / 'src/clickagent/advantage_runtimes.py')[:12]
        folder = OUT / 'setup/smoke' / label
        receipt = folder / 'smoke_receipt.json'
        if receipt.exists():
            d = read(receipt)
            if d['complete'] and d['adapter_sha256'] == sha(ROOT / 'src/clickagent/advantage_runtimes.py') and d['worker_sha256'] == sha(ROOT / 'scripts/run_advantage_inference_v17.py'):
                return True, None
    else:
        folder = OUT / 'inference' / label
        receipt = folder / 'completion.json'
        if receipt.exists():
            for name, expected in read(receipt)['hashes'].items():
                assert sha(name) == expected, name
            return True, None
    logs = OUT / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, OPENBLAS_NUM_THREADS='4', OMP_NUM_THREADS='4', PYTHONIOENCODING='utf-8',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1', PYTHONDONTWRITEBYTECODE='1')
    stdout_path, stderr_path = logs / f'{label}.stdout.log', logs / f'{label}.stderr.log'
    with stdout_path.open('a', encoding='utf-8') as stdout, stderr_path.open('a', encoding='utf-8') as stderr:
        proc = subprocess.Popen([sys.executable, '-B', '-X', 'utf8', '-u',
            str(ROOT / 'scripts/run_advantage_inference_v17.py'), '--method', method,
            '--pixels', str(pixels), '--stage', stage], cwd=ROOT, env=env, stdout=stdout, stderr=stderr)
        while proc.poll() is None:
            child = read(folder / 'status.json') if (folder / 'status.json').exists() else {}
            status('preflight' if stage == 'smoke' else 'inference', job=label,
                   child_pid=proc.pid, child_phase=child.get('phase'),
                   completed=child.get('completed'), total=child.get('total'))
            time.sleep(30)
    if proc.returncode == 0:
        assert receipt.exists()
        return True, None
    error = stderr_path.read_text(encoding='utf-8')[-12000:]
    child = read(folder / 'status.json') if (folder / 'status.json').exists() else {}
    error += '\n' + child.get('error', '')
    resource = any(k in error.lower() for k in ['cuda out of memory', 'cuda error: out of memory', 'not enough memory', 'allocation on device', 'paging file is too small'])
    failure = dict(job=label, exit_code=proc.returncode, resource_limit=resource,
                   error=error, stderr=str(stderr_path), no_accuracy_assigned=True)
    write(OUT / 'failures' / f'{label}_{int(time.time())}.json', failure)
    return False, failure


def freeze():
    p = OUT / 'experiments_protocol.json'
    files = [Path(__file__), ROOT / 'scripts/run_advantage_inference_v17.py',
             ROOT / 'src/clickagent/advantage_runtimes.py', ROOT / 'src/clickagent/advantage_validation.py',
             ROOT / 'scripts/report_advantage_validation_v17.py', OUT / 'plan.json',
             OUT / 'data/cohorts.json', OUT / 'data_efficiency/completion.json']
    if p.exists():
        for name, expected in read(p)['source_and_input_hashes'].items():
            assert sha(name) == expected, name
        return
    spec = dict(registered_at_utc=datetime.now(timezone.utc).isoformat(),
        source_and_input_hashes={str(q): sha(q) for q in files},
        controlled_jobs=CONTROLLED, stronger_jobs=STRONGER,
        primary_accuracy_family=['ours vs TAG', 'ours vs GUI-AIMA', 'ours vs MVP'],
        inference='Source-stratified screenshot bootstrap2000;98.333% family intervals for3 primary comparisons. Fixed models; reused screenshots.',
        speed='Same3 baseline family; ratio of paired per-screenshot means over2 repeats. Separate S1 accuracy in timed cohort.',
        robustness='2 perturbations times3 references;12 adjusted tests for absolute accuracy and differential change. Small-drop claims require absolute accuracy support too.',
        stronger='60 paired screenshots only; separate tables from the300 query controlled comparison.',
        resource_policy='Controlled arms must complete. Higher-budget resource failures are recorded as unavailable, never misses or wins.',
        mainline_unchanged=True)
    write(p, spec)


def complete(resource_failures):
    frozen = read(OUT / 'experiments_protocol.json')
    for name, expected in frozen['source_and_input_hashes'].items():
        assert sha(name) == expected, name
    artifacts = [OUT / 'plan.json', OUT / 'experiments_protocol.json', OUT / 'data_efficiency/completion.json']
    for method, pixels, stage in CONTROLLED:
        p = OUT / 'inference' / f'{method}_p{pixels}_{stage}' / 'completion.json'
        assert p.exists()
        for name, expected in read(p)['hashes'].items():
            assert sha(name) == expected, name
        artifacts.append(p)
    evidence = analyze()
    write(OUT / 'analysis/final_evidence.json', evidence)
    artifacts.append(OUT / 'analysis/final_evidence.json')
    write(OUT / 'completion.json', dict(complete=True, mainline_unchanged=True,
        completed_at_utc=datetime.now(timezone.utc).isoformat(),
        artifact_sha256={str(p): sha(p) for p in artifacts}, resource_failures=resource_failures,
        scope='Diagnostic follow-up; no new independent benchmark or automatic main-model selection.'))
    phase = 'complete_with_resource_limits' if resource_failures else 'complete'
    status(phase, controlled_jobs=len(CONTROLLED), stronger_jobs=len(STRONGER), resource_failures=len(resource_failures))
    current_path = ROOT / 'CURRENT_RESEARCH.json'
    current = read(current_path)
    if current.get('active_experiment', {}).get('run_id') == 'advantage_validation_v17':
        current['active_experiment'].pop('report_path', None)
        current['active_experiment'].update(status=phase, completion_path=str(OUT / 'completion.json'),
            completed_at_utc=read(OUT / 'completion.json')['completed_at_utc'],
            analysis_path=str(OUT / 'analysis/results.txt'), results_path=str(OUT / 'analysis/final_evidence.json'))
        write(current_path, current)


def main():
    resource_failures = []
    try:
        for path in [ROOT / 'models/GUI-AIMA-3B/clickagent_download_verified_v17.json']:
            wait_for(path, 'waiting_for_reference_weights')
        while not any(ready(ROOT / 'models' / name / 'clickagent_download_verified_v17.json')
                      for name in ['MiniCPM-Llama3-V-2_5', 'MiniCPM-Llama3-V-2_5_modelscope_staging']):
            status('waiting_for_reference_weights', waiting_for='TAG official hash verification')
            time.sleep(30)
        # Keep all scientific GPU measurements isolated from the curve-training job.
        wait_for(OUT / 'data_efficiency/completion.json', 'waiting_for_data_efficiency')
        for method in ['ours', 'ours_s1', 'qwen', 'ui_tars', 'aima', 'mvp_single', 'mvp', 'tag', 'ours_gui']:
            ok, failure = worker(method, 262144, 'smoke')
            if not ok:
                raise RuntimeError('Controlled adapter preflight failed: ' + json.dumps(failure, ensure_ascii=False))
        freeze()
        for method, pixels, stage in CONTROLLED:
            ok, failure = worker(method, pixels, stage)
            if not ok:
                raise RuntimeError('Controlled inference failed: ' + json.dumps(failure, ensure_ascii=False))
        for method, pixels, stage in STRONGER:
            ok, failure = worker(method, pixels, stage)
            if not ok:
                if not failure['resource_limit']:
                    raise RuntimeError('Higher-budget adapter failed: ' + json.dumps(failure, ensure_ascii=False))
                resource_failures.append(failure)
        complete(resource_failures)
    except Exception:
        status('failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
