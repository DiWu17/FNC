"""Prospective erratum: public-only ScreenSpot-Pro evaluation after V1 history failure.

The V1 reserve and Venus confirmations collided with an earlier held-out run.
This version preserves that failed attempt, uses the same frozen 39 models and
scientific feature/scoring engine, and permits only label-free extraction and
prediction sealing for the complete official benchmark. There is deliberately
no test-label analysis command here. A new independent cohort must be selected,
predicted and sealed under a separate prospective protocol before any full
benchmark labels are opened.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
import run_paper_scale_eval_v1 as engine

SOURCE_RUN = ROOT / 'runs/paper_scale_ablation_v1'
RUN = ROOT / 'runs/paper_scale_ablation_v2'
OUT = RUN / 'eval'
CORRECTION = SOURCE_RUN / 'data/exposure_correction_v2.json'
HISTORY = SOURCE_RUN / 'data/full_history_overlap_corrected_v2.json'
POST_V32 = SOURCE_RUN / 'data/post_v32_exposure_addendum_v1.json'
PUBLIC = SOURCE_RUN / 'data/full_public.json'
ERRATUM = ROOT / 'research/paper/SCALE_ABLATION_ERRATUM_V2.md'
V1_STATUS = SOURCE_RUN / 'eval/chain_status.json'
V1_PROTOCOL = SOURCE_RUN / 'eval/reserve/protocol.json'
V1_ERROR = SOURCE_RUN / 'eval/logs/chain/extract-reserve.stderr.log'
PROTOCOL = RUN / 'protocol.json'
STATUS = RUN / 'status.json'
LOGS = RUN / 'logs'
STAGES = ('register', 'feature-parity', 'archive-parity',
          'parity-old_val', 'parity-new_development',
          'prepare-full', 'extract-full', 'score-full')
EXPECTED_HISTORY = {
    'pre_v32_historical_exact', 'pre_v32_historical_near_le4',
    'v18_confirmation_observed_exact_rgb', 'v18_confirmation_observed_near_le4',
    'v32_diagnostic_observed', 'v32_reserve_previously_unforwarded',
}

# The V1 module remains byte-for-byte unchanged. Point only its output at the
# new run directory; its frozen model registry and archived scientific sources
# continue to come from V1/V16/V18. All calls pass through the V2 protocol gate.
engine.OUT = OUT
_engine_verify_public = engine.verify_public


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def old_failure():
    for path in (V1_STATUS, V1_PROTOCOL, V1_ERROR):
        require(path.exists(), 'V1 failure receipt missing: ' + str(path))
    status = read(V1_STATUS)
    require(status.get('phase') == 'failed' and 'extract-reserve' in status.get('error', ''),
            'V1 failure receipt has changed')
    for relative in ('eval/reserve/features/index.json', 'eval/reserve/predictions.json',
                     'eval/reserve/analysis.json'):
        require(not (SOURCE_RUN / relative).exists(),
                'V1 reserve unexpectedly proceeded past the pre-forward history guard')
    for relative in ('eval/reserve/features/cases', 'eval/reserve/features/arrays',
                     'eval/reserve/predictions/cases'):
        folder = SOURCE_RUN / relative
        require(not folder.exists() or not any(folder.iterdir()),
                'V1 reserve contains a held-out feature or prediction case')
    return {str(path): sha(path) for path in (V1_STATUS, V1_PROTOCOL, V1_ERROR)}


def corrected_history():
    for path in (CORRECTION, HISTORY, POST_V32, PUBLIC):
        require(path.exists(), 'Corrected public history audit not sealed: ' + str(path))
    correction, history, public = read(CORRECTION), read(HISTORY), read(PUBLIC)
    require(read(POST_V32).get('complete') is True,
            'Post-V32 exposure audit is incomplete')
    require(correction.get('complete') is True and history.get('complete') is True,
            'Corrected exposure audit incomplete')
    require(history.get('n') == 1581 and isinstance(history.get('rows'), list) and
            len(history['rows']) == len(public) == 1581,
            'Corrected full benchmark count differs')
    rows_bytes = (json.dumps(history['rows'], ensure_ascii=False, sort_keys=True, indent=2)
                  + '\n').encode('utf-8')
    require(hashlib.sha256(rows_bytes).hexdigest() == history.get('rows_sha256'),
            'Corrected history row seal differs')
    by_id = {r['sample_id']: r for r in history['rows']}
    require(len(by_id) == 1581 and set(by_id) == {r['sample_id'] for r in public},
            'Corrected history IDs differ from public benchmark')
    for case in public:
        row = by_id[case['sample_id']]
        require(row['pixel_sha256'] == case['pixel_sha256'] and
                row['history_status'] in EXPECTED_HISTORY and
                not any(key in row for key in ('target_geometry', 'labels', 'bbox', 'hits')) and
                'target_geometry' not in case and 'labels' not in case,
                'Corrected history/public identity or label separation failed')
    actual_counts = Counter(row['history_status'] for row in history['rows'])
    require(dict(actual_counts) == history.get('history_status_counts'),
            'Corrected history status counts differ from sealed rows')
    return correction, history, public


def register():
    v1_receipts = old_failure()
    correction, history, public = corrected_history()
    require(ERRATUM.exists(), 'Prospective prose erratum missing')
    old_audit = read(SOURCE_RUN / 'data/audit.json')
    official_sha = old_audit['source_sha256'][
        'data/public_v1/manifests/screenspot_pro.jsonl']
    expected = dict(
        complete=True,
        protocol='Prospective V2 erratum; V1 independent cohorts invalidated by omitted earlier held-out exposure',
        prior_failed_run='runs/paper_scale_ablation_v1/eval',
        prior_failure_receipts=v1_receipts,
        prior_model_registry_sha256=sha(SOURCE_RUN / 'models/frozen.json'),
        engine_script_sha256=sha(engine.__file__),
        v2_script_sha256=sha(__file__),
        correction_sha256=sha(CORRECTION),
        history_sha256=sha(HISTORY),
        post_v32_exposure_sha256=sha(POST_V32),
        prose_erratum_sha256=sha(ERRATUM),
        full_public_sha256=sha(PUBLIC),
        official_manifest_sha256_from_preparation_audit=official_sha,
        corrected_history_counts={key: sum(r['history_status'] == key for r in history['rows'])
                                  for key in sorted(EXPECTED_HISTORY)},
        forbidden_cohorts=['reserve58_v1', 'venus143_v1'],
        allowed_stages=list(STAGES),
        full_benchmark_independent=False,
        full_labels_may_open=False,
        independent_confirmation='pending new prospective cohort and sealed predictions',
        test_labels_opened=False,
        test_model_forward_calls_at_registration=0,
    )
    if PROTOCOL.exists():
        require(read(PROTOCOL) == expected, 'V2 protocol changed after registration')
    else:
        engine.write(PROTOCOL, expected)
    print(json.dumps({'stage': 'register', 'protocol_sha256': sha(PROTOCOL),
                      'history_counts': expected['corrected_history_counts'],
                      'full_labels_may_open': False}, ensure_ascii=False), flush=True)


def gate():
    require(PROTOCOL.exists(), 'Register V2 erratum before evaluation')
    registered = read(PROTOCOL)
    require(registered['complete'] is True and registered['v2_script_sha256'] == sha(__file__) and
            registered['engine_script_sha256'] == sha(engine.__file__) and
            registered['prior_model_registry_sha256'] == sha(SOURCE_RUN / 'models/frozen.json') and
            registered['correction_sha256'] == sha(CORRECTION) and
            registered['history_sha256'] == sha(HISTORY) and
            registered['post_v32_exposure_sha256'] == sha(POST_V32) and
            registered['prose_erratum_sha256'] == sha(ERRATUM) and
            registered['full_public_sha256'] == sha(PUBLIC) and
            registered['full_labels_may_open'] is False,
            'V2 prospective protocol or source changed')
    require(registered['prior_failure_receipts'] == old_failure(),
            'V1 failure receipt changed')
    return registered


def verify_public_v2(cohort):
    require(cohort == 'full', 'V2 permits only the descriptive full benchmark')
    registered = gate()
    spec, rows = _engine_verify_public(cohort)
    require(spec['v2_script_sha256'] == sha(__file__) and
            spec['erratum_sha256'] == sha(PROTOCOL) and
            spec['corrected_history_sha256'] == registered['history_sha256'] and
            spec['labels_opened'] is False,
            'Full public protocol is not bound to V2 erratum')
    return spec, rows


engine.verify_public = verify_public_v2


def _locked_label_analysis(*_args, **_kwargs):
    raise RuntimeError('V2 full benchmark labels stay locked until a separate prospective independent protocol')


engine.analyze = _locked_label_analysis


def prepare_full():
    registered = gate()
    table = engine.all_models_sealed()
    folder = OUT / 'full'
    spec_path = folder / 'protocol.json'
    if spec_path.exists():
        verify_public_v2('full')
        return
    _, _, source_rows = corrected_history()
    rows = [{k: row[k] for k in engine.CASE_KEYS} for row in source_rows]
    require(len(rows) == 1581 and len({r['sample_id'] for r in rows}) == 1581,
            'Full public benchmark ID drift')
    engine.write(folder / 'public.json', rows)
    source_sha = {
        str(PUBLIC): sha(PUBLIC),
        str(CORRECTION): sha(CORRECTION),
        str(HISTORY): sha(HISTORY),
        str(POST_V32): sha(POST_V32),
        str(ERRATUM): sha(ERRATUM),
        str(PROTOCOL): sha(PROTOCOL),
        str(engine.IMAGE_INDEX): sha(engine.IMAGE_INDEX),
        # This file contains GT. Its digest comes from the sealed preparation
        # audit; do not open it until a later, separately registered analysis.
        str(engine.SSPRO): registered['official_manifest_sha256_from_preparation_audit'],
    }
    spec = dict(complete=True, cohort='full', n=len(rows),
                public_sha256=sha(folder / 'public.json'), source_sha256=source_sha,
                model_hashes={name: entry['sha256'] for name, entry in table.items()},
                model_seal_sha256=sha(SOURCE_RUN / 'models/frozen.json'),
                v16_seal_sha256=sha(engine.V16 / 'models/frozen.json'),
                v18_seal_sha256=sha(engine.V18 / 'models/frozen.json'),
                script_sha256=sha(engine.__file__), v2_script_sha256=sha(__file__),
                erratum_sha256=sha(PROTOCOL),
                corrected_history_sha256=sha(HISTORY),
                metric='Original K16 forced Top1, canonical union-index tie',
                public_only=True, labels_opened=False, created_at_utc=now())
    engine.write(spec_path, spec)
    print(json.dumps({'stage': 'prepare-full', 'n': len(rows),
                      'models': len(table), 'labels_opened': False}, ensure_ascii=False), flush=True)


def output_for(stage):
    if stage == 'register':
        return PROTOCOL
    if stage == 'feature-parity':
        return OUT / 'parity_features.json'
    if stage == 'archive-parity':
        return OUT / 'parity_archives.json'
    if stage.startswith('parity-'):
        return OUT / 'parity_full' / (stage.removeprefix('parity-') + '.json')
    return OUT / 'full' / {'prepare-full': 'protocol.json',
                           'extract-full': 'features/index.json',
                           'score-full': 'predictions.json'}[stage]


def latest_event():
    path = OUT / 'events.jsonl'
    if not path.exists():
        return None
    with path.open('rb') as stream:
        stream.seek(0, os.SEEK_END)
        stream.seek(max(0, stream.tell() - 16384))
        lines = stream.read().decode('utf-8', errors='replace').splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            pass
    return None


def status(phase, **fields):
    previous = read(STATUS) if STATUS.exists() else {}
    record = dict(phase=phase, updated_at_utc=now(), controller_pid=os.getpid(),
                  completed_stages=previous.get('completed_stages', []))
    record.update(fields)
    engine.write(STATUS, record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def run_all_public(device, poll_seconds):
    LOGS.mkdir(parents=True, exist_ok=True)
    for stage in STAGES:
        command = [sys.executable, '-B', '-X', 'utf8', '-u', str(Path(__file__)),
                   '--stage', stage, '--device', device]
        out_path = LOGS / f'{stage}.stdout.log'
        err_path = LOGS / f'{stage}.stderr.log'
        started = time.monotonic()
        status('running', stage=stage, command=command,
               stdout_log=str(out_path), stderr_log=str(err_path))
        with out_path.open('ab') as stdout, err_path.open('ab') as stderr:
            header = ('\n=== ' + now() + ' ' + stage + ' ===\n').encode()
            stdout.write(header); stderr.write(header)
            stdout.flush(); stderr.flush()
            process = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            last_report = 0.0
            try:
                while process.poll() is None:
                    elapsed = time.monotonic() - started
                    if elapsed - last_report >= 30 or last_report == 0:
                        event = latest_event()
                        matching = (event is not None and
                                    event.get('cohort') == 'full' and
                                    event.get('stage') == ('extract_public' if stage == 'extract-full'
                                                           else 'score_public' if stage == 'score-full'
                                                           else None))
                        progress = ({key: event.get(key) for key in ('completed', 'total', 'elapsed_seconds')}
                                    if matching else None)
                        status('running', stage=stage, child_pid=process.pid,
                               elapsed_stage_seconds=round(elapsed, 1),
                               latest_event=event, progress=progress,
                               stdout_log=str(out_path), stderr_log=str(err_path))
                        last_report = elapsed
                    time.sleep(poll_seconds)
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    process.wait(timeout=30)
                raise
        require(process.returncode == 0,
                f'{stage} exited {process.returncode}; inspect {err_path}')
        output = output_for(stage)
        require(output.exists() and read(output).get('complete') is True,
                'Stage did not seal expected output: ' + str(output))
        previous = read(STATUS)
        done = list(dict.fromkeys([*previous.get('completed_stages', []), stage]))
        status('stage_complete', stage=stage, completed_stages=done,
               output=str(output), output_sha256=sha(output),
               elapsed_stage_seconds=round(time.monotonic() - started, 1))
    seal = OUT / 'full/predictions.json'
    receipt = dict(complete=True, scope='Public predictions only; no benchmark labels opened',
                   full_predictions_sha256=sha(seal),
                   corrected_history_sha256=sha(HISTORY),
                   erratum_sha256=sha(PROTOCOL),
                   independent_confirmation='pending', labels_opened=False)
    completion = RUN / 'public_completion.json'
    if completion.exists():
        require(read(completion) == receipt, 'V2 public completion drift')
    else:
        engine.write(completion, receipt)
    status('complete_public', complete=True, completion=str(completion),
           predictions_sha256=sha(seal), labels_opened=False)


def dry_run():
    receipts = old_failure()
    ready = CORRECTION.exists() and HISTORY.exists() and POST_V32.exists()
    if ready:
        _, history, _ = corrected_history()
    else:
        history = None
    table = engine.models(False)
    require(len(table) == 39, 'Frozen 3k/full/matched3k registry must contain 39 models')
    info = dict(dry_run=True, v1_failure_receipts=receipts,
                corrected_history_ready=ready,
                correction_sha256=sha(CORRECTION) if ready else None,
                corrected_history_sha256=sha(HISTORY) if ready else None,
                post_v32_exposure_sha256=sha(POST_V32) if ready else None,
                corrected_history_counts=history['history_status_counts'] if ready else None,
                prose_erratum_exists=ERRATUM.exists(),
                prose_erratum_sha256=sha(ERRATUM) if ERRATUM.exists() else None,
                frozen_model_count=len(table),
                frozen_model_registry_sha256=sha(SOURCE_RUN / 'models/frozen.json'),
                planned_stages=list(STAGES),
                existing_v2_protocol=PROTOCOL.exists(),
                labels_opened=False, model_forward_calls=0)
    print(json.dumps(info, ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=(*STAGES, 'all-public', 'dry-run'),
                        default='dry-run')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--poll-seconds', type=int, default=15)
    args = parser.parse_args()
    require(1 <= args.poll_seconds <= 30, 'Poll interval must be 1..30 seconds')
    os.chdir(ROOT)
    if args.stage == 'dry-run':
        dry_run()
    elif args.stage == 'all-public':
        try:
            run_all_public(args.device, args.poll_seconds)
        except BaseException as exc:
            status('failed', complete=False, error=repr(exc))
            raise
    elif args.stage == 'register':
        register()
    else:
        gate()
        if args.stage == 'feature-parity':
            engine.feature_parity(3)
        elif args.stage == 'archive-parity':
            engine.archive_parity()
        elif args.stage.startswith('parity-'):
            engine.parity(args.stage.removeprefix('parity-'), args.device, 0, False)
        elif args.stage == 'prepare-full':
            prepare_full()
        elif args.stage == 'extract-full':
            engine.extract_public('full')
        else:
            engine.score_public('full', args.device)


if __name__ == '__main__':
    main()
