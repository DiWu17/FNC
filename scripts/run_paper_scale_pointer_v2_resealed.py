"""Prospective PointerBench confirmation of the frozen 3k/full K16 scorers.

Pro500 is primary and Sheets500 is secondary. Public inputs are extracted and
all 39 frozen models scored before either label file is opened. This script
reuses the V1 scientific feature/scoring functions, with cache reuse disabled
for the new source, under a separate V2 pointer protocol. Full ScreenSpot-Pro
predictions must also be sealed before the Pointer labels can be unlocked.
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
import numpy as np
import run_paper_scale_eval_v1 as engine

RUN = ROOT / 'runs/paper_scale_ablation_v2'
OUT = RUN / 'eval'
DATA = RUN / 'pointerbench'
PUBLIC_SOURCE = DATA / 'public.jsonl'
LABEL_SOURCE = DATA / 'labels.jsonl'
AUDIT = DATA / 'audit.json'
PROVENANCE = DATA / 'provenance.json'
OLD_INCIDENT = DATA / 'preseal_label_access_incident.json'
INCIDENT = DATA / 'preseal_label_access_addendum_v2.json'
OLD_PROTOCOL = DATA / 'evaluation_protocol.json'
PROTOCOL = DATA / 'evaluation_protocol_v2.json'
STATUS = DATA / 'evaluation_status.json'
PUBLIC_COMPLETION = DATA / 'public_completion.json'
READOUT = DATA / 'readout.txt'
LOGS = DATA / 'evaluation_logs'
FULL_PROTOCOL = RUN / 'protocol.json'
FULL_PREDICTIONS = OUT / 'full/predictions.json'
FULL_COMPLETION = RUN / 'public_completion.json'
CORRECTION = ROOT / 'runs/paper_scale_ablation_v1/data/exposure_correction_v2.json'
HISTORY = ROOT / 'runs/paper_scale_ablation_v1/data/full_history_overlap_corrected_v2.json'
POST_V32 = ROOT / 'runs/paper_scale_ablation_v1/data/post_v32_exposure_addendum_v1.json'
ERRATUM = ROOT / 'research/paper/SCALE_ABLATION_ERRATUM_V2.md'
REVISION = '5a421276ad8bfe813116df04482f81a6201fef13'
SUITES = ('pro', 'sheets')
PUBLIC_STAGES = ('register', 'prepare-pro', 'extract-pro', 'score-pro',
                 'prepare-sheets', 'extract-sheets', 'score-sheets')
ANALYSIS_STAGES = ('analyze-pro', 'analyze-sheets')

engine.OUT = OUT
_engine_verify_public = engine.verify_public
# The independent source must be processed afresh. In particular, do not read
# earlier per-case prediction archives as a cache inventory during extraction.
engine.archived_sources = lambda: ({}, {})


def now():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def jsonl(path):
    with Path(path).open('r', encoding='utf-8-sig') as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def source_audit():
    for path in (PUBLIC_SOURCE, AUDIT, PROVENANCE, OLD_INCIDENT, INCIDENT,
                 OLD_PROTOCOL, FULL_PROTOCOL, CORRECTION, HISTORY, POST_V32, ERRATUM):
        require(path.exists(), 'Prospective Pointer source missing: ' + str(path))
    audit, provenance = read(AUDIT), read(PROVENANCE)
    require(audit.get('complete') is True and
            audit.get('independent_history_audit_passed') is True and
            audit.get('counts') == {'pro': 500, 'sheets': 500},
            'Pointer source or exact/near history audit not approved')
    require(provenance.get('revision') == REVISION,
            'PointerBench source revision changed')
    require(sha(PROVENANCE) == audit.get('provenance_sha256'),
            'Pointer provenance digest changed since source audit')
    require(sha(PUBLIC_SOURCE) == audit.get('public_sha256'),
            'Pointer public manifest digest changed')
    incident = read(INCIDENT)
    superseded = read(OLD_PROTOCOL)
    require(superseded.get('complete') is True and
            superseded.get('preseal_label_access_incident_sha256') == sha(OLD_INCIDENT) and
            superseded.get('model_forward_calls_at_registration') == 0 and
            superseded.get('public_source_sha256') == sha(PUBLIC_SOURCE),
            'Superseded Pointer registration changed')
    require(incident.get('complete') is True and
            incident.get('supersedes_incident_sha256') == sha(OLD_INCIDENT) and
            incident.get('supersedes_pointer_protocol_sha256') == sha(OLD_PROTOCOL) and
            incident.get('identified_distinct_examples_at_least', 0) >= 7 and
            incident.get('exhaustive_count_known') is False and
            incident.get('evaluation_pipeline_label_access_before_prediction_seal') is False,
            'Pointer non-blind pre-seal access addendum is not recorded')
    public = jsonl(PUBLIC_SOURCE)
    require(len(public) == 1000 and
            [q['suite'] for q in public] == ['pro'] * 500 + ['sheets'] * 500 and
            len({q['sample_id'] for q in public}) == 1000,
            'Pointer suite size, order or ID uniqueness changed')
    for q in public:
        require(q['dataset'] == 'PointerBench' and q['image_size'] == [1024, 768] and
                q['suite'] in SUITES and
                set(q) == set(engine.CASE_KEYS) | {'suite', 'dhash256'},
                'Pointer public row contains a missing field, GT or wrong coordinate frame')
    return audit, provenance, public


def register():
    audit, provenance, public = source_audit()
    full = read(FULL_PROTOCOL)
    require(full['complete'] is True and full['full_labels_may_open'] is False and
            full['engine_script_sha256'] == sha(engine.__file__) and
            full['correction_sha256'] == sha(CORRECTION) and
            full['history_sha256'] == sha(HISTORY) and
            full['post_v32_exposure_sha256'] == sha(POST_V32) and
            full['prose_erratum_sha256'] == sha(ERRATUM),
            'V2 full public-only protocol is not sealed')
    receipt = dict(
        complete=True, purpose='Frozen 39-model PointerBench confirmation',
        primary='PointerBench Pro500', secondary='PointerBench Sheets500',
        revision=REVISION, n=1000, pro_n=500, sheets_n=500,
        coordinate_frame=[1024, 768], metric='Original K16 forced Top1 point-in-bbox',
        model_registry_sha256=full['prior_model_registry_sha256'],
        full_protocol_sha256=sha(FULL_PROTOCOL),
        engine_script_sha256=sha(engine.__file__),
        pointer_script_sha256=sha(__file__),
        source_audit_sha256=sha(AUDIT), provenance_sha256=sha(PROVENANCE),
        superseded_pointer_protocol_sha256=sha(OLD_PROTOCOL),
        superseded_incident_sha256=sha(OLD_INCIDENT),
        preseal_label_access_incident_sha256=sha(INCIDENT),
        public_source_sha256=sha(PUBLIC_SOURCE),
        labels_source_sha256_unopened=audit['labels_sha256'],
        corrected_exposure_sha256=sha(CORRECTION),
        corrected_history_sha256=sha(HISTORY),
        post_v32_exposure_sha256=sha(POST_V32),
        prose_erratum_sha256=sha(ERRATUM),
        no_archived_feature_reuse=True,
        evaluation_pipeline_predictions_before_label_loading=True,
        full_predictions_before_any_labels=True,
        mainline_model_unchanged=True,
        at_least_seven_ground_truth_rows_previewed_before_prediction_seal=True,
        exhaustive_preseal_preview_count_known=False,
        strict_blind_confirmation=False,
        evaluation_pipeline_labels_opened=False,
        model_forward_calls_at_registration=0,
        stages=[*PUBLIC_STAGES, *ANALYSIS_STAGES],
    )
    if PROTOCOL.exists():
        require(read(PROTOCOL) == receipt, 'Pointer protocol differs from frozen registration')
    else:
        engine.write(PROTOCOL, receipt)
    print(json.dumps({'stage': 'register', 'n': len(public),
                      'revision': REVISION, 'labels_opened': False,
                      'protocol_sha256': sha(PROTOCOL)}, ensure_ascii=False), flush=True)


def gate():
    require(PROTOCOL.exists(), 'Register Pointer protocol first')
    p = read(PROTOCOL)
    require(p['complete'] is True and p['pointer_script_sha256'] == sha(__file__) and
            p['engine_script_sha256'] == sha(engine.__file__) and
            p['model_registry_sha256'] == sha(engine.RUN / 'models/frozen.json') and
            p['full_protocol_sha256'] == sha(FULL_PROTOCOL) and
            p['source_audit_sha256'] == sha(AUDIT) and
            p['provenance_sha256'] == sha(PROVENANCE) and
            p['superseded_pointer_protocol_sha256'] == sha(OLD_PROTOCOL) and
            p['superseded_incident_sha256'] == sha(OLD_INCIDENT) and
            p['preseal_label_access_incident_sha256'] == sha(INCIDENT) and
            p['public_source_sha256'] == sha(PUBLIC_SOURCE) and
            p['corrected_exposure_sha256'] == sha(CORRECTION) and
            p['corrected_history_sha256'] == sha(HISTORY) and
            p['post_v32_exposure_sha256'] == sha(POST_V32) and
            p['prose_erratum_sha256'] == sha(ERRATUM) and
            p['evaluation_pipeline_labels_opened'] is False and
            p['at_least_seven_ground_truth_rows_previewed_before_prediction_seal'] is True and
            p['exhaustive_preseal_preview_count_known'] is False and
            p['strict_blind_confirmation'] is False,
            'Pointer frozen identity or V2 erratum changed')
    return p


def verify_public_pointer(cohort):
    require(cohort in SUITES, 'Pointer source supports only predeclared suites')
    protocol = gate()
    spec, rows = _engine_verify_public(cohort)
    require(spec['pointer_protocol_sha256'] == sha(PROTOCOL) and
            spec['pointer_script_sha256'] == sha(__file__) and
            spec['public_only'] is True and spec['labels_opened'] is False and
            len(rows) == 500 and all(q['image_size'] == [1024, 768] for q in rows),
            'Pointer suite does not match frozen public protocol')
    return spec, rows


engine.verify_public = verify_public_pointer


def prepare(cohort):
    require(cohort in SUITES, 'Unknown Pointer suite')
    gate()
    table = engine.all_models_sealed()
    folder = OUT / cohort
    if (folder / 'protocol.json').exists():
        verify_public_pointer(cohort)
        return
    _, _, public = source_audit()
    selected = [q for q in public if q['suite'] == cohort]
    require(len(selected) == 500, 'Pointer suite count changed')
    rows = [{key: q[key] for key in engine.CASE_KEYS} for q in selected]
    engine.write(folder / 'public.json', rows)
    source_sha = {str(PUBLIC_SOURCE): sha(PUBLIC_SOURCE), str(AUDIT): sha(AUDIT),
                  str(PROVENANCE): sha(PROVENANCE), str(OLD_INCIDENT): sha(OLD_INCIDENT),
                  str(INCIDENT): sha(INCIDENT), str(OLD_PROTOCOL): sha(OLD_PROTOCOL),
                  str(PROTOCOL): sha(PROTOCOL),
                  str(FULL_PROTOCOL): sha(FULL_PROTOCOL), str(CORRECTION): sha(CORRECTION),
                  str(HISTORY): sha(HISTORY), str(POST_V32): sha(POST_V32),
                  str(ERRATUM): sha(ERRATUM)}
    spec = dict(complete=True, cohort=cohort, n=500,
                public_sha256=sha(folder / 'public.json'), source_sha256=source_sha,
                model_hashes={name: entry['sha256'] for name, entry in table.items()},
                model_seal_sha256=sha(engine.RUN / 'models/frozen.json'),
                v16_seal_sha256=sha(engine.V16 / 'models/frozen.json'),
                v18_seal_sha256=sha(engine.V18 / 'models/frozen.json'),
                script_sha256=sha(engine.__file__),
                pointer_script_sha256=sha(__file__),
                pointer_protocol_sha256=sha(PROTOCOL),
                metric='Original K16 forced Top1 point-in-bbox, official 1024x768',
                public_only=True, labels_opened=False, created_at_utc=now())
    engine.write(folder / 'protocol.json', spec)
    print(json.dumps({'stage': f'prepare-{cohort}', 'n': 500,
                      'models': len(table), 'labels_opened': False}, ensure_ascii=False), flush=True)


def full_public_sealed():
    require(FULL_COMPLETION.exists() and FULL_PREDICTIONS.exists(),
            'Finish complete ScreenSpot-Pro public predictions before Pointer inference')
    full_receipt = read(FULL_COMPLETION)
    full = read(FULL_PREDICTIONS)
    require(full_receipt['complete'] is True and full_receipt['labels_opened'] is False and
            full_receipt['erratum_sha256'] == sha(FULL_PROTOCOL) and
            full_receipt['full_predictions_sha256'] == sha(FULL_PREDICTIONS) and
            full['complete'] is True and full['labels_opened'] is False and full['n'] == 1581,
            'Full benchmark public prediction seal is incomplete or changed')
    return full_receipt


def all_predictions_sealed():
    protocol = gate()
    require(PUBLIC_COMPLETION.exists(),
            'Finish the registered Pro500+Sheets500 public-only stage chain first')
    pointer_receipt = read(PUBLIC_COMPLETION)
    require(pointer_receipt['complete'] is True and
            pointer_receipt['labels_opened'] is False and
            pointer_receipt['pointer_protocol_sha256'] == sha(PROTOCOL),
            'Pointer public completion receipt changed')
    full_public_sealed()
    suites = {}
    for cohort in SUITES:
        spec, rows = verify_public_pointer(cohort)
        path = OUT / cohort / 'predictions.json'
        require(path.exists(), 'Both Pointer suites must have sealed predictions')
        seal = read(path)
        require(seal['complete'] is True and seal['labels_opened'] is False and
                seal['n'] == len(rows) == 500 and
                seal['public_sha256'] == spec['public_sha256'] and
                seal['model_hashes'] == spec['model_hashes'],
                'Pointer prediction seal mismatch: ' + cohort)
        for row in rows:
            sid = row['sample_id']
            case = OUT / cohort / f'predictions/cases/{engine.key(sid)}.json'
            require(sha(case) == seal['case_hashes'][sid],
                    'Sealed Pointer case changed: ' + sid)
        suites[cohort] = (spec, rows, seal, path)
        require(pointer_receipt['prediction_sha256'][cohort] == sha(path),
                'Pointer public completion/prediction seal differs')
    return protocol, suites


def analyze(cohort):
    require(cohort in SUITES, 'Unknown Pointer analysis suite')
    protocol, suites = all_predictions_sealed()
    # The evaluation pipeline's first GT read occurs only after all 2,581
    # predictions (500 + 500 + 1,581) have complete public-only seals.
    require(sha(LABEL_SOURCE) == protocol['labels_source_sha256_unopened'],
            'Pointer label source changed since prospective registration')
    labels_rows = jsonl(LABEL_SOURCE)
    require(len(labels_rows) == 1000 and len({q['sample_id'] for q in labels_rows}) == 1000,
            'Pointer labels are incomplete or duplicated')
    labels = {q['sample_id']: q['target_geometry'] for q in labels_rows}
    for suite in SUITES:
        require({q['sample_id'] for q in suites[suite][1]} <= set(labels),
                'Pointer labels miss a predeclared query')
    spec, rows, seal, seal_path = suites[cohort]
    output = OUT / cohort / 'analysis.json'
    if output.exists():
        prior = read(output)
        require(prior['complete'] is True and
                prior['predictions_sha256'] == sha(seal_path) and
                prior['labels_sha256'] == sha(LABEL_SOURCE),
                'Existing Pointer analysis differs from sealed predictions or labels')
        return
    methods = spec['model_hashes']
    hits = {name: [] for name in methods}
    details = []
    for row in rows:
        sid = row['sample_id']
        case = OUT / cohort / f'predictions/cases/{engine.key(sid)}.json'
        prediction = read(case)
        require(sha(case) == seal['case_hashes'][sid] and
                prediction['sample_id'] == sid and set(prediction['methods']) == set(methods),
                'Pointer scored case changed: ' + sid)
        outcomes = {}
        for name, result in prediction['methods'].items():
            point = result['point']
            value = bool(point is not None and engine.point_in_geometry(
                (np.asarray(point) / np.asarray([1024, 768])).tolist(), labels[sid]))
            outcomes[name] = value
            hits[name].append(value)
        details.append(dict(sample_id=sid, dataset='PointerBench', suite=cohort,
                            application=row['application'],
                            application_group=row['application_group'],
                            pixel_sha256=row['pixel_sha256'], hits=outcomes))
    method_scores = {name: dict(hits=sum(values), accuracy=sum(values) / 500)
                     for name, values in hits.items()}
    summary = {scale: {variant: dict(
        seed_accuracies=[method_scores[f'{scale}_{variant}_s{s}']['accuracy']
                         for s in (1, 2, 3)],
        mean_accuracy=float(np.mean([method_scores[f'{scale}_{variant}_s{s}']['accuracy']
                                     for s in (1, 2, 3)])))
                       for variant in engine.VARIANTS} for scale in ('3k', 'full')}
    control = [method_scores[f'matched3k_aligned_s{s}']['accuracy'] for s in (1, 2, 3)]
    summary['matched3k_aligned'] = dict(seed_accuracies=control,
                                       mean_accuracy=float(np.mean(control)))
    contrasts, rgb_images = engine.paired_intervals(hits, rows)
    analysis = dict(complete=True, cohort=cohort,
                    role='predeclared primary' if cohort == 'pro' else 'predeclared secondary',
                    n=500, unique_rgb_images=rgb_images,
                    coordinate_frame=[1024, 768],
                    metric='Original K16 forced Top1 point-in-bbox',
                    predictions_sha256=sha(seal_path), labels_sha256=sha(LABEL_SOURCE),
                    pointer_protocol_sha256=sha(PROTOCOL),
                    methods=method_scores, summary=summary, contrasts=contrasts,
                    primary_family='(3k/full) x (no_evidence/spatial_shuffle); use ci98_75',
                    analyzed_at_utc=now())
    engine.write(output, analysis)
    engine.write(OUT / cohort / 'analysis_details.json', details)
    print(json.dumps({'stage': f'analyze-{cohort}', 'n': 500,
                      'mean_full_aligned': summary['full']['aligned']['mean_accuracy'],
                      'mean_3k_aligned': summary['3k']['aligned']['mean_accuracy']},
                     ensure_ascii=False), flush=True)


def output_for(stage):
    if stage == 'register':
        return PROTOCOL
    action, cohort = stage.split('-', 1)
    return OUT / cohort / {'prepare': 'protocol.json',
                           'extract': 'features/index.json',
                           'score': 'predictions.json',
                           'analyze': 'analysis.json'}[action]


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
    prior = read(STATUS) if STATUS.exists() else {}
    value = dict(phase=phase, updated_at_utc=now(),
                 completed_stages=prior.get('completed_stages', []))
    value.update(fields)
    engine.write(STATUS, value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def readout():
    reports = {cohort: read(OUT / cohort / 'analysis.json') for cohort in SUITES}
    lines = [
        'PointerBench 独立确认：Pro500 为预声明主队列，Sheets500 为预声明副队列。',
        '固定原始 K16 点命中、官方 1024×768 坐标；同一冻结39模型，三种子。',
        '两套 Pointer 与完整 ScreenSpot-Pro 预测均已先封存，之后评估流水线才读取 Pointer 标签。',
        '协议偏差：预测前至少7个不同样本的GT在文档/查看器/行预览中可见；确切总数未知。',
        '仍保留全部预声明样本；此队列不是严格盲测，未按可见GT调整模型或指标。',
        'PointerBench 为合成 GUI 来源，跨来源结论不自动外推至真实桌面代理执行。',
    ]
    for cohort in SUITES:
        result = reports[cohort]
        n = result['n']
        lines += ['', f'{cohort.upper()}，n={n}，RGB 图像簇={result["unique_rgb_images"]}。',
                  '规模 | 消融 | S1 | S2 | S3 | 三种子平均准确率']
        for scale in ('3k', 'full'):
            for variant in engine.VARIANTS:
                scores = [result['methods'][f'{scale}_{variant}_s{s}']['hits']
                          for s in (1, 2, 3)]
                mean = result['summary'][scale][variant]['mean_accuracy']
                lines.append(f'{scale} | {variant} | {scores[0]}/{n} | {scores[1]}/{n} | '
                             f'{scores[2]}/{n} | {100*mean:.2f}%')
        control = [result['methods'][f'matched3k_aligned_s{s}']['hits']
                   for s in (1, 2, 3)]
        mean = result['summary']['matched3k_aligned']['mean_accuracy']
        lines.append(f'matched3k | aligned | {control[0]}/{n} | {control[1]}/{n} | '
                     f'{control[2]}/{n} | {100*mean:.2f}%')
        lines.append('配对差：百分点；95% RGB图像簇区间；各预定种子差。')
        for name, item in result['contrasts'].items():
            seeds = ', '.join(f'{100*x:+.2f}' for x in item['per_seed_delta'])
            ci = ', '.join(f'{100*x:+.2f}' for x in item['ci95'])
            lines.append(f'{name}: {100*item["delta"]:+.2f}; [{ci}]; [{seeds}]')
        lines.append('四项主机制比较的同时覆盖98.75%区间（百分点）：')
        for scale in ('3k', 'full'):
            for variant in ('no_evidence', 'spatial_shuffle'):
                name = f'{scale}_aligned_vs_{variant}'
                ci = ', '.join(f'{100*x:+.2f}' for x in result['contrasts'][name]['ci98_75'])
                lines.append(f'{name}: [{ci}]')
    lines += ['', '以上区间按 RGB 图像簇重采样，条件化于固定三次拟合；'
              '不据此声称对 MVP 或真实点击执行任务的总体优势。', '']
    temp = READOUT.with_suffix('.txt.tmp')
    temp.write_text('\n'.join(lines), encoding='utf-8')
    os.replace(temp, READOUT)
    return READOUT


def run_stages(stages, device, poll_seconds):
    LOGS.mkdir(parents=True, exist_ok=True)
    for stage in stages:
        cmd = [sys.executable, '-B', '-X', 'utf8', '-u', str(Path(__file__)),
               '--stage', stage, '--device', device]
        stdout_path = LOGS / f'{stage}.stdout.log'
        stderr_path = LOGS / f'{stage}.stderr.log'
        started = time.monotonic()
        status('running', stage=stage, command=cmd,
               stdout_log=str(stdout_path), stderr_log=str(stderr_path))
        with stdout_path.open('ab') as stdout, stderr_path.open('ab') as stderr:
            header = ('\n=== ' + now() + ' ' + stage + ' ===\n').encode()
            stdout.write(header); stderr.write(header)
            stdout.flush(); stderr.flush()
            process = subprocess.Popen(cmd, cwd=ROOT, stdout=stdout, stderr=stderr,
                                       creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
            last_report = 0.0
            try:
                while process.poll() is None:
                    elapsed = time.monotonic() - started
                    if elapsed - last_report >= 30 or last_report == 0:
                        event = latest_event()
                        expected = ('extract_public' if stage.startswith('extract-') else
                                    'score_public' if stage.startswith('score-') else None)
                        progress = ({key: event.get(key) for key in
                                     ('completed', 'total', 'elapsed_seconds')}
                                    if event and event.get('stage') == expected and
                                    event.get('cohort') == stage.split('-', 1)[-1] else None)
                        status('running', stage=stage, child_pid=process.pid,
                               elapsed_stage_seconds=round(elapsed, 1),
                               latest_event=event, progress=progress,
                               stdout_log=str(stdout_path), stderr_log=str(stderr_path))
                        last_report = elapsed
                    time.sleep(poll_seconds)
            except BaseException:
                if process.poll() is None:
                    process.terminate(); process.wait(timeout=30)
                raise
        require(process.returncode == 0,
                f'{stage} exited {process.returncode}; inspect {stderr_path}')
        output = output_for(stage)
        require(output.exists() and read(output).get('complete') is True,
                'Stage did not seal output: ' + str(output))
        prior = read(STATUS)
        done = list(dict.fromkeys([*prior.get('completed_stages', []), stage]))
        status('stage_complete', stage=stage, completed_stages=done,
               output=str(output), output_sha256=sha(output),
               elapsed_stage_seconds=round(time.monotonic() - started, 1))
    if stages == PUBLIC_STAGES:
        receipt = dict(complete=True, scope='Pro500+Sheets500 public predictions',
                       pointer_protocol_sha256=sha(PROTOCOL),
                       prediction_sha256={cohort: sha(OUT / cohort / 'predictions.json')
                                          for cohort in SUITES},
                       labels_opened=False)
        if PUBLIC_COMPLETION.exists():
            require(read(PUBLIC_COMPLETION) == receipt, 'Pointer public receipt drift')
        else:
            engine.write(PUBLIC_COMPLETION, receipt)
        status('complete_public', complete=True, labels_opened=False,
               public_completion=str(PUBLIC_COMPLETION))
    else:
        path = readout()
        status('complete_analysis', complete=True, labels_opened=True,
               readout=str(path), readout_sha256=sha(path))


def dry_run():
    ready = all(path.exists() for path in (PUBLIC_SOURCE, AUDIT, PROVENANCE,
                                           OLD_INCIDENT, INCIDENT, OLD_PROTOCOL,
                                           FULL_PROTOCOL, CORRECTION, HISTORY,
                                           POST_V32, ERRATUM))
    if ready:
        source_audit()
    print(json.dumps(dict(dry_run=True, source_ready=ready,
                          revision=REVISION, suites={'pro': 500, 'sheets': 500},
                          public_stages=list(PUBLIC_STAGES),
                          analysis_stages=list(ANALYSIS_STAGES),
                          labels_opened=False, model_forward_calls=0),
                     ensure_ascii=False, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=(*PUBLIC_STAGES, *ANALYSIS_STAGES,
                                            'all-public', 'all-analysis', 'dry-run'),
                        default='dry-run')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--poll-seconds', type=int, default=15)
    args = parser.parse_args()
    require(1 <= args.poll_seconds <= 30, 'Poll interval must be 1..30 seconds')
    os.chdir(ROOT)
    if args.stage == 'dry-run':
        dry_run()
    elif args.stage in ('all-public', 'all-analysis'):
        try:
            run_stages(PUBLIC_STAGES if args.stage == 'all-public' else ANALYSIS_STAGES,
                       args.device, args.poll_seconds)
        except BaseException as exc:
            status('failed', complete=False, error=repr(exc))
            raise
    elif args.stage == 'register':
        register()
    else:
        gate()
        action, cohort = args.stage.split('-', 1)
        if action == 'prepare':
            prepare(cohort)
        elif action == 'extract':
            full_public_sealed()
            engine.extract_public(cohort)
        elif action == 'score':
            full_public_sealed()
            engine.score_public(cohort, args.device)
        else:
            analyze(cohort)


if __name__ == '__main__':
    main()
