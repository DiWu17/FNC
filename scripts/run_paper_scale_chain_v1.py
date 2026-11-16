"""Resumable, text-only controller for the frozen 3k/full ablation study.

Wait for the 18-slot full-data model seal, verify observed development parity,
then run reserve58 -> Venus143 -> official ScreenSpot-Pro1581. Every test cohort
has public extraction, scoring and only then explicit label analysis. This
controller never selects a seed, model, checkpoint or policy from outcomes.
"""
from __future__ import annotations

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs/paper_scale_ablation_v1'
OUT = RUN / 'eval'
EVAL = ROOT / 'scripts/run_paper_scale_eval_v1.py'
STATUS = OUT / 'chain_status.json'
LOG_DIR = OUT / 'logs/chain'
STAGES = (
    'feature-parity', 'archive-parity', 'parity-old_val', 'parity-new_development',
    'prepare-reserve', 'extract-reserve', 'score-reserve', 'analyze-reserve',
    'prepare-venus', 'extract-venus', 'score-venus', 'analyze-venus',
    'prepare-full', 'extract-full', 'score-full', 'analyze-full',
)
VARIANTS = ('aligned', 'no_evidence', 'spatial_shuffle',
            'attention_only', 'hidden_only', 'token_mean')
SEEDS = (20261311, 20261312, 20261313)


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


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8')
    os.replace(temp, path)


def atomic_text(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(value, encoding='utf-8')
    os.replace(temp, path)


def state(phase, **extra):
    old = read(STATUS) if STATUS.exists() else {}
    completed = old.get('completed_stages', [])
    value = dict(phase=phase, updated_at_utc=now(), controller_pid=os.getpid(),
                 completed_stages=completed, stages_total=len(STAGES))
    value.update(extra)
    atomic_json(STATUS, value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def check_only(device='cuda:0', show_plan=False):
    """No model forward, no labels opened, no stage launched."""
    for path in (EVAL, Path(__file__)):
        ast.parse(path.read_text(encoding='utf-8'))
    expected_stages = (
        'feature-parity', 'archive-parity', 'parity-old_val', 'parity-new_development',
        *(f'{action}-{cohort}' for cohort in ('reserve', 'venus', 'full')
          for action in ('prepare', 'extract', 'score', 'analyze')),
    )
    if STAGES != expected_stages:
        raise ValueError('Prediction-seal/label-unlock stage order changed')
    required = [RUN / 'data/audit.json', RUN / 'data/venus_audit.json',
                RUN / 'data/reserve_public.json', RUN / 'data/venus_public.json',
                RUN / 'data/full_public.json']
    for path in required:
        if not path.exists():
            raise FileNotFoundError(path)
    public_counts = {}
    for cohort, count in (('reserve', 58), ('venus', 143), ('full', 1581)):
        rows = read(RUN / f'data/{cohort}_public.json')
        if len(rows) != count or any('target_geometry' in row or 'labels' in row for row in rows):
            raise ValueError('Public cohort size/label separation changed: ' + cohort)
        public_counts[cohort] = len(rows)
    audit = read(RUN / 'data/audit.json')
    venus = read(RUN / 'data/venus_audit.json')
    if not audit['training']['reserve_no_exact_or_near_training_overlap'] or not venus['independent_history_audit_passed']:
        raise ValueError('History isolation audit failed')
    model_path = RUN / 'models/frozen.json'
    model_ready = False
    if model_path.exists():
        seal = read(model_path)
        expected_models = {f'full_{variant}_s{s}' for variant in VARIANTS
                           for s in (1, 2, 3)}
        entries = seal.get('models', {})
        if seal.get('complete') is not True or set(entries) != expected_models:
            raise ValueError('Full model seal does not contain the fixed 18 slots')
        for variant in VARIANTS:
            for s, seed in enumerate(SEEDS, 1):
                entry = entries[f'full_{variant}_s{s}']
                if (entry['variant'] != variant or entry['seed'] != seed or
                        entry['queries'] != 13373 or entry['steps'] != 13376 or
                        sha(entry['path']) != entry['sha256']):
                    raise ValueError(f'Full model identity or hash drift: {variant}_s{s}')
        model_ready = True
    result = dict(check_only=True, eval_script_sha256=sha(EVAL),
                  data_counts=public_counts, full_models_frozen=model_ready,
                  planned_stages=list(STAGES), stage_order_validated=True,
                  labels_opened=False,
                  model_forward_calls=0)
    if show_plan:
        result['stage_plan'] = [dict(stage=stage, command=command(stage, device),
                                     output=str(output_for(stage))) for stage in STAGES]
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)


def wait_for_models(interval, timeout_hours):
    started = time.monotonic()
    last_report = 0.0
    path = RUN / 'models/frozen.json'
    while True:
        if path.exists():
            seal = read(path)
            if seal.get('complete') and len(seal.get('models', {})) == 18:
                state('models_frozen', model_seal_sha256=sha(path))
                return
        training = RUN / 'status.json'
        progress = read(training) if training.exists() else {}
        if progress.get('phase') == 'failed':
            raise RuntimeError('Training failed before the full model seal; inspect ' + str(training))
        elapsed = time.monotonic() - started
        if elapsed >= timeout_hours * 3600:
            raise TimeoutError('Timed out waiting for full ablation model seal')
        if elapsed - last_report >= 30 or last_report == 0:
            state('waiting_for_models', training_phase=progress.get('phase'),
                  training_model=progress.get('model'),
                  training_step=progress.get('step'),
                  elapsed_wait_seconds=round(elapsed, 1))
            last_report = elapsed
        time.sleep(min(interval, 30))


def command(stage, device):
    if stage == 'feature-parity':
        args = ['--stage', 'feature-parity', '--device', 'cpu']
    elif stage == 'archive-parity':
        args = ['--stage', 'archive-parity', '--device', 'cpu']
    elif stage.startswith('parity-'):
        args = ['--stage', 'parity', '--cohort', stage.removeprefix('parity-'),
                '--device', device]
    else:
        args = ['--stage', stage, '--device', device]
    return [sys.executable, '-B', '-X', 'utf8', '-u', str(EVAL), *args]


def output_for(stage):
    if stage == 'feature-parity':
        return OUT / 'parity_features.json'
    if stage == 'archive-parity':
        return OUT / 'parity_archives.json'
    if stage.startswith('parity-'):
        return OUT / 'parity_full' / (stage.removeprefix('parity-') + '.json')
    action, cohort = stage.split('-', 1)
    return OUT / cohort / dict(prepare='protocol.json', extract='features/index.json',
                               score='predictions.json', analyze='analysis.json')[action]


def latest_eval_event():
    path = OUT / 'events.jsonl'
    if not path.exists():
        return None
    with path.open('rb') as stream:
        stream.seek(0, os.SEEK_END)
        size = stream.tell()
        stream.seek(max(0, size - 16384))
        lines = stream.read().decode('utf-8', errors='replace').splitlines()
    for line in reversed(lines):
        try:
            return json.loads(line)
        except json.JSONDecodeError:
            continue
    return None


def stage_progress(stage, event):
    if not isinstance(event, dict):
        return None
    if stage.startswith('extract-'):
        expected = ('extract_public', stage.removeprefix('extract-'))
    elif stage.startswith('score-'):
        expected = ('score_public', stage.removeprefix('score-'))
    else:
        return None
    if (event.get('stage'), event.get('cohort')) != expected:
        return None
    completed, total = event.get('completed'), event.get('total')
    if not isinstance(completed, int) or not isinstance(total, int) or total <= 0:
        return None
    return dict(completed=completed, total=total,
                fraction=round(completed / total, 6),
                elapsed_seconds=event.get('elapsed_seconds'))


def run_stage(stage, device, interval):
    output = output_for(stage)
    cmd = command(stage, device)
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    stdout = LOG_DIR / f'{stage}.stdout.log'
    stderr = LOG_DIR / f'{stage}.stderr.log'
    started = time.monotonic()
    state('running_stage', current_stage=stage, command=cmd, output=str(output),
          stdout_log=str(stdout), stderr_log=str(stderr))
    flags = getattr(subprocess, 'CREATE_NO_WINDOW', 0)
    with stdout.open('ab') as out, stderr.open('ab') as err:
        header = ('\n=== ' + now() + ' ' + stage + ' ===\n').encode()
        out.write(header)
        err.write(header)
        out.flush()
        err.flush()
        process = subprocess.Popen(cmd, cwd=ROOT, stdout=out, stderr=err,
                                   creationflags=flags)
        last_report = 0.0
        try:
            while True:
                returncode = process.poll()
                if returncode is not None:
                    break
                elapsed = time.monotonic() - started
                if elapsed - last_report >= 30 or last_report == 0:
                    event = latest_eval_event()
                    state('running_stage', current_stage=stage, child_pid=process.pid,
                          elapsed_stage_seconds=round(elapsed, 1),
                          latest_eval_event=event,
                          progress=stage_progress(stage, event),
                          stdout_log=str(stdout), stderr_log=str(stderr))
                    last_report = elapsed
                time.sleep(min(interval, 30))
        except BaseException:
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=30)
            raise
    if returncode != 0:
        raise RuntimeError(f'{stage} exited {returncode}; inspect {stderr} and {stdout}')
    if not output.exists():
        raise RuntimeError(f'{stage} exited 0 without expected output {output}')
    receipt = read(output)
    if receipt.get('complete') is not True:
        raise RuntimeError(f'{stage} output is not complete: {output}')
    old = read(STATUS)
    done = list(dict.fromkeys([*old.get('completed_stages', []), stage]))
    state('stage_complete', completed_stages=done, last_completed_stage=stage,
          output=str(output), output_sha256=sha(output),
          elapsed_stage_seconds=round(time.monotonic() - started, 1))


def percent(value):
    return f'{100 * value:+.2f}'


def interval(values):
    return '[' + ', '.join(percent(x) for x in values) + ']'


def directional_statement(title, contrast):
    delta = contrast['delta']
    lo, hi = contrast['ci95']
    direction = '全量较高' if delta > 0 else '全量较低' if delta < 0 else '两者相同'
    separation = '95%配对区间不含0' if lo > 0 or hi < 0 else '95%配对区间含0'
    return (f'{title}：{direction} {abs(100*delta):.2f} 个百分点；'
            f'95% RGB图像簇区间 {interval([lo, hi])}；{separation}。')


def final_readout():
    results = {cohort: read(OUT / cohort / 'analysis.json')
               for cohort in ('reserve', 'venus', 'full')}
    lines = [
        '3k 与全量评分器贡献消融：固定 K16 候选，三个预定种子',
        '所有准确率均为整数点击点落入有效目标框；无候选/无输出保留在分母。',
        '全量训练清单 13,373（有候选拟合 12,886），3k 清单 2,973；'
        '模型均为 22,050 参数，同一冻结 Qwen3-VL-8B 与 V12 候选提议器。',
        '注意：全量数据源配比和总更新步数也变化；matched3k 控制仅匹配全量更新步数。',
    ]
    for cohort, label in (('reserve', 'ScreenSpot-Pro 未见保留集58'),
                          ('venus', 'VenusBench-GD 跨来源未见集143'),
                          ('full', 'ScreenSpot-Pro 官方完整1581')):
        result = results[cohort]
        n = result['n']
        lines += ['', f'{label}；RGB图像簇={result["unique_rgb_images"]}，题数={n}。']
        lines.append('训练量 | 消融 | S1 | S2 | S3 | 三种子平均准确率')
        for scale in ('3k', 'full'):
            for variant in ('aligned', 'no_evidence', 'spatial_shuffle',
                            'attention_only', 'hidden_only', 'token_mean'):
                hits = [result['methods'][f'{scale}_{variant}_s{s}']['hits'] for s in (1, 2, 3)]
                mean = result['summary'][scale][variant]['mean_accuracy']
                lines.append(f'{scale} | {variant} | {hits[0]}/{n} | {hits[1]}/{n} | '
                             f'{hits[2]}/{n} | {100*mean:.2f}%')
        hits = [result['methods'][f'matched3k_aligned_s{s}']['hits'] for s in (1, 2, 3)]
        mean = result['summary']['matched3k_aligned']['mean_accuracy']
        lines.append(f'matched3k | aligned | {hits[0]}/{n} | {hits[1]}/{n} | '
                     f'{hits[2]}/{n} | {100*mean:.2f}%')
        lines.append('配对比较 | 三种子均值差（百分点） | 95% RGB图像簇区间 | 各种子差（百分点）')
        for name, contrast in result['contrasts'].items():
            seed = ', '.join(percent(x) for x in contrast['per_seed_delta'])
            lines.append(f'{name} | {percent(contrast["delta"])} | '
                         f'{interval(contrast["ci95"])} | [{seed}]')
        lines.append('主机制四比较的同时覆盖 98.75% 区间（百分点）：')
        for scale in ('3k', 'full'):
            for variant in ('no_evidence', 'spatial_shuffle'):
                key = f'{scale}_aligned_vs_{variant}'
                lines.append(f'{key}: {interval(result["contrasts"][key]["ci98_75"])}')
        lines.append('区间条件化于预定的固定三次拟合；三种子本身并非重采样单位。')
        lines.append('训练规模判断（仅描述固定协议的配对差）：')
        lines.append(directional_statement('全量对原始3k',
                                           result['contrasts']['full_vs_3k/aligned']))
        lines.append(directional_statement('全量对等更新步3k',
                                           result['contrasts']['full_aligned_vs_matched3k']))
        if cohort == 'reserve':
            lines.append('此队列是同来源未见确认，仅58题；区间宽时不应把方向解释成确定增益。')
        elif cohort == 'venus':
            lines.append('此队列是跨来源未见确认，143题；与 ScreenSpot-Pro 分开解释。')
        if cohort == 'full':
            lines.append('完整榜单含大量历史已观察/近似图片和本次58图；仅供官方整体表现与暴露分层，不是独立泛化确认。')
            lines.append('历史暴露分层（full_aligned 对 3k_aligned，三种子均值）：')
            for category, item in result['history_strata']['history_status'].items():
                lines.append(f'{category}: n={item["n"]}, '
                             f'3k={100*item["aligned"]["3k"]["seed_mean_accuracy"]:.2f}%, '
                             f'full={100*item["aligned"]["full"]["seed_mean_accuracy"]:.2f}%, '
                             f'delta={item["full_minus_3k_pp"]:+.2f}pp')
    lines += ['', '解释边界：消融只识别在固定 V12 候选池内，空间对齐内部证据对评分器的增量作用。',
              'V12 候选提议器本身也读取内部证据；不能把评分器去证据等同于全系统去证据。',
              '58与143分别是小样本同来源和跨来源独立确认；两者按顺序预测封存后才读取标签。',
              '同图像不同指令由RGB图像簇共同重采样；三种子不当作独立测试样本。',
              '不从本实验推断真实点击执行成功率、动作类型、与 MVP 的总体优势或纯数据量因果。',
              '数据效率与泛化判断应先看两组未见确认的 full_vs_3k/aligned 配对区间，'
              '再将完整1581作为带历史暴露分层的描述性结果。', '']
    target = OUT / 'readout.txt'
    atomic_text(target, '\n'.join(lines))
    return target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-only', action='store_true',
                        help='Read-only audit and planned command list; no model forward')
    parser.add_argument('--dry-run', action='store_true',
                        help='Read-only frozen-registry and ordered stage/command audit')
    parser.add_argument('--device', default='cuda:0',
                        help='Readout device for parity/scoring; extraction follows frozen V15 device map')
    parser.add_argument('--poll-seconds', type=int, default=15)
    parser.add_argument('--wait-hours', type=float, default=24)
    args = parser.parse_args()
    if not 1 <= args.poll_seconds <= 30 or args.wait_hours <= 0:
        parser.error('Poll must be 1..30 seconds and wait-hours positive')
    os.chdir(ROOT)
    if args.check_only or args.dry_run:
        check_only(args.device, show_plan=args.dry_run)
        return
    try:
        check_only(args.device)
        wait_for_models(args.poll_seconds, args.wait_hours)
        for stage in STAGES:
            run_stage(stage, args.device, args.poll_seconds)
        report = final_readout()
        state('complete', complete=True, readout=str(report), readout_sha256=sha(report))
    except BaseException as exc:
        state('failed', complete=False, error=repr(exc))
        raise


if __name__ == '__main__':
    main()
