"""Add external references to sealed V16 cases without changing its experiment."""
from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/external_baselines_v16'
V16 = ROOT / 'runs/contribution_v16'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def status(phase, **fields):
    value = dict(phase=phase, complete=phase == 'complete', pid=os.getpid(),
                 updated_at_utc=datetime.now(timezone.utc).isoformat(), **fields)
    write(OUT / 'status.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def register():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT / 'protocol.json').exists():
        return verify()
    assert not (OUT / 'ui_tars').exists(), 'Never register after reference inference'
    completion = read(V16 / 'completion.json')
    assert completion['complete'] and completion['adopted_mainline_unchanged']
    for name, expected in completion['artifact_sha256'].items():
        assert digest(name) == expected, name
    rows = [json.loads(s) for s in (V16 / 'data/independent_test.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(rows) == len({r['sample_id'] for r in rows}) == 300
    manifest = []
    for row in rows:
        converted = dict(row, experiment_split='test')
        assert {k: v for k, v in converted.items() if k != 'experiment_split'} == {
            k: v for k, v in row.items() if k != 'experiment_split'}
        manifest.append(json.dumps(converted, ensure_ascii=False))
    (OUT / 'manifest.jsonl').write_text('\n'.join(manifest) + '\n', encoding='utf-8')
    sources = [Path(__file__), ROOT / 'scripts/run_ui_tars_baseline.py',
               ROOT / 'scripts/audit_ui_tars_baseline.py', ROOT / 'src/clickagent/model_baselines.py',
               ROOT / 'src/clickagent/public_evaluation.py', ROOT / 'scripts/run_target_token_experiment.py',
               ROOT / 'runs/ui_tars_official_metadata.json',
               ROOT / 'models/UI-TARS-1.5-7B/clickagent_download_verified.json',
               ROOT / 'models/UI-TARS-1.5-7B/config.json',
               ROOT / 'models/UI-TARS-1.5-7B/preprocessor_config.json',
               V16 / 'completion.json', V16 / 'evaluation/independent/results.json',
               V16 / 'evaluation/independent/details.json', V16 / 'data/independent_test.jsonl',
               OUT / 'manifest.jsonl']
    protocol = dict(
        registered_at_utc=datetime.now(timezone.utc).isoformat(),
        kind='Post-V16 external-baseline extension on previously evaluated fixed screenshots',
        purpose='Assess complete-method competitiveness; ablations are not external baselines.',
        n=300, sources={'ScreenSpot-Pro': 150, 'VenusBench-GD': 150},
        hashes={str(p): digest(p) for p in sources},
        model=read(ROOT / 'models/UI-TARS-1.5-7B/clickagent_download_verified.json')['official_model'],
        model_revision=read(ROOT / 'models/UI-TARS-1.5-7B/clickagent_download_verified.json')['official_revision'],
        recipe='Unmodified existing official-prompt UI-TARS runtime: FP16, greedy64, max_pixels262144, EOS required, processed-image absolute coordinates, all300 denominator.',
        result_policy='Preserve all outputs; no prompt, parser, seed, checkpoint or cohort changes based on outcomes. Failures count as misses.',
        comparison='Full adopted3-seed mean, original Qwen completed-lax, our candidates plus GUI verifier, and UI-TARS. No ablations counted as external model competitors.',
        statistics='Exploratory source-stratified paired bootstrap2000 on fixed-model seed-averaged differences;95% intervals. Existing V16 primary tests remain unchanged.',
        limitations=['Previously evaluated cohort; extension is not a new independent confirmation.',
                     '262144 pixels is a controlled per-view cap, not a reproduction of published high-resolution scores.',
                     'Our method has additional local views; this is not a matched-FLOPs comparison.',
                     'Existing GUI reference is our proposal pool plus the2B verifier, not the full GUI-Actor grounding system.',
                     'No demonstrated latency, total-compute or sample-efficiency superiority.'],
        pending_external_panel=[
            {'method': 'TAG', 'role': 'Closest frozen attention grounding reference', 'status': 'not_reproduced', 'source': 'https://arxiv.org/html/2412.10840v1'},
            {'method': 'GUI-Actor and LiteTrain', 'role': 'Complete action-head system; verifier reference alone is insufficient', 'status': 'not_reproduced', 'source': 'https://github.com/microsoft/GUI-Actor'},
            {'method': 'GUI-AIMA', 'role': 'Learned intrinsic-attention aggregation', 'status': 'not_reproduced', 'source': 'https://github.com/sjz5202/GUI-AIMA'},
            {'method': 'Qwen3-VL plus MVP', 'role': 'Same-backbone multi-view reference', 'status': 'not_reproduced', 'source': 'https://github.com/ZJUSCL/MVP'},
            {'method': 'Qwen3-VL plus Re-Prefill', 'role': 'Same-backbone internal-attention inference enhancement', 'status': 'not_reproduced', 'source': 'https://arxiv.org/html/2605.12549v1'},
            {'method': 'Qwen3-VL official grounding prompt and resolution', 'role': 'Check low native score under existing32-token CLICK/WAIT and low-resolution recipe', 'status': 'not_reproduced'}],
    )
    write(OUT / 'protocol.json', protocol)
    status('registered', total=300)
    return protocol


def verify():
    protocol = read(OUT / 'protocol.json')
    for name, expected in protocol['hashes'].items():
        assert digest(name) == expected, f'Sealed input/source changed: {name}'
    return protocol


def run_driver(script):
    with (OUT / f'{Path(script).stem}.stdout.log').open('a', encoding='utf-8') as stdout, \
         (OUT / f'{Path(script).stem}.stderr.log').open('a', encoding='utf-8') as stderr:
        subprocess.run([sys.executable, '-B', '-X', 'utf8', '-u', str(ROOT / 'scripts' / script),
                        '--output', str(OUT)], cwd=ROOT, stdout=stdout, stderr=stderr, check=True)


def summarize():
    verify()
    ui = read(OUT / 'ui_tars/summary.json')
    audit = read(OUT / 'ui_tars/final_audit.json')
    assert ui['complete'] and ui['n'] == audit['same_test_images'] == 300 and audit['complete']
    details = read(V16 / 'evaluation/independent/details.json')
    predictions = [json.loads(s) for s in (OUT / 'ui_tars/predictions.jsonl').read_text(encoding='utf-8').splitlines()]
    lookup = {r['sample_id']: r for r in predictions}
    assert len(lookup) == 300 and set(lookup) == {r['sample_id'] for r in details}
    cases = {r['sample_id']: r for r in [json.loads(s) for s in (OUT / 'manifest.jsonl').read_text(encoding='utf-8').splitlines()]}
    from clickagent.public_evaluation import point_in_geometry
    for row in predictions:
        c = cases[row['sample_id']]
        p = row['point']
        expected = p is not None and point_in_geometry([p[0] / c['image_size'][0], p[1] / c['image_size'][1]], c['target_geometry'])
        assert row['hit'] == bool(expected)
    assert sum(r['hit'] for r in predictions) == ui['hits']
    full_seeds = np.asarray([[r['methods'][f'aligned_s{i}']['top1'] for r in details] for i in [1, 2, 3]], dtype=float)
    full = full_seeds.mean(0)
    external = np.asarray([lookup[r['sample_id']]['hit'] for r in details], dtype=float)
    native = np.asarray([r['methods']['qwen_native_completed_lax']['top1'] for r in details], dtype=float)
    gui = np.asarray([r['methods']['gui_verifier_k16']['top1'] for r in details], dtype=float)
    sources = np.asarray([r['source'] for r in details])
    rng = np.random.default_rng(20261105)
    draws = np.concatenate([rng.choice(np.flatnonzero(sources == s), size=(2000, int((sources == s).sum())), replace=True) for s in sorted(set(sources))], axis=1)
    delta = full - external
    ci = np.quantile(delta[draws].mean(1), [.025, .975])
    vectors = {'our_full': full, 'our_candidates_plus_gui_verifier': gui,
               'qwen_native_completed_lax': native, 'ui_tars_1_5_7b': external}
    methods = {k: dict(mean_accuracy_percent=float(v.mean() * 100),
                      by_source_percent={s: float(v[sources == s].mean() * 100) for s in sorted(set(sources))}) for k, v in vectors.items()}
    for k in methods:
        methods[k]['observed_rank'] = 1 + sum(v['mean_accuracy_percent'] > methods[k]['mean_accuracy_percent'] + 1e-10 for v in methods.values())
    result = dict(complete=True, n=300, methods=methods,
                  full_vs_ui_tars=dict(delta_pp=float(delta.mean() * 100), exploratory_ci95_pp=(ci * 100).tolist(),
                  conditional_on_fixed_models=True,
                  per_seed=[dict(hits=int(v.sum()), wins=int(((v == 1) & (external == 0)).sum()), losses=int(((v == 0) & (external == 1)).sum())) for v in full_seeds]),
                  scope='Fixed previously evaluated V16 subset; external extension only; no official full-benchmark or new confirmation claim.')
    write(OUT / 'comparison.json', result)
    table = ''.join(f'<tr><td>{v["observed_rank"]}</td><td>{html.escape(k)}</td><td>{v["mean_accuracy_percent"]:.2f}%</td></tr>' for k, v in sorted(methods.items(), key=lambda kv: -kv[1]['mean_accuracy_percent']))
    (OUT / 'report.html').write_text('<!doctype html><html lang="zh"><meta charset="utf-8"><title>V16 外部基线补充比较</title><style>body{font:17px system-ui;max-width:1000px;margin:40px auto;line-height:1.7}td,th{border:1px solid #ccc;padding:10px}table{border-collapse:collapse}</style><h1>固定300条上的外部基线补充比较</h1><p>V16结果、训练集和主模型保持固定；这是已评估子集上的后续比较。</p><table><tr><th>观测排名</th><th>方法</th><th>准确率</th></tr>'+table+'<p>GUI项使用我们的候选与外部验证器，非完整GUI-Actor系统。尚未完成TAG、GUI-AIMA、MVP和Re-Prefill复现。该表不能证明对全部已有工作的优势。</p><pre>'+html.escape(json.dumps(result, ensure_ascii=False, indent=2))+'</pre></html>', encoding='utf-8')
    files = [OUT / 'protocol.json', OUT / 'comparison.json', OUT / 'ui_tars/summary.json', OUT / 'ui_tars/final_audit.json', OUT / 'ui_tars/predictions.jsonl']
    write(OUT / 'completion.json', dict(complete=True, completed_at_utc=datetime.now(timezone.utc).isoformat(), artifact_sha256={str(p): digest(p) for p in files}, mainline_unchanged=True))
    status('complete', completed=300, total=300)


def main():
    register()
    try:
        verify()
        status('running_ui_tars', total=300)
        run_driver('run_ui_tars_baseline.py')
        status('auditing_ui_tars', total=300)
        run_driver('audit_ui_tars_baseline.py')
        summarize()
    except Exception as error:
        status('failed', error=repr(error))
        raise


if __name__ == '__main__':
    main()
