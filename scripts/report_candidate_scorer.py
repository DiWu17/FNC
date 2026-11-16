"""Render audited V11 results, preserving the prespecified primary and cohorts.

No model imports/inference, fitting, bootstrap recomputation or test selection.
Incomplete audits/results are refused before loading outcome details or writing
outputs. Figures use static CPU plotting; PNGs are embedded in one HTML file.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


VARIANTS = ('internal_full', 'internal_pooled', 'internal_global', 'visual')
BUDGETS = (4, 8, 16, 32)
SEEDS = ('s1', 's2')
PRIMARY = 'internal_full_s1_K16'
ROBUSTNESS = 'internal_full_s2_K16'
BASELINES = ('old_critic_K16', 'gui_actor_K16', 'gui_actor_K4', 'global_top1', 'qwen_native_completed_lax', 'ui_tars')
COLORS = dict(internal_full='#227d9c', internal_pooled='#ce8d36', internal_global='#748d52', visual='#956aa6')
LABELS = dict(internal_full='Internal full', internal_pooled='Early token pool', internal_global='Global view only', visual='Dedicated visual')
STAGE_KEYS = ('hits', 'empty_pools', 'nonempty_pool_without_correct_point', 'selector_missed_correct_point')
STAGE_LABELS = ('Hit', 'Empty pool', 'No correct proposed point', 'Wrong selection in covered pool')
STAGE_COLORS = ('#299b86', '#a8afb7', '#d9a454', '#cc6b65')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def rows(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def expected_methods():
    return {f'{variant}_{seed}_K{budget}' for variant in VARIANTS for seed in SEEDS for budget in BUDGETS} | set(BASELINES)


def head_names(budget=16):
    return [f'{v}_{s}_K{budget}' for v in VARIANTS for s in SEEDS]


def primary_table_names():
    return head_names()+['old_critic_K16', 'gui_actor_K16', 'global_top1', 'qwen_native_completed_lax', 'ui_tars']


def short(name):
    for variant in VARIANTS:
        if name.startswith(variant+'_'):
            return LABELS[variant]+' / '+name[len(variant)+1:].replace('_K', ' / K')
    return {'old_critic_K16': 'Old point critic / K16', 'gui_actor_K16': 'GUI-Actor / K16',
        'gui_actor_K4': 'GUI-Actor / K4', 'global_top1': 'Global top1',
        'qwen_native_completed_lax': 'Qwen native', 'ui_tars': 'UI-TARS 1.5-7B'}.get(name, name)


def color(name):
    return next((c for variant, c in COLORS.items() if name.startswith(variant+'_')), '#66717e')


def number(value, digits=1):
    return '—' if value is None else f'{value:.{digits}f}'


def rate(value):
    return '—' if value is None else f'{100*value:.1f}%'


def hit_text(value):
    return f"{value['hits']}/{value['n']} ({rate(value['accuracy'])})"


def interval(value, percent=True):
    scale = 100 if percent else 1
    return f"[{scale*value[0]:.1f}, {scale*value[1]:.1f}]"


def hashed_paths(mapping):
    return {str(Path(path).resolve()): digest for path, digest in mapping.items()}


def load_bundle(out):
    """Fail closed before reading fresh outcome details or creating figures."""
    out = Path(out)
    audit = read(out/'final_audit.json')
    require(audit.get('complete') is True and set(audit['splits']) == {'test', 'diagnostic'}, 'Complete two-cohort final audit required')
    frozen, protocol = read(out/'frozen.json'), read(out/'protocol.json')
    require(frozen.get('complete') is True and frozen['fresh_test_features_at_freeze'] == 0, 'Invalid freeze')
    require(audit['frozen_sha256'] == sha(out/'frozen.json'), 'Frozen identity changed after audit')
    require(audit['protocol_sha256'] == sha(out/'protocol.json'), 'Protocol changed after audit')
    require(frozen['primary'] == 'internal_full_s1' and frozen['primary_budget'] == 16, 'Primary must stay protocol-selected')
    require(protocol['primary_budget'] == 16 and protocol['budgets'] == list(BUDGETS), 'Unexpected candidate budgets')
    require(protocol['variants'] == list(VARIANTS), 'Expected all four variants')
    require(sha(out/'manifest.jsonl') == protocol['manifest_sha256'], 'Manifest changed')
    require(sha(out/'diagnostic_manifest.jsonl') == protocol['diagnostic_manifest_sha256'], 'Diagnostic manifest changed')
    require(sha(out/'data_selection.json') == protocol['data_selection_sha256'], 'Selection changed')
    for path, digest in audit['source_sha256'].items():
        require(sha(Path(path)) == digest, f'Auditor/evaluator changed: {path}')
    expected = expected_methods()
    require(len(expected) == 38, 'Internal method specification error')
    checkpoint_names = {f'{v}_{s}' for v in VARIANTS for s in SEEDS}
    require(set(frozen['models']) == checkpoint_names, 'Expected all eight frozen checkpoints')
    sealed_files = hashed_paths(frozen['files'])
    histories = {}
    for name in sorted(checkpoint_names):
        entry = frozen['models'][name]; path = Path(entry['file'])
        require(sha(path) == entry['sha256'], f'Checkpoint changed: {name}')
        history_path = path.with_suffix('.json')
        require(sealed_files.get(str(path.resolve())) == entry['sha256'], f'Checkpoint missing from freeze: {name}')
        require(sealed_files.get(str(history_path.resolve())) == sha(history_path), f'Training history changed after freeze: {name}')
        value = read(history_path)
        require(value['complete'] and value['checkpoint_sha256'] == entry['sha256'], f'Incomplete model: {name}')
        require(value['protocol_sha256'] == audit['protocol_sha256'], f'Training protocol drift: {name}')
        require([r['epoch'] for r in value['history']] == list(range(protocol['optimization']['epochs']+1)), f'Incomplete trajectory: {name}')
        selected = max(value['history'], key=lambda r: (r['validation']['hits'], -r['validation']['loss']))
        require(value['selected'] == entry['selected'] == selected, f'Validation selection mismatch: {name}')
        require(value['selected_validation_replay']['hits'] == selected['validation']['hits'], f'Validation replay mismatch: {name}')
        histories[name] = value
    evidence = hashed_paths(audit['evaluation_sha256'])
    summaries, details, predictions = {}, {}, {}
    for split in ('test', 'diagnostic'):
        folder = out/'evaluation'/split
        for filename in ('summary.json', 'details.json', 'predictions.jsonl'):
            path = folder/filename
            require(evidence.get(str(path.resolve())) == sha(path), f'Audited evaluation changed: {path}')
        summary = read(folder/'summary.json')
        require(summary['complete'] and summary['n'] == 192 and summary['split'] == split, f'Incomplete {split}')
        require(summary['primary'] == PRIMARY and summary['robustness'] == ROBUSTNESS, 'Test outcome cannot change primary')
        require(summary['exploratory_reused_images'] == (split == 'diagnostic'), 'Cohort meaning changed')
        require(summary['selection_used_this_cohort'] is False, 'Evaluation used for selection')
        require(summary['frozen_sha256'] == audit['frozen_sha256'] and summary['protocol_sha256'] == audit['protocol_sha256'], 'Evaluation identity drift')
        require(set(summary['methods']) == expected, 'Missing/extra methods')
        check = audit['splits'][split]
        require(check['n'] == 192 and check['methods'] == 38 and check['predictions_checked'] == 192*38, 'Audit coverage incomplete')
        require(check['replay']['complete'] and check['replay']['integer_points_exact'], 'CPU replay failed')
        values = read(folder/'details.json'); predicted = rows(folder/'predictions.jsonl')
        require(len(values) == len({v['sample_id'] for v in values}) == 192, 'Detail sample coverage incomplete')
        by_id = {v['sample_id']: v for v in values}
        lookup = {(v['sample_id'], v['method']): v for v in predicted}
        require(len(predicted) == len(lookup) == 192*38, 'Prediction coverage incomplete')
        for value in values:
            require(set(value['methods']) == expected, 'Incomplete sample methods')
            for method, pred in value['methods'].items():
                row = dict(sample_id=value['sample_id'], dataset=value['dataset'], method=method, **pred)
                require(lookup.get((value['sample_id'], method)) == row, 'Details/predictions differ')
        for method, m in summary['methods'].items():
            ms = [v['methods'][method] for v in values]
            require(m['n'] == 192 and m['hits'] == sum(v['hit'] for v in ms), f'Metric mismatch: {method}')
            require(abs(m['accuracy']-m['hits']/192) < 1e-12, 'Accuracy denominator mismatch')
            require(m['points'] == sum(v['point'] is not None for v in ms), 'Point coverage mismatch')
            if m['pool_budget'] is not None:
                require(sum(m[k] for k in STAGE_KEYS) == 192, 'Failure stages lose samples')
                require(m['pool_available'] == summary['pools'][str(m['pool_budget'])]['oracle'], 'Pool oracle mismatch')
        require(len(summary['paired_comparisons']) == 61, 'Missing planned paired comparisons')
        for comparison in summary['paired_comparisons']:
            method, reference = comparison['method'], comparison['reference']
            require(method in expected and reference in expected, 'Unknown paired method')
            diff = np.array([int(v['methods'][method]['hit'])-int(v['methods'][reference]['hit']) for v in values])
            require(comparison['wins'] == int((diff > 0).sum()) and comparison['losses'] == int((diff < 0).sum())
                    and comparison['ties'] == int((diff == 0).sum()), 'Paired counts mismatch')
            require(abs(comparison['delta']-diff.mean()) < 1e-12, 'Paired delta mismatch')
            ci = comparison['ci95']
            require(len(ci) == 2 and np.isfinite(ci).all() and -1 <= ci[0] <= ci[1] <= 1, 'Invalid proportion CI')
        summaries[split], details[split], predictions[split] = summary, by_id, predicted
    # Manifests/GT are first used here, AFTER both completed audited outcomes.
    manifest = rows(out/'manifest.jsonl')
    cases = {'test': {c['sample_id']: c for c in manifest if c['experiment_split'] == 'test'},
             'diagnostic': {c['sample_id']: c for c in rows(out/'diagnostic_manifest.jsonl')}}
    for split in cases:
        require(set(cases[split]) == set(details[split]), 'Manifest/evaluation cohort mismatch')
    require(not (set(cases['test']) & set(cases['diagnostic'])), 'Cohorts overlap')
    selection = read(out/'data_selection.json'); selection_audit = read(out/'data_selection_audit.json')
    require(selection['complete'] and selection_audit['complete'], 'Data selection audit incomplete')
    require(selection_audit['manifest_sha256'] == protocol['manifest_sha256'], 'Data selection audit drift')
    return dict(out=out, audit=audit, frozen=frozen, protocol=protocol, histories=histories,
                summaries=summaries, details=details, predictions=predictions, cases=cases,
                selection=selection, selection_audit=selection_audit)


def save_figure(fig, folder, name):
    files = []
    for extension in ('png', 'pdf'):
        path = folder/(name+'.'+extension)
        fig.savefig(path, dpi=160, bbox_inches='tight', facecolor='white'); files.append(path)
    plt.close(fig)
    return files


def plot_design(folder):
    fig, ax = plt.subplots(figsize=(14, 7.2)); ax.axis('off'); ax.set(xlim=(0, 1), ylim=(0, 1))
    boxes = [(.02, .73, .26, .20, 'Frozen Qwen3-VL + decoder\nFull frame + independent crop\nGlobal Top(K/2) + crop Top(K/2)'),
             (.37, .73, .26, .20, 'Shared actual candidate points\nK16 primary; K4/8/32 secondary\nHeads score fixed coordinates'),
             (.73, .73, .25, .20, 'Supervision: training points only\nMulti-positive listwise + null + BCE\nValidation selects epoch; seed1 fixed'),
             (.03, .33, .44, .28, 'INTERNAL route: new trainable projections\nAll target tokens x 96 native head identities\n3x3 image + 5x5 premerge neighborhoods\nGlobal image-token attention; both views at every point\nCandidate-conditioned evidence -> token max/logmeanexp\n96 head channels -> set comparison + null'),
             (.54, .33, .43, .28, 'DEDICATED VISUAL route: frozen Qwen vision\nONE extra full-frame encode up to 1,048,576 pixels\nShared multi-scale point-centered ROI features\nAll generated target tokens condition each candidate\nCandidate-set comparison + null\nNo GUI-Actor features, labels or teacher scores'),
             (.03, .04, .44, .16, 'Matched internal controls (same parameters)\nEarly token pool: homogenize tokens BEFORE interaction\nGlobal only: mask crop; retain global neighborhoods'),
             (.54, .04, .43, .16, 'Common evaluation + external references\nFixed integer point inside target geometry\nSame-pool GUI-Actor / old critic; Qwen / UI-TARS\nStatic positive localization, not executed UI actions')]
    for x, y, w, h, text in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor='#eef4f7', edgecolor='#718b9d', lw=1.2))
        ax.text(x+w/2, y+h/2, text, ha='center', va='center', fontsize=9.7, linespacing=1.45)
    for a, b in (((.28, .83), (.37, .83)), ((.63, .83), (.73, .83)), ((.50, .73), (.25, .61)),
                 ((.50, .73), (.75, .61)), ((.25, .33), (.25, .20)), ((.75, .33), (.75, .20))):
        ax.annotate('', b, a, arrowprops=dict(arrowstyle='->', color='#45687d', lw=1.5))
    return save_figure(fig, folder, 'design')


def plot_scores(summary, folder, name):
    names = primary_table_names(); methods = summary['methods']; y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(11.8, 7.6))
    values = np.array([methods[n]['accuracy']*100 for n in names])
    ci = np.array([methods[n]['ci95'] for n in names])*100
    ax.barh(y, values, color=[color(n) for n in names], alpha=.82)
    # Draw interval endpoints directly, even if a percentile interval does not
    # contain the point estimate. Counts occupy a separate right-hand column.
    ax.hlines(y, ci[:, 0], ci[:, 1], color='#202c35', lw=1)
    ax.vlines(ci[:, 0], y-.08, y+.08, color='#202c35', lw=1)
    ax.vlines(ci[:, 1], y-.08, y+.08, color='#202c35', lw=1)
    oracle = summary['pools']['16']['oracle']/summary['n']*100
    ax.axvline(oracle, ls='--', color='#a2583d', lw=1.4, label='K16 pool oracle (pool methods only)')
    for i, n in enumerate(names):
        ax.text(112, i, f"{methods[n]['hits']}/{summary['n']}", va='center', fontsize=9, ha='right')
    ax.set(yticks=y, yticklabels=[short(n) for n in names], xlabel='Point hit rate (%)', xlim=(0, 114), xticks=np.arange(0, 101, 20),
           title='Fresh independent test / K16' if summary['split'] == 'test' else 'Reused v9 cohort / exploratory only')
    ax.invert_yaxis(); ax.grid(axis='x', alpha=.2); ax.set_axisbelow(True)
    ax.legend(loc='upper center', bbox_to_anchor=(.5, -.095), frameon=False, fontsize=9)
    fig.tight_layout()
    return save_figure(fig, folder, name)


def plot_pool_curves(summary, folder):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5)); x = np.arange(4)
    oracle = [summary['pools'][str(k)]['oracle']/summary['n']*100 for k in BUDGETS]
    axes[0].plot(x, oracle, '--D', color='#343d46', label='Candidate oracle', lw=2)
    for variant in VARIANTS:
        for seed in SEEDS:
            names = [f'{variant}_{seed}_K{k}' for k in BUDGETS]
            for ax, metric in zip(axes, ('accuracy', 'conditional_selection_accuracy')):
                vals = [summary['methods'][n][metric] for n in names]
                ax.plot(x, [v*100 if v is not None else np.nan for v in vals],
                    '-o' if seed == 's1' else ':s', color=COLORS[variant], alpha=1 if seed == 's1' else .6,
                    label=LABELS[variant]+' / '+seed, lw=1.7 if seed == 's1' else 1.1, markersize=4)
    axes[0].scatter([0, 2], [summary['methods'][f'gui_actor_K{k}']['accuracy']*100 for k in (4, 16)],
                     marker='*', s=100, c='#ad4e46', label='GUI-Actor reference', zorder=5)
    for ax, title in zip(axes, ('Coverage and selected point hit', 'Selection given a correct point exists')):
        ax.set(xticks=x, xticklabels=BUDGETS, xlabel='Prespecified candidate budget', ylabel='Rate (%)', ylim=(0, 103), title=title)
        ax.grid(alpha=.18)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, -.11), ncol=4, fontsize=8.5, frameon=False)
    fig.tight_layout(rect=(0, .055, 1, 1))
    return save_figure(fig, folder, 'pool_and_selection')


def plot_failure(summary, folder):
    names = head_names()+['old_critic_K16', 'gui_actor_K16']
    fig, ax = plt.subplots(figsize=(12.6, 6.8)); left = np.zeros(len(names)); y = np.arange(len(names))
    for key, label, c in zip(STAGE_KEYS, STAGE_LABELS, STAGE_COLORS):
        values = np.array([summary['methods'][n][key] for n in names])/summary['n']*100
        ax.barh(y, values, left=left, color=c, label=label)
        for i, value in enumerate(values):
            if value >= 5:
                ax.text(left[i]+value/2, i, str(summary['methods'][names[i]][key]), ha='center', va='center', fontsize=9)
        left += values
    ax.set(yticks=y, yticklabels=[short(n) for n in names], xlim=(0, 100), xlabel='All 192 samples (%)', title='Fresh K16 failure decomposition; empty cases retained')
    ax.invert_yaxis(); ax.legend(loc='upper center', bbox_to_anchor=(.5, -.12), ncol=2, frameon=False, fontsize=9)
    fig.tight_layout()
    return save_figure(fig, folder, 'failure_stages')


def plot_datasets(summary, folder):
    names = [PRIMARY, ROBUSTNESS, 'internal_pooled_s1_K16', 'internal_global_s1_K16', 'visual_s1_K16', 'gui_actor_K16', 'ui_tars']
    datasets = sorted(summary['methods'][PRIMARY]['datasets']); fig, axes = plt.subplots(1, len(datasets), figsize=(15, 5.8), sharey=True)
    for ax, dataset in zip(np.atleast_1d(axes), datasets):
        values = [summary['methods'][n]['datasets'][dataset]['accuracy']*100 for n in names]
        ax.bar(np.arange(len(names)), values, color=[color(n) for n in names])
        ax.set(xticks=np.arange(len(names)), xticklabels=[short(n) for n in names], title=dataset, ylim=(0, 103))
        ax.tick_params(axis='x', labelrotation=60, labelsize=8); ax.grid(axis='y', alpha=.2); ax.set_axisbelow(True)
        for i, n in enumerate(names):
            value = summary['methods'][n]['datasets'][dataset]
            ax.text(i, values[i]+1, f"{value['hits']}/{value['n']}", ha='center', fontsize=8)
    np.atleast_1d(axes)[0].set_ylabel('Point hit rate (%)'); fig.tight_layout()
    return save_figure(fig, folder, 'datasets')


def plot_training(bundle, folder):
    histories = bundle['histories']; names = [f'{v}_{s}' for v in VARIANTS for s in SEEDS]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8)); y = np.arange(len(names))
    train = [histories[n]['selected_training_evaluation']['hits']/histories[n]['selected_training_evaluation']['n']*100 for n in names]
    val = [histories[n]['selected_validation_replay']['hits']/histories[n]['selected_validation_replay']['n']*100 for n in names]
    axes[0].barh(y-.17, train, height=.32, label='Train at selected epoch', color='#83acba')
    axes[0].barh(y+.17, val, height=.32, label='Validation at same epoch', color='#d4a464')
    axes[0].set(yticks=y, yticklabels=[short(n+'_K16') for n in names], xlim=(0, 100), xlabel='K16 point hit rate (%)')
    axes[0].invert_yaxis(); axes[0].legend(loc='lower right', fontsize=8, frameon=False)
    for n in names:
        h = histories[n]; variant = n.rsplit('_s', 1)[0]
        values = [row['validation']['hits']/row['validation']['n']*100 for row in h['history']]
        line, = axes[1].plot([row['epoch'] for row in h['history']], values,
            '-' if n.endswith('s1') else ':', color=COLORS[variant], label=short(n+'_K16'), lw=1.4)
        selected = h['selected']; axes[1].scatter(selected['epoch'], selected['validation']['hits']/selected['validation']['n']*100,
                                               color=line.get_color(), marker='o' if n.endswith('s1') else 's', s=25)
    axes[1].set(xlabel='Training epoch', ylabel='Validation K16 point hit rate (%)', title='Validation curves and selected epochs')
    axes[1].legend(loc='upper center', bbox_to_anchor=(.5, -.15), ncol=2, fontsize=8, frameon=False)
    for ax in axes:
        ax.grid(alpha=.17); ax.set_axisbelow(True)
    fig.tight_layout()
    return save_figure(fig, folder, 'training_validation')


def key_comparisons(summary):
    refs = [ROBUSTNESS]+[f'{v}_{s}_K16' for v in VARIANTS[1:] for s in SEEDS]+['old_critic_K16', 'gui_actor_K16', 'qwen_native_completed_lax', 'ui_tars']
    lookup = {(c['method'], c['reference']): c for c in summary['paired_comparisons']}
    return [lookup[PRIMARY, name] for name in refs]


def plot_paired(summary, folder):
    comparisons = key_comparisons(summary); fig, ax = plt.subplots(figsize=(12.3, 6.8))
    for i, value in enumerate(comparisons):
        lo, hi = np.array(value['ci95'])*100; delta = value['delta']*100
        ax.plot([lo, hi], [i, i], color='#286f8d', lw=2); ax.scatter(delta, i, color='#204f66', s=30)
    ax.axvline(0, color='#8e969e', lw=1, ls='--')
    ax.set(yticks=np.arange(len(comparisons)), yticklabels=[short(c['reference']) for c in comparisons],
           xlabel='Primary minus reference (percentage points), paired 95% CI', title='Fresh cohort: fixed primary comparisons')
    ax.invert_yaxis(); ax.grid(axis='x', alpha=.18); fig.tight_layout()
    return save_figure(fig, folder, 'paired_intervals')


def plot_compute(summary, folder):
    timing = summary['timing']; fig, axes = plt.subplots(1, 3, figsize=(17.5, 6), gridspec_kw={'width_ratios': [1.5, 1.1, .7]})
    names = head_names(); y = np.arange(8)
    for marker, key, label in (('o', 'median_seconds', 'Median'), ('|', 'p95_seconds', 'p95')):
        values = [timing['cached_heads'][n][key]*1000 for n in names]
        axes[0].scatter(values, y, marker=marker, s=65, label=label, color='#286f8d' if marker == 'o' else '#d59649')
    axes[0].set(yticks=y, yticklabels=[short(n) for n in names], xlabel='CPU milliseconds / case', title='Cached K16 head scoring + conversion')
    axes[0].invert_yaxis(); axes[0].legend(loc='upper center', bbox_to_anchor=(.5, -.13), ncol=2, frameon=False)
    stages = [('Qwen full view', timing['frozen_qwen_extraction']['global_view']), ('Qwen old crop', timing['frozen_qwen_extraction']['local_view']),
              ('Extra visual encoding', timing['extra_visual_encoding']), ('Native text baseline', timing['native'])]
    for i, (label, item) in enumerate(stages):
        if item['median_seconds'] is not None:
            axes[1].plot([item['median_seconds'], item['p95_seconds']], [i, i], color='#74868b', lw=2)
            axes[1].scatter(item['median_seconds'], i, color='#74868b', s=30)
    axes[1].set(yticks=np.arange(4), yticklabels=[x[0] for x in stages], xlabel='Seconds / recorded stage', title='Separate extraction stages; do not sum')
    axes[1].invert_yaxis()
    gui = timing['gui_reference']['measured_batches']
    if gui['n']:
        axes[2].bar(['Median', 'p95'], [gui['median_seconds'], gui['p95_seconds']], color=['#9a728e', '#c4a4bd'])
    else:
        axes[2].text(.5, .5, 'Scores reused\nNo new batch timings', ha='center', va='center', transform=axes[2].transAxes)
    axes[2].set(ylabel='Seconds / GPU batch', title='GUI-Actor reference batches')
    for ax in axes:
        ax.grid(alpha=.18); ax.set_axisbelow(True)
    fig.tight_layout()
    return save_figure(fig, folder, 'compute_stages')


def select_examples(bundle, limit):
    values = bundle['details']['test']; cases = bundle['cases']['test']; chosen = []
    predicates = [
        ('内部选择正确、GUI验证器选择错误', 'gui_actor_K16', lambda d: d['methods'][PRIMARY]['hit'] and not d['methods']['gui_actor_K16']['hit']),
        ('候选包含正确点，但内部排序失误', 'gui_actor_K16', lambda d: not d['methods'][PRIMARY]['hit'] and d['pools']['16']['point_oracle'] and d['methods']['gui_actor_K16']['hit']),
        ('候选池本身没有正确点', 'visual_s1_K16', lambda d: not d['pools']['16']['point_oracle']),
        ('第二种子救回的首种子失误', ROBUSTNESS, lambda d: not d['methods'][PRIMARY]['hit'] and d['methods'][ROBUSTNESS]['hit']),
        ('完整交互胜过早池化的案例', 'internal_pooled_s1_K16', lambda d: d['methods'][PRIMARY]['hit'] and not d['methods']['internal_pooled_s1_K16']['hit']),
        ('早池化控制胜过完整交互的案例', 'internal_pooled_s1_K16', lambda d: not d['methods'][PRIMARY]['hit'] and d['methods']['internal_pooled_s1_K16']['hit']),
        ('专用视觉路线救回内部路线失误', 'visual_s1_K16', lambda d: not d['methods'][PRIMARY]['hit'] and d['methods']['visual_s1_K16']['hit']),
        ('crop视图带来改善的案例', 'internal_global_s1_K16', lambda d: d['methods'][PRIMARY]['hit'] and not d['methods']['internal_global_s1_K16']['hit']),
        ('普通命中示例', 'gui_actor_K16', lambda d: d['methods'][PRIMARY]['hit']),
    ]
    used = set()
    for label, reference, predicate in predicates:
        if len(chosen) >= limit:
            break
        found = next((sid for sid in sorted(values) if sid not in used and predicate(values[sid])), None)
        if found is not None:
            chosen.append(dict(label=label, reference=reference, detail=values[found], case=cases[found])); used.add(found)
    return chosen


def plot_example(example, path):
    case, detail, reference = example['case'], example['detail'], example['reference']
    with Image.open(case['image']) as opened:
        image = opened.convert('RGB')
    w, h = image.size; geometry = case['target_geometry']
    polygons = ([[[a, b], [c, b], [c, d], [a, d], [a, b]] for a, b, c, d in geometry['boxes']]
                if geometry['type'] == 'bbox' else [list(p)+[p[0]] for p in geometry['polygons']])
    vertices = np.concatenate(polygons); low, high = vertices.min(0), vertices.max(0)
    margin = np.maximum((high-low)*1.25, [120/w, 120/h])
    box = [round(max(0, low[0]-margin[0])*w), round(max(0, low[1]-margin[1])*h),
           round(min(1, high[0]+margin[0])*w), round(min(1, high[1]+margin[1])*h)]
    overview = image.copy(); overview.thumbnail((2200, 1500))
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={'width_ratios': [1.4, 1]})
    axes[0].imshow(overview, extent=[0, w, h, 0]); axes[1].imshow(image.crop(box), extent=[box[0], box[2], box[3], box[1]])
    for ax in axes:
        points = detail['pools']['16']['points']
        if points:
            arr = np.asarray(points); ax.scatter(arr[:, 0], arr[:, 1], s=14, facecolors='none', edgecolors='#b0b0b0', alpha=.65, label='Actual K16 pool')
        for poly in polygons:
            xy = np.asarray(poly)*[w, h]; ax.plot(xy[:, 0], xy[:, 1], color='#50e046', lw=2)
        for method, marker, c, label in ((PRIMARY, '+', '#1c8fe1', 'Internal full S1'), (reference, 'x', '#f58437', short(reference))):
            point = detail['methods'][method]['point']
            if point is not None:
                ax.scatter(*point, marker=marker, c=c, s=120, lw=2, label=label)
        ax.axis('off')
    axes[0].set(xlim=(0, w), ylim=(h, 0), title='Original screenshot, full target geometry and real candidates')
    axes[1].set(xlim=(box[0], box[2]), ylim=(box[3], box[1]), title='Original-pixel target neighborhood (report only)')
    handles, labels = axes[0].get_legend_handles_labels(); fig.legend(handles, labels, loc='lower center', ncol=3, frameon=False, fontsize=9)
    fig.tight_layout(rect=(0, .06, 1, 1)); fig.savefig(path, dpi=150, bbox_inches='tight'); plt.close(fig)


def table_html(headers, values):
    def cell(value, tag):
        return '<'+tag+'>'+html.escape(str(value))+'</'+tag+'>'
    return '<div class="table"><table><thead><tr>'+''.join(cell(v, 'th') for v in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join(cell(v, 'td') for v in row)+'</tr>' for row in values)+'</tbody></table></div>'


def render_report(bundle, max_examples=6):
    out = bundle['out']; folder = out/'evaluation'; assets = folder/'assets'; assets.mkdir(parents=True, exist_ok=True)
    test, diagnostic = (bundle['summaries'][s] for s in ('test', 'diagnostic')); methods = test['methods']
    files = plot_design(assets)+plot_scores(test, assets, 'fresh_k16')+plot_scores(diagnostic, assets, 'diagnostic_k16')
    files += plot_pool_curves(test, assets)+plot_failure(test, assets)+plot_datasets(test, assets)
    files += plot_training(bundle, assets)+plot_paired(test, assets)+plot_compute(test, assets)
    examples = select_examples(bundle, max_examples)
    for i, example in enumerate(examples, 1):
        path = assets/f'example_{i:02d}.png'; plot_example(example, path); example['path'] = path; files.append(path)
    document = []
    def heading(text, level=2):
        document.append(f'<h{level}>'+html.escape(text)+f'</h{level}>')
    def paragraph(text, note=False):
        document.append('<p'+(' class="note"' if note else '')+'>'+html.escape(text)+'</p>')
    def table(headers, values):
        document.append(table_html(headers, values))
    def figure(name, caption):
        path = assets/(name+'.png')
        document.append('<figure><img alt="'+html.escape(caption, quote=True)+'" src="data:image/png;base64,'+
                        base64.b64encode(path.read_bytes()).decode()+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>')
    heading('V11：内部候选评分器与专用视觉验证器', 1)
    paragraph(f"预先指定的内部主模型 {PRIMARY} 在新独立192题命中 {hit_text(methods[PRIMARY])}，同方案第二种子为 {hit_text(methods[ROBUSTNESS])}。同一K16候选池的 GUI-Actor 参考为 {hit_text(methods['gui_actor_K16'])}，旧point critic为 {hit_text(methods['old_critic_K16'])}。全部路线、种子与预算均保留，没有按测试分数更换主模型。")
    paragraph('主指标是静态正例截图中固定整数点是否落入目标几何。候选坐标由同一冻结旧解码器生成，新模型只选择点；这不是实际操作系统执行成功率、拒绝安全性、多步任务成功或零token推理。冻结Qwen仍先自由生成目标名称，再读取这些token自身的内部状态。', True)
    heading('方法设计与变量')
    paragraph('内部路线读取原始三层目标query [T,12288]、h15图像位置内容[P,4096]、premerge视觉[4P,1152]和全部目标token×96头注意力。每个点在全图与旧crop两个视图查询3×3粗格、5×5细格及全图语义；点不在crop内时屏蔽该视图。每个candidate-token-head先与目标和候选内容交互，再做token max/logmeanexp；96头通道保留到后续MLP，最后对实际候选集合与null联合比较。')
    paragraph('pooled控制在交互前沿token均值化query与每头字段，再广播；它保留96个头身份，是早期信息损失控制，不是精确复现旧v9。global控制仅屏蔽crop视图，全图内的局部邻域仍保留。三种内部配置参数完全一致。')
    paragraph('专用视觉路线只加载同一Qwen的视觉权重，额外将整张截图以最高1,048,576像素编码一次；多尺度ROI共享这些特征，现有完整目标名称状态条件化每个点，再做候选集合比较。没有使用GUI-Actor的权重、特征、评分或教师标签。它与内部路线像素预算和新增计算不同，因此不是同计算量结构消融。')
    figure('design', '两路线共享真实候选、监督与最终点计分；GUI-Actor是独立外部参考。')
    opt = bundle['protocol']['optimization']
    paragraph(f"训练为 {opt['epochs']} epoch，lr={opt['learning_rate']}、weight decay={opt['weight_decay']}、累积{opt['accumulation']}条、clip={opt['gradient_clip']}；每次从真实4/8/16/32前缀池随机选预算并打乱候选次序。多正例listwise包含显式none，并加适量类平衡BCE；GT只给实际已提议点标注，没有GT坐标注入。873条训练query保留同图多目标，未宣称额外加入跨query对比损失。")
    table(['配置','种子','参数','验证选中epoch','选模规则'], [[name, h['seed'], h['parameters'], h['selected']['epoch'], '验证K16命中优先，平局较低loss'] for name, h in bundle['histories'].items()])
    heading('数据、冻结与读取边界')
    counts = bundle['selection']['counts']; audit = bundle['selection_audit']
    table(['部分','query数','作用'], [['训练', counts['train'], '原873条真实train，实际生成候选'], ['验证', counts['val'], '旧172条，仅它选择epoch'],
        ['独立测试', counts['test'], '三公开来源各64，冻结后才提取'], ['旧v9诊断', 192, '重复使用过的图像，只作次要探索']])
    paragraph(f"原训练/验证保持内容一致；新测试用RGB身份与dHash≤4排除历史及跨split近重复。数据审计解码核验 {audit['all_cohort_unique_files_rgb_verified']} 个唯一图像文件，历史无法解析数为 {audit['historical_unresolved']}。哈希排除不能保证发现所有语义相似图像。")
    table(['跨split','左/右唯一图','RGB重复','dHash≤4对数'], [[name, f"{v['left_unique_images']} / {v['right_unique_images']}", v['exact_pixel_overlaps'], v['dhash_le_4_pairs']] for name, v in audit['cross_split_overlap'].items()])
    heading('主表：独立测试，同池K16与同图系统参考')
    body = []
    for name in primary_table_names():
        m = methods[name]
        body.append([name, hit_text(m), interval(m['ci95'])+'%', f"{m['points']}/192", m['pool_available'] if m['pool_available'] is not None else '不受本候选池限制', rate(m['conditional_selection_accuracy'])])
    table(['方法','实际点命中','95%区间','输出点','候选oracle题数','有正确候选时选对率'], body)
    paragraph('共同K16 pool方法的命中最多等于该池oracle；原生Qwen与UI-TARS自由生成坐标，不受这个oracle约束。global_top1是同一冻结全图解码器首选点，不是新增评分器。区间是按数据集分层、按截图聚类的bootstrap，不包含完整训练种子不确定性。')
    figure('fresh_k16', '固定协议顺序列出全部两种子；竖线只约束使用K16池的方法。')
    heading('候选覆盖与选择能力分别看')
    table(['预算','实际总点数','平均点数','空池题数','池中有正确点','oracle比例'], [[k, test['pools'][str(k)]['total_points'], number(test['pools'][str(k)]['total_points']/192, 2), test['pools'][str(k)]['empty'], f"{test['pools'][str(k)]['oracle']}/192", rate(test['pools'][str(k)]['oracle']/192)] for k in BUDGETS])
    paragraph('预算采用globalTop(K/2)+独立旧cropTop(K/2)，先限额后按整数坐标去重，不补位；所以实际点数可小于K。每个K都用自己的真实小集合重新前向，绝不将K32集合输出分数切片伪作小预算结果。K16预先固定为主，K4/8/32只检查预算敏感性。')
    figure('pool_and_selection', '覆盖扩大与排序改进是不同问题；右图分母只包含候选池有正确点的题。')
    table(['方法']+[f'K{k}命中 / 有正确点时选对率' for k in BUDGETS], [[short(f'{v}_{s}_K16')]+[f"{test['methods'][f'{v}_{s}_K{k}']['hits']}/192 / {rate(test['methods'][f'{v}_{s}_K{k}']['conditional_selection_accuracy'])}" for k in BUDGETS] for v in VARIANTS for s in SEEDS])
    heading('错误分环节与来源切片')
    names = head_names()+['old_critic_K16', 'gui_actor_K16']
    table(['方法','命中','空池','非空但无正确点','有正确点却排错'], [[name]+[methods[name][k] for k in STAGE_KEYS] for name in names])
    figure('failure_stages', '每行四部分严格加总192；没有因名称缺失或空池而缩小分母。')
    datasets = sorted(methods[PRIMARY]['datasets'])
    table(['方法']+datasets, [[name]+[f"{methods[name]['datasets'][d]['hits']}/{methods[name]['datasets'][d]['n']}" for d in datasets] for name in primary_table_names()])
    figure('datasets', '来源切片帮助检查总体差异是否只集中在一个数据集；全部两种子数值见表。')
    heading('训练/验证差距与学习曲线')
    body = []
    for name, h in bundle['histories'].items():
        a, b = h['selected_training_evaluation'], h['selected_validation_replay']
        body.append([name, h['selected']['epoch'], f"{a['hits']}/{a['n']}", f"{b['hits']}/{b['n']}",
                     number(100*(a['hits']/a['n']-b['hits']/b['n']), 2)+'pp', number(a['loss'], 3)+' / '+number(b['loss'], 3),
                     f"{a['oracle']}/{a['n']} / {b['oracle']}/{b['n']}"])
    table(['轨迹','epoch','训练K16命中','验证K16命中','train-val差','同K16 loss train/val','pool oracle train/val'], body)
    paragraph('训练重放按全部873条记录和K16计算，与随机预算训练过程中的即时train_loss分母不同；训练与验证的候选oracle也不同。差距提示泛化问题，不能单凭它断言某一分支导致过拟合。所有epoch只看旧验证集，测试没有学习曲线或选模作用。')
    figure('training_validation', '左：各自验证选中epoch的训练/验证；右：完整验证轨迹。')
    heading('固定主方案的配对差异')
    comparisons = key_comparisons(test)
    table(['主方案减去参考','胜 /负 /平','差值pp','配对95% CI (pp)'], [[c['reference'], f"{c['wins']} / {c['losses']} / {c['ties']}", number(c['delta']*100, 2), interval(c['ci95'])] for c in comparisons])
    figure('paired_intervals', '使用评测脚本已计算的配对区间，不在报告中重新bootstrap；Δ=主模型−参考。')
    document.append('<details><summary>查看全部61个预定配对比较</summary>'+table_html(['方法','参考','胜/负/平','Δpp','95%CI pp'],
        [[c['method'], c['reference'], f"{c['wins']}/{c['losses']}/{c['ties']}", number(c['delta']*100, 2), interval(c['ci95'])] for c in test['paired_comparisons']])+'</details>')
    paragraph('这些比较未作多重检验校正。两种子用于展示稳健性，不足以估计完整训练随机性；不能按这些测试差值切换主模型或再选预算。')
    heading('null输出：单列的辅助分析')
    table(['方法','强制选择命中','null策略命中','学习到的拒绝数','null策略输出点数'], [[name, methods[name]['hits'],
        methods[name]['optional_null_policy']['hits'], methods[name]['optional_null_policy']['learned_rejections'], methods[name]['optional_null_policy']['points']] for name in head_names()])
    paragraph('主指标始终选择非空池中的最高分点；null策略只作辅助，不改主表。null监督的是本候选池是否存在命中点，类平衡BCE和listwise分数不是校准的点击许可概率；本正例集也不能测禁用、缺失或禁止点击的误放行率。')
    heading('计算与时延拆开报告')
    t = test['timing']
    table(['CPU缓存头 /K16','样本数','总秒数','median ms','p95 ms'], [[name, t['cached_heads'][name]['n'], number(t['cached_heads'][name]['total_seconds'], 3), number(t['cached_heads'][name]['median_seconds']*1000, 2), number(t['cached_heads'][name]['p95_seconds']*1000, 2)] for name in head_names()])
    stage_rows = [('冻结Qwen全图目标状态', t['frozen_qwen_extraction']['global_view']), ('冻结Qwen旧crop目标状态', t['frozen_qwen_extraction']['local_view']),
                  ('额外高分辨率视觉编码', t['extra_visual_encoding']), ('原生文本坐标', t['native']), ('GUI-Actor GPU batch', t['gui_reference']['measured_batches'])]
    table(['分项记录阶段','记录数','总秒数','median s','p95 s'], [[label, value['n'], number(value['total_seconds'], 3), number(value['median_seconds'], 3), number(value['p95_seconds'], 3)] for label, value in stage_rows])
    paragraph('缓存头正式前向均在CPU，计入输入tensor转换与小模型前向；它排除了此前Qwen提取与专用视觉编码。GUI参考按GPU batch记录，不能把batch中位数当每图或每点时延。各阶段来自不同运行、缓存边界和设备，不能相加伪称实测端到端时间，也不能直接据此声称整体提速。图像编码共享不自动等于更快，新增分辨率的成本单列。')
    paragraph('运行包含并行调度，部分参考批次曾与Qwen共享GPU，计时受到资源竞争影响。该参考前缀辅助任务因显存压力停止，19条已完整写出的结果保留，其余由正式参考阶段补齐；评分函数、提示、批次与候选没有改变。调度详情记录于scheduling_notes.json，这些组件时间不是资源隔离的性能测量。')
    figure('compute_stages', 'CPU头、GPU提取阶段和GUI batch采用各自单位；没有构造总端到端柱状图。')
    heading('旧v9 192题：仅作次要探索')
    paragraph('这批截图已用于v9/v10分析，不能替代新独立测试，也不参与本轮训练或epoch选择。GUI分数复用旧K32公共缓存，没有新增GUI测时；显示分数只用于解释迁移与候选选择行为。', True)
    dm = diagnostic['methods']
    table(['方法','旧诊断命中','旧诊断K16有正确点时选对率'], [[name, hit_text(dm[name]), rate(dm[name]['conditional_selection_accuracy'])] for name in primary_table_names()])
    figure('diagnostic_k16', '重复图像上的结果不作为确认性收益证据；新测试仍是主表。')
    if examples:
        heading('事后可视化：改善、失败和候选缺失')
        paragraph('以下GT只用于正式评测完成后的报告展示。按预定现象分类，在每类中取sample_id排序首例且不重复，最多九张；不是随机样本或总体证据。绿线为完整目标几何，淡圈为实际K16候选，蓝点为主模型，橙点为对应参考。右侧只裁原始像素显示，没有重新输入模型。')
        for example in examples:
            heading(example['label'], 3)
            c, d = example['case'], example['detail']
            paragraph(f"{c['dataset']} · {c['sample_id']} · {c['instruction']}；模型生成名称：{d['target_name']}；参考：{example['reference']}。")
            figure(example['path'].stem, '两方法实际选择的点；远离目标的错误点请在左侧全图观察。')
    heading('审计、复现与结论边界')
    a = bundle['audit']
    table(['项目','审计证据'], [['冻结文件检查', a['frozen_files_checked']], ['训练文件检查', a['training_files_checked']], ['外部参考检查', a['external_files_checked']],
        ['新测试完整点计分', f"{a['splits']['test']['predictions_checked']} / 192×38"], ['旧诊断完整点计分', f"{a['splits']['diagnostic']['predictions_checked']} / 192×38"],
        ['CPU确定性回放', '两集合均完成；整数点精确一致，分数按审计容差'], ['协议SHA256', a['protocol_sha256']], ['冻结索引SHA256', a['frozen_sha256']]])
    for split in ('test', 'diagnostic'):
        replay = a['splits'][split]['replay']
        paragraph(f"{split} 的CPU回放范围：{len(replay.get('case_ids', []))}条按身份哈希预定的非空样本，head={replay.get('heads', '未提供')}，预算={replay.get('budgets', '未提供')}，head前向={replay.get('head_forwards', '未提供')}、旧解码器前向={replay.get('raw_decoder_forwards', '未提供')}。这是小样本补充回放，不是把全部八模型的全部前向重新执行一次。")
    for text in [
        '新测试仅192题、每来源64题；训练仍只有873 query/568图，参数增加仍可能过拟合。旧验证已多轮使用，新测试若据其继续设计模型就需要另一批新测试。',
        '模型生成目标名可能错误或缺失；候选池也可能没有任何正确点，评分器无法凭排序修复缺失坐标。候选oracle只约束当前固定提议策略，不是VLM通用上限。',
        '完整内部结构避免了机械地先丢弃token/head，但不能保证学习到了有效交互。参数匹配控制仅隔离所声明的信息路径；专用视觉路线还有额外分辨率和编码计算。',
        '两种子、截图bootstrap与配对比较不能消除来源偏移、近重复漏检或验证集反复试验的影响。没有按test更改模型、预算或阈值。',
        '本轮只评静态正例点选择；没有重新证明agent动作意图门控，也没有测真实GUI执行、多步计划或负例安全性。',
    ]:
        paragraph(text)
    paragraph('复现：python scripts/evaluate_candidate_scorer.py；python scripts/audit_candidate_scorer.py；python scripts/report_candidate_scorer.py。报告只读完成后的封存与评测文件，不训练、不调用GPU、不重选模型。PNG/PDF保存在assets；HTML内嵌全部PNG，可独立打开。')
    document.append('<p>补充材料：<a href="compression/report.html">测试前固定的K16→Top4候选压缩诊断</a>；<a href="test/exploratory_head_error_analysis.md">新192题评分头错例分析</a>；<a href="diagnostic/exploratory_error_analysis.md">旧192题探索性分析</a>。压缩诊断须在本页完整主审计后独立运行，属于训练期间追加的次要项目；错例分析不用于重新选模。</p>')
    style = "body{font:16px/1.75 system-ui,'Microsoft YaHei',sans-serif;max-width:1380px;margin:32px auto;padding:0 22px;color:#253544;background:#f7f9fa}h1,h2{line-height:1.35}h2{margin-top:38px}.note{background:#fff1d4;padding:14px;border-left:4px solid #ca913d}.table{overflow-x:auto}table{border-collapse:collapse;width:100%;background:white;font-size:14px}td,th{padding:8px 10px;border-bottom:1px solid #dce3e8;text-align:left;vertical-align:top}th{background:#e9f0f5}figure{margin:24px 0}img{max-width:100%;background:white}figcaption{font-size:14px;color:#5a6974}p{overflow-wrap:anywhere}details{margin:20px 0}summary{cursor:pointer;font-weight:600}"
    report = folder/'report.html'
    report.write_text('<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>内部评分器与专用视觉验证器 V11</title><style>'+style+'</style></head><body>'+''.join(document)+'</body></html>', encoding='utf-8')
    integrity = dict(complete=True, report=str(report.resolve()), report_sha256=sha(report), source_sha256=sha(Path(__file__)),
        final_audit_sha256=sha(out/'final_audit.json'), primary=PRIMARY, robustness=ROBUSTNESS,
        test_model_selection=False, fresh_primary=True, diagnostic_secondary=True, examples=len(examples),
        evaluation_sha256=bundle['audit']['evaluation_sha256'],
        data_selection_audit_sha256=sha(out/'data_selection_audit.json'),
        assets_sha256={str(p.resolve()): sha(p) for p in files})
    (folder/'report_integrity.json').write_text(json.dumps(integrity, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    return integrity


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('runs/candidate_scorer_v11'))
    parser.add_argument('--max-examples', type=int, default=6)
    args = parser.parse_args(argv)
    require(0 <= args.max_examples <= 9, '--max-examples must be0..9')
    result = render_report(load_bundle(args.output), args.max_examples)
    print('REPORT '+result['report'], flush=True)


if __name__ == '__main__':
    main()
