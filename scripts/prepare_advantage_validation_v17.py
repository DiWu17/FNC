"""Lock V17 questions and label-blind cohorts before new experiment outputs."""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from clickagent.advantage_validation import balanced_cases, nested_groups

OUT = ROOT / 'runs/advantage_validation_v17'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def main():
    path = OUT / 'plan.json'
    if path.exists():
        spec = read(path)
        for name, expected in spec['input_hashes'].items():
            assert sha(name) == expected, name
        print('Existing V17 plan and inputs verified')
        return
    assert not (OUT / 'data_efficiency/models').exists()
    main = ROOT / 'runs/quality_readout_v15_3k'
    v16 = ROOT / 'runs/contribution_v16'
    independent_path = v16 / 'data/independent_test.jsonl'
    cases = [json.loads(s) for s in independent_path.read_text(encoding='utf-8').splitlines()]
    assert len(cases) == len({c['pixel_sha256'] for c in cases}) == 300
    selected = balanced_cases(cases, 30, 'v17-speed-robustness')
    entries = read(main / 'data/old_train_index.json') + read(main / 'features/new_train/index.json')['cases']
    assert len(entries) == 2973
    data_path = OUT / 'data/cohorts.json'
    write(data_path, dict(full_test=cases, speed_robustness=selected,
                          warmup=read(main / 'features/new_reserve/index.json')['cases'][:2]))
    subsets = {}
    for fraction in [.1, .25, .5, 1.]:
        selected_entries = nested_groups(entries, fraction)
        tag = f'n{round(fraction * 100):03d}'
        p = OUT / f'data/{tag}_train.json'
        write(p, selected_entries)
        new_count = sum(e['role'] == 'new_train' for e in selected_entries)
        subsets[tag] = dict(fraction=fraction, queries=len(selected_entries),
                            rgb_images=len({e['group'] for e in selected_entries}),
                            by_source=dict(Counter(e['source'] for e in selected_entries)),
                            path=str(p), sha256=sha(p),
                            nominal8_steps=8 * ((len(selected_entries) + 7) // 8),
                            upstream873_plus_selected_new_queries=873 + new_count)
    files = [independent_path, data_path, main / 'data/old_train_index.json',
             main / 'features/new_train/index.json', main / 'models/frozen.json',
             v16 / 'models/frozen.json', v16 / 'independent/index.json',
             Path(__file__), ROOT / 'src/clickagent/advantage_validation.py']
    revisions = {name: subprocess.check_output(['git', '-C', str(ROOT / 'third_party' / name),
                                               'rev-parse', 'HEAD'], text=True).strip()
                 for name in ['TAG', 'GUI-AIMA', 'MVP']}
    spec = dict(registered_at_utc=datetime.now(timezone.utc).isoformat(),
        mainline='quality_readout_v15_3k', mainline_unchanged=True,
        purpose='Find measured strengths and weaknesses against external methods, without precommitting to positive results.',
        input_hashes={str(p): sha(p) for p in files}, official_repositories=revisions,
        scope='Exploratory extensions of the previously evaluated V16 screenshots. No new independent-test claim.',
        training_subsets=subsets,
        data_efficiency=dict(variants=['aligned', 'no_evidence'],
            seeds=[20261311, 20261312, 20261313], architecture_parameters=22050,
            fixed_updates=2976, second_budget='8 nominal query passes, batch8; same as parent full regime',
            evaluation=['old_val172', 'new_development300', 'reused_independent300'],
            full_size='Reuse authenticated V15_3k full and V16 no-evidence models. Never pick a seed or model from test.',
            target='Report curves, paired differences and the smallest observed fraction retaining90% of full accuracy; threshold crossing is descriptive.',
            limits='Only readout-data efficiency conditional on fixed873-query trained upstream. TAG uses zero grounding training labels; released AIMA training is not matched.'),
        speed=dict(cases=60, per_source=30, repetitions=2, warmup_examples=2,
            primary='Single-request wall time including image decoding, all VLM passes, proposal/feature extraction and point selection. No cached-feature speed claim.',
            summaries=['p50', 'p95', 'mean', 'paired bootstrap ratio', 'peak CUDA allocated/reserved', 'model load time separately', 'forward count', 'processed pixels'],
            exclusions='Model loading, one-time warmup and evidence/report disk writes excluded equally; preprocessing and CPU/GPU transfers included.',
            concurrency='One benchmark GPU job at a time; record GPU model, dtype, software and actual model placement.',
            our_seed='S1 fixed for timing; report all3 scorer predictions, without choosing bestseed or ensemble.'),
        robustness=dict(cases=60, conditions=['clean', 'jpeg70', 'border24'],
            target='Full end-to-end regeneration for every condition; map points back to original geometry only for scoring.',
            metrics=['absolute perturbed accuracy', 'clean-to-perturbed paired change', 'clean-correct retention', 'prediction displacement in original coordinates'],
            limits='Visual invariance only; source-stratified scores do not establish unseen-app or semantic paraphrase robustness.'),
        external_methods=['qwen_official_coordinate', 'ui_tars_1_5_7b', 'TAG_original_MiniCPM', 'GUI_AIMA_3B', 'Qwen3VL_MVP'],
        inference_budgets=dict(controlled='262144 pixels per Qwen/AIMA/UI view; TAG keeps native slicing after same public image cap, declared separately.',
            stronger_reference='Add Qwen/UI single-view1048576 and MVP paper-style resolution on fixed60; no low-resolution win generalized to publisher-default inference.',
            recipes='Lock each adapter and development-only parity checks before cohort inference. FP16 on P5000; no unsupportedBF16/FlashAttention speed claims.'),
        superiority_rules=['Accuracy: paired positive interval against named baseline within declared input/view budget; no single global superiority claim.',
            'Speed: end-to-end paired latency benefit and accuracy reported jointly; a fast but less accurate method is a tradeoff.',
            'Data: fixed-update and fixed-exposure curves both reported, including upstream supervised budget.',
            'Robustness: report perturbation loss plus absolute accuracy; failure on clean images cannot masquerade as robustness.',
            'Test-set labels never enter adapters, crop choice, perturbations, subset selection or model tuning.',
            'Unsupported adapter or resource failures are recorded explicitly and never converted into an apparent accuracy win.'])
    write(path, spec)
    write(OUT / 'status.json', dict(phase='preparing', complete=False, updated_at_utc=datetime.now(timezone.utc).isoformat()))
    print(json.dumps({'registered': True, 'subsets': subsets, 'speed_robustness': len(selected)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
