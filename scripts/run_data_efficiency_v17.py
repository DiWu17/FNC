"""Readout-only learning curves with fixed-update and fixed-exposure controls."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
import numpy as np
import torch
import run_quality_readout_v15_3k as prior
import run_contribution_v16 as contribution
from clickagent.evidence_retention import RetentionModel

MASTER = ROOT / 'runs/advantage_validation_v17'
OUT = MASTER / 'data_efficiency'
MAIN = ROOT / 'runs/quality_readout_v15_3k'
V16 = ROOT / 'runs/contribution_v16'
read, write, sha = prior.read, prior.write, prior.sha


def status(phase, **fields):
    row = dict(phase=phase, complete=phase == 'complete', pid=__import__('os').getpid(),
               updated_at_utc=datetime.now(timezone.utc).isoformat(), **fields)
    write(OUT / 'status.json', row)
    with (OUT / 'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps(row, ensure_ascii=False), flush=True)


def register():
    OUT.mkdir(parents=True, exist_ok=True)
    p = OUT / 'protocol.json'
    if p.exists():
        spec = read(p)
        for path, expected in spec['hashes'].items():
            assert sha(path) == expected, path
        return spec
    assert not (OUT / 'models').exists()
    plan = read(MASTER / 'plan.json')
    hashes = {str(MASTER / 'plan.json'): sha(MASTER / 'plan.json'), str(Path(__file__)): sha(__file__),
              **{str(ROOT / name): expected for name, expected in read(V16 / 'protocol.json')['source_identity'].items()}}
    # V16's source identity includes the adapters and all reused training/evaluation mathematics.
    for path, expected in hashes.items():
        assert sha(path) == expected, path
    for v in plan['training_subsets'].values():
        hashes[v['path']] = v['sha256']
    spec = dict(registered_at_utc=datetime.now(timezone.utc).isoformat(), hashes=hashes,
                plan_sha256=sha(MASTER / 'plan.json'), question=plan['data_efficiency'],
                subsets=plan['training_subsets'], new_fits=36, original_files_read_only=True,
                performance_cost_note='Training elapsed logs are operational only; no cross-model training-time claim from these curves.')
    write(p, spec)
    status('registered', new_fits=36)
    return spec


@contextmanager
def adapters(variant):
    previous = prior.OUT, prior.status, prior.tensors
    prior.OUT, prior.status = OUT, status
    prior.tensors = lambda records, mean, std, ignored, device: contribution.build_inputs(records, mean, std, variant, device)
    try:
        yield
    finally:
        prior.OUT, prior.status, prior.tensors = previous


def train():
    spec = register()
    full_entries = read(MASTER / 'data/n100_train.json')
    status('loading_train', queries=len(full_entries))
    records = prior.base.load_records(full_entries)
    lookup = {r['sample_id']: r for r in records}
    registry = {}
    for budget in ['fixed_updates', 'nominal8']:
        for subset, info in spec['subsets'].items():
            selected = read(info['path'])
            fit = [lookup[e['sample_id']] for e in selected]
            steps = 2976 if budget == 'fixed_updates' else info['nominal8_steps']
            for variant in ['aligned', 'no_evidence']:
                arm = f'{budget}_{subset}_{variant}'
                for i, seed in enumerate(prior.SEEDS, 1):
                    tag = f'{arm}_s{i}'
                    if subset == 'n100':
                        parent = MAIN / f'models/expanded_s{i}.json' if variant == 'aligned' else V16 / f'models/no_evidence_s{i}.json'
                        receipt = read(parent)
                        assert sha(receipt['checkpoint_path']) == receipt['checkpoint_sha256']
                    else:
                        with adapters(variant):
                            prior.train_one(arm, fit, seed, steps, 'cuda:0')
                        receipt = read(OUT / f'models/{tag}.json')
                    registry[tag] = dict(path=receipt['checkpoint_path'], sha256=receipt['checkpoint_sha256'],
                                         budget=budget, subset=subset, variant=variant, seed=seed,
                                         queries=len(fit), steps=steps,
                                         reused_full_size=subset == 'n100')
    write(OUT / 'models/frozen.json', dict(complete=True, models=registry,
          protocol_sha256=sha(OUT / 'protocol.json'), trained_fits=36, unique_checkpoints=len({v['sha256'] for v in registry.values()})))
    status('models_frozen', model_slots=len(registry))
    del records, lookup, fit
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def evaluate():
    register()
    frozen = read(OUT / 'models/frozen.json')
    assert frozen['complete'] and frozen['protocol_sha256'] == sha(OUT / 'protocol.json')
    cohorts = {'old_val': prior.base.old_entries('val'),
               'new_development': read(MAIN / 'features/new_reserve/index.json')['cases'],
               'reused_independent': read(V16 / 'independent/index.json')['cases']}
    all_results = {}
    for cohort, entries in cohorts.items():
        records = prior.base.load_records(entries)
        cache, vectors, methods = {}, {}, {}
        status('evaluating_' + cohort, models_completed=0, models_total=len(frozen['models']))
        for name, info in frozen['models'].items():
            assert sha(info['path']) == info['sha256']
            if info['sha256'] in cache:
                hits = cache[info['sha256']]
            else:
                saved = torch.load(info['path'], map_location='cpu', weights_only=True)
                model = RetentionModel().to('cuda:0').eval()
                model.load_state_dict(saved['state_dict'])
                hits = []
                for r in records:
                    if r['n']:
                        inputs, _ = contribution.build_inputs([r], saved['mean'].numpy(), saved['std'].numpy(), info['variant'], 'cuda:0')
                        scores = model(**inputs).float().cpu().numpy()
                        pool = np.sort(r['baseline'])
                        pick = int(pool[np.argsort(-scores[pool], kind='stable')[0]]) if len(pool) else None
                    else:
                        pick = None
                    hits.append(bool(pick is not None and r['labels'][pick]))
                cache[info['sha256']] = hits
                del model
            vectors[name] = hits
            methods[name] = dict(hits=sum(hits), accuracy=sum(hits) / len(records), **info)
            if len(methods) % 6 == 0:
                status('evaluating_' + cohort, models_completed=len(methods), models_total=len(frozen['models']))
        comparisons = {}
        for budget in ['fixed_updates', 'nominal8']:
            for subset in ['n010', 'n025', 'n050', 'n100']:
                full = [vectors[f'{budget}_{subset}_aligned_s{i}'] for i in [1, 2, 3]]
                no = [vectors[f'{budget}_{subset}_no_evidence_s{i}'] for i in [1, 2, 3]]
                reference = [vectors[f'{budget}_n100_aligned_s{i}'] for i in [1, 2, 3]]
                comparisons[f'{budget}_{subset}_aligned_vs_no_evidence'] = prior.mean_seed_comparison(full, no, entries)
                comparisons[f'{budget}_{subset}_aligned_vs_full_data'] = prior.mean_seed_comparison(full, reference, entries)
        result = dict(complete=True, n=len(records), methods=methods, comparisons=comparisons,
                      sample_ids=[r['sample_id'] for r in records], hit_vectors=vectors,
                      scope='Readout-only data curves on reused exploratory/development cohorts')
        write(OUT / f'evaluation/{cohort}.json', result)
        all_results[cohort] = {k: result[k] for k in ['n', 'methods', 'comparisons', 'scope']}
        del records
        gc.collect()
    write(OUT / 'results.json', dict(complete=True, cohorts=all_results,
          limitation='Frozen proposer already trained on873 labels; curves do not show entire-system training from only the plotted readout queries.'))
    files = [OUT / 'protocol.json', OUT / 'models/frozen.json', OUT / 'results.json']
    write(OUT / 'completion.json', dict(complete=True, artifact_sha256={str(p): sha(p) for p in files}, mainline_unchanged=True))
    status('complete', new_fits=36)


def main():
    torch.set_num_threads(4)
    try:
        train()
        evaluate()
    except Exception:
        status('failed', error=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
