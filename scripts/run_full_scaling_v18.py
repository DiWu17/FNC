"""Independent full-data run, reusing immutable V15 caches; no HTML output.

Only output/status/cache-location adapters wrap the original extraction and
training mathematics. Adopted 3k checkpoints are read-only references.
"""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import argparse
from collections import Counter
from datetime import datetime, timezone
import gc
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
import numpy as np
import torch
import run_quality_readout_v15_3k as prior
from clickagent.evidence_retention import RetentionModel

OUT = ROOT / 'runs/full_scaling_v18'
MAIN = ROOT / 'runs/quality_readout_v15_3k'
ORIGINAL = ROOT / 'runs/quality_readout_v15'
read, write, key, digest = prior.read, prior.write, prior.key, prior.digest


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def status(phase, **fields):
    row = dict(phase=phase, complete=phase == 'complete', pid=os.getpid(),
               updated_at_utc=datetime.now(timezone.utc).isoformat(), **fields)
    write(OUT / 'status.json', row)
    p = OUT / 'events.jsonl'; p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('a', encoding='utf-8') as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps(row, ensure_ascii=False), flush=True)


def verify_protocol():
    spec = read(OUT / 'protocol.json')
    for p, expected in spec['hashes'].items():
        assert sha(p) == expected, p
    return spec


def prepare():
    if (OUT / 'protocol.json').exists():
        return verify_protocol()
    for name, expected in read(MAIN / 'protocol.json')['source_identity'].items():
        assert sha(ROOT / name) == expected, name
    prep_path = ORIGINAL / 'analysis/data_suitability/qc1_candidate/preparation.json'
    prep = read(prep_path)
    manifest = prep['candidate_manifests']['train.jsonl']
    assert sha(manifest['path']) == manifest['sha256']
    cases = prior.base.rows(manifest['path'])
    old = read(MAIN / 'data/old_train_index.json')
    selected_3k = prior.base.rows(MAIN / 'data/new_train.jsonl')
    assert len(cases) == 12500 and len(old) == 873
    assert {c['sample_id'] for c in selected_3k} <= {c['sample_id'] for c in cases}
    old_val = prior.base.old_entries('val')
    new_val = read(MAIN / 'features/new_reserve/index.json')['cases']
    train_rgb = {c['pixel_sha256'] for c in cases} | {e['group'] for e in old}
    assert not train_rgb & {e['group'] for e in old_val + new_val}
    dev_cases = prior.base.rows(MAIN / 'data/new_reserve.jsonl')
    assert not {(c['dataset'], c['source_group']) for c in cases} & {(c['dataset'], c['source_group']) for c in dev_cases}
    cases = sorted(cases, key=lambda c: key(c['sample_id']))
    p = OUT / 'data/new_train.jsonl'; p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(''.join(json.dumps(c, ensure_ascii=False)+'\n' for c in cases), encoding='utf-8')
    write(OUT / 'data/old_train_index.json', old)
    write(OUT / 'data/development_indices.json', dict(old_val=old_val, new_development=new_val))
    inputs = [Path(__file__), prep_path, Path(manifest['path']), MAIN / 'models/frozen.json',
        MAIN / 'completion.json', MAIN / 'protocol.json', MAIN / 'data/old_train_index.json',
        MAIN / 'features/new_train/index.json', MAIN / 'features/new_reserve/index.json',
        OUT / 'data/new_train.jsonl', OUT / 'data/old_train_index.json', OUT / 'data/development_indices.json']
    for name in read(MAIN / 'protocol.json')['source_identity']:
        inputs.append(ROOT / name)
    n = len(old) + len(cases)
    spec = dict(registered_at_utc=datetime.now(timezone.utc).isoformat(),
        hashes={str(p): sha(p) for p in inputs}, training_queries=n, new_queries=len(cases),
        seeds=list(prior.SEEDS), steps=8*((n+7)//8), readout_parameters=22050,
        arms=dict(full='All13373 cleaned queries, same architecture/loss, fresh seeded initialization',
                  matched3k='Original2973 queries, same full-data update count and fresh seeded initialization',
                  original3k='Read-only original2976-update checkpoints'),
        frozen_upstream='Original Qwen3-VL-8B, proposer and extraction contract, unchanged',
        evaluation='Only already-used old172/new300 development; union Top1 primary, K16 secondary; all3 seeds. No new independent-test claim.',
        cache='Exact row identity plus base/label/state/mask/pipeline hashes; reuse original/3k files read-only. New raw/compact outputs only in this directory.',
        optimizer='Original AdamW2e-4, wd.02, batch8, gradient clip1, unchanged union listwise/null+.25 balanced BCE; train-only scaler; fixed final step.',
        no_mainline_replacement=True, html=False,
        limits=['Full source mixture differs from balanced3k; improvement cannot be attributed only to quantity.',
                'Existing90-query semantic review remains pending; no fully manually certified-label claim.',
                'Only the readout is trained; frozen proposal coverage cannot be changed by readout scaling.'])
    write(OUT / 'protocol.json', spec)
    status('registered', queries=n, steps=spec['steps'])
    return spec


class FullExtractor(prior.Extractor):
    def view(self, case, transform):
        ident = digest(dict(case=case, transform=transform, contract=self.contract))
        p = MAIN / 'raw' / f'{ident}.npz'
        if not p.exists():
            return super().view(case, transform)
        with np.load(p, allow_pickle=False) as z:
            meta = json.loads(str(z['metadata']))
            arrays = {k: z[k].copy() for k in ('target_attention', 'target_hidden', 'image_hidden', 'vision_hidden')}
        assert meta['input_identity'] == ident and meta['own_token_forward']
        assert meta['all_backbone_parameters_frozen'] and meta['ground_truth_in_model_input'] is False
        return arrays, meta, dict(feature_path=str(p), feature_sha256=sha(p), transform=transform)

    def case(self, public):
        p = MAIN / f'pipeline/{key(public["sample_id"])}.json'
        if not p.exists():
            return super().case(public)
        pipe = read(p)
        assert pipe['complete'] and pipe['input_identity'] == digest(public)
        for v in [dict(feature_path=pipe['global_feature_path'], feature_sha256=pipe['global_feature_sha256'])] + pipe['local_views']:
            assert sha(v['feature_path']) == v['feature_sha256']
        return pipe, p


def extract():
    spec = verify_protocol()
    index = OUT / 'features/new_train/index.json'
    if index.exists():
        assert read(index)['complete']
        return
    cases = prior.base.rows(OUT / 'data/new_train.jsonl')
    cached = {e['sample_id']: e for e in read(MAIN / 'features/new_train/index.json')['cases']}
    records = []; reused = computed = 0; new_seconds = 0.; started = time.perf_counter(); extractor = None
    status('extracting_new_train', completed=0, total=len(cases), reused=0, computed=0)
    for i, c in enumerate(cases, 1):
        dest = OUT / f'features/new_train/cases/{key(c["sample_id"])}.json'
        original = ORIGINAL / f'features/new_train/cases/{key(c["sample_id"])}.json'
        entry = read(dest) if dest.exists() else cached.get(c['sample_id']) or (read(original) if original.exists() else None)
        if entry is not None:
            prior.authenticate_entry(entry, c)
            if not dest.exists():
                write(dest, dict(entry, reused_by='full_scaling_v18'))
            reused += 1
        else:
            tick = time.perf_counter()
            if extractor is None:
                extractor = FullExtractor()
            pipe, pp = extractor.case({k: c[k] for k in prior.base.PUBLIC})
            b, labels, state, mask = prior.compact(c, pipe, pp)
            bp = OUT / f'features/new_train/arrays/{key(c["sample_id"])}.npz'
            prior.base.npz(bp, **b)
            paths = [bp.with_suffix(s) for s in ['.labels.npy', '.state.npy', '.mask.npy']]
            for p, a in zip(paths, [labels, state, mask]):
                temp = p.with_suffix(p.suffix+'.tmp')
                with temp.open('wb') as stream:
                    np.save(stream, a, allow_pickle=False)
                os.replace(temp, p)
            lp, sp, mp = paths
            entry = dict(sample_id=c['sample_id'], split='train', role='new_train', source=c['dataset'],
                group=c['pixel_sha256'], source_group=c['source_group'], case_identity=digest(c),
                n=len(b['points']), matched_count=len(b['baseline']), path=str(bp), sha256=sha(bp),
                labels_path=str(lp), labels_sha256=sha(lp), state_path=str(sp), state_sha256=sha(sp),
                mask_path=str(mp), mask_sha256=sha(mp), pipeline_path=str(pp), pipeline_sha256=sha(pp))
            write(dest, entry); computed += 1; new_seconds += time.perf_counter()-tick
        records.append(entry)
        if i % 5 == 0 or i == len(cases):
            status('extracting_new_train', completed=i, total=len(cases), reused=reused, computed=computed,
                elapsed_seconds=time.perf_counter()-started,
                mean_new_case_seconds=new_seconds/computed if computed else None)
    write(index, dict(complete=True, cases=records, n=len(records), protocol_sha256=sha(OUT / 'protocol.json')))
    del extractor; gc.collect(); torch.cuda.empty_cache()
    status('training_features_complete', new_queries=len(records))


def train():
    spec = verify_protocol()
    registry = {}
    for arm in ['full', 'matched3k']:
        entries = read(OUT / 'data/old_train_index.json') + read(OUT / 'features/new_train/index.json')['cases'] if arm == 'full' else read(MAIN / 'data/old_train_index.json') + read(MAIN / 'features/new_train/index.json')['cases']
        status('loading_train', arm=arm, queries=len(entries))
        records = prior.base.load_records(entries)
        for s, seed in enumerate(prior.SEEDS, 1):
            prior.train_one(arm, records, seed, spec['steps'], 'cuda:0')
            receipt = read(OUT / f'models/{arm}_s{s}.json')
            assert receipt['complete'] and sha(receipt['checkpoint_path']) == receipt['checkpoint_sha256']
            registry[f'{arm}_s{s}'] = dict(path=receipt['checkpoint_path'], sha256=receipt['checkpoint_sha256'], queries=len(entries), steps=spec['steps'])
        del records; gc.collect(); torch.cuda.empty_cache()
    for s in [1, 2, 3]:
        e = read(MAIN / f'models/expanded_s{s}.json')
        registry[f'original3k_s{s}'] = dict(path=e['checkpoint_path'], sha256=e['checkpoint_sha256'], queries=2973, steps=2976)
    write(OUT / 'models/frozen.json', dict(complete=True, models=registry, protocol_sha256=sha(OUT / 'protocol.json')))
    status('models_frozen', new_models=6, model_slots=9)


@torch.inference_mode()
def evaluate():
    frozen = read(OUT / 'models/frozen.json')
    for cohort, entries in read(OUT / 'data/development_indices.json').items():
        status('evaluating_'+cohort, models_completed=0, models_total=9)
        records = prior.base.load_records(entries)
        methods = {}; vectors = {}; secondary = {}
        for name, info in frozen['models'].items():
            assert sha(info['path']) == info['sha256']
            saved = torch.load(info['path'], map_location='cpu', weights_only=True)
            model = RetentionModel().to('cuda:0').eval(); model.load_state_dict(saved['state_dict'])
            union_hits = []; k16_hits = []
            for r in records:
                if r['n']:
                    inputs, _ = prior.tensors([r], saved['mean'].numpy(), saved['std'].numpy(), 'aligned', 'cuda:0')
                    scores = model(**inputs).float().cpu().numpy()
                    pick = int(np.argmax(scores)); pool = np.sort(r['baseline'])
                    kpick = int(pool[np.argsort(-scores[pool], kind='stable')[0]]) if len(pool) else None
                else:
                    pick = kpick = None
                union_hits.append(bool(pick is not None and r['labels'][pick]))
                k16_hits.append(bool(kpick is not None and r['labels'][kpick]))
            vectors[name] = union_hits; secondary[name] = k16_hits
            methods[name] = dict(union_hits=sum(union_hits), union_accuracy_percent=100*np.mean(union_hits),
                                 K16_hits=sum(k16_hits), K16_accuracy_percent=100*np.mean(k16_hits))
            status('evaluating_'+cohort, models_completed=len(methods), models_total=9, last_model=name)
            del model
        comparisons = {}
        for ref in ['original3k', 'matched3k']:
            comparisons['full_vs_'+ref] = prior.mean_seed_comparison(
                [vectors[f'full_s{s}'] for s in [1, 2, 3]], [vectors[f'{ref}_s{s}'] for s in [1, 2, 3]], entries)
        write(OUT / f'evaluation/{cohort}.json', dict(complete=True, n=len(records), methods=methods,
            comparisons=comparisons, hit_vectors=vectors, K16_hit_vectors=secondary,
            sample_ids=[r['sample_id'] for r in records], scope='Development only; no new independent confirmation'))
        del records; gc.collect()
    files = [OUT / 'protocol.json', OUT / 'features/new_train/index.json', OUT / 'models/frozen.json']
    lines = ['全量数据评分器：开发评估；原3000例模型保持不变。']
    for cohort in ['old_val', 'new_development']:
        p = OUT / f'evaluation/{cohort}.json'; files.append(p); r = read(p)
        for arm in ['original3k', 'matched3k', 'full']:
            value = np.mean([r['methods'][f'{arm}_s{s}']['union_accuracy_percent'] for s in [1, 2, 3]])
            lines.append(f'{cohort} / {arm}: 完整候选Top1三种子均值={value:.2f}%')
        lines.append(json.dumps(r['comparisons'], ensure_ascii=False))
    (OUT / 'results.txt').write_text('\n'.join(lines)+'\n', encoding='utf-8'); files.append(OUT / 'results.txt')
    verify_protocol()
    write(OUT / 'completion.json', dict(complete=True, completed_at_utc=datetime.now(timezone.utc).isoformat(),
        artifact_sha256={str(p): sha(p) for p in files}, mainline_unchanged=True, independent_confirmation=False))
    status('complete', training_queries=13373, new_models=6)


def main():
    p = argparse.ArgumentParser(); p.add_argument('--stage', choices=['prepare', 'extract', 'train', 'evaluate', 'all'], default='all')
    args = p.parse_args()
    prior.OUT = OUT; prior.status = status
    torch.set_num_threads(4)
    try:
        prepare()
        if args.stage in ['extract', 'all']: extract()
        if args.stage in ['train', 'all']: train()
        if args.stage in ['evaluate', 'all']: evaluate()
    except BaseException:
        status('failed', error=traceback.format_exc()); raise


if __name__ == '__main__':
    main()
