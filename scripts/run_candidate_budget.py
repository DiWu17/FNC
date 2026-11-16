"""V10 paired candidate-budget diagnostic; all K share one fresh point-score cache."""
import argparse
import json
from pathlib import Path
import shutil
import time

import torch
from PIL import Image

from clickagent.action_benchmark import sha256, write_json
from clickagent.candidate_budget import BUDGETS, build_pools, coordinate_key
from point_aligned_data import read
from run_gui_verifier import VerifierRuntime, MODEL
from run_target_token_experiment import key, read_jsonl

BASE = Path('runs/context_verifier_v9')
OUT = Path('runs/candidate_budget_v10')
SOURCES = ['src/clickagent/candidate_budget.py', 'scripts/run_candidate_budget.py',
           'scripts/evaluate_candidate_budget.py']


def cases():
    return [c for c in read_jsonl(OUT / 'manifest.jsonl') if c['experiment_split'] == 'test']


def seal_identity():
    old = read(BASE / 'inference_identity.json')
    files = dict(old['sources'])
    files.update({path: sha256(Path(path)) for path in SOURCES})
    for path in old['sources']:
        assert sha256(Path(path)) == old['sources'][path], path
    for path in ['protocol.json', 'manifest.jsonl', 'sealed_pools.json']:
        source = OUT / path
        files[str(source)] = sha256(source)
    for path in [BASE / 'inference_identity.json', BASE / 'evaluation/predictions.jsonl',
                 BASE / 'evaluation/summary.json', BASE / 'evaluation/audit.json',
                 MODEL / 'clickagent_download_verified.json']:
        files[str(path)] = sha256(path)
    for c in cases():
        sid = key(c['sample_id'])
        for folder in ['pipeline', 'gui_verifier']:
            path = BASE / folder / f'{sid}.json'
            files[str(path)] = sha256(path)
    return dict(files=files, model_verified_manifest=read(MODEL / 'clickagent_download_verified.json'))


def assert_seal():
    identity = read(OUT / 'identity.json')
    for path, digest in identity['files'].items():
        assert sha256(Path(path)) == digest, path
    return identity


def prepare():
    OUT.mkdir(exist_ok=True)
    if (OUT / 'identity.json').exists():
        assert_seal()
        return
    source_cases = [c for c in read_jsonl(BASE / 'manifest.jsonl') if c['experiment_split'] == 'test']
    assert len(source_cases) == len({c['sample_id'] for c in source_cases}) == 192
    assert read(BASE / 'evaluation/audit.json')['passed']
    (OUT / 'manifest.jsonl').write_text(''.join(json.dumps(c, ensure_ascii=False) + '\n' for c in source_cases), encoding='utf-8')
    all_pools = []
    for c in source_cases:
        pipeline = read(BASE / 'pipeline' / f'{key(c["sample_id"])}.json')
        pools = build_pools(pipeline)
        assert [v['point'] for v in pools['4']] == [v['point'] for v in pipeline['pools']['fixed_pool']]
        assert all(0 <= item['point'][0] < c['image_size'][0] and 0 <= item['point'][1] < c['image_size'][1]
                   for pool in pools.values() for item in pool)
        all_pools.append(dict(sample_id=c['sample_id'], pools=pools, ground_truth_in_proposals=False))
    write_json(OUT / 'sealed_pools.json', all_pools)
    protocol = dict(version=10, kind='exploratory reused-v9-192 candidate-budget diagnostic',
        independent_new_test=False, source=str(BASE), n=192, budgets=list(BUDGETS),
        pool_rule='globalTop(K/2) + independentOldCropTop(K/2), exact integer dedup after quota, no refill',
        unchanged='frozen v9 decoded coordinates, source ranks, images, instructions, crops, model, prompt and image budget',
        verifier='official GUI-Actor-Verifier-2B, crop mode, max_pixels262144, FP16, SDPA',
        score='True-vs-False log-odds, first tied candidate in each pool wins; no acceptance threshold',
        common_cache='re-score each distinct K32 point once in fixed chunks of6; every K uses this SAME fresh score',
        batch_size=6, seed=20261102, primary_analysis='all four K, no test-selected winning K',
        legacy_control='K4 coordinates must equal v9; archive exact old73 replay and fresh-score drift separately',
        compute='common-cache actual inference time; per-K point/pixel/call budget, not independent end-to-end latency',
        metric='point inside any valid GT region, inclusive; absent outputs stay in192 denominator',
        labels='GT used only for post-inference evaluation; no new training or policy tuning',
        generalization='diagnostic curves on already analyzed screenshots; a selected next policy requires fresh validation/test')
    write_json(OUT / 'protocol.json', protocol)
    identity = seal_identity()
    write_json(OUT / 'identity.json', identity)
    for path in SOURCES:
        destination = OUT / 'source_snapshot' / path
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    write_json(OUT / 'frozen.json', dict(frozen_at=time.time(), scored_cases_at_freeze=0,
        identity_sha256=sha256(OUT / 'identity.json')))
    print(json.dumps(dict(stage='prepared', n=192, unique_K32_points=sum(len(r['pools']['32']) for r in all_pools))), flush=True)


def infer():
    assert_seal()
    torch.set_num_threads(4)
    torch.manual_seed(20261102)
    rows = cases()
    sealed = {r['sample_id']: r for r in read(OUT / 'sealed_pools.json')}
    folder = OUT / 'verifier'
    folder.mkdir(exist_ok=True)
    pending = [c for c in rows if not (folder / f'{key(c["sample_id"])}.json').exists()
               or not read(folder / f'{key(c["sample_id"])}.json').get('complete')]
    started = time.time()
    if pending:
        runtime = VerifierRuntime()
        write_json(OUT / 'runtime.json', dict(**runtime.metadata, seed=20261102, fixed_max_batch_size=6))
        for i, c in enumerate(pending, 1):
            # Whitelist model inputs. No GT geometry or hit labels cross this boundary.
            public = {k: c[k] for k in ['sample_id', 'image', 'image_size', 'image_sha256', 'instruction']}
            sid = key(public['sample_id']);path = folder / f'{sid}.json'
            pool = sealed[public['sample_id']]['pools']['32']
            points = [v['point'] for v in pool]
            saved = read(path) if path.exists() else dict(sample_id=public['sample_id'], records={}, batches=[], complete=False,
                ground_truth_in_model_inputs=False, original_image_sha256=public['image_sha256'])
            assert sha256(Path(public['image'])) == public['image_sha256']
            with Image.open(public['image']) as im:
                image = im.convert('RGB')
            assert list(image.size) == public['image_size']
            # Chunk positions are fixed by the sealed largest pool, including resume.
            for start in range(0, len(points), 6):
                chunk = points[start:start + 6]
                if all(coordinate_key(p) in saved['records'] for p in chunk):
                    continue
                assert not any(coordinate_key(p) in saved['records'] for p in chunk), 'partial atomic chunk'
                results = runtime.score(image, public['instruction'], chunk, 'crop')
                for record in results:
                    saved['records'][coordinate_key(record['point'])] = record
                saved['batches'].append(dict(start=start, points=chunk, seconds=results[0]['batch_elapsed_seconds']))
                write_json(path, saved)
            assert set(saved['records']) == {coordinate_key(p) for p in points}
            saved['complete'] = True
            write_json(path, saved)
            complete = sum(read(p).get('complete', False) for p in folder.glob('*.json'))
            write_json(OUT / 'progress.json', dict(phase='verifier', completed=complete, total=192,
                last_sample_id=public['sample_id'], current_case_points=len(points),
                this_invocation_seconds=time.time()-started))
            print(json.dumps(dict(stage='verifier', i=i, pending=len(pending), complete=complete, points=len(points))), flush=True)
    assert_seal()
    assert all(read(folder / f'{key(c["sample_id"])}.json')['complete'] for c in rows)
    write_json(OUT / 'completion.json', dict(complete=True, n=192, full_qwen_rerun=False,
        verifier_cases=192, actual_unique_points=sum(len(v['pools']['32']) for v in sealed.values())))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['prepare', 'infer', 'all'], default='all')
    args = parser.parse_args()
    if args.stage in ('prepare', 'all'):prepare()
    if args.stage in ('infer', 'all'):infer()


if __name__ == '__main__':main()
