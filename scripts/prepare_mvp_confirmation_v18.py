"""Reserve fresh screenshots without consulting predictions; no model calls."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'scripts'))
import prepare_contribution_v16 as prep

OUT = ROOT/'runs/mvp_confirmation_v18'


def main():
    receipt = OUT/'data/selection.json'
    if receipt.exists():
        value = prep.read_json(receipt)
        for p, expected in value['input_sha256'].items(): assert prep.sha(p) == expected, p
        assert prep.sha(OUT/'data/heldout.jsonl') == value['manifest_sha256']
        print('Previously reserved fresh screenshots authenticated; no reselection.', flush=True)
        return
    prep.OUT = OUT
    prep.SEED = 20261105
    # The prior150-per-source availability check found only143 unused Venus
    # screenshots under the unchanged exclusion rule; no predictions existed.
    prep.QUOTAS = {'screenspot_pro': 140, 'venusbench_gd': 140}
    paths, ignored = prep.historical_manifests()
    infos, pixels, hashes, identities, historical_rows = prep.resolve_history(paths)
    print(f'History authenticated: {len(paths)} manifests; {len(pixels)} unique RGB images.', flush=True)
    sources = {s:ROOT/f'data/public_v1/manifests/{s}.jsonl' for s in prep.QUOTAS}
    selected, stats = prep.choose({s:prep.read_rows(p) for s,p in sources.items()}, infos, pixels, hashes, quotas=prep.QUOTAS)
    with ThreadPoolExecutor(max_workers=2) as pool:
        for i, (case, actual) in enumerate(zip(selected, pool.map(prep.image_info, [c['image'] for c in selected])), 1):
            for a,b in [('pixel_sha256','pixel_sha256'),('dhash256','dhash256'),('image_sha256','file_sha256'),('image_size','image_size')]:
                assert case[a] == actual[b], (case['sample_id'],a)
            if i % 50 == 0: print(f'Redecoded reserved screenshots: {i}/{len(selected)}', flush=True)
    manifest = OUT/'data/heldout.jsonl'; prep.write_rows(manifest, selected)
    inputs = sorted(set(Path(p) for p in identities) | set(sources.values()) | {Path(__file__), ROOT/'scripts/prepare_contribution_v16.py', ROOT/'scripts/prepare_candidate_scorer.py', ROOT/'scripts/prepare_semantic_head_data.py'})
    prep.write_json(receipt, dict(complete=True, reserved_at_utc=datetime.now(timezone.utc).isoformat(),
        seed=prep.SEED, quotas=prep.QUOTAS, selected_queries=len(selected), selected_unique_rgb=len({r['pixel_sha256'] for r in selected}),
        selection_key='sha256(20261105:independent_v16:sample_id), inherited selector namespace; new seed and historical exclusion pool',
        availability=stats, input_sha256={str(p):prep.sha(p) for p in inputs}, manifest_sha256=prep.sha(manifest),
        historical_rows=historical_rows, protected_unique_rgb=len(pixels), protected_unique_dhash=len(hashes),
        historical_manifests=[str(p) for p in paths], ignored_artifacts=ignored,
        exclusions='Exact RGB and dHash256 Hamming<=4 against all discovered experiment manifests, entire imported train/reserve pool and within the new subset. One query per screenshot.',
        model_outputs_used_for_selection=False, no_model_execution=True, images_redecoded=True,
        feature_extraction_started=False,
        availability_adjustment='Before any held-out prediction,150-per-source request found only143 strict-isolation Venus images. Final quota140 per source; exact/near exclusions unchanged. Initial failure log retained.',
        gate='Freeze all compared checkpoints, candidate policy and evaluation protocol before any held-out inference.',
        scope='Reserved fresh image-isolated subset, not a full official benchmark, unseen-application claim or guarantee of pretraining independence.'))
    print(f'Reserved {len(selected)} fresh screenshots; no features or predictions generated.', flush=True)


if __name__ == '__main__': main()
