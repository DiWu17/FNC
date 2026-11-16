"""Reuse v7 train/val, exclude all historical images, seal a fresh v8 test."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path

from prepare_semantic_head_data import (HammingIndex, discover_history, image_path, image_info,
    read_rows, read_json, sha, key, valid_public_geometry, write_json, write_rows)


OUT = Path('runs/point_aligned_v8')
BASE = Path('runs/semantic_head_v7')

def path_key(value,manifest=None):
    path=Path(value)
    if not path.is_absolute() and manifest is not None:path=Path(manifest).parent/path
    return os.path.normcase(os.path.abspath(path))


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'data_selection.json').exists():
        assert read_json(OUT/'data_selection.json')['manifest_sha256'] == sha(OUT/'manifest.jsonl')
        print('DATA ALREADY SEALED'); return
    base = read_rows(BASE/'manifest.jsonl')
    retained = [r for r in base if r['experiment_split'] in ('train','val')]
    history, _ = discover_history(Path.cwd(), OUT)
    rows = [(p, r) for p in history for r in read_rows(p)]
    print(f'HISTORY {len(history)} manifests {len(rows)} rows',flush=True)
    infos = {path_key(r['image']): r for r in read_json('data/conditioned_v3/public_image_index.json')['images']}
    for r in read_rows('data/conditioned_v3/training_pool_b8ed555f5e93.jsonl'):
        infos.setdefault(path_key(r['image']), r)
    for r in read_rows(BASE/'data_selection_image_hash_cache.jsonl'):
        infos.setdefault(path_key(r['image']), r)
    pixels = set(); hashes = set(); unresolved=[]
    control_sources={}
    for path, row in rows:
        if not row.get('image') and not row.get('image_path'): continue
        info = row if row.get('pixel_sha256') and row.get('dhash256') else infos.get(path_key(row.get('image',row.get('image_path')),path))
        if info is None:
            control = path.parent/'control.json'
            if control.exists():
                if control not in control_sources:
                    cr=read_json(control); original=Path(cr.get('source_manifest',''))
                    control_sources[control]=(original,{r['sample_id']:r for r in read_rows(original)}) if original.is_file() and sha(original)==cr.get('source_manifest_sha256') else None
                cached=control_sources[control]
                if cached is not None:
                    original,source_rows=cached;source=source_rows.get(row.get('sample_id'))
                    if source:
                        info=source if source.get('pixel_sha256') and source.get('dhash256') else infos.get(path_key(source.get('image',source.get('image_path')),original))
        if info is None:
            unresolved.append(dict(manifest=str(path),sample_id=row.get('sample_id')));continue
        pixels.add(info['pixel_sha256']);hashes.add(int(info['dhash256'],16))
    assert not unresolved, unresolved[:10]
    historical_pixels=len(pixels);historical_dhashes=len(hashes)
    print(f'RESOLVED {historical_pixels} images',flush=True)
    tree=HammingIndex(hashes); fresh=[]; sources=[]; counts={}; seed=20261111
    for slug in ('screenspot_v2','screenspot_pro','venusbench_gd'):
        path=Path('data/public_v1/manifests')/(slug+'.jsonl');sources.append(path)
        eligible=[r for r in read_rows(path) if r.get('split','test')=='test' and r.get('expected_actionable') and valid_public_geometry(r)]
        eligible=[dict(r,pixel_sha256=infos[path_key(r['image'])]['pixel_sha256'],
            dhash256=infos[path_key(r['image'])]['dhash256'],image_sha256=infos[path_key(r['image'])]['file_sha256']) for r in eligible]
        eligible.sort(key=lambda r:key(seed,'fresh_v8',r['sample_id']))
        selected=[]; rejected=Counter()
        for r in eligible:
            if r['pixel_sha256'] in pixels: rejected['exact']+=1;continue
            if tree.near(int(r['dhash256'],16),4):rejected['near4']+=1;continue
            selected.append(dict(r,experiment_split='test',split='test',training_extension=False))
            pixels.add(r['pixel_sha256']);tree.add(int(r['dhash256'],16))
            if len(selected)==64:break
        assert len(selected)==64,(slug,len(selected))
        fresh+=selected;counts[slug]=dict(selected=64,rejected=dict(rejected))
        print('SELECTED '+slug+' 64',flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        actual=list(pool.map(image_info,[r['image'] for r in fresh]))
    for c,info in zip(fresh,actual):
        assert c['image_sha256']==info['file_sha256'] and c['pixel_sha256']==info['pixel_sha256']
        assert c['dhash256']==info['dhash256'] and c['image_size']==info['image_size']
    manifest=retained+fresh
    assert len({r['sample_id'] for r in manifest})==len(manifest)
    write_rows(OUT/'manifest.jsonl',manifest)
    write_json(OUT/'data_selection.json',dict(complete=True,seed=seed,counts=dict(Counter(r['experiment_split'] for r in manifest)),
        quotas=counts,retained_train_val_source=str(BASE/'manifest.jsonl'),retained_source_sha256=sha(BASE/'manifest.jsonl'),
        historical_manifest_sha256={str(p):sha(p) for p in history},historical_pixel_count=historical_pixels,historical_dhash_count=historical_dhashes,
        public_manifest_sha256={str(p):sha(p) for p in sources},fresh_pixels_verified=192,near_exclusion_radius=4,
        selection_used_predictions=False,manifest_sha256=sha(OUT/'manifest.jsonl')))
    print('PREPARED '+str(OUT/'manifest.jsonl'),flush=True)


if __name__=='__main__':main()
