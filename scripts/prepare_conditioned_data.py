"""Image-disjoint training/validation from a train corpus, fresh public test."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from clickagent.action_benchmark import sha256,write_json
from run_target_token_experiment import read_json,read_jsonl,key


ROOT=Path(__file__).resolve().parents[1];DATA=ROOT/'data/conditioned_v3';OUTPUT=ROOT/'runs/conditioned_v3'
SEED=20261101


def write_rows(path,rows):path.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf-8')


def prepare_test(index):
    path=DATA/'test_manifest.jsonl'
    if path.exists():return read_jsonl(path)
    by_path={r['image']:r for r in index['images']}
    prior=list((ROOT/'runs/public_v1').glob('*/selected_manifest.jsonl'))
    prior += [ROOT/f'runs/{run}/manifest.jsonl' for run in ('candidate_v2','target_token_v1','semantic_target_v2')]
    used={by_path[c['image']]['pixel_sha256'] for p in prior for c in read_jsonl(p)}
    chosen=[];seen=set(used)
    for ds in ('screenspot_v2','screenspot_pro','osworld_g','venusbench_gd'):
        rows=read_jsonl(ROOT/f'data/public_v1/manifests/{ds}.jsonl')
        rows.sort(key=lambda c:key(f'{SEED}:test:{c["sample_id"]}'))
        selected=[]
        for c in rows:
            if not c['expected_actionable'] or not c.get('target_geometry'):continue
            info=by_path[c['image']]
            if info['pixel_sha256'] in seen:continue
            seen.add(info['pixel_sha256'])
            selected.append(dict(c,experiment_split='test',image_sha256=info['file_sha256'],
                pixel_sha256=info['pixel_sha256'],dhash256=info['dhash256']))
            if len(selected)==48:break
        if len(selected)!=48:raise ValueError(f'Insufficient untouched images: {ds}: {len(selected)}')
        chosen.extend(selected)
    write_rows(path,chosen)
    write_json(DATA/'test_selection.json',{'seed':SEED,'n':len(chosen),'per_dataset':48,
        'prior_manifest_files':[str(p) for p in prior],'prior_pixel_hashes_excluded':len(used),
        'test_pixel_images_unique':len({c['pixel_sha256'] for c in chosen}),
        'model_outputs_used_for_selection':False,'reference_image_index_sha256':sha256(DATA/'public_image_index.json')})
    return chosen


def iou(a,b):
    inter=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)


def select_distinct_targets(rows,limit=2):
    selected=[]
    for r in sorted(rows,key=lambda c:key(f'{SEED}:target:{c["sample_id"]}')):
        box=r['target_geometry']['boxes'][0]
        if all(iou(box,s['target_geometry']['boxes'][0])<.2 and r['instruction'].casefold()!=s['instruction'].casefold() for s in selected):
            selected.append(r)
        if len(selected)==limit:break
    return selected


def prepare_training(index,test,train_images=512,val_images=128):
    verified=read_json(DATA/'training_download_verified.json');assert verified['all_sha_verified']
    pool_key=hashlib.sha256((sha256(DATA/'training_download_verified.json')+sha256(DATA/'public_image_index.json')+sha256(Path(__file__))).encode()).hexdigest()[:12]
    cache=DATA/f'training_pool_{pool_key}.jsonl';stats_path=DATA/f'training_pool_{pool_key}_stats.json'
    images=DATA/'source_images';images.mkdir(exist_ok=True)
    stats=Counter();groups=defaultdict(list)
    if cache.exists():
        for r in read_jsonl(cache):groups[r['pixel_sha256']].append(r)
        stats.update(read_json(stats_path))
    else:
        import pyarrow.parquet as pq
        reference_pixels={r['pixel_sha256'] for r in index['images']}
        reference_phashes=[int(r['dhash256'],16) for r in index['images']]
        encoded={}
        info_cache=DATA/('source_image_info_'+sha256(DATA/'public_image_index.json')[:12]);info_cache.mkdir(exist_ok=True)
        def identify(item):
            file_hash,data=item;cached=info_cache/f'{file_hash}.json';destination=images/f'{file_hash}.img'
            if cached.exists():
                info=read_json(cached)
                assert info['file_sha256']==file_hash
                if info['allowed'] and not destination.exists():destination.write_bytes(data)
                return file_hash,info
            with Image.open(io.BytesIO(data)) as opened:
                im=opened.convert('RGB');w,h=im.size
                pixel=hashlib.sha256(f'{w}x{h}:RGB:'.encode()+im.tobytes()).hexdigest()
                gray=np.asarray(im.convert('L').resize((17,16),Image.Resampling.LANCZOS))
                phash=np.packbits((gray[:,1:]>gray[:,:-1]).ravel()).tobytes().hex()
            near=any((int(phash,16)^v).bit_count()<=4 for v in reference_phashes)
            info={'allowed':pixel not in reference_pixels and not near,'pixel':pixel,'phash':phash,
                'size':[w,h],'file_sha256':file_hash}
            if info['allowed'] and not destination.exists():destination.write_bytes(data)
            temporary=cached.with_suffix('.tmp');write_json(temporary,info);temporary.replace(cached)
            return file_hash,info
        pool=ThreadPoolExecutor(max_workers=4)
        for name,digest in verified['files'].items():
            path=DATA/'training_shards'/name
            if sha256(path)!=digest:raise ValueError('Training shard changed '+name)
            parquet=pq.ParquetFile(path)
            for batch in parquet.iter_batches(batch_size=12,columns=['dataset','uuid','image','bbox','instruction']):
                raw_rows=batch.to_pylist();identified=[];pending={}
                for raw in raw_rows:
                    data=raw['image']['bytes'];file_hash=hashlib.sha256(data).hexdigest();identified.append((raw,file_hash))
                    if file_hash not in encoded:pending[file_hash]=data
                for file_hash,info in pool.map(identify,pending.items()):
                    encoded[file_hash]=info;stats['unique_source_images']+=1
                    if not info['allowed']:stats['images_excluded_public_exact_or_near']+=1
                for raw,file_hash in identified:
                    stats['source_rows']+=1
                    info=encoded[file_hash]
                    if not info['allowed']:stats['rows_excluded_public_exact_or_near']+=1;continue
                    w,h=info['size'];box=[float(x) for x in raw['bbox']];text=raw['instruction']
                    if len(box)!=4 or not (0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h) or not isinstance(text,str) or not text.strip():
                        stats['invalid_annotation_rows']+=1;continue
                    norm=[box[0]/w,box[1]/h,box[2]/w,box[3]/h]
                    case={'sample_id':'salesforce:'+raw['uuid'],'dataset':'Salesforce/'+raw['dataset'],
                        'source_split':'train','official_training_split':True,'instruction':text.strip(),
                        'image':str((images/f'{file_hash}.img').resolve()),'image_size':[w,h],
                        'image_sha256':file_hash,'pixel_sha256':info['pixel'],'dhash256':info['phash'],
                        'expected_actionable':True,'task_kind':'grounding_present',
                        'target_geometry':{'type':'bbox','boxes':[norm]},'group_id':info['pixel'],
                        'training_repo_revision':verified['revision']}
                    groups[info['pixel']].append(case)
            print(f'Parsed {name}: {len(groups)} eligible unique images',flush=True)
        pool.shutdown(wait=True)
        rows=[r for group in groups.values() for r in group]
        # Some shards repeat exact annotation UUIDs. Deduplicate before grouping.
        unique={r['sample_id']:r for r in rows};groups=defaultdict(list)
        for r in unique.values():groups[r['pixel_sha256']].append(r)
        write_rows(cache,list(unique.values()));write_json(stats_path,dict(stats))
    buckets=defaultdict(list)
    for pixel,rows in groups.items():
        selected=select_distinct_targets(rows)
        if selected:buckets[selected[0]['dataset']].append((pixel,selected))
    for ds in buckets:buckets[ds].sort(key=lambda r:key(f'{SEED}:image:{r[0]}'))
    round_robin=[]
    while any(buckets.values()):
        for ds in sorted(buckets):
            if buckets[ds]:round_robin.append(buckets[ds].pop(0))
    val=[];train=[];used_phashes=[]
    # Keep near-duplicate scenes together by conservatively excluding any later
    # image whose dHash is within 4 bits of an already selected scene.
    for pixel,rows in round_robin:
        phash=int(rows[0]['dhash256'],16)
        if any((phash^v).bit_count()<=4 for v in used_phashes):continue
        if len(val)<val_images:val.append((pixel,rows))
        elif len(train)<train_images:train.append((pixel,rows))
        else:break
        used_phashes.append(phash)
    if len(train)!=train_images or len(val)!=val_images:
        raise ValueError(f'Need more official training shards: train images {len(train)}/{train_images}, val {len(val)}/{val_images}')
    rows=[]
    for split,items in (('train',train),('val',val)):
        for pixel,targets in items:rows.extend(dict(r,experiment_split=split) for r in targets)
    train_pixels={r['pixel_sha256'] for r in rows if r['experiment_split']=='train'}
    val_pixels={r['pixel_sha256'] for r in rows if r['experiment_split']=='val'}
    test_pixels={r['pixel_sha256'] for r in test}
    assert not train_pixels&val_pixels and not (train_pixels|val_pixels)&test_pixels
    manifest=rows+test
    OUTPUT.mkdir(exist_ok=True)
    write_rows(OUTPUT/'manifest.jsonl',manifest)
    write_json(OUTPUT/'selection.json',{'seed':SEED,'training_source':verified,
        'counts':dict(Counter(r['experiment_split'] for r in manifest)),
        'unique_images':{'train':len(train_pixels),'val':len(val_pixels),'test':len(test_pixels)},
        'all_training_images_excluded_all_public_exact_and_near_duplicates':True,
        'train_val_near_duplicates_excluded':True,'per_image_max_training_targets':2,
        'training_sources':dict(Counter(r['dataset'] for r in rows if r['experiment_split']=='train')),
        'preparer_source_sha256':sha256(Path(__file__)),
        'all_splits_pixel_disjoint':True,'pool_stats':dict(stats),
        'test_selection_sha256':sha256(DATA/'test_selection.json'),'manifest_sha256':sha256(OUTPUT/'manifest.jsonl')})
    print(json.dumps(read_json(OUTPUT/'selection.json'),ensure_ascii=False),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--phase',choices=('test','all'),default='all')
    args=p.parse_args();index=read_json(DATA/'public_image_index.json');assert index['complete']
    test=prepare_test(index)
    if args.phase=='test':print(f'Frozen untouched public test: {len(test)}');return
    prepare_training(index,test)


if __name__=='__main__':main()
