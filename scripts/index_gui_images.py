"""Canonical pixel hashes and perceptual fingerprints for leakage screening."""
import concurrent.futures
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from clickagent.action_benchmark import sha256,write_json


def fingerprint(path):
    with Image.open(path) as opened:
        im=opened.convert('RGB');w,h=im.size
        digest=hashlib.sha256(f'{w}x{h}:RGB:'.encode()+im.tobytes()).hexdigest()
        gray=np.asarray(im.convert('L').resize((17,16),Image.Resampling.LANCZOS))
    bits=(gray[:,1:]>gray[:,:-1]).ravel()
    phash=np.packbits(bits).tobytes().hex()
    return {'image':str(path),'file_sha256':sha256(path),'pixel_sha256':digest,'dhash256':phash,'image_size':[w,h]}


def main():
    dest=Path('data/conditioned_v3/public_image_index.json')
    paths=set()
    manifests=sorted(Path('data/public_v1/manifests').glob('*.jsonl'))
    for manifest in manifests:
        paths.update(json.loads(line)['image'] for line in manifest.read_text(encoding='utf-8').splitlines())
    rows={}
    if dest.exists():rows={r['image']:r for r in json.loads(dest.read_text())['images']}
    pending=[p for p in sorted(paths) if p not in rows]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        for i,r in enumerate(pool.map(fingerprint,pending),1):
            rows[r['image']]=r
            if i%200==0:
                write_json(dest,{'images':list(rows.values()),'complete':False})
                print(f'Indexed {i}/{len(pending)}',flush=True)
    write_json(dest,{'images':[rows[p] for p in sorted(paths)],'complete':True,
        'manifests':{str(p):sha256(p) for p in manifests},'canonical':'sha256(width x height + RGB bytes)',
        'near_duplicate_screen':'256-bit horizontal dHash; Hamming <= 4 conservatively excludes training images'})
    print(f'COMPLETE indexed {len(rows)} referenced public screenshots',flush=True)


if __name__=='__main__':main()
