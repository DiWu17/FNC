"""Inspect verified complete shard ranges while the corpus download continues."""
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as pq


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    root=Path('data/conditioned_v3');meta=json.loads((root/'salesforce_repository.json').read_text())
    stats=[]
    for item in meta['siblings']:
        name=Path(item['rfilename']).name
        if not name.endswith('.parquet'):continue
        target=root/'training_shards'/name
        if not target.exists():
            pieces=[root/'download_parts_8mb'/f'{name}.{i}.bin' for i in range(0,item['size'],8*1024*1024)]
            if not all(p.exists() and p.stat().st_size==min(8*1024*1024,item['size']-i*8*1024*1024) for i,p in enumerate(pieces)):continue
            temporary=target.with_suffix('.preflight')
            with temporary.open('wb') as dst:
                for piece in pieces:dst.write(piece.read_bytes())
            assert digest(temporary)==item['lfs']['sha256']
            temporary.replace(target)
        assert digest(target)==item['lfs']['sha256']
        sources={};images={};rows=0
        for batch in pq.ParquetFile(target).iter_batches(batch_size=8,columns=['dataset','image']):
            for row in batch.to_pylist():
                source=row['dataset'];sources[source]=sources.get(source,0)+1;rows+=1
                images.setdefault(source,set()).add(hashlib.sha256(row['image']['bytes']).hexdigest())
        stats.append(dict(file=name,rows=rows,sources=sources,unique_images={k:len(v) for k,v in images.items()}))
        print(json.dumps(stats[-1]),flush=True)
    (root/'shard_preflight.json').write_text(json.dumps(stats,indent=2)+'\n')


if __name__=='__main__':main()
