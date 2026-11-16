"""Download pinned, SHA-verified GUI training shards, not benchmark labels."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time
from urllib.parse import urlparse,parse_qs


ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'data/conditioned_v3'
SHARDS=(0,8,15,23,30,38,45,53,60,74)
CHUNK=8*1024*1024


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def main():
    meta=json.loads((DEST/'salesforce_repository.json').read_text())
    files=[f for f in meta['siblings'] if f['rfilename'].endswith('.parquet')]
    selected=[files[i] for i in SHARDS]
    part=DEST/'download_parts_8mb';part.mkdir(exist_ok=True)
    folder=DEST/'training_shards';folder.mkdir(exist_ok=True)
    logs=DEST/'download_logs';logs.mkdir(exist_ok=True)
    # Reuse complete 8-MiB prefixes from the earlier interrupted 32-MiB
    # transfers. Final full-file official SHA remains mandatory.
    for f in selected:
        name=Path(f['rfilename']).name
        for old in (DEST/'download_parts').glob(name+'.*'):
            if old.suffix not in ('.bin','.partial'):continue
            start=int(old.name.split('.')[-2])
            with old.open('rb') as src:
                offset=start
                while offset<f['size']:
                    length=min(CHUNK,f['size']-offset);data=src.read(length)
                    if len(data)!=length:break
                    target=part/f'{name}.{offset}.bin'
                    if not target.exists():target.write_bytes(data)
                    offset+=length
    jobs=[];pending=[]
    for f in selected:
        target=folder/Path(f['rfilename']).name
        if target.exists() and target.stat().st_size==f['size'] and digest(target)==f['lfs']['sha256']:continue
        pending.append(f)
        for start in range(0,f['size'],CHUNK):jobs.append((f,start,min(f['size'],start+CHUNK)-1))
    # Reuse still-valid public CDN redirects. Unique resolver query strings
    # bypass useful cached redirects and can time out at the HF host itself.
    # The signed public links cover the whole file, not a particular range.
    direct={}
    header_paths=list(part.glob('*.headers'))+list(DEST.glob('probe_*.headers'))
    for headers in header_paths:
        for url in re.findall(r'^location:\s*(\S+)',headers.read_text(errors='replace'),re.I|re.M):
            parsed=urlparse(url)
            if not parsed.hostname or not parsed.hostname.endswith('cdn.hf.co'):continue
            expires=int(parse_qs(parsed.query).get('Expires',['0'])[0])
            if expires<time.time()+180:continue
            for f in pending:
                name=Path(f['rfilename']).name
                if name in url:direct[name]=url
    def fetch(job):
        f,start,end=job;name=Path(f['rfilename']).name;target=part/f'{name}.{start}.bin';length=end-start+1
        if target.exists() and target.stat().st_size==length:return
        temp=target.with_suffix('.partial');piece=target.with_suffix('.piece');headers=target.with_suffix('.headers')
        # An interrupted curl may leave an uncommitted, valid range prefix.
        received=temp.stat().st_size if temp.exists() else 0
        ranges=re.findall(r'content-range:\s*bytes\s+(\d+)-(\d+)/',headers.read_text(errors='replace'),re.I) if headers.exists() else []
        if piece.exists() and ranges and int(ranges[-1][0])==start+received:
            data=piece.read_bytes()
            if 0<len(data)<=length-received:
                with temp.open('ab') as dst:dst.write(data)
        for attempt in range(12):
            received=temp.stat().st_size if temp.exists() else 0
            if received==length:temp.replace(target);return
            begin=start+received
            resolver=f"https://hf-mirror.com/datasets/Salesforce/grounding_dataset/resolve/{meta['sha']}/{f['rfilename']}"
            url=direct.get(name,resolver) if attempt%4!=3 else resolver
            if piece.exists():piece.unlink()
            with (logs/f'{name}.{start}.small.log').open('ab') as log:
                code=subprocess.run(['curl.exe','--fail','--location','--connect-timeout','10','--max-time','90',
                    '--header','Cache-Control: no-cache',
                    '--range',f'{begin}-{end}','--dump-header',str(headers),'--output',str(piece),url],stdout=log,stderr=log).returncode
            ranges=re.findall(r'content-range:\s*bytes\s+(\d+)-(\d+)/',headers.read_text(errors='replace'),re.I) if headers.exists() else []
            for redirect in re.findall(r'^location:\s*(\S+)',headers.read_text(errors='replace'),re.I|re.M) if headers.exists() else []:
                parsed=urlparse(redirect)
                if parsed.hostname and parsed.hostname.endswith('cdn.hf.co'):direct[name]=redirect
            if piece.exists() and code in (0,18,28,56) and ranges and int(ranges[-1][0])==begin:
                data=piece.read_bytes()
                if 0<len(data)<=end-begin+1:
                    with temp.open('ab') as dst:dst.write(data)
            if temp.exists() and temp.stat().st_size==length:
                temp.replace(target);return
        raise RuntimeError(f'Failed {name}:{start} after resumable attempts')
    print(f'Downloading {len(selected)} train shards / {len(jobs)} ranges',flush=True)
    errors=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        for i,future in enumerate(concurrent.futures.as_completed([pool.submit(fetch,j) for j in jobs]),1):
            try:future.result()
            except Exception as exc:errors.append(str(exc));print('ERROR '+str(exc),flush=True)
            if i%16==0 or i==len(jobs):print(f'{i}/{len(jobs)} ranges',flush=True)
    if errors:raise RuntimeError(errors)
    for f in pending:
        target=folder/Path(f['rfilename']).name;temp=target.with_suffix('.assembled')
        with temp.open('wb') as dst:
            for start in range(0,f['size'],CHUNK):
                with (part/f'{target.name}.{start}.bin').open('rb') as src:
                    for b in iter(lambda:src.read(8*1024*1024),b''):dst.write(b)
        if digest(temp)!=f['lfs']['sha256']:raise ValueError('SHA mismatch '+target.name)
        temp.replace(target);print('VERIFIED '+target.name,flush=True)
    record={'repo':'Salesforce/grounding_dataset','revision':meta['sha'],'official_split':'train','shard_indices':SHARDS,
        'files':{Path(f['rfilename']).name:f['lfs']['sha256'] for f in selected},'all_sha_verified':True,
        'source':'https://huggingface.co/datasets/Salesforce/grounding_dataset','download_mirror':'https://hf-mirror.com'}
    (DEST/'training_download_verified.json').write_text(json.dumps(record,indent=2)+'\n')
    print('COMPLETE verified GUI training shards',flush=True)


if __name__=='__main__':main()
