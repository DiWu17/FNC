"""Resumable parallel native-curl ranges for verified public ModelScope weights."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
from urllib.parse import quote

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/"models/UI-TARS-1.5-7B"
REV="34beb1c6dfc6956d4ad58a2fdf6c6da38ecd8808"
CHUNK=128*1024*1024


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def url(name):
    return 'https://modelscope.cn/api/v1/models/bytedance-community/UI-TARS-1.5-7B/repo?Revision='+REV+'&FilePath='+quote(name,safe='')


def main():
    files=json.loads((ROOT/'runs/ui_tars_modelscope_files.json').read_text())['Data']['Files']
    hf=json.loads((ROOT/'runs/ui_tars_official_metadata.json').read_text())
    official={r['rfilename']:r for r in hf['siblings']}
    chosen=[r for r in files if r['Path'].endswith(('.json','.txt','.safetensors'))]
    part_dir=DEST/'range_parts';part_dir.mkdir(exist_ok=True)
    logs=DEST/'range_logs';logs.mkdir(exist_ok=True)
    jobs=[];weights=[]
    for r in chosen:
        name=r['Path'];target=DEST/name
        if '/' in name or '\\' in name:raise ValueError(name)
        if name.endswith('.safetensors'):
            assert r['Sha256']==official[name]['lfs']['sha256']
        if target.exists() and target.stat().st_size==r['Size'] and sha(target)==r['Sha256']:
            print('VERIFIED cached '+name,flush=True);continue
        weights.append(r)
        for start in range(0,r['Size'],CHUNK):
            end=min(r['Size'],start+CHUNK)-1
            jobs.append((name,start,end))
    def fetch(job):
        name,start,end=job;path=part_dir/f'{name}.{start}.bin'
        length=end-start+1
        if path.exists() and path.stat().st_size==length:return
        temp=path.with_suffix('.partial')
        with (logs/(path.name+'.log')).open('wb') as log:
            cmd=['curl.exe','--fail','--location','--retry','4','--retry-delay','2','--connect-timeout','30',
                 '--max-time','600','--range',f'{start}-{end}','--output',str(temp),url(name)]
            code=subprocess.run(cmd,stdout=log,stderr=log).returncode
        if code or temp.stat().st_size!=length:raise RuntimeError(f'Range failed {name}:{start}, exit={code}')
        temp.replace(path)
    print(f'Parallel ranges: {len(jobs)}, workers=24, chunk={CHUNK}',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=24) as pool:
        for i,f in enumerate(concurrent.futures.as_completed([pool.submit(fetch,j) for j in jobs]),1):
            f.result()
            if i%8==0 or i==len(jobs):print(f'Ranges complete {i}/{len(jobs)}',flush=True)
    for r in weights:
        target=DEST/r['Path'];temp=target.with_suffix(target.suffix+'.assembled')
        h=hashlib.sha256()
        with temp.open('wb') as out:
            for start in range(0,r['Size'],CHUNK):
                with (part_dir/f'{r["Path"]}.{start}.bin').open('rb') as src:
                    for b in iter(lambda:src.read(8*1024*1024),b''):out.write(b);h.update(b)
        if temp.stat().st_size!=r['Size'] or h.hexdigest()!=r['Sha256']:raise RuntimeError('SHA mismatch '+r['Path'])
        temp.replace(target)
        print('VERIFIED '+r['Path'],flush=True)
    record={'source':'ModelScope bytedance-community mirror','official_model':'ByteDance-Seed/UI-TARS-1.5-7B',
        'official_revision':hf['sha'],'mirror_revision':REV,'official_weight_lfs_sha_matches':True,
        'all_download_sha256_verified':True,'files':{r['Path']:{'size':r['Size'],'sha256':r['Sha256']} for r in chosen}}
    (DEST/'clickagent_download_verified.json').write_text(json.dumps(record,indent=2)+'\n')
    print('COMPLETE verified official UI-TARS weights',flush=True)


if __name__=='__main__':main()
