"""Native-curl ModelScope mirror download, verified against official LFS SHAs."""
import concurrent.futures
import hashlib
import json
from pathlib import Path
import subprocess
import time
from urllib.parse import quote


ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/"models/UI-TARS-1.5-7B"


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    files=json.loads((ROOT/'runs/ui_tars_modelscope_files.json').read_text(encoding='utf-8'))['Data']['Files']
    official=json.loads((ROOT/'runs/ui_tars_official_metadata.json').read_text(encoding='utf-8'))
    by_name={r['rfilename']:r for r in official['siblings']}
    chosen=[r for r in files if r['Path'].endswith(('.json','.txt','.safetensors'))]
    for r in chosen:
        if r['Path'].endswith('.safetensors'):
            assert by_name[r['Path']]['lfs']['sha256']==r['Sha256'],r['Path']
    DEST.mkdir(exist_ok=True)
    (DEST/'native_download_logs').mkdir(exist_ok=True)
    def download(record):
        name=record['Path']
        if '/' in name or '\\' in name:raise ValueError('Unexpected nested file')
        target=DEST/name
        if target.exists() and target.stat().st_size==record['Size'] and digest(target)==record['Sha256']:
            print('VERIFIED cached '+name,flush=True);return
        part=target.with_suffix(target.suffix+'.curlpart')
        url='https://modelscope.cn/api/v1/models/bytedance-community/UI-TARS-1.5-7B/repo?Revision=34beb1c6dfc6956d4ad58a2fdf6c6da38ecd8808&FilePath='+quote(name,safe='')
        with (DEST/'native_download_logs'/(name+'.log')).open('wb') as log:
            cmd=['curl.exe','--fail','--location','--retry','4','--retry-delay','3','--connect-timeout','30',
                 '--max-time','7200','--continue-at','-','--output',str(part),url]
            result=subprocess.run(cmd,stdout=log,stderr=log)
        if result.returncode:raise RuntimeError(f'curl failed for {name}, exit {result.returncode}')
        if part.stat().st_size!=record['Size'] or digest(part)!=record['Sha256']:
            raise RuntimeError('Size/SHA mismatch '+name)
        part.replace(target)
        print('VERIFIED '+name,flush=True)
    print(f'Downloading {len(chosen)} files; bytes={sum(r["Size"] for r in chosen)}',flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=7) as pool:
        futures=[pool.submit(download,r) for r in chosen]
        for future in concurrent.futures.as_completed(futures):future.result()
    record={'source':'ModelScope bytedance-community mirror','official_model':'ByteDance-Seed/UI-TARS-1.5-7B',
        'official_revision':official['sha'],'mirror_revision':'34beb1c6dfc6956d4ad58a2fdf6c6da38ecd8808',
        'official_weight_lfs_sha_matches':True,'all_download_sha256_verified':True,
        'files':{r['Path']:{'size':r['Size'],'sha256':r['Sha256']} for r in chosen}}
    (DEST/'clickagent_download_verified.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print('COMPLETE verified official UI-TARS weights',flush=True)


if __name__=='__main__':main()
