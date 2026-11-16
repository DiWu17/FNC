"""Download safe public model assets from ModelScope; verify official HF hashes."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import requests

ROOT = Path('models/GUI-Actor-Verifier-2B')


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def main():
    ROOT.mkdir(exist_ok=True)
    probes = json.loads(Path('runs/context_verifier_v9/verifier_host_probe.json').read_text(encoding='utf-8'))
    ms = probes[0]['body']; hf = probes[2]['body']
    assert ms['Success'] and hf['id'] == 'microsoft/GUI-Actor-Verifier-2B'
    official = {v['rfilename']: v for v in hf['siblings']}
    files = [v for v in ms['Data']['Files'] if v['Path'] in official and
             v['Path'] not in {'.gitattributes', 'training_args.bin', 'trainer_state.json'}]
    def fetch(info):
        name=info['Path']; dest=ROOT/name; part=dest.with_suffix(dest.suffix+'.part')
        url=f'https://modelscope.cn/models/microsoft/GUI-Actor-Verifier-2B/resolve/master/{name}'
        if not dest.exists():
            offset=part.stat().st_size if part.exists() else 0
            response=requests.get(url,stream=True,headers={'Range':f'bytes={offset}-'} if offset else {},timeout=(30,120))
            response.raise_for_status()
            if offset and response.status_code != 206: offset=0
            with part.open('ab' if offset else 'wb') as f:
                written=offset; report=written
                for data in response.iter_content(4*1024*1024):
                    f.write(data);written+=len(data)
                    if written-report>=256*1024*1024:
                        print(f'{name} {written}/{info["Size"]}',flush=True);report=written
            assert part.stat().st_size==info['Size'],name
            part.replace(dest)
        sha=digest(dest);assert sha==info['Sha256'],name
        raw=official[name]
        if raw.get('lfs'): assert sha==raw['lfs']['sha256'],name
        else:
            data=dest.read_bytes();git=hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()
            assert git==raw['blobId'],(name,git,raw['blobId'])
        print('VERIFIED '+name,flush=True)
        return dict(file=name,size=dest.stat().st_size,sha256=sha,official_hf_blob=raw['blobId'])
    with ThreadPoolExecutor(max_workers=4) as pool: result=list(pool.map(fetch,files))
    (ROOT/'clickagent_download_verified.json').write_text(json.dumps(dict(complete=True,
        official_model='microsoft/GUI-Actor-Verifier-2B',official_revision=hf['sha'],
        source='ModelScope microsoft mirror',files=result,remote_code_executed=False),indent=2),encoding='utf-8')


if __name__=='__main__': main()
