"""Inspect public model hosts; never execute downloaded repository code."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import requests


OUT = Path('runs/context_verifier_v9')


def probe(url):
    try:
        r = requests.get(url, timeout=25)
        return dict(url=url, status=r.status_code, body=r.json() if r.status_code == 200 else r.text[:160])
    except Exception as e:
        return dict(url=url, error=type(e).__name__+': '+str(e)[:160])


def main():
    OUT.mkdir(exist_ok=True)
    urls = [f'https://modelscope.cn/api/v1/models/{name}/repo/files?Revision=master&Recursive=true'
            for name in ['microsoft/GUI-Actor-Verifier-2B', 'AI-ModelScope/GUI-Actor-Verifier-2B']]
    urls += ['https://huggingface.co/api/models/microsoft/GUI-Actor-Verifier-2B?blobs=true',
             'https://hf-mirror.com/api/models/microsoft/GUI-Actor-Verifier-2B?blobs=true']
    with ThreadPoolExecutor(max_workers=4) as pool:
        result = list(pool.map(probe, urls))
    (OUT/'verifier_host_probe.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    for item in result:
        body = item.get('body', {})
        print(json.dumps(dict(url=item['url'], status=item.get('status'), error=item.get('error'),
            sha=body.get('sha') if isinstance(body,dict) else None,
            files=len(body.get('siblings',[])) if isinstance(body,dict) else None)), flush=True)


if __name__ == '__main__':
    main()
