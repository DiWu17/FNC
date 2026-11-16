"""Accelerate pinned AIMA weights with independently verified HTTP ranges."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import time

import requests
import download_advantage_models_v17 as original

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'models/GUI-AIMA-3B'
CHUNK = 64 * 1024 * 1024


def main():
    metadata = json.loads((ROOT / 'runs/advantage_validation_v17/setup/aima_official_metadata.json').read_text(encoding='utf-8'))
    parts = DEST / 'range_parts_v17'
    parts.mkdir(parents=True, exist_ok=True)
    weights = [r for r in metadata['siblings'] if r['rfilename'].endswith('.safetensors')]
    jobs = []
    for row in weights:
        target = DEST / row['rfilename']
        if target.exists() and original.sha(target) == row['lfs']['sha256']:
            continue
        size = row['lfs']['size']
        jobs.extend((row['rfilename'], start, min(start + CHUNK, size) - 1) for start in range(0, size, CHUNK))

    def fetch(job):
        name, start, end = job
        path = parts / f'{name}.{start}.range'
        length = end - start + 1
        if path.exists() and path.stat().st_size == length:
            return
        url = f'https://huggingface.co/smz8599/GUI-AIMA-3B/resolve/{metadata["sha"]}/{name}?download=true'
        for attempt in range(6):
            try:
                temp = path.with_suffix('.partial')
                offset = temp.stat().st_size if temp.exists() else 0
                assert 0 <= offset <= length, (name, start, offset, length)
                if offset < length:
                    resume_start = start + offset
                    with requests.get(url, headers={'Range': f'bytes={resume_start}-{end}'}, stream=True, timeout=(30, 120)) as response:
                        response.raise_for_status()
                        assert response.status_code == 206 and response.headers.get('Content-Range', '').startswith(f'bytes {resume_start}-{end}/'), response.headers.get('Content-Range')
                        with temp.open('ab' if offset else 'wb') as stream:
                            for chunk in response.iter_content(4 * 1024 * 1024):
                                stream.write(chunk)
                assert temp.stat().st_size == length
                temp.replace(path)
                return
            except Exception as error:
                print(f'Range retry {name}:{start} {type(error).__name__}', flush=True)
                if attempt == 5:
                    raise
                time.sleep(min(20, 2 ** attempt))

    print(f'AIMA ranges {len(jobs)}, workers=12', flush=True)
    with ThreadPoolExecutor(max_workers=12) as pool:
        for i, future in enumerate(as_completed([pool.submit(fetch, j) for j in jobs]), 1):
            future.result()
            if i % 8 == 0 or i == len(jobs):
                print(f'Ranges {i}/{len(jobs)}', flush=True)
    for row in weights:
        target = DEST / row['rfilename']
        if target.exists() and original.sha(target) == row['lfs']['sha256']:
            continue
        assembled = target.with_suffix('.safetensors.ranges_assembled')
        h = hashlib.sha256()
        with assembled.open('wb') as output:
            for start in range(0, row['lfs']['size'], CHUNK):
                with (parts / f'{row["rfilename"]}.{start}.range').open('rb') as stream:
                    for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                        h.update(chunk)
                        output.write(chunk)
        assert assembled.stat().st_size == row['lfs']['size'] and h.hexdigest() == row['lfs']['sha256']
        assembled.replace(target)
        print('VERIFIED ' + row['rfilename'], flush=True)
    original.MODELS = {'aima': ('smz8599/GUI-AIMA-3B', 'GUI-AIMA-3B')}
    original.main()


if __name__ == '__main__':
    main()
