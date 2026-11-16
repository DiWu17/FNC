"""Download pinned public reference assets and verify publisher file hashes."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import time

import requests

ROOT = Path(__file__).resolve().parents[1]
SETUP = ROOT / 'runs/advantage_validation_v17/setup'
MODELS = {'aima': ('smz8599/GUI-AIMA-3B', 'GUI-AIMA-3B'),
          'tag': ('openbmb/MiniCPM-Llama3-V-2_5', 'MiniCPM-Llama3-V-2_5')}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    for name, (repo, folder) in MODELS.items():
        metadata_path = SETUP / f'{name}_official_metadata.json'
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        dest = ROOT / 'models' / folder
        dest.mkdir(parents=True, exist_ok=True)
        files = [f for f in metadata['siblings'] if '/' not in f['rfilename'] and
                 f['rfilename'].endswith(('.json', '.safetensors', '.model', '.txt'))]
        print(json.dumps({'model': repo, 'revision': metadata['sha'],
                          'files': len(files)}, ensure_ascii=False), flush=True)

        def fetch(info):
            filename = info['rfilename']
            assert Path(filename).name == filename
            path = dest / filename
            part = path.with_suffix(path.suffix + '.part')
            lfs = info.get('lfs')
            expected_size = info.get('size') or (lfs or {}).get('size')
            for attempt in range(8):
                try:
                    if not path.exists():
                        offset = part.stat().st_size if part.exists() else 0
                        response = requests.get(
                            f'https://huggingface.co/{repo}/resolve/{metadata["sha"]}/{filename}?download=true',
                            headers={'Range': f'bytes={offset}-'} if offset else {},
                            stream=True, timeout=(30, 120))
                        response.raise_for_status()
                        if offset and (response.status_code != 206 or not response.headers.get('Content-Range', '').startswith(f'bytes {offset}-')):
                            offset = 0
                        with part.open('ab' if offset else 'wb') as stream:
                            count = offset
                            report = count
                            for chunk in response.iter_content(4 * 1024 * 1024):
                                stream.write(chunk)
                                count += len(chunk)
                                if count - report >= 512 * 1024 * 1024:
                                    print(f'{repo} {filename}: {count}/{expected_size}', flush=True)
                                    report = count
                        if expected_size is not None:
                            assert part.stat().st_size == expected_size, filename
                        part.replace(path)
                    actual_sha = sha(path)
                    if lfs:
                        assert actual_sha == lfs['sha256'], filename
                    else:
                        data = path.read_bytes()
                        assert hashlib.sha1(f'blob {len(data)}\0'.encode() + data).hexdigest() == info['blobId'], filename
                    print(f'VERIFIED {repo} {filename}', flush=True)
                    return {'filename': filename, 'bytes': path.stat().st_size, 'sha256': actual_sha}
                except Exception as error:
                    print(f'RETRY {repo} {filename} attempt={attempt + 1}: {type(error).__name__}', flush=True)
                    if attempt == 7:
                        raise
                    time.sleep(min(30, 2 ** attempt))

        with ThreadPoolExecutor(max_workers=4) as pool:
            result = list(pool.map(fetch, files))
        receipt = dict(complete=True, official_model=repo, official_revision=metadata['sha'],
                       official_metadata_sha256=sha(metadata_path), all_files_verified=True,
                       files=result, remote_code_executed=False)
        (dest / 'clickagent_download_verified_v17.json').write_text(
            json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
