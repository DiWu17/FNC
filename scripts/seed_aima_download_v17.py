"""Reuse complete prefix chunks from the existing public-weight download.

This only saves duplicate network transfer. The range downloader still checks the
assembled file against the publisher's SHA256 before publishing any checkpoint.
"""
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEST = (ROOT / 'models/GUI-AIMA-3B').resolve()
CHUNK = 64 * 1024 * 1024


def main():
    metadata = json.loads((ROOT / 'runs/advantage_validation_v17/setup/aima_official_metadata.json').read_text(encoding='utf-8'))
    parts = DEST / 'range_parts_v17'
    assert DEST.parent == (ROOT / 'models').resolve()
    parts.mkdir(exist_ok=True)
    records = []
    for row in metadata['siblings']:
        name = row['rfilename']
        if not name.endswith('.safetensors'):
            continue
        source = DEST / (name + '.part')
        if not source.exists():
            continue
        available = min(source.stat().st_size, row['lfs']['size'])
        copied = skipped = 0
        with source.open('rb') as stream:
            for start in range(0, available, CHUNK):
                length = min(CHUNK, row['lfs']['size'] - start)
                if start + length > available:
                    break
                target = parts / f'{name}.{start}.range'
                if target.exists() and target.stat().st_size == length:
                    skipped += 1
                    continue
                stream.seek(start)
                data = stream.read(length)
                if len(data) != length:
                    break
                temp = target.with_suffix('.seed_partial')
                temp.write_bytes(data)
                temp.replace(target)
                copied += 1
        records.append(dict(file=name, prefix_bytes=available, copied_chunks=copied, existing_chunks=skipped))
    report = dict(records=records, weights_not_yet_verified=True,
                  verification='Whole assembled publisher SHA256 remains mandatory in download_aima_ranges_v17.py')
    (ROOT / 'runs/advantage_validation_v17/setup/aima_range_resume.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
