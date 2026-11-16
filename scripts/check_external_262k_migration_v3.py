"""Public-only migration preflight; verifies model files and first images."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_external_262k_panel_v3 as panel


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def check():
    protocol = panel.verify_protocol()
    qwen = json.loads((ROOT / 'runs/pointer_mvp_same_cohort_v1/protocol.json').read_text(encoding='utf-8'))
    for name, item in qwen['model_shards'].items():
        path = ROOT / 'models/Qwen3-VL-8B-Instruct' / name
        assert path.stat().st_size == item['size_bytes'] and sha(path) == item['sha256'], name
        print('verified qwen', name, flush=True)
    tag = json.loads((ROOT / 'models/MiniCPM-Llama3-V-2_5/clickagent_download_verified_v17.json').read_text(encoding='utf-8'))
    assert tag['complete'] and tag['all_files_verified']
    for item in tag['files']:
        path = ROOT / 'models/MiniCPM-Llama3-V-2_5' / item['filename']
        assert path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], path
        print('verified tag', item['filename'], flush=True)
    for suite in panel.COUNTS:
        row = panel.public(suite)[0]
        path = panel.resolve_image_path(row['image'])
        assert sha(path) == row['image_sha256'], suite
        print('verified first image', suite, flush=True)
    print('PREFLIGHT_OK', protocol['run_id'], panel.sha(panel.PROTOCOL), flush=True)


if __name__ == '__main__':
    check()
