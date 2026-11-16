"""Build and verify a public-only, cross-platform 262k baseline transfer archive.

No label file is opened or included. The archive contains only the registered
V3 inference inputs, existing label-free Pointer predictions, two verified
model releases, reference code, and the images in the fixed public manifests.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import run_external_262k_panel_v3 as panel

OUT = ROOT / 'transfer'
ARCHIVE = OUT / 'ClickAgent_262k_trainfree_public_v3.tar.zst'
PART = OUT / 'ClickAgent_262k_trainfree_public_v3.tar.zst.part'
STATUS = OUT / 'ClickAgent_262k_trainfree_public_v3.status.json'
SIDECAR = OUT / 'ClickAgent_262k_trainfree_public_v3.sha256'
MANIFEST_MEMBER = 'ClickAgent/TRANSFER_MANIFEST.json'
CODE_DIRS = (
    'third_party/MVP/qwen3_vl',
    'third_party/TAG/MiniCPM_Llama3_V_2_5',
    'third_party/env_v17/mvp/qwen_vl_utils',
)
CODE_FILES = (
    'src/clickagent/__init__.py',
    'src/clickagent/advantage_runtimes.py',
    'scripts/run_external_262k_panel_v3.py',
    'scripts/queue_external_262k_panel_v3.py',
    'scripts/check_external_262k_migration_v3.py',
    'scripts/build_external_262k_transfer_archive_v3.py',
    'third_party/MVP/mvp_sspro_qwen3vl.py',
    'third_party/TAG/core_utils/tag.py',
    'third_party/TAG/eval_mm/screenspot/prompt_tag.py',
    'third_party/TAG/eval_mm/screenspot/eval_TAG.py',
)
MODEL_DIRS = (
    'models/Qwen3-VL-8B-Instruct',
    'models/MiniCPM-Llama3-V-2_5',
)
SMALL_FILES = (
    'README.md', 'pyproject.toml', 'requirements.txt',
    'paper/EXTERNAL_262K_BASELINES_MIGRATION.md',
    'runs/external_262k_panel_v3/protocol.json',
    'runs/paper_scale_ablation_v2/pointerbench/public.jsonl',
    'runs/mmbench_gui_new_source_v1/public.json',
    'runs/paper_scale_ablation_v2/eval/full/public.json',
    'runs/pointer_mvp_same_cohort_v1/protocol.json',
    'runs/pointer_mvp_same_cohort_v1/public_completion.json',
)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write_status(phase, **fields):
    record = dict(phase=phase, at_utc=datetime.now(timezone.utc).isoformat(), **fields)
    OUT.mkdir(parents=True, exist_ok=True)
    temp = STATUS.with_suffix('.json.tmp')
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    os.replace(temp, STATUS)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def relative(path):
    path = Path(path)
    resolved = path.resolve(strict=True)
    root = ROOT.resolve()
    if not resolved.is_relative_to(root) or not path.is_file() or path.is_symlink():
        raise ValueError('Unsafe transfer input: ' + str(path))
    return path.relative_to(ROOT).as_posix()


def allowed(path):
    parts = path.relative_to(ROOT).parts
    if any(part in ('__pycache__', '.git', '.cache', '.modelscope-cache',
                    '.pytest_cache') for part in parts):
        return False
    if path.suffix in ('.pyc', '.pyo'):
        return False
    if path.name.endswith(('.incomplete', '.part', '.tmp')):
        return False
    return True


def collect():
    protocol = panel.verify_protocol()
    expected = dict(protocol['source_sha256'])
    prior = read(ROOT / 'runs/pointer_mvp_same_cohort_v1/protocol.json')
    for name, item in prior['model_shards'].items():
        expected['models/Qwen3-VL-8B-Instruct/' + name] = item['sha256']
    tag = read(ROOT / 'models/MiniCPM-Llama3-V-2_5/clickagent_download_verified_v17.json')
    if not (tag['complete'] and tag['all_files_verified']):
        raise ValueError('TAG model receipt incomplete')
    for item in tag['files']:
        expected['models/MiniCPM-Llama3-V-2_5/' + item['filename']] = item['sha256']

    files = {}

    def add(path, category, digest=None):
        path = Path(path)
        rel = relative(path)
        if rel in files:
            if digest and files[rel]['expected_sha256'] not in (None, digest):
                raise ValueError('Conflicting image hash: ' + rel)
            if digest:
                files[rel]['expected_sha256'] = digest
            return
        files[rel] = dict(path=path, category=category,
                          size_bytes=path.stat().st_size,
                          expected_sha256=digest or expected.get(rel))

    for name in CODE_DIRS + MODEL_DIRS:
        directory = ROOT / name
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        category = 'model' if name.startswith('models/') else 'code'
        for base, dirs, names in os.walk(directory):
            dirs[:] = [d for d in dirs if allowed(Path(base) / d)]
            for filename in names:
                path = Path(base) / filename
                if allowed(path) and (category == 'model' or path.suffix == '.py'):
                    add(path, category)
    for name in CODE_FILES:
        add(ROOT / name, 'code')
    for name in SMALL_FILES:
        add(ROOT / name, 'protocol_and_docs')
    prior_predictions = ROOT / 'runs/pointer_mvp_same_cohort_v1/predictions'
    if not prior_predictions.is_dir():
        raise FileNotFoundError(prior_predictions)
    for path in prior_predictions.rglob('*'):
        if path.is_file() and allowed(path):
            add(path, 'prior_public_predictions')
    for suite in panel.COUNTS:
        for row in panel.public(suite):
            add(panel.resolve_image_path(row['image']), 'public_image', row['image_sha256'])

    for rel in expected:
        if rel not in files:
            raise ValueError('Frozen source omitted from archive: ' + rel)
    entries = sorted(files.items())
    total = sum(value['size_bytes'] for _, value in entries)
    return entries, expected, total


class HashingReader:
    def __init__(self, stream):
        self.stream = stream
        self.hash = hashlib.sha256()
        self.count = 0

    def read(self, n=-1):
        data = self.stream.read(n)
        self.hash.update(data)
        self.count += len(data)
        return data


def build(entries, total):
    if ARCHIVE.exists() or PART.exists():
        raise FileExistsError('Archive or partial archive already exists; inspect before rebuilding')
    compressor = shutil.which('zstd')
    if not compressor:
        raise RuntimeError('zstd executable required')
    OUT.mkdir(parents=True, exist_ok=True)
    write_status('packing', file_count=len(entries), total_input_bytes=total,
                 completed_files=0, completed_input_bytes=0)
    started = time.monotonic()
    proc = subprocess.Popen([compressor, '-T0', '-3', '-q', '-o', str(PART)],
                            stdin=subprocess.PIPE)
    manifest_files = []
    done = 0
    try:
        if proc.stdin is None:
            raise RuntimeError('zstd input pipe missing')
        with tarfile.open(fileobj=proc.stdin, mode='w|',
                          format=tarfile.PAX_FORMAT, bufsize=4 * 1024 * 1024) as tar:
            tar.copybufsize = 4 * 1024 * 1024
            for i, (rel, item) in enumerate(entries, 1):
                arcname = 'ClickAgent/' + rel
                info = tarfile.TarInfo(arcname)
                info.size = item['size_bytes']
                info.mode = 0o644
                info.mtime = 0
                with item['path'].open('rb') as source:
                    reader = HashingReader(source)
                    tar.addfile(info, reader)
                if reader.count != info.size:
                    raise IOError('Input changed while packing: ' + rel)
                digest = reader.hash.hexdigest()
                if item['expected_sha256'] and digest != item['expected_sha256']:
                    raise ValueError('Frozen SHA mismatch while packing: ' + rel)
                manifest_files.append(dict(path=rel, size_bytes=info.size,
                                           sha256=digest, category=item['category']))
                done += info.size
                if i % 100 == 0 or info.size >= 256 * 1024 * 1024 or i == len(entries):
                    rate = done / max(time.monotonic() - started, 1)
                    eta = (total - done) / rate if rate else None
                    write_status('packing', file_count=len(entries), total_input_bytes=total,
                                 completed_files=i, completed_input_bytes=done,
                                 elapsed_seconds=round(time.monotonic() - started, 1),
                                 estimated_remaining_seconds=round(eta, 1) if eta is not None else None)
            payload = dict(complete=True, format='tar.zst',
                           public_only=True, labels_included=False,
                           run_id='external_262k_panel_v3',
                           protocol_sha256=panel.sha(panel.PROTOCOL),
                           file_count=len(manifest_files), total_input_bytes=total,
                           files=manifest_files)
            data = (json.dumps(payload, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
            info = tarfile.TarInfo(MANIFEST_MEMBER)
            info.size = len(data)
            info.mode = 0o644
            info.mtime = 0
            tar.addfile(info, io.BytesIO(data))
    finally:
        if proc.stdin is not None and not proc.stdin.closed:
            proc.stdin.close()
    if proc.wait() != 0:
        raise RuntimeError('zstd compression failed; partial archive retained')
    write_status('packed_unverified', input_bytes=total, archive_bytes=PART.stat().st_size,
                 file_count=len(entries))


def verify_deep(entries):
    decompressor = shutil.which('zstd')
    if not decompressor:
        raise RuntimeError('zstd executable required')
    proc = subprocess.Popen([decompressor, '-d', '-c', str(PART)], stdout=subprocess.PIPE)
    observed = {}
    manifest = None
    try:
        if proc.stdout is None:
            raise RuntimeError('zstd output pipe missing')
        with tarfile.open(fileobj=proc.stdout, mode='r|',
                          format=tarfile.PAX_FORMAT, bufsize=4 * 1024 * 1024) as tar:
            for member in tar:
                if not member.isfile():
                    raise ValueError('Unexpected archive member type: ' + member.name)
                stream = tar.extractfile(member)
                if stream is None:
                    raise IOError('Unreadable archive member: ' + member.name)
                if member.name == MANIFEST_MEMBER:
                    manifest = json.load(stream)
                    continue
                if not member.name.startswith('ClickAgent/'):
                    raise ValueError('Unsafe archive path: ' + member.name)
                digest = hashlib.sha256()
                n = 0
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(block)
                    n += len(block)
                if n != member.size or member.name in observed:
                    raise ValueError('Archive entry size/duplicate mismatch: ' + member.name)
                observed[member.name[11:]] = dict(size_bytes=n, sha256=digest.hexdigest())
        # tar stops at end-of-archive blocks; drain zstd to its true EOF before
        # closing the pipe so decompression cannot fail with a broken pipe.
        for _ in iter(lambda: proc.stdout.read(8 * 1024 * 1024), b''):
            pass
    finally:
        if proc.stdout is not None and not proc.stdout.closed:
            proc.stdout.close()
    if proc.wait() != 0:
        raise RuntimeError('zstd decompression/integrity check failed')
    if manifest is None or manifest.get('file_count') != len(entries):
        raise ValueError('Transfer manifest absent/incomplete')
    if manifest['protocol_sha256'] != panel.sha(panel.PROTOCOL):
        raise ValueError('Protocol SHA in archive differs')
    expected = {row['path']: dict(size_bytes=row['size_bytes'], sha256=row['sha256'])
                for row in manifest['files']}
    if expected != observed or len(expected) != len(entries):
        raise ValueError('Deep archive file SHA comparison failed')
    write_status('archive_verified', file_count=len(entries), archive_bytes=PART.stat().st_size)
    return manifest


def sha_file(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    entries, _, total = collect()
    counts = {}
    for _, item in entries:
        category = item['category']
        old = counts.get(category, (0, 0))
        counts[category] = (old[0] + 1, old[1] + item['size_bytes'])
    print(json.dumps({'files': len(entries), 'input_bytes': total,
                      'categories': {k: {'files': n, 'bytes': b}
                                     for k, (n, b) in counts.items()}},
                     ensure_ascii=False), flush=True)
    if args.plan_only:
        return
    build(entries, total)
    verify_deep(entries)
    os.replace(PART, ARCHIVE)
    digest = sha_file(ARCHIVE)
    SIDECAR.write_text(f'{digest}  {ARCHIVE.name}\n', encoding='ascii')
    write_status('complete', archive=str(ARCHIVE), archive_bytes=ARCHIVE.stat().st_size,
                 archive_sha256=digest, sha256_sidecar=str(SIDECAR),
                 file_count=len(entries), total_input_bytes=total,
                 labels_included=False)


if __name__ == '__main__':
    main()
