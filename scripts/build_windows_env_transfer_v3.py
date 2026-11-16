"""Package and verify the existing Windows Python environment for V3 inference.

This is a snapshot of the local Conda prefix, not a conda-pack relocation.
Extract it alongside the public V3 transfer archive, then smoke-test the
interpreter and model imports on the target Windows x64 host before inference.
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
OUT = ROOT / 'transfer'
STEM = 'ClickAgent_windows_py311_runtime_v3'
ARCHIVE = OUT / f'{STEM}.tar.zst'
PART = OUT / f'{STEM}.tar.zst.part'
STATUS = OUT / f'{STEM}.status.json'
SIDECAR = OUT / f'{STEM}.sha256'
MANIFEST_MEMBER = 'ClickAgent/ENV_TRANSFER_MANIFEST.json'
SOURCE_DIRS = (
    '.venv',
    'third_party/env_v17/tag',
    'third_party/env_v17/common',
)
EXTRA_FILES = (
    'paper/EXTERNAL_262K_BASELINES_MIGRATION.md',
    'scripts/build_windows_env_transfer_v3.py',
)
EXCLUDED_DIRS = {'__pycache__', '.pytest_cache', '.cache', '.git'}
EXCLUDED_FILES = {'__editable__.clickagent_experiments-0.1.0.pth'}
EXCLUDED_EDITABLE_METADATA = 'clickagent_experiments-0.1.0.dist-info'
STREAM_BUFFER_BYTES = 4 * 1024 * 1024


def write_status(phase: str, **fields: object) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    record = dict(phase=phase, at_utc=datetime.now(timezone.utc).isoformat(), **fields)
    temp = STATUS.with_suffix('.json.tmp')
    temp.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n',
                    encoding='utf-8')
    os.replace(temp, STATUS)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def relative(path: Path) -> str:
    resolved = path.resolve(strict=True)
    if (not resolved.is_relative_to(ROOT.resolve()) or
            not path.is_file() or path.is_symlink()):
        raise ValueError('Unsafe environment input: ' + str(path))
    return path.relative_to(ROOT).as_posix()


def allowed(path: Path) -> bool:
    parts = path.relative_to(ROOT).parts
    if any(part in EXCLUDED_DIRS for part in parts):
        return False
    if EXCLUDED_EDITABLE_METADATA in parts:
        return False
    if path.name in EXCLUDED_FILES or path.suffix in ('.pyc', '.pyo'):
        return False
    if path.name.endswith(('.incomplete', '.part', '.tmp')):
        return False
    return True


def collect() -> tuple[list[tuple[str, dict]], int]:
    if sys.platform != 'win32':
        raise RuntimeError('The source environment is Windows only')
    env = ROOT / '.venv'
    if not (env / 'conda-meta/history').is_file():
        raise RuntimeError('Expected the existing Conda prefix at .venv')
    if Path(sys.prefix).resolve() != env.resolve():
        raise RuntimeError('Run this builder using .venv\\python.exe')
    if sys.version_info[:2] != (3, 11) or sys.maxsize <= 2**32:
        raise RuntimeError('Expected the existing 64-bit CPython 3.11 environment')

    files: dict[str, dict] = {}
    for name in SOURCE_DIRS:
        directory = ROOT / name
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        category = 'python_conda_prefix' if name == '.venv' else 'v17_overlay'
        for base, dirs, names in os.walk(directory, followlinks=False):
            dirs[:] = [d for d in dirs if allowed(Path(base) / d) and
                       not (Path(base) / d).is_symlink()]
            for filename in names:
                path = Path(base) / filename
                if allowed(path):
                    rel = relative(path)
                    if rel in files:
                        raise ValueError('Duplicate environment input: ' + rel)
                    files[rel] = dict(path=path, category=category,
                                      size_bytes=path.stat().st_size)
    for name in EXTRA_FILES:
        path = ROOT / name
        rel = relative(path)
        if rel in files:
            raise ValueError('Duplicate environment input: ' + rel)
        files[rel] = dict(path=path, category='documentation',
                          size_bytes=path.stat().st_size)
    if not any(rel.endswith('/python.exe') for rel in files):
        raise RuntimeError('Python executable missing from environment snapshot')
    entries = sorted(files.items())
    return entries, sum(item['size_bytes'] for _, item in entries)


class HashingReader:
    def __init__(self, source):
        self.source = source
        self.hash = hashlib.sha256()
        self.count = 0

    def read(self, n=-1):
        data = self.source.read(n)
        self.hash.update(data)
        self.count += len(data)
        return data


def pack(entries: list[tuple[str, dict]], total: int) -> None:
    if ARCHIVE.exists() or PART.exists():
        raise FileExistsError('Final or partial environment archive already exists')
    compressor = shutil.which('zstd')
    if not compressor:
        raise RuntimeError('zstd executable required')
    OUT.mkdir(parents=True, exist_ok=True)
    write_status('packing', file_count=len(entries), total_input_bytes=total,
                 completed_files=0, completed_input_bytes=0)
    started = time.monotonic()
    process = subprocess.Popen([compressor, '-T0', '-3', '-q', '-o', str(PART)],
                               stdin=subprocess.PIPE)
    manifest_files = []
    done = 0
    try:
        if process.stdin is None:
            raise RuntimeError('zstd input pipe missing')
        with tarfile.open(fileobj=process.stdin, mode='w|',
                          format=tarfile.PAX_FORMAT,
                          bufsize=STREAM_BUFFER_BYTES) as tar:
            tar.copybufsize = STREAM_BUFFER_BYTES
            for i, (rel, item) in enumerate(entries, 1):
                info = tarfile.TarInfo('ClickAgent/' + rel)
                info.size = item['size_bytes']
                info.mode = 0o644
                info.mtime = 0
                with item['path'].open('rb') as source:
                    reader = HashingReader(source)
                    tar.addfile(info, reader)
                if reader.count != info.size:
                    raise IOError('Environment input changed while packing: ' + rel)
                manifest_files.append(dict(path=rel, size_bytes=info.size,
                                           sha256=reader.hash.hexdigest(),
                                           category=item['category']))
                done += info.size
                if i % 500 == 0 or info.size >= 256 * 1024 * 1024 or i == len(entries):
                    elapsed = max(time.monotonic() - started, 1)
                    rate = done / elapsed
                    write_status('packing', file_count=len(entries),
                                 total_input_bytes=total,
                                 completed_files=i, completed_input_bytes=done,
                                 elapsed_seconds=round(elapsed, 1),
                                 estimated_remaining_seconds=round((total - done) / rate, 1)
                                 if rate else None)
            payload = dict(complete=True, format='tar.zst',
                           source_os='Windows', python_abi='cp311-win_amd64',
                           environment_kind='raw Conda prefix snapshot',
                           conda_pack_used=False, relocation_validated=False,
                           extract_into='ClickAgent/',
                           required_companion='ClickAgent_262k_trainfree_public_v3.tar.zst',
                           excluded_editable_pointer=True,
                           file_count=len(manifest_files),
                           total_input_bytes=total, files=manifest_files)
            data = (json.dumps(payload, ensure_ascii=False, indent=2) + '\n').encode('utf-8')
            info = tarfile.TarInfo(MANIFEST_MEMBER)
            info.size = len(data)
            info.mode = 0o644
            info.mtime = 0
            tar.addfile(info, io.BytesIO(data))
    finally:
        if process.stdin is not None and not process.stdin.closed:
            process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('zstd compression failed; partial archive retained')
    write_status('packed_unverified', file_count=len(entries), input_bytes=total,
                 archive_bytes=PART.stat().st_size)


def verify_deep(entries: list[tuple[str, dict]]) -> None:
    decompressor = shutil.which('zstd')
    if not decompressor:
        raise RuntimeError('zstd executable required')
    process = subprocess.Popen([decompressor, '-d', '-c', str(PART)],
                               stdout=subprocess.PIPE)
    observed = {}
    manifest = None
    try:
        if process.stdout is None:
            raise RuntimeError('zstd output pipe missing')
        with tarfile.open(fileobj=process.stdout, mode='r|',
                          format=tarfile.PAX_FORMAT,
                          bufsize=STREAM_BUFFER_BYTES) as tar:
            for member in tar:
                if not member.isfile():
                    raise ValueError('Unexpected member type: ' + member.name)
                stream = tar.extractfile(member)
                if stream is None:
                    raise IOError('Unreadable member: ' + member.name)
                if member.name == MANIFEST_MEMBER:
                    if manifest is not None:
                        raise ValueError('Duplicate environment manifest')
                    manifest = json.load(stream)
                    continue
                if not member.name.startswith('ClickAgent/'):
                    raise ValueError('Unsafe archive path: ' + member.name)
                rel = member.name[len('ClickAgent/'):]
                if rel.startswith('/') or '..' in Path(rel).parts or rel in observed:
                    raise ValueError('Unsafe or duplicate member: ' + member.name)
                digest = hashlib.sha256()
                size = 0
                for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
                    digest.update(block)
                    size += len(block)
                if size != member.size:
                    raise ValueError('Archive entry size mismatch: ' + rel)
                observed[rel] = dict(size_bytes=size, sha256=digest.hexdigest())
        for _ in iter(lambda: process.stdout.read(8 * 1024 * 1024), b''):
            pass
    finally:
        if process.stdout is not None and not process.stdout.closed:
            process.stdout.close()
    if process.wait() != 0:
        raise RuntimeError('zstd decompression or integrity check failed')
    if manifest is None or manifest.get('file_count') != len(entries):
        raise ValueError('Environment manifest missing or incomplete')
    expected = {row['path']: dict(size_bytes=row['size_bytes'], sha256=row['sha256'])
                for row in manifest['files']}
    if len(expected) != len(entries) or expected != observed:
        raise ValueError('Environment archive deep SHA verification failed')
    write_status('archive_verified', file_count=len(entries),
                 archive_bytes=PART.stat().st_size)


def sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    entries, total = collect()
    categories = {}
    for _, item in entries:
        category = item['category']
        n, size = categories.get(category, (0, 0))
        categories[category] = (n + 1, size + item['size_bytes'])
    print(json.dumps(dict(file_count=len(entries), input_bytes=total,
                          categories={k: dict(files=n, bytes=size)
                                      for k, (n, size) in categories.items()})),
          flush=True)
    if args.plan_only:
        return
    pack(entries, total)
    verify_deep(entries)
    os.replace(PART, ARCHIVE)
    digest = sha_file(ARCHIVE)
    SIDECAR.write_text(f'{digest}  {ARCHIVE.name}\n', encoding='ascii')
    write_status('complete', archive=str(ARCHIVE), archive_bytes=ARCHIVE.stat().st_size,
                 archive_sha256=digest, sha256_sidecar=str(SIDECAR),
                 file_count=len(entries), total_input_bytes=total,
                 relocation_validated=False)


if __name__ == '__main__':
    main()
