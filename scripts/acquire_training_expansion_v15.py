"""Pinned public training-data acquisition in a new, resumable directory."""
from __future__ import annotations

import argparse
import concurrent.futures
from collections import Counter, defaultdict
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tarfile
import time
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from clickagent.public_data import download_file, safe_path
from clickagent.training_expansion_data import local_image_path

OUT = ROOT / 'data/expansion_v15'
RUN = ROOT / 'runs/dataset_expansion_v15'
REPOS = {
    'groundcua': ('ServiceNow/GroundCUA', 'ServiceNow__GroundCUA.json'),
    'showui_web': ('zonghanHZH/ShowUI-web-8k', 'zonghanHZH__ShowUI-web-8k.json'),
    'amex': ('zonghanHZH/AMEX-8k', 'zonghanHZH__AMEX-8k.json'),
}


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    tmp.replace(path)


@lru_cache(maxsize=None)
def identity(name):
    repo, filename = REPOS[name]
    meta = read(OUT / 'discovery' / filename)
    assert meta['id'] == repo
    files = {f['rfilename']: f for f in meta['siblings']}
    if name == 'groundcua':
        # HF's /api/datasets response stops at100000 files. Complete the three
        # truncated image directories from pinned, individually verified trees.
        for path in (OUT / 'discovery').glob('GroundCUA_images_*.json'):
            for entry in read(path):
                if entry['type'] == 'file':
                    files[entry['path']] = dict(entry, rfilename=entry['path'], blobId=entry['oid'])
    return repo, meta['sha'], files


def transfer(name, item):
    repo, revision, _ = identity(name)
    relative = item.get('rfilename', item.get('path'))
    lfs = item.get('lfs') or {}
    published = lfs.get('sha256', lfs.get('oid'))
    git_sha = item.get('blobId', item.get('oid')) if not published else None
    encoded = quote(relative, safe='/')
    dest = safe_path(OUT / 'raw' / name, relative)
    return download_file([
        f'https://huggingface.co/datasets/{repo}/resolve/{revision}/{encoded}?download=true',
        f'https://hf-mirror.com/datasets/{repo}/resolve/{revision}/{encoded}?download=true',
    ], dest, size=item['size'], sha256=published, git_sha1=git_sha, retries=2)


def bootstrap():
    jobs = []
    for name in REPOS:
        _, _, files = identity(name)
        jobs.append((name, files['README.md']))
        if name == 'groundcua':
            root = read(OUT / 'discovery/GroundCUA_root_tree.json')
            jobs.append((name, next(f for f in root if f.get('path') == 'instruction_tuning.tar.gz')))
        else:
            jobs.append((name, files['metadata/hf_train.json']))
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {pool.submit(transfer, name, item): (name, item) for name, item in jobs}
        for future in concurrent.futures.as_completed(futures):
            name, item = futures[future]
            result = future.result()
            print(json.dumps(dict(stage='metadata', dataset=name,
                  file=item.get('rfilename', item.get('path')), bytes=result['size'],
                  sha256=result['sha256'], seconds=time.monotonic()-started)), flush=True)
    write(RUN / 'bootstrap_complete.json', dict(complete=True, repos={name: identity(name)[:2] for name in REPOS}))


def stable(text):
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def source_group(name, image):
    relative = image.removeprefix('images/')
    if name == 'groundcua':
        return relative.split('/')[0]
    if name == 'showui_web':
        root = relative.split('/')[0].casefold()
        return re.sub(r'^\d+_(?:chrome|edge|firefox|safari)_', '', root)
    return re.sub(r'-\d+\.[^.]+$', '', Path(relative).name)


def balanced_select(items, limit):
    buckets = defaultdict(list)
    for item in items:
        buckets[item['source_group']].append(item)
    groups = sorted(buckets, key=lambda g: stable('20261103:group:' + g))
    for group in groups:
        buckets[group].sort(key=lambda r: stable('20261103:image:' + r['relative_image']))
    selected = []
    level = 0
    while len(selected) < limit:
        added = False
        for group in groups:
            if level < len(buckets[group]):
                selected.append(buckets[group][level]); added = True
                if len(selected) == limit:
                    break
        if not added:
            break
        level += 1
    return selected


def select():
    assert read(RUN / 'bootstrap_complete.json')['complete']
    path = RUN / 'selection.json'
    assert not path.exists(), 'Selection already fixed; use the existing selection'
    selected = []
    stats = {}
    for name in ('showui_web', 'amex'):
        _, _, files = identity(name)
        raw = read(OUT / 'raw' / name / 'metadata/hf_train.json')
        cases = []
        for index, row in enumerate(raw):
            relative = 'images/' + row['img_url'].removeprefix('images/')
            if relative not in files:
                continue
            cases.append(dict(dataset=name, relative_image=relative, item=files[relative],
                 source_group=source_group(name, relative), expected_size=row['img_size'],
                 annotations=[dict(instruction=e.get('instruction'), bbox=e.get('bbox'),
                   source_record=index, source_element=j, annotation_file='metadata/hf_train.json')
                   for j, e in enumerate(row['element'])]))
        chosen = balanced_select(cases, 1000)
        selected.extend(chosen)
        stats[name] = dict(available_images=len(cases), metadata_rows=len(raw), selected_images=len(chosen),
                          source_groups=len({r['source_group'] for r in chosen}))
    _, _, files = identity('groundcua')
    archive = OUT / 'raw/groundcua/instruction_tuning.tar.gz'
    with tarfile.open(archive, 'r|gz') as bundle:
        chosen_by_path = None
        selected_ids = None
        valid_members = []
        for member in bundle:
            if not member.isfile() or not member.name.endswith('.json'):
                continue
            with bundle.extractfile(member) as stream:
                rows = json.load(stream)
            assert isinstance(rows, list), member.name
            valid_members.append(dict(name=member.name, rows=len(rows)))
            if chosen_by_path is None:
                candidates = {}
                for row in rows:
                    if not isinstance(row, dict) or not isinstance(row.get('instruction'), str):
                        continue
                    relative = f"images/{row.get('platform')}/{row.get('id')}.png"
                    if relative in files and isinstance(row.get('bbox'), list):
                        candidates[relative] = dict(dataset='groundcua', relative_image=relative,
                            item=files[relative], source_group=row['platform'], expected_size=None, annotations=[])
                chosen = balanced_select(list(candidates.values()), 2000)
                assert len(chosen) == 2000
                chosen_by_path = {r['relative_image']: r for r in chosen}
                selected_ids = {Path(p).stem for p in chosen_by_path}
                stats['groundcua'] = dict(available_first_instruction_images=len(candidates),
                    selected_images=len(chosen), source_groups=len({r['source_group'] for r in chosen}))
            for index, row in enumerate(rows):
                if not isinstance(row, dict) or row.get('id') not in selected_ids:
                    continue
                relative = f"images/{row.get('platform')}/{row.get('id')}.png"
                if relative not in chosen_by_path:
                    continue
                if isinstance(row.get('instruction'), str) and isinstance(row.get('bbox'), list):
                    chosen_by_path[relative]['annotations'].append(dict(instruction=row['instruction'],
                        bbox=row['bbox'], source_record=index, annotation_file=member.name,
                        category=row.get('category'), text=row.get('text')))
            print(json.dumps(dict(stage='groundcua_instruction_scan', member=member.name, rows=len(rows))), flush=True)
            del rows
        assert all(r['annotations'] for r in chosen_by_path.values())
        selected.extend(chosen_by_path.values())
        stats['groundcua']['instruction_archive_members'] = valid_members
    assert len(selected) == 4000
    for row in selected:
        row['key'] = stable(row['dataset'] + ':' + row['relative_image'])[:24]
    discovery = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in (OUT / 'discovery').glob('*.json')}
    write(path, dict(complete=True, seed=20261103, selection_rule='Group-balanced round robin with fixed hash order, no model predictions.',
        scope='Initial4000-image training import, not a complete download of the three source datasets.',
        sources={name: dict(repo=identity(name)[0], revision=identity(name)[1]) for name in REPOS},
        stats=stats, cases=selected, discovery_sha256=discovery,
        groundcua_truncated_inventory_completed=True, source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    write(RUN / 'work_state.json', dict(complete=False, phase='image_selection_fixed', planned_images=4000))
    print(json.dumps(dict(stage='selection_fixed', stats=stats), ensure_ascii=False), flush=True)


def transfer_image(row):
    name = row['dataset']; item = row['item']
    repo, revision, _ = identity(name)
    relative = row['relative_image']
    destination = local_image_path(OUT / 'raw' / name, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    lfs = item.get('lfs') or {}
    expected = lfs.get('sha256', lfs.get('oid'))
    git_sha = item.get('blobId', item.get('oid')) if not expected else None
    from clickagent.public_data import file_hash
    def valid(path):
        return path.is_file() and path.stat().st_size == item['size'] and (
            file_hash(path) == expected if expected else file_hash(path, 'sha1', git_blob=True) == git_sha)
    if valid(destination):
        return dict(cached=True, bytes=item['size'])
    part = destination.with_name(destination.name + '.part')
    encoded = quote(relative, safe='/')
    errors = []
    started = time.monotonic()
    for attempt in range(4):
        host = 'huggingface.co' if attempt % 2 == 0 else 'hf-mirror.com'
        url = f'https://{host}/datasets/{repo}/resolve/{revision}/{encoded}?download=true'
        command = ['curl.exe', '--fail', '--location', '--silent', '--show-error',
                   '--connect-timeout', '12', '--max-time', '90', '--speed-time', '25', '--speed-limit', '1024',
                   '--output', str(part)]
        if part.is_file() and part.stat().st_size:
            if valid(part):
                part.replace(destination); break
            command += ['--continue-at', '-']
        result = subprocess.run(command + [url], capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=100)
        if valid(part):
            part.replace(destination); break
        errors.append(dict(attempt=attempt+1, host=host, returncode=result.returncode, error=result.stderr[-250:]))
        if result.returncode == 33 or (part.is_file() and part.stat().st_size >= item['size']):
            part.replace(part.with_name(part.name + f'.bad-{time.time_ns()}'))
        time.sleep(min(attempt+1, 3))
    assert valid(destination), json.dumps(dict(dataset=name, file=relative, errors=errors))
    receipt = dict(complete=True, repo=repo, revision=revision, relative_path=relative,
                   size=item['size'], sha256=file_hash(destination), published_sha256=expected,
                   published_git_sha1=git_sha, seconds=time.monotonic()-started, prior_errors=errors)
    write(destination.with_name(destination.name + '.download.json'), receipt)
    return dict(cached=False, bytes=item['size'])


def download_images(workers):
    selection = read(RUN / 'selection.json')
    jobs = selection['cases']
    done = Counter(); errors = []; started = time.monotonic(); transferred = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        pending = {pool.submit(transfer_image, row): row for row in jobs}
        for i, future in enumerate(concurrent.futures.as_completed(pending), 1):
            row = pending[future]
            try:
                result = future.result(); done[row['dataset']] += 1; transferred += result['bytes']
            except Exception as error:
                errors.append(dict(dataset=row['dataset'], file=row['relative_image'], error=str(error)))
                print(json.dumps(dict(stage='download_error', dataset=row['dataset'], file=row['relative_image'])), flush=True)
            if i % 25 == 0 or i == len(jobs):
                print(json.dumps(dict(stage='image_download', finished=i, n=len(jobs), verified=dict(done),
                      errors=len(errors), verified_mib=round(transferred/2**20,1), seconds=round(time.monotonic()-started,1))), flush=True)
    write(RUN / 'download_complete.json', dict(complete=not errors, verified=dict(done), n=sum(done.values()),
         errors=errors, selection_sha256=hashlib.sha256((RUN/'selection.json').read_bytes()).hexdigest()))
    assert not errors, f'{len(errors)} downloads pending; resume the same command'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--stage', choices=['bootstrap', 'select', 'download'], required=True)
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()
    if args.stage == 'bootstrap':
        bootstrap()
    elif args.stage == 'select':
        select()
    elif args.stage == 'download':
        download_images(args.workers)


if __name__ == '__main__':
    main()
