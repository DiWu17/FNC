"""Normalize, exclude historical screenshots, and prepare V15 training manifests."""
from __future__ import annotations

from collections import Counter, defaultdict
import concurrent.futures
import hashlib
import html
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from clickagent.training_expansion_data import digest, screenshot_identity, merge_annotations, NearIndex, select_targets, reserve_groups, local_image_path

DATA = ROOT / 'data/expansion_v15'
RUN = ROOT / 'runs/dataset_expansion_v15'
LABELS = dict(groundcua='GroundCUA', showui_web='ShowUI-web', amex='AMEX')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, data):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name+'.tmp')
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temporary.replace(path)


def write_rows(path, rows):
    with Path(path).open('w', encoding='utf-8') as output:
        for row in rows:
            output.write(json.dumps(row, ensure_ascii=False)+'\n')


def legacy_files():
    base = ROOT / 'runs/multiregion_v13'
    seal = read(base / 'completion.json'); found = {}
    for root, field in ((base, 'artifact_sha256'), (ROOT, 'docs_sha256')):
        for name, expected in seal[field].items():
            path = root / name
            assert digest(path) == expected, path
            found[str(path)] = expected
    return found


def identify(row):
    path = local_image_path(DATA / 'raw' / row['dataset'], row['relative_image'])
    cache = DATA / 'image_index' / (row['key']+'.json')
    if cache.exists():
        try:
            saved = read(cache)
        except json.JSONDecodeError:
            saved = None
        if saved is not None:
            assert saved['image_sha256'] == digest(path)
            return row, saved
    info = screenshot_identity(path)
    expected = (row['item'].get('lfs') or {}).get('sha256', (row['item'].get('lfs') or {}).get('oid'))
    assert expected is None or expected == info['image_sha256'], path
    info.update(dataset=row['dataset'], source_group=row['source_group'], relative_image=row['relative_image'], image=str(path.resolve()))
    write(cache, info)
    return row, info


def main():
    started = time.monotonic()
    assert read(RUN / 'download_complete.json')['complete']
    selection = read(RUN / 'selection.json')
    initial_seals = legacy_files()
    public_path = ROOT / 'data/conditioned_v3/public_image_index.json'
    public = read(public_path); assert public['complete']
    base_path = ROOT / 'runs/evidence_retention_v14/base/index.json'
    historical = read(base_path)['cases']
    cases_by_id = {}
    for split in ('train', 'val'):
        for line in (ROOT / f'runs/multiregion_v13/data/{split}.jsonl').read_text(encoding='utf-8').splitlines():
            row = json.loads(line); cases_by_id[row['sample_id']] = row
    exact = {}; near = NearIndex()
    for image in public['images']:
        ref = dict(kind='public_benchmark', image=image['image'])
        exact[image['pixel_sha256']] = ref
        near.add(int(image['dhash256'], 16), ref)
    for item in historical:
        old = cases_by_id[item['sample_id']]
        ref = dict(kind='historical_training_or_validation', sample_id=item['sample_id'], split=item['split'])
        exact.setdefault(item['group'], ref)
        near.add(int(old['dhash256'], 16), ref)
    stats = defaultdict(Counter); identified = []; errors = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(identify, row): row for row in selection['cases']}
        for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
            row = futures[future]
            try:
                identified.append(future.result())
            except Exception as error:
                errors.append(dict(dataset=row['dataset'], image=row['relative_image'], error=str(error)))
                stats[row['dataset']]['invalid_image'] += 1
            if i % 100 == 0:
                print(json.dumps(dict(stage='decode_hash', i=i, n=4000, seconds=round(time.monotonic()-started,1))), flush=True)
    accepted = []; rejected = []; new_near = NearIndex(); new_pixels = {}
    order = dict(groundcua=0, showui_web=1, amex=2)
    identified.sort(key=lambda pair: (order[pair[0]['dataset']], hashlib.sha256(pair[0]['key'].encode()).hexdigest()))
    for raw, image in identified:
        source = raw['dataset']; stat = stats[source]; stat['downloaded_images'] += 1
        reason = None; match = None
        if raw['expected_size'] and list(raw['expected_size']) != image['image_size']:
            reason = 'metadata_image_size_mismatch'
        if image['pixel_sha256'] in exact:
            match = exact[image['pixel_sha256']]; reason = match['kind']+'_exact'
        elif (match := near.find(int(image['dhash256'], 16))) is not None:
            reason = match['kind']+'_near'
        if reason is None and image['pixel_sha256'] in new_pixels:
            reason = 'new_import_exact'; match = new_pixels[image['pixel_sha256']]
        elif reason is None and (match := new_near.find(int(image['dhash256'], 16))) is not None:
            reason = 'new_import_near'
        if reason:
            stat[reason] += 1
            rejected.append(dict(dataset=source, image=raw['relative_image'], reason=reason, reference=match)); continue
        targets, target_stats = merge_annotations(raw['annotations'], image['image_size'])
        stat.update(target_stats)
        if not targets:
            stat['images_without_valid_target'] += 1
            rejected.append(dict(dataset=source, image=raw['relative_image'], reason='no_valid_target')); continue
        image['targets'] = targets
        image['key'] = raw['key']
        accepted.append(image)
        ref = dict(dataset=source, image=raw['relative_image'])
        new_pixels[image['pixel_sha256']] = ref
        new_near.add(int(image['dhash256'], 16), ref)
        stat['accepted_unique_screenshots'] += 1
    reserves = reserve_groups(accepted)
    all_rows = []; selected_rows = []; image_index = []
    for image in accepted:
        source = image['dataset']; split = 'val' if (source, image['source_group']) in reserves else 'train'
        rows = []
        for target in image['targets']:
            identifier = hashlib.sha256((image['pixel_sha256']+'\0'+target['instruction']).encode()).hexdigest()[:24]
            rows.append(dict(sample_id=f'expansion15:{source}:{identifier}', dataset=LABELS[source],
                instruction=target['instruction'], image=image['image'], image_size=image['image_size'],
                image_sha256=image['image_sha256'], pixel_sha256=image['pixel_sha256'], dhash256=image['dhash256'],
                group_id=image['pixel_sha256'], source_group=image['source_group'],
                source_group_type=('application' if source=='groundcua' else 'website_collection' if source=='showui_web' else 'episode_proxy'),
                experiment_split=split, split=split, split_role='new_training' if split=='train' else 'new_development_reserve',
                source_split='published_training_corpus', training_repo=selection['sources'][source]['repo'],
                training_repo_revision=selection['sources'][source]['revision'], source_image=image['relative_image'],
                expected_actionable=True, task_kind='grounding_present',
                target_geometry=dict(type='bbox', boxes=target['boxes']), source_annotations=target['source_annotations']))
        selected = select_targets(rows, 4)
        assert selected
        all_rows.extend(rows); selected_rows.extend(selected)
        stats[source][split+'_images'] += 1
        stats[source][split+'_all_queries'] += len(rows)
        stats[source][split+'_selected_queries'] += len(selected)
        image_index.append({k: v for k, v in image.items() if k != 'targets'} | dict(experiment_split=split, all_queries=len(rows), selected_queries=len(selected)))
    manifest = DATA / 'manifests'; manifest.mkdir(parents=True, exist_ok=True)
    files = {
        'pool_all.jsonl': all_rows,
        'train_all.jsonl': [r for r in all_rows if r['experiment_split']=='train'],
        'train.jsonl': [r for r in selected_rows if r['experiment_split']=='train'],
        'validation_reserve_all.jsonl': [r for r in all_rows if r['experiment_split']=='val'],
        'validation_reserve.jsonl': [r for r in selected_rows if r['experiment_split']=='val'],
    }
    legacy_train = [json.loads(l) for l in (ROOT / 'runs/multiregion_v13/data/train.jsonl').read_text(encoding='utf-8').splitlines()]
    files['combined_train.jsonl'] = legacy_train + files['train.jsonl']
    assert len({r['sample_id'] for r in files['combined_train.jsonl']}) == len(files['combined_train.jsonl'])
    for filename, rows in files.items():
        write_rows(manifest / filename, rows)
    train = files['train.jsonl']; val = files['validation_reserve.jsonl']
    assert not {r['pixel_sha256'] for r in train} & {r['pixel_sha256'] for r in val}
    assert not {(r['dataset'],r['source_group']) for r in train} & {(r['dataset'],r['source_group']) for r in val}
    assert all(0 <= x <= 1 for row in all_rows for box in row['target_geometry']['boxes'] for x in box)
    for source in LABELS:
        stats[source]['train_source_groups'] = len({r['source_group'] for r in train if r['dataset']==LABELS[source]})
        stats[source]['reserve_source_groups'] = len({r['source_group'] for r in val if r['dataset']==LABELS[source]})
        assert stats[source]['train_images'] > 0
    assert legacy_files() == initial_seals
    write(RUN / 'image_index.json', dict(complete=True, images=image_index))
    write(RUN / 'excluded_images.json', rejected)
    write(RUN / 'image_errors.json', errors)
    result = dict(complete=not errors, initial_batch_images=4000, stats={k:dict(v) for k,v in stats.items()},
        new_unique_images=len(accepted), new_training_images=len({r['pixel_sha256'] for r in train}),
        new_reserved_images=len({r['pixel_sha256'] for r in val}), new_training_queries=len(train),
        new_training_all_queries=len(files['train_all.jsonl']), reserved_queries=len(val),
        combined_training_queries=len(files['combined_train.jsonl']), combined_training_images=568+len({r['pixel_sha256'] for r in train}),
        manifests={name:dict(path=str((manifest/name).resolve()), rows=len(rows),sha256=digest(manifest/name)) for name,rows in files.items()},
        selection_sha256=digest(RUN/'selection.json'), public_reference_sha256=digest(public_path), legacy_base_index_sha256=digest(base_path),
        excluded_public_reference_images=len(public['images']), legacy_completion_hashes=initial_seals,
        curation='Reject exact RGB and dHash<=4 matches to all6745 indexed public benchmark screenshots, historical train/val, and earlier retained new images. Merge identical source queries into multiple-valid-box supervision. Select at most4 non-overlapping target queries per retained screenshot; full labels also preserved.',
        reservation='10% of source groups held out before model training. GroundCUA: software; ShowUI: website collection; AMEX: episode proxy. Separation applies within newly imported data; globally unseen applications relative to all legacy/backbone data is not established.',
        source_licenses=dict(groundcua=dict(published='MIT'),showui_web=dict(curated='Apache-2.0',original='No explicit license in original ShowUI-web card checked; original provenance retained'),amex=dict(curated='Apache-2.0',original='CC-BY-4.0')),
        warnings=['This is an initial training import, not complete copies of the source datasets.','No model fitting, Qwen features or performance evaluation has run on these data.','Source instruction quality has not been exhaustively manually verified; dataset QA sample is descriptive.','AMEX may share source lineage with Aria-UI mobile; source names alone do not guarantee independent content.','New reserve is a development reserve, not an untouched official benchmark or a proven global application holdout.'],
        source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [Path(__file__),ROOT/'src/clickagent/training_expansion_data.py']},
        elapsed_seconds=time.monotonic()-started)
    write(RUN / 'results.json', result)
    write(RUN / 'work_state.json', dict(complete=not errors,phase='normalized_training_import_complete',model_training_started=False,qwen_feature_extraction_started=False))
    render_report(result)
    print(json.dumps({k:v for k,v in result.items() if k not in ['legacy_completion_hashes','manifests','source_sha256']},ensure_ascii=False),flush=True)


def render_report(result):
    parts=['<h1>ClickAgent：训练数据扩充首批接入</h1><p>2026-11-03 · 三套训练来源 · 原V13/V14实验原件保持只读</p>',
           '<p><strong>新增训练截图 '+str(result['new_training_images'])+' 张，精选定位任务 '+str(result['new_training_queries'])+' 条；合并原873条后为 '+str(result['combined_training_queries'])+' 条。</strong></p>',
           '<p>原训练568张不同截图，合并后为 '+str(result['combined_training_images'])+' 张。完整标签另存。尚未提取Qwen内部特征、训练模型或进行效果评测。</p>',
           '<table><tr><th>数据来源</th><th>计划下载图</th><th>排重后图</th><th>训练图</th><th>保留图</th><th>精选训练任务</th><th>训练组 / 保留组</th></tr>']
    for source in ('groundcua','showui_web','amex'):
        s=result['stats'][source]
        values=[LABELS[source],2000 if source=='groundcua' else 1000,s['accepted_unique_screenshots'],s['train_images'],s['val_images'],s['train_selected_queries'],f"{s['train_source_groups']} / {s['reserve_source_groups']}"]
        parts.append('<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in values)+'</tr>')
    parts.append('</table><p>训练组在新增数据内部按软件/网站/任务代理隔离。AMEX没有经验证的应用ID，不能将任务代理隔离称为应用未见。</p>')
    parts.append('<h2>可直接使用的清单</h2><ul>')
    for name,info in result['manifests'].items():
        parts.append('<li><a href="../../data/expansion_v15/manifests/'+name+'">'+name+'</a>：'+str(info['rows'])+' 条</li>')
    parts.append('</ul><p>train.jsonl：每图至多4个空间分离的定位任务；train_all.jsonl：保留全部训练标注。combined_train.jsonl仅合并原训练集，不加入原验证或测试集。validation_reserve为新增开发保留集。</p>')
    parts.append('<h2>接入方式与边界</h2><p>原始截图逐文件核对发布方SHA256或Git对象哈希；RGB哈希排除改编码重复，256位dHash距离≤4保守排除近似图。对全部6745张已索引公共评测图、历史训练/验证图和新数据内部进行排重。相同指令的多个有效区域合并为多框监督，未将它们互相当成负例。</p>')
    parts.append('<p>数据来源：<a href="https://huggingface.co/datasets/ServiceNow/GroundCUA">GroundCUA</a> · <a href="https://huggingface.co/datasets/zonghanHZH/ShowUI-web-8k">ShowUI-web训练整理子集</a> · <a href="https://huggingface.co/datasets/zonghanHZH/AMEX-8k">AMEX训练整理子集</a>。整理子集不算新的独立数据源，原始来源与许可字段一并保留。</p>')
    parts.append('<h2>质量核对样例</h2><p>下图按固定哈希选每个来源两张已接入训练图，框与文字仅用于人工核对，不用于模型输入。</p><img style="max-width:100%" src="qa_samples.png">')
    parts.append('<p><a href="results.json">完整统计、排重规则和哈希</a> · <a href="excluded_images.json">排除记录</a> · <a href="selection.json">冻结下载清单</a></p>')
    style="body{max-width:1100px;margin:36px auto;padding:0 20px;background:#f8fafc;color:#203449;font:16px/1.7 'Segoe UI','Microsoft YaHei',sans-serif}table{border-collapse:collapse;width:100%;background:white}td,th{padding:10px;border-bottom:1px solid #d9e2ed;text-align:left}th{background:#e5edf7}a{color:#176e9e}"
    (RUN/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>训练数据扩充V15</title><style>'+style+'</style><body>'+''.join(parts)+'</body></html>',encoding='utf-8')


if __name__ == '__main__':
    main()
