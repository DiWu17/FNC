"""Select a fresh, image-isolated confirmation subset without model outputs."""
from __future__ import annotations
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from prepare_semantic_head_data import HammingIndex, image_info, key, read_json, read_rows, sha, valid_public_geometry, write_json, write_rows
from prepare_candidate_scorer import resolve_history, path_key

OUT=ROOT/'runs/contribution_v16'
QUOTAS={'screenspot_pro':150,'venusbench_gd':150}
SEED=20261104


def historical_manifests():
    result=subprocess.run(['rg','--files','--hidden','--no-ignore',str(ROOT/'runs'),'-g','*.jsonl'],capture_output=True,text=True,check=True)
    included=[];ignored=[]
    for name in result.stdout.splitlines():
        p=Path(name).resolve();parts=[s.lower() for s in p.relative_to(ROOT/'runs').parts]
        artifact=any('pytest' in s or s.startswith('refinement_tests') or s=='source_archive' for s in parts)
        relevant='manifest' in p.name.lower() or 'data' in parts[:-1]
        if OUT==p or OUT in p.parents or artifact or not relevant:
            ignored.append(str(p));continue
        included.append(p)
    # Conservatively protect the entire imported training/development image pool,
    # beyond the2973/300 actually used. These rows carry verified RGB/dHash IDs.
    included.extend(ROOT/'data/expansion_v15_qc1_candidate/manifests'/n for n in ('train.jsonl','validation_reserve.jsonl'))
    return sorted(set(included)),ignored


def choose(rows, infos, pixels, hashes, quotas=QUOTAS):
    tree=HammingIndex(hashes);used=set(pixels);selected=[];availability={}
    for source,quota in quotas.items():
        eligible=[r for r in rows[source] if r.get('split')=='test' and r.get('task_kind')=='grounding'
            and r.get('expected_actionable') is True and valid_public_geometry(r)]
        retained=[];rejected=Counter()
        for r in eligible:
            info=infos[path_key(r['image'])]
            if info['pixel_sha256'] in pixels:rejected['history_exact']+=1
            elif HammingIndex.near(tree,int(info['dhash256'],16),4):rejected['history_or_prior_selected_near']+=1
            else:retained.append(r)
        taken=[]
        for r in sorted(retained,key=lambda r:key(SEED,'independent_v16',r['sample_id'])):
            info=infos[path_key(r['image'])];pixel=info['pixel_sha256'];h=int(info['dhash256'],16)
            if pixel in used or tree.near(h,4):rejected['within_confirmation_exact_or_near']+=1;continue
            original=r.get('metadata',{})
            group=original.get('application') or r.get('group_id') or pixel
            taken.append(dict(r,source_split='test',split='independent_test',experiment_split='independent_test',
                pixel_sha256=pixel,dhash256=info['dhash256'],image_sha256=info['file_sha256'],source_group=str(group)))
            used.add(pixel);tree.add(h)
            if len(taken)==quota:break
        availability[source]=dict(eligible_queries=len(eligible),before_within_subset_exclusion=len(retained),selected=len(taken),rejected=dict(rejected))
        if len(taken)!=quota:raise ValueError(f'{source}: required{quota} strictly isolated screenshots, found{len(taken)}; exclusions not relaxed')
        selected.extend(taken)
    return selected,availability


def main():
    output=OUT/'data/independent_test.jsonl';receipt=OUT/'data/independent_selection.json'
    if receipt.exists():
        saved=read_json(receipt)
        for name,expected in saved['input_sha256'].items():assert sha(name)==expected,name
        assert sha(output)==saved['manifest_sha256']
        print('Existing confirmation cohort authenticated; no reselection.');return
    paths,ignored=historical_manifests()
    infos,pixels,hashes,identity_paths,historical_rows=resolve_history(paths)
    print(json.dumps(dict(stage='history_resolved',manifests=len(paths),rows=historical_rows,rgb=len(pixels)),ensure_ascii=False),flush=True)
    sources={s:ROOT/f'data/public_v1/manifests/{s}.jsonl' for s in QUOTAS}
    selected,stats=choose({s:read_rows(p) for s,p in sources.items()},infos,pixels,hashes)
    print(json.dumps(dict(stage='confirmation_selected',availability=stats),ensure_ascii=False),flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        for r,actual in zip(selected,pool.map(image_info,[r['image'] for r in selected])):
            for a,b in [('pixel_sha256','pixel_sha256'),('dhash256','dhash256'),('image_sha256','file_sha256'),('image_size','image_size')]:
                assert r[a]==actual[b],(r['sample_id'],a)
    write_rows(output,selected)
    tools=[Path(__file__),ROOT/'scripts/prepare_semantic_head_data.py',ROOT/'scripts/prepare_candidate_scorer.py']
    inputs=sorted(set(Path(p) for p in identity_paths)|set(sources.values())|set(tools))
    write_json(receipt,dict(complete=True,created_at_utc=datetime.now(timezone.utc).isoformat(),seed=SEED,quotas=QUOTAS,
        selection_key='sha256(20261104:independent_v16:sample_id)',availability=stats,
        input_sha256={str(p):sha(p) for p in inputs},manifest_sha256=sha(output),historical_manifests=[str(p) for p in paths],
        ignored_artifacts=ignored,historical_rows=historical_rows,protected_unique_rgb=len(pixels),protected_dhash=len(hashes),
        selected_queries=len(selected),selected_rgb=len({r['pixel_sha256'] for r in selected}),
        source_groups={s:len({r['source_group'] for r in selected if r['source']==s}) for s in QUOTAS},
        exclusions='RGB exact and dHash256 Hamming<=4 against all discovered project histories, all imported train/reserve images, and within selected confirmation across sources; one query per screenshot.',
        no_model_execution=True,model_outputs_used_for_selection=False,selected_images_redecoded_and_verified=True,
        final_feature_extraction_requires_all_ablation_checkpoints_frozen=True,
        scope='Fresh300-query image-isolated confirmation subset of pinned public benchmarks, not a full official benchmark score or a guarantee of unseen apps/pretraining.',
        primary_sources={'screenspot_pro':'https://github.com/likaixin2000/ScreenSpot-Pro-GUI-Grounding','venusbench_gd':'https://huggingface.co/datasets/inclusionAI/VenusBench-GD'}))
    print(json.dumps(dict(complete=True,queries=len(selected),availability=stats)),flush=True)


if __name__=='__main__':main()
