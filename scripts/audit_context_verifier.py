"""Replay v9 prediction/scoring and check the new multi-image interfaces."""
from collections import Counter
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from clickagent.action_benchmark import sha256,write_json
from clickagent.point_aligned_inference import unique_points,map_crop_point
from clickagent.public_evaluation import point_in_geometry
from point_aligned_data import read
from run_target_token_experiment import key,read_jsonl
from run_context_verifier import symmetric_choice
from evaluate_context_verifier import score
from seal_context_verifier import identity,OUT


def check_visual_interface(root,cases):
    local_count=0;comparisons=0;order_masses=[];gui_records=0;gui_masses=[];gui_tokens=Counter()
    for c in cases:
        sid=key(c['sample_id']);p=read(root/'pipeline'/f'{sid}.json')
        assert p['sample_id']==c['sample_id'] and not p['ground_truth_in_model_inputs']
        points=p['global_result']['all_points'] if p['global_result']['candidates'] else []
        for point in points:
            assert 0<=point[0]<c['image_size'][0] and 0<=point[1]<c['image_size'][1]
        if p['old_crop_transform']:
            for candidate in p['old_crop_candidates']:
                assert candidate['point']==map_crop_point(candidate['local_point'],p['old_crop_transform'])
        if p['context_locals']:assert p['context_locals'][0]['transform']==p['old_crop_transform']
        for i,local in enumerate(p['context_locals']):
            transform=local['transform'];path=root/f'context_features_{i}'/f'{sid}.npz'
            with np.load(path,allow_pickle=False) as z:
                meta=json.loads(str(z['metadata']));gh,gw=meta['grid_shape'];count=gh*gw
                assert meta['own_token_forward'] and meta['all_backbone_parameters_frozen'] and not meta['ground_truth_in_model_input']
                grids=np.asarray(meta['all_image_grids']);assert grids.shape==(2,3) and (grids[:,0]==1).all()
                # The overview is manually resized toward64K; processor grid
                # rounding may increase it slightly. Every image still uses
                # the declared262144 hard processor maximum.
                assert (np.prod(grids,axis=1)*16*16<=262144).all()
                assert tuple(grids[1,1:]//2)==(gh,gw)
                assert meta['vision_offset']==int(np.prod(grids[0]))
                assert meta['image_content_positions']==meta['all_image_positions'][1]
                assert len(meta['image_content_positions'])==count
                assert z['image_hidden'].shape==(count,4096) and z['vision_hidden'].shape==(count*4,1152)
                assert z['target_attention'].shape==(len(meta['target_raw_indices']),96,count)
                assert z['target_hidden'].shape==(len(meta['target_raw_indices']),12288)
                assert meta['transform']==transform
                assert all(v>=1 for v in meta['target_raw_indices'])
                assert meta['target_query_absolute_positions']==[meta['input_tokens']+v-1 for v in meta['target_raw_indices']]
                local_count+=1
            for candidate in local['candidates']:
                assert candidate['point']==map_crop_point(candidate['local_point'],transform)
        for name,pool in p['pools'].items():
            assert len(pool)<=4 and len({tuple(v['point']) for v in pool})==len(pool)
            comp=p['comparisons'][name]
            assert comp['symmetric']==symmetric_choice(comp['forward'],comp['reverse'],pool)
            for direction in ['forward','reverse']:
                r=comp[direction]
                assert not r['ground_truth_in_model_inputs']
                if not pool:continue
                assert r['per_candidate_pixel_budget']==262144
                assert len(r['order'])==len(pool) and set(map(tuple,r['order']))==set(tuple(v['point']) for v in pool)
                assert not r['generated_coordinates']
                assert len(r['image_grids'])==1+len(pool)
                assert len(r['scores'])==len(pool)+1
                assert np.isfinite(r['scores']).all() and np.isfinite(r['choice_mass'])
                assert r['labels']==[chr(ord('A')+i) for i in range(len(pool))]+['N']
                order_masses.append(r['choice_mass']);comparisons+=1
        # Exact v8 candidate policy reconstructed without any geometry labels.
        g=[dict(v,source='global') for v in p['global_result']['candidates']]
        assert p['pools']['fixed_pool']==unique_points(g[:2],p['old_crop_candidates'][:2],limit=4)
        c0=p['context_locals'][0]['candidates'] if p['context_locals'] else []
        c1=p['context_locals'][1]['candidates'] if len(p['context_locals'])>1 else []
        assert p['pools']['context_pool']==unique_points(g[:1],c0[:1],c1[:1],g[1:2],c0[1:2],c1[1:2],g[2:4],limit=4)
        for coordinates,value in p['verification'].items():
            x,y=map(int,coordinates.split(','));raw=read(root/'verifier'/f'{sid}_{x}_{y}.json');assert value==raw
            assert raw['instruction']==c['instruction'] and not raw['ground_truth_in_model_input']
            yes=torch.tensor(raw['yes_logit'],dtype=torch.float32);no=torch.tensor(raw['no_logit'],dtype=torch.float32)
            assert float(torch.exp(yes-torch.logaddexp(yes,no)))==raw['score']
        gui=read(root/'gui_verifier'/f'{sid}.json')
        expected={tuple(v['point']) for v in unique_points(*p['pools'].values())}
        for mode,records in gui['scores'].items():
            assert set(tuple(v['point']) for v in records)==expected
            for record in records:
                assert abs(record['log_odds']-(record['true_logit']-record['false_logit']))<1e-5
                assert np.isfinite([record['true_logit'],record['false_logit'],record['label_mass']]).all()
                assert not record['ground_truth_in_model_input']
                assert int(np.prod(record['image_grid_thw']))*14*14<=262144
                gui_records+=1
                gui_masses.append(record['label_mass']);gui_tokens[record['preferred_token'].strip()]+=1
    return dict(local_archives_checked=local_count,comparative_passes_checked=comparisons,gui_records_checked=gui_records,
        candidate_letter_mass_min=min(order_masses) if order_masses else None,
        candidate_letter_mass_median=float(np.median(order_masses)) if order_masses else None,
        gui_label_mass_min=min(gui_masses) if gui_masses else None,
        gui_label_mass_median=float(np.median(gui_masses)) if gui_masses else None,
        gui_preferred_token_counts=dict(gui_tokens))


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['validation','test'],default='test');args=p.parse_args()
    torch.set_num_threads(4);source_digest=sha256(Path(__file__))
    cases=[c for c in read_jsonl(OUT/'manifest.jsonl') if c['experiment_split']==('val' if args.stage=='validation' else 'test')]
    root=OUT/'validation_smoke' if args.stage=='validation' else OUT
    if args.stage=='validation':cases=sorted(cases,key=lambda c:key(c['sample_id']))[:12]
    else:assert read(OUT/'inference_identity.json')==identity()
    # Preserve the previously audited geometry/scoring convention as well as
    # the v9 frozen prediction source; none of these legacy helpers was edited.
    for path,digest in read('runs/point_aligned_v8/evaluation/summary.json')['evaluation_sources_sha256'].items():
        assert sha256(Path(path))==digest,path
    interface=check_visual_interface(root,cases)
    if args.stage=='test':
        manifest=read(OUT/'data_selection.json');assert manifest['fresh_pixels_verified']==192
        for path,d in manifest['historical_manifest_sha256'].items():assert sha256(Path(path))==d
        assert len(set(c['pixel_sha256'] for c in cases))==len(cases)==192
        for c in cases:
            assert sha256(Path(c['image']))==c['image_sha256']
            with Image.open(c['image']) as im:assert list(im.size)==c['image_size']
        for suffix in ['s1','s2']:
            h=read(OUT/'models'/f'point_critic_{suffix}.json');assert len(h['history'])==26 and h['complete']
            for path,digest in h['sources'].items():assert sha256(Path(path))==digest
            for split in ['train','val']:assert read(OUT/f'critic_cache_{split}.json')['sources']==h['sources']
            best=max(h['history'],key=lambda v:(v['validation']['hits'],-v['validation']['loss']))
            assert best==h['selected'] and sha256(OUT/'models'/f'point_critic_{suffix}.pt')==h['sha256']
        assert read(OUT/'frozen.json')['fresh_features_at_freeze']==0
        # All fresh outputs must have been collected after source/model sealing.
        import os
        frozen_at=read(OUT/'frozen.json')['frozen_at']
        assert all(os.path.getmtime(path)>=frozen_at for path in (OUT/'features').glob('*.npz'))
    archived=read_jsonl(root/'evaluation/predictions.jsonl')
    replay,_,_=score(root,cases,args.stage=='test')
    assert archived==replay,'CPU point-critic/selection replay changed predictions'
    byid={c['sample_id']:c for c in cases}
    for r in archived:
        c=byid[r['sample_id']];point=r['point']
        hit=bool(point is not None and point_in_geometry([point[0]/c['image_size'][0],point[1]/c['image_size'][1]],c['target_geometry']))
        assert r['hit']==hit
    assert source_digest==sha256(Path(__file__))
    result=dict(passed=True,n=len(cases),rows_replayed=len(replay),full_qwen_rerun=False,
        critic_and_selection_replay='CPU exact',all_scores_common_region_metric=True,
        source_sha256=source_digest,**interface)
    write_json(root/'evaluation/audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
