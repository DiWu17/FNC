"""Independent CPU replay, source identity, transform and scoring checks for v8."""
from collections import Counter
import argparse
import json
import hashlib
import time
from pathlib import Path

import numpy as np
from PIL import Image
import torch

from clickagent.action_benchmark import sha256,write_json
from clickagent.point_aligned_inference import proposal_crop,map_crop_point,unique_points,VERIFIER_PROMPT
from evaluate_conditioned_readouts import score_point
from point_aligned_data import OUT,BASE,read,archive_record
from point_aligned_predict import load_predictors,predict_record
from run_point_aligned_inference import identity
from run_target_token_experiment import key,read_jsonl
from clickagent.public_diagnostics import parse_offline_lax_click


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait-for-scores',action='store_true');args=parser.parse_args()
    audit_source_digest=sha256(Path(__file__))
    torch.set_num_threads(4);protocol=read(OUT/'protocol.json');frozen=read(OUT/'frozen_readouts.json')
    assert read(OUT/'inference_identity.json')==identity()
    assert read(OUT/'completion.json')['complete']
    for p,digest in protocol['training_sources_sha256'].items():assert sha256(Path(p))==digest
    for p,digest in read(OUT/'data_selection.json')['historical_manifest_sha256'].items():assert sha256(Path(p))==digest
    cases=read_jsonl(OUT/'manifest.jsonl');old=read_jsonl(BASE/'manifest.jsonl')
    assert [c for c in cases if c['experiment_split']!='test']==[c for c in old if c['experiment_split'] in ('train','val')]
    for job in protocol['jobs']:
        h=read(OUT/'models'/f'{job["name"]}_history.json')
        assert h['complete'] and h['protocol_sha256']==sha256(OUT/'protocol.json')
        assert len(h['history'])==21 and [v['optimizer_steps'] for v in h['history']]==[0]+[75]*20
        chosen=max(h['history'],key=lambda r:(r['validation']['hits'],r['validation']['regions'],-r['validation']['loss']))
        assert chosen['epoch']==h['selected']['epoch']==frozen['models'][job['name']]['selected']['epoch']
    assert len({read(OUT/'models'/f'{j["name"]}_history.json')['parameters'] for j in protocol['jobs']})==1
    models=load_predictors();tested=0;checked_verifications=0;cache_digests={};posthoc_points={};float64_score_differences=[]
    tests=[c for c in cases if c['experiment_split']=='test'];assert len(tests)==192
    for i,c in enumerate(tests):
        sid=key(c['sample_id']);fp=OUT/'features'/f'{sid}.npz'
        assert fp.stat().st_mtime>=frozen['frozen_at']
        assert sha256(Path(c['image']))==c['image_sha256']
        if c['sample_id'].startswith('screenspot_pro:'):
            w,h=c['image_size'];original=c['metadata']['original']
            np.testing.assert_allclose(np.asarray(c['target_geometry']['boxes'][0])*[w,h,w,h],original['bbox'],rtol=0,atol=1e-7)
            assert original['img_size']==c['image_size']
        r=archive_record(fp,c,labels=False)
        assert r['metadata']['sample_id']==c['sample_id'] and r['metadata']['image_sha256']==c['image_sha256']
        assert r['metadata']['instruction_sha256']==hashlib.sha256(c['instruction'].encode('utf-8')).hexdigest()
        assert r['metadata']['all_backbone_parameters_frozen'] and r['metadata']['own_token_forward']
        assert not r['metadata']['ground_truth_in_model_input']
        # Strip gold fields before calling the replay predictor.
        stripped=dict(r,case={'sample_id':c['sample_id'],'image_size':c['image_size']})
        predicted=predict_record(stripped,models);stored=read(OUT/'global'/f'{sid}.json')['predictions']
        assert set(predicted)==set(stored)
        for method,value in predicted.items():
            assert value==stored[method],(c['sample_id'],method)
            tested+=1
        pipeline=read(OUT/'pipeline'/f'{sid}.json');g=stored['joint_connected_s1']
        for directory in ('native','crop_native'):
            native_path=OUT/directory/f'{sid}.json'
            if not native_path.exists():continue
            raw=read(native_path);complete=raw['terminated_by_eos'] or raw['generated_text'].endswith(('\n','\r'))
            normalized=parse_offline_lax_click(raw['generated_text']) if complete else None
            expected=[round(v*(size-1)/1000) for v,size in zip(normalized,raw['image_size'])] if normalized else None
            assert expected==raw['completed_lax_point']
            assert raw['instruction']==c['instruction'] and not raw['ground_truth_in_model_inputs']
            if directory=='crop_native':assert raw['original_point']==map_crop_point(expected,pipeline['transform'])
        cp=OUT/'crop_features'/f'{sid}.npz';crop_candidates=[]
        if g['candidates']:
            with Image.open(c['image']) as image:crop,transform=proposal_crop(image,g['candidates'][0]['parent_index'],r['metadata']['grid_shape'])
            assert pipeline['transform']==transform
            local_case={'sample_id':c['sample_id'],'image_size':list(crop.size)}
            local=archive_record(cp,local_case,labels=False)
            assert local['metadata']['transform']==transform
            crop_result=predict_record(local,models,only=['joint_connected_s1'])['joint_connected_s1']
            assert crop_result==pipeline['crop_result']
            crop_candidates=[dict(v,local_point=v['point'],point=map_crop_point(v['point'],transform),source='crop') for v in crop_result['candidates']]
            assert pipeline['crop_point']==(crop_candidates[0]['point'] if crop_candidates else None)
        global_candidates=[dict(v,source='global') for v in g['candidates']]
        assert pipeline['global_pool']==unique_points(global_candidates[:4])
        assert pipeline['combined_pool']==unique_points(global_candidates[:2],crop_candidates[:2],limit=4)
        filled=unique_points(pipeline['combined_pool'],pipeline['global_pool'],limit=4)
        winner=max(filled,key=lambda v:pipeline['verification'][','.join(map(str,v['point']))]['score']) if filled else None
        posthoc_points[c['sample_id']]=winner['point'] if winner else None
        for name,pool in [('global_verified_point',pipeline['global_pool']),('combined_verified_point',pipeline['combined_pool'])]:
            winner=max(pool,key=lambda v:pipeline['verification'][','.join(map(str,v['point']))]['score']) if pool else None
            assert pipeline[name]==(winner['point'] if winner else None)
        for point_string,v in pipeline['verification'].items():
            point=list(map(int,point_string.split(',')));vp=OUT/'verifier'/f'{sid}_{point[0]}_{point[1]}.json'
            assert read(vp)==v and v['system_prompt']==VERIFIER_PROMPT and v['instruction']==c['instruction']
            assert v['transform']['point']==point and not v['ground_truth_in_model_input'] and not v['generated_coordinates']
            yes,no=torch.tensor([v['yes_logit'],v['no_logit']],dtype=torch.float32)
            # Replay the actual archived computation, including its float32
            # subtraction at large logits; do not substitute a float64 sigmoid.
            score=float((yes-torch.logaddexp(yes,no)).exp())
            assert score==v['score'] and 0<=v['yes_no_mass']<=1.00001
            ideal=1/(1+np.exp(np.clip(v['no_logit']-v['yes_logit'],-700,700)))
            float64_score_differences.append(abs(ideal-v['score']))
            checked_verifications+=1
        for directory in ('features','global','native','pipeline','crop_features','crop_native'):
            path=OUT/directory/f'{sid}.{"npz" if directory in ("features","crop_features") else "json"}'
            if path.exists():cache_digests[str(path)]=sha256(path)
        if i%24==0:print(f'REPLAY {i+1}/192',flush=True)
    if args.wait_for_scores:
        deadline=time.monotonic()+1800
        while not (OUT/'evaluation'/'summary.json').exists():
            if time.monotonic()>deadline:raise TimeoutError('Readout replay passed but final scoring has not finished within30 minutes')
            time.sleep(5)
    assert read(OUT/'ui_tars'/'summary.json')['complete']
    rows=read_jsonl(OUT/'evaluation'/'predictions.jsonl');summary=read(OUT/'evaluation'/'summary.json')
    for p,digest in summary['evaluation_sources_sha256'].items():assert sha256(Path(p))==digest
    assert sha256(OUT/'evaluation'/'posthoc_pool_fill_declaration.json')==summary['posthoc_pool_fill_declaration_sha256']
    if summary['pro_input_resolution_audit_sha256']:
        assert sha256(OUT/'evaluation'/'label_visual_checks'/'pro_audit.json')==summary['pro_input_resolution_audit_sha256']
    if summary['verifier_precision_diagnostic_sha256']:
        assert sha256(OUT/'evaluation'/'verifier_precision_diagnostic.json')==summary['verifier_precision_diagnostic_sha256']
    lookup={c['sample_id']:c for c in tests};counts=Counter()
    for row in rows:
        assert row['hit']==bool(score_point(row['point'],lookup[row['sample_id']]))
        if 'failure_stage' in row:assert (row['failure_stage']=='hit')==row['hit']
        if row.get('posthoc'):assert row['point']==posthoc_points[row['sample_id']]
        counts[row['method']]+=int(row['hit'])
    assert {m:v['hits'] for m,v in summary['methods'].items()}==dict(counts)
    assert sha256(Path(__file__))==audit_source_digest,'Audit source changed during replay'
    write_json(OUT/'evaluation'/'audit.json',dict(passed=True,fresh_cases=192,global_predictions_replayed=tested,
        marked_point_records_checked=checked_verifications,scored_predictions=len(rows),
        all_readout_replay_device='cpu',full_qwen_rerun=False,parameters_matched=True,
        ground_truth_absent_from_prediction_replay=True,all_train_epochs_and_validation_choices_checked=True,
        inference_identity_sha256=sha256(OUT/'inference_identity.json'),cache_digests=cache_digests,
        audit_source_sha256=audit_source_digest,verifier_score_replay='Exact original float32 logaddexp and exp',
        max_difference_from_float64_sigmoid=max(float64_score_differences),
        limitations=['Qwen logits and vision states are archived evidence, not independently rerun in audit.',
            'Zero-shot Yes/No scores rank points; they are not calibrated correctness probabilities.']))
    print('AUDIT PASSED',flush=True)


if __name__=='__main__':main()
