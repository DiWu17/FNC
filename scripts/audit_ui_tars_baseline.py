"""Replay coordinate conversion and verify the external model's provenance."""
import argparse
import hashlib
import json
from pathlib import Path

from clickagent.action_benchmark import sha256,write_json
from clickagent.model_baselines import parse_ui_tars_point
from clickagent.public_evaluation import point_in_geometry
from run_target_token_experiment import read_json,read_jsonl,key


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runs/semantic_target_v2'))
    args=parser.parse_args()
    output=args.output;folder=output/'ui_tars';model=Path('models/UI-TARS-1.5-7B')
    cases=[c for c in read_jsonl(output/'manifest.jsonl') if c['experiment_split']=='test']
    summary=read_json(folder/'summary.json');assert summary['complete'] and summary['n']==len(cases)
    verified=read_json(model/'clickagent_download_verified.json')
    official=read_json(Path('runs/ui_tars_official_metadata.json'))
    assert verified['official_revision']==official['sha']==summary['official_revision']
    small={}
    lfs={}
    for f in official['siblings']:
        name=f['rfilename']
        if f.get('lfs') and name in verified['files']:
            lfs[name]=verified['files'][name]['sha256']==f['lfs']['sha256']
        if f.get('lfs') or not name.endswith(('.json','.txt')):continue
        data=(model/name).read_bytes()
        blob=hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()
        small[name]=blob==f['blobId']
    assert all(small.values())
    assert all(lfs.values()) and len(lfs)>=7
    predictions={r['sample_id']:r for r in read_jsonl(folder/'predictions.jsonl')}
    assert set(predictions)=={c['sample_id'] for c in cases}
    identity=read_json(folder/'identity.json')
    assert identity['manifest_sha256']==sha256(output/'manifest.jsonl')
    assert identity['driver_sha256']==sha256(Path('scripts/run_ui_tars_baseline.py'))
    assert identity['runtime_source_sha256']==sha256(Path('src/clickagent/model_baselines.py'))
    runtime=read_json(folder/'runtime.json')
    assert runtime['trainable_parameters']==0 and runtime['max_pixels']==262144
    assert not runtime['checkpoint_loading_info']['missing_keys'] and not runtime['checkpoint_loading_info']['mismatched_keys']
    parsed=0;completed=0
    for c in cases:
        raw=read_json(folder/f"{key(c['sample_id'])}.json")
        assert raw['sample_id']==c['sample_id'] and raw['ground_truth_in_model_input'] is False
        point=parse_ui_tars_point(raw['generated_text'],raw['processed_size'],raw['image_size']) if raw['terminated_by_eos'] else None
        assert point==raw['point']==predictions[c['sample_id']]['point']
        w,h=c['image_size']
        hit=point is not None and point_in_geometry([point[0]/w,point[1]/h],c['target_geometry'])
        assert hit==predictions[c['sample_id']]['hit']
        parsed+=point is not None;completed+=raw['terminated_by_eos']
    result={'complete':True,'same_test_images':len(cases),'coordinate_conversions_replayed':len(cases),
        'official_nonweight_git_blobs_match':small,'official_lfs_hashes_match':lfs,
        'official_weight_sha_verification':verified['all_download_sha256_verified'],
        'official_revision':verified['official_revision'],'trainable_parameters':0,
        'valid_coordinate_outputs':parsed,'terminated_by_eos':completed,
        'source_hashes_match':True,'max_pixels':262144}
    write_json(folder/'final_audit.json',result);print(json.dumps(result),flush=True)


if __name__=='__main__':main()
