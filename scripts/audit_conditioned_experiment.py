"""Independent data split, source identity and point scoring audit."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from clickagent.action_benchmark import sha256,write_json
from clickagent.action_model import parse_text_action
from clickagent.public_diagnostics import parse_offline_lax_click
from clickagent.public_evaluation import point_in_geometry
from run_target_token_experiment import read_json,read_jsonl,key


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runs/conditioned_v3'))
    args=parser.parse_args();out=args.output;data=Path('data/conditioned_v3')
    cases=read_jsonl(out/'manifest.jsonl');by_id={c['sample_id']:c for c in cases}
    assert len(by_id)==len(cases)
    selection=read_json(out/'selection.json');freeze=read_json(out/'frozen_readouts.json')
    assert selection['preparer_source_sha256']==sha256(Path('scripts/prepare_conditioned_data.py'))
    training=read_json(out/'training_identity.json');feature=read_json(out/'feature_identity.json')
    assert selection['manifest_sha256']==training['manifest_sha256']==feature['manifest_sha256']==sha256(out/'manifest.jsonl')
    assert training['feature_identity_sha256']==sha256(out/'feature_identity.json')
    assert freeze['training_identity_sha256']==sha256(out/'training_identity.json')
    assert freeze['test_features_at_freeze']==freeze['test_labels_used_for_fitting']==0
    assert freeze['primary_model']=='full'
    assert set(freeze['models'])=={'full','attention_only','no_image','no_mass','no_query','no_attention','static_96'}
    assert training['source_sha256']==freeze['head_sources']
    assert training['protocol_sha256']==sha256(Path('docs/CONDITIONED_SPATIAL_PROTOCOL.md'))
    evaluation_identity=read_json(out/'evaluation/identity.json')
    assert evaluation_identity['evaluation_source_sha256']==sha256(Path('scripts/evaluate_conditioned_readouts.py'))
    assert evaluation_identity['frozen_readouts_sha256']==sha256(out/'frozen_readouts.json')
    for identity in (feature,training):
        for path,digest in identity['source_sha256'].items():assert sha256(Path(path))==digest
    pinned_path=data/'qwen_weight_identity.json'
    assert feature['qwen_weight_identity_sha256']==sha256(pinned_path)
    backbone=read_json(pinned_path)
    for name,item in backbone['files'].items():
        path=Path(backbone['model_path'])/name
        assert path.stat().st_size==item['bytes'] and sha256(path)==item['sha256']
    for name,r in freeze['models'].items():
        assert sha256(out/'models'/r['file'])==r['sha256']
        hist=read_json(out/'models'/f'{name}_history.json')
        ranks=[(item['val']['hits'],-item['val']['loss_on_named']) for item in hist['history']]
        assert max(range(len(ranks)),key=lambda i:ranks[i])==hist['best_epoch']==r['best_epoch']
        assert hist['checkpoint_sha256']==r['sha256'] and hist['complete']
    assert read_json(out/'feature_audit.json')['audited_samples']==len(cases)
    test=[c for c in cases if c['experiment_split']=='test'];assert len(test)==192
    assert Counter(c['dataset'] for c in test)=={ds:48 for ds in ('ScreenSpot-v2','ScreenSpot-Pro','OSWorld-G','VenusBench-GD')}
    assert read_json(out/'ui_tars/final_audit.json')['same_test_images']==len(test)
    trainval=[c for c in cases if c['experiment_split']!='test']
    assert all(c['official_training_split'] and c['source_split']=='train' for c in trainval)
    index=read_json(data/'public_image_index.json');public={r['pixel_sha256'] for r in index['images']}
    public_dhash=[int(r['dhash256'],16) for r in index['images']]
    unique={}
    for c in cases:
        previous=unique.get(c['pixel_sha256'])
        assert previous is None or previous['experiment_split']==c['experiment_split']
        unique[c['pixel_sha256']]=c
    pixels=Counter();hashes=[]
    for c in unique.values():
        with Image.open(c['image']) as opened:
            im=opened.convert('RGB');w,h=im.size
            pixel=hashlib.sha256(f'{w}x{h}:RGB:'.encode()+im.tobytes()).hexdigest()
            gray=np.asarray(im.convert('L').resize((17,16),Image.Resampling.LANCZOS))
            dhash=np.packbits((gray[:,1:]>gray[:,:-1]).ravel()).tobytes().hex()
        assert [w,h]==c['image_size'] and pixel==c['pixel_sha256'] and dhash==c['dhash256']
        assert sha256(Path(c['image']))==c['image_sha256']
        pixels[c['experiment_split']]+=1
        if c['experiment_split']=='test':continue
        assert pixel not in public
        dhash=int(dhash,16)
        assert all((dhash^v).bit_count()>4 for v in public_dhash)
        assert all((dhash^v).bit_count()>4 for v in hashes)
        hashes.append(dhash)
    assert dict(pixels)==selection['unique_images']
    fresh=read_json(data/'test_selection.json');by_path={r['image']:r for r in index['images']}
    used={by_path[c['image']]['pixel_sha256'] for p in fresh['prior_manifest_files'] for c in read_jsonl(Path(p))}
    assert not {c['pixel_sha256'] for c in test}&used
    assert len(used)==fresh['prior_pixel_hashes_excluded']
    assert sha256(data/'test_selection.json')==selection['test_selection_sha256']
    freeze_time=(out/'frozen_readouts.json').stat().st_mtime_ns
    for c in test:
        assert (out/'features'/f"{key(c['sample_id'])}.npz").stat().st_mtime_ns>=freeze_time
        native=read_json(out/'native'/f"{key(c['sample_id'])}.json")
        assert native['sample_id']==c['sample_id'] and native['ground_truth_in_model_input'] is False
        complete=native['terminated_by_eos'] or native['generated_text'].endswith(('\n','\r'))
        strict=parse_text_action(native['generated_text'],final=True) if complete else None
        point=None
        if strict and strict['action']=='click':
            point=[round(v*(s-1)/1000) for v,s in zip(strict['normalized_point'],c['image_size'])]
        assert point==native['point']
        lax=parse_offline_lax_click(native['generated_text']) if complete else None
        lax_point=[round(v*(s-1)/1000) for v,s in zip(lax,c['image_size'])] if lax else None
        assert lax_point==native['completed_lax_point']
    rows=read_jsonl(out/'evaluation/all_predictions.jsonl');counts=Counter()
    for r in rows:
        c=by_id[r['sample_id']];point=r['point'];w,h=c['image_size']
        hit=point is not None and point_in_geometry([point[0]/w,point[1]/h],c['target_geometry'])
        assert hit==r['hit'];counts[r['method']]+=1
    assert all(n==len(test) for n in counts.values())
    result={'complete':True,'unique_images_rehashed':dict(pixels),'sample_counts':dict(Counter(c['experiment_split'] for c in cases)),
        'official_train_split_only_for_training':True,'train_val_test_pixel_disjoint':True,
        'train_val_near_duplicates_excluded':True,'all_public_images_excluded_from_training':len(public),
        'previous_experiment_images_excluded_from_test':len(used),'test_features_written_after_readout_freeze':True,
        'validation_only_checkpoint_choices_replayed':True,'all_point_hits_replayed':len(rows),
        'same_test_denominator_per_method':dict(counts),'source_checkpoint_hashes_match':True,
        'native_format_conversion_replayed':len(test),'no_test_model_selection':True,
        'manifest_sha256':sha256(out/'manifest.jsonl'),'frozen_readouts_sha256':sha256(out/'frozen_readouts.json')}
    result['qwen_local_checkpoint_hashes_unchanged']=True
    write_json(out/'final_audit.json',result)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
