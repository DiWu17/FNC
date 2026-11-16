"""Freeze v9 models and inference BEFORE any new test visual features exist."""
import argparse
import importlib.metadata
from pathlib import Path
import shutil
import time

from clickagent.action_benchmark import sha256,write_json
from point_aligned_data import read

OUT=Path('runs/context_verifier_v9')
SOURCES=['src/clickagent/context_verifier.py','src/clickagent/point_critic.py',
    'scripts/run_context_verifier.py','scripts/train_context_point_critic.py','scripts/run_gui_verifier.py',
    'scripts/seal_context_verifier.py','scripts/prepare_context_verifier.py',
    'scripts/evaluate_context_verifier.py','scripts/prepare_point_aligned.py','scripts/prepare_semantic_head_data.py',
    'scripts/point_aligned_predict.py','scripts/point_aligned_data.py','scripts/run_point_aligned_inference.py',
    'src/clickagent/conditioned_features.py','src/clickagent/target_token.py','src/clickagent/action_model.py',
    'src/clickagent/model_features.py','src/clickagent/point_aligned_readout.py','src/clickagent/point_aligned_inference.py',
    'src/clickagent/candidate_ranker.py','src/clickagent/semantic_head_ranker.py','src/clickagent/attention_field_ranker.py',
    'src/clickagent/model_baselines.py','scripts/run_ui_tars_baseline.py']


def protocol():
    return dict(version=9,seed=20261121,train_queries=873,validation_queries=172,fresh_test_queries=192,
        manifest_sha256=sha256(OUT/'manifest.jsonl'),data_selection_sha256=sha256(OUT/'data_selection.json'),
        primary='context_pool__point_critic_s1',fixed_pool_primary='fixed_pool__point_critic_s1',
        backbone_frozen=True,readout_frozen='v8 joint_connected_s1',
        critic=dict(seeds=[20261122,20261123],epochs=25,learning_rate=3e-4,batch_queries=12,width=128,
            epoch_selection='old validation globalTop4 hits, then lower loss, earlier epoch',
            features='all candidates use frozen global name-token and image states plus actual point',
            losses='multi-positive listwise + balanced BCE + .25 hard-negative pairwise',
            train_points='top64 frozen candidate points + 6 annotation positives + boundary negatives + same-image targets + random8',
            validation_points='unmodified globalTop4, no GT-proposed points'),
        fixed_pool='exact v8 globalTop2 + independent-cropTop2, dedup, at most4',
        context_pool='globalTop1, contextROI0Top1, contextROI1Top1, globalTop2, then round-robin fill; at most4',
        contextual_views=dict(maximum_rois=2,roi_nms_iou=.3,overview_pixels=65536,local_max_pixels=262144,
            crop='unchanged v8 4parentcells, minimum256 original px, upsample2x',
            target_name='tentative hypothesis, may correct it; original instruction retained'),
        comparative=dict(overview_pixels=65536,per_candidate_pixels=262144,local_side=768,
            scores='single-token candidate letter logits plus N; forward/reverse normalized log scores averaged',
            primary_abstains_on_N=True,forced_choice_secondary=True),
        gui_verifier=dict(model='microsoft/GUI-Actor-Verifier-2B',max_pixels=262144,
            views=['exact v8 marked768 crop','official marked full image'],dtype='float16',prompt='official model card'),
        yes_no_secondary='rerank identical archived Yes/No logits by stable log-odds to isolate probability saturation',
        critic_feature_rounding='FP16-rounded input features, FP32 utility head, identical to training/validation cache',
        cross_view_secondary='same critic, standardize within each view, exclude generating view, average; fallback global score',
        evaluation='point inside any GT region, inclusive; unavailable predictions remain in denominator',
        selection_metric='hits / cases with correct candidate in fixed pool; also rescues, harms, coverage',
        test_labels_used_for_training_or_selection=False)


def identity():
    old=read('runs/point_aligned_v8/frozen_readouts.json')
    for p,d in old['sources'].items():assert sha256(Path(p))==d,p
    for p,d in read('runs/point_aligned_v8/inference_identity.json')['sources'].items():assert sha256(Path(p))==d,p
    files=[OUT/'models'/f'point_critic_{s}.pt' for s in ['s1','s2']]
    files+=[Path('runs/point_aligned_v8/models')/old['models']['joint_connected_s1']['file']]
    return dict(protocol_sha256=sha256(OUT/'protocol.json'),manifest_sha256=sha256(OUT/'manifest.jsonl'),
        sources={p:sha256(Path(p)) for p in SOURCES},models={str(p):sha256(p) for p in files},
        qwen_identity_sha256=sha256(Path('data/conditioned_v3/qwen_weight_identity.json')),
        gui_verifier_identity_sha256=sha256(Path('models/GUI-Actor-Verifier-2B/clickagent_download_verified.json')),
        ui_tars_identity_sha256=sha256(Path('models/UI-TARS-1.5-7B/clickagent_download_verified.json')),
        libraries={p:importlib.metadata.version(p) for p in ['torch','transformers','numpy','Pillow']})


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['protocol','freeze'],required=True);args=parser.parse_args()
    if args.stage=='protocol':
        p=protocol()
        if (OUT/'protocol.json').exists():assert read(OUT/'protocol.json')==p
        else:write_json(OUT/'protocol.json',p)
        print('PROTOCOL SAVED');return
    assert read(OUT/'protocol.json')==protocol()
    assert read(OUT/'validation_smoke/completion.json')['complete']
    assert read(OUT/'validation_smoke/gui_verifier/completion.json')['complete']
    assert read(OUT/'validation_smoke/evaluation/audit.json')['passed']
    assert not list((OUT/'features').glob('*.npz'))
    for s in ['s1','s2']:assert read(OUT/'models'/f'point_critic_{s}.json')['complete']
    current=identity();path=OUT/'inference_identity.json'
    if path.exists():assert read(path)==current
    else:write_json(path,current)
    folder=OUT/'source_snapshot';folder.mkdir(exist_ok=True)
    for p in SOURCES:shutil.copyfile(p,folder/Path(p).name)
    write_json(OUT/'frozen.json',dict(complete=True,frozen_at=time.time(),fresh_features_at_freeze=0,
        primary=protocol()['primary'],identity_sha256=sha256(path)))
    print('V9 INFERENCE AND MODELS FROZEN',flush=True)


if __name__=='__main__':main()
