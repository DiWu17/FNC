"""Seal the v8 matched training matrix before training or new test features."""
import shutil
from pathlib import Path

from clickagent.action_benchmark import sha256,write_json
from point_aligned_data import OUT,BASE,read,frozen_source_identity
from train_point_aligned import SOURCES


def main():
    OUT.mkdir(exist_ok=True);(OUT/'models').mkdir(exist_ok=True)
    if (OUT/'protocol.json').exists():
        p=read(OUT/'protocol.json');assert p['manifest_sha256']==sha256(OUT/'manifest.jsonl');print('PROTOCOL ALREADY SEALED');return
    selection=read(OUT/'data_selection.json');assert selection['complete']
    jobs=[]
    variants=[('region_fixed','region','fixed',False),('point_fixed','point','fixed',False),
              ('joint_detached','point','own',False),('joint_connected','point','own',True)]
    for suffix,seed in [('s1',20261112),('s2',20261113)]:
        for name,objective,decoder,connect in variants:
            jobs.append(dict(name=name+'_'+suffix,suffix=suffix,seed=seed,objective=objective,decoder=decoder,connect_fine=connect))
    snapshot=OUT/'source_snapshot';snapshot.mkdir(exist_ok=True)
    for source in SOURCES:shutil.copyfile(source,snapshot/Path(source).name)
    warm=[BASE/'models'/f'semantic_hard_{s}__point.pt' for s in ['s1','s2']]
    write_json(OUT/'protocol.json',dict(version=1,primary_global='joint_connected_s1',primary_pipeline='joint_connected_s1__crop_verified',
        jobs=jobs,epochs=20,training_queries_per_epoch=600,batch_size=8,updates_per_epoch=75,
        learning_rate=.0001,region_aux_weight=.25,weight_decay=.01,candidate_removal_probability=0,
        selection='Validation CPU integer point hits, then parent overlap, lower objective, earlier epoch; includes epoch0',
        training_sources_sha256={p:sha256(Path(p)) for p in SOURCES},warm_start_sha256={str(p):sha256(p) for p in warm},
        source_cache_identity=frozen_source_identity(),manifest_sha256=sha256(OUT/'manifest.jsonl'),
        data_selection_sha256=sha256(OUT/'data_selection.json'),train_queries=873,val_queries=172,test_queries=192,
        preparation_source_sha256=sha256(Path('scripts/prepare_point_aligned.py')),
        max_pixels_per_qwen_pass=262144,backbone_frozen=True,test_outputs_absent_before_freeze=True,
        point_objective='-log(sum probability of any currently valid candidate point), null if none; shared child/offset auxiliary and .25 region auxiliary',
        fine_connection='Identical fine adapter; only direct routed-field gradient is detached versus connected. Shared base query/vision projections remain trainable in both.',
        inference=dict(crop_proposal='primary global top1 parent center',crop_span_in_parent_cells=4,crop_min_side_original_px=256,
            crop_scale=2,readout_pool='global top2 plus local crop top2, exact integer duplicates removed, at most4',
            verifier='Frozen Qwen screenshot+instruction with a single hollow red circle; Yes/No first-token likelihood, max score; no threshold',
            verifier_crop_original_px=768,verifier_crop_scale=1,verifier_does_not_generate_coordinates=True,
            crop_native_coordinate_reference=True,main_crop_and_verifier_only_for_primary_seed=True,
            validation_smoke='First12 by fixed sample_id SHA, no test policy selection'),
        limitations=['Static positive pointing; no permission/absence false-click claim.',
            'New trained controls warm-start old v7 weights; compare their matched continuation, not from-scratch data efficiency.',
            'Joint-detached vs connected isolates direct fine-gradient connection; fixed vs own also changes point targets and decoder.',
            'All final readout predictions use CPU as canonical numerical execution; Qwen is FP16 on two GPUs.',
            'Zero-shot frozen visual verifier is not the trained GUI-Actor verifier.',
            'Crop and verifier add visual inference budget and have no acceleration claim.',
            'Two seeds apply to global training; primary-seed proposed crop is shared only within its inference comparisons.']))
    print('SEALED v8 eight jobs, 20 epochs, fresh192',flush=True)


if __name__=='__main__':main()
