"""Seal all v7 hypotheses, compute budget and primary before new VLM features."""
from collections import Counter
import json
from pathlib import Path

from clickagent.action_benchmark import sha256, write_json
from clickagent.semantic_head_experiment import build_readout
from run_target_token_experiment import read_jsonl


def main():
    out=Path('runs/semantic_head_v7');path=out/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text());assert old['manifest_sha256']==sha256(out/'manifest.jsonl')
        print('ALREADY SEALED');return
    rows=read_jsonl(out/'manifest.jsonl');selection=out/'data_selection.json';assert selection.exists()
    base=[c for c in rows if c['experiment_split']=='train' and not c.get('training_extension')]
    val=[c for c in rows if c['experiment_split']=='val'];test=[c for c in rows if c['experiment_split']=='test']
    diagnostic=[c for c in rows if c['experiment_split']=='diagnostic'];extra=[c for c in rows if c.get('training_extension')]
    assert len(base)==618 and len(val)==172 and len(test)==len(diagnostic)==192
    assert len({c['sample_id'] for c in rows})==len(rows)
    assert not any((out/'features').glob('*.npz')) if (out/'features').exists() else True
    jobs=[]
    conditions=[('flat_base','flat','base','random'),('static_base','static','base','random'),
        ('semantic_base','semantic','base','random'),('semantic_expanded','semantic','expanded','random'),
        ('semantic_hard','semantic','expanded','grouped_hard')]
    for suffix,seed in [('s1',20261104),('s2',20261105)]:
        for name,architecture,data,recipe in conditions:
            jobs.append(dict(name=name+'_'+suffix,architecture=architecture,data=data,recipe=recipe,seed=seed))
    baselines=['runs/candidate_ranker_v5/models/full.pt','runs/attention_field_v6/models/all.pt',
        'runs/attention_field_v6/models/all_replica.pt','runs/phrase_v4/models/local.pt']
    protected=['src/clickagent/candidate_ranker.py','src/clickagent/attention_field_ranker.py','scripts/train_attention_field_ranker.py',
        'scripts/evaluate_attention_field_ranker.py','runs/candidate_ranker_v5/evaluation/predictions.jsonl',
        'runs/attention_field_v6/evaluation/predictions.jsonl','runs/phrase_v4/evaluation/predictions.jsonl']
    import numpy as np
    parameter_counts={}
    for arch in ['flat','static','semantic']:
        model=build_readout(arch,np.ones(96)/96)
        parameter_counts[arch]=dict(total=sum(p.numel() for p in model.parameters()),core=sum(p.numel() for p in model.core.parameters()))
    assert parameter_counts['static']==parameter_counts['semantic']
    protocol=dict(question='Early target-conditioned head routing, point-goal selection/quality, and real same-image hard examples.',
        primary='semantic_hard_s1__quality',primary_score_mode='combined',primary_decoder='common frozen v5 full fine points',
        manifest_sha256=sha256(out/'manifest.jsonl'),data_selection_sha256=sha256(selection),
        data_preparation_sha256=sha256(Path('scripts/prepare_semantic_head_data.py')),
        train_queries_base=len(base),train_queries_extra=len(extra),train_queries_expanded=len(base)+len(extra),
        validation_queries=len(val),diagnostic_queries=len(diagnostic),fresh_test_queries=len(test),
        fresh_test_quotas=dict(Counter(c['dataset'] for c in test)),fresh_test_near_exclusion_radius=4,
        jobs=jobs,epochs=24,training_queries_per_epoch=600,batch_size=8,updates_per_epoch=75,
        width=32,head_width=16,routing_groups=4,learning_rate=.0003,weight_decay=.01,
        candidate_removal_probability=.25,hard_negative_weight=.25,hard_negative_margin=.5,
        checkpoint_selectors={'region':['rank.parent_overlap','rank.fixed_point_hits','lower_rank_loss','earlier_epoch'],
            'point':['rank.fixed_point_hits','rank.parent_overlap','lower_rank_loss','earlier_epoch'],
            'quality':['combined.fixed_point_hits','combined.parent_overlap','lower_quality_loss','earlier_epoch']},
        point_quality='Separate class-balanced per-case BCE predicts frozen-v5-point hit; all inputs detached, gradient clipping separate. Uncalibrated ranking score.',
        combined_score='candidate score + logsigmoid(detached-branch point quality logit)',
        hard_recipe='First sample a common set of 600 queries without replacement per seed/epoch, with identical per-query removal decisions. Random versus hard then changes only their order: shuffle screenshot groups and selected queries within groups. Add 0.25 softplus(0.5 + mean-logsumexp(other-target negative scores) - mean-logsumexp(current-positive scores)).',
        candidates='Same union of all generated-name token/head argmax cells, no GT proposal injection or expansion.',
        fixed_fine_checkpoint=baselines[0],fixed_fine_sha256=sha256(Path(baselines[0])),
        baseline_checkpoint_sha256={str(Path(p)):sha256(Path(p)) for p in baselines},
        unchanged_artifact_sha256={str(Path(p)):sha256(Path(p)) for p in protected},
        head_prior='runs/phrase_v4/evaluation/head_token_profile/train_only_head_prior.json',
        parameter_counts=parameter_counts,max_pixels=262144,backbone_trainable_parameters=0,
        test_features_required_absent_until_freeze=True,test_used_for_fitting_or_checkpoint_selection=False,
        diagnostic_reuse=True,comparison_plan='10 trajectories with 3 validation selectors each; primary fixed s1 combined; report s2 in full, no test winner selection.',
        limitations=['Static positive pointing, no action-permission/negative false-click claim.',
            'Expanded versus base changes the real data composition but matches total update count; random versus grouped_hard holds the expanded queries fixed.',
            'Early semantic/static physical parameter counts match. Flat architecture has a different parameter count.',
            'Primary scores share a previously trained frozen fine decoder, so its candidate-point oracle is a ceiling for these primary scores, not for Qwen.',
            'Existing validation reused over research rounds; final fresh test is the independent check.'])
    write_json(path,protocol);print('SEALED '+json.dumps(dict(extra=len(extra),fresh=len(test),jobs=len(jobs),parameter_counts=parameter_counts)),flush=True)


if __name__=='__main__':main()
