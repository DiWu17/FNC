"""Replay sealed audit with persisted JSON count-map key semantics only.

The original auditor/evaluator and every forecast artifact remain byte-identical.
Only the two explicitly named count-distribution maps may convert int keys to
their JSON string form; counts, scores, points, intervals and all other checks
retain exact equality. Original failed audit and diagnostic are preserved.
"""
from pathlib import Path
import json

from clickagent.action_benchmark import sha256, write_json
from crossfit_protocol import OUT, read


def persisted_summary(summary):
    changes=[]
    allowed={'candidate_count_distribution','candidate_size_counts'}
    def visit(value,path):
        if isinstance(value,dict):
            integer_keys=[k for k in value if type(k) is int]
            if integer_keys:
                assert path and path[-1] in allowed, ('Unexpected numeric dictionary key',path)
                assert all(type(k) is int and k>=0 for k in value), ('Mixed/colliding count keys',path)
                assert all(type(v) is int and v>=0 for v in value.values()), ('Invalid count',path)
                changes.append(dict(path='/'.join(path),keys_before=list(value),keys_after=[str(k) for k in value]))
                return {str(k):v for k,v in value.items()}
            assert all(isinstance(k,str) for k in value), ('Unexpected dictionary key type',path)
            return {k:visit(v,path+[k]) for k,v in value.items()}
        if isinstance(value,list):return [visit(v,path+[str(i)]) for i,v in enumerate(value)]
        return value
    return visit(summary,[]),changes


def main():
    import torch
    # Match the sealed original auditor CLI and canonical proposer decoding.
    torch.set_num_threads(4)
    import audit_crossfit_candidates as original
    import evaluate_crossfit_candidates as evaluator
    original_path=Path('scripts/audit_crossfit_candidates.py')
    evaluator_path=Path('scripts/evaluate_crossfit_candidates.py')
    assert sha256(original_path)==read(OUT/'pretest_audit.json')['auditor_sha256']
    assert sha256(evaluator_path)==read(OUT/'evaluation/test/summary.json')['evaluator_sha256']
    thread_diagnostic=OUT/'checks/audit_thread_replay_diagnostic.json'
    thread_replay=read(thread_diagnostic)
    assert torch.get_num_threads()==thread_replay['canonical_threads']==4
    assert all(thread_replay['canonical_all_fields_bitwise_equal'].values())
    evaluate=evaluator.evaluate_records
    normalization=[]
    def evaluate_persisted(*args,**kwargs):
        result=evaluate(*args,**kwargs)
        result['summary'],changes=persisted_summary(result['summary'])
        normalization.extend(changes)
        return result
    evaluator.evaluate_records=evaluate_persisted
    result=original.run_audit(OUT,'final')
    assert len(normalization)==8, normalization
    result['serialization_audit_adapter']=dict(
        reason='Strict dictionary equality compared in-memory int count keys with persisted JSON str keys',
        only_allowed_count_maps_normalized=True,normalizations=normalization,
        numeric_values_points_scores_intervals_unchanged=True,
        cpu_threads=torch.get_num_threads(),
        thread_replay_diagnostic_sha256=sha256(thread_diagnostic),
        original_auditor_sha256=sha256(original_path),original_evaluator_sha256=sha256(evaluator_path),
        adapter_sha256=sha256(Path(__file__)),
        diagnostic_sha256=sha256(OUT/'checks/final_audit_serialization_diagnostic.json'),
        regression_tests_sha256=sha256(Path('tests/test_crossfit_serialization_audit.py')))
    for p in (Path(__file__),Path('tests/test_crossfit_serialization_audit.py'),OUT/'checks/final_audit_serialization_diagnostic.json',
              thread_diagnostic,Path('scripts/probe_crossfit_audit_threads.py')):
        result['input_sha256'][str(p)]=sha256(p)
    write_json(OUT/'final_audit.json',result)
    print(json.dumps(dict(complete=True,stage='final',serialization_count_maps=len(normalization))),flush=True)


if __name__=='__main__':main()
