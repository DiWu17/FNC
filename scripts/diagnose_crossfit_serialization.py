"""Verify saved statistics, preserving the original serialization diagnosis."""
from crossfit_protocol import OUT,read,rows
from clickagent.action_benchmark import write_json
from audit_crossfit_serialization import persisted_summary
from evaluate_crossfit_candidates import evaluate_records

dest=OUT/'evaluation/test'
saved=read(dest/'summary.json')
recomputed=evaluate_records(rows(OUT/'data/test.jsonl'),rows(dest/'predictions.jsonl'),rows(dest/'pools.jsonl'))['summary']
normalized,changes=persisted_summary(recomputed)
assert len(changes)==8
assert all(saved[k]==value for k,value in normalized.items()), 'A numerical or non-count-map statistic changed'
path=OUT/'checks/final_audit_serialization_diagnostic.json'
if path.exists():
    assert read(path)['json_roundtrip_equal'] is True
else:
    write_json(path,dict(json_roundtrip_equal=True,normalizations=changes,
        purpose='Persisted count-map keys only; all recomputed numeric fields exactly equal'))
print('SAVED STATISTICS EXACTLY MATCH; ONLY8 JSON COUNT-KEY MAPS NORMALIZED',flush=True)
