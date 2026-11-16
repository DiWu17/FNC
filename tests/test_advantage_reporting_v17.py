import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def module():
    spec = importlib.util.spec_from_file_location('v17_report_test', ROOT / 'scripts/report_advantage_validation_v17.py')
    obj = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding='utf-8')


def test_repeated_timings_are_clustered_to60_screenshots(tmp_path):
    report = module()
    report.OUT = tmp_path
    for method, seconds in [('ours_s1_p262144_speed', 1.), ('tag_p262144_all', 2.)]:
        rows = [dict(sample_id=str(i), source='a' if i < 30 else 'b', repetition=r, hit=1.,
                     measured=dict(elapsed_seconds=seconds + r * .1, peak_allocated_bytes=[100, 100]))
                for r in [0, 1] for i in range(60)]
        save(tmp_path / 'inference' / method / 'speed/results.json', dict(rows=rows))
    result = report.collect()
    assert result['speed']['ours_s1_p262144_speed']['timed_calls'] == 120
    assert result['paired_speed']['tag']['n'] == 60
    assert result['paired_speed']['tag']['estimate'] == pytest.approx(1.05 / 2.05)


def test_zero_clean_accuracy_cannot_be_called_robustness_advantage(tmp_path):
    report = module()
    report.OUT = tmp_path
    for method, hit in [('ours_p262144_all', 0), ('tag_p262144_all', 1)]:
        clean = [dict(sample_id=str(i), source='a' if i < 30 else 'b', condition='clean',
                      point=None if hit == 0 else [10, 20], hit=hit, seed_hits=[bool(hit)]) for i in range(60)]
        changed = [{**r, 'condition': condition} for condition in ['jpeg70', 'border24'] for r in clean]
        save(tmp_path / 'inference' / method / 'robustness/results.json', dict(clean_rows=clean, rows=changed))
    result = report.collect()
    assert result['robustness']['ours_p262144_all']['jpeg70']['clean_correct_retention'] is None
    assert result['claims']['tag']['robustness']['jpeg70'] == 'not_established'
