import sys
from pathlib import Path
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import run_quality_readout_v15_pilot as pilot


def fixtures():
    return [dict(sample_id=f'{source}-{i}',dataset=source) for source in pilot.SOURCES for i in range(35)]


def test_balanced_prefix_is_nested_and_order_independent():
    a=fixtures();small=pilot.select_subset(a,15);large=pilot.select_subset(a,30)
    assert len(small)==45 and len(large)==90
    assert {r['sample_id'] for r in small}<={r['sample_id'] for r in large}
    assert pilot.select_subset(list(reversed(a)),30)==large
    assert all(sum(r['dataset']==s for r in large)==30 for s in pilot.SOURCES)


def test_no_candidate_or_success_filter_changes_selection():
    a=fixtures();before=pilot.select_subset(a,30)
    after=pilot.select_subset([dict(r,n=0,model_hit=False,cache_available=False) for r in a],30)
    assert [r['sample_id'] for r in before]==[r['sample_id'] for r in after]


def test_missing_fixed_prefix_is_not_silently_replaced():
    with pytest.raises(ValueError,match='Not enough source cases'):pilot.select_subset(fixtures(),36)


def test_archived_baseline_update_budget_is_matched():
    assert pilot.STEPS==8*((851+7)//8)==856
