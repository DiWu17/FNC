import json
from pathlib import Path
import sys
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_crossfit_serialization import persisted_summary


def test_real_retained_and_pool_counts_match_persisted_json():
    from evaluate_crossfit_candidates import evaluate_records
    c=dict(sample_id='one',source='one',image_size=[100,100],
           target_geometry=dict(type='bbox',boxes=[[.1,.1,.3,.3]]))
    points=[dict(point=[20,20]),dict(point=[80,80])]
    p=dict(sample_id='one',method='probe_top4_gui',budget=16,point=[20,20],
           retained_points=[[20,20],[80,80]],n_verified=2)
    result=evaluate_records([c],[p],[dict(sample_id='one',pools={'16':points})],bootstrap_reps=10)['summary']
    saved=json.loads(json.dumps(result))
    assert result!=saved  # Regression trigger: int map keys become strings in JSON.
    normalized,changes=persisted_summary(result)
    assert normalized==saved and len(changes)==2
    assert normalized['methods']['probe_top4_gui__K16']['hits']==1


def test_scalar_metrics_are_never_coerced_or_rounded():
    x={'hits':1,'accuracy':1/3,'ci95':[-.02604166666666663,.08854166666666663]}
    y,changes=persisted_summary(x)
    assert y==x and changes==[] and type(y['hits']) is int
    assert type(y['accuracy']) is float
    wrong={**x,'hits':2}
    assert persisted_summary(wrong)[0]!=y


@pytest.mark.parametrize('bad',[
    {'unexpected':{4:1}},
    {'candidate_size_counts':{4:1,'4':2}},
    {'candidate_count_distribution':{4:1.0}},
    {'candidate_size_counts':{-1:1}},
])
def test_only_nonnegative_integer_count_maps_can_change_keys(bad):
    with pytest.raises(AssertionError):persisted_summary(bad)
