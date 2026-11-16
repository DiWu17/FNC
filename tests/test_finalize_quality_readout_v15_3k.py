import copy
import sys
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from finalize_quality_readout_v15_3k import validate_details


def fixture():
    details=[dict(sample_id='a',source='s',methods={'model':dict(top1=True,top4=True,union_top1=False)})]
    result=dict(complete=True,n=1,methods={'model':dict(top1_hits=1,top4_coverage=1,union_top1_hits=0,accuracy=1.)},
        by_source={'s':dict(n=1,hits={'model':1})})
    return details,result


def test_valid_list_details_do_not_require_dict_complete_flag():
    details,result=fixture();validate_details(details,result,['model'])


def test_missing_case_rejected():
    details,result=fixture()
    with pytest.raises(ValueError,match='details count'):validate_details([],result,['model'])


def test_wrong_aggregate_or_model_rejected():
    details,result=fixture();bad=copy.deepcopy(result);bad['methods']['model']['top1_hits']=0
    with pytest.raises(ValueError,match='aggregate mismatch'):validate_details(details,bad,['model'])
    with pytest.raises(ValueError,match='result models'):validate_details(details,result,['model','missing'])


def test_bad_details_schema_rejected():
    details,result=fixture()
    with pytest.raises(ValueError,match='must be a list'):validate_details({'cases':details},result,['model'])
