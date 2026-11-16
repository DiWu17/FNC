import numpy as np
import pytest

from clickagent.semantic_target import core_indices, temporal_map


def meta_for(pieces):
    prefix="ACTION: CLICK\nTARGET:"
    prefixes=[prefix]
    for piece in pieces:
        prefix+=piece
        prefixes.append(prefix)
    return {"generated_prefixes":prefixes,"stages":{"target_mean":list(range(2,len(prefixes)+1))}}


def test_core_filters_generic_words_but_keeps_name_subtokens():
    meta=meta_for([" the"," Save"," button"])
    assert core_indices(meta)==[3]
    raw=np.arange(5*2*3,dtype=np.float32).reshape(5,2,3)/1000
    np.testing.assert_array_equal(temporal_map(raw,meta,"core_mean"),raw[3])


def test_literal_generic_label_and_multibyte_parts_are_retained():
    assert core_indices(meta_for([" Menu"]))==[2]
    meta={"generated_prefixes":["ACTION: CLICK\nTARGET:","ACTION: CLICK\nTARGET: �","ACTION: CLICK\nTARGET: 保","ACTION: CLICK\nTARGET: 保存"],
          "stages":{"target_mean":[2,3,4]}}
    assert core_indices(meta)==[2,3,4]


def test_first2_uses_actual_generated_tokens_and_mass_weighting_is_convex():
    meta=meta_for([" Py","Charm"," icon"])
    raw=np.ones((5,2,3),np.float32)*.01
    raw[2]=.02;raw[3]=.04;raw[4]=.08
    np.testing.assert_allclose(temporal_map(raw,meta,"target_first2"),.03)
    result=temporal_map(raw,meta,"mass_weighted_core")
    assert (result>=.02).all() and (result<=.04).all()
    np.testing.assert_allclose(result,(.02*.02+.04*.04)/(.02+.04))


def test_missing_target_remains_a_failure_for_every_policy():
    assert temporal_map(np.zeros((1,96,4)),{"stages":{}},"core_mean") is None


def test_invalid_policy_is_rejected():
    with pytest.raises(ValueError):
        temporal_map(np.zeros((3,96,4)),meta_for([" Save"]),"oracle_best_test_token")
