import numpy as np
from clickagent.prompt_readout_v25 import (select_readouts,should_stop,select_arrays,
    normalized_to_pixel,map_view_point,coordinate_consensus,PROMPTS,compose_arrays)
from clickagent.target_token import TARGET_NAME_PROMPT


def prefixes(text):return [text[:i+1] for i in range(len(text))]


def test_original_prompt_is_exact_and_stage_uses_own_token_indices():
    assert PROMPTS['name']==TARGET_NAME_PROMPT
    text='ACTION: CLICK\nTARGET: Save\n'
    s,m=select_readouts(text,prefixes(text),ended_by_eos=False,capped=False)
    assert ''.join(text[i-1] for i in s['target'])=='Save'
    assert m['target_available'] and not s['coordinate']


def test_framed_roles_are_not_mixed_into_target_readout():
    text='ACTION: CLICK\n<ground><container>toolbar</container><location>upper_left</location><target>Save</target></ground>'
    s,_=select_readouts(text,prefixes(text),ended_by_eos=False,capped=False)
    assert ''.join(text[i-1] for i in s['target'])=='Save'
    assert ''.join(text[i-1] for i in s['location'])=='upper_left'
    assert should_stop(text,'context')
    assert not should_stop(text.replace('</ground>',''),'context')
    assert set(s['target'])<=set(s['phrase'])


def test_coordinate_collection_does_not_stop_after_target():
    before='ACTION: CLICK\nTARGET: Save\n'
    assert should_stop(before,'name') and not should_stop(before,'coordinate')
    text=before+'POINT: 500, 250\n'
    assert should_stop(text,'coordinate')
    s,m=select_readouts(text,prefixes(text),ended_by_eos=False,capped=False)
    assert m['normalized_point']==[500,250]
    assert ''.join(text[i-1] for i in s['coordinate'])=='500, 250'
    assert ''.join(text[i-1] for i in s['separator'])==','


def test_incomplete_out_of_bounds_and_explanations_are_not_coordinates():
    prefix='ACTION: CLICK\nTARGET: Save\nPOINT: '
    for tail in ['1001, 2\n','-1, 2\n','500, 250 explanation\n','(500, 250]\n']:
        s,m=select_readouts(prefix+tail,prefixes(prefix+tail),ended_by_eos=False,capped=False)
        assert m['normalized_point'] is None and not s['coordinate']
    text=prefix+'500, 250'
    s,m=select_readouts(text,prefixes(text),ended_by_eos=False,capped=True)
    assert not s['coordinate']
    s,m=select_readouts(text,prefixes(text),ended_by_eos=True,capped=False)
    assert m['normalized_point']==[500,250]


def test_complete_compatible_coordinate_formats_are_registered():
    for pair in ['500, 250','(500,250)','[500, 250]','500 250']:
        text='ACTION: CLICK\nTARGET: Save\nPOINT: '+pair+'\n'
        _,m=select_readouts(text,prefixes(text),ended_by_eos=False,capped=False)
        assert m['normalized_point']==[500,250]


def test_actual_subtoken_spans_and_selected_arrays_align():
    chunks=['ACTION',': CLICK\n','TARGET',': Save\n','POINT',': ','500',',',' 250','\n']
    pp=[''.join(chunks[:i+1]) for i in range(len(chunks))];text=pp[-1]
    s,_=select_readouts(text,pp,ended_by_eos=False,capped=False)
    assert s['target']==[4] and s['coordinate']==[7,8,9] and s['separator']==[8]
    a=dict(all_attention=np.arange(20).reshape(10,1,2),all_hidden=np.arange(30).reshape(10,3),
           image_hidden=np.zeros((2,4)),vision_hidden=np.zeros((8,5)))
    m=dict(readout_stages=s,generated_token_ids=list(range(10)),input_tokens=100)
    z,meta=select_arrays(a,m,'coordinate')
    assert np.array_equal(z['target_attention'],a['all_attention'][[6,7,8]])
    assert meta['target_query_absolute_positions']==[106,107,108]


def test_coordinate_geometry_and_consensus_never_use_labels():
    assert normalized_to_pixel([1000,500],[2001,1001])==[2000,500]
    assert map_view_point([200,100],dict(scale=2,box=[20,30,500,500]))==[120,80]
    assert coordinate_consensus([[1,2],[300,300],[4,5],None])==[1,2]
    assert coordinate_consensus([None,None]) is None


def test_visual_composition_preserves_entity_query_and_mass_before_pruning():
    arrays=dict(all_attention=np.array([[[.4,.5,.08,.02]],[[.8,.01,.18,.01]]],np.float32),
        all_hidden=np.array([[1,2,3],[9,9,9]],np.float16),image_hidden=np.zeros((4,4)),vision_hidden=np.zeros((16,5)))
    meta=dict(readout_stages=dict(target=[1],location=[2],container=[],phrase=[1,2]),
        generated_token_ids=[1,2],input_tokens=20,grid_shape=[2,2],ground_fields=dict(location='upper_left'))
    base,_=compose_arrays(arrays,meta,'entity_mean');product,_=compose_arrays(arrays,meta,'role_product')
    added,_=compose_arrays(arrays,meta,'role_sum');gate,_=compose_arrays(arrays,meta,'location_gate')
    assert base['target_attention'][0,0].argmax()==1
    assert product['target_attention'][0,0].argmax()==0
    for x in [product,added,gate]:
        assert np.allclose(x['target_attention'].sum(-1),base['target_attention'].sum(-1))
        assert np.array_equal(x['target_hidden'],base['target_hidden'])
        assert np.isfinite(x['target_attention']).all()


def test_malformed_frames_and_unknown_modifiers_are_not_silently_guessed():
    text='ACTION: CLICK\n<ground><container>NONE</container><location>unknown</location><target>Save</target></ground>'
    s,m=select_readouts(text,prefixes(text),ended_by_eos=False,capped=False)
    assert not s['container'] and not s['location'] and m['ground_complete']
    bad=text.replace('<target>Save</target>','<target>Save</target><target>Exit</target>')
    s,m=select_readouts(bad,prefixes(bad),ended_by_eos=False,capped=False)
    assert not s['target'] and not m['ground_complete']
