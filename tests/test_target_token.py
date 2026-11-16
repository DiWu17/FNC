from types import SimpleNamespace
import threading

import numpy as np
import pytest

from clickagent.target_token import select_stages


def prefixes(pieces):
    return ["".join(pieces[:i+1]) for i in range(len(pieces))]


def test_target_span_uses_generated_subwords_not_action_or_marker():
    p=prefixes(["ACTION",":"," CLICK","\n","TARGET",":"," Save"," button","\n"])
    stages,meta=select_stages(p[-1],p,ended_by_eos=False,token_cap_reached=False)
    assert stages["action_word"]==[3]
    assert stages["target_marker"]==[6]
    assert stages["target_mean"]==[7,8]
    assert stages["target_first"]==[7] and stages["target_last"]==[8]
    assert meta["target_text"]=="Save button"


def test_cap_does_not_turn_partial_target_name_into_complete_span():
    p=prefixes(["ACTION: CLICK\nTARGET:"," Sett"])
    stages,meta=select_stages(p[-1],p,ended_by_eos=False,token_cap_reached=True)
    assert "target_mean" not in stages and not meta["target_available"]
    stages,meta=select_stages(p[-1],p,ended_by_eos=True,token_cap_reached=False)
    assert stages["target_mean"]==[2]


def test_incremental_multibyte_rewrites_keep_both_actual_tokens():
    p=["ACTION: CLICK\nTARGET:","ACTION: CLICK\nTARGET: �","ACTION: CLICK\nTARGET: 保",
       "ACTION: CLICK\nTARGET: 保存","ACTION: CLICK\nTARGET: 保存\n"]
    stages,_=select_stages(p[-1],p,ended_by_eos=False,token_cap_reached=False)
    assert stages["target_mean"]==[2,3,4]


def test_none_target_is_missing_not_an_attention_selected_answer():
    p=prefixes(["ACTION: WAIT\nTARGET:"," NONE","\n"])
    stages,meta=select_stages(p[-1],p,ended_by_eos=False,token_cap_reached=False)
    assert not meta["target_available"] and "target_mean" not in stages
    assert meta["action"]=="WAIT"


def test_marker_token_crossing_into_name_is_not_punctuation_control():
    p=prefixes(["ACTION: CLICK\nTARGET",": Save","\n"])
    stages,_=select_stages(p[-1],p,ended_by_eos=False,token_cap_reached=False)
    assert stages["target_mean"]==[2]
    assert "target_marker" not in stages


def test_runtime_forwards_every_real_target_token_before_its_map(monkeypatch):
    import clickagent.target_token as target
    torch=pytest.importorskip("torch")
    pieces={10:"ACTION: CLICK\nTARGET:",11:" Save",12:"\n"}
    state={"last":3,"forwards":[]}

    def logits(next_id):
        tensor=torch.full((1,1,100),-100.)
        tensor[0,0,next_id]=1
        return SimpleNamespace(logits=tensor)

    class Recorder:
        def __init__(self,*args):pass
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def begin(self,*args):pass
        def snapshot(self):
            return {"raw_image_attention":np.full((96,1),state["last"],np.float32),
                    "hidden":np.asarray([state["last"]],np.float32),"head_order":[str(i) for i in range(96)]}

    runtime=target.TargetTokenRuntime.__new__(target.TargetTokenRuntime)
    runtime._lock=threading.Lock()
    runtime.config=SimpleNamespace(layers=(15,28,36))
    runtime.eos_ids={99}
    runtime.processor=SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda ids,**kw:"".join(pieces[i] for i in ids)))
    runtime.model=lambda **kwargs:logits(10)
    runtime._prepare_image=lambda *args:({"input_ids":torch.tensor([[1,2,3]])},np.asarray([1]),(1,1))
    def forward(token_id,*args):
        state["forwards"].append(token_id)
        state["last"]=token_id
        return logits({10:11,11:12,12:99}[token_id]),None
    runtime._forward_token=forward
    monkeypatch.setattr(target,"RawSelectedLayerRecorder",Recorder)
    monkeypatch.setattr(target,"_sync_cuda",lambda:None)
    arrays,meta=runtime.infer_target_name(SimpleNamespace(size=(32,32)),"Click Save")
    assert state["forwards"]==[10,11,12]
    assert arrays["raw_attention"][:,0,0].tolist()==[3,10,11,12]
    assert meta["stages"]["target_mean"]==[2]
    assert meta["query_token_indices"]==[2,3,4,5]
    assert meta["generated_token_ids"]==[10,11,12]


def test_heat_cell_overlap_is_separate_from_center_click_accuracy():
    import runpy
    overlap=runpy.run_path("scripts/run_target_token_experiment.py")["overlap_cells"]
    geometry={"type":"bbox","boxes":[[.05,.05,.10,.10]]}
    assert overlap(geometry,(2,2)).tolist()==[True,False,False,False]
    poly={"type":"polygon","polygons":[[[.05,.05],[.10,.05],[.10,.10],[.05,.10]]]}
    assert overlap(poly,(2,2)).tolist()==[True,False,False,False]
    # The strip crosses both top cells, but no bottom cell.
    poly={"type":"polygon","polygons":[[[.1,.1],[.9,.1],[.9,.2],[.1,.2]]]}
    assert overlap(poly,(2,2)).tolist()==[True,True,False,False]
