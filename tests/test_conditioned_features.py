from types import SimpleNamespace
import threading

import numpy as np
import pytest


def test_image_features_are_actual_prefill_positions_and_hooks_preserve_logits():
    import torch
    from transformers import Qwen3VLConfig,Qwen3VLForConditionalGeneration
    from clickagent.conditioned_features import SpatialContentRecorder
    config=Qwen3VLConfig(text_config={'vocab_size':100,'hidden_size':32,'intermediate_size':64,
        'num_hidden_layers':3,'num_attention_heads':4,'num_key_value_heads':2,'head_dim':8,
        'rope_scaling':{'rope_type':'default','mrope_section':[1,1,2]}},
        vision_config={'hidden_size':32,'out_hidden_size':32,'intermediate_size':64,'depth':3,
        'num_heads':4,'deepstack_visual_indexes':[0,1,2]},image_token_id=95,video_token_id=96,
        vision_start_token_id=97,vision_end_token_id=98)
    config._attn_implementation='sdpa'
    model=Qwen3VLForConditionalGeneration(config).eval().requires_grad_(False)
    ids=torch.tensor([[1,2,3,4,5]])
    reference={}
    def capture(module,args,result):reference['hidden']=result.detach().cpu().clone()
    handle=model.model.language_model.layers[0].register_forward_hook(capture)
    with torch.inference_mode():
        no_hook=model(input_ids=ids,use_cache=True,logits_to_keep=1).logits
        with SpatialContentRecorder(model,(1,3),image_layer=1) as recorder:
            recorder.begin(np.asarray([1,3]),(1,2))
            out=model(input_ids=ids,use_cache=True,logits_to_keep=1)
            torch.testing.assert_close(out.logits,no_hook)
            expected=reference['hidden'][0,[1,3]].to(torch.float16).numpy()
            assert np.array_equal(recorder.image_hidden,expected)
            assert recorder.image_hidden.shape==(2,32)
            prefill=recorder.image_hidden.copy()
            model(input_ids=torch.tensor([[6]]),past_key_values=out.past_key_values,
                cache_position=torch.tensor([5]),use_cache=True,logits_to_keep=1)
            assert np.array_equal(recorder.image_hidden,prefill)
            assert recorder.snapshot()['hidden'].shape==(64,)
    handle.remove()
    assert not model.model.language_model.layers[0]._forward_hooks
    assert not model.model.language_model.layers[0].self_attn._forward_hooks


def test_conditioned_collector_keeps_only_real_name_token_rows(monkeypatch):
    import torch
    import clickagent.conditioned_features as module
    pieces={10:'ACTION: CLICK\nTARGET:',11:' Save',12:'\n'};state={'last':3,'forwarded':[]}
    def logits(n):
        values=torch.full((1,1,100),-100.);values[0,0,n]=1
        return SimpleNamespace(logits=values)
    class Model:
        def __call__(self,**kw):return logits(10)
        def parameters(self):return iter(())
    class Recorder:
        def __init__(self,*args):
            self.image_hidden=np.asarray([[7.,8.]],np.float16)
            self.vision_hidden=np.arange(8,dtype=np.float16).reshape(4,2)
        def __enter__(self):return self
        def __exit__(self,*args):pass
        def begin(self,*args):pass
        def snapshot(self):return {'raw_image_attention':np.full((96,1),state['last'],np.float32),
            'hidden':np.asarray([state['last']],np.float32),'head_order':[str(i) for i in range(96)]}
    runtime=module.ConditionedFeatureRuntime.__new__(module.ConditionedFeatureRuntime)
    runtime._lock=threading.Lock();runtime.config=SimpleNamespace(layers=(15,28,36));runtime.eos_ids={99}
    runtime.model=Model();runtime.processor=SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda ids,**kw:''.join(pieces[i] for i in ids)))
    runtime._prepare_image=lambda *args:({'input_ids':torch.tensor([[1,2,3]])},np.asarray([1]),(1,1))
    def forward(token,*args):
        state['last']=token;state['forwarded'].append(token)
        return logits({10:11,11:12,12:99}[token]),None
    runtime._forward_token=forward
    monkeypatch.setattr(module,'SpatialContentRecorder',Recorder);monkeypatch.setattr(module,'_sync_cuda',lambda:None)
    arrays,meta=runtime.infer_spatial_features(SimpleNamespace(size=(32,32)),'Click Save')
    assert state['forwarded']==[10,11,12]
    assert arrays['target_attention'][:,0,0].tolist()==[11]
    assert arrays['target_hidden'][:,0].tolist()==[11]
    assert arrays['prefill_attention'][0,0]==3
    assert meta['target_raw_indices']==[2] and meta['target_query_absolute_positions']==[4]
    assert meta['image_content_positions']==[1] and meta['ground_truth_in_model_input'] is False


def test_vision_block_order_is_restored_to_fine_raster_without_averaging():
    from clickagent.conditioned_features import unmerge_raster
    grouped=np.arange(8).reshape(8,1)
    raster=unmerge_raster(grouped,(1,2)).reshape(2,4)
    assert raster.tolist()==[[0,1,4,5],[2,3,6,7]]


def test_real_vision_merger_capture_preserves_multimodal_logits():
    import torch
    from transformers import Qwen3VLConfig,Qwen3VLForConditionalGeneration
    from clickagent.conditioned_features import SpatialContentRecorder,unmerge_raster
    config=Qwen3VLConfig(text_config={'vocab_size':100,'hidden_size':32,'intermediate_size':64,
        'num_hidden_layers':3,'num_attention_heads':4,'num_key_value_heads':2,'head_dim':8,
        'rope_scaling':{'rope_type':'default','mrope_section':[1,1,2]}},
        vision_config={'hidden_size':32,'out_hidden_size':32,'intermediate_size':64,'depth':3,
        'num_heads':4,'patch_size':2,'temporal_patch_size':2,'num_position_embeddings':16,
        'deepstack_visual_indexes':[0,1,2]},image_token_id=95,video_token_id=96,
        vision_start_token_id=97,vision_end_token_id=98)
    config._attn_implementation='sdpa'
    model=Qwen3VLForConditionalGeneration(config).eval().requires_grad_(False)
    inputs={'input_ids':torch.tensor([[97,95,95,98,2]]),'pixel_values':torch.randn(8,24),
        'image_grid_thw':torch.tensor([[1,2,4]])}
    actual={}
    def capture(module,args):actual['fine']=args[0].detach().cpu().to(torch.float16).numpy().copy()
    handle=model.model.visual.merger.register_forward_pre_hook(capture)
    with torch.inference_mode():
        reference=model(**inputs,use_cache=True,logits_to_keep=1).logits
        with SpatialContentRecorder(model,(1,3),image_layer=1) as recorder:
            recorder.begin(np.asarray([1,2]),(1,2))
            output=model(**inputs,use_cache=True,logits_to_keep=1)
            torch.testing.assert_close(reference,output.logits)
            assert recorder.vision_hidden.shape==(8,32)
            assert np.array_equal(recorder.vision_hidden,unmerge_raster(actual['fine'],(1,2)))
            assert recorder.image_hidden.shape==(2,32)
    handle.remove()
    assert not model.model.visual.merger._forward_pre_hooks
