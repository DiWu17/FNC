import importlib
import json
from pathlib import Path
import sys
import pytest

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
worker=importlib.import_module('run_mvp_confirmation_worker_v18')


def save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value),encoding='utf-8')


def test_inference_gate_requires_complete_training_and_unchanged_model(tmp_path,monkeypatch):
    folder=tmp_path/'confirmation'
    monkeypatch.setattr(worker,'ROOT',tmp_path)
    monkeypatch.setattr(worker,'OUT',folder)
    model=tmp_path/'frozen.pt';model.write_bytes(b'frozen_model')
    save(folder/'protocol.json',dict(hashes={}))
    save(folder/'freeze.json',dict(protocol_sha256=worker.sha(folder/'protocol.json'),models={'one':dict(path=str(model),sha256=worker.sha(model))}))
    with pytest.raises(FileNotFoundError):worker.authenticated()
    save(tmp_path/'runs/full_scaling_v18/completion.json',dict(complete=False))
    with pytest.raises(AssertionError):worker.authenticated()
    save(tmp_path/'runs/full_scaling_v18/completion.json',dict(complete=True))
    worker.authenticated()
    model.write_bytes(b'modified_model')
    with pytest.raises(AssertionError):worker.authenticated()


def test_resumed_prediction_is_bound_to_case_and_final_freeze(tmp_path):
    p=tmp_path/'row.json'
    case=dict(sample_id='case',image_sha256='image')
    assert worker.cached_result(p,case,'freeze') is None
    save(p,dict(sample_id='case',image_sha256='image',freeze_sha256='freeze'))
    assert worker.cached_result(p,case,'freeze')['sample_id']=='case'
    with pytest.raises(AssertionError):worker.cached_result(p,case,'different_freeze')
    with pytest.raises(AssertionError):worker.cached_result(p,dict(sample_id='other',image_sha256='image'),'freeze')
