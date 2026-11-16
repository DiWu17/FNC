"""Pin the existing ModelScope checkpoint by its actual local file hashes."""
import json
from pathlib import Path

from clickagent.action_benchmark import sha256,write_json


def main():
    model=Path('models/Qwen3-VL-8B-Instruct');dest=Path('data/conditioned_v3/qwen_weight_identity.json')
    names=sorted(p.name for p in model.iterdir() if p.is_file() and p.suffix in ('.safetensors','.json','.txt'))
    record={'model_path':str(model.resolve()),'download_source':'ModelScope',
        'model_id':'Qwen/Qwen3-VL-8B-Instruct','files':{}}
    for name in names:
        path=model/name;record['files'][name]={'sha256':sha256(path),'bytes':path.stat().st_size}
        print('HASHED '+name,flush=True)
    assert len([name for name in names if name.endswith('.safetensors')])==4
    record['note']='Actual local checkpoint identity, not a claim of an independently checked upstream commit.'
    if dest.exists():
        assert json.loads(dest.read_text(encoding='utf-8'))==record,'Backbone changed; use a new experiment'
    else:write_json(dest,record)
    print('PINNED '+str(dest),flush=True)


if __name__=='__main__':main()
