"""Fresh fixed-cohort accuracy; no training, label-conditioned inference or HTML."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
OUT = ROOT/'runs/mvp_confirmation_v18'


def read(p): return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024), b''): h.update(chunk)
    return h.hexdigest()


def write(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    temp = p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    temp.replace(p)


def status(phase, **fields):
    row = dict(phase=phase, complete=phase=='complete', updated_at_utc=datetime.now(timezone.utc).isoformat(), **fields)
    write(OUT/'worker_status.json', row); print(json.dumps(row, ensure_ascii=False), flush=True)


def authenticated():
    spec = read(OUT/'protocol.json'); freeze = read(OUT/'freeze.json')
    for p, expected in spec['hashes'].items(): assert sha(p)==expected,p
    assert freeze['protocol_sha256']==sha(OUT/'protocol.json')
    for name, model in freeze['models'].items(): assert sha(model['path'])==model['sha256'],name
    assert read(ROOT/'runs/full_scaling_v18/completion.json')['complete']
    return spec, freeze


def geometry_hit(point, case):
    from clickagent.public_evaluation import point_in_geometry
    return bool(point is not None and point_in_geometry([point[i]/case['image_size'][i] for i in [0,1]], case['target_geometry']))


def cached_result(path, case, freeze_hash):
    if not path.exists(): return None
    row = read(path)
    assert row['sample_id']==case['sample_id'] and row['image_sha256']==case['image_sha256']
    assert row['freeze_sha256']==freeze_hash
    return row


def ours(cases, freeze):
    import numpy as np
    import torch
    import run_full_scaling_v18 as full
    import run_frozen_geometry_v18 as geom
    from clickagent.evidence_retention import RetentionModel
    from clickagent.frozen_geometry_v18 import pipeline_geometry
    torch.set_num_threads(4)
    full.prior.OUT = OUT/'features'
    full.prior.status = lambda phase,**kw: status('features_'+phase,**kw)
    models = {}
    for name, info in freeze['models'].items():
        saved = torch.load(info['path'], map_location='cpu', weights_only=True)
        model = RetentionModel().to('cuda:0').eval().requires_grad_(False)
        model.load_state_dict(saved['state_dict']); models[name]=(model,saved)
    extractor = None; records = []; token = sha(OUT/'freeze.json')
    mode = freeze['geometry_mode']
    for i, case in enumerate(cases, 1):
        path = OUT/f'predictions/ours/{full.key(case["sample_id"])}.json'
        cached = cached_result(path, case, token)
        if cached is not None:
            records.append(cached); continue
        assert sha(case['image'])==case['image_sha256']
        if extractor is None: extractor=full.FullExtractor()
        pipe, _ = extractor.case({k:case[k] for k in full.prior.base.PUBLIC})
        views = [dict(feature_path=pipe['global_feature_path'],feature_sha256=pipe['global_feature_sha256'])]+pipe['local_views']
        raw = []
        for v in views:
            assert sha(v['feature_path'])==v['feature_sha256']
            with np.load(v['feature_path'],allow_pickle=False) as z:
                raw.append(({k:z[k].copy() for k in ['target_attention','target_hidden','image_hidden']},json.loads(str(z['metadata']))))
        outputs = {}
        for geometry in ['original']+([mode] if mode!='original' else []):
            changed = pipeline_geometry(pipe,case['image_size'],geometry,full.prior.base.build_c1_c3_pools)
            r = geom.feature_record(changed,dict(image_size=case['image_size']),raw)
            for arm in ['original3k','matched3k','full']:
                if arm=='matched3k' and geometry!='original': continue
                points=[]
                for s in [1,2,3]:
                    model,saved=models[f'{arm}_s{s}']
                    with torch.inference_mode():
                        if r['n']:
                            inputs,_=full.prior.tensors([r],saved['mean'].numpy(),saved['std'].numpy(),'aligned','cuda:0')
                            scores=model(**inputs).float().cpu().numpy(); point=r['points'][int(scores.argmax())].tolist()
                        else: point=None
                    points.append(point)
                outputs[arm+('_union' if geometry=='original' else '_geometry')]=points
        if mode=='original':
            for arm in ['original3k','full']: outputs[arm+'_geometry']=outputs[arm+'_union']
        # Annotation access for scoring occurs after all public-only predictions.
        row=dict(sample_id=case['sample_id'],source=case['dataset'],image_sha256=case['image_sha256'],
            pixel_sha256=case['pixel_sha256'],freeze_sha256=token,points=outputs,
            hits={name:[geometry_hit(p,case) for p in points] for name,points in outputs.items()},
            views=len(views),
            processed_pixels=sum(int(np.prod(meta['grid_shape']))*1024 for _,meta in raw),
            backbone_forward_calls=sum(1+len(meta['generated_token_ids']) for _,meta in raw),
            ground_truth_in_model_inputs=False)
        write(path,row);records.append(row)
        if i%5==0 or i==len(cases):status('ours',completed=i,total=len(cases))
    write(OUT/'predictions/ours.json',dict(complete=True,rows=records,freeze_sha256=token))


def mvp(cases, freeze):
    from PIL import Image
    from clickagent.advantage_runtimes import configure_paths
    configure_paths('mvp')
    from clickagent.advantage_runtimes import MVPAdapter
    import torch
    torch.set_num_threads(4)
    runtime=None;records=[];token=sha(OUT/'freeze.json')
    for i,case in enumerate(cases,1):
        key=hashlib.sha256(case['sample_id'].encode()).hexdigest()[:24]
        path=OUT/f'predictions/mvp/{key}.json'
        cached=cached_result(path,case,token)
        if cached is not None:
            records.append(cached);continue
        assert sha(case['image'])==case['image_sha256']
        if runtime is None:
            runtime=MVPAdapter(OUT/'mvp_runtime',max_pixels=262144)
            write(OUT/'mvp_runtime/metadata.json',runtime.metadata)
        with Image.open(case['image']) as opened:image=opened.convert('RGB')
        before=runtime.counter.snapshot()
        try:
            with torch.inference_mode():output=runtime.infer(image,case['instruction'])
        finally:image.close()
        assert output['ground_truth_in_model_input'] is False
        after=runtime.counter.snapshot()
        row=dict(sample_id=case['sample_id'],source=case['dataset'],image_sha256=case['image_sha256'],
            pixel_sha256=case['pixel_sha256'],freeze_sha256=token,point=output['point'],
            hit=geometry_hit(output['point'],case),output=output,
            processed_pixels=after['processed_pixels']-before['processed_pixels'],
            backbone_forward_calls=after['backbone_forward_calls']-before['backbone_forward_calls'],
            ground_truth_in_model_inputs=False)
        write(path,row);records.append(row)
        if i%5==0 or i==len(cases):status('mvp',completed=i,total=len(cases))
    write(OUT/'predictions/mvp.json',dict(complete=True,rows=records,freeze_sha256=token))


def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['ours','mvp'],required=True);args=p.parse_args()
    _,freeze=authenticated()
    cases=[json.loads(s) for s in (OUT/'data/heldout.jsonl').read_text(encoding='utf-8').splitlines()]
    assert len(cases)==280==len({c['pixel_sha256'] for c in cases})
    status('loading_'+args.method,total=len(cases))
    {'ours':ours,'mvp':mvp}[args.method](cases,freeze)
    authenticated();status('complete',method=args.method,n=len(cases))


if __name__=='__main__':
    try: main()
    except BaseException:
        status('failed',error=traceback.format_exc());raise
