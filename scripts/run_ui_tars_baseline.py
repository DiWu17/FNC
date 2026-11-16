"""Run the verified public UI-TARS model on the identical sealed test cohort."""
import argparse
import json
from pathlib import Path

from clickagent.action_benchmark import sha256,write_json
from clickagent.model_baselines import UITarsRuntime
from clickagent.public_evaluation import point_in_geometry
from run_target_token_experiment import read_json,read_jsonl,key


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runs/semantic_target_v2'))
    parser.add_argument('--model',type=Path,default=Path('models/UI-TARS-1.5-7B'))
    parser.add_argument('--limit',type=int)
    parser.add_argument('--split',choices=('dev','test'),default='test')
    args=parser.parse_args()
    verified=read_json(args.model/'clickagent_download_verified.json')
    assert verified['all_download_sha256_verified'] and verified['official_weight_lfs_sha_matches']
    dev_predictions=args.output/'ui_tars_dev_check/predictions.jsonl'
    if args.split=='test' and dev_predictions.exists():
        if not any(r['point'] is not None for r in read_jsonl(dev_predictions)):
            raise ValueError('All development outputs failed the point interface; inspect before running test')
    cases=[r for r in read_jsonl(args.output/'manifest.jsonl') if r['experiment_split']==args.split]
    folder=args.output/('ui_tars' if args.split=='test' else 'ui_tars_dev_check');folder.mkdir(exist_ok=True)
    identity={'manifest_sha256':sha256(args.output/'manifest.jsonl'),'model_revision':verified['official_revision'],
        'driver_sha256':sha256(Path(__file__)),'runtime_source_sha256':sha256(Path('src/clickagent/model_baselines.py')),
        'split':args.split,'max_pixels':262144}
    if (folder/'identity.json').exists() and read_json(folder/'identity.json')!=identity:
        raise ValueError('Baseline source/model changed; use a new output folder')
    write_json(folder/'identity.json',identity)
    pending=[r for r in cases if not (folder/f"{key(r['sample_id'])}.json").exists()]
    if args.limit:pending=pending[:args.limit]
    if pending:
        from PIL import Image
        runtime=UITarsRuntime(args.model)
        write_json(folder/'runtime.json',runtime.metadata)
        for i,case in enumerate(pending,1):
            assert sha256(Path(case['image']))==case['image_sha256']
            with Image.open(case['image']) as im:image=im.convert('RGB')
            result=runtime.infer(image,case['instruction'])
            result.update(sample_id=case['sample_id'],dataset=case['dataset'])
            write_json(folder/f"{key(case['sample_id'])}.json",result)
            print(f"[{i}/{len(pending)}] {case['dataset']} point={result['point']} tokens={result['generated_tokens']} text={result['generated_text']!r}",flush=True)
    rows=[]
    for case in cases:
        path=folder/f"{key(case['sample_id'])}.json"
        if not path.exists():continue
        r=read_json(path);point=r['point'];w,h=case['image_size']
        correct=point is not None and point_in_geometry([point[0]/w,point[1]/h],case['target_geometry'])
        rows.append(dict(sample_id=case['sample_id'],dataset=case['dataset'],point=point,hit=bool(correct)))
    (folder/'predictions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows),encoding='utf-8')
    summary={'n':len(rows),'hits':sum(r['hit'] for r in rows),'complete':len(rows)==len(cases),
        'datasets':{ds:{'n':sum(r['dataset']==ds for r in rows),'hits':sum(r['hit'] for r in rows if r['dataset']==ds)} for ds in sorted({r['dataset'] for r in rows})},
        'model':verified['official_model'],'official_revision':verified['official_revision'],'max_pixels':262144}
    write_json(folder/'summary.json',summary);print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
