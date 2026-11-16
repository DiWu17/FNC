"""Validation-only real-image alignment and local-attention distribution check."""
import json
from pathlib import Path
import numpy as np
from clickagent.action_benchmark import sha256,write_json

ROOT=Path('runs/context_verifier_v9/validation_smoke')


def cosine(a,b):
    return float(np.mean(np.sum(a*b,-1)/(np.linalg.norm(a,axis=-1)*np.linalg.norm(b,axis=-1)+1e-12)))


def main():
    rows=[]
    for path in sorted((ROOT/'context_features_0').glob('*.npz')):
        with np.load(path,allow_pickle=False) as z:
            cm=json.loads(str(z['metadata']));cv=z['vision_hidden'].astype(np.float32);ci=z['image_hidden'].astype(np.float32)
            ca=z['target_attention'].copy();cq=z['target_hidden'].astype(np.float32)
        with np.load(ROOT/'old_crop_features'/path.name,allow_pickle=False) as z:
            om=json.loads(str(z['metadata']));ov=z['vision_hidden'].astype(np.float32);oi=z['image_hidden'].astype(np.float32)
            oa=z['target_attention'].copy();oq=z['target_hidden'].astype(np.float32)
        assert cv.shape==ov.shape and cm['grid_shape']==om['grid_shape'] and cm['transform']==om['transform']
        same=bool(cm.get('target_available') and om.get('target_available') and cm.get('target_text')==om.get('target_text'))
        row=dict(sample_id=cm['sample_id'],vision_exact_equal=bool(np.array_equal(cv,ov)),vision_cosine=cosine(cv,ov),
            vision_relative_l2=float(np.linalg.norm(cv-ov)/np.linalg.norm(ov)),image_prefix_cosine=cosine(ci,oi),
            old_name=om.get('target_text'),context_name=cm.get('target_text'),same_name=same,
            old_action=om.get('action'),context_action=cm.get('action'),
            old_local_attention_mass=float(oa.sum(-1).mean()) if len(oa) else None,
            context_local_attention_mass=float(ca.sum(-1).mean()) if len(ca) else None,
            target_state_cosine_same_name=cosine(cq,oq) if same and cq.shape==oq.shape else None)
        rows.append(row)
    same=[r for r in rows if r['same_name']]
    summary=dict(n=len(rows),vision_exact_equal_cases=sum(r['vision_exact_equal'] for r in rows),
        image_prefix_cosine_mean=float(np.mean([r['image_prefix_cosine'] for r in rows])),same_name_cases=len(same),
        mean_local_attention_mass_ratio_same_name=float(np.mean([r['context_local_attention_mass']/r['old_local_attention_mass'] for r in same])))
    write_json(ROOT.parent/'evaluation/context_validation_alignment.json',dict(complete=True,split='original_validation_only',
        summary=summary,rows=rows,source_sha256=sha256(Path(__file__)),
        warning='Post-hoc interface/mechanism diagnosis. Lower local attention mass is not proof of worse localization or proof that remaining attention goes to overview rather than text.'))
    print(json.dumps(summary),flush=True)


if __name__=='__main__':main()
