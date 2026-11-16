"""Training-input-QA supplement registered before any V31 dev observation.

Same candidate pairs and model, changing only local visual context128->256.
Original128-pixel primary remains unchanged and is always reported.
"""
from collections import Counter
from pathlib import Path
import argparse
import shutil
import sys
import numpy as np
from PIL import Image,ImageDraw,ImageFont,ImageFilter
import run_reinspection_v31 as run
from clickagent.reinspection_v31 import joint_vector,MODES,scores,decide

PARENT=run.OUT;OUT=PARENT/'context256'

def canvas256(image,left,right,blur=False):
    w,h=image.size;canvas=Image.new('RGB',(512,512),(235,235,235));geometry={}
    scale=min(512/w,256/h);rw,rh=max(1,round(w*scale)),max(1,round(h*scale));ox,oy=(512-rw)//2,(256-rh)//2
    with image.resize((rw,rh),Image.Resampling.LANCZOS) as im:
        if blur:im=im.filter(ImageFilter.GaussianBlur(8))
        canvas.paste(im,(ox,oy))
    for j,p in enumerate([left,right]):
        cw,ch=min(256,w),min(256,h);x0=int(np.clip(round(p[0]-cw/2),0,w-cw));y0=int(np.clip(round(p[1]-ch/2),0,h-ch))
        with image.crop((x0,y0,x0+cw,y0+ch)).resize((256,256),Image.Resampling.LANCZOS) as patch:
            if blur:patch=patch.filter(ImageFilter.GaussianBlur(8))
            canvas.paste(patch,(j*256,256))
        geometry[str(j)]=dict(box=[x0,y0,x0+cw,y0+ch],local_center=[j*256+(p[0]-x0)*256/cw,256+(p[1]-y0)*256/ch])
    draw=ImageDraw.Draw(canvas)
    try:font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',15)
    except OSError:font=ImageFont.load_default()
    for j,(p,label,color) in enumerate(zip([left,right],['LEFT','RIGHT'],[(0,165,20),(180,0,180)])):
        x,y=ox+p[0]*rw/w,oy+p[1]*rh/h;draw.ellipse((x-5,y-5,x+5,y+5),outline=color,width=2)
        x,y=geometry[str(j)]['local_center'];draw.ellipse((x-8,y-8,x+8,y+8),outline=color,width=2)
        draw.rectangle((j*256,256,j*256+70,276),fill='white');draw.text((j*256+3,257),label,fill=color,font=font)
    draw.line((0,255,511,255),fill='black',width=2);draw.line((255,256,255,511),fill='black',width=2)
    return canvas,dict(overview_resize=[rw,rh],overview_offset=[ox,oy],candidates=[list(left),list(right)],
        local=geometry,blurred=blur,ground_truth_used=False,canvas_size=[512,512],local_original_extent=256)

def preflight():
    with Image.new('RGB',(900,700),'white') as image:
        for blur in [False,True]:
            canvas,g=canvas256(image,[20,30],[880,680],blur);assert canvas.size==(512,512)
            for j,p in enumerate([[20,30],[880,680]]):
                box=g['local'][str(j)]['box'];center=g['local'][str(j)]['local_center']
                assert box[2]-box[0]==256 and box[3]-box[1]==256
                restored=[center[0]-j*256+box[0],center[1]-256+box[1]]
                np.testing.assert_allclose(restored,p)
            canvas.close()
    return dict(complete=True,transform_checks=4,matched_canvas_size=[512,512],original_extent=256)

def prepare():
    if (OUT/'protocol.json').exists():return
    parent=run.verify();state=run.read(PARENT/'status.json')
    assert not (PARENT/'frozen.json').exists(),'Register before fitting/development to avoid outcome-driven supplement'
    assert state['phase'].endswith('train') or state['phase']=='training_observations'
    OUT.mkdir(parents=True,exist_ok=True);run.write(OUT/'preflight.json',preflight())
    for name in ['public','labels','prompt_timing']:
        p=OUT/'data'/f'{name}.json';p.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(PARENT/'data'/f'{name}.json',p)
    for name in ['baseline_input_index.json','mvp_input_index.json']:shutil.copyfile(PARENT/name,OUT/name)
    deps=[Path(__file__),PARENT/'protocol.json',*map(Path,parent['hashes'])]
    spec=dict(registered_at_utc=run.now(),reason='Training-image QA showed128px crops clip visible labels;compare256px context without changing candidates or finalimagebudget.',
        parent_stage_at_registration=state,parent_primary_unchanged=True,registered_before_parent_model_fit=True,
        intervention='Original local extent128->256. Same512x512 composite,overview,predictedpoints,marks,prompt,fullinstruction,4096Dprefillfeatures,forward/reverse orders,clear/blur controls,kernelridge.01,threshold.1,folds,labels. No new candidate generation.',
        comparisons=['context256_combined-parent128_combined','context256_combined-context256_blurred','context256_combined-short_single'],
        statistics='All100primary,3pairedcomparisons98.333%source/RGBbootstrap;new72/old28secondary95%. Supplemental pilot,not independent confirmation. Both128and256reported,no winner adoption.',
        hashes={str(p):run.sha(p) for p in deps+[OUT/'preflight.json',OUT/'data/public.json',OUT/'data/labels.json',OUT/'baseline_input_index.json',OUT/'mvp_input_index.json']},
        backbone_inventory=parent['backbone_inventory'])
    run.write(OUT/'protocol.json',spec)
    # Input preview uses first available eligible training record,never outcomes.
    for c in run.read(PARENT/'data/public.json'):
        p=PARENT/'features'/(run.key(c['sample_id'])+'.npz')
        if c['cohort']!='train' or not p.exists():continue
        _,m=run.load(p)
        if not m['eligible']:continue
        with Image.open(c['image']) as source:im=source.convert('RGB')
        canvas,g=canvas256(im,m['base'],m['alternative']);canvas.save(OUT/'preview_train.png');canvas.close();im.close();break
    print('Registered256px context supplement before parent model fit.',flush=True)

def verify():
    spec=run.read(OUT/'protocol.json')
    for p,h in spec['hashes'].items():assert run.sha(p)==h,p
    return spec

def parent_path(c,part):return PARENT/part/(run.key(c['sample_id'])+'.npz')

def collect(rt,c):
    a,pair=run.load(parent_path(c,'features'))
    if not pair['eligible']:
        run.npz(run.path(c,'features'),a,pair);return
    assert run.sha(c['image'])==c['image_sha256']
    with Image.open(c['image']) as source:image=source.convert('RGB')
    try:
        for blur in [False,True]:
            for reverse in [False,True]:
                part=f'judge/{"blur" if blur else "clear"}_{int(reverse)}';p=run.path(c,part)
                if p.exists():continue
                left,right=(pair['alternative'],pair['base']) if reverse else (pair['base'],pair['alternative'])
                canvas,g=canvas256(image,left,right,blur)
                try:hidden,m=rt.judge(canvas,c['instruction'])
                finally:canvas.close()
                m.update(sample_id=c['sample_id'],base=pair['base'],alternative=pair['alternative'],reverse=reverse,blur=blur,geometry=g,skipped=False)
                run.npz(p,dict(hidden=hidden),m)
        clear=[run.load(run.path(c,f'judge/clear_{r}')) for r in [0,1]];blurred=[run.load(run.path(c,f'judge/blur_{r}')) for r in [0,1]]
        a['content']=joint_vector(clear[0][0]['hidden'],clear[1][0]['hidden']);a['blurred']=joint_vector(blurred[0][0]['hidden'],blurred[1][0]['hidden'])
        pair['judge_seconds']=sum(m['elapsed_seconds'] for _,m in clear+blurred)
        pair['workload_seconds']['content_policy']=pair['workload_seconds']['diverse_consensus']+sum(m['elapsed_seconds'] for _,m in clear)
        def u(m):
            p=np.exp(np.asarray(m['logits'])-max(m['logits']));p/=p.sum();return float(p[1]-p[0])
        pair['zero_shot_utility']=.5*(u(clear[0][1])-u(clear[1][1]))
        pair['local_original_extent']=256;pair['parent_feature_sha256']=run.sha(parent_path(c,'features'))
        run.npz(run.path(c,'features'),a,pair)
    finally:image.close()

def analyze():
    verify();rows=run.read(OUT/'predictions.json')['rows'];parent={r['sample_id']:r for r in run.read(PARENT/'predictions.json')['rows']};result={};lines=[]
    for cohort in ['all','new_development','old_val']:
        rr=[r for r in rows if cohort=='all' or r['cohort']==cohort];w=run.bootstrap_weights(rr);summary={name:100*float(np.mean([r['hits'][name] for r in rr])) for name in rr[0]['hits']}
        comparisons={}
        for tag,a,b in [('context-parent128','combined','parent128'),('context-blurred','combined','blurred'),('context-base','combined','short_single')]:
            diff=np.asarray([int(r['hits'][a])-int(parent[r['sample_id']]['hits']['combined'] if b=='parent128' else r['hits'][b]) for r in rr])
            level=1-.05/3 if cohort=='all' else .95;alpha=(1-level)/2;ci=(100*np.quantile(w@diff,[alpha,1-alpha])).tolist()
            comparisons[tag]=dict(delta_pp=100*float(diff.mean()),ci_pp=ci,positive=bool(ci[0]>0))
        replacements={mode:dict(switches=sum(r['methods'][mode]!=r['methods']['short_single'] for r in rr),repairs=sum(r['switches'][mode]['repair'] for r in rr),harms=sum(r['switches'][mode]['harm'] for r in rr)) for mode in MODES}
        result[cohort]=dict(n=len(rr),percent=summary,comparisons=comparisons,replacements=replacements)
        lines += [f'{cohort}: n={len(rr)}',str(summary),str(comparisons),str(replacements)]
    run.write(OUT/'results.json',dict(complete=True,**result));(OUT/'readout.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    cases=run.read(OUT/'data/public.json');frozen=run.read(OUT/'frozen.json');freeze=datetime_timestamp(frozen['frozen_at_utc']);checks=Counter()
    for c in cases:
        original_a,original_m=run.load(parent_path(c,'features'));a,m=run.load(run.path(c,'features'))
        assert original_m['base']==m['base'] and original_m['alternative']==m['alternative']
        np.testing.assert_array_equal(original_a['geometry'],a['geometry']);checks['unchanged_pairs']+=1
        if original_m['eligible']:
            for blur in ['clear','blur']:
                for reverse in [0,1]:
                    p=run.path(c,f'judge/{blur}_{reverse}');hidden,meta=run.load(p)
                    assert not meta['ground_truth_in_model_input'] and meta['geometry']['local_original_extent']==256
                    if c['cohort']!='train':assert p.stat().st_mtime>=freeze
                    checks['real_joint_prefills']+=1
    for row in frozen['models'].values():run.load_model(row)
    run.write(OUT/'verification.json',dict(complete=True,at_utc=run.now(),checks=dict(checks),all_primary_gates={k:v['positive'] for k,v in result['all']['comparisons'].items()},mainline_unchanged=True))
    run.write(OUT/'completion.json',dict(complete=True,at_utc=run.now(),results_sha256=run.sha(OUT/'results.json'),verification_sha256=run.sha(OUT/'verification.json')))
    run.status('complete',queries=212,context_extent=256)

def datetime_timestamp(s):
    from datetime import datetime
    return datetime.fromisoformat(s).timestamp()

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','all','analyze']);args=parser.parse_args()
    if args.stage=='prepare':prepare();return
    verify();assert run.read(PARENT/'completion.json')['complete'];run.OUT=OUT
    if args.stage=='analyze':analyze();return
    cases=run.read(OUT/'data/public.json');train=[c for c in cases if c['cohort']=='train'];dev=[c for c in cases if c['cohort']!='train']
    run.status('loading_backbone_context256');rt=run.Runtime()
    for i,c in enumerate(train,1):
        if not run.path(c,'features').exists():collect(rt,c)
        run.status('training_observations',completed=i,total=112)
    if not (OUT/'frozen.json').exists():run.train(train)
    for i,c in enumerate(dev,1):
        if not run.path(c,'features').exists():collect(rt,c)
        run.status('development_observations',completed=i,total=100)
    run.evaluate(dev);analyze()

if __name__=='__main__':main()
