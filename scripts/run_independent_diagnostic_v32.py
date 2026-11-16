"""Frozen V31-prefix / MVP comparison on history-excluded ScreenSpot-Pro RGBs."""
import os
for _k,_v in dict(PYTHONDONTWRITEBYTECODE='1',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4').items():os.environ.setdefault(_k,_v)
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import json
from pathlib import Path
import subprocess
import sys
import time
import traceback
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_pair_latency_v30 as base
import run_point_prefix_latency_v31 as fast
from prepare_semantic_head_data import HammingIndex,image_info,valid_public_geometry
from prepare_candidate_scorer import path_key
from clickagent.grounded_composition_v28 import pixel
from clickagent.public_evaluation import point_in_geometry
OUT=ROOT/'runs/independent_diagnostic_v32'
SEED='v32_20261110_independent_diagnosis'
RADIUS=4

def write(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(v,ensure_ascii=False,indent=2,default=base.default,allow_nan=False)+'\n',encoding='utf-8')
    for attempt in range(100):
        try:os.replace(temp,p);return
        except PermissionError:
            if attempt==99:raise
            time.sleep(.1)

def status(method,phase,**fields):
    v=dict(method=method,phase=phase,complete=phase=='complete',pid=os.getpid(),updated_at_utc=base.now(),**fields)
    write(OUT/method/'status.json',v);print(json.dumps(v,ensure_ascii=False,default=base.default),flush=True)

def rows(p):
    if p.suffix=='.jsonl':return [json.loads(s) for s in p.read_text(encoding='utf-8-sig').splitlines() if s.strip()]
    x=base.read(p)
    return x if isinstance(x,list) else x.get('rows',x.get('cases',[]))

def stratified_take(pool,n,namespace):
    groups=defaultdict(list)
    for c in pool:groups[(c['application_group'],c['ui_type'])].append(c)
    quotas={k:int(n*len(v)//len(pool)) for k,v in groups.items()}
    order=sorted(groups,key=lambda k:(-(n*len(groups[k])/len(pool)-quotas[k]),base.key(str(k))))
    for k in order[:n-sum(quotas.values())]:quotas[k]+=1
    chosen=[]
    for k,v in sorted(groups.items()):chosen.extend(sorted(v,key=lambda q:base.key(SEED+'|'+namespace+'|'+q['sample_id']))[:quotas[k]])
    assert len(chosen)==n
    return sorted(chosen,key=lambda q:base.key(SEED+'|execution|'+q['sample_id']))

def prepare():
    if (OUT/'protocol.json').exists():return verify()
    OUT.mkdir(parents=True,exist_ok=True);status('controller','auditing_history')
    manifest=ROOT/'data/public_v1/manifests/screenspot_pro.jsonl';public=rows(manifest)
    names=subprocess.check_output(['rg','--files','runs','data','-g','*.jsonl','-g','*public.json','-g','warmup.json','-g','!*download*'],cwd=ROOT,text=True,encoding='utf-8').splitlines()
    history=[];cache_paths=[]
    for name in names:
        p=ROOT/name;s=str(p).lower().replace('\\','/')
        if any(x in s for x in ['independent_diagnostic_v32','/raw/','/tools/','pytest','source_archive','snapshot','/tmp','refinement_tests','documentation_at_closeout']):continue
        if 'hash_cache' in p.name:cache_paths.append(p);continue
        if '/data/public_v1/' in s:continue
        if p.name in ['public.json','warmup.json','manifest.jsonl','cases.jsonl'] or ('manifest' in p.name and p.suffix=='.jsonl') or ('/data/' in s and p.name in ['train.jsonl','val.jsonl','test.jsonl','combined_train.jsonl','validation_reserve.jsonl','test_manifest.jsonl']) or p.name.startswith('training_pool_'):
            history.append(p)
    index=ROOT/'data/conditioned_v3/public_image_index.json'
    infos={path_key(c['image']):c for c in base.read(index)['images']}
    for p in cache_paths:
        for c in rows(p):
            if c.get('image') and c.get('pixel_sha256') and c.get('dhash256'):infos.setdefault(path_key(c['image']),c)
    source_rows={};history_rows=[];missing={};unresolved=[];inputs=[manifest,index,*history,*cache_paths]
    for file_index,p in enumerate(history,1):
        for c in rows(p):
            if not isinstance(c,dict):continue
            if c.get('sample_id') and c.get('image') and Path(c['image']).is_absolute():source_rows[c['sample_id']]=c
            if c.get('image') or c.get('image_path') or c.get('pixel_sha256'):history_rows.append((p,c))
        if file_index%20==0:status('controller','reading_history_manifests',completed=file_index,total=len(history))
    resolved=[];controls={};path_cache={}
    for row_index,(p,c) in enumerate(history_rows,1):
        if c.get('pixel_sha256') and c.get('dhash256'):resolved.append(c);continue
        name=c.get('image',c.get('image_path'))
        if not name:unresolved.append(dict(manifest=str(p),sample_id=c.get('sample_id'),reason='no_image_or_both_hashes'));continue
        cached_path=(str(p.parent),name)
        if cached_path in path_cache:resolved.append(path_cache[cached_path]);continue
        key=path_key(name,p);info=infos.get(key)
        if info is None and not Path(key).is_file():
            control=p.parent/'control.json'
            if control not in controls:
                controls[control]=None
                if control.is_file():
                    cr=base.read(control);src=Path(cr.get('source_manifest',''))
                    if src.is_file() and base.sha(src)==cr.get('source_manifest_sha256'):
                        inputs.extend([control,src]);controls[control]=(src,{q.get('sample_id'):q for q in rows(src)})
            if controls[control] is not None:
                src,source_index=controls[control];original=source_index.get(c.get('sample_id'))
                if original:key=path_key(original.get('image',original.get('image_path')),src);info=infos.get(key)
            if info is None and not Path(key).is_file() and c.get('sample_id') in source_rows:key=path_key(source_rows[c['sample_id']]['image']);info=infos.get(key)
        if info is not None:resolved.append(info);path_cache[cached_path]=info
        elif Path(key).is_file():missing[key]=None;pending=dict(pending_path=key);resolved.append(pending);path_cache[cached_path]=pending
        else:unresolved.append(dict(manifest=str(p),sample_id=c.get('sample_id'),image=name))
        if row_index%10000==0:status('controller','resolving_history_images',completed=row_index,total=len(history_rows))
    write(OUT/'history_resolution_initial.json',dict(manifests=len(history),rows=len(history_rows),cached=len(resolved)-sum('pending_path' in q for q in resolved),missing_files=len(missing),unresolved=unresolved))
    assert not unresolved,('Historical image references unresolved',unresolved[:5])
    if missing:
        with ThreadPoolExecutor(max_workers=6) as pool:
            for n,(name,info) in enumerate(zip(sorted(missing),pool.map(image_info,sorted(missing))),1):
                missing[name]=info
                if n%100==0 or n==len(missing):status('controller','hashing_historical_RGB',completed=n,total=len(missing))
        write(OUT/'additional_history_hashes.json',list(missing.values()))
    resolved=[missing[q['pending_path']] if 'pending_path' in q else q for q in resolved]
    hist_pixels={q['pixel_sha256'] for q in resolved};hist_hash={int(q['dhash256'],16) for q in resolved};tree=HammingIndex(hist_hash)
    candidates=[];rejected=Counter()
    for c in public:
        if not c.get('expected_actionable') or not valid_public_geometry(c):rejected['invalid_geometry']+=1;continue
        info=infos[path_key(c['image'])]
        if info['pixel_sha256'] in hist_pixels:rejected['historical_exact']+=1;continue
        if tree.near(int(info['dhash256'],16),RADIUS):rejected['historical_near_le4']+=1;continue
        m=c['metadata'];candidates.append(dict(c,image_sha256=info['file_sha256'],pixel_sha256=info['pixel_sha256'],dhash256=info['dhash256'],application=m['application'],application_group=m['application_group'],ui_type=m['ui_type']))
    available=[];used=set();near=HammingIndex()
    for c in sorted(candidates,key=lambda q:base.key(SEED+'|canonicalRGB|'+q['sample_id'])):
        if c['pixel_sha256'] in used or near.near(int(c['dhash256'],16),RADIUS):continue
        available.append(c);used.add(c['pixel_sha256']);near.add(int(c['dhash256'],16))
    n=len(available);diag_n=200 if n>=300 else min(200,2*n//3);reserve_n=min(100,n-diag_n)
    write(OUT/'availability.json',dict(total_queries=len(public),historical_unique_RGB=len(hist_pixels),rejected=dict(rejected),remaining_queries=len(candidates),mutually_distinct_near_excluded_RGB=n,diagnostic_n=diag_n,reserve_n=reserve_n,exclusion_radius=RADIUS,exclusion_relaxed=False))
    assert diag_n>=20,('Insufficient strictly fresh diagnostic images',n)
    diag=stratified_take(available,diag_n,'diagnostic');chosen={q['sample_id'] for q in diag}
    reserve=stratified_take([q for q in available if q['sample_id'] not in chosen],reserve_n,'reserve')
    verify_cases=diag+reserve
    with ThreadPoolExecutor(max_workers=4) as pool:
        for i,(c,actual) in enumerate(zip(verify_cases,pool.map(image_info,[q['image'] for q in verify_cases])),1):
            for k in ['pixel_sha256','dhash256','image_size']:assert c[k]==actual[k],(c['sample_id'],k)
            assert c['image_sha256']==actual['file_sha256']
            if i%25==0 or i==len(verify_cases):status('controller','verifying_selected_RGB',completed=i,total=len(verify_cases))
    keys=['sample_id','dataset','image','image_size','instruction','image_sha256','pixel_sha256','dhash256','application','application_group','ui_type']
    for name,cohort in [('public',diag),('reserve_public',reserve)]:write(OUT/'data'/f'{name}.json',[{k:q[k] for k in keys} for q in cohort])
    write(OUT/'data/labels.json',{q['sample_id']:q['target_geometry'] for q in diag})
    write(OUT/'data/reserve_labels.json',{q['sample_id']:q['target_geometry'] for q in reserve})
    write(OUT/'data/warmup.json',base.read(ROOT/'runs/pair_latency_v30/data/warmup.json'))
    write(OUT/'historical_exclusion.json',dict(complete=True,history_manifest_paths=[str(q) for q in history],historical_RGB=sorted(hist_pixels),historical_dhash256=[f'{v:064x}' for v in sorted(hist_hash)],unresolved=[],near_radius=4,diagnostic_reserve_mutually_near_excluded=True))
    imports=[Path(__file__),ROOT/'scripts/run_pair_latency_v30.py',ROOT/'scripts/run_point_prefix_latency_v31.py',ROOT/'src/clickagent/point_prefix_v31.py',ROOT/'src/clickagent/advantage_runtimes.py',ROOT/'src/clickagent/coordinate_calibration_v29.py',ROOT/'src/clickagent/grounded_composition_v28.py',ROOT/'scripts/prepare_semantic_head_data.py',ROOT/'scripts/prepare_candidate_scorer.py']
    prior=base.read(ROOT/'runs/pair_latency_v30/protocol.json')
    inputs+=imports+[Path(q) for q in prior['hashes']]+list((OUT/'data').glob('*.json'))+[OUT/'historical_exclusion.json',OUT/'availability.json']
    for q,h in prior['hashes'].items():assert base.sha(q)==h,q
    tests=subprocess.run([sys.executable,'-B','-X','utf8','-m','unittest','discover','-s','tests','-p','test_independent_diagnostic_v32.py','-v'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    write(OUT/'unit_tests.json',dict(exit_code=tests.returncode,output=tests.stdout+tests.stderr));assert tests.returncode==0,tests.stdout+tests.stderr
    write(OUT/'protocol.json',dict(registered_at_utc=base.now(),authorization='试试吧',purpose='Fixed-model paired accuracy,latency and error diagnosis on history-excluded official ScreenSpot-Pro subset;not full leaderboard,not final untouched confirmation.',
        selection='Exclude all discovered historical used/prepared training and evaluation manifests,exactRGB and256bitdHash<=4;mutualexclusion across diagnostic and reserve. One instruction per distinctRGB,chosen by fixedSHA namespace. Proportional application_group/UI-type stratification. No outcomes used. Target200diagnostic+100reserve;below300available use floor(2N/3)diagnostic capped200 and remainderreserve capped100. No exclusion relaxation.',
        diagnostic_n=diag_n,reserve_n=reserve_n,backbone=prior['backbone'],fairness=prior['fairness'],ours='Exact V31 fullprompt pointprefix runtime,V29keep_all_free,no learned gate. Sixfixedheads original3k/full s1/2/3,report all and means,predeclared main original3k_s1,no ensembling or seed choice.',
        mvp=prior['mvp'],inference='Each diagnostic query freshly inferred once per method,separate sequential processes,ours thenMVP,warmup oldalreadyexposedimage. No reserveforward,no labels loaded byworker. No training,thresholdsearch or promotion.',
        statistics='Primary paired difference original3k_s1 minusMVP;95%RGBbootstrap stratified by application_group;exacttwo-sideddiscordantbinomialtest;all6heads and2familymeans descriptive. Query=RGB here. Latency includesdecode,preprocess,allforwards,conditionalfallback,score;excludesfilebyte/hash,load,warmup,writes.',
        diagnosis='GT read only after both methods complete. Correct observed candidate but finalwrong=selectionloss;else no crop intersectsGT=cropmiss;else localdecodefailure. Report overlappingflags too. Observed candidates are actual native outputs plusmaterializedfallbackA32,not latent fullpooloracle. Additional trace collection has noVLMforwards.',
        limits='Subset of officialbenchmark after strict history+near filtering;not representative officialfull1581score. Diagnostic becomes development afterinspection;reserve remains untouched. Nativeprompts/hooks/device maps differ,per-viewpixelcap equal,totalworkmeasured.',
        hashes={str(q):base.sha(q) for q in sorted(set(inputs))},model_file_inventory=prior['model_file_inventory']))
    status('controller','registered',diagnostic_n=diag_n,reserve_n=reserve_n)

def verify():
    p=base.read(OUT/'protocol.json')
    for f,h in p['hashes'].items():assert base.sha(f)==h,f
    for e in p['model_file_inventory']:
        st=Path(e['path']).stat();assert st.st_size==e['bytes'] and st.st_mtime_ns==e['mtime_ns']
    return p

class TraceRuntime(fast.PrefixRuntime):
    def legacy_a(self,image,instruction,stages):
        points,scores=super().legacy_a(image,instruction,stages)
        self.observed_fallback=dict(points=points.tolist(),scores=scores.tolist())
        return points,scores
    def infer(self,image,instruction):
        self.observed_fallback=None
        output=super().infer(image,instruction);observed=[]
        for stage in output['stages']:
            if stage['part'] not in ['global','free','locked']:continue
            p=pixel(stage['point'],stage['image_size'],None if stage['part']=='global' else output['crop'])
            if p is not None:observed.append(dict(source=stage['part'],point=p))
        if self.observed_fallback:
            observed.extend(dict(source='fallback_A32',point=p) for p in self.observed_fallback['points'])
        output['observed_candidates']=observed;output['observed_fallback']=self.observed_fallback
        return output

def worker(method):
    spec=verify()
    from clickagent.advantage_runtimes import configure_paths,MVPAdapter
    configure_paths('mvp' if method=='mvp' else 'coordinate')
    import torch
    from PIL import Image
    torch.set_num_threads(4);folder=OUT/method
    status(method,'loading_model');started=time.perf_counter()
    rt=TraceRuntime(folder) if method=='ours' else MVPAdapter(folder,262144)
    if method=='mvp':base.instrument_mvp(rt,torch)
    base.sync(torch);write(folder/'runtime.json',dict(load_seconds=time.perf_counter()-started,metadata=rt.metadata,torch=torch.__version__,gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))
    def once(c,warm=False):
        payload=Path(c['image']).read_bytes();assert base.hashlib.sha256(payload).hexdigest()==c['image_sha256']
        if method=='mvp':rt.generation_records=[]
        before=rt.counter.snapshot();base.sync(torch);started=time.perf_counter()
        with Image.open(BytesIO(payload)) as im: image=im.convert('RGB')
        try:output=rt.infer(image,c['instruction'])
        finally:image.close()
        base.sync(torch);elapsed=time.perf_counter()-started;after=rt.counter.snapshot()
        assert output['ground_truth_in_model_input'] is False
        if method=='mvp':output.update(stages=rt.generation_records.copy(),generation_calls=len(rt.generation_records))
        row=dict(sample_id=c['sample_id'],method=method,warmup=warm,elapsed_seconds=elapsed,counter={k:after[k]-before[k] for k in before},output=output,image_sha256=c['image_sha256'])
        write(folder/('warmup' if warm else 'cases')/(base.key(c['sample_id'])+'.json'),row);return row
    for c in base.read(OUT/'data/warmup.json'):once(c,True)
    cases=base.read(OUT/'data/public.json');pred=[]
    for i,c in enumerate(cases,1):
        file=folder/'cases'/(base.key(c['sample_id'])+'.json')
        row=base.read(file) if file.exists() else once(c)
        assert row['sample_id']==c['sample_id'] and row['image_sha256']==c['image_sha256']
        pred.append(row);status(method,'inference',completed=i,total=len(cases),seconds=row['elapsed_seconds'],generation_calls=row['output'].get('generation_calls'))
    write(folder/'results.json',dict(complete=True,rows=pred));status(method,'complete',queries=len(pred))

def hit(point,c,label):
    return bool(point is not None and all(np.isfinite(point)) and point_in_geometry([point[0]/c['image_size'][0],point[1]/c['image_size'][1]],label))

def crop_overlap(box,c,label):
    w,h=c['image_size'];left,top,right,bottom=box
    fractions=[]
    for x0,y0,x1,y1 in label['boxes']:
        x0,x1=x0*w,x1*w;y0,y1=y0*h,y1*h
        area=max(0,min(right,x1)-max(left,x0))*max(0,min(bottom,y1)-max(top,y0))
        fractions.append(area/((x1-x0)*(y1-y0)))
    return max(fractions,default=0.)

def classify(final_hit,any_candidate,crop_fraction):
    if final_hit:return 'correct'
    if any_candidate:return 'selection_loss'
    if crop_fraction<=0:return 'crop_miss'
    return 'local_decode_failure'

def analyze():
    spec=verify();cases=base.read(OUT/'data/public.json');labels=base.read(OUT/'data/labels.json')
    ours=base.read(OUT/'ours/results.json');mvp=base.read(OUT/'mvp/results.json');assert ours['complete'] and mvp['complete']
    om={q['sample_id']:q for q in ours['rows']};mm={q['sample_id']:q for q in mvp['rows']};assert set(om)==set(mm)=={c['sample_id'] for c in cases}
    pred=[]
    for c in cases:
        sid=c['sample_id'];o=om[sid]['output'];m=mm[sid]['output'];label=labels[sid];assert label['type']=='bbox'
        head_hits=[hit(p,c,label) for p in o['seed_points']];mh=hit(m['point'],c,label)
        oc=o['observed_candidates'];ref=m.get('reference_result') or {};mc=[dict(source=p['stage'],point=[int(v) for v in p['point']]) for p in ref.get('all_predictions',[]) if p.get('point') is not None]
        ofrac=crop_overlap(o['crop']['box'],c,label)
        mfracs=[crop_overlap(p['region'],c,label) for p in ref.get('subimage_predictions',[]) if p.get('region')]
        oa=any(hit(p['point'],c,label) for p in oc);ma=any(hit(p['point'],c,label) for p in mc)
        pred.append(dict(sample_id=sid,application=c['application'],application_group=c['application_group'],ui_type=c['ui_type'],head_hits=head_hits,mvp_hit=mh,
            ours_diagnosis=dict(category=classify(head_hits[0],oa,ofrac),any_observed_candidate_correct=oa,crop_GT_fraction=ofrac,free_hit=next((hit(p['point'],c,label) for p in oc if p['source']=='free'),False),reason=o['reason']),
            mvp_diagnosis=dict(category=classify(mh,ma,max(mfracs,default=0.)),any_observed_candidate_correct=ma,crop_GT_fractions=mfracs),
            ours_seconds=om[sid]['elapsed_seconds'],mvp_seconds=mm[sid]['elapsed_seconds']))
    y=np.array([q['head_hits'] for q in pred],int);z=np.array([q['mvp_hit'] for q in pred],int);diff=y[:,0]-z
    rng=np.random.default_rng(20261132);samples=np.empty(10000);strata=defaultdict(list)
    for i,c in enumerate(cases):strata[c['application_group']].append(i)
    for b in range(len(samples)):
        ix=np.concatenate([rng.choice(v,len(v),replace=True) for v in strata.values()]);samples[b]=diff[ix].mean()
    wins=int((diff==1).sum());losses=int((diff==-1).sum());nd=wins+losses
    from scipy.stats import binomtest
    result=dict(complete=True,n=len(cases),head_hits={name:int(y[:,i].sum()) for i,name in enumerate(base.NAMES)},head_accuracy_percent={name:100*float(y[:,i].mean()) for i,name in enumerate(base.NAMES)},
        family_mean_percent=dict(original3k=100*float(y[:,:3].mean()),full=100*float(y[:,3:].mean())),mvp_hits=int(z.sum()),mvp_accuracy_percent=100*float(z.mean()),
        primary=dict(head='original3k_s1',wins=wins,losses=losses,both_correct=int(((y[:,0]==1)&(z==1)).sum()),both_wrong=int(((y[:,0]==0)&(z==0)).sum()),delta_pp=100*float(diff.mean()),ci95_pp=(100*np.quantile(samples,[.025,.975])).tolist(),exact_two_sided_p=binomtest(wins,nd,.5).pvalue if nd else 1.),
        latency={m:dict(mean_seconds=float(np.mean([q[m+'_seconds'] for q in pred])),median_seconds=float(np.median([q[m+'_seconds'] for q in pred]))) for m in ['ours','mvp']},
        diagnosis={m:dict(Counter(q[m+'_diagnosis']['category'] for q in pred)) for m in ['ours','mvp']},reserve_queries_not_run=spec['reserve_n'],mainline_unchanged=True)
    result['by_application_group']={g:dict(n=len(ix),ours_hits=int(y[ix,0].sum()),mvp_hits=int(z[ix].sum())) for g,ix in strata.items()}
    write(OUT/'predictions.json',dict(rows=pred));write(OUT/'results.json',result)
    lines=['V32：固定提速版本与MVP的严格未见截图子集对比',f'诊断{len(cases)}张独立RGB/查询；另保留{spec["reserve_n"]}张未运行截图。不是官方1581题整榜；诊断样本查看后转为开发证据。',
        '固定完整提示+目标/坐标早停，V29原生局部点保留，所有模型冻结，训练和调参均为0。',
        '全部固定评分头命中：'+str(result['head_hits']), '两系列均值%：'+str(result['family_mean_percent']),f'MVP命中：{int(z.sum())}/{len(cases)}，{result["mvp_accuracy_percent"]:.2f}%。',
        '预定主头original3k_s1配对比较：'+str(result['primary']),'单题耗时：'+str(result['latency']),'失败归因：'+str(result['diagnosis']),
        '正确候选指实际已生成的原生点及实际回退时构造的A32；未额外生成潜在候选，因此不是完整候选器的覆盖上限。',
        '错误分类按正确候选被漏选、裁剪与GT完全不相交、可见但局部未定位的顺序互斥归类。裁剪含部分目标也算相交，另保存目标覆盖比例。',
        '双方推理不读取真实GT；MVP原记录中的dummy bbox正确性字段不用于评价。所有准确率在双方输出完成后独立按正式归一化目标区域计算。']
    (OUT/'readout.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    for method in ['ours','mvp']:assert not (OUT/method/'reserve').exists()
    write(OUT/'verification.json',dict(complete=True,at_utc=base.now(),protocol_hashes_verified=len(spec['hashes']),paired_queries=len(pred),head_decisions=len(pred)*6,reserve_not_run=True,mainline_unchanged=True))
    write(OUT/'completion.json',dict(complete=True,at_utc=base.now(),results_sha256=base.sha(OUT/'results.json'),readout_sha256=base.sha(OUT/'readout.txt'),verification_sha256=base.sha(OUT/'verification.json')))
    reg=base.read(ROOT/'CURRENT_RESEARCH.json');assert reg['run_id']=='quality_readout_v15_3k';reg['active_experiment'].update(status='complete',phase='analyzed_and_verified',results_path=str(OUT/'readout.txt'));write(ROOT/'CURRENT_RESEARCH.json',reg)
    status('controller','complete',results_path=str(OUT/'readout.txt'));print('\n'.join(lines),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','ours','mvp','analyze','all']);args=parser.parse_args()
    if args.stage=='prepare':prepare();return
    if args.stage in ['ours','mvp']:worker(args.stage);return
    if args.stage=='analyze':analyze();return
    spec=verify();reg=base.read(ROOT/'CURRENT_RESEARCH.json');reg['active_experiment']=dict(run_id='independent_diagnostic_v32',status='running',phase='frozen_paired_inference',protocol_path=str(OUT/'protocol.json'),status_path=str(OUT/'controller/status.json'),mainline_unchanged=True);write(ROOT/'CURRENT_RESEARCH.json',reg)
    for method in ['ours','mvp']:
        folder=OUT/method;folder.mkdir(parents=True,exist_ok=True)
        if (folder/'results.json').exists() and base.read(folder/'results.json')['complete']:continue
        for _ in range(30):
            p=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True)
            if p.returncode==0 and max(int(s) for s in p.stdout.splitlines() if s.strip())<3000:break
            time.sleep(5)
        else:raise RuntimeError('GPU busy;will not start duplicate backbone')
        status('controller','running_'+method,total=spec['diagnostic_n'])
        with (folder/'stdout.log').open('a',encoding='utf-8') as stdout,(folder/'stderr.log').open('a',encoding='utf-8') as stderr:
            child=subprocess.run([sys.executable,'-B','-X','utf8','-u',str(Path(__file__).resolve()),method],cwd=ROOT,stdout=stdout,stderr=stderr)
        assert child.returncode==0,f'{method} failed;see stderr.log'
    analyze()

if __name__=='__main__':
    try:main()
    except BaseException as error:
        status('controller','failed',error=repr(error),traceback=traceback.format_exc());raise
