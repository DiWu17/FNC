"""V33 matched-cost supplementary regions; fixed full prompt and frozen models."""
import argparse
from collections import Counter,defaultdict
import csv
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_independent_diagnostic_v32 as prior
import run_reinspection_v31 as cache
import run_point_prefix_latency_v31 as fast
from clickagent.observation_regions_v33 import ARMS,coordinate_distribution,plan_regions
from clickagent.reinspection_v31 import proposal_observation,plan_diverse,consensus
from clickagent.grounded_composition_v28 import GLOBAL_PROMPT,LOCAL_PROMPT,pixel
import numpy as np
from PIL import Image
OUT=ROOT/'runs/observation_regions_v33';V28=ROOT/'runs/grounded_composition_v28';V32=prior.OUT
read=prior.base.read;write=prior.write;sha=prior.base.sha;key=prior.base.key

def status(phase,**fields):
    v=dict(phase=phase,updated_at_utc=prior.base.now(),pid=prior.os.getpid(),complete=phase=='complete',**fields)
    write(OUT/'status.json',v);print(prior.json.dumps(v,ensure_ascii=False,default=prior.base.default),flush=True)

def prepare():
    if (OUT/'protocol.json').exists():return verify()
    OUT.mkdir(parents=True,exist_ok=True)
    p=prior.verify();old=cache.read(V28/'data/public.json');train=[c for c in old if c['cohort']=='train'];dev=read(V32/'data/public.json');assert len(train)==112 and len(dev)==115
    public_keys=['sample_id','dataset','image','image_size','instruction','image_sha256']
    cases=[{k:c[k] for k in public_keys}|dict(cohort='train',source=c['source']) for c in train]+[dict(c,cohort='development',source=c['application_group']) for c in dev]
    write(OUT/'data/public.json',cases)
    old_labels=read(V28/'data/labels.json');write(OUT/'data/train_labels.json',{c['sample_id']:old_labels[c['sample_id']] for c in train})
    write(OUT/'data/development_labels.json',read(V32/'data/labels.json'))
    baseline={};inputs=[]
    for c in train:
        gp=V28/'raw/global'/(key(c['sample_id'])+'.npz');bp=V28/'grounded'/(key(c['sample_id'])+'.npz');inputs.extend([gp,bp])
        _,gm=cache.load(gp);_,bm=cache.load(bp);native=bm['native_points']
        baseline[c['sample_id']]=dict(first_crop=gm['planned_local_transform'],point=native['free'],global_point=native['global'],global_archive=str(gp),cohort='train',native_points=native)
    old_rows=read(V32/'ours/results.json')['rows'];assert len(old_rows)==len(dev)
    for r in old_rows:
        o=r['output'];g=next(s for s in o['stages'] if s['part']=='global')
        baseline[r['sample_id']]=dict(first_crop=o['crop'],point=o['point'],seed_points=o['seed_points'],global_reference=g,cohort='development',elapsed_seconds=r['elapsed_seconds'])
    write(OUT/'data/baseline.json',baseline)
    tests=subprocess.run([sys.executable,'-B','-X','utf8','-m','unittest','discover','-s','tests','-p','test_observation_regions_v33.py','-v'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    write(OUT/'unit_tests.json',dict(exit_code=tests.returncode,output=tests.stdout+tests.stderr));assert tests.returncode==0,tests.stdout+tests.stderr
    files=[Path(__file__),ROOT/'src/clickagent/observation_regions_v33.py',ROOT/'tests/test_observation_regions_v33.py',OUT/'unit_tests.json',V32/'protocol.json',V32/'ours/results.json',V32/'mvp/results.json',V32/'final_completion.json',V28/'data/public.json',V28/'data/labels.json',*inputs,*map(Path,p['hashes']),*list((OUT/'data').glob('*.json'))]
    write(OUT/'protocol.json',dict(registered_at_utc=prior.base.now(),authorization='继续',purpose='Improve where to observe on the unchanged fullprompt+causalprefix baseline. Matched one-extra-local-view mechanism study,no scorer/backbone training,not independent final confirmation.',
        data='112existingTRAIN RGB used only for cached-global geometric audit,no local TRAIN generation or fittedparameters. Reuse115V32RGB as now-observed development cohort.58reserveRGB neverforwarded. No new independent test.',
        arms=list(ARMS),algorithm='nearby:640pxcrop at baselinepoint;name_fullpool:softmax(zscore)overALLunique frozen-own-proposerpoints,not top64;coordinate_mass:equal normalized96head attention over actual generated p-value own-forwardtokens. Guidedarms share all learnedpoints+patchcenter anchors,exact crop geometry,IoU<=.25preference,andmass*(1-IoU). Random hashes the SAMEunique eligibleboxes. Missing name/coordinate evidence falls backtomatchedrandom,never dropscase. No hyperparametersearch.',
        coordinate_indices='Actual incremental decoding character-span mapping uses zero-based content indices and matching all_attention rows;stored indices additionally1based. Both x and y and comma insidecomplete p field. No retokenized answer,teacherforcedprefix,or fabricatedtoken.',
        inference='OriginalGLOBAL_PROMPT160token ceiling with actual t/p prefixstop;LOCAL_PROMPT56 unchanged. Frozen8B/ownproposer/all6oldheads unchanged;no new head. TRAIN uses actual V28 full-global cached states before referencegeneration. DEVglobalprefix freshly rerun and must match V32point,target,text exactly before newcrops. Same input and per-view262144pixelbudget.4localarms separately executed even when cropboxesidentical.',
        primary='All115development,3pairedcomparisons with98.333%RGB/source-stratifiedbootstrap:coordinate_mass candidateunion minus name_fullpool,nearby,random_matched. Candidateunion uses unchanged V32primarypoint OR one extraactualpoint. GToracle is diagnostic,not deployment accuracy. No adaptivearmselection.',
        secondary='Extra view validpoint/hit,GTcropintersection/fullcoverage,repairavailability on38/115baselinehits,potential harm if alwaysreplace. Exact-point3candidateconsensus(global,unchangedbase,extra),14pixel complete-link with basepreferredtie;reportall6oldheads andfamilymeans. No learnedreplacementrule. ExistingMVP115predictions only historical same-development reference,unequal viewbudget.',
        geometry='Same640original-pixelmaximumcrop,2xresize;coordinateandnameguidedarms shareanchorboxes;top64V31planner included only as geometryaudit,not extraVLMarm. AllGTcropmetrics computed outside planner.',
        costs='Record actualglobal and4local calls,counters,tokens andseconds. Baseline observations shared/cached and newglobal replay duplicates an existing observation;sum workload is not separatelymeasuredendtoendlatency. No speedadvantage claim.',
        hashes={str(f):sha(f) for f in sorted(set(files))},model_file_inventory=p['model_file_inventory']))
    status('registered',training_RGB=112,development_RGB=115,reserve_unrun=58,new_development_generation_calls=575)

def verify():
    p=read(OUT/'protocol.json')
    for f,h in p['hashes'].items():assert sha(f)==h,f
    for e in p['model_file_inventory']:
        s=Path(e['path']).stat();assert s.st_size==e['bytes'] and s.st_mtime_ns==e['mtime_ns']
    return p

def make_plans(c,arrays,meta,proposer):
    from clickagent.crossfit_proposer import predict_raw
    baseline=read(OUT/'data/baseline.json')[c['sample_id']]
    raw,m=proposal_observation(arrays,meta);proposal=predict_raw(proposer,raw,m,c['image_size'],64)
    probability,coord=coordinate_distribution(arrays,meta);started=time.perf_counter()
    plans=plan_regions(c['image_size'],meta['grid_shape'],baseline['first_crop'],baseline['point'],proposal,probability,c['sample_id'])
    old,oldinfo=plan_diverse(proposal['all_points'],proposal['all_scores'],c['image_size'],baseline['first_crop']['box'])
    value=dict(sample_id=c['sample_id'],cohort=c['cohort'],plans=plans,top64_geometry_control=dict(transform=old,metadata=oldinfo),proposal=proposal,coordinate_evidence=coord,coordinate_probability=probability.tolist(),planner_seconds=time.perf_counter()-started,ground_truth_used=False)
    write(OUT/'plans'/(key(c['sample_id'])+'.json'),value);return value

def geometry_train():
    verify()
    from clickagent.crossfit_proposer import load_crossfit_proposer
    model=load_crossfit_proposer(ROOT/'runs/crossfit_candidates_v12/proposers/full.pt','cpu')[0].eval().requires_grad_(False)
    cases=[c for c in read(OUT/'data/public.json') if c['cohort']=='train'];baselines=read(OUT/'data/baseline.json')
    for i,c in enumerate(cases,1):
        dest=OUT/'plans'/(key(c['sample_id'])+'.json')
        if not dest.exists():
            arrays,meta=cache.load(baselines[c['sample_id']]['global_archive']);make_plans(c,arrays,meta,model)
        status('training_geometry',completed=i,total=len(cases))
    labels=read(OUT/'data/train_labels.json');summary={}
    for arm in [*ARMS,'top64_geometry_control']:
        fractions=[]
        for c in cases:
            p=read(OUT/'plans'/(key(c['sample_id'])+'.json'));tr=p[arm]['transform'] if arm=='top64_geometry_control' else p['plans'][arm]['transform']
            fractions.append(prior.crop_overlap(tr['box'],c,labels[c['sample_id']]))
        summary[arm]=dict(intersects=sum(f>0 for f in fractions),full_ge99pct=sum(f>=.99 for f in fractions),n=len(cases))
    write(OUT/'training_geometry.json',dict(complete=True,arms=summary,no_fit=True,no_development_labels_used=True));status('training_geometry_complete',arms=summary)

def collect():
    verify()
    from clickagent.advantage_runtimes import configure_paths
    configure_paths('coordinate')
    import torch
    torch.set_num_threads(4);status('loading_model');rt=fast.PrefixRuntime(OUT/'runtime')
    write(OUT/'runtime.json',dict(metadata=rt.metadata,torch=torch.__version__,gpus=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))
    cases=[c for c in read(OUT/'data/public.json') if c['cohort']=='development'];baselines=read(OUT/'data/baseline.json')
    def generate(im,ins,prompt,cap):
        before=rt.counter.snapshot();a,m=rt.runtime.infer(im,ins,prompt,cap);after=rt.counter.snapshot();m['counter']={k:after[k]-before[k] for k in before};return a,m
    for i,c in enumerate(cases,1):
        assert sha(c['image'])==c['image_sha256'];b=baselines[c['sample_id']]
        with Image.open(c['image']) as opened:image=opened.convert('RGB')
        try:
            gp=OUT/'raw/global'/(key(c['sample_id'])+'.npz')
            if not gp.exists():
                a,m=generate(image,c['instruction'],GLOBAL_PROMPT,160);m.update(sample_id=c['sample_id'],image_sha256=c['image_sha256']);cache.npz(gp,a,m)
            a,m=cache.load(gp);ref=b['global_reference']
            assert m['point']==ref['point'] and m['target']==ref['target'] and m['generated_text']==ref['text'],('Globalparity',c['sample_id'])
            pp=OUT/'plans'/(key(c['sample_id'])+'.json');plan=read(pp) if pp.exists() else make_plans(c,a,m,rt.proposer);del a
            for arm in ARMS:
                dest=OUT/'raw'/arm/(key(c['sample_id'])+'.npz')
                if not dest.exists():
                    tr=plan['plans'][arm]['transform']
                    with image.crop(tr['box']).resize(tuple(tr['crop_size']),Image.Resampling.LANCZOS) as local:
                        a,lm=generate(local,c['instruction'],LOCAL_PROMPT,56)
                    lm.update(sample_id=c['sample_id'],arm=arm,transform=tr,world_point=pixel(lm['point'],lm['image_size'],tr),image_sha256=c['image_sha256'],ground_truth_in_model_input=False)
                    cache.npz(dest,{},lm);del a
                status('development_observations',completed=i-1,total=len(cases),arm=arm,current_case=i)
            status('development_observations',completed=i,total=len(cases),arm='case_complete')
        finally:image.close()
    status('observations_complete',completed=len(cases),total=len(cases))

def analyze():
    spec=verify();cases=[c for c in read(OUT/'data/public.json') if c['cohort']=='development'];labels=read(OUT/'data/development_labels.json');baselines=read(OUT/'data/baseline.json');rows=[];parity=0
    for c in cases:
        sid=c['sample_id'];b=baselines[sid];_,gm=cache.load(OUT/'raw/global'/(key(sid)+'.npz'));ref=b['global_reference'];assert gm['point']==ref['point'] and gm['target']==ref['target'] and gm['generated_text']==ref['text'];parity+=1
        gp=pixel(gm['point'],c['image_size']);plan=read(OUT/'plans'/(key(sid)+'.json'));base_hits=[prior.hit(p,c,labels[sid]) for p in b['seed_points']];arms={}
        for arm in ARMS:
            _,m=cache.load(OUT/'raw'/arm/(key(sid)+'.npz'));point=m['world_point'];assert point==pixel(m['point'],m['image_size'],m['transform']);assert m['transform']==plan['plans'][arm]['transform'];h=prior.hit(point,c,labels[sid])
            selected=[consensus([gp,base,point],base,14) for base in b['seed_points']];sh=[prior.hit(p,c,labels[sid]) for p in selected]
            arms[arm]=dict(point=point,extra_hit=h,extra_valid=point is not None,union_hits=[bool(v or h) for v in base_hits],consensus_points=selected,consensus_hits=sh,
                crop_GT_fraction=prior.crop_overlap(m['transform']['box'],c,labels[sid]),additional_seconds=m['elapsed_seconds'],tokens=len(m['generated_token_ids']),counter=m['counter'],coordinate_fallback=plan['plans'][arm].get('fallback_to_matched_random',False))
        rows.append(dict(sample_id=sid,source=c['source'],base_points=b['seed_points'],base_hits=base_hits,arms=arms,coordinate_available=plan['coordinate_evidence']['available'],baseline_seconds=b['elapsed_seconds']))
    n=len(rows);base=np.array([q['base_hits'][0] for q in rows],bool);rng=np.random.default_rng(20261133);groups=defaultdict(list)
    for i,q in enumerate(rows):groups[q['source']].append(i)
    bootstrap=np.zeros((10000,n))
    for b in range(len(bootstrap)):
        ix=np.concatenate([rng.choice(v,len(v),replace=True) for v in groups.values()]);bootstrap[b]=np.bincount(ix,minlength=n)/n
    summaries={}
    for arm in ARMS:
        extra=np.array([q['arms'][arm]['extra_hit'] for q in rows],bool);fractions=np.array([q['arms'][arm]['crop_GT_fraction'] for q in rows]);cons=np.array([q['arms'][arm]['consensus_hits'] for q in rows],bool)
        summaries[arm]=dict(extra_hits=int(extra.sum()),baseline_plus_extra_oracle_hits=int((base|extra).sum()),extra_repairs_available=int((~base&extra).sum()),potential_harms_if_replace_all=int((base&~extra).sum()),
            valid_points=sum(q['arms'][arm]['extra_valid'] for q in rows),crop_intersects=int((fractions>0).sum()),crop_full_ge99pct=int((fractions>=.99).sum()),
            consensus_head_hits={name:int(cons[:,j].sum()) for j,name in enumerate(prior.base.NAMES)},consensus_family_mean_percent=dict(original3k=100*float(cons[:,:3].mean()),full=100*float(cons[:,3:].mean())),
            consensus_primary_repairs=int((~base&cons[:,0]).sum()),consensus_primary_harms=int((base&~cons[:,0]).sum()),mean_extra_seconds=float(np.mean([q['arms'][arm]['additional_seconds'] for q in rows])),fallback_random_cases=sum(q['arms'][arm]['coordinate_fallback'] for q in rows))
    comparisons={};co=np.array([q['arms']['coordinate_mass']['union_hits'][0] for q in rows],int)
    for control in ['name_fullpool','nearby','random_matched']:
        diff=co-np.array([q['arms'][control]['union_hits'][0] for q in rows],int);alpha=.05/3/2;ci=(100*np.quantile(bootstrap@diff,[alpha,1-alpha])).tolist()
        comparisons['coordinate-'+control]=dict(delta_pp=100*float(diff.mean()),ci_pp=ci,level=1-.05/3,positive=bool(ci[0]>0))
    mvp=read(V32/'results.json');result=dict(complete=True,n=n,baseline_primary_hits=int(base.sum()),arms=summaries,primary_comparisons=comparisons,coordinate_evidence_available=sum(q['coordinate_available'] for q in rows),historical_same_development_MVP_hits=mvp['mvp_hits'],training_geometry=read(OUT/'training_geometry.json'),reserve_unrun=58,mainline_unchanged=True)
    write(OUT/'predictions.json',dict(rows=rows));write(OUT/'results.json',result)
    lines=['V33：完整提示早停下，补充观察区域的同预算对照','112训练图仅复用历史全图状态做几何检查，无新训练、无参数拟合；115张V32已观察截图作为开发，58张保留图未运行。',
        f'固定V32主头{int(base.sum())}/{n}；每个分支额外执行一次相同640像素/2倍局部生成。新增全图重放与V32目标、坐标、原始文本逐项一致。',
        '四分支：'+str(summaries),'预登记三比较，98.333%区间，候选并集诊断：'+str(comparisons),
        f'有效坐标字段注意力{result["coordinate_evidence_available"]}/{n}；缺失按登记随机回退并保留样本。',
        '候选并集使用GT判是否任一点正确，是诊断可达率，不是已实现准确率。实际可部署对照是固定14像素完整链接共识，原生基线点优先平局，全部六头分别报告。',
        f'同115开发图历史MVP={mvp["mvp_hits"]}；本轮预算不同，不据此宣称独立优越性或速度优势。共享缓存视图的耗时不能视为完整系统独立计时。',
        '不读取58张保留图进行推理，不按开发分挑分支或评分头，不自动替换采用模型。']
    (OUT/'readout.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    verify();write(OUT/'verification.json',dict(complete=True,at_utc=prior.base.now(),global_text_target_point_parity=parity,local_coordinate_replays=n*len(ARMS),source_hashes_verified=len(spec['hashes']),reserve_unrun=True,mainline_unchanged=True))
    write(OUT/'completion.json',dict(complete=True,at_utc=prior.base.now(),results_sha256=sha(OUT/'results.json'),readout_sha256=sha(OUT/'readout.txt'),verification_sha256=sha(OUT/'verification.json')))
    reg=read(ROOT/'CURRENT_RESEARCH.json');assert reg['run_id']=='quality_readout_v15_3k';reg['active_experiment'].update(status='complete',phase='analyzed_and_verified',results_path=str(OUT/'readout.txt'));write(ROOT/'CURRENT_RESEARCH.json',reg);status('complete',results_path=str(OUT/'readout.txt'));print('\n'.join(lines),flush=True)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['prepare','geometry','collect','analyze','all']);args=parser.parse_args()
    if args.stage=='prepare':prepare();return
    if args.stage=='geometry':geometry_train();return
    if args.stage=='collect':collect();return
    if args.stage=='analyze':analyze();return
    verify();reg=read(ROOT/'CURRENT_RESEARCH.json');reg['active_experiment']=dict(run_id='observation_regions_v33',status='running',phase='matched_cost_regions',protocol_path=str(OUT/'protocol.json'),status_path=str(OUT/'status.json'),mainline_unchanged=True);write(ROOT/'CURRENT_RESEARCH.json',reg)
    if not (OUT/'training_geometry.json').exists():geometry_train()
    collect();analyze()

if __name__=='__main__':
    try:main()
    except BaseException as error:status('failed',error=repr(error),traceback=traceback.format_exc());raise
