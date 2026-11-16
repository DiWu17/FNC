"""Train-only same-RGB/same-target-text counterfactual region matching.

The two candidates are ORIGINAL ANNOTATION region centres. This is an oracle
candidate diagnostic, never a deployment grounding score or an added dev point.
"""
import os
for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ.setdefault(k,'4')
import sys,json,re,itertools,traceback
from pathlib import Path
from collections import defaultdict
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import run_grounded_composition_v28 as run
from run_visual_feedback_v23 import read,write,sha,hit
from clickagent.grounded_composition_v28 import visual_features,fit,infer,identity
OUT=ROOT/'runs/grounded_composition_v28_paired_queries';BIG=ROOT/'runs/grounded_composition_v28_representation3k'

def status(phase,**fields):
    row=dict(phase=phase,updated_at_utc=run.now(),complete=phase=='complete',**fields);write(OUT/'status.json',row);print(row,flush=True)
def normalized(text):return re.sub(r'\s+',' ',str(text).casefold()).strip(' \t\r\n\"\'')
def metadata(path):
    with np.load(path,allow_pickle=False) as z:return json.loads(str(z['metadata']))

def prepare():
    assert not (OUT/'plan.json').exists()
    paths=[Path(__file__),ROOT/'src/clickagent/grounded_composition_v28.py',BIG/'data/public.json',BIG/'data/indices.json',BIG/'protocol.json']
    write(OUT/'plan.json',dict(registered_at_utc=run.now(),source_sha256=sha(Path(__file__)),input_sha256={str(p):sha(p) for p in paths},
        data='Original2973 TRAINING queries only, no V28dev labels/predictions. Group sameRGB/source AND exactly normalized generatedglobaltargettext(nonNONE,complete). Different instructions, disjoint annotated target-region centres. One pair/RGB, fixedSHA choice. Generated name equality does not assert both visible labels were human-verified.',
        candidate_regime='Oracle two ORIGINAL ANNOTATED region centres, same two alternatives for both questions, opposite labels. This tests conditional region discrimination given candidates, NOT end-to-end grounding accuracy. No annotation enters VLM prompts, no VLM calls; mainV28 pools/models/results unchanged.',
        features='Same frozen image encoding for both queries, each own actualname-token96head attention. SameV28centered576 globalfeatures; optional smallest EXACTLYSHARED cached crop containing both candidates, separate576features, otherwisezero. Compare global-only vs shared-local vs query-collapsed(mean ofbothqueries) control. Keep1152-dimensional lineararchitecture fixed, base0,ridge.3, balanced pairloss. Query-collapsed input cannot solve both labels; expected50% question/0%both without query distinction.',
        evaluation='3 source-stratified RGB folds, both questions alwaysheldtogether. Also leave-source-out onlyif>=2sources. Fixeduntrained equal-head all-token-mean baseline. Report questionaccuracy and both-questions-correct pairs, aliasing ofcandidatefeatures; small/ambiguous pairing explicitly disclosed. No hyperparameter/model selection, no comparison of these oracle scores againstMVP grounding.',new_vlm_calls=0,mainline_unchanged=True))

def mine():
    cc=[c for c in read(BIG/'data/public.json') if c['cohort']=='train'];lookup={}
    for file in [ROOT/'runs/multiregion_v13/data/train.jsonl',ROOT/'runs/quality_readout_v15_3k/data/new_train.jsonl']:
        for line in file.read_text(encoding='utf-8-sig').splitlines():
            c=json.loads(line);lookup[c['sample_id']]=c['target_geometry']
    grouped=defaultdict(list);excluded=defaultdict(int)
    for i,c in enumerate(cc,1):
        m=metadata(c['views'][0]['feature_path']);name=normalized(m.get('target_text',''))
        if not m.get('target_available',False) or not name or name in ['none','unknown','n/a']:excluded['unavailable_name']+=1;continue
        geom=lookup[c['sample_id']]
        if geom['type']!='bbox':excluded['not_bbox']+=1;continue
        box=max(geom['boxes'],key=lambda b:(b[2]-b[0])*(b[3]-b[1]));p=(np.asarray(box[:2])+np.asarray(box[2:]))/2*np.asarray(c['image_size'])
        if not hit(p.tolist(),c,geom):excluded['invalid_centre']+=1;continue
        grouped[c['group'],c['source'],name].append(dict(case=c,geometry=geom,point=p.tolist(),name=name))
        if i%300==0:status('mining_training_pairs',completed=i,total=len(cc))
    possible=defaultdict(list)
    for (rgb,source,name),rows in grouped.items():
        for a,b in itertools.combinations(rows,2):
            ca,cb=a['case'],b['case']
            if ca['image_size']!=cb['image_size'] or normalized(ca['instruction'])==normalized(cb['instruction']):continue
            if hit(a['point'],cb,b['geometry']) or hit(b['point'],ca,a['geometry']):continue
            if np.linalg.norm((np.asarray(a['point'])-b['point'])/ca['image_size'])<.02:continue
            aa,bb=sorted([a,b],key=lambda x:x['case']['sample_id']);possible[rgb].append(dict(a=aa,b=bb,group=rgb,source=source,name=name))
    pairs=[min(rows,key=lambda p:run.digest([p['a']['case']['sample_id'],p['b']['case']['sample_id'],'v28_pair'])) for rows in possible.values()]
    pairs.sort(key=lambda p:run.digest([p['group'],'v28_pair_order']));write(OUT/'pairs.json',dict(pairs=pairs,excluded=dict(excluded),oracle_candidates=True,original_training_queries=len(cc)))
    status('paired_data_mined',rgb_pairs=len(pairs),queries=2*len(pairs),sources=sorted({p['source'] for p in pairs}));return pairs

def read_view(view):
    assert sha(view['feature_path'])==view['feature_sha256']
    with np.load(view['feature_path'],allow_pickle=False) as z:return z['target_attention'],z['vision_hidden'],json.loads(str(z['metadata']))
def signature(v):return json.dumps({k:v['transform'][k] for k in ['box','scale','crop_size','original_size']},sort_keys=True)
def features(pairs):
    results=[]
    for i,pair in enumerate(pairs,1):
        ca,cb=pair['a']['case'],pair['b']['case'];points=np.asarray([pair['a']['point'],pair['b']['point']]);size=ca['image_size']
        aa,va,ma=read_view(ca['views'][0]);ab,vb,mb=read_view(cb['views'][0])
        if ma['grid_shape']!=mb['grid_shape']:continue
        ga,_=visual_features(aa,va,ma['grid_shape'],points,size);gb,_=visual_features(ab,va,mb['grid_shape'],points,size)
        la=np.zeros_like(ga);lb=np.zeros_like(gb);shared=None
        other={signature(v):v for v in cb['views'][1:]};common=[]
        for v in ca['views'][1:]:
            key=signature(v)
            if key not in other:continue
            box=np.asarray(v['transform']['box'])
            if ((points>=box[:2])&(points<box[2:])).all():common.append((int(np.prod(box[2:]-box[:2])),v,other[key]))
        if common:
            _,a,b=min(common,key=lambda x:x[0]);aa,vv,ma=read_view(a);ab,_,mb=read_view(b)
            if ma['grid_shape']==mb['grid_shape']:
                la,_=visual_features(aa,vv,ma['grid_shape'],points,size,a['transform']);lb,_=visual_features(ab,vv,mb['grid_shape'],points,size,b['transform']);shared=a['transform']
        full=np.stack([np.c_[ga,la],np.c_[gb,lb]]);glob=full.copy();glob[:,:,576:]=0;collapsed=np.repeat(full.mean(0,keepdims=True),2,axis=0)
        row=dict(pair=pair,global_alias=bool(np.max(np.abs(ga[0]-ga[1]))<1e-7 and np.max(np.abs(gb[0]-gb[1]))<1e-7),shared_alias=bool(np.max(np.abs(full[0,0]-full[0,1]))<1e-7 and np.max(np.abs(full[1,0]-full[1,1]))<1e-7),shared=shared)
        path=OUT/f'features/{run.key(pair["group"])}.npz';run.npz(path,global_only=glob,shared=full,collapsed=collapsed,metadata=np.asarray(json.dumps(dict(group=pair['group'],oracle_candidate_diagnostic=True))))
        row['features_path']=str(path);row['features_sha256']=sha(path);results.append(row);status('paired_features',completed=i,total=len(pairs))
    write(OUT/'features.json',results);return results

def records(rows,mode):
    rr=[]
    for row in rows:
        x=run.load_npz(row['features_path'])[0][mode];p=row['pair']
        for j in [0,1]:rr.append(dict(sample_id=p['group']+str(j),source=p['source'],group=p['group'],x=x[j].astype(np.float64),base=np.zeros(2),y=np.asarray([j==0,j==1]),points=np.asarray([p['a']['point'],p['b']['point']]),size=np.asarray(p['a']['case']['image_size'])))
    return rr
def predictions(model,rr):return [dict(sample_id=r['sample_id'],group=r['group'],hit=bool(r['y'][int(infer(model,r).argmax())])) for r in rr]
def accuracy(rows):
    gg=defaultdict(list)
    for r in rows:gg[r['group']].append(r['hit'])
    return dict(queries=len(rows),rgb_pairs=len(gg),question_percent=100*float(np.mean([r['hit'] for r in rows])) if rows else None,both_correct_percent=100*float(np.mean([all(v) for v in gg.values()])) if gg else None)

def evaluate(rows):
    summaries={};all_rows=[];assignment={};sources=sorted({r['pair']['source'] for r in rows})
    for source in sources:
        for j,r in enumerate(sorted([r for r in rows if r['pair']['source']==source],key=lambda r:run.digest([r['pair']['group'],'paired_fold']))):assignment[r['pair']['group']]=j%3
    for mode in ['global_only','shared','collapsed']:
        rr=records(rows,mode);held=[];by_source=[]
        for fold in range(3):
            fitting=[r for r in rr if assignment[r['group']]!=fold];testing=[r for r in rr if assignment[r['group']]==fold]
            model=fit(fitting,.3,True) if fitting else identity(1152);held+=predictions(model,testing)
        if len(sources)>=2:
            for source in sources:
                model=fit([r for r in rr if r['source']!=source],.3,True);by_source+=predictions(model,[r for r in rr if r['source']==source])
        model=fit(rr,.3,True) if rr else identity(1152);train=predictions(model,rr)
        run.npz(OUT/f'models/{mode}.npz',weight=model['weight'],mean=model['mean'],std=model['std'],metadata=np.asarray(json.dumps(dict(mode=mode,oracle_diagnostic_only=True))))
        summaries[mode]=dict(train=accuracy(train),RGB_OOF=accuracy(held),source_OOF=accuracy(by_source),weight_norm=float(np.linalg.norm(model['weight'])));all_rows.append(dict(mode=mode,held=held,source_held=by_source))
        status('paired_probe_fitted',mode=mode,result=summaries[mode])
    result=dict(oracle_candidate_diagnostic=True,NOT_main_grounding_accuracy=True,new_vlm_calls=0,paired_rgb=len(rows),sources=sources,shared_local_pairs=sum(r['shared'] is not None for r in rows),global_alias_pairs=sum(r['global_alias'] for r in rows),shared_alias_pairs=sum(r['shared_alias'] for r in rows),summaries=summaries)
    write(OUT/'results.json',result);write(OUT/'predictions.json',all_rows)
    (OUT/'readout.txt').write_text('同图、同模型目标名称、不同指令/标注区域的二选一诊断。\n候选是两个原标注区域的中心；这是给定候选后的区域辨别，不是端到端定位，不与MVP主榜准确率比较。\n同图两问题一起留出；query-collapsed控制不能同时解对两问题。名称来自模型，未包装成人工核验的可见文字标签。\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    if not (OUT/'plan.json').exists():prepare()
    plan=read(OUT/'plan.json');assert plan['source_sha256']==sha(Path(__file__))
    for p,h in plan['input_sha256'].items():assert sha(p)==h,p
    rows=features(mine());evaluate(rows)
    paths=[p for p in OUT.rglob('*') if p.is_file() and p.name not in ['status.json','completion.json']]
    write(OUT/'completion.json',dict(complete=True,source_sha256=sha(Path(__file__)),artifact_sha256={str(p):sha(p) for p in paths},new_vlm_calls=0));status('complete',rgb_pairs=len(rows))

if __name__=='__main__':
    try:main()
    except BaseException as e:status('failed',error=repr(e),traceback=traceback.format_exc());raise
