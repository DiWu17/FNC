"""Post-hoc scientific plots over frozen V21 outputs. No model inference/fitting."""
from collections import Counter,defaultdict
from datetime import datetime,timezone
import hashlib,json,re,sys,textwrap
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from clickagent.mvp_repair_v21 import stable_key
from clickagent.public_evaluation import point_in_geometry
from clickagent.frozen_selection_v18 import evidence

RUN=ROOT/'runs/mvp_repair_v21'
OUT=RUN/'candidate_value_analysis'
SIGNALS={'scorer':'score','closer_to_MVP':'negative_distance','spatial_view_support':'spatial_support',
         'exact_view_count':'exact_views','active_view_count':'active_views',
         'seed_top8_count':'seed_top8_count','local_contrast':'local_contrast'}

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def hit(p,c,label):return p is not None and point_in_geometry((np.asarray(p,dtype=float)/c['image_size']).tolist(),label)
def auc(y,s):
    pos=s[y];neg=s[~y]
    if not len(pos) or not len(neg):return None
    return float(((pos[:,None]>neg).astype(float)+.5*(pos[:,None]==neg)).mean())
def interval(values,seed=20262122):
    a=np.asarray(values,float)
    if not len(a):return None
    rng=np.random.default_rng(seed)
    draws=np.asarray([a[rng.integers(len(a),size=len(a))].mean() for _ in range(2000)])
    return dict(n_screenshots=len(a),mean=float(a.mean()),ci95=np.quantile(draws,[.025,.975]).tolist())
def distance(points,anchor,size):
    return np.linalg.norm(np.asarray(points)-np.asarray(anchor),axis=-1)/np.hypot(*size)
def value_class(mvp_hit,point_hit):
    return ('redundant_correct' if point_hit else 'distractor') if mvp_hit else ('potential_repair' if point_hit else 'failed_alternative')
def strict_coord_output(text):
    # Diagnostic of syntax only; no change to official parsed MVP coordinates.
    return bool(re.fullmatch(r'\s*[\(\[]?\s*-?\d+(?:\.\d+)?\s*,\s*-?\d+(?:\.\d+)?\s*[\)\]]?\s*',str(text)))

def extract():
    complete=read(RUN/'completion.json');assert complete['complete']
    for p,h in complete['artifact_sha256'].items():assert sha(p)==h,p
    public=read(RUN/'data/public.json');labels=read(RUN/'data/labels.json');props=read(RUN/'data/proposals.json')
    indices=read(ROOT/'runs/bottleneck_v19/data/indices.json')
    index={e['sample_id']:e for co in ['new_development','old_val'] for e in indices[co]}
    freeze=read(RUN/'predictions_frozen.json');primary=read(RUN/'results.json')['hit_vectors'];cases=[];candidate_rows=[]
    for j,c in enumerate(public):
        sid=c['sample_id'];cp=RUN/f'cases/{stable_key(sid)}.json';assert sha(cp)==freeze['cases'][str(cp)]
        r=read(cp);m=read(r['mvp_path'])['output']['reference_result'];p=props[sid];e=index[sid]
        assert sha(e['path'])==e['sha256']
        with np.load(e['path'],allow_pickle=False) as z:
            pts=z['points'].copy();base=z['base'].copy();active=z['active'].copy()
        assert pts.tolist()==p['points'];scores=np.asarray(p['scores'],float)
        hits=np.asarray([hit(q,c,labels[sid]) for q in pts],bool)
        assert sha(e['labels_path'])==e['labels_sha256']
        np.testing.assert_array_equal(hits,np.load(e['labels_path'],allow_pickle=False)>.5)
        native={tuple(q) for q in r['native_pool']};mhit=hit(r['mvp_point'],c,labels[sid]);nhit=any(hit(q,c,labels[sid]) for q in native)
        assert mhit==primary['mvp'][0][j]
        n=len(pts);ranks=np.empty_like(scores,dtype=int)
        for s in range(3):ranks[s,np.argsort(-scores[s],kind='stable')]=np.arange(1,n+1)
        support,_=evidence(pts,base,c['image_size'])
        d=distance(pts,r['mvp_point'],c['image_size'])
        dn=np.min(np.linalg.norm(pts[:,None]-np.asarray(list(native))[None],axis=-1),axis=1)/np.hypot(*c['image_size'])
        with Image.open(c['image']) as im:
            gray=np.asarray(im.convert('L'))
        contrast=[]
        for x,y in pts:
            patch=gray[max(0,y-16):min(len(gray),y+17),max(0,x-16):min(gray.shape[1],x+17)]
            contrast.append(float(patch.std())/255)
        fixed={'distance':d,'negative_distance':-d,'nearest_native_distance':dn,'spatial_support':support,
               'exact_views':base[:,30]*4,'active_views':active.sum(1),'seed_top8_count':(ranks<=8).sum(0),
               'local_contrast':np.asarray(contrast),'global_origin':base[:,0]>.5,'local_origin':base[:,[7,14,21]].any(1)}
        native_wrong=not mhit;seeds=[];point_index={tuple(q):i for i,q in enumerate(pts)}
        for s in range(3):
            pool=r['augmented_pools'][s];selected=r['augmented_results'][s];extra=[q for q in pool if tuple(q) not in native]
            selected_hit=hit(selected['point'],c,labels[sid]);pool_hit=any(hit(q,c,labels[sid]) for q in pool)
            assert selected_hit==primary['augmented_verify'][s][j]
            assert hit(r['original_seed_points'][s],c,labels[sid])==primary['original'][s][j]
            best_rank=int(ranks[s,hits].min()) if hits.any() else None
            ordered=np.argsort(-scores[s],kind='stable');ordered=[i for i in ordered if list(pts[i])!=r['mvp_point']]
            oracle={str(k):bool(hits[ordered[:k]].any()) for k in [1,2,4,8,16,32,64]}
            selected_new=selected['point'] is not None and tuple(selected['point']) not in native
            sf=dict(seed=s+1,extra=extra,pool_hit=pool_hit,selected_hit=selected_hit,best_correct_rank=best_rank,
                    oracle=oracle,repair=native_wrong and selected_hit,harm=mhit and not selected_hit,
                    selected_new=selected_new,order_agreement=not selected['order_disagrees'],selected_point=selected['point'])
            # Scores are already aligned to canonical candidate identities in V21.
            ix=selected['selected_candidate_index']
            sf['verifier_margin_over_anchor']=float(selected['scores'][ix]-selected['scores'][0]) if ix is not None else 0.
            sf['shortlist_correct_extra_count']=sum(hit(q,c,labels[sid]) for q in extra)
            seeds.append(sf)
            for q in extra:
                i=point_index[tuple(q)];h=bool(hits[i]);rr=dict(sample_id=sid,source=c['source'],cohort=c['cohort'],seed=s+1,
                    point=q,mvp_hit=mhit,point_hit=h,value_class=value_class(mhit,h),native_pool_hit=nhit,
                    incremental_coverage=not nhit and h,rank=int(ranks[s,i]),score=float(scores[s,i]),
                    score_z=float((scores[s,i]-scores[s].mean())/max(scores[s].std(),1e-9)),
                    selected=q==selected['point'],order_agreement=sf['order_agreement'],verifier_margin_over_anchor=sf['verifier_margin_over_anchor'])
                rr.update({k:float(v[i]) for k,v in fixed.items()});candidate_rows.append(rr)
        # Within-screen AUC avoids treating hundreds of points as independent cases.
        aucs={}
        for s in range(3):
            feat=dict(fixed,score=scores[s]);aucs[str(s+1)]={}
            for scope,mask in [('full',np.ones(n,bool)),('top16',ranks[s]<=16)]:
                aucs[str(s+1)][scope]={name:auc(hits[mask],feat[key][mask]) for name,key in SIGNALS.items()}
        best_i=int(np.flatnonzero(hits)[np.argmin(ranks[0,hits])]) if hits.any() else None
        badsyntax=not strict_coord_output(m['final_prediction'].get('output',''))
        near_native_pair=min((np.linalg.norm(np.asarray(a)-b) for ia,a in enumerate(r['native_pool']) for b in r['native_pool'][ia+1:]),default=0)
        cases.append(dict(**c,mvp_point=r['mvp_point'],mvp_hit=mhit,native_pool=r['native_pool'],native_pool_hit=nhit,
            full_pool_hit=bool(hits.any()),n_full=n,seeds=seeds,aucs=aucs,strict_coordinate_format_failure=badsyntax,
            mvp_output=m['final_prediction'].get('output',''),mvp_min_native_pair_distance=near_native_pair,
            best_correct_point=pts[best_i].tolist() if best_i is not None else None,
            full_positive_count=int(hits.sum()),label=labels[sid]))
        if (j+1)%20==0:print(f'feature extraction {j+1}/100',flush=True)
    write(OUT/'cases.json',cases);write(OUT/'shortlisted_candidates.json',candidate_rows)
    return cases,candidate_rows

def summarize(cases,rows):
    result={}
    for co in ['new_development','old_val','all']:
        cc=[c for c in cases if co=='all' or c['cohort']==co];out={}
        for s in range(3):
            wrong=[c for c in cc if not c['mvp_hit']];reachable=[c for c in wrong if c['full_pool_hit']]
            rr=[r for r in rows if r['seed']==s+1 and (co=='all' or r['cohort']==co)]
            counts=Counter(r['value_class'] for r in rr);hist=Counter()
            for c in reachable:
                rank=c['seeds'][s]['best_correct_rank'];label='1' if rank==1 else '2' if rank==2 else '3-4' if rank<=4 else '5-8' if rank<=8 else '9-16' if rank<=16 else '>16'
                hist[label]+=1
            sub=dict(n=len(cc),mvp_wrong=len(wrong),reachable_wrong=len(reachable),shortlisted_candidate_counts=dict(counts),
                best_correct_rank_histogram=dict(hist),oracle_recovery_counts={str(k):sum(c['seeds'][s]['oracle'][str(k)] for c in wrong) for k in [1,2,4,8,16,32,64]},
                actual_aug_repair=sum(c['seeds'][s]['repair'] for c in cc),actual_aug_harm=sum(c['seeds'][s]['harm'] for c in cc),
                shortlist_contains_on_mvp_wrong=sum(c['seeds'][s]['pool_hit'] for c in wrong),
                ranking_aucs={scope:{sig:interval([c['aucs'][str(s+1)][scope][sig] for c in reachable if c['aucs'][str(s+1)][scope][sig] is not None]) for sig in SIGNALS} for scope in ['full','top16']})
            switches=[c for c in cc if c['seeds'][s]['selected_new']]
            sub['new_point_switches']=dict(n=len(switches),repairs=sum(c['seeds'][s]['repair'] for c in switches),harms=sum(c['seeds'][s]['harm'] for c in switches),
                failed_alternative=sum(not c['mvp_hit'] and not c['seeds'][s]['selected_hit'] for c in switches),redundant_correct=sum(c['mvp_hit'] and c['seeds'][s]['selected_hit'] for c in switches))
            # Describe fixed signals on actual repaired/broken selections; never tune a gate.
            sub['switch_features']={}
            for cls in ['repair','harm']:
                cr=[r for r in rr if r['selected'] and ((not r['mvp_hit'] and r['point_hit']) if cls=='repair' else (r['mvp_hit'] and not r['point_hit']))]
                sub['switch_features'][cls]=dict(n=len(cr),medians={k:float(np.median([r[k] for r in cr])) if cr else None for k in ['rank','distance','spatial_support','exact_views','active_views','score_z','local_contrast','verifier_margin_over_anchor']},order_agreement_count=sum(r['order_agreement'] for r in cr))
            out[str(s+1)]=sub
        result[co]=out
    result['mvp_syntax_diagnostic']=dict(n=100,strict_format_failures=sum(c['strict_coordinate_format_failure'] for c in cases),
        incorrect_among_format_failures=sum(c['strict_coordinate_format_failure'] and not c['mvp_hit'] for c in cases),
        description='Syntax-only posthoc flag. Official parsed predictions left unchanged; noncanonical text is not automatically an invalid prediction.')
    write(OUT/'statistics.json',result);return result

def plot(cases,rows,stats):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    colors=['#2878B5','#DD8452','#55A868','#C44E52']
    ss=stats['new_development'];s1=ss['1'];xx=np.arange(7);kk=[1,2,4,8,16,32,64]
    fig,ax=plt.subplots(1,3,figsize=(14.5,4.5))
    labels=['MVP errors','Our full pool\ncan reach','Augmented pool\ncan reach','Actually\nrepaired']
    vals=[s1['mvp_wrong'],s1['reachable_wrong'],s1['shortlist_contains_on_mvp_wrong'],s1['actual_aug_repair']]
    bars=ax[0].bar(range(4),vals,color=['#999999',colors[0],colors[0],colors[2]])
    ax[0].bar_label(bars);ax[0].set_xticks(range(4),labels,fontsize=8);ax[0].set_ylabel('Screenshots (new dev72, fixed S1)')
    ax[0].set_ylim(0,max(vals)*1.2);ax[0].set_title('Where corrective potential is lost')
    mat=np.asarray([[ss[str(s)]['oracle_recovery_counts'][str(k)] for k in kk] for s in [1,2,3]])
    ax[1].fill_between(xx,mat.min(0),mat.max(0),alpha=.18,color=colors[0],label='Range across 3 fixed fits')
    ax[1].plot(xx,mat[0],'-o',color=colors[0],label='Fixed S1')
    ax[1].axhline(s1['reachable_wrong'],color='gray',linestyle=':',label='Full-pool ceiling')
    ax[1].set_xticks(xx,kk);ax[1].set_xlabel('Top K original-score points (oracle only)');ax[1].set_ylabel('MVP errors with a correct point')
    ax[1].set_ylim(0,s1['mvp_wrong']);ax[1].set_title('Correct points often rank below top2');ax[1].legend(fontsize=7)
    cats=['potential_repair','redundant_correct','failed_alternative','distractor']
    vals=[s1['shortlisted_candidate_counts'].get(k,0) for k in cats]
    bars=ax[2].bar(range(4),vals,color=[colors[2],colors[0],'#999999',colors[3]])
    ax[2].bar_label(bars);ax[2].set_xticks(range(4),['Correct when\nMVP wrong','Correct when\nMVP correct','Wrong when\nMVP wrong','Wrong when\nMVP correct'],fontsize=8)
    ax[2].set_ylabel('Actually added points, fixed S1');ax[2].set_title('Correctness and extra value differ');ax[2].margins(y=.15)
    fig.suptitle('Frozen V21: descriptive candidate analysis on reused development data',fontsize=12)
    fig.tight_layout();fig.savefig(OUT/'value_and_shortlist.png',dpi=170);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(12.5,4.6));names=list(SIGNALS)
    for ax,scope,title in zip(axes,['full','top16'],['Entire original candidate pool','Harder comparison: top16 only']):
        for j,sig in enumerate(names):
            v=s1['ranking_aucs'][scope][sig]
            if v is None:continue
            lo,hi=v['ci95'];ax.plot([lo,hi],[j,j],color=colors[0],linewidth=2);ax.plot(v['mean'],j,'o',color=colors[0])
            ax.text(1.01,j,f'{v["mean"]:.2f} (n={v["n_screenshots"]})',transform=ax.get_yaxis_transform(),va='center',fontsize=8)
        ax.axvline(.5,color='gray',linestyle=':');ax.set_xlim(0,1);ax.set_yticks(range(len(names)),['Original score','Closer to MVP','Spatial view support','Exact view count','Containing view count','3-fit top8 count','Local gray contrast'])
        ax.invert_yaxis();ax.set_xlabel('Mean within-screenshot AUC (95% screenshot bootstrap)');ax.set_title(title,fontsize=10)
    fig.suptitle('Can existing public signals rank correct points? MVP-wrong, reachable new-dev cases, S1',fontsize=11)
    fig.tight_layout(w_pad=3.5);fig.savefig(OUT/'feature_ranking.png',dpi=170,bbox_inches='tight');plt.close(fig)

    fig,axes=plt.subplots(1,3,figsize=(13.5,4.5));selected=[r for r in rows if r['seed']==1 and r['selected'] and ((not r['mvp_hit'] and r['point_hit']) or (r['mvp_hit'] and not r['point_hit']))]
    for cls,color,marker in [('repair',colors[2],'o'),('harm',colors[3],'x')]:
        rr=[r for r in selected if (not r['mvp_hit'] if cls=='repair' else r['mvp_hit'])]
        for ax,key,label in zip(axes,['rank','spatial_support','verifier_margin_over_anchor'],['Original scorer rank','Spatial view support','Verifier score advantage over MVP']):
            ax.scatter([100*r['distance'] for r in rr],[r[key] for r in rr],color=color,marker=marker,label=f'{cls} (n={len(rr)})',alpha=.8,s=45)
            ax.set_xlabel('Distance from MVP (% of screenshot diagonal)');ax.set_ylabel(label)
    axes[0].set_yscale('log');axes[0].legend(fontsize=8)
    fig.suptitle('Observed switches: repairs and harmful replacements overlap (all100, fixed S1)',fontsize=11)
    fig.tight_layout();fig.savefig(OUT/'repair_vs_harm.png',dpi=170);plt.close(fig)

def examples(cases):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle,Polygon
    chosen=[]
    conflict='expansion15:groundcua:b69866ebe0c7f23360377d15'
    rules=[('label_conflict',lambda c:c['sample_id']==conflict),
           ('repair_near',lambda c:c['seeds'][0]['repair'] and c['seeds'][0]['selected_new'] and distance([c['seeds'][0]['selected_point']],c['mvp_point'],c['image_size'])[0]<=.03),
           ('repair_far',lambda c:c['seeds'][0]['repair'] and c['seeds'][0]['selected_new'] and distance([c['seeds'][0]['selected_point']],c['mvp_point'],c['image_size'])[0]>.03),
           ('harm',lambda c:c['seeds'][0]['harm'] and c['seeds'][0]['selected_new']),
           ('shortlist_miss',lambda c:not c['mvp_hit'] and c['full_pool_hit'] and not c['seeds'][0]['pool_hit']),
           ('verification_miss',lambda c:not c['mvp_hit'] and c['seeds'][0]['pool_hit'] and not c['seeds'][0]['selected_hit'])]
    used=set();(OUT/'examples').mkdir(exist_ok=True)
    for kind,rule in rules:
        eligible=[c for c in cases if c['sample_id'] not in used and rule(c)]
        if not eligible:continue
        c=min(eligible,key=lambda c:stable_key(c['sample_id']+'|candidate_value_example'));used.add(c['sample_id'])
        with Image.open(c['image']) as image:im=np.asarray(image.convert('RGB'))
        w,h=c['image_size'];geo=c['label'];s=c['seeds'][0]
        if geo['type']=='bbox':
            box=max(geo['boxes'],key=lambda b:(b[2]-b[0])*(b[3]-b[1]));center=np.asarray([(box[0]+box[2])*w/2,(box[1]+box[3])*h/2])
        else:center=np.asarray(geo['polygons'][0]).mean(0)*[w,h]
        coords=[('M',c['mvp_point'],'#2878B5')]+[(str(i+1),p,'#DD8452') for i,p in enumerate(s['extra'])]
        if kind in ['shortlist_miss','label_conflict']:coords.append(('C',c['best_correct_point'],'#AA4499'))
        def draw(ax):
            ax.imshow(im)
            if geo['type']=='bbox':
                for x0,y0,x1,y1 in geo['boxes']:ax.add_patch(Rectangle((x0*w,y0*h),(x1-x0)*w,(y1-y0)*h,fill=False,edgecolor='#00A050',linewidth=2))
            else:
                for p in geo['polygons']:ax.add_patch(Polygon(np.asarray(p)*[w,h],fill=False,edgecolor='#00A050',linewidth=2))
            for i,(label,p,color) in enumerate(coords):
                ax.scatter(*p,s=80,facecolors='none',edgecolors=color,linewidths=1.6)
                ax.annotate(label,p,xytext=(8,10 if i%2 else -18),textcoords='offset points',color=color,fontsize=11,fontweight='bold',bbox=dict(facecolor='white',alpha=.85,edgecolor='none',pad=1))
            ax.scatter(*s['selected_point'],s=110,marker='*',color='#C44E52',edgecolor='white',linewidth=.5)
            ax.set_xticks([]);ax.set_yticks([])
        def zoom(ax,pt):
            span=min(512,max(w,h));x=max(0,min(w-span,float(pt[0])-span/2));y=max(0,min(h-span,float(pt[1])-span/2))
            ax.set_xlim(x,min(w,x+span));ax.set_ylim(min(h,y+span),y)
        fig,axes=plt.subplots(1,3,figsize=(14.4,5.7));
        for ax in axes:draw(ax)
        axes[0].set_xlim(0,w);axes[0].set_ylim(h,0);axes[0].set_title('Full screenshot')
        zoom(axes[1],center);axes[1].set_title('Target context (GT used for analysis crop)')
        zoom(axes[2],s['selected_point']);axes[2].set_title('Chosen point context')
        instruction='\n'.join(textwrap.wrap(c['instruction'],width=125))
        fig.suptitle(f'{kind} | {c["source"]} | MVP hit={c["mvp_hit"]}, final hit={s["selected_hit"]}\n{instruction}',fontsize=10,y=.98)
        fig.text(.5,.02,'Green outline: target | Blue M: MVP | Orange 1/2: new supplemental points | Red star: final | Purple C: missed correct point',ha='center',fontsize=9)
        fig.tight_layout(rect=[0,.05,1,.88]);path=OUT/f'examples/{kind}.png';fig.savefig(path,dpi=145);plt.close(fig)
        chosen.append(dict(kind=kind,sample_id=c['sample_id'],path=str(path),instruction=c['instruction'],source=c['source'],
            selection='Deterministic minimum SHA within outcome stratum;illustrative,not prevalence evidence',mvp_hit=c['mvp_hit'],final_hit=s['selected_hit'],
            extra_points=s['extra'],chosen_point=s['selected_point'],best_correct_point=c['best_correct_point'],best_correct_rank=s['best_correct_rank'],
            mvp_output_format_flag=c['strict_coordinate_format_failure']))
    write(OUT/'examples.json',chosen)

def main():
    OUT.mkdir(exist_ok=True);assert read(OUT/'analysis_plan.json')['kind']=='posthoc_descriptive_analysis'
    cases,rows=extract();stats=summarize(cases,rows);plot(cases,rows,stats);examples(cases)
    write(OUT/'analysis_completion.json',dict(complete=True,completed_at_utc=datetime.now(timezone.utc).isoformat(),
        screenshots=100,training=False,new_model_inference=False,posthoc_descriptive=True,
        script_sha256=sha(__file__),artifacts={str(p):sha(p) for p in OUT.rglob('*') if p.is_file() and p.name!='analysis_completion.json'}))
    print(json.dumps({'complete':True,'new_dev_s1':stats['new_development']['1'],'all_switches_s1':stats['all']['1']['switch_features'],'syntax':stats['mvp_syntax_diagnostic']},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
