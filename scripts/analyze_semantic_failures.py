"""Post-hoc failure attribution and negative controls; no inference changes."""
import json
from collections import Counter
from pathlib import Path

import numpy as np

from clickagent.action_benchmark import write_json
from clickagent.action_grounding import AttentionPointGrounder,_pixel_centers
from clickagent.public_evaluation import point_in_geometry
from clickagent.semantic_target import STOP,temporal_map
from run_semantic_target import load_feature
from run_target_token_experiment import read_json,read_jsonl


def distance_to_geometry(point,geometry,size):
    if point is None:return None
    w,h=size;x,y=point
    if point_in_geometry([x/w,y/h],geometry):return 0.
    if geometry['type']=='bbox':
        return float(min(np.hypot(max(a*w-x,0,x-c*w),max(b*h-y,0,y-d*h)) for a,b,c,d in geometry['boxes']))
    distances=[]
    for polygon in geometry['polygons']:
        vertices=np.asarray(polygon)*[w,h]
        for a,b in zip(vertices,np.roll(vertices,-1,axis=0)):
            vector=b-a;t=np.clip(np.dot(np.asarray(point)-a,vector)/(np.dot(vector,vector) or 1.),0.,1.)
            distances.append(np.linalg.norm(np.asarray(point)-(a+t*vector)))
    return float(min(distances))


def main():
    output=Path('runs/semantic_target_v2');dest=output/'failure_analysis';dest.mkdir(exist_ok=True)
    cases=[c for c in read_jsonl(output/'manifest.jsonl') if c['experiment_split']=='test']
    diagnostics={r['sample_id']:r for r in read_jsonl(output/'diagnostics/cases.jsonl')}
    predictions={(r['method'],r['sample_id']):r for r in read_jsonl(output/'predictions.jsonl')}
    grounder=AttentionPointGrounder.load('runs/action_v3/grounder/grounder.joblib')
    choice=read_json(output/'frozen_selection.json')['selected']
    records=[];topk={k:Counter() for k in (1,3,5,10)};masks=[];peaks=[];single=[]
    for c in cases:
        sid=c['sample_id'];d=diagnostics[sid]
        raw,meta=load_feature(output,c,choice['variant'])
        attention=temporal_map(raw,meta,choice['policy'])
        gh,gw=meta['grid_shape'];w,h=c['image_size'];x,y=_pixel_centers((gh,gw),c['image_size'])
        center_mask=np.asarray([point_in_geometry([a/w,b/h],c['target_geometry']) for a,b in zip(x,y)],bool)
        from run_target_token_experiment import overlap_cells
        overlap=overlap_cells(c['target_geometry'],(gh,gw))
        r=dict(d,native=predictions['qwen_native_completed_lax',sid],
            layer15=predictions['optimized_layer15_mean',sid],
            patch_pixel_size=[w/gw,h/gh],positive_centers=int(center_mask.sum()),patch_count=gh*gw)
        r['point_distance_to_target_original_pixels']=distance_to_geometry(d['optimized']['point'],c['target_geometry'],c['image_size'])
        if attention is None:
            peaks.append(None);single.append(None);r['first_token_piece']=None
        else:
            from clickagent.action_grounding import patch_features
            features=patch_features(attention,(gh,gw))
            probs=grounder.pipeline.predict_proba(features)[:,list(grounder.pipeline.classes_).index(1)]
            order=np.argsort(-probs,kind='stable')
            assert [int(x[order[0]]),int(y[order[0]])]==d['optimized']['point']
            for k in topk:
                topk[k]['point_candidate_hit']+=bool(center_mask[order[:k]].any())
                topk[k]['cell_candidate_overlap']+=bool(overlap[order[:k]].any())
            head_peaks=attention.argmax(-1)
            peaks.append(np.stack(((head_peaks%gw+.5)/gw,(head_peaks//gw+.5)/gh),axis=-1))
            patch=int(order[0]);single.append(np.asarray([(patch%gw+.5)/gw,(patch//gw+.5)/gh]))
            index=meta['stages']['target_first'][0]
            before='' if index==1 else meta['generated_prefixes'][index-2]
            after=meta['generated_prefixes'][index-1]
            common=0
            for a,b in zip(before,after):
                if a!=b:break
                common+=1
            r['first_token_piece']=after[common:]
        masks.append(center_mask);records.append(r)
    # Shuffle entire sets of head peaks between different images within each
    # dataset. Preserve peak correlations and typical GUI positional bias.
    # Re-map source normalized cell positions onto each destination's own grid.
    head_matrix=np.zeros((len(cases),len(cases)),bool)
    point_matrix=np.zeros_like(head_matrix)
    for i,c in enumerate(cases):
        gh,gw=records[i]['grid_shape']
        for j in range(len(cases)):
            if peaks[j] is not None:
                xy=peaks[j];indices=np.floor(xy[:,1]*gh).astype(int)*gw+np.floor(xy[:,0]*gw).astype(int)
                head_matrix[i,j]=masks[i][indices].any()
            if single[j] is not None:
                a,b=single[j];point_matrix[i,j]=masks[i][int(b*gh)*gw+int(a*gw)]
    observed_heads=int(np.trace(head_matrix));assert observed_heads==sum(r.get('any_individual_head_peak_hits',False) for r in records)
    assert int(np.trace(point_matrix))==20
    rng=np.random.default_rng(20261101);head_null=[];point_null=[]
    groups=[np.asarray([i for i,c in enumerate(cases) if c['dataset']==ds]) for ds in sorted({c['dataset'] for c in cases})]
    for _ in range(5000):
        hs=0;ps=0
        for group in groups:
            shuffled=rng.permutation(group)
            while np.any(shuffled==group):shuffled=rng.permutation(group)
            hs+=head_matrix[group,shuffled].sum();ps+=point_matrix[group,shuffled].sum()
        head_null.append(int(hs));point_null.append(int(ps))
    native_misses=[r for r in records if r['native']['hit'] and not r['optimized']['hit']]
    def null_report(values,observed):return {'observed':observed,'shuffle_mean':float(np.mean(values)),
        'shuffle_95_range':np.percentile(values,[2.5,97.5]).tolist(),'fraction_shuffle_at_least_observed':float(np.mean(np.asarray(values)>=observed))}
    first_generic=[r for r in records if r['first_token_piece'] and r['first_token_piece'].strip().casefold() in STOP]
    summary={'n':128,'native_correct_ours_wrong':len(native_misses),
        'native_correct_ours_wrong_buckets':dict(Counter(r['optimized']['failure_stage'] for r in native_misses)),
        'native_correct_ours_wrong_cell_overlap':sum(r['optimized']['cell_overlap'] for r in native_misses),
        'center_reachable_and_selected_cell_overlap':[
            {'grid_center_can_hit':a,'selected_cell_overlap':b,'n':sum(r['grid_center_can_hit']==a and r['optimized']['cell_overlap']==b for r in records)} for a in (True,False) for b in (True,False)],
        'perfect_within_selected_cell_upper_bound':sum(r['optimized']['cell_overlap'] for r in records),
        'boundary_tolerance_diagnostic':{str(t):{'hits':sum(r['point_distance_to_target_original_pixels'] is not None and r['point_distance_to_target_original_pixels']<=t+1e-8 for r in records)} for t in (0,2,5,10,20)},
        'top_k_candidate_oracles':{str(k):dict(v) for k,v in topk.items()},
        'raw_layer15_correct_main_wrong':sum(r['layer15']['hit'] and not r['optimized']['hit'] for r in records),
        'raw_layer15_wrong_main_correct':sum(not r['layer15']['hit'] and r['optimized']['hit'] for r in records),
        'first_token_is_standalone_generic_word':len(first_generic),
        'first_generic_word_histogram':dict(Counter(r['first_token_piece'].strip().casefold() for r in first_generic)),
        'head_oracle_shuffle_control':null_report(head_null,observed_heads),
        'main_point_shuffle_control':null_report(point_null,20),
        'note':'Post-hoc diagnosis only. Top-k/oracle/counterfactual bounds are not deployable accuracy. Shuffle uses other screenshots within each dataset, not a causal or application-controlled test.'}
    write_json(dest/'summary.json',summary)
    (dest/'cases.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf-8')
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    # Outcome-selected examples for manual semantic review; no tuning.
    chosen=['screenspot_v2:000762:mobile:428','screenspot_v2:000322:desktop:322',
        'osworld_g:000301:rpngAhRMhE-0','screenspot_v2:001156:web:321']
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle,Polygon
    from PIL import Image
    plt.rcParams['font.family']=['Microsoft YaHei','DejaVu Sans']
    fig,axes=plt.subplots(2,2,figsize=(16,12))
    by_id={c['sample_id']:c for c in cases};r_by_id={r['sample_id']:r for r in records}
    for ax,sid in zip(axes.ravel(),chosen):
        c=by_id[sid];r=r_by_id[sid];w,h=c['image_size']
        with Image.open(c['image']) as im:ax.imshow(im.convert('RGB'))
        geom=c['target_geometry']
        if geom['type']=='bbox':
            for a,b,e,f in geom['boxes']:ax.add_patch(Rectangle((a*w,b*h),(e-a)*w,(f-b)*h,fill=False,edgecolor='lime',linewidth=2))
        else:
            for poly in geom['polygons']:ax.add_patch(Polygon(np.asarray(poly)*[w,h],fill=False,edgecolor='lime',linewidth=2))
        for name,color in (('optimized','red'),('native','cyan')):
            if r[name]['point'] is not None:ax.scatter(*r[name]['point'],c=color,marker='+',s=110,linewidth=2)
        gh,gw=r['grid_shape'];point=r['optimized']['point']
        if point:
            col=min(gw-1,int(point[0]/w*gw));row=min(gh-1,int(point[1]/h*gh))
            ax.add_patch(Rectangle((col*w/gw,row*h/gh),w/gw,h/gh,fill=False,edgecolor='red',linewidth=1.5))
        ax.set_title(c['instruction']+'\nGenerated name: '+str(r['target_text']),fontsize=10,wrap=True)
        ax.axis('off')
    fig.suptitle('Post-hoc examples: green=target, red=our point/cell, cyan=native Qwen point',fontsize=12)
    fig.tight_layout();fig.savefig(dest/'examples.png',dpi=140);plt.close(fig)
    # One example where the unchanged layer-15 mean points correctly, but the
    # downstream 96-head readout selects a different settings row.
    sid=chosen[0];c=by_id[sid];r=r_by_id[sid];w,h=c['image_size']
    raw,meta=load_feature(output,c,choice['variant']);attention=temporal_map(raw,meta,choice['policy'])
    layer_heat=attention[:32].mean(0);layer_heat/=layer_heat.sum()
    probs=grounder.pipeline.predict_proba(patch_features(attention,meta['grid_shape']))[:,1]
    with Image.open(c['image']) as im:im=im.convert('RGB')
    fig,axes=plt.subplots(1,3,figsize=(11,10))
    for ax,title,heat,point in zip(axes,('Screenshot','Layer 15 mean attention (relative)','96-head logistic score (not action confidence)'),
            (None,layer_heat,probs),(None,r['layer15']['point'],r['optimized']['point'])):
        ax.imshow(im)
        if heat is not None:
            ax.imshow(heat.reshape(meta['grid_shape']),extent=(0,w,h,0),cmap='magma',alpha=.55,interpolation='nearest')
        for a,b,e,f in c['target_geometry']['boxes']:ax.add_patch(Rectangle((a*w,b*h),(e-a)*w,(f-b)*h,fill=False,edgecolor='lime',linewidth=2))
        if point:ax.scatter(*point,c='cyan' if title.startswith('Layer') else 'red',marker='+',s=120,linewidth=2)
        ax.set_title(title,fontsize=10,wrap=True);ax.axis('off')
    fig.suptitle('Instruction: view books setting | Generated target: Books | Green = correct target row',fontsize=12)
    fig.tight_layout();fig.savefig(dest/'books_readout.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
