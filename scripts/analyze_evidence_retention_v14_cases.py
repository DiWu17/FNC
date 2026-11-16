"""Posthoc explanation of frozen V14 retention outputs; no policy changes."""
from __future__ import annotations
import hashlib
import html
import json
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle,Polygon

ROOT=next(p for p in Path(__file__).resolve().parents
          if (p/'src/clickagent').is_dir() and (p/'runs/evidence_retention_v14').is_dir())
RUN=ROOT/'runs/evidence_retention_v14'


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))


def occupied(points,size):
    if not len(points):return 0
    cells=np.clip(np.floor(np.asarray(points)/np.asarray(size)*8).astype(int),0,7)
    return len({tuple(p) for p in cells})


def illustrate(case,row,dest):
    data=Path(case['image']).read_bytes()
    assert hashlib.sha256(data).hexdigest()==case['image_sha256']
    with Image.open(case['image']) as im:image=np.asarray(im.convert('RGB'))
    w,h=case['image_size'];fig,axes=plt.subplots(1,2,figsize=(14,6),layout='constrained')
    for ax,name in zip(axes,('baseline','aligned_s1')):
        ax.imshow(image)
        for b in case['target_geometry'].get('boxes',[]):
            x0,y0,x1,y1=b;ax.add_patch(Rectangle((x0*w,y0*h),(x1-x0)*w,(y1-y0)*h,fill=False,edgecolor='#00ef75',linewidth=2,zorder=10))
        for p in case['target_geometry'].get('polygons',[]):
            ax.add_patch(Polygon([(x*w,y*h) for x,y in p],fill=False,edgecolor='#00ef75',linewidth=2,zorder=10))
        points=np.asarray(row['points']).reshape(-1,2)[row['pools'][name]]
        if len(points):ax.scatter(points[:,0],points[:,1],s=30,marker='x',color='#ff354d' if name=='baseline' else '#15c4ff')
        ax.set_title(f"{name}: {len(points)} points, reachable={row['coverage'][name]}")
        ax.set_axis_off()
    fig.suptitle(case['instruction'],fontsize=11)
    fig.savefig(dest,dpi=150);plt.close(fig)


def main():
    rows=[json.loads(l) for l in (RUN/'experiment2/pools.jsonl').read_text(encoding='utf-8').splitlines()]
    rows=sorted([r for r in rows if r['split']=='val'],key=lambda r:r['sample_id'])
    cases={r['sample_id']:r for r in (json.loads(l) for l in (ROOT/'runs/multiregion_v13/data/val.jsonl').read_text(encoding='utf-8').splitlines())}
    output=RUN/'analysis';output.mkdir(exist_ok=True)
    selected=[]
    for label,predicate in [('coverage_gain',lambda r:r['coverage']['aligned_s1'] and not r['coverage']['baseline']),
                             ('coverage_loss',lambda r:r['coverage']['baseline'] and not r['coverage']['aligned_s1'])]:
        row=next((r for r in rows if predicate(r)),None)
        if row is None:continue
        case=cases[row['sample_id']];image=label+'.png';illustrate(case,row,output/image)
        selected.append(dict(label=label,sample_id=row['sample_id'],instruction=case['instruction'],source=case['dataset'],
            coverage={name:row['coverage'][name] for name in row['coverage']},image=image,
            selection_rule='First lexicographic sample_id for primaryS1 coverage gain/loss; fixed output explanation only'))
    spatial={}
    base={e['sample_id']:e for e in load(RUN/'base/index.json')['cases'] if e['split']=='val'}
    for policy in ('baseline','rank_s1','rank_s2','aligned_s1','aligned_s2'):
        values=[];correct=slots=0;per_reachable=[]
        for r in rows:
            case=cases[r['sample_id']];points=np.asarray(r['points']).reshape(-1,2)[r['pools'][policy]]
            values.append(occupied(points,case['image_size']))
            labels=np.load(base[r['sample_id']]['labels_path'],allow_pickle=False)
            n=int(labels[r['pools'][policy]].sum());correct+=n;slots+=len(points)
            if n:per_reachable.append(n)
        spatial[policy]=dict(n=172,mean_occupied_8x8_cells=float(np.mean(values)),
            correct_point_instances=correct,point_slots=slots,point_precision=correct/max(slots,1),
            mean_correct_points_when_reachable=float(np.mean(per_reachable)),
            interpretation='Posthoc spread and correct-point density, not deployed accuracy; does not establish that diversity/density causes final outcome differences.')
    summary=dict(complete=True,posthoc=True,policy_or_hyperparameter_changes=False,
        experiment2_results_sha256=hashlib.sha256((RUN/'experiment2/results.json').read_bytes()).hexdigest(),examples=selected,spatial=spatial)
    (output/'case_analysis.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    body=['<h1>候选保留：主种子的救回与损失案例</h1>',
        '<p>本页为冻结输出的事后解释，不改变候选、模型或超参数。按 sample_id 排序，分别取第一条可达性救回和损失；同时展示正负案例。绿框为有效标注，红叉为原配额，蓝叉为新增内部证据方法。所有点原本都已解码。</p>']
    for e in selected:
        body.extend(['<h2>'+html.escape(e['label'])+'</h2><p>'+html.escape(e['sample_id']+' · '+e['instruction'])+'</p>',
                     '<img style="max-width:100%" src="'+e['image']+'">'])
    body.append('<h2>空间分布描述</h2><pre>'+html.escape(json.dumps(spatial,ensure_ascii=False,indent=2))+'</pre><p>固定8×8网格占用仅用于描述聚集程度，没有调网格、选择新策略或推断多样性的因果作用。</p>')
    (output/'report.html').write_text('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>V14 保留案例</title><body style="max-width:1200px;margin:40px auto;font:16px/1.7 sans-serif">'+''.join(body)+'</body></html>',encoding='utf-8')
    (output/'source.py').write_bytes(Path(__file__).read_bytes())
    print(json.dumps(summary,ensure_ascii=False))


if __name__=='__main__':main()
