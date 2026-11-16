"""Audit frozen-scorer development studies and write text/static figures."""
from pathlib import Path
import hashlib
import json
import os

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'runs/frozen_geometry_v18/analysis'


def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    verified={}
    for run in ['frozen_alternatives_v18','frozen_geometry_v18','coordinate_control_v18']:
        folder=ROOT/'runs'/run;receipt=read(folder/'completion.json')
        assert receipt['complete']
        for p,h in receipt['artifact_sha256'].items():assert sha(p)==h,p
        protocol=read(folder/'protocol.json')
        for p,h in protocol.get('hashes',protocol.get('input_hashes',{})).items():assert sha(p)==h,p
        verified[run]=dict(completion_sha256=sha(folder/'completion.json'))
    frozen=read(ROOT/'runs/quality_readout_v15_3k/models/frozen.json')['models']
    for s in [1,2,3]:
        r=frozen[f'expanded_s{s}'];assert sha(r['path'])==r['sha256']
    geometry=ROOT/'runs/frozen_geometry_v18'
    replay=read(geometry/'completion.json')['replay']
    assert replay['point_agreement']==replay['n_seed_predictions']==1416
    indices=read(ROOT/'runs/full_scaling_v18/data/development_indices.json')
    by_source={}
    for cohort in ['old_val','new_development']:
        geo=read(geometry/f'evaluation/{cohort}.json')
        fixed=read(ROOT/f'runs/coordinate_control_v18/evaluation/{cohort}.json')
        by_source[cohort]={}
        for source in sorted({e['source'] for e in indices[cohort]}):
            ids=[i for i,e in enumerate(indices[cohort]) if e['source']==source]
            import numpy as np
            by_source[cohort][source]=dict(n=len(ids),
                original=100*float(np.asarray(geo['hit_vectors']['original'])[:,ids].mean()),
                center_rescore=100*float(np.asarray(geo['hit_vectors']['coarse_center'])[:,ids].mean()),
                center_fixed_selection=100*float(np.asarray(fixed['hit_vectors']['coarse_center'])[:,ids].mean()))
    OUT.mkdir(parents=True,exist_ok=True)
    lines=['V18：全量训练期间，冻结现有评分器的并行开发实验解读',
        '现有2973例模型、提议器和8B骨干权重均未更改；未覆盖主线。',
        '第一组：完整候选直接argmax在新开发集为58.67%。所有5个简单跨视图融合规则的点估计均更低，未得到提升证据。',
        '第二组：候选改为粗网格中心并重建特征/重新评分，新开发58.67%→62.67%，旧开发52.71%→54.07%。',
        '第三组：保持原选中候选来源，仅移动最终坐标，新开发58.67%→63.11%，旧开发52.71%→53.49%。',
        '新开发集仅坐标移动的差值+4.44百分点，描述性95%区间[+1.22,+7.79]。',
        '重建/重排相对仅移动坐标为-0.44百分点，95%区间[-1.55,+0.56]，没有额外收益证据。',
        '这些结果提示当前精细坐标解码值得检查，不足以证明所有细化偏移有害；原候选来源、跨视图重复与数据分布均会影响结论。',
        '来源表现存在权衡：粗中心重评分改善GroundCUA及ShowUI-web，AMEX下降，不是统一的鲁棒性收益。',
        '全部是多次使用的开发集、多个探索性对比；不是MVP测试截图上的成绩，不可与38.67%直接比较。',
        '已有1416个种子级原始预测CPU回放全部与缓存选点一致，分数最大误差约5.13e-6。',
        '新280张截图在任何预测前独立预留；全量模型完成并冻结后，自动进行MVP直接对照。联合三项主比较使用98.333%区间。',
        '', '分来源结果：',json.dumps(by_source,ensure_ascii=False,indent=2)]
    (OUT/'interpretation.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    os.environ.setdefault('MPLCONFIGDIR',str(OUT/'matplotlib_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels=['View RRF','Support weight 0.25','Support weight 0.50','Support weight 1.00','Top-4 consensus',
        'Coarse center + rescore','Half to center + rescore','Child center + rescore','Coarse center, fixed selection']
    fig,axes=plt.subplots(1,2,figsize=(12,6.8),sharey=True,layout='constrained')
    for ax,cohort,title in zip(axes,['old_val','new_development'],['Old development (172 queries)','New development (300 queries / 206 RGB groups)']):
        rules=read(ROOT/f'runs/frozen_alternatives_v18/evaluation/{cohort}.json')
        geo=read(geometry/f'evaluation/{cohort}.json')
        fixed=read(ROOT/f'runs/coordinate_control_v18/evaluation/{cohort}.json')
        rows=[rules['comparisons'][m] for m in ['view_rrf','support_025','support_050','support_100','top4_consensus']]
        rows += [geo['comparisons'][m] for m in ['coarse_center','half_to_center','child_center']]
        rows += [fixed['contrasts']['coarse_center']['coordinate_only_vs_original']]
        for i,r in enumerate(rows):
            x=r['delta_pp'];lo,hi=r['ci95_pp']
            ax.errorbar(x,i,xerr=[[x-lo],[hi-x]],fmt='o',capsize=3,color='#246ea1' if x>0 else '#ae4a42')
        ax.axvline(0,color='gray',lw=1);ax.grid(axis='x',alpha=.2);ax.set_title(title)
        ax.set_xlabel('Hit-rate change vs full-union argmax (pp)')
    axes[0].set_yticks(range(len(labels)),labels);axes[0].invert_yaxis()
    fig.suptitle('Frozen scorer: candidate fusion and coordinate decoding\nFixed 3-seed mean; descriptive paired 95% intervals; reused development only',fontsize=12)
    fig.savefig(OUT/'frozen_trials.png',dpi=180);plt.close(fig)
    files=[OUT/'interpretation.txt',OUT/'frozen_trials.png']
    audit=dict(complete=True,verified_studies=verified,original_prediction_replay=replay,
        original3k_checkpoints_unchanged=True,by_source=by_source,
        artifacts_sha256={str(p):sha(p) for p in files},finalizer_sha256=sha(__file__))
    (OUT/'verification.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('\n'.join(lines[:13]),flush=True)


if __name__=='__main__':main()
