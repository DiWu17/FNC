"""Additive closeout: preserve registered V19 and its posthoc audit unchanged."""
from datetime import datetime,timezone
from pathlib import Path
import json
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import run_bottleneck_v19 as probe

RUN=probe.OUT
AUDIT=RUN/'analysis/permutation_audit'
OUT=RUN/'analysis/closeout'
ARMS=['original','capacity','visual','visual_shuffled','listwise_only']
LABELS=['Original\n22k','Capacity\n67k','Local content\n87k','Shuffled content\n87k','Listwise only\n67k']


def plot(primary,audit):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':180})
    fig,axes=plt.subplots(1,3,figsize=(19,6.5),layout='constrained',gridspec_kw={'width_ratios':[1.25,1,1]})
    ax=axes[0];x=np.arange(5)
    for offset,cohort,color,label in [(-.24,'train','#537a9d','Train'),(0,'old_val','#acb7c4','Old development'),(.24,'new_development','#d69242','New development')]:
        values=[primary['cohorts'][cohort]['methods'][a]['accuracy_percent'] for a in ARMS]
        ax.bar(x+offset,values,.23,color=color,label=label)
    ax.set(xticks=x,xticklabels=LABELS,ylim=(0,85),ylabel='Point accuracy (%)',title='Fit improves; development does not')
    ax.tick_params(axis='x',labelsize=9);ax.legend(loc='upper left',frameon=False,fontsize=9)
    ax.text(0,-.22,'Same 2973 training queries and 2976 updates.\nFixed candidate pools; means of 3 fits; no ensemble.',transform=ax.transAxes,va='top',fontsize=9)
    ax=axes[1];names=['capacity','visual_vs_capacity','visual_alignment','loss_objective']
    for i,name in enumerate(names):
        r=primary['cohorts']['new_development']['comparisons'][name];v=r['delta_pp'];lo,hi=r['ci_pp']
        ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color='#537a9d',capsize=4)
        ax.annotate(f'{v:+.2f}',(v,i),xytext=(0,10),textcoords='offset points',ha='center',fontsize=9)
    ax.axvline(0,color='black',ls='--',lw=1)
    ax.set(yticks=range(4),yticklabels=['Capacity - original','Local - capacity','Aligned - shuffled','Listwise - capacity'],ylim=(3.7,-.7),xlim=(-6,7),xlabel='Accuracy difference (pp)',title='Registered new-development contrasts')
    ax.text(0,-.22,'98.75% paired bootstrap intervals; family 4.\nHistorically used development, not independent test.',transform=ax.transAxes,va='top',fontsize=9)
    ax=axes[2]
    pairs=[('train','C_shuffle_minus_aligned','Train: aligned C, shuffled input'),('new_development','C_shuffle_minus_aligned','New dev: aligned C, shuffled input'),
           ('train','D_reshuffle_minus_fixed','Train: shuffled D, new permutation'),('new_development','D_reshuffle_minus_fixed','New dev: shuffled D, new permutation')]
    for i,(cohort,name,_) in enumerate(pairs):
        r=audit['cohorts'][cohort]['comparisons'][name];v=r['delta_pp'];lo,hi=r['ci95_pp']
        ax.errorbar(v,i,xerr=[[v-lo],[hi-v]],fmt='o',color='#976549',capsize=4)
        ax.annotate(f'{v:+.2f}',(v,i),xytext=(0,10),textcoords='offset points',ha='center',fontsize=9)
    ax.axvline(0,color='black',ls='--',lw=1)
    ax.set(yticks=range(4),yticklabels=[p[2] for p in pairs],ylim=(3.7,-.7),xlim=(-23,5),xlabel='Fixed-weight perturbation difference (pp)',title='Posthoc spatial-association audit')
    ax.text(0,-.22,'Descriptive 95% intervals; no new training.\nD: mean over 3 fixed input permutations, not an ensemble.',transform=ax.transAxes,va='top',fontsize=9)
    fig.suptitle('V19 bottleneck diagnosis | Capacity, local content, and loss controls',fontsize=14)
    fig.savefig(OUT/'bottleneck_summary.png',bbox_inches='tight');plt.close(fig)


def main():
    OUT.mkdir(parents=True,exist_ok=True);probe.verify();inputs={}
    for folder in [RUN,AUDIT]:
        receipt=probe.read(folder/'completion.json');assert receipt['complete']
        inputs[str(folder/'completion.json')]=probe.sha(folder/'completion.json')
        for p,h in receipt['artifact_sha256'].items():assert probe.sha(p)==h,p;inputs[p]=h
    primary=probe.read(RUN/'results.json');audit=probe.read(AUDIT/'results.json');frozen=probe.read(RUN/'models/frozen.json')
    for model in frozen['models'].values():assert probe.sha(model['path'])==model['sha256']
    replay=0;max_error=0
    for cohort in ['old_val','new_development']:
        r=probe.read(RUN/f'evaluation/{cohort}.json')['original_replay'];assert r['point_agreement']==r['total']
        replay+=r['total'];max_error=max(max_error,r['max_score_error'])
    assert replay==1416 and max_error<1e-4
    paired=[]
    for s in [1,2,3]:
        for a,b in [('capacity','listwise_only'),('visual','visual_shuffled')]:
            ra=probe.read(RUN/f'models/{a}_s{s}.json');rb=probe.read(RUN/f'models/{b}_s{s}.json')
            assert ra['initial_state_sha256']==rb['initial_state_sha256'];paired.append([a,b,s])
    lines=['V19 瓶颈区分：完整结论',
           '12个新拟合与事后置换核查均已完成。原3k权重不变，所有结果保留三个种子；没有用fresh280调参或重测MVP。',
           '', '固定条件：2973条训练清单，2878条有候选进入拟合，95条空候选按原规则跳过；同一完整候选、2976更新、批8、AdamW2e-4、无早停。',
           '方法 | 参数量 | 训练准确率% | 旧开发172准确率% | 新开发300准确率%']
    for arm in ARMS:
        parameters=22050 if arm=='original' else probe.read(RUN/f'models/{arm}_s1.json')['parameter_count']
        values=[primary['cohorts'][c]['methods'][arm]['accuracy_percent'] for c in ['train','old_val','new_development']]
        lines.append(f'{arm} | {parameters} | '+' | '.join(f'{v:.2f}' for v in values))
    lines += ['', '预先登记的四项新开发比较（98.75%区间，四比较族）：']
    for name,r in primary['cohorts']['new_development']['comparisons'].items():
        lines.append(f'{name}: {r["delta_pp"]:+.2f}pp, 区间{r["ci_pp"]}, 达到+2pp且下界>0门槛={r["meets_registered_positive_gate"]}')
    lines += ['', '结论应如何调整：',
              '1. 当前这轮改进的直接障碍是把训练拟合转化为开发集选点能力。三倍容量提高训练准确率，却未提高两套开发集准确率。数据未变，候选覆盖也未变。',
              '2. 加入局部视觉分支使训练59.33%升到71.48%，但新开发58.67%降到57.67%，旧开发52.71%降到46.51%；当前视觉读出没有净泛化收益。',
              '3. 同结构、同初始化的固定打乱版本在训练也达到71.10%。追加固定权重核查中，给它换一套打乱对应，训练降到57.66%，新开发只从55.89%到55.15%。这直接显示该扩展模型学到了训练样本特定的对应关系，其收益不能外推到开发样本。',
              '4. 正确视觉版本在新开发被打乱后57.67%降到52.44%（事后描述性差-5.22pp，95%区间约[-8.47,-2.09]）。模型确实依赖局部对应；不能把没有净收益解释成局部视觉信息完全无用。',
              '5. 去掉辅助BCE相对扩容模型仅+0.78pp，校正区间跨零，没有支持它是当前主要修复点。',
              '', '证据强度与边界：',
              '已观察到：扩容/加特征能提高训练拟合；固定随机对应也能被拟合；这些训练收益没有改善开发准确率。',
              '较有依据的优先级：提高候选选择的跨界面泛化、抑制对样本特定关系的记忆，先于继续扩大当前读出或直接拼接更多缓存向量。',
              '尚未锁定：监督噪声、训练分布覆盖、表示与语言对齐、正则化/优化各自的因果份额。不能把原评分器全部错误都归为过拟合，也不能排除其他容量或视觉表示方案。',
              '本次视觉探针只使用原分辨率缓存的固定32维投影，不等于测试了所有局部视觉方案或更高分辨率验证器。',
              '原小评分器本身的训练/新开发条件命中率72.38%/68.48%；大视觉版本87.20%/67.32%。大模型出现的记忆现象不能不加区分地套到原模型上。',
              '这是复用开发集的机制诊断；所有区间条件于固定拟合。置换核查是事后描述性补充，不扩充原四比较族的主结论。没有独立超越MVP的证据。',
              '', '保留原3k主线。下一轮应围绕可迁移的语义—区域对应与抑制样本记忆设计新对照，数据/监督质量与视觉表示仍需分别验证。',
              '完整原始主比较：'+str(RUN/'results.txt'),
              '事后置换核查：'+str(AUDIT/'readout.txt')]
    (OUT/'readout.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    plot(primary,audit)
    probe.write(OUT/'verification.json',dict(complete=True,verified_at_utc=probe.now(),input_sha256=inputs,
        checkpoint_hashes_verified=15,original_point_replay_agreement=replay,original_max_score_error=max_error,
        paired_initialization_controls=paired,counterfactual_C_D_replays=6*(2973+172+300),
        original_registered_results_unchanged=True,source_sha256=probe.sha(__file__),mainline_unchanged=True))
    files=[OUT/n for n in ['readout.txt','bottleneck_summary.png','verification.json']]
    probe.write(OUT/'completion.json',dict(complete=True,completed_at_utc=probe.now(),artifact_sha256={str(p):probe.sha(p) for p in files}))
    print(json.dumps(dict(complete=True,readout=str(OUT/'readout.txt'),original_replay_agreement=replay,all15_checkpoint_hashes_verified=True)))


if __name__=='__main__':main()
