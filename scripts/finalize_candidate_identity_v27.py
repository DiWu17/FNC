"""Deliver primary V27 and its explicitly recorded normalization correction."""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
from datetime import datetime,timezone
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import run_candidate_identity_v27 as run
from run_visual_feedback_v23 import read,write,sha


def now():return datetime.now(timezone.utc).isoformat()


def main():
    out=run.OUT;control=out/'scaling_control';verified=0
    for comp in [out/'completion.json',control/'completion.json']:
        for p,h in read(comp)['artifact_sha256'].items():assert sha(p)==h,p;verified+=1
    run.verify()
    for p,h in read(control/'protocol.json')['sealed_sha256'].items():assert sha(p)==h,p
    primary=read(out/'results.json');corrected=read(control/'results.json');diagnosis=read(out/'mechanism_diagnosis.json')
    frozen=read(control/'models/frozen.json');identities={arm:sum(r['ridge'] is None for r in frozen['models'].values() if r['arm']==arm) for arm in run.ARMS}
    oldrows=read(out/'predictions.json')['rows'];lookup={(r['sample_id'],r['model'],r['arm']):r for r in oldrows}
    newrows=read(control/'predictions.json')['rows'];newlookup={(r['sample_id'],r['model'],r['arm']):r for r in newrows}
    cases=read(run.V25/'data/public.json');cc=[c for c in cases if c['cohort']=='new_development']
    changes={}
    for family in ['original3k','full']:
        repairs=harms=0
        for c in cc:
            for seed in [1,2,3]:
                a=newlookup[c['sample_id'],f'{family}_s{seed}','full'];b=lookup[c['sample_id'],f'{family}_s{seed}','A']
                repairs+=a['hit'] and not b['hit'];harms+=b['hit'] and not a['hit']
        changes[family]=dict(mean_repairs=repairs/3,mean_harms=harms/3)
    info=diagnosis['summary']['new_development']['candidate_case_counts']['newly_covered_cases']
    newly=[]
    for c in info:
        r=dict(sample_id=c['sample_id'],instruction=c['instruction'],by_family={})
        for f in ['original3k','full']:
            r['by_family'][f]=[newlookup[c['sample_id'],f'{f}_s{s}','full']['hit'] for s in [1,2,3]]
        newly.append(r)
    write(out/'final_interpretation.json',dict(normalization_correction_exploratory=True,
        primary_artifacts_unchanged=True,corrected_full_vs_original_changes=changes,
        corrected_zero_residual_selections=identities,newly_covered_outcomes=newly,
        primary_and_correction_artifact_hashes_verified=verified,new_vlm_calls=0,independent_test=False))
    lines=['V27最终结论（含归一化修正）：补充候选有覆盖价值；当前缓存视觉/候选比较/容器读出未提高最终命中。','',
        '范围：56原训练RGB训练新增<=322系数残差；36历史复用开发RGB评估（新24主比较/旧12稳定性）。原3k/全量各3个评分器保持冻结。',
        '所有实验、训练选择、预测重放和文件核验已完成；0次新增VLM调用。没有替换已采用的V15_3k主线。','',
        '最终应使用以下修正尺度结果（新24图，三个固定拟合均值，不是72个独立样本）：',
        '方案 | 3k评分器 | 全量评分器']
    r=corrected['cohorts']['new_development']['methods'];p=primary['cohorts']['new_development']['methods']
    table=[('原方法','old','A'),('合并候选，冻结原评分','old','union_frozen'),
        ('仅来源/原分数校准','new','prior'),('细视觉参照（修正尺度）','new','visual'),
        ('细视觉+候选比较（修正尺度）','new','pair_visual'),('完整弱容器上下文（修正尺度）','new','full'),
        ('打乱空间对应控制（修正尺度）','new','shuffled')]
    for label,regime,arm in table:
        src=p if regime=='old' else r
        lines.append(f'{label} | {src["original3k"][arm]["accuracy_percent"]:.2f}% | {src["full"][arm]["accuracy_percent"]:.2f}%')
    lines+=['','1. 补充候选的价值与限度',
        '完整候选覆盖17/24→19/24，新增覆盖任务为Latest和Sofascore。并集平均362.58个点，原A平均184.58个点。',
        '最多16点时，朴素合并与A8+B新增8保护池覆盖均75.00%；配额保护没有额外收益。合并提升了同预算候选覆盖，却未提升最终命中。',
        '修正完整模块选中这两个新增目标的种子情况：'+str(newly),'',
        '2. 归一化设置的问题与修正',
        '最初归一化使用std下限0.05。训练集full_s1候选池中，视觉/候选比较通道std中位数约0.00149/0.00083，容器上下文约0.219。前者被相对压低几十倍，容器项主导了一些误选。',
        '这影响了对第一轮负结果的解释。因此保留全部原协议/代码/模型/预测，单独登记scaling_control，按每个训练折的实际std归一化（下限1e-4，常量通道特殊处理）。损失、候选、特征、训练数据、LOSO选模和零残差回退均固定。',
        '修正不是按开发最高分挑选归一化。它在主实验结果已知后发起，属于探索性修正；不能作为新的独立确认。原完整模块52.78%/52.78%，修正后55.56%/58.33%，仍低于原方法56.94%/62.50%。','',
        '3. 当前能支持的判断',
        '修正后细视觉与候选比较没有超越来源校准；完整上下文没有优于打乱对应控制。全量完整相对原方法-4.17pp，校正99.6875%区间[-16.67,4.43]pp，没有正收益证据。',
        '修正完整相对原方法的修复/伤害（每种子均值）：'+str(changes),
        '新增模块的数据仍很少：原训练56个查询，约41–43个有正负候选排序监督。修正后零残差回退数（每分支共6拟合）：'+str(identities)+'。视觉和候选比较各5/6拟合回退；这表示该训练设定未学出稳定收益，不能断言视觉信息本身无用。',
        '旧12图上，修正完整3k50.00%（原47.22%），全量47.22%（原52.78%）；没有形成一致改善。','',
        '4. 未被这轮试验回答的问题',
        '本轮只读已有分辨率的缓存。没有按补充候选重新裁剪、放大、重新编码截图，不能将结果解释为高分辨率局部重看无效。',
        '容器注意力质心只是弱上下文参照，并非真实对象锚点。任意“左/右/位于两者之间”的对象关系尚未实现和检验。',
        '候选视觉向量不完全相同也不等于包含可迁移目标语义；严格判别力与几何精度仍需更有针对性的实验。','',
        '处理决定：保留覆盖提升的观察，不采用当前新增评分模块。若继续，应优先做相同观测预算下的新候选引导局部重编码，以及真实对象锚点的显式关系对照；先验证新增证据，再决定扩训练。',
        '数据与计算限制：36图均为历史开发图，三个种子不独立。部署需A/B两套观测，本轮0新增调用源于缓存复用。没有独立泛化、同预算速度或超过MVP的结论。','',
        '核验：5项测试；988历史缓存哈希；92并集重建；432原A/B预测；1296主实验+1080修正实验决策重放；640归一化前后等价性检查，最大误差5.78e-15。']
    (out/'conclusion_final.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    plot(out,p,r)
    paths=[Path(__file__),out/'conclusion_final.txt',out/'final_interpretation.json',out/'summary_final_zh.png',
        out/'completion.json',out/'delivery_manifest.json',control/'completion.json',control/'protocol.json',control/'final_audit.json']
    write(out/'final_delivery_manifest.json',dict(complete=True,finalized_at_utc=now(),main_execution_exit_code=0,
        normalization_execution_exit_code=0,analysis_exit_code=0,artifacts={str(p):sha(p) for p in paths}))
    write(out/'final_status.json',dict(complete=True,phase='primary_correction_analysis_and_audit_complete',updated_at_utc=now(),
        main_execution_exit_code=0,normalization_execution_exit_code=0,analysis_exit_code=0,new_vlm_calls=0,mainline_unchanged=True))
    reg=read(ROOT/'CURRENT_RESEARCH.json');before=read(out/'registry_before.json');assert reg['models']==before['models'] and reg['run_id']==before['run_id']
    reg['active_experiment'].update(status='complete',phase='primary_correction_analysis_and_audit_complete',
        status_path=str(out/'final_status.json'),analysis_path=str(out/'conclusion_final.txt'),static_figure_path=str(out/'summary_final_zh.png'),
        final_delivery_manifest_path=str(out/'final_delivery_manifest.json'),normalization_execution_exit_code=0)
    for entry in reg['completed_experiments']:
        if entry['run_id']==out.name:entry.update(analysis_path=str(out/'conclusion_final.txt'),normalization_control_path=str(control/'completion.json'))
    write(ROOT/'CURRENT_RESEARCH.json',reg)
    write(ROOT/'research/decisions/20261109_candidate_identity_v27_normalization_closeout.json',dict(completed_at_utc=now(),
        mainline_unchanged=True,analysis_path=str(out/'conclusion_final.txt'),normalization_correction_documented=True,
        conclusion='Fine visual/competition residual shows no advantage under the tested training regime even after correcting relative feature scaling. True high-resolution re-encoding and explicit object relations remain untested.'))
    (out/'SESSION_STATE.txt').write_text('V27主实验、归一化设置修正对照、两轮预测重放、最终中文分析与静态图、历史文件核验全部完成。退出码均0。无运行中任务。原方法不替换。最终结论conclusion_final.txt；原始结果与第一次报告均保留。\n',encoding='utf-8')
    print((out/'conclusion_final.txt').read_text('utf-8'))


def plot(out,old,new):
    os.environ.setdefault('MPLCONFIGDIR',str(out/'mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(14,5.8),layout='constrained')
    labels=['原方法','合并\n冻结评分','视觉\n原尺度','视觉\n修正尺度','完整\n原尺度','完整\n修正尺度','打乱\n修正尺度']
    items=[('old','A'),('old','union_frozen'),('old','visual'),('new','visual'),('old','full'),('new','full'),('new','shuffled')]
    for ax,f,color,name in zip(axes,['original3k','full'],['#3e6ca8','#c86f36'],['3k评分器','全量评分器']):
        values=[(old if src=='old' else new)[f][arm]['accuracy_percent'] for src,arm in items]
        bars=ax.bar(np.arange(len(values)),values,color=color,width=.65)
        for i in [2,4]:bars[i].set_alpha(.4)
        ax.bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
        ax.axhline(old[f]['A']['accuracy_percent'],color='#555555',ls='--',lw=1)
        ax.set_xticks(np.arange(len(values)),labels);ax.set_ylim(0,78);ax.set_ylabel('最终点击命中率（%）');ax.set_title(name)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('V27｜主实验与归一化设置修正：修正后仍未优于原方法\n新24张历史开发图、三个固定拟合均值；浅色为原设置；修正属于事后探索，无独立确认',fontsize=13)
    fig.savefig(out/'summary_final_zh.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
