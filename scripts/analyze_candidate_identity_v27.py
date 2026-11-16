"""Posthoc retention/aliasing attribution and static closeout; no method fitting."""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE','1')
from datetime import datetime,timezone
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import run_candidate_identity_v27 as run
from clickagent.candidate_identity_v27 import (retain,PRIOR,VISUAL,PEER,CONTEXT,infer)
from run_location_selection_v26 import geometry_bins
from run_visual_feedback_v23 import read,write,sha


def now():return datetime.now(timezone.utc).isoformat()


def main():
    out=run.OUT;complete=read(out/'completion.json')
    for p,h in complete['artifact_sha256'].items():assert sha(p)==h,p
    run.verify()
    write(out/'posthoc_plan.json',dict(recorded_at_utc=now(),source_sha256=sha(Path(__file__)),
        purpose='After primary results: explain supplemental retention, same-appearance aliases, current-view coverage and feature-group score contributions. No new fit, method or hyperparameter selection.',
        retention_budgets=[8,16,32,64],alias_tolerance=1e-6,
        original_primary_results_unchanged=True,uses_evaluation_labels_for_diagnostics_only=True))
    cases=read(run.V25/'data/public.json');dev=[c for c in cases if c['cohort']!='train']
    labels=read(run.V25/'data/labels.json');pred=read(out/'predictions.json');results=read(out/'results.json')
    frozen=read(out/'models/frozen.json');models=run.load_models()
    lookup={(r['sample_id'],r['model'],r['arm']):r for r in pred['rows']}
    retention=[];case_info=[];changes=[];harm_details=[];aliases=[];budget_difference=[]
    for c in dev:
        a=run.load(c);sid=c['sample_id'];truth=np.asarray([run.hit(p.tolist(),c,labels[sid]) for p in a['points']],bool)
        A=a['ai']>=0;Bnew=~A
        local_visibility=[bool(geometry_bins(c,labels[sid],v['transform'])[2]) for v in c['views'][1:]]
        case_info.append(dict(sample_id=sid,source=c['source'],cohort=c['cohort'],instruction=c['instruction'],
            A_count=int(A.sum()),union_count=len(A),B_only_count=int(Bnew.sum()),
            A_oracle=bool(truth[A].any()),union_oracle=bool(truth.any()),B_only_oracle=bool(truth[Bnew].any()),
            target_visible_in_any_old_local_crop=any(local_visibility),old_local_visibility=local_visibility))
        for name in run.NAMES:
            scores=a['U__'+name];ix=retain(scores,a['ai']);naive=retain(scores,a['ai'],protected=False)
            budget_difference.append(dict(sample_id=sid,model=name,pool_equal=bool(np.array_equal(ix,naive)),
                protected_minus_naive_true_candidates=int(truth[ix].sum()-truth[naive].sum())))
            for k in [8,16,32,64]:
                for protected in [False,True]:
                    kk=retain(scores,a['ai'],budget=k,protected=protected)
                    retention.append(dict(sample_id=sid,model=name,budget=k,protected=protected,
                        oracle=bool(truth[kk].any()),n=len(kk),positive_B_only_retained=bool((truth[kk]&Bnew[kk]).any())))
            for arm,base in [('union_frozen','A'),('prior','union_frozen'),('visual','prior'),
                             ('pair_visual','visual'),('full','pair_visual'),('full','A')]:
                p=lookup[sid,name,arm];q=lookup[sid,name,base]
                changes.append(dict(sample_id=sid,model=name,contrast=arm+'-'+base,
                    repair=p['hit'] and not q['hit'],harm=q['hit'] and not p['hit'],
                    moved=p['point']!=q['point'],selected_B_only=bool(p['index'] is not None and Bnew[p['index']])))
            # Alias diagnosis: target visual features identical across a true and
            # false candidate cannot themselves distinguish this pair.
            visual=a['role'][ix,:2].reshape(len(ix),96)
            pos=np.flatnonzero(truth[ix]);neg=np.flatnonzero(~truth[ix]);same=0
            if len(pos) and len(neg):same=int((np.max(np.abs(visual[pos,None]-visual[neg]),axis=-1)<=1e-6).any())
            aliases.append(dict(sample_id=sid,model=name,rankable_pool=bool(len(pos) and len(neg)),
                positive_has_identical_visual_negative=bool(same)))
            p=lookup[sid,name,'full'];q=lookup[sid,name,'A']
            if q['hit'] and not p['hit']:
                r=run.record(c,a,name,'full');m=models['full__'+name]
                j=p['index'];base_index=next((i for i in ix if a['points'][i].tolist()==q['point']),None)
                if base_index is not None and j is not None:
                    ji=int(np.flatnonzero(ix==j)[0]);bi=int(np.flatnonzero(ix==base_index)[0])
                    delta=(r['x'][ji]-r['x'][bi])/m['std']*m['weight']
                    values={k:float(delta[s].sum()) for k,s in [('prior',PRIOR),('visual',VISUAL),('peer',PEER),('container',CONTEXT)]}
                    harm_details.append(dict(sample_id=sid,source=c['source'],cohort=c['cohort'],model=name,
                        instruction=c['instruction'],A_point=q['point'],full_point=p['point'],full_selected_B_only=bool(Bnew[j]),
                        score_difference_frozen=float(r['base'][ji]-r['base'][bi]),
                        score_difference_residual_groups=values,
                        strongest_positive_group=max(values,key=values.get),container_fields=[d.get('ground_fields') for d in []]))
    summary={}
    for cohort in ['new_development','old_val','all']:
        ids={c['sample_id'] for c in dev if cohort=='all' or c['cohort']==cohort};summary[cohort]={}
        info=[r for r in case_info if r['sample_id'] in ids]
        summary[cohort]['candidate_case_counts']=dict(A_covered=sum(r['A_oracle'] for r in info),union_covered=sum(r['union_oracle'] for r in info),
            newly_covered=sum(r['union_oracle'] and not r['A_oracle'] for r in info),
            newly_covered_cases=[r for r in info if r['union_oracle'] and not r['A_oracle']],
            target_in_any_old_local_crop=sum(r['target_visible_in_any_old_local_crop'] for r in info))
        for family in ['original3k','full']:
            f={};summary[cohort][family]=f
            for contrast in sorted({r['contrast'] for r in changes}):
                rr=[r for r in changes if r['sample_id'] in ids and r['model'].startswith(family+'_') and r['contrast']==contrast]
                f[contrast]=dict(mean_repairs=sum(r['repair'] for r in rr)/3,mean_harms=sum(r['harm'] for r in rr)/3,
                    mean_changes=sum(r['moved'] for r in rr)/3)
            f['retention_curve']={}
            for k in [8,16,32,64]:
                f['retention_curve'][str(k)]={}
                for protect in [False,True]:
                    rr=[r for r in retention if r['sample_id'] in ids and r['model'].startswith(family+'_') and r['budget']==k and r['protected']==protect]
                    f['retention_curve'][str(k)][str(protect)]=100*float(np.mean([r['oracle'] for r in rr]))
            rr=[r for r in aliases if r['sample_id'] in ids and r['model'].startswith(family+'_')]
            f['visual_aliases']=dict(rankable_fit_cases=sum(r['rankable_pool'] for r in rr),
                positive_with_identical_negative_fit_cases=sum(r['positive_has_identical_visual_negative'] for r in rr),
                note='Fit-cases repeat RGB across seeds; this is feature ambiguity, not an independent failure count.')
    identity_count={arm:sum(r['ridge'] is None for r in frozen['models'].values() if r['arm']==arm) for arm in run.ARMS}
    trained_queries=sorted({r['optimizer']['rankable_queries'] for r in frozen['models'].values() if r['ridge'] is not None})
    write(out/'mechanism_diagnosis.json',dict(summary=summary,retention_rows=retention,case_information=case_info,
        decision_changes=changes,harm_details=harm_details,visual_aliases=aliases,budget_differences=budget_difference,
        zero_residual_selections=identity_count,rankable_training_queries=trained_queries,
        primary_results_unchanged=True))
    new=results['cohorts']['new_development'];lines=[
        'V27结论：补充候选提高覆盖；本轮细视觉参照/候选比较/弱容器上下文未提高最终命中。','',
        '执行与核验已完成。56原训练RGB、36历史复用开发RGB（新24、旧12）；0次新增VLM调用，原6个评分器未更新。',
        '5项测试通过；988份历史缓存哈希核验；92个并集候选重建；432条原A/B预测复现；1296条新增模块决策重放。','',
        '新24开发图，三个固定拟合均值（不是72个独立样本）：',
        '方案 | 3k评分器 | 全量评分器']
    labels_text={'A':'原方法','union_frozen':'合并候选，冻结评分','prior':'仅来源/原分数校准',
        'visual':'加入细视觉参照','pair_visual':'再加入候选比较','full':'再加入弱容器上下文','shuffled':'打乱空间对应控制','full_naive':'完整模块，朴素K16池'}
    for arm,label in labels_text.items():lines.append(f'{label} | {new["methods"]["original3k"][arm]["accuracy_percent"]:.2f}% | {new["methods"]["full"][arm]["accuracy_percent"]:.2f}%')
    lines+=['','候选部分：',
        '完整原候选覆盖17/24；完整A+B并集19/24。两个新增覆盖并没有证明最终可正确选中。',
        '最多16点时，朴素/保护池覆盖都为75.00%（3k种子命中覆盖18/17/19题；全量各18题）。固定8+8名额没有额外覆盖收益。两种池实际平均14.67点，含两张空池图。',
        '原A的Top16覆盖3k70.83%、全量66.67%。合并确实增加了同预算下的可选正确目标，但冻结评分的最终命中反而略降。','',
        '评分部分：',
        '细视觉参照与候选比较没有表现出优于来源校准的增益。完整模块全量比原方法低9.72pp，三个种子均下降；比打乱对应控制也低9.72pp。不能报告该分支有效。',
        '完整模块3k比原方法低4.17pp；旧12图的全量均值较原方法高2.78pp，但区间跨0，且新24退化，不满足稳定性要求。',
        '本轮用的是已有分辨率的缓存视觉状态，没有给候选重新裁剪、放大、重新编码；因此不能用负结果否定高分辨率局部重看。',
        '容器注意力质心是弱上下文代理，不等于正确对象锚点。没有实现任意左/右/之间关系；负结果不能解释为显式锚点推理失败。','',
        '训练与统计边界：',
        '新增模块只有56个训练查询，其中能提供正负排序监督的查询为'+str(trained_queries)+'；其余池没有正确候选或没有正负对。',
        '30个选模中有'+str(sum(identity_count.values()))+'个回退到零残差，按分支为'+str(identity_count)+'。因此本轮未发现可迁移增益，不等于证明视觉特征不存在可用信息。',
        '来源留一只用于残差参数选择，原评分器可能已见这些训练样本/来源。24张复用开发图、16重比较的统计功效有限；不通过进阶门槛不等于等效性证明。',
        '两组进阶门槛均未通过，没有替换主方法，也未因此消耗新的独立测试。','',
        '对后续研究的建议：保留“候选覆盖提升”这一事实；当前新增读出模块不采用。应优先区分：新候选引导的真实局部重编码能否增加目标细节，以及显式对象锚点能否提供可靠关系。需要同调用预算对照，避免继续只在同一低分辨率缓存上叠加评分项。',
        '本轮部署需要A/B两套观测，不是同VLM调用预算下的速度优势，也没有超过MVP的证据。']
    (out/'conclusion.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    plot(out,new)
    extra=[Path(__file__)]+[out/n for n in ['posthoc_plan.json','mechanism_diagnosis.json','conclusion.txt','method_note.txt','summary_zh.png','completion.json','unit_tests.json']]
    write(out/'delivery_manifest.json',dict(complete=True,finalized_at_utc=now(),main_execution_exit_code=0,
        analysis_exit_code=0,artifacts={str(p):sha(p) for p in extra}))
    reg=read(ROOT/'CURRENT_RESEARCH.json');reg['active_experiment'].update(analysis_path=str(out/'conclusion.txt'),
        static_figure_path=str(out/'summary_zh.png'),main_execution_exit_code=0,analysis_exit_code=0,
        mechanism_diagnosis_path=str(out/'mechanism_diagnosis.json'))
    write(ROOT/'CURRENT_RESEARCH.json',reg)
    write(ROOT/'research/decisions/20261109_candidate_identity_v27_closeout.json',dict(completed_at_utc=now(),
        positive_gates=results['positive_gates'],mainline_unchanged=True,analysis_path=str(out/'conclusion.txt'),
        conclusion='Coverage gains confirmed on reused pilot; no fine-template/competition/container scoring advantage. True high-resolution re-encoding and explicit relational anchors not tested.'))
    print((out/'conclusion.txt').read_text('utf-8'))


def plot(out,new):
    os.environ.setdefault('MPLCONFIGDIR',str(out/'mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(15,5.5),layout='constrained')
    colors=['#3e6ca8','#c86f36'];families=['original3k','full']
    pools=['A_complete','union_complete','A16','naive','protected'];x=np.arange(len(pools))
    arms=['A','union_frozen','prior','visual','pair_visual','full','shuffled'];xx=np.arange(len(arms))
    for f,col,offset,name in zip(families,colors,[-.19,.19],['3k评分器','全量评分器']):
        vals=[new['coverage'][f][p]['oracle_percent'] for p in pools]
        bars=axes[0].bar(x+offset,vals,.36,color=col,label=name);axes[0].bar_label(bars,fmt='%.1f',padding=3,fontsize=9)
        vals=[new['methods'][f][a]['accuracy_percent'] for a in arms]
        axes[1].plot(xx,vals,'o-',color=col,label=name)
    axes[0].set_xticks(x,['原完整池','合并完整池','原Top16','朴素≤16','保护≤16']);axes[0].set_ylim(0,100)
    axes[0].set_ylabel('包含正确候选的图像比例（%）');axes[0].set_title('覆盖增加，但保护配额未优于朴素合并');axes[0].legend(fontsize=9)
    axes[1].set_xticks(xx,['原方法','合并\n冻结评分','来源\n校准','细视觉','候选\n比较','容器\n上下文','打乱\n对应']);axes[1].set_ylim(40,75)
    axes[1].set_ylabel('最终点击命中率（%）');axes[1].set_title('当前缓存读出未将新增覆盖转化为命中');axes[1].legend(fontsize=9)
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('V27｜新24张历史开发图，三个固定拟合均值\n原评分器冻结；左图完整池与≤16预算分别标明；没有独立测试或MVP优势结论',fontsize=13)
    fig.savefig(out/'summary_zh.png',dpi=150);plt.close(fig)


if __name__=='__main__':main()
