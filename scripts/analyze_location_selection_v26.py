"""Posthoc grid-resolution diagnosis and human closeout, not method selection."""
import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
from datetime import datetime, timezone
from pathlib import Path
import sys
import json
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
import run_location_selection_v26 as run
from run_visual_feedback_v23 import read,write,sha


def now():return datetime.now(timezone.utc).isoformat()


def allowed_grid(points,size,geometry,n):
    xy=np.asarray(points,float).reshape(-1,2)/np.asarray(size)
    cells=np.clip(np.floor(xy*n),0,n-1).astype(int)
    lo=cells/n;hi=lo+1/n;allowed=np.zeros(len(xy),bool)
    for box in geometry['boxes']:
        box=np.asarray(box,float)
        allowed|=((box[2:]>=lo-1e-12).all(1)&(box[:2]<=hi+1e-12).all(1))
    return allowed


def main():
    out=run.OUT;complete=read(out/'completion.json')
    for p,h in complete['artifact_sha256'].items():assert sha(p)==h,p
    audit=read(out/'final_audit.json');assert audit['complete']
    write(out/'posthoc_plan.json',dict(recorded_at_utc=now(),source_sha256=sha(Path(__file__)),
        status='Posthoc after primary results: no new method or hyperparameter selection.',
        question='If three-by-three GT bins cannot repair original A, what annotation resolution can?',
        grids=[3,6,12,24],base='A frozen original models, same complete A candidate set',
        diagnostic='GT-bbox intersecting cells only; all true candidates retained. Optimistic annotation-informed analysis, NOT a deployable accuracy or novelty claim.',
        primary_result_unchanged=True,new_vlm_calls=0))
    cases=read(run.V25/'data/public.json');dev=[c for c in cases if c['cohort']!='train']
    labels=read(run.V25/'data/labels.json');recs=run.records(dev)
    pred=read(out/'predictions.json');results=read(out/'results.json');cal=read(out/'calibration.json')
    lookup={(r['sample_id'],r['model'],r['arm']):r for r in pred['rows']}
    cov={r['sample_id']:r for r in pred['coverage']}
    spatial={r['sample_id']:r for r in pred['spatial'] if r['global_view']}
    grid_rows=[];failure_rows=[]
    for c in dev:
        sid=c['sample_id'];r=recs[sid];truth=run.labels_at(c,r['points'],labels)
        for model in run.NAMES:
            s=r['scores']['A__'+model];a=run.choose(s);a_hit=bool(a is not None and truth[a])
            for n in [3,6,12,24]:
                allowed=allowed_grid(r['points'],c['image_size'],labels[sid],n)
                assert allowed[truth].all()
                ix=np.flatnonzero(allowed);j=int(ix[s[ix].argmax()]) if len(ix) else a
                hit=bool(j is not None and truth[j]);assert not a_hit or hit
                grid_rows.append(dict(sample_id=sid,model=model,grid=n,index=j,hit=hit,
                    baseline_hit=a_hit,allowed_candidates=int(allowed.sum()),label_informed=True))
            if not a_hit:
                failure_rows.append(dict(sample_id=sid,cohort=c['cohort'],source=c['source'],model=model,
                    instruction=c['instruction'],covered=cov[sid]['covered'],selected_point=r['points'][a].tolist() if a is not None else None,
                    target_geometry=labels[sid],global_fields=spatial[sid]['fields'],
                    selected_bin=int(run.cell_ids(r['points'][a:a+1],c['image_size'])[0]) if a is not None else None,
                    target_bins=spatial[sid]['target_bins']))
    grid_summary={};attribution={};changes={};cv={}
    for cohort in ['new_development','old_val','all']:
        cc=[c for c in dev if cohort=='all' or c['cohort']==cohort];ids={c['sample_id'] for c in cc}
        grid_summary[cohort]={};attribution[cohort]={};changes[cohort]={}
        for family in ['original3k','full']:
            grid_summary[cohort][family]={str(n):100*float(np.mean([r['hit'] for r in grid_rows if r['sample_id'] in ids and r['model'].startswith(family+'_') and r['grid']==n])) for n in [3,6,12,24]}
            rr=[r for r in failure_rows if r['sample_id'] in ids and r['model'].startswith(family+'_')]
            attribution[cohort][family]=dict(mean_errors_per_fit=len(rr)/3,
                mean_no_correct_candidate=sum(not r['covered'] for r in rr)/3,
                mean_covered_selection_errors=sum(r['covered'] for r in rr)/3,
                mean_covered_errors_same_target_bins=sum(r['covered'] and r['selected_bin'] in r['target_bins'] for r in rr)/3)
            changes[cohort][family]={}
            for arm,base in [('A_position','A'),('B','A'),('B_position','B')]:
                repair=harm=moved=0
                for c in cc:
                    for seed in [1,2,3]:
                        a=lookup[c['sample_id'],f'{family}_s{seed}',arm];b=lookup[c['sample_id'],f'{family}_s{seed}',base]
                        repair+=a['hit'] and not b['hit'];harm+=b['hit'] and not a['hit'];moved+=a['point']!=b['point']
                changes[cohort][family][arm+'-'+base]=dict(mean_repairs=repair/3,mean_harms=harm/3,mean_changed_points=moved/3)
    for family in ['original3k','full']:
        cv[family]={}
        for base in ['A','B']:
            rr=[r for r in cal['train_leave_source_out'] if r['model'].startswith(family+'_') and r['base']==base]
            cv[family][base]=dict(original_accuracy=100*float(np.mean([r['original_hit'] for r in rr])),
                position_accuracy=100*float(np.mean([r['position_hit'] for r in rr])),
                note='Only scalar fitting excludes source; base heads can have seen training samples/sources.')
    write(out/'posthoc_diagnosis.json',dict(grid_summary=grid_summary,grid_rows=grid_rows,
        error_attribution=attribution,changes=changes,training_scalar_LOSO=cv,failures=failure_rows,
        label_informed_diagnostics_not_deployable=True))
    main=results['cohorts']['new_development'];old=results['cohorts']['old_val']
    lines=['V26结论：固定候选的显式九宫格位置约束未带来可靠提升。','',
        '执行完成：56训练RGB校准、36历史复用开发RGB评估，0次新增大模型生成，原6个评分器权重保持不变。',
        '5项边界/解析/置换/几何测试通过；92个固定候选集合一致；216条原预测复现；44,496次决策重放通过。','',
        '主要结果（新24张复用开发图，三个固定种子准确率均值，不是三个独立数据集）：',
        '组别 | 3k | 全量']
    for name,arm in [('原方法','A'),('原方法+位置软约束','A_position'),('结构化描述中的名称，固定原候选','B'),
                     ('结构化名称+位置软约束','B_position'),('原方法+静态中心','A_center'),('原方法+打乱位置（100次均值）','A_shuffled')]:
        lines.append(f'{name} | {main["methods"]["original3k"][arm]["accuracy_percent"]:.2f}% | {main["methods"]["full"][arm]["accuracy_percent"]:.2f}%')
    lines+=['','解释：',
        f'1. 24题中原候选覆盖{main["covered"]}/24。全量原评分器每个种子均15/24；9个错误中7个没有正确候选，另2个有正确候选但选中了同一目标九宫格内的干扰点。这个错误分解仅针对本次固定A候选和小开发子集，不能外推到所有实验。',
        '2. 全图描述按预设等分九宫格衡量：19/24有有效位置，12/19与目标标注格相交，5/24缺失或无效。位置名称有自然语言边界歧义，此统计不等于判定所有不相交描述都是幻觉。',
        '3. 给原方法提供真实目标相交的九宫格，3k仍56.94%、全量仍62.50%。九宫格无法排除上述同格干扰；只改善粗位置可靠性不足以解决这批误选。',
        '4. 在同一候选集与同一冻结评分器下，B名称比A分别低8.33、12.50个百分点。这个对照隔离了候选变化，但仍包含提示词/查询状态分布变化。不能据此断言描述里的信息已经消失，也不能断言所有角色化评分器都会失败。',
        '5. B全量加位置仅比B提高1.39pp，16比较校正99.6875%区间[0,6.94]pp；与打乱位置相比-0.57pp，区间[-5.43,3.54]pp。未证明有效的位置语义收益。',
        '6. 旧12张上，全量A原52.78%，加位置50.00%；3k两者47.22%。四个进阶门槛全部未通过，因此本轮未扩展到新的独立测试，也未替换主方法。','',
        '事后网格分辨率诊断（只用于理解瓶颈，使用GT，不是方法分数）：',
        json.dumps(grid_summary['new_development'],ensure_ascii=False),
        '细网格GT筛选仍保留原候选。其改善表示更细的位置证据有机会排除干扰，不能说明我们已能生成或读出这种证据。','',
        '下一步建议：先保住并修复正确候选覆盖，再在固定候选中测试细粒度目标身份/局部视觉或锚点关系。当前绝对九宫格约束暂停扩展；不据此投入全量重新训练。',
        '本轮B名称在冻结旧评分器下退化，后续若研究角色化读出，应保留实体/位置/锚点角色与token间关系，并为其匹配训练与对照，避免把分布不匹配直接解释成思路上限。','',
        '计算与泛化限制：本次0新增VLM调用源于缓存复用。部署A+位置需要额外B全图描述；B组还依赖B多视图状态与A候选，不能宣称同调用预算或免费加速。36张均为历史开发图，没有新的独立泛化或超过MVP的结论。',
        '粗GT格与细GT格只在独立诊断分支使用；生产预测不读取标注。标注仅在原训练集用于选择位置系数、在开发集用于评估。']
    (out/'conclusion.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    plot(out,main,grid_summary['new_development'])
    extra=[Path(__file__)]+[out/n for n in ['posthoc_plan.json','posthoc_diagnosis.json','conclusion.txt','method_note.txt','summary_zh.png','unit_tests.json','completion.json']]
    write(out/'delivery_manifest.json',dict(complete=True,finalized_at_utc=now(),main_execution_exit_code=0,
        primary_audit_passed=True,posthoc_source_sha256=sha(Path(__file__)),artifacts={str(p):sha(p) for p in extra}))
    reg=read(ROOT/'CURRENT_RESEARCH.json');reg['active_experiment'].update(analysis_path=str(out/'conclusion.txt'),
        static_figure_path=str(out/'summary_zh.png'),main_execution_exit_code=0,posthoc_diagnosis_path=str(out/'posthoc_diagnosis.json'))
    write(ROOT/'CURRENT_RESEARCH.json',reg)
    write(ROOT/'research/decisions/20261109_location_selection_v26_closeout.json',dict(completed_at_utc=now(),
        primary_positive_gates=results['positive_gates'],mainline_unchanged=True,analysis_path=str(out/'conclusion.txt'),
        next_priority='Candidate coverage plus fine-grained target identity/local discrimination; coarse nine-bin position expansion not supported.'))
    print((out/'conclusion.txt').read_text('utf-8'))


def plot(out,main,grid):
    os.environ.setdefault('MPLCONFIGDIR',str(out/'mplconfig'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    font=Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():plt.rcParams['font.family']=FontProperties(fname=str(font)).get_name()
    plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(13.6,5.5),layout='constrained')
    families=['original3k','full'];colors=['#3e6ca8','#c86f36']
    names=['原方法','原方法\n+位置','新描述名称','新描述名称\n+位置']
    arms=['A','A_position','B','B_position'];x=np.arange(len(arms))
    for f,col,offset,label in zip(families,colors,[-.19,.19],['3k评分器','全量评分器']):
        values=[main['methods'][f][a]['accuracy_percent'] for a in arms]
        bars=axes[0].bar(x+offset,values,.36,label=label,color=col)
        axes[0].bar_label(bars,fmt='%.1f',padding=3,fontsize=10)
        axes[1].plot([3,6,12,24],[grid[f][str(n)] for n in [3,6,12,24]],'o-',label=label,color=col)
    axes[0].set_xticks(x,names);axes[0].set_ylim(0,80);axes[0].set_ylabel('点击命中率（%）')
    axes[0].set_title('实际证据：固定原候选，位置约束未带来可靠收益');axes[0].legend(loc='upper right',fontsize=9)
    axes[1].axhline(100*main['covered']/main['n'],color='#777777',ls='--',label='原候选覆盖上限')
    axes[1].set_xticks([3,6,12,24],['3×3','6×6','12×12','24×24']);axes[1].set_ylim(40,80)
    axes[1].set_xlabel('使用真实目标标注限定的网格分辨率');axes[1].set_ylabel('诊断命中率（%）')
    axes[1].set_title('事后GT诊断：细位置可排除更多同格干扰');axes[1].legend(fontsize=9,loc='lower right')
    for ax in axes:ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('V26｜新24张历史开发图，3个固定拟合均值；全部组候选相同\n右图使用标注，只是事后瓶颈诊断；没有独立测试或MVP优势结论',fontsize=13)
    fig.savefig(out/'summary_zh.png',dpi=160);plt.close(fig)


if __name__=='__main__':main()
