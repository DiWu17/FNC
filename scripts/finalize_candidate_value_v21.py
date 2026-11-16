"""Readable interpretation and supplemental geometry/QA; no new policy evaluation."""
import sys,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
import analyze_candidate_value_v21 as a

def main():
    out=a.OUT;stats=a.read(out/'statistics.json');cases=a.read(out/'cases.json');rows=a.read(out/'shortlisted_candidates.json')
    assert len(cases)==100 and len({c['group'] for c in cases})==100
    primary=a.read(a.RUN/'results.json')
    for co in ['new_development','old_val','all']:
        for seed in range(3):
            x=stats[co][str(seed+1)];m=primary['cohorts'][co]['methods']['augmented_verify']
            assert x['actual_aug_repair']==m['repaired_seed_counts'][seed]
            assert x['actual_aug_harm']==m['broken_seed_counts'][seed]
    s=stats['new_development']['1'];all_s=stats['all']['1']
    def geometry(c):
        g=c['label'];w,h=c['image_size']
        if g['type']=='bbox':b=max(g['boxes'],key=lambda b:(b[2]-b[0])*(b[3]-b[1]))
        else:
            p=max(g['polygons'],key=len);p=np.asarray(p);b=[*p.min(0),*p.max(0)]
        return dict(width_px=(b[2]-b[0])*w,height_px=(b[3]-b[1])*h,area_fraction=(b[2]-b[0])*(b[3]-b[1]))
    geometry_stats={}
    for name,pred in [('mvp_correct',lambda c:c['mvp_hit']),('mvp_wrong_full_reachable',lambda c:not c['mvp_hit'] and c['full_pool_hit']),('shortlist_miss',lambda c:not c['mvp_hit'] and c['full_pool_hit'] and not c['seeds'][0]['pool_hit'])]:
        cc=[c for c in cases if c['cohort']=='new_development' and pred(c)];gg=[geometry(c) for c in cc]
        geometry_stats[name]=dict(n=len(cc),medians={k:float(np.median([g[k] for g in gg])) for k in ['width_px','height_px','area_fraction']})
    syntax_cases=[c for c in cases if c['strict_coordinate_format_failure']]
    syntax=dict(flagged=2,flagged_case_ids=[c['sample_id'] for c in syntax_cases],repair_counts_per_seed=[sum(c['seeds'][s]['repair'] for c in syntax_cases) for s in range(3)],
        scope='Posthoc text syntax diagnostic, not automatic invalidation; baseline unchanged.')
    origin={}
    for co in ['new_development','all']:
        origin[co]={}
        for cls in ['potential_repair','redundant_correct','failed_alternative','distractor']:
            rr=[r for r in rows if r['seed']==1 and r['value_class']==cls and (co=='all' or r['cohort']==co)]
            origin[co][cls]=dict(n_points=len(rr),global_origin=int(sum(r['global_origin'] for r in rr)),local_origin=int(sum(r['local_origin'] for r in rr)))
    conflict_id='expansion15:groundcua:b69866ebe0c7f23360377d15'
    source=None
    for line in (ROOT/'runs/quality_readout_v15_3k/data/new_reserve.jsonl').read_text(encoding='utf-8-sig').splitlines():
        c=json.loads(line)
        if c['sample_id']==conflict_id:source=c;break
    review=dict(posthoc=True,scope='Six outcome-stratified screenshot panels inspected;not an exhaustive annotation quality audit.',
        suspected_conflict=dict(sample_id=conflict_id,instruction=source['instruction'],source_annotation_text=source['source_annotations'][0]['text'],
            source_bbox=source['target_geometry'],observation='Instruction says select desktop.7z, but source annotation text is287 and box is at status bar. MVP points to file row. This is an instruction-label ambiguity,not evidence alone of semantic ranking failure.',
            action='Preserve all original labels/scores; flag for separate data QA before new training.'),
        illustrative_patterns={'repair_near':'MVP and added points lie in same input, but only added point overlaps tiny annotated text cursor;precision matters.',
            'repair_far':'Added point reaches more-options control in the content panel instead of similar bottom-navigation control;context matters.',
            'harm':'Correct lift-arrow candidate and MVP are present, but verifier switches to adjacent toolbar icon;hard negative discrimination matters.',
            'verification_miss':'Correct annotated erase point is available but verifier picks another toolbar location;word-to-region matching remains ambiguous.'})
    a.write(out/'qualitative_review.json',review);a.write(out/'additional_diagnostics.json',dict(geometry=geometry_stats,mvp_syntax=syntax,candidate_origin=origin))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.font_manager import FontProperties
    fp=FontProperties(fname='C:/Windows/Fonts/msyh.ttc');plt.rcParams['font.family']=fp.get_name();plt.rcParams['axes.unicode_minus']=False
    fig,axes=plt.subplots(1,2,figsize=(12.4,4.6))
    h=s['best_correct_rank_histogram'];vv=[h.get('1',0)+h.get('2',0),h.get('3-4',0)+h.get('5-8',0),h.get('9-16',0),h.get('>16',0)]
    bars=axes[0].bar(range(4),vv,color=['#2878B5','#DD8452','#DD8452','#DD8452']);axes[0].bar_label(bars,padding=3)
    axes[0].set_xticks(range(4),['前 2 名','第 3—8 名','第 9—16 名','16 名以后']);axes[0].set_ylabel('截图数');axes[0].set_ylim(0,9)
    axes[0].set_title('15 例完整池含正确点：首个正确点排在哪里？')
    names=['scorer','spatial_view_support','closer_to_MVP','local_contrast'];labels=['现有评分','跨视图区域支持','离 MVP 更近','局部对比度'];yy=np.arange(4)
    for offset,scope,color,label in [(-.18,'full','#2878B5','整个候选池（15 张）'),(.18,'top16','#DD8452','仅前 16 候选（12 张）')]:
        vals=[s['ranking_aucs'][scope][k]['mean'] for k in names];bars=axes[1].barh(yy+offset,vals,.36,color=color,label=label)
        axes[1].bar_label(bars,fmt='%.2f',padding=3,fontsize=9)
    axes[1].set_yticks(yy,labels);axes[1].invert_yaxis();axes[1].axvline(.5,color='gray',linestyle=':');axes[1].set_xlim(0,1.05)
    axes[1].set_xlabel('同一截图内排序 AUC（0.5 为随机，1 为完美）');axes[1].legend(fontsize=8,loc='lower right');axes[1].set_title('筛掉明显错误容易，分清相似高分候选更难')
    fig.suptitle('V21 候选价值分析：新增开发集 72 张，固定 S1；回溯分析，不是独立验证',fontsize=12)
    fig.tight_layout();fig.savefig(out/'readout_zh.png',dpi=180);plt.close(fig)

    lines=['有价值的补充候选：V21 回溯分析','',
        '范围：100张历史复用开发截图（新增72/旧28），一图一查询。没有训练、没有新模型推理、没有改动任何V21主结果。',
        '主图固定S1；S2/S3只做敏感性描述，不挑最好种子，不把三拟合或数万个候选当独立样本。排序区间按截图重采样，属于探索性关联。','',
        '1. 什么叫有价值',
        '潜在纠错点：MVP错、该点对；新增覆盖点：MVP原生候选都不对、该点对；实际修复：复核最终真的选中正确点。',
        'MVP已经对时的正确补充点只是冗余；错误补充点只有被选中才造成误改。几何新坐标不等于新的语义区域。',
        '理想直接监督是相对MVP的命中增量 y(候选)-y(MVP)：+1纠错、0无净变化、-1误改。这些是训练/分析标签，部署时不可知。','',
        '2. 价值损失在哪里（新增72，S1）',
        f'MVP错22例，完整我方候选能覆盖15例；补充短名单覆盖7例，实际修复6例，同时误改6例。',
        '15例中的首个正确点：前2名7例，第3—8名1例，第9—16名4例，16名以后3例。',
        '只取前2名会漏8/15；扩大到前16名的oracle覆盖为12/22个MVP错误，前32为14/22。这只是正确点是否存在，绝不是新方法准确率。',
        '实际新增135个点：11个潜在纠错点、65个冗余正确点、26个在MVP错误时仍错误的点、33个在MVP正确时的干扰点。候选点数不等于独立截图数。','',
        '3. 可见特征有没有用（新增72，S1）']
    for name,key in [('评分器分数','scorer'),('距离MVP更近','closer_to_MVP'),('跨视图区域支持','spatial_view_support'),('跨视图完全同坐标计数','exact_view_count'),('包含点的视图数','active_view_count'),('三拟合前8名计数','seed_top8_count'),('局部灰度对比度','local_contrast')]:
        full=s['ranking_aucs']['full'][key];top=s['ranking_aucs']['top16'][key]
        lines.append(f'{name}：全池AUC={full["mean"]:.3f}（n={full["n_screenshots"]}），前16内AUC={top["mean"]:.3f}，95%截图bootstrap区间{[round(x,3) for x in top["ci95"]]}（n={top["n_screenshots"]}）。')
    lines += ['前16分析只包含其中仍有正负候选的12张，不包含正确点在前16之外的3张；因此还须结合排名漏失图。',
        '评分器全池AUC约0.94，不能推出前2选择可靠；它只需压过大量明显错误点就能获得高AUC。三种子前16评分AUC约0.64—0.79，结论仍需独立检验。',
        '跨视图完全同坐标计数几乎恒定为1，不能当有效共识；区域支持虽能排除明显错误，在高分难候选中没有显示出可靠排序力。',
        '全部100张、S1：14个潜在纠错补充点全部来自局部视图；但51个干扰点中40个也来自局部视图。来源只能提示放大视图有用，不能据此判断新点值得替换；来源生成量不同，不能直接比较因果效果。',
        '低层对比度不是指令—控件语义，不建议直接按更鲜明/更近/更多视图去替换。','',
        '4. 实际修复与误改的区别（全部100，S1；仅8次修复/10次误改）',
        '修复与误改的候选都在高分前列：排名中位数分别1和1.5；分数、距离和跨视图支持分布明显重叠。',
        '复核相对MVP的分数优势中位数分别11.13和4.71；换序一致分别4/8与3/10。其他种子的修复也有较大分数优势，但仍重叠。',
        '因此“稳定且明显优于MVP”是可检验的线索，尚不是可靠置信度，也没有在这些数据上挑阈值或采用新门控。','',
        '5. 截图展示和数据问题',
        'repair_near：文字光标微小位置修正；repair_far：不同上下文的more-options控件；harm：lift箭头与相邻工具图标混淆。',
        'shortlist_miss与verification_miss分别展示筛选前丢失和复核选错。绿色GT和围绕GT的诊断裁剪仅用于事后解释，不是模型输入。',
        'label_conflict：指令要求选desktop.7z，源标注文字却是287，框在状态栏。保持原标签与评分，另行标为指令/标注歧义，不能把此例当纯模型语义错误。',
        '另有2条MVP最终文本不是规范坐标对，其中1条被补充方案修复；这部分收益包含输出解析/裁剪失败的恢复，不全是细粒度语义辨别。没有修改官方解析或基线。',
        '局部小框几何描述（仅诊断标签，不能作输入）：'+str(geometry_stats),'',
        '6. 如何挑：下一步可执行假设',
        '第一步保留现有分数作粗筛，但训练/评价指标应直接针对“MVP漏掉时能找回多少”，并衡量误改风险，而不只计算全池排序AUC。',
        '第二步在候选之间显式比较完整指令、当前窗口/浮层上下文、目标局部图像与MVP局部图像，用同屏相似控件和邻近图标作难负例。',
        '第三步分开建模纠错收益和误改风险，低证据时保留MVP。顺序一致性及相对分数优势可作输入线索，阈值必须在训练/开发流程确定后用新独立数据检验。',
        '候选多样性也应围绕不同控件/语义区域，而不是只要求坐标相距14像素；本轮没有验证新的区域划分或组合规则。',
        '这个方向与已做V20的辅助对应学习不同：这里建议直接监督相对MVP的+1/0/-1纠错目标，是否有效尚待实验。',
        '扩候选或提高门槛都不能自动带来净收益；本轮只定位问题、提出可检验假设，没有证明新选择策略更好。']
    (out/'analysis.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    registry=a.read(ROOT/'CURRENT_RESEARCH.json');registry.setdefault('descriptive_analyses',[])
    entry=dict(run_id='mvp_repair_v21_candidate_value',status='complete',parent='mvp_repair_v21',analysis_path=str(out/'analysis.txt'),mainline_unchanged=True,posthoc=True)
    registry['descriptive_analyses']=[e for e in registry['descriptive_analyses'] if e.get('run_id')!=entry['run_id']]+[entry]
    a.write(ROOT/'CURRENT_RESEARCH.json',registry)
    a.write(out/'analysis_completion.json',dict(complete=True,completed_at_utc=a.datetime.now(a.timezone.utc).isoformat(),
        screenshots=100,source_scripts={str(p):a.sha(p) for p in [Path(__file__),ROOT/'scripts/analyze_candidate_value_v21.py']},
        training=False,new_model_inference=False,posthoc_descriptive=True,primary_result_replay=True,all_candidate_labels_match_authenticated_cache=True,
        artifacts={str(p):a.sha(p) for p in out.rglob('*') if p.is_file() and p.name not in ['analysis_completion.json','visual_review.json']}))
    print('Readable analysis, geometry diagnostics, qualitative QA and Chinese figure complete.',flush=True)

if __name__=='__main__':main()
