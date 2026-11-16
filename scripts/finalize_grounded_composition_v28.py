"""Interpret all predeclared V28 results and verify the final delivered files."""
import os,sys,json,subprocess,shutil,time,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'scripts')]
import numpy as np
from run_visual_feedback_v23 import read,write,sha,bootstrap,hit
import run_grounded_composition_v28 as run
from clickagent.prompt_readout_v25 import coordinate_consensus
OUT=run.OUT

def main():
    spec=run.verify();result=read(OUT/'results.json');summary=read(OUT/'mechanisms/summary.json')
    assert read(OUT/'verified_repairs/frozen.json')['complete']
    with (OUT/'verified_evaluate.stdout.log').open('wb') as stdout,(OUT/'verified_evaluate.stderr.log').open('wb') as stderr:
        extra=subprocess.run([sys.executable,'-B','-X','utf8','-u',str(ROOT/'scripts/run_grounded_composition_v28_verified_repairs.py'),'evaluate'],cwd=ROOT,stdout=stdout,stderr=stderr)
    assert extra.returncode==0,'Verified repair evaluation failed; inspect its stderr'
    verified_repairs=read(OUT/'verified_repairs/results.json');verified_frozen=read(OUT/'verified_repairs/frozen.json')
    assert read(OUT/'component_controls/frozen.json')['complete']
    with (OUT/'components_evaluate.stdout.log').open('wb') as stdout,(OUT/'components_evaluate.stderr.log').open('wb') as stderr:
        component_exit=subprocess.run([sys.executable,'-B','-X','utf8','-u',str(ROOT/'scripts/run_grounded_composition_v28_components.py'),'evaluate'],cwd=ROOT,stdout=stdout,stderr=stderr)
    assert component_exit.returncode==0,'Component evaluation failed; inspect its stderr'
    components=read(OUT/'component_controls/results.json')
    large=ROOT/'runs/grounded_composition_v28_representation3k'
    while not (large/'completion.json').exists():
        if (large/'status.json').exists():assert read(large/'status.json')['phase']!='failed',read(large/'status.json')
        print('Waiting for cached2973 representation follow-up',flush=True);time.sleep(10)
    representation3k=read(large/'results.json')
    paired=ROOT/'runs/grounded_composition_v28_paired_queries'
    while not (paired/'completion.json').exists():
        if (paired/'status.json').exists():assert read(paired/'status.json')['phase']!='failed',read(paired/'status.json')
        time.sleep(10)
    paired_result=read(paired/'results.json')
    import probe_grounded_composition_v28_paired_queries as paired_probe
    fixed_baselines={}
    for mode in ['global_only','shared','collapsed']:
        rr=paired_probe.records(read(paired/'features.json'),mode);pp=[]
        for r in rr:
            scores=r['x'][:,384:480].mean(1)+r['x'][:,960:1056].mean(1)
            pp.append(dict(group=r['group'],hit=bool(r['y'][int(scores.argmax())])))
        fixed_baselines[mode]=paired_probe.accuracy(pp)
    write(paired/'fixed_baselines.json',dict(no_fitting=True,no_policy_selection=True,results=fixed_baselines))
    (paired/'readout_final.txt').write_text((paired/'readout.txt').read_text(encoding='utf-8')+'\n预登记的无训练等权头均值参照：\n'+json.dumps(fixed_baselines,ensure_ascii=False,indent=2)+'\n注意：自动挖掘的同模型名称不等价于人工确认的同可见文字；配对语义质量审计另见quality_review.json（如有）。\n',encoding='utf-8')
    aux=[paired/'fixed_baselines.json',paired/'readout_final.txt']
    if (paired/'quality_review.json').exists():aux.append(paired/'quality_review.json')
    write(paired/'final_completion.json',dict(complete=True,base_completion_sha256=sha(paired/'completion.json'),additional_artifact_sha256={str(p):sha(p) for p in aux}))
    backbone_files=read(run.V23/'protocol.json')['backbone_identity']
    for path,r in backbone_files.items():
        st=Path(path).stat();assert st.st_size==r['bytes'] and st.st_mtime_ns==r['mtime_ns'],path
    assert read(OUT/'execution.json')['complete'] and read(OUT/'mechanisms/completion.json')['complete'] and read(OUT/'training_transfer/completion.json')['complete']
    checks=subprocess.run([sys.executable,'-B','-X','utf8','-m','unittest','discover','-s','tests','-p','test_grounded_composition_v28.py','-v'],cwd=ROOT,capture_output=True,text=True,encoding='utf-8')
    write(OUT/'verification_tests.json',dict(exit_code=checks.returncode,output=checks.stdout+checks.stderr,current_source_sha256=run.identity()));assert checks.returncode==0
    changed=[];verified=0
    for p,h in read(OUT/'completion.json')['artifact_sha256'].items():
        actual=sha(p)
        if actual!=h:
            # The parent executor appends its finish exit-code AFTER the child
            # writes completion. This is a receipt update, never a result edit.
            assert Path(p).resolve()==(OUT/'execution.json').resolve(),p
            execution=read(p);assert execution['complete'] and all(r['exit_code']==0 for r in execution['phases'])
            changed.append(dict(path=p,before=h,after=actual,reason='Parent executor finalized phase exit receipts after child completion; scientific inputs/models/results unchanged.'))
        else:verified+=1
    for folder in ['mechanisms','training_transfer','verified_repairs','component_controls']:
        for p,h in read(OUT/folder/'completion.json')['artifact_sha256'].items():assert sha(p)==h,p
    for p,h in read(large/'completion.json')['artifact_sha256'].items():assert sha(p)==h,p
    for p,h in read(paired/'completion.json')['artifact_sha256'].items():assert sha(p)==h,p
    new=result['new_development'];old=result['old_val'];gates={}
    def supported(family,a,b):return new['contrasts'][family+'/'+a+'-'+b]['ci_pp'][0]>0
    for family in ['original3k','full']:
        gates[family]=dict(representation=supported(family,'repr_centered','repr_raw') and supported(family,'repr_centered','repr_uniform') and supported(family,'repr_centered','repr_shuffled'),
            identity=supported(family,'grounded','no_identity'),relation=supported(family,'grounded','no_relation') and supported(family,'grounded','reversed_relation'),
            targeted_supervision=supported(family,'targeted','grounded'),more_data=supported(family,'combined','grounded'),overall=supported(family,'combined','A'))
    # Descriptive observation audit: distinguish legitimate missing-target crops
    # from formatting failures. These annotations never influence observations.
    cases=read(OUT/'data/public.json');labels=read(OUT/'data/labels.json');quality=[]
    dev72=[c for c in cases if c['cohort']=='new_development']
    prediction_lookup={(r['sample_id'],r['old_head'],r['arm']):r for part in ['predictions_stage1.json','predictions_stage2.json'] for r in read(OUT/part)['rows']}
    group_sensitivity={}
    for family in ['original3k','full']:
        for a,b in spec['contrasts']:
            difference=np.asarray([[float(prediction_lookup[c['sample_id'],f'{family}_s{s}',a]['hit'])-float(prediction_lookup[c['sample_id'],f'{family}_s{s}',b]['hit']) for c in dev72] for s in [1,2,3]])
            group_sensitivity[family+'/'+a+'-'+b]=bootstrap(difference,dev72,.9975,group_field='source_group')
    write(OUT/'source_group_sensitivity.json',dict(descriptive_sensitivity=True,grouping_note='Original metadata groups vary by source: application/site/session/image; this is not uniformly app-held-out testing. Same20 contrasts/99.75%level asRGBprimary, no policy selection.',contrasts=group_sensitivity))
    for c in cases:
        metas={}
        for part in ['global','free','locked']:
            with np.load(run.dest(c,'raw/'+part),allow_pickle=False) as z:metas[part]=json.loads(str(z['metadata']))
        box=np.asarray(metas['global']['planned_local_transform']['box'])/np.tile(c['image_size'],2)
        visible=any(b[0]>=box[0] and b[1]>=box[1] and b[2]<=box[2] and b[3]<=box[3] for b in labels[c['sample_id']]['boxes'])
        overlap=any(min(b[2],box[2])>max(b[0],box[0]) and min(b[3],box[3])>max(b[1],box[1]) for b in labels[c['sample_id']]['boxes'])
        classes={}
        for part,m in metas.items():
            pfield=re.search(r'<p>(.*?)</p>',m['generated_text'],re.S)
            rawpoint=pfield.group(1).strip().upper() if pfield else ''
            if m['complete']:
                classes[part]='complete_point' if m['point'] is not None else ('complete_abstention' if rawpoint=='NONE' else 'complete_invalid_coordinate')
            else:classes[part]='bare_NONE' if m['generated_text'].strip().upper()=='NONE' else 'malformed'
        quality.append(dict(sample_id=c['sample_id'],cohort=c['cohort'],target_fully_in_local_crop=visible,target_overlaps_local_crop=overlap,output_classes=classes))
    qs={}
    for co in ['train','new_development','old_val']:
        rr=[r for r in quality if r['cohort']==co]
        qs[co]=dict(n=len(rr),target_fully_in_crop=sum(r['target_fully_in_local_crop'] for r in rr),target_completely_outside_crop=sum(not r['target_overlaps_local_crop'] for r in rr),outputs={part:{cls:sum(r['output_classes'][part]==cls for r in rr) for cls in ['complete_point','complete_abstention','complete_invalid_coordinate','bare_NONE','malformed']} for part in ['global','free','locked']})
    write(OUT/'observation_quality.json',dict(descriptive=True,annotation_used_only_for_audit=True,summary=qs,rows=quality))
    # Exact additive score attribution for reachable errors, never a new policy.
    attributions=[]
    for name in run.NAMES:
        row=read(OUT/'models_stage2/frozen.json')['models']['combined__'+name];model=run.load_npz(row['path'])[0]
        for c in [c for c in cases if c['cohort']!='train']:
            r=run.record(c,name,'combined');scores=run.infer(model,r)
            if not len(scores):continue
            chosen=int(scores.argmax());truth=np.asarray([hit(p.tolist(),c,labels[c['sample_id']]) for p in r['points']],bool)
            if truth[chosen] or not truth.any():continue
            pos=np.flatnonzero(truth);correct=int(pos[scores[pos].argmax()]);delta=(r['x'][chosen]-r['x'][correct])/model['std']*model['weight']
            parts=dict(old_prior=float(r['base'][chosen]-r['base'][correct]),visual=float(delta[:1152].sum()),native_geometry=float(delta[1152:1169].sum()),object_relation=float(delta[1169:].sum()))
            margin=float(scores[chosen]-scores[correct]);assert abs(sum(parts.values())-margin)<1e-7
            attributions.append(dict(sample_id=c['sample_id'],cohort=c['cohort'],old_head=name,chosen_point=r['points'][chosen].tolist(),diagnostic_GT_positive=r['points'][correct].tolist(),wrong_minus_positive_margin=margin,parts=parts,identical_visual_features=bool(np.max(np.abs(r['x'][chosen,:1152]-r['x'][correct,:1152]))<1e-7)))
    write(OUT/'score_attribution.json',dict(descriptive_frozen_model_only=True,GT_used_only_to_choose_diagnostic_positive=True,rows=attributions))
    # Necessary nonlearned references: extra observations alone vs learned use.
    reference_rows=[]
    for c in [c for c in cases if c['cohort']!='train']:
        g,gm=run.load(c,'grounded');consensus=coordinate_consensus([gm['native_points'][p] for p in ['global','free','locked']])
        for si,name in enumerate(run.NAMES):
            oldpoint=prediction_lookup[c['sample_id'],name,'A']['point'];s=g['scores'][si]
            priorpoint=g['points'][int(s.argmax())].tolist() if len(s) else None
            for arm,point in [('augmented_prior',priorpoint),('native_consensus_then_A',consensus if consensus is not None else oldpoint)]:
                reference_rows.append(dict(sample_id=c['sample_id'],old_head=name,arm=arm,point=point,hit=hit(point,c,labels[c['sample_id']])))
    ref_lookup={(r['sample_id'],r['old_head'],r['arm']):r for r in reference_rows};reference_results={}
    for co in ['new_development','old_val']:
        cc=[c for c in cases if c['cohort']==co];ss={};dd={}
        for family in ['original3k','full']:
            ss[family]={};full=np.asarray([[prediction_lookup[c['sample_id'],f'{family}_s{s}','combined']['hit'] for c in cc] for s in [1,2,3]],float)
            for arm in ['augmented_prior','native_consensus_then_A']:
                val=np.asarray([[ref_lookup[c['sample_id'],f'{family}_s{s}',arm]['hit'] for c in cc] for s in [1,2,3]],float)
                ss[family][arm]=dict(percent=float(val.mean()*100),seed_hits=val.sum(1).astype(int).tolist());dd[family+'/combined-'+arm]=bootstrap(full-val,cc,.9875 if co=='new_development' else .95)
        reference_results[co]=dict(summary=ss,contrasts=dd)
    write(OUT/'nonlearned_references.json',dict(secondary_only=True,selection_performed=False,description='Same observed pool. augmented_prior uses A logits/native median prior without learned residual; consensus uses priorV25 complete-link14px grouping of global/free/locked native coordinates, fallback A only if no native point. No new fitting or VLM calls.',rows=reference_rows,results=reference_results))
    lines=['V28：三个环节均已执行、分析和核验','',
        '范围：112个原训练集独立RGB（随机56+失败定向56），100个历史开发RGB（新72+旧28）。冻结Qwen3-VL-8B与全部旧评分器；新增模块每个拟合仅56/112例，不能称作新增模块3k/全量训练。','',
        '采样统计更正：matched_count实际为匹配预算数量，原报告误读为正确覆盖数。真实标签重算随机组覆盖47/56，失败组覆盖19/56；失败组漏目标37/56，其中空池19。实际采样是在关系指令、非空预算失败、空预算失败间交替，未按真实覆盖做分层。截图ID、实际训练标签、全部原有分支不变。原始登记/元数据及更正记录一起保留。','',
        '1. 表示：先去图像公共成分，再归一化和组合；保留96头、前4个token位置及全部token均值/最小值。固定A32候选，原始/去均值/均匀/打乱等容量对照均报告。',
        '2. 一致性与关系：新全图目标、原生坐标、具名参照对象框；同一候选驱动局部裁剪分别自由定位和附加全图目标名称提示定位。名称提示不能保证同名实例身份正确。关系作用于预测对象框边界，使用非线性方向/内部/邻近约束及两关系合取；另有去关系和反转关系对照。','3. 监督：同样模型、损失与正则，随机56与失败定向56等量对照，合并112，并以不采近邻硬负点作对照。保留三拟合/两旧评分器系列；没有按开发最高值挑种子。','',
        '新开发72图：准确率为三个固定拟合均值，单位%。']
    arms=['A','repr_raw','repr_centered','repr_uniform','repr_shuffled','no_identity','no_relation','grounded','reversed_relation','targeted','combined','combined_nohard']
    lines.append('方法 | 旧3k系列 | 旧全量系列')
    for arm in arms:lines.append(f'{arm} | {new["summary"]["original3k"][arm]["percent"]:.2f} | {new["summary"]["full"][arm]["percent"]:.2f}')
    lines+=['','同观测、不训练新增评分器的参照（单独描述性登记，不替换原主实验）：']
    for arm in ['augmented_prior','native_consensus_then_A']:
        lines.append(f'{arm}: 旧3k {reference_results["new_development"]["summary"]["original3k"][arm]["percent"]:.2f}%；旧全量 {reference_results["new_development"]["summary"]["full"][arm]["percent"]:.2f}%。')
    lines+=['','缓存表示扩量（2973训练查询，原先56例结果之后单独登记；开发仍复用同100图）：']
    for mode in ['raw','centered','shuffled']:
        lines.append(f'{mode}: 旧3k {representation3k["new_development"]["summary"]["original3k"][mode]["percent"]:.2f}%；旧全量 {representation3k["new_development"]["summary"]["full"][mode]["percent"]:.2f}%。')
    lines+=['','训练内同图/同模型目标描述配对诊断：',
        f'自动挖掘仅{paired_result["paired_rgb"]}对；给定两个原标注区域中心，按RGB留出，禁止把其准确率与MVP主榜相比。自动相同模型名称未逐例等价于相同可见标签，且可能含语义相同指令却标到不同控件的歧义。',
        '配对小样本结果仅用于诊断数据与表示缺口：'+json.dumps(paired_result['summaries'],ensure_ascii=False)]
    lines+=['','原生坐标对照（全图/局部自由/局部全图名称提示）：'+str(new['native']),'候选覆盖：'+str(new['coverage']),'', '合并112训练相对原A：']
    for family in ['original3k','full']:
        contrast=new['contrasts'][family+'/combined-A'];lines.append(f'{family}: {contrast["delta_pp"]:+.2f}个百分点，99.75% RGB分层bootstrap区间 {contrast["ci_pp"]}；旧28变化 {old["summary"][family]["combined"]["percent"]-old["summary"][family]["A"]["percent"]:+.2f}个百分点。')
    lines+=['','更正后单独登记的标签核验纠错监督（原有主实验全部保留）：']
    for arm in ['verified_repair','matched_random','all112_priority']:
        count=verified_frozen['models'][arm+'__original3k_s1']['new_training_rgb']
        lines.append(f'{arm} 训练{count} RGB: 旧3k {verified_repairs["new_development"]["summary"]["original3k"][arm]["percent"]:.2f}%；旧全量 {verified_repairs["new_development"]["summary"]["full"][arm]["percent"]:.2f}%。')
    lines+=['这些是单独登记的探索性二级对照，区间与全部种子见verified_repairs/readout.txt，没有按开发最高值替换方法。','', '登记对比的区间下界是否大于0：',json.dumps(gates,ensure_ascii=False,indent=2),'','实际训练监督覆盖：',json.dumps(summary['training'],ensure_ascii=False,indent=2),'','新72失败分解：',json.dumps(summary['decomposition'],ensure_ascii=False,indent=2),'','同截图历史MVP参考（预算不同，非独立确认）：']
    lines+=['新72评分组件对照（同一候选池，保留旧先验）：']
    for arm in ['geometry_random','visual_random','geometry_combined','visual_combined']:
        lines.append(f'{arm}: 旧3k {components["new_development"]["summary"]["original3k"][arm]["percent"]:.2f}%；旧全量 {components["new_development"]["summary"]["full"][arm]["percent"]:.2f}%。')
    for co in ['new_development','old_val']:
        comp=summary['cached_MVP_comparison'][co];lines.append(f'{co}: MVP {comp["MVP_percent"]:.2f}%；V28 combined 旧3k {comp["V28"]["original3k"]["combined"]["percent"]:.2f}%，旧全量 {comp["V28"]["full"]["combined"]["percent"]:.2f}%。')
    lines+=['','解释边界：',
        '去均值有效性需要相对原始/均匀/打乱均改善；仅旧开发改善或训练拟合改善不支持稳定泛化。',
        '身份锁定依赖全图先解析正确；不同视图标签不同也可能是同义词。显式参照框是模型预测，没有独立锚点标注，不能把关系失败全部归因于关系算子。',
        '反转关系分支也重新训练了权重，可能学会反向利用其特征；它不是完全移除参照信息的控制。去关系对照检验的是在相同原生坐标/视觉证据条件下的增量。',
        '本轮真实参照条件数量见results.json mechanisms；若很少，关系对照的结论受检验力限制。',
        '候选增补来自原生模型坐标；新增坐标使用旧A logit中位数作中性先验，不能称为旧评分器在新坐标上的原生预测。',
        '为让消融共用同一候选池，A32由六个旧评分头共同形成并保护各头第一名；3k/全量列表示最后评分头系列，不代表端到端系统仅使用该系列的训练预算。不能据此主张纯3k数据效率。',
        '112仍是小样本机制实验。近邻框外负点不保证都是同名控件；没有把自动挖掘任务包装成逐例人工验证的重复名称基准。',
        '原A与MVP成本不同：V28另外增加全图1次、局部2次模型调用。不能将缓存运行时间称为部署加速。',
        '历史开发集已多轮观察；区间是当前拟合在复用RGB上的不确定性，不是独立泛化或种子总体置信区间。来源及原始source_group留出仅隔离新增模块，旧评分器已见原训练数据。source_group粒度依来源而异，部分仅为会话或截图，不能统一称为跨应用测试。附source_group聚类bootstrap敏感性，不按较有利的区间挑结论。','',
        '执行成本：'+str(result['cost']),'拒绝/格式与裁剪可见性审计：'+json.dumps(qs,ensure_ascii=False),'主线保持 V15_3k。本轮没有自动更换主线，也没有独立测试宣称超过MVP。']
    (OUT/'conclusion.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    # Static scientific figure, with the same predeclared mean-of-three values.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(15,5),sharey=True)
    panels=[('Representation',['A','repr_raw','repr_centered','repr_shuffled'],['A','Raw','Centered','Shuffled']),('Identity / relations',['A','no_identity','no_relation','grounded'],['A','Free view','No relation','Grounded']),('Supervision',['grounded','targeted','combined','combined_nohard'],['Random56','Targeted56','Combined112','No hard neg.'])]
    for ax,(title,aa,ll) in zip(axes,panels):
        x=np.arange(len(aa))
        for j,family in enumerate(['original3k','full']):ax.bar(x+(j-.5)*.36,[new['summary'][family][a]['percent'] for a in aa],.36,label=family)
        ax.set_xticks(x,ll,rotation=28,ha='right');ax.set_title(title);ax.set_ylim(0,100);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Accuracy (%)');axes[-1].legend();fig.suptitle('V28 reused development: 72 RGBs, mean of 3 fixed fits; unequal VLM cost vs A')
    fig.tight_layout();fig.savefig(OUT/'summary.png',dpi=170);plt.close(fig)
    write(OUT/'final_verification.json',dict(passed=True,unit_tests_exit_code=0,scientific_artifacts_verified=verified,backbone_identity_files_verified=len(backbone_files),expected_parent_receipt_updates=changed,baseline_prediction_parity=summary['original_prediction_parity'],positive_gates=gates,mainline_unchanged=True))
    reg=read(ROOT/'CURRENT_RESEARCH.json');reg['active_experiment'].update(status='complete',analysis_path=str(OUT/'conclusion.txt'),static_figure_path=str(OUT/'summary.png'),final_verification_path=str(OUT/'final_verification.json'),training_transfer_path=str(OUT/'training_transfer/readout.txt'),mechanisms_path=str(OUT/'mechanisms/readout.txt'),positive_gates=gates)
    reg['active_experiment'].update(verified_repair_results_path=str(OUT/'verified_repairs/readout.txt'),component_results_path=str(OUT/'component_controls/readout.txt'),representation_scaling_path=str(large/'readout.txt'))
    reg['active_experiment']['paired_query_probe_path']=str(paired/'readout_final.txt')
    write(ROOT/'CURRENT_RESEARCH.json',reg)
    archive=OUT/'source_archive_postanalysis';archive.mkdir(parents=True,exist_ok=True)
    for script in ['analyze_grounded_composition_v28_mechanisms.py','analyze_grounded_composition_v28_transfer.py','finalize_grounded_composition_v28.py','execute_grounded_composition_v28.py','execute_grounded_composition_v28_followups.py','run_grounded_composition_v28_verified_repairs.py','run_grounded_composition_v28_components.py','run_grounded_composition_v28_representation3k.py','probe_grounded_composition_v28_paired_queries.py','resume_grounded_composition_v28_reporting.py']:
        shutil.copyfile(ROOT/'scripts'/script,archive/script)
    paths=[p for p in OUT.rglob('*') if p.is_file() and p.suffix in ['.json','.npz','.txt','.png','.py'] and 'offload' not in p.parts and p.name!='delivery_manifest.json']
    write(OUT/'delivery_manifest.json',dict(complete=True,mainline_unchanged=True,artifact_sha256={str(p):sha(p) for p in paths}))
    print('\n'.join(lines),flush=True)

if __name__=='__main__':main()
