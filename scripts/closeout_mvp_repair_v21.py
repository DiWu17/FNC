"""Replay frozen public decisions; descriptive coverage/selection decomposition."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import numpy as np
import run_mvp_repair_v21 as run
from clickagent.mvp_repair_v21 import make_pools,combine_orders,presentation_order,stable_key
from clickagent.public_evaluation import point_in_geometry


def plot_summary(out,results):
    """Presentation-only companion; primary sealed plot/results stay untouched."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,3,figsize=(14.2,4.4))
    arms=['original','mvp','native_verify','augmented_verify'];x=np.arange(4)
    for offset,cohort,color,label in [(-.18,'new_development','#2878B5','New dev (72)'),(.18,'old_val','#D65F32','Old dev (28)')]:
        y=[results['cohorts'][cohort]['methods'][a]['accuracy_percent'] for a in arms]
        bars=axes[0].bar(x+offset,y,.36,color=color,label=label)
        axes[0].bar_label(bars,fmt='%.1f',fontsize=7,padding=2)
    axes[0].set_xticks(x,['Original 3k','MVP','Native verify','+Our points'],rotation=15)
    axes[0].set_ylim(0,100);axes[0].set_ylabel('Accuracy (%)');axes[0].legend(fontsize=8)
    axes[0].set_title('Fixed policies; no seed selection',fontsize=10)
    for j,a in enumerate(['native_verify','augmented_verify']):
        m=results['cohorts']['new_development']['methods'][a]
        for offset,key,color,label in [(-.17,'repaired_seed_counts','#2878B5','MVP errors repaired'),(.17,'broken_seed_counts','#D65F32','MVP correct broken')]:
            bars=axes[1].bar(j+offset,float(np.mean(m[key])),.34,color=color,label=label if j==0 else None)
            axes[1].bar_label(bars,fmt='%.2f',fontsize=8,padding=2)
    axes[1].set_xticks([0,1],['Native verify','+Our points'])
    axes[1].set_ylabel('New dev case count (3-fit mean)');axes[1].legend(fontsize=8)
    axes[1].set_title('Repairs and damage relative to MVP',fontsize=10);axes[1].margins(y=.22)
    contrasts=results['cohorts']['new_development']['comparisons']
    for j,key in enumerate(['native_verification','our_candidate_value','hybrid_net_gain']):
        c=contrasts[key];lo,hi=c['ci_pp'];delta=c['delta_pp']
        axes[2].plot([lo,hi],[j,j],color='#2878B5',linewidth=2)
        axes[2].plot(delta,j,'o',color='#2878B5')
        axes[2].annotate(f'{delta:+.2f} [{lo:+.2f}, {hi:+.2f}]',(delta,j),xytext=(0,10),textcoords='offset points',ha='center',fontsize=7)
    axes[2].axvline(0,color='gray',linewidth=1)
    axes[2].set_yticks(range(3),['Native - MVP','+Our - Native','+Our - MVP'])
    axes[2].set_ylim(-.45,2.6);axes[2].invert_yaxis();axes[2].margins(x=.25)
    axes[2].set_xlabel('Accuracy difference (percentage points)')
    axes[2].set_title('New dev: adjusted 98.333% intervals',fontsize=10)
    fig.suptitle('V21 development pilot: matched verification budget; supplemental proposals cost extra',fontsize=11)
    fig.tight_layout();path=out/'closeout_summary.png';fig.savefig(path,dpi=170);plt.close(fig)
    return path


def main():
    out=run.OUT;run.verify();completion=run.read(out/'completion.json');assert completion['complete']
    for p,h in completion['artifact_sha256'].items():assert run.sha(p)==h
    plan=run.read(out/'coverage_decomposition_plan.json');assert plan['before_primary_label_analysis']
    cases=run.read(out/'data/public.json');labels=run.read(out/'data/labels.json');proposals=run.read(out/'data/proposals.json')
    results=run.read(out/'results.json');freeze=run.read(out/'predictions_frozen.json');protocol=run.sha(out/'protocol.json')
    for p,h in freeze['cases'].items():assert run.sha(p)==h
    def hit(point,c):return bool(point is not None and point_in_geometry([point[j]/c['image_size'][j] for j in [0,1]],labels[c['sample_id']]))
    artifacts={};decisions_replayed=0;per_case=[]
    for c in cases:
        assert run.sha(c['image'])==c['image_sha256']
        path=out/f'cases/{stable_key(c["sample_id"])}.json';r=run.read(path);artifacts[str(path)]=run.sha(path)
        assert r['protocol_sha256']==protocol and r['image_sha256']==c['image_sha256'] and not r['ground_truth_in_model_inputs']
        mp=run.read(r['mvp_path']);assert mp['protocol_sha256']==protocol and mp['image_sha256']==c['image_sha256']
        artifacts[r['mvp_path']]=run.sha(r['mvp_path']);p=proposals[c['sample_id']]
        rebuilt=[make_pools(mp['output'],p['points'],score,c['image_size']) for score in p['scores']]
        assert r['native_pool']==rebuilt[0]['native'] and r['augmented_pools']==[x['augmented'] for x in rebuilt]
        for pool,selected in [(r['native_pool'],r['native_result'])]+list(zip(r['augmented_pools'],r['augmented_results'])):
            if not pool:
                assert selected['point']==r['mvp_point'] and not selected['verification_paths'];continue
            observations=[]
            for reverse,vpath in zip([False,True],selected['verification_paths']):
                obs=run.read(vpath);assert obs['pool']==pool and obs['image_sha256']==c['image_sha256'] and obs['protocol_sha256']==protocol
                assert obs['order']==presentation_order(c['sample_id'],len(pool),reverse)
                assert not obs['ground_truth_in_model_inputs'] and obs['counter']['backbone_forward_calls']==1
                assert obs['image_grids']==[[1,32,32]]*(len(pool)+1)
                assert obs['counter']['processed_pixels']==(len(pool)+1)*512*512
                observations.append(obs);artifacts[vpath]=run.sha(vpath)
            replay=combine_orders(pool,pool[0],observations)
            for field in ['point','fallback_to_mvp','selected_candidate_index','order_disagrees','per_order_choice']:assert replay[field]==selected[field]
            np.testing.assert_allclose(replay['scores'],selected['scores'],rtol=0,atol=1e-12);decisions_replayed+=1
        mvp=hit(r['mvp_point'],c);native=any(hit(q,c) for q in r['native_pool']);full=any(hit(q,c) for q in p['points'])
        b=hit(r['native_result']['point'],c);stats=[]
        for s in range(3):
            pool=r['augmented_pools'][s];aug=any(hit(q,c) for q in pool);point=r['augmented_results'][s]['point'];selected=hit(point,c)
            extra={tuple(q) for q in pool}-{tuple(q) for q in r['native_pool']}
            new_point=point is not None and tuple(point) in extra
            assert not (mvp and native and not aug),'MVP anchor lost'
            stats.append(dict(mvp_wrong=not mvp,full_reachable_mvp_wrong=not mvp and full,
                native_miss_aug_gain=not native and aug,native_hit_aug_loss=native and not aug,
                full_reachable_but_aug_miss=not mvp and full and not aug,
                aug_reachable_but_not_selected=not mvp and aug and not selected,
                mvp_wrong_k0_or_k1=not mvp and len(pool)<=1,
                repaired_via_new_point=not mvp and selected and new_point,
                broken_via_new_point=mvp and not selected and new_point,
                candidate_value_repairs=not b and selected,candidate_value_harms=b and not selected,
                genuinely_new_candidate_count=len(extra)))
        per_case.append(dict(sample_id=c['sample_id'],cohort=c['cohort'],source=c['source'],mvp_hit=mvp,stats=stats))
    decomposition={}
    for cohort in ['new_development','old_val','all']:
        rows=[r for r in per_case if cohort=='all' or r['cohort']==cohort]
        fields=list(rows[0]['stats'][0]);seed_counts={k:[sum(row['stats'][s][k] for row in rows) for s in range(3)] for k in fields}
        decomposition[cohort]=dict(n=len(rows),seed_counts=seed_counts,three_fit_means={k:float(np.mean(v)) for k,v in seed_counts.items()})
        methods=results['cohorts'][cohort]['methods'];difference=np.mean(seed_counts['candidate_value_repairs'])-np.mean(seed_counts['candidate_value_harms'])
        assert abs(100*difference/len(rows)-(methods['augmented_verify']['accuracy_percent']-methods['native_verify']['accuracy_percent']))<1e-9
    summary=dict(complete=True,analyzed_at_utc=run.now(),descriptive_only=True,plan_sha256=run.sha(out/'coverage_decomposition_plan.json'),
        cohorts=decomposition,per_case=per_case,public_decisions_replayed=decisions_replayed,raw_artifact_sha256=artifacts,
        note='Coverage bucket counts are conditional on this fixed matched-k shortlist. Different ways of selecting more supplemental points were not tested. No independent result or new model selection.')
    run.write(out/'supplementary.json',summary)
    lines=['V21最终解读：MVP候选补充与视觉复核，100张开发截图。','',
        '四组均已完成，原3k三种子权重和MVP8B骨干冻结；不挑种子，不做三种子集成；两种显示顺序的分数按预案平均。',
        '新增开发72条、旧开发28条，均一截图一查询；这些是历史复用开发数据。',
        '两种复核逐样本候选数、MVP保留点、提示词、512概览/512局部图预算与两次显示顺序相同。',
        '补充方案另外需要原候选提议管线；复核预算匹配不等于完整管线算力匹配。','']
    for cohort in ['new_development','old_val']:
        r=results['cohorts'][cohort];lines.append(f'{cohort}, n={r["n"]}')
        for arm,m in r['methods'].items():lines.append(f'{arm}: {m["accuracy_percent"]:.2f}%, 种子命中{m["seed_hits"]}, 候选覆盖{m.get("pool_coverage_percent")}; 修正{m.get("repaired_seed_counts")},破坏{m.get("broken_seed_counts")}')
        for name,comp in r['comparisons'].items():lines.append(f'{name}: {comp["delta_pp"]:+.2f}pp; {comp["interval_percent"]:.3f}%区间{comp["ci_pp"]}')
        d=decomposition[cohort]['three_fit_means'];lines += ['分解（三固定拟合均值，例数）：'+str(d),'']
    advance=results['advance_to_new_independent_test'];gates=results['primary_positive_gates']
    if advance:
        interpretation='达到本轮开发门槛，支持继续做新独立验证；尚不能确认超越MVP，尤其仍需相同总计算预算的更强MVP对照。'
    elif gates['native_verification'] and not gates['our_candidate_value']:
        interpretation='开发收益主要支持额外视觉复核，本轮尚未建立我方补充候选的额外贡献。'
    else:
        interpretation='当前方案未达到登记的候选价值/净收益门槛；保留原主线。结合候选覆盖与复核选错分解确定下一轮问题。'
    nm=results['cohorts']['new_development']['methods'];nd=decomposition['new_development']['three_fit_means']
    lines += ['判断：'+interpretation,'',
        f'关键读数：新增开发集复核短名单覆盖从{nm["native_verify"]["pool_coverage_percent"]:.2f}%升至{nm["augmented_verify"]["pool_coverage_percent"]:.2f}%，准确率却从{nm["native_verify"]["accuracy_percent"]:.2f}%降至{nm["augmented_verify"]["accuracy_percent"]:.2f}%。',
        f'补充方案相对MVP平均修复{float(np.mean(nm["augmented_verify"]["repaired_seed_counts"])):.2f}例、误改{float(np.mean(nm["augmented_verify"]["broken_seed_counts"])):.2f}例；候选可用性增加尚未形成可靠净收益。',
        f'MVP答错{nd["mvp_wrong"]:.0f}例，其中原完整池可覆盖{nd["full_reachable_mvp_wrong"]:.0f}例，仍有{nd["full_reachable_but_aug_miss"]:.2f}例未进入补充短名单。这个损失发生在复核之前。',
        f'换序后的单次选择也有不稳定：100张中原生复核{results["diagnostics"]["native_order_disagreements"]}张前后不一致，补充复核三拟合分别{results["diagnostics"]["augmented_order_disagreements_per_seed"]}张；最终采用登记的双顺序分数平均。',
        '据此下一步应分别解决有价值候选进入短名单，以及是否可靠替换MVP保留点；本轮不能把失败唯一归因于模型参数量、数据量或语义能力。',
        '完整候选池有正确点，不保证当前只选两个补充点能把它带入复核；必须区分短名单漏选与复核误判。',
        'MVP重复预测导致k<=1、或MVP本身无效时，本轮严格匹配k的方案无法增加不同候选；不能据此否定更宽预算的候选补充。',
        '复核使用合法答案首token的约束分数，并非自由生成的推理过程；没有训练或置信阈值调参。',
        '额外点与顺序对照均由公开预测构造，标注只在全部预测冻结后用于评分。','',
        '计时面板：'+str(results['timing_summary']),
        '上述总时间为分别实测模块耗时相加的估算，不是联合端到端计时；模型加载排除，不能将缓存读取视为速度收益。',
        '新旧来源样本很少、开发集复用、同应用相关性及固定拟合限制结论；不宣称全榜SOTA或跨应用泛化。',
        '完整性：100张原图哈希校验；原3k300条命中记录重放一致；全部公共复核选择由原始分数重放一致。',
        '未读取原fresh280、未改原模型、未自动采用、未生成HTML。']
    (out/'interpretation.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    verification=dict(complete=True,verified_at_utc=run.now(),images_verified=len(cases),public_decisions_replayed=decisions_replayed,
        original_hit_replay=results['original_hit_replay'],matched_candidate_and_pixel_budgets=True,source_protocol_verified=True,
        raw_artifacts=len(artifacts),supplementary_sha256=run.sha(out/'supplementary.json'),mainline_unchanged=True)
    run.write(out/'verification.json',verification)
    decision=dict(recorded_at_utc=run.now(),run_id=out.name,status='development_pilot_complete_no_automatic_adoption',
        primary_positive_gates=gates,advance_to_new_independent_test=advance,interpretation=interpretation,
        mainline_run_id='quality_readout_v15_3k',analysis_path=str(out/'interpretation.txt'))
    dp=ROOT/'research/decisions/20261108_mvp_repair_v21_closeout.json';run.write(dp,decision)
    reg=run.read(ROOT/'CURRENT_RESEARCH.json');assert reg['active_experiment']['run_id']==out.name
    reg['active_experiment'].update(analysis_path=str(out/'interpretation.txt'),verification_path=str(out/'verification.json'),
        supplementary_path=str(out/'supplementary.json'),decision_path=str(dp),phase='inference_timing_replay_and_analysis_complete')
    plot=plot_summary(out,results)
    reg['active_experiment']['summary_plot_path']=str(plot)
    reg['next_research_objective']['status']='mvp_repair_v21_development_pilot_complete_MVP_target_pending'
    reg['next_research_objective']['latest_v21_findings']=dict(primary_positive_gates=gates,advance_to_new_independent_test=advance,
        new_development_accuracy={a:m['accuracy_percent'] for a,m in results['cohorts']['new_development']['methods'].items()},
        new_development_decomposition=decomposition['new_development']['three_fit_means'],interpretation=interpretation,
        analysis_path=str(out/'interpretation.txt'),mainline_unchanged=True,independent_confirmation=False)
    run.write(ROOT/'CURRENT_RESEARCH.json',reg)
    files=[out/p for p in ['completion.json','results.json','supplementary.json','verification.json','interpretation.txt','visual_qa_inputs.json']]+[dp,Path(__file__),plot]
    run.write(out/'closeout_completion.json',dict(complete=True,completed_at_utc=run.now(),artifact_sha256={str(p):run.sha(p) for p in files},
        mainline_unchanged=True,primary_exit_code=0))
    print('\n'.join(lines),flush=True)


if __name__=='__main__':main()
