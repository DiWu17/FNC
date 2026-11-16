"""Replay V22 saved scores and write bounded mechanism interpretation."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'scripts'),str(ROOT/'src')]
import numpy as np
import run_structure_readout_v22 as r

def main():
    out=r.OUT;spec=r.verify();completion=r.read(out/'completion.json');assert completion['complete']
    for p,h in completion['artifact_sha256'].items():assert r.sha(p)==h,p
    indices=r.read(out/'data/indices.json');public=r.read(out/'data/public.json');frozen=r.read(out/'models/frozen.json');assert frozen['complete']
    text=r.read(out/'text/index.json');spatial=r.read(out/'spatial/index.json');assert text['complete'] and spatial['complete']
    fresh=r.read(out/'corrected_evaluation.execution.json');assert fresh['fresh_import'] and fresh['prior_parent_stopped_before_any_predictions']
    assert fresh['scientific_source_sha256']==r.sha(ROOT/'scripts/run_structure_readout_v22.py') and fresh['model_freeze_sha256']==r.sha(out/'models/frozen.json')
    text_coverage=0
    for entries in indices.values():
        for e in entries:
            if e['n']:
                import hashlib
                h=hashlib.sha256(public[e['sample_id']]['instruction'].encode()).hexdigest();assert h in text['records'];assert not text['records'][h]['truncated'];text_coverage+=1
    models=frozen['models']
    for s in [1,2,3]:
        for width in [32,64]:
            a,b=[models[f'{co}{width}_s{s}'] for co in ['pooled','ordered']]
            assert a['parameter_count']==b['parameter_count'] and a['initial_state_sha256']==b['initial_state_sha256']
        for arm in r.ARMS:
            m=models[f'{arm}_s{s}'];assert r.sha(m['checkpoint_path'])==m['checkpoint_sha256'] and m['steps']==2976 and m['fitting_queries']==2878
    results=r.read(out/'results.json');decomp={};replay=0
    for cohort,entries in indices.items():
        ev=r.read(out/f'evaluation/{cohort}.json');assert r.sha(ev['scores_path'])==ev['scores_sha256'];assert ev['sample_ids']==[e['sample_id'] for e in entries]
        label=[]
        for e in entries:
            assert r.sha(e['labels_path'])==e['labels_sha256'];label.append(np.load(e['labels_path'],allow_pickle=False)>.5)
        assert sum(y.any() for y in label)==ev['oracle_hits']
        with np.load(ev['scores_path'],allow_pickle=False) as z:
            offsets=z['offsets'];assert np.array_equal(offsets,np.r_[0,np.cumsum([e['n'] for e in entries])])
            for name in models:
                flat=z[name];assert len(flat)==offsets[-1] and np.isfinite(flat).all()
                for i,y in enumerate(label):
                    score=flat[offsets[i]:offsets[i+1]];hit=bool(len(score) and y[int(score.argmax())]);assert hit==ev['hit_vectors'][name][i];replay+=1
        prior=r.read(r.prior.OUT/f'evaluation/{cohort}.json');assert prior['sample_ids']==ev['sample_ids']
        for arm in ['original','capacity']:
            for s in [1,2,3]:assert prior['hit_vectors'][f'{arm}_s{s}']==ev['hit_vectors'][f'{arm}_s{s}']
        rows={};original=np.asarray([ev['hit_vectors'][f'original_s{s}'] for s in [1,2,3]],bool)
        oracle=np.asarray([y.any() for y in label],bool)
        for arm in r.ARMS:
            v=np.asarray([ev['hit_vectors'][f'{arm}_s{s}'] for s in [1,2,3]],bool)
            rows[arm]=dict(repaired_original_seed_counts=(v&~original).sum(1).astype(int).tolist(),broken_original_seed_counts=(~v&original).sum(1).astype(int).tolist(),
                wrong_despite_candidate_seed_counts=(~v&oracle[None]).sum(1).astype(int).tolist(),no_correct_candidate=int((~oracle).sum()))
        decomp[cohort]=rows
    r.write(out/'error_decomposition.json',dict(complete=True,cohorts=decomp,predictions_replayed=replay))
    new=results['cohorts']['new_development'];old=results['cohorts']['old_val'];train=results['cohorts']['train'];gates=results['primary_positive_gates']
    if gates['structure_small'] or gates['structure_large']:
        conclusion='在本次固定表示与读出中，保留结构至少在一个登记容量下达到正向门槛；仍需结合净收益、另一容量及未来独立数据判断。'
    else:conclusion='本次完整指令文本序列与3x3空间序列未建立相对同架构池化对照的登记净收益；不能据此认定所有信息压缩都无害或方法已达理论上限。'
    lines=['V22 自有方法对照：最终解释','',
        '所有12个新拟合、训练/开发评估、原始分数重放均已完成。原3k三个权重不变。',
        '固定2973条训练清单、2878非空拟合、2976步、批8、三种子；候选坐标与旧172/新300开发查询完全相同。',
        '两个新表示都保留原32base+294state，并读取同一套完整指令文本编码和局部空间内容。pooled将词/细格内容平均后重复；ordered保留顺序和格位置。每个容量下参数、初始化、输入长度/掩码与训练批次一致。',
        '这隔离的是新增读出中的结构保留；新读出对原版同时改变内容、结构、容量和提取成本，不能只归因于结构。','']
    for cohort,data in [('训练',train),('旧开发',old),('新开发',new)]:
        lines.append(f'{cohort},n={data["n"]},固定候选可达{data["oracle_hits"]}')
        for name,m in data['methods'].items():lines.append(f'{name}: {m["accuracy_percent"]:.2f}%，三种子命中{m["seed_hits"]}，可达条件{m["conditional_accuracy_percent"]:.2f}%')
        for name,c in data['comparisons'].items():lines.append(f'{name}: {c["delta_pp"]:+.2f}pp；{c["interval_percent"]:.2f}%区间{c["ci_pp"]}')
        lines.append('')
    lines+=['参数量（同容量两种表示完全一致）：'+str(spec['parameter_counts']),
        '判断：'+conclusion,'主门槛：'+str(gates),'达到后续独立验证候选门槛：'+str(results['advance_to_independent_test']),
        '新增读出相对原版的修复/误改，三种子分别统计：'+str(decomp['new_development']),
        '冻结模型输入干预：'+str(new['input_interventions']),
        '单独池化文本/空间的输入干预是分布变化诊断；不能替代分别训练的文本×空间因子试验。',
        '如果训练改善而开发无益，提示该读出利用新增特征仍缺乏泛化；若两者均无益，还存在表示、投影与优化不足等解释。',
        '完整指令来自冻结8B的独立纯文本因果编码；并非图像条件下全部指令状态。每token固定64维投影；局部视觉使用原缓存分辨率，未新增高分辨率图像。',
        '因此阴性结果不否定更高维、联合图文或其他读出；阳性结果也不等同于独立超越MVP。',
        '全部开发集历史复用，所有三种子均报告，不挑终点或最优容量、不自动采用、未访问原fresh280，无HTML。']
    (out/'interpretation.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    verification=dict(complete=True,verified_at_utc=r.now(),new_model_score_decisions_replayed=replay,all_12_models_frozen_before_evaluation=True,
        identical_parameter_and_initialization_pairs_verified=True,full_instruction_feature_coverage=text_coverage,archived_reference_hit_vectors_exact=True,
        source_and_checkpoint_identity_verified=True,fresh_corrected_evaluation_verified=True,training_checkpoints_unchanged_after_evaluation_restart=True,mainline_unchanged=True)
    r.write(out/'verification.json',verification)
    dp=ROOT/'research/decisions/20261109_structure_readout_v22_closeout.json';r.write(dp,dict(run_id=out.name,completed_at_utc=r.now(),primary_positive_gates=gates,advance_to_independent_test=results['advance_to_independent_test'],interpretation=conclusion,analysis_path=str(out/'interpretation.txt'),mainline_unchanged=True))
    reg=r.read(ROOT/'CURRENT_RESEARCH.json');assert reg['run_id']=='quality_readout_v15_3k' and reg['active_experiment']['run_id']==out.name
    reg['active_experiment'].update(phase='training_evaluation_replay_and_analysis_complete',analysis_path=str(out/'interpretation.txt'),verification_path=str(out/'verification.json'),decision_path=str(dp))
    reg['next_research_objective']['status']='v22_structure_control_complete_MVP_target_pending'
    reg['next_research_objective']['latest_v22_findings']=dict(primary_positive_gates=gates,advance_to_independent_test=results['advance_to_independent_test'],interpretation=conclusion,new_development_accuracy={n:v['accuracy_percent'] for n,v in new['methods'].items()},analysis_path=str(out/'interpretation.txt'),mainline_unchanged=True)
    r.write(ROOT/'CURRENT_RESEARCH.json',reg)
    files=[out/p for p in ['completion.json','results.json','interpretation.txt','verification.json','error_decomposition.json']]+[dp,Path(__file__)]
    r.write(out/'closeout_completion.json',dict(complete=True,completed_at_utc=r.now(),artifact_sha256={str(p):r.sha(p) for p in files},mainline_unchanged=True))
    print('\n'.join(lines),flush=True)
if __name__=='__main__':main()
