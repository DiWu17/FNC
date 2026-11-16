"""Interpret the completed V29 run without rewriting sealed results."""
from pathlib import Path
import sys
import json
from datetime import datetime
from collections import defaultdict
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'scripts')]
import numpy as np
import run_coordinate_calibration_v29 as run
OUT = run.OUT


def main():
    run.verify();primary = run.read(OUT/'completion.json');extra = run.read(OUT/'anchored_comparator/completion.json')
    checked = 0
    for manifest in [primary, extra]:
        assert manifest['complete']
        for path, expected in manifest['artifact_sha256'].items():
            assert run.sha(path) == expected, path;checked += 1
    result = run.read(OUT/'results.json');anchor = run.read(OUT/'anchored_comparator/results.json')
    frozen = run.read(OUT/'frozen.json');cases = [c for c in run.read(OUT/'data/public.json') if c['cohort'] != 'train']
    public = {c['sample_id']: c for c in cases};labels = run.read(run.PRIOR/'data/labels.json')
    rows = run.read(OUT/'predictions.json')['rows']+run.read(OUT/'anchored_comparator/predictions.json')['rows']
    lookup = {(r['sample_id'],r['old_head'],r['arm']):r for r in rows};points = {};input_checks = 0
    frozen_time = datetime.fromisoformat(frozen['frozen_at_utc']).timestamp()
    all_cases = run.read(OUT/'data/public.json');train_ids = {c['sample_id'] for c in all_cases if c['cohort']=='train'}
    for c in all_cases:
        p = run.feature_path(c)
        with np.load(p,allow_pickle=False) as z:
            m = json.loads(str(z['metadata']));oldn = m['old_count']
            assert m['labels_used'] is False
            assert np.isfinite(z['x']).all() and np.isfinite(z['source_base']).all()
            assert not z['source_base'][:,oldn:].any(), c['sample_id']
            points[c['sample_id']] = z['points'].copy()
        if c['cohort']!='train': assert p.stat().st_mtime >= frozen_time
        input_checks += 1
    group_map = {c['sample_id']:(c['source'],str(c['source_group'])) for c in all_cases}
    fold_count = 0
    for audit in run.read(OUT/'training_fold_audit.json'):
        assert set(audit['training_ids'])|set(audit['validation_ids']) == train_ids
        for split in [audit]+audit['inner_splits']:
            assert not set(split['training_ids']) & set(split['validation_ids'])
            assert not {group_map[s] for s in split['training_ids']} & {group_map[s] for s in split['validation_ids']}
            fold_count += 1
    calibrated = {}
    for name in run.NAMES:
        calibrated[name] = {}
        for mode in ['fixed','temperature','source']:
            model = run.load_model(frozen['models'][name+'__'+mode]);calibrated[name][mode] = float(model['prior_coef'])
        gate = run.load_model(frozen['models'][name+'__gate'])
        assert set(gate['training_ids']) <= train_ids
        calibrated[name]['gate_discordant_queries'] = gate['discordant_queries']
    summary = {};bins = {};retained = 0;point_checks = 0
    for co in ['new_development','old_val']:
        cc = [c for c in cases if c['cohort']==co];summary[co] = {};bins[co] = {}
        for family in ['original3k','full']:
            summary[co][family] = {};bins[co][family] = {}
            for arm in ['source_calibrated','gate_equal','gate_conservative','anchored_equal','anchored_conservative']:
                decisions = [];bb = [defaultdict(float) for _ in range(4)]
                for s in [1,2,3]:
                    name = f'{family}_s{s}';counts = defaultdict(int)
                    for c in cc:
                        sid = c['sample_id'];out = lookup[sid,name,arm];free = lookup[sid,name,'native_free'];detail = out['detail'];pool = points[sid]
                        if out['point'] is not None: assert any(np.array_equal(out['point'],p) for p in pool)
                        point_checks += 1
                        if detail.get('reason')=='keep_exact_free':
                            assert out['point']==free['point'];retained += 1
                        if free['point'] is None:
                            counts['free_missing'] += 1;counts['fallback_hits'] += out['hit'];continue
                        counts['free_available'] += 1;counts['coordinate_changes'] += out['point']!=free['point']
                        counts['repairs'] += (not free['hit']) and out['hit'];counts['harms'] += free['hit'] and not out['hit']
                        prob = detail.get('probability');rival = detail.get('rival_index')
                        if prob is not None:
                            assert 0<=prob<=1 and rival is not None
                            other_hit = run.hit(pool[rival].tolist(),c,labels[sid]);b = bb[min(int(prob*4),3)]
                            b['decision_count'] += 1;b['probability_sum'] += prob
                            b['rival_better'] += bool(other_hit and not free['hit']);b['free_better'] += bool(free['hit'] and not other_hit)
                            if bool(other_hit)!=bool(free['hit']): b['conditional_brier_sum'] += (prob-float(other_hit))**2
                    decisions.append(dict(old_head=name,**dict(counts)))
                keys = ['free_available','free_missing','fallback_hits','coordinate_changes','repairs','harms']
                summary[co][family][arm] = dict(per_fit=decisions,mean_of_three={k:float(np.mean([d.get(k,0) for d in decisions])) for k in keys})
                bins[co][family][arm] = [dict(range=[i/4,(i+1)/4],**dict(b), conditional_rival_better_fraction=(b['rival_better']/(b['rival_better']+b['free_better']) if b['rival_better']+b['free_better'] else None)) for i,b in enumerate(bb)]
    run.write(OUT/'calibration_summary.json',dict(prior_coefficients=calibrated,retention=summary,
              caveat='New/oldcandidate calibration here concerns relative ranking. Gate probabilities are conditional on exactlyonecorrect and may be miscalibrated after maximizing over rivals. No error-rate guarantee.'))
    run.write(OUT/'gate_probability_audit.json',dict(posthoc_descriptive_only=True,
              observation_unit='Decisions across three fixed fits; repeatedRGBs are not independent tasks. Not used to retune thresholds.',bins=bins))
    lines = [
        '当前方法已记录到docs/CURRENT_METHOD.md；V29旧先验/新候选校准和原始局部坐标保留已实现、训练、评估及核验。',
        '',
        '工程上的两处修正：',
        '1. 旧分数先验不再必须以系数1压住新证据。temperature分支与固定系数对照共享同一66维特征和损失，仅学习alpha；source分支只用真实A坐标的旧分数作A内标准化，新坐标的未知旧分数设为缺失零，并通过明确的原生来源特征学习。中位数填充值不会进入该分支的推理分数或特征；为保持干预可比，训练硬负例抽样仍沿用旧排序，不等于训练各步骤都删除了旧排序。',
        '2. 自由局部有效时保留原始整数坐标，不吸附回旧候选网格、不平均坐标。一个19维输入的正则门控估计替换是否更好；仅严格超过预登记0.75时换点，0.5作为探索对照。自由点缺失时由来源校准排序回退。',
        '',
        '实验：固定V28的112训练、100复用开发截图与A32+至多3原生点；没有新调用8B。18个校准排序器、6个门控、60个外层分组/来源留出，以及外层训练内的嵌套3折都已完成；所有全训练模型冻结后才做V29开发推理。旧主干与六个旧头不改。',
        '旧评分头已经学习过原训练集，交叉拟合只隔离新模块，不能称整个系统从未见这些来源。新72/旧28都已多轮观察，不是独立测试。',
        '',
        '新开发72图：准确率%，旧3k／旧全量，三固定拟合均值（不集成）：',
    ]
    for arm in run.ARMS:
        lines.append(arm+': '+' / '.join(f'{result["new_development"]["summary"][family][arm]["percent"]:.2f}' for family in ['original3k','full']))
    lines += ['', '旧开发28图：']
    for arm in ['native_free','free_then_A','consensus','source_calibrated','gate_equal','gate_conservative','v28_geometry']:
        lines.append(arm+': '+' / '.join(f'{result["old_val"]["summary"][family][arm]["percent"]:.2f}' for family in ['original3k','full']))
    lines += [
        '',
        '校准的实际收益：',
        '学习出的temperature系数0.645—0.746，source系数0.666—0.767，均低于原固定1。相同特征下temperature比固定先验新72提高1.39/0.93个百分点；来源缺失处理再增加0.46/0.00。校正区间下界均未大于0，旧28也未显示一致增量。因此先验偏强是值得修正的实现问题，但本轮尚未证明它单独解释主要差距。',
        '新的source_calibrated为63.89/62.50%，仍低于原生自由点69.44%。它在有效自由点上每拟合改动47—56次，救回2—3例，却破坏6—8例；同一目标内部改点不会改变命中，不能把所有坐标变化都算成破坏。',
        '',
        '保留规则的实际收益：',
        '预登记保守门控在新72两系列均69.44%，保住自由点基线；0.5门控为63.89/63.43%，更容易误替换。0.75门控仍改动2—9个有效自由坐标；除3k_s2有1救回1破坏外，其余拟合均0救回0破坏。不能把同准确率说成逐点相同或已经学会修复错误。',
        '旧28保守门控57.14%，比原生自由53.57%的1/28增益全部来自自由点缺失时的校准回退；有效自由点上没有净修复。简单三视图共识为60.71%，MVP历史参考67.86%。新72自由点与历史MVP同69.44%仍不构成全域追平或同成本优势。',
        '14项主比较校正后，只有旧3k系列保守方案相对原A显示正下界（+14.35个百分点，约[0.46,29.17]）；相对直接保留自由点为0，相对共识/同观测几何分支的区间包含0。整体相对A的上涨不能归功于学习门控，也包含额外视觉观测。',
        '',
        '训练样本稀缺的单列对照：',
        '原top-rival门控每头仅12—14条条件有效训练查询。查看这一训练统计后、尚未查看V29新开发结果时，单独登记了原生自由点与全部候选的比较模型；训练有效查询70张、1357个比较对，但它们不是1357个独立样本。全部原主实验保持不变。',
        '该补充的0.75版本新72仅56.48/56.94%，0.5版本53.70/54.63%，表现明显更差；训练内分组/来源留出也不如保留自由点。这说明在现有相对几何表示下，扩大比较对不够，最大化候选后的置信度尤其不能直接当作替换正确率。这个模型保留作负结果，不设为默认。',
        '',
        '现在可以怎样使用：',
        '新候选保留准确原生坐标；未知旧分数保持缺失，由显式来源特征处理；有效自由局部点默认保留，来源校准用于缺失时的回退。保守门控作为已实现的研究比较层保留，不能把它写成已证实优于直接局部点的贡献。采用主线仍保持V15_3k。',
        '后续真正需要的新证据是：在自由点错误、其他候选正确时能够区分两者的局部内容/指令对应，以及候选选择后的置信度校准。仅减少旧先验或继续扩大同类几何比较对没有解决这一语义判别缺口。',
        '',
        '可复查入口：protocol.json、frozen.json、results.json、retention_diagnostics.json、training_cv_summary.json；二级比较见anchored_comparator/plan.json及results.json。另附gate_probability_audit.json，固定概率区间只作事后诊断，没有据此重新调阈值。',
    ]
    (OUT/'interpretation.txt').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    aa = ['A','v28_geometry','fixed_prior','source_calibrated','native_free','gate_equal','gate_conservative']
    ll = ['A','V28 geometry','Fixed prior','Calibrated','Free native','Gate >.5','Gate >.75']
    fig, axes = plt.subplots(1,2,figsize=(13,5),sharey=True)
    for ax, co in zip(axes,['new_development','old_val']):
        x=np.arange(len(aa))
        for j,family in enumerate(['original3k','full']):
            vals=[result[co]['summary'][family][a]['percent'] for a in aa]
            ax.bar(x+(j-.5)*.36,vals,.36,label=family)
        ax.axhline(69.4444444444 if co=='new_development' else 67.8571428571,color='black',ls='--',lw=1,label='Historical MVP (unequal cost)')
        ax.set_title('Reused new72' if co=='new_development' else 'Reused old28');ax.set_xticks(x,ll,rotation=27,ha='right');ax.set_ylim(0,100);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].set_ylabel('Accuracy (%)');axes[1].legend(fontsize=8)
    fig.suptitle('V29: fixed candidate pool / mean of 3 fits / extra observations vs A')
    fig.tight_layout();fig.savefig(OUT/'summary.png',dpi=170);plt.close(fig)
    run.write(OUT/'closeout_verification.json',dict(passed=True,at_utc=run.now(),immutable_artifacts_checked=checked,
              feature_cases_checked=input_checks,nested_splits_group_disjoint_checked=fold_count,decision_points_checked=point_checks,
              exact_retention_decisions_checked=retained,original_and_consensus_prediction_parity=1200,
              no_newpoint_imputed_prior_in_source_inference_features=True,development_features_after_model_freeze=True,mainline_unchanged=True))
    print('INTERPRETATION_AND_CLOSEOUT_VERIFIED',checked,'artifacts',flush=True)


if __name__=='__main__':main()
