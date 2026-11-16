"""Add a readable interpretation without altering any sealed experiment result."""
import json
import hashlib
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'runs/grounded_composition_v28'


def read(p):
    return json.loads(Path(p).read_text(encoding='utf-8'))


def write(p, value):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    assert read(ROOT / 'research/executions/v28_followups.json')['complete']
    verification = read(OUT / 'final_verification.json')
    assert verification['passed'] and read(OUT / 'execution.json')['complete']
    delivery = read(OUT / 'delivery_manifest.json')
    assert delivery['complete']
    print('VERIFY_FINAL_DELIVERY', len(delivery['artifact_sha256']), flush=True)
    for i, (p, expected) in enumerate(delivery['artifact_sha256'].items(), 1):
        assert sha(p) == expected, p
        if i % 400 == 0:
            print('VERIFIED', i, flush=True)
    paired = ROOT / 'runs/grounded_composition_v28_paired_queries'
    extra = read(paired / 'final_completion.json')
    assert sha(paired / 'completion.json') == extra['base_completion_sha256']
    for p, expected in extra['additional_artifact_sha256'].items():
        assert sha(p) == expected, p
    result = read(OUT / 'results.json')
    components = read(OUT / 'component_controls/results.json')
    repaired = read(OUT / 'verified_repairs/results.json')
    mechanisms = read(OUT / 'mechanisms/summary.json')
    references = read(OUT / 'nonlearned_references.json')['results']
    big = read(ROOT / 'runs/grounded_composition_v28_representation3k/results.json')
    cases = read(OUT / 'data/public.json')
    attribution = read(OUT / 'score_attribution.json')['rows']
    quality = read(OUT / 'observation_quality.json')['summary']
    new, old = result['new_development'], result['old_val']
    component_diagnostics = {}
    for family in ['original3k', 'full']:
        rr = [r for r in attribution if r['cohort'] == 'new_development' and r['old_head'].startswith(family)]
        component_diagnostics[family] = dict(
            reachable_error_decisions_across_three_fits=len(rr),
            identical_visual_features=sum(r['identical_visual_features'] for r in rr),
            largest_positive_wrong_margin_term=dict(Counter(max(r['parts'], key=r['parts'].get) for r in rr)),
            positive_wrong_margin_counts={k: sum(r['parts'][k] > 1e-8 for r in rr) for k in ['old_prior', 'visual', 'native_geometry', 'object_relation']})
    write(OUT / 'interpretation_attribution_summary.json', dict(
        descriptive_only=True, unit='Wrong-versus-best-annotated-positive decisions across all three fixed fits; repeated RGBs are not independent samples.',
        caveat='Additive score terms describe the frozen fitted model; they do not identify a causal effect of removing/refitting a component.',
        new72=component_diagnostics))

    def acc(cohort, family, arm):
        return result[cohort]['summary'][family][arm]['percent']

    lines = [
        '三个环节已全部执行并核验。当前最明确的实现瓶颈是：新增视觉表示与候选选择尚未形成可靠、可迁移的指令—区域判别能力。局部高分辨率观测有用，但学到的评分器没有充分利用它。三项主贡献检验均未通过；保持原研究主线，不能宣称超过MVP。',
        '',
        '本轮新增模块训练112张原训练截图（随机56、失败定向56），开发100张历史已观察截图（主分析新72、补充旧28）。冻结视觉语言模型及六个旧评分器，完成636次新调用、全部对照训练和审计。主分析准确率为三个固定拟合均值，不是集成，也没有挑最优种子。旧3k/全量列仅指评分头；共用A32由六个头形成，不能解释为纯3k训练预算。',
        '',
        '新开发72张，准确率%，两列分别旧3k／旧全量：',
    ]
    for label, arm in [('原A评分器', 'A'), ('随机56+全部新特征', 'grounded'), ('失败定向56', 'targeted'), ('合并112', 'combined')]:
        lines.append(f'{label}: {acc("new_development", "original3k", arm):.2f} / {acc("new_development", "full", arm):.2f}')
    for label, arm in [('仅几何随机56', 'geometry_random'), ('仅视觉合并112', 'visual_combined')]:
        lines.append(f'{label}: {components["new_development"]["summary"]["original3k"][arm]["percent"]:.2f} / {components["new_development"]["summary"]["full"][arm]["percent"]:.2f}')
    for label, arm in [('标签核验补充正例32（二级探索）', 'verified_repair'), ('来源/数量匹配随机32（二级探索）', 'matched_random')]:
        lines.append(f'{label}: {repaired["new_development"]["summary"]["original3k"][arm]["percent"]:.2f} / {repaired["new_development"]["summary"]["full"][arm]["percent"]:.2f}')
    lines += ['', '同次观测的原生输出与简单规则：', '全图 / 局部自由 / 局部名称提示: ' + ' / '.join(f'{new["native"][p]:.2f}' for p in ['global', 'free', 'locked'])]
    for arm in ['augmented_prior', 'native_consensus_then_A']:
        lines.append(arm + ': ' + ' / '.join(f'{references["new_development"]["summary"][f][arm]["percent"]:.2f}' for f in ['original3k', 'full']))
    lines += [
        f'同72张历史MVP参考 {mechanisms["cached_MVP_comparison"]["new_development"]["MVP_percent"]:.2f}%。局部自由输出在这72张上数值相同，旧28为{old["native"]["free"]:.2f}%、MVP {mechanisms["cached_MVP_comparison"]["old_val"]["MVP_percent"]:.2f}%。同分不代表逐例相同、同成本或独立复现优越性。',
        '',
        '第一环节：改善表示。保留实际token、96个头与全图/局部通道，比较原始余弦、先去公共成分、均匀注意力、打乱注意力。新72上去公共成分49.54/51.85%，原A55.09/56.48%；未建立增益。固定等权的近邻错误点AUC：去公共成分0.542，打乱0.570（54张可比较图），说明余弦数值更有差异并不自动等于目标语义更准确。',
        '为区分小样本拟合不稳，又用2973条缓存训练查询（2156张独立RGB）拟合相同表示：去公共成分' + ' / '.join(f'{big["new_development"]["summary"][f]["centered"]["percent"]:.2f}' for f in ['original3k', 'full']) + '%，恢复了56例训练的回退，但未可靠超过原A和打乱对照。它是小样本结果之后登记的探索性扩量，不是独立确认。当前不能把先去公共成分/多token视觉组合写成已证实的核心贡献。',
        '',
        '第二环节：跨视图一致性和对象关系。主模型相对不附加全图目标名称的分支，两系列均+1.85个百分点，校正区间包含0。原生局部输出反而由自由69.44%降为名称提示63.89%：5例自由正确/提示错误，1例相反。名称提示可能稳定错误全图解释，且不能区分真正同名实例；旧28的变化方向也不同，不能宣称锁定普遍有害。',
        '显式参照关系相对去关系仅+0.46/-0.46个百分点。54/72例含可解析预测参照，但固定关键词规则仅匹配5条明确关系指令，该规则不是完整关系标注。具名框准确性也没有独立人工标注，反转关系还可重新学习权重。因此当前只能确认这套实现没有显示可靠增量，不能据此否定所有关系组合方法。',
        '',
        '第三环节：纠错监督与硬负例。失败定向56比随机56高3.24/2.31个百分点，合并112比随机56高1.85/1.85；预设校正区间均包含0。只有采到错误并不保证有可学的纠正目标：失败组原候选漏37/56，原生补点救回10例，补后仍漏27例；随机组原漏9例、救回6例。训练112补后覆盖79例，其中真正有正负候选可做排序拟合76例。mechanisms中augmented_rankable字段实为补后覆盖数，不能将79误说成79例有效正负对。',
        '另用训练标签核验“旧3k多数头选错且新原生点确实命中”的39例，按来源上限取得32例，与同数量/来源随机32例相比+2.78/+2.31个百分点，98.75%区间仍包含0，旧28也不一致。优先保留补充正例是合理的训练设计线索，目前还不是确定贡献。',
        '',
        '为什么仍然选不好：',
        f'完整原候选覆盖56/72（{new["coverage"]["A_complete"]:.2f}%），共用A32覆盖54/72（{new["coverage"]["A32"]:.2f}%），补充原生点后61/72（{new["coverage"]["augmented"]:.2f}%）。候选池改善是真的，但11张仍完全没有正确点；合并模型在有正确候选时仍分别错17/16张（旧3k/全量的三拟合均值），可选却选错比完全漏候选更突出。',
        '几何单独分支已达到62.50/64.35%，高于相应全部特征分支；仅视觉分支合并训练仅50.00/51.85%。分支差异区间仍宽，但现有结果不支持把总体上涨归因于新增视觉语义组合。原生局部自由点比学习后的选择更好，直接说明当前残差评分/先验校准仍会丢掉已有正确答案。',
        '逐项错误分数归因见 interpretation_attribution_summary.json；旧先验、新视觉、原生几何、对象关系的精确加和见score_attribution.json。该归因只是已冻结模型的分数分解，不等于逐项删除重训后的因果效果。',
        '六个固定拟合合计99次可达却选错的决定中，推动错误相对正确点分数最高的项分别为旧先验45次、原生几何34次、视觉20次；这是重复截图上的决定计数，不能当作99张独立样本。只有1次错误点/正确点的视觉特征完全相同，说明本轮大部分排序失败不能仅用候选落入同一视觉网格解释。固定旧先验权重和新增点中位数先验值得单独校准检验。',
        '局部裁剪也仍有上游损失：新72中10张目标与裁剪完全不相交，只有51张至少一个完整标注框被裁剪包含。即便后续局部判断再准确，错误的观察位置也会限制可用信息；需要保留全图补救或重新裁剪的机制。',
        '来源差异明显：局部自由重看在AMEX/ShowUI/GroundCUA分别83.33/79.17/45.83%，全量系列合并模型为87.50/65.28/34.72%。当前跨应用桌面场景仍弱。',
        '',
        '数据审计也发现具体缺口：自动挖掘6对同模型名称样本，经逐例看图，没有确认到干净的同可见名称消歧对；包含命名漂移、产品标题/颜色控件粒度混淆、Click/Tap等价指令却标不同加号，以及一例目标疑似滚出截图。后一例原始ShowUI元数据已含同样框，发布截图校验值一致，未发现这一例导入坐标转换错误。保留所有样本和原结果，没有事后删样涨分；这6对不是噪声率的随机估计。',
        '',
        '下一步收敛建议：保留局部自由重看作为强基线，优先解决原生局部点与A候选之间的选择/可信度校准；训练样本必须确有可用正确候选和难负例，并补充经过检查的同图不同指令、真实同名实例样本。新增视觉表示必须超过相同几何和打乱对照，整体必须超过不训练的简单规则，再冻结方案做新的独立留出集及同成本MVP比较。当前不应继续堆复杂组合并直接写成有效贡献。',
        '',
        '统计与执行边界：',
    ]
    for family in ['original3k', 'full']:
        q = new['contrasts'][family + '/combined-A']
        lines.append(f'{family}合并112相对原A {q["delta_pp"]:+.2f}个百分点，99.75%按来源/RGB分层bootstrap区间{q["ci_pp"]}。')
    lines += [
        '所有主贡献门槛：' + json.dumps(verification['positive_gates'], ensure_ascii=False),
        '新旧开发集已多轮观察，所有区间是当前固定拟合在复用图像上的诊断，不是独立泛化确认；另附source_group聚类敏感性，原组粒度随来源为应用/网站/会话/截图，不能统称跨应用测试。',
        '训练采样matched_count语义误读、XML空白解析兼容，以及最终重复sample_id汇总错误均有保留原件的更正记录；后者仅机械修复输出序列化，不改任何候选、权重、得分或生成，不触发新调用。',
        '当前输出格式为Python、JSON、纯文本和静态研究图，没有新HTML看板。新调用总成本约127.70分钟模型计时，不能当部署延迟或加速结论。所有六个旧评分器与主干身份核验一致；基线600个坐标完全复放一致。原V15_3k主线保持。',
        '原cost.parse_failures=69的语义是不完整XML：其中62次为裸NONE、7次为其他格式异常；另有2次XML完整但坐标非法。裸NONE作为弃权保留，不能把69全说成模型格式错误。',
        '',
        '观测有效性与裁剪可见性（仅事后标注审计，不参与生成/打分）：' + json.dumps(quality, ensure_ascii=False),
    ]
    (OUT / 'interpretation.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    now = datetime.now(timezone.utc).isoformat()
    decision_path = ROOT / 'research/decisions/20261110_grounded_composition_v28_closeout.json'
    findings = dict(complete=True, scope='Reused-development mechanism experiment; no independent MVP-superiority claim.',
                    primary_positive_gates=verification['positive_gates'],
                    new72={f: dict(A=new['summary'][f]['A']['percent'], combined=new['summary'][f]['combined']['percent'],
                                   geometry_random=components['new_development']['summary'][f]['geometry_random']['percent']) for f in ['original3k', 'full']},
                    native=new['native'], coverage=new['coverage'], MVP_historical_new72_percent=69.44444444444444,
                    conclusion='Selection/calibration and transferable visual discrimination remain limiting; local observation has signal, added visual/relation features lack demonstrated incremental benefit.',
                    mainline_unchanged=True, interpretation_path=str(OUT / 'interpretation.txt'), decision_path=str(decision_path))
    write(decision_path, dict(at_utc=now, run_id=OUT.name, decision='Complete all3 stages; preserve V15_3k adopted mainline; do not promote a new policy on reused-development scores.',
                              findings=findings, delivery_manifest_sha256=sha(OUT / 'delivery_manifest.json'),
                              reporting_recovery_path=str(OUT / 'amendments/03_reporting_duplicate_sample_id/record.json')))
    reg = read(ROOT / 'CURRENT_RESEARCH.json')
    reg['active_experiment'].update(status='complete', phase='three_stages_analyzed_verified_and_interpreted',
                                    interpretation_path=str(OUT / 'interpretation.txt'), closeout_decision_path=str(decision_path),
                                    reporting_recovery_path=str(OUT / 'amendments/03_reporting_duplicate_sample_id/record.json'))
    entry = dict(reg['active_experiment'], completed_at_utc=now, delivery_manifest_path=str(OUT / 'delivery_manifest.json'))
    reg['completed_experiments'] = [e for e in reg['completed_experiments'] if e['run_id'] != OUT.name] + [entry]
    reg['next_research_objective'].update(status='v28_complete_transferable_selection_bottleneck_MVP_not_surpassed', latest_v28_findings=findings)
    write(ROOT / 'CURRENT_RESEARCH.json', reg)
    write(OUT / 'registry_at_closeout.json', reg)
    write(OUT / 'closeout_verification.json', dict(passed=True, at_utc=now, all_three_stages_complete=True,
                                                 delivery_artifact_hashes_verified=len(delivery['artifact_sha256']),
                                                 paired_additional_hashes_verified=len(extra['additional_artifact_sha256']),
                                                 delivery_manifest_sha256=sha(OUT / 'delivery_manifest.json'), mainline_unchanged=True))
    artifacts = [OUT / 'interpretation.txt', OUT / 'interpretation_attribution_summary.json', OUT / 'registry_at_closeout.json',
                 OUT / 'closeout_verification.json', decision_path, Path(__file__)]
    write(OUT / 'interpretation_manifest.json', dict(complete=True, at_utc=now,
                                                   preserved_delivery_manifest_sha256=sha(OUT / 'delivery_manifest.json'),
                                                   artifact_sha256={str(p): sha(p) for p in artifacts}))
    continuation = ROOT / 'research/executions/v28_continuation.json'
    checkpoint = read(continuation)
    checkpoint.update(complete=True, current_phase='All stages, followups, human interpretation and full delivery hash verification complete.',
                      interpretation_path=str(OUT / 'interpretation.txt'), manifest_path=str(OUT / 'interpretation_manifest.json'))
    write(continuation, checkpoint)
    print('CLOSEOUT_COMPLETE', now, flush=True)


if __name__ == '__main__':
    main()
