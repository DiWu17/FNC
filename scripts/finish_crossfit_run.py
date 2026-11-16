"""Finish registered secondary analyses and documentation after main audit."""
from __future__ import annotations
import json
from pathlib import Path
import sys
import time

from crossfit_protocol import OUT, read
from clickagent.action_benchmark import write_json
from run_crossfit_pipeline import run


def append_once(path, content):
    marker = '<!-- CROSSFIT_CANDIDATES_V12 -->'
    text = path.read_text(encoding='utf-8-sig')
    block = marker + '\n' + content.strip() + '\n'
    if marker in text:
        assert block in text, 'Preserve differing previous V12 documentation: ' + str(path)
    else:
        path.write_text(text.rstrip() + '\n\n' + block, encoding='utf-8')


def docs():
    data = read(OUT / 'evaluation/test/summary.json')
    shift = read(OUT / 'evaluation/training_pool_shift.json')
    pair = next(p for p in data['paired_comparisons'] if p['method']=='oof_s1__K16' and p['reference']=='in_sample_s1__K16')
    methods = data['methods']; lines = []
    for suffix in ('s1', 's2'):
        for arm in ('in_sample', 'oof'):
            m = methods[arm + '_' + suffix + '__K16']
            lines.append(f'| {arm}_{suffix} | {m["hits"]}/192 | {100*m["accuracy"]:.2f}% |')
    text = '\n'.join(lines)
    content = f'''## V12：交叉拟合训练候选的对照

保持 V11 内部完整评分器的架构、损失和优化参数；五个从相同随机初态训练的空间提议器按整图分折，为每题分别生成 OOF 和 in-sample 训练候选。六个提议器均不继承旧监督权重、旧教师点或学习过的 head prior。873 条训练指令、568 张截图，原172验证用于选模，历史及近重复排除后的新192题用于主测试。

两组在同一全训练部署提议器 D 的 K16 候选池上评测，oracle 为 {data['pools']['16']['oracle']}/192。

| 固定评分器 | 新192命中 | 命中率 |
|---|---:|---:|
{text}

预定主比较 OOF S1 − in-sample S1：{pair['wins']} 赢、{pair['losses']} 输，差值 {pair['delta']*100:+.2f} 个百分点，截图簇配对95%区间 [{pair['ci95'][0]*100:.2f}, {pair['ci95'][1]*100:.2f}]。S2是稳健性检查，不按测试选择种子。同池 GUI K16 命中 {methods['gui_actor__K16']['hits']}/192，Qwen 原生坐标命中 {methods['qwen_native_completed_lax']['hits']}/192。

训练池 K16 覆盖：in-sample {shift['arms']['in_sample']['total']['oracle16']}/873，OOF {shift['arms']['oof']['total']['oracle16']}/873。候选覆盖与评分器误排在报告中分开统计。命中任一有效目标区域即可，不要求等于标注中心；空池失败，全部192题进入分母。

该对照检验交叉拟合协议整体效果；循环配对还改变了被排除的整折，不是仅干预单图是否参与训练的纯因果实验。原172验证重复用于上游与评分器选模，独立结论以冻结后的新192为准。旧V11-192仅作旧V8池上的迁移诊断；本轮新D和新测试集的绝对分数不与旧轮次直接比高低。

见 [完整报告](../runs/crossfit_candidates_v12/evaluation/report.html)、[主测试](../runs/crossfit_candidates_v12/evaluation/test/report.html)、[预定协议](../runs/crossfit_candidates_v12/protocol.json)、[最终审计](../runs/crossfit_candidates_v12/final_audit.json)。本轮为离线实验，未自动替换原方法的运行入口，未评估真实点击执行或负指令权限误点率。
'''
    append_once(Path('docs/CURRENT_METHOD.md'), content)
    append_once(Path('README.md'), '''## V12 交叉拟合候选对照

已完成保持评分器结构与损失不变的 in-sample / OOF 训练候选对照、两个指定种子及新192题独立测试。见 [完整报告](runs/crossfit_candidates_v12/evaluation/report.html) 和 [方法记录](docs/CURRENT_METHOD.md)。主测试、候选覆盖、误排与旧池迁移诊断分别报告；原方法运行入口未自动切换。
''')


def main():
    while not (OUT / 'final_audit.json').exists():
        time.sleep(5)
    assert read(OUT / 'final_audit.json')['complete']
    if (OUT / 'finish_complete.json').exists():
        assert read(OUT / 'finish_complete.json')['complete']; return
    run('migration', ['scripts/run_crossfit_scorer.py', '--stage', 'migration'])
    dest = OUT / 'evaluation/migration'
    run('migration_evaluate', ['scripts/evaluate_crossfit_candidates.py',
        '--cases', str(dest/'cases.jsonl'), '--predictions', str(dest/'predictions.jsonl'),
        '--pools', str(dest/'pools.jsonl'), '--out', str(dest)])
    run('report', ['scripts/build_crossfit_report.py'])
    docs()
    write_json(OUT / 'finish_complete.json', dict(complete=True, created_at=time.time(),
        report=str(OUT/'evaluation/report.html'), main_audit_complete=True,
        migration_secondary_complete=True, documentation_complete=True, manual_visual_qa_pending=True))
    print('REGISTERED ANALYSES AND DOCUMENTATION COMPLETE', flush=True)


if __name__ == '__main__':
    main()
