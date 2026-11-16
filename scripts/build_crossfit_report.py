"""Assemble audited V12 results, with training-pool and migration diagnostics."""
from html import escape
import json
from pathlib import Path

from crossfit_protocol import OUT, read
from clickagent.action_benchmark import sha256


def pct(value):
    return '—' if value is None else f'{value*100:.1f}%'


def main():
    final = read(OUT / 'final_audit.json'); assert final['complete']
    test = read(OUT / 'evaluation/test/summary.json')
    assert sha256(OUT / 'evaluation/test/summary.json') == final['final']['summary_sha256']
    recovery = final.get('serialization_audit_adapter')
    if recovery:
        assert sha256(Path('scripts/audit_crossfit_serialization.py')) == recovery['adapter_sha256']
        assert sha256(OUT / 'checks/final_audit_serialization_diagnostic.json') == recovery['diagnostic_sha256']
        assert sha256(OUT / 'checks/audit_thread_replay_diagnostic.json') == recovery['thread_replay_diagnostic_sha256']
    val = read(OUT / 'evaluation/val/summary.json')
    shift = read(OUT / 'evaluation/training_pool_shift.json')
    migration = read(OUT / 'evaluation/migration/summary.json')
    interpretation_path = OUT / 'evaluation/interpretation.json'
    interpretation = read(interpretation_path) if interpretation_path.exists() else {}
    for path, expected in interpretation.get('input_sha256', {}).items():
        assert sha256(Path(path)) == expected, 'Stale report interpretation: ' + path
    interpretation_html = ''.join('<p>' + escape(p) + '</p>' for p in interpretation.get('paragraphs', []))
    audit_note = ('<p>审计说明：原审计将内存整数计数键与JSON字符串键作严格类型比较。'
        '本轮保留原封存源码，通过显式适配仅规范化8处候选数量分布的键；'
        '命中、坐标、得分、区间及其他检查保持原有严格比较。'
        '<a href="../../../scripts/audit_crossfit_serialization.py">适配源码</a> · '
        '<a href="../checks/final_audit_serialization_diagnostic.json">逐字段诊断</a> · '
        '<a href="../checks/gui_parallel_execution.json">GUI并行执行记录</a></p>'
        if final.get('serialization_audit_adapter') else '')
    method_rows = []
    names = [a + '_' + s + '__K16' for s in ('s1','s2') for a in ('in_sample','oof')]
    names += ['gui_actor__K16', 'gui_actor__K4', 'qwen_native_completed_lax']
    for name in names:
        m = test['methods'][name]; old = migration['methods'].get(name)
        v = val['methods'].get(name)
        method_rows.append(f'<tr><td>{escape(name)}</td><td>{m["hits"]}/{m["n"]} ({pct(m["accuracy"])})</td>'
            f'<td>{pct(m.get("conditional_selection_accuracy"))}</td><td>{v["hits"] if v else "—"}</td>'
            f'<td>{old["hits"] if old else "—"}</td></tr>')
    pairs = [p for p in test['paired_comparisons'] if p['method'] in ('oof_s1__K16','oof_s2__K16')
             and p['reference'] == p['method'].replace('oof','in_sample')]
    paired_rows = ''.join(f'<tr><td>{escape(p["method"])}</td><td>{p["wins"]}/{p["losses"]}</td>'
        f'<td>{p["delta"]*100:+.2f}</td><td>[{p["ci95"][0]*100:.2f}, {p["ci95"][1]*100:.2f}]</td></tr>' for p in pairs)
    pool_rows = []
    for arm in ('in_sample','oof'):
        m = shift['arms'][arm]['total']
        pool_rows.append(f'<tr><td>{arm}</td><td>{m["oracle16"]}/{m["n"]}</td>'
            f'<td>{pct(m["conditional_random_top1"])}</td><td>{pct(m["quota4_retention"])}</td>'
            f'<td>{m["local_named"]}</td></tr>')
    conditional_rows = []
    for arm in ('in_sample','oof'):
        for seed in ('s1','s2'):
            name = arm + '_' + seed
            meta = shift['models'][name]; tr = meta['selected_training_evaluation']
            v = val['methods'][name + '__K16']; te = test['methods'][name + '__K16']
            conditional_rows.append(f'<tr><td>{name}</td><td>{meta["selected"]["epoch"]}</td>'
                f'<td>{tr["hits"]}/{tr["oracle"]} ({pct(tr["hits"]/tr["oracle"] if tr["oracle"] else None)})</td>'
                f'<td>{pct(v["conditional_selection_accuracy"])}</td><td>{pct(te["conditional_selection_accuracy"])}</td></tr>')
    pool = test['pools']['16']
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>V12 交叉拟合候选实验</title>
<style>body{{font:16px system-ui,sans-serif;max-width:1100px;margin:36px auto;padding:0 24px;color:#223247;line-height:1.65}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{border:1px solid #d8e0e9;padding:9px;text-align:left}}th{{background:#edf3f8}}img{{max-width:100%}}code{{background:#eef2f7;padding:2px 5px}}a{{color:#096da5}}</style>
<h1>V12：训练候选是否需要严格交叉拟合</h1>
<p>冻结 Qwen3-VL-8B，保持 V11 内部评分器架构与损失。五个从相同随机初态训练的空间提议器，按整张截图分折：每题由未见其截图的模型生成 OOF 候选，由见过其截图的循环配对模型生成 in-sample 候选。两组评分器在同一全训练部署提议器 D 的候选上评测。</p>
<p>训练 873 条指令 / 568 张截图，验证 172 条，主测试为历史和近重复排除后的新 192 题，三个来源各 64 题。六个提议器均无旧监督 warm start、旧教师点或学习过的 head prior。共用部署 D 的 K16 oracle：{pool['oracle']}/192；空池 {pool['empty']}。评分器更新次数、初始状态和逐题预算表配对锁定。</p>
<h2>主结果</h2><table><tr><th>方法</th><th>新192命中</th><th>池可达时选对率</th><th>原172验证命中（用于选模）</th><th>旧V11-192迁移诊断</th></tr>{''.join(method_rows)}</table>
<p>任意点落入任一有效目标区域即命中，含边界；不要求等于标注中心。全体样本进入分母。旧192使用旧V8候选池，且已被分析，仅作迁移诊断；它不与新192构成同样本对照。</p>
<h2>配对差值：OOF − in-sample</h2><table><tr><th>指定种子</th><th>赢 / 输</th><th>命中率差 pp</th><th>配对95%区间 pp</th></tr>{paired_rows}</table>
<p>S1 为预定主比较，S2 为稳健性；不能事后挑高分种子。区间按来源分层、截图簇配对 bootstrap 2000 次，仅反映冻结模型的样本不确定性。循环配对还改变了被排除的整折，检验的是交叉拟合协议整体效应，不能解释成只干预一张图的纯因果效应。</p>
<h2>训练候选分布</h2><table><tr><th>组别</th><th>K16覆盖</th><th>可达题随机Top1期望</th><th>quota K4保住K16目标</th><th>有局部名称状态</th></tr>{''.join(pool_rows)}</table>
<h2>拟合与泛化</h2><table><tr><th>评分器</th><th>选定epoch</th><th>训练可达题选对</th><th>验证可达题选对</th><th>新测试可达题选对</th></tr>{''.join(conditional_rows)}</table>
<p>训练列使用各自训练池；验证与新测试两臂各自使用完全相同的 D 池。候选覆盖、局部名称可用性及正例比例的变化是协议干预的组成，报告它们用于解释，不按测试结果补点或配平。</p>
<h2>完整证据</h2><p><a href="test/report.html">独立测试全部预算、来源与失败分解</a> · <a href="val/report.html">共用D验证</a> · <a href="migration/report.html">旧池迁移诊断</a> · <a href="training_pool_shift.json">训练候选分布</a> · <a href="../pretest_audit.json">测试前审计</a> · <a href="../final_audit.json">最终审计</a> · <a href="../protocol.json">预定协议</a></p>
{audit_note}
<h2>结果解读</h2>{interpretation_html}
<img src="test/accuracy.png" alt="独立测试方法命中率与区间">
<p>Top4后GUI属于预定次要分析，候选验证数量是计数代理；复用缓存得分不能作为端到端实测提速。此轮没有执行真实桌面点击，也不评估禁止点击等负指令的权限误点率。</p></html>'''
    (OUT / 'evaluation/report.html').write_text(page, encoding='utf-8')
    print(json.dumps(dict(complete=True, report=str(OUT/'evaluation/report.html'), pairs=pairs)), flush=True)


if __name__ == '__main__':
    main()
