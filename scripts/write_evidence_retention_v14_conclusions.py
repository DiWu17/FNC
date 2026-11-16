"""Attach a readable, explicitly scoped interpretation to completed results."""
from __future__ import annotations

import hashlib
import html
import json
from pathlib import Path

ROOT = next(p for p in Path(__file__).resolve().parents if (p / 'src/clickagent').is_dir())
RUN = ROOT / 'runs/evidence_retention_v14'


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def esc(value):
    return html.escape(str(value))


def main():
    assert read(RUN / 'work_state.json')['complete']
    assert read(RUN / 'checks/final_output_audit.json')['complete']
    note = read(RUN / 'conclusions.json')
    direct = read(RUN / 'analysis/direct_quality_readout/results.json')
    top4 = read(RUN / 'analysis/quality_top4_gui/results.json')
    assert direct['posthoc'] and top4['posthoc'] and top4['complete']
    parts = ['<section id="completed-conclusions"><h2>本轮结论与研究记录</h2>',
             '<p><strong>' + esc(note['research_status']) + '</strong></p>',
             '<table><tr><th>按顺序检验的问题</th><th>本轮结论</th></tr>']
    for answer in note['three_questions']:
        parts.append('<tr><td>' + esc(answer['question']) + '</td><td>' + esc(answer['answer']) + '</td></tr>')
    parts.append('</table><p>' + esc(note['contribution_decision']) + '</p>')
    parts.append('<h3>事后诊断：冻结新质量分数直接读出</h3><p>以下改变了读出方式。它们提供下一步假设，不替代三项预定实验。各分支均展示两个固定种子，没有挑最佳种子。</p>')
    parts.append('<table><tr><th>质量分数</th><th>旧 K16 直接 Top1</th><th>旧 K16 质量 Top4→GUI</th><th>全部已解码点直接 Top1</th><th>全部已解码点质量 Top4→GUI</th></tr>')
    for family in ('rank', 'aligned', 'permuted'):
        values = []
        for pool in ('old_pool', 'decoded_union'):
            values.append(' / '.join(str(direct['hits'][pool][f'{family}_s{s}']) for s in (1, 2)))
            keys = [pool + '__' + f'{family}_s{s}' for s in (1, 2)]
            values.append(' / '.join(str(top4['methods'][key]['hits']) for key in keys) if all(key in top4['methods'] for key in keys) else '未评估')
        parts.append('<tr><td>' + esc(family) + '</td>' + ''.join('<td>' + esc(v) + '</td>' for v in values) + '</tr>')
    parts.append('</table><p>分母均为172。旧 K16 保持原候选池；全部已解码点分支读取更多候选的质量分数，二者分别报告。两种 Top4 分支都仅从共同已缓存 GUI 分数读取4个候选，不增加 GUI 调用；没有给未覆盖坐标补分。</p>')
    for field, title in (('posthoc_interpretation', '这些诊断说明什么'), ('next_step', '下一步收敛方向')):
        parts.append('<h3>' + title + '</h3><p>' + esc(note[field]) + '</p>')
    parts.append('<h3>可复核材料</h3><p>')
    links = [('conclusions.json', '研究决策原始记录'),
             ('checks/final_output_audit.json', '最终输出核对'),
             ('analysis/report.html', '候选得失与空间分布诊断'),
             ('analysis/direct_quality_readout/results.json', '直接读出诊断'),
             ('analysis/quality_top4_gui/plan.json', 'Top4 诊断计划与时点'),
             ('analysis/quality_top4_gui/results.json', 'Top4 诊断结果'),
             ('experiment3/execution_io_compat.json', '执行期间 I/O 修复记录'),
             ('checks/gui_restart_consistency.json', '重启前后数值校准')]
    parts.append(' · '.join('<a href="' + path + '">' + label + '</a>' for path, label in links))
    parts.append('</p><p>V13 封存材料保持只读。训练873、验证172，截图 RGB 组无交叉；验证集有历史复用。结果是静态点命中，没有新的独立测试或真实点击结果。实验期间的文件写入故障及恢复已保留；不声称这次共享缓存执行测得各方法的端到端速度。</p></section>')
    path = RUN / 'report.html'
    content = path.read_text(encoding='utf-8')
    assert 'id="completed-conclusions"' not in content, 'Interpretation was already attached'
    position = content.index('<h2>问题一：')
    path.write_text(content[:position] + ''.join(parts) + content[position:], encoding='utf-8')
    (RUN / 'analysis/conclusions_report_source.py').write_bytes(Path(__file__).read_bytes())
    files = [path, RUN / 'conclusions.json', RUN / 'analysis/conclusions_report_source.py',
             RUN / 'checks/final_output_audit.json', RUN / 'analysis/quality_top4_gui/plan.json',
             RUN / 'analysis/quality_top4_gui/results.json', RUN / 'analysis/direct_quality_readout/results.json']
    manifest = dict(complete=True, scope='Reporting only; original experiment results and protocols unchanged.',
                    files={str(p.relative_to(RUN)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files})
    (RUN / 'report_completion.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == '__main__':
    main()
