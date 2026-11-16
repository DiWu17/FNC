"""Export a self-contained HTML research report and scientific PNG figures."""
from collections import Counter
import html
import json
from pathlib import Path
import textwrap

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
from PIL import Image
from shapely.affinity import scale
from shapely.geometry import Point

from clickagent.action_benchmark import write_json
from clickagent.spatial_geometry import shape_from_geometry
from run_target_token_experiment import key, read_json, read_jsonl


LABELS = {'first': 'First-token map', 'uniform': 'Uniform phrase fusion', 'global': 'Global phrase selector',
          'local': 'Position-specific phrase selector', 'first_replica': 'First-token map (seed 2)', 'local_replica': 'Local selector (seed 2)',
          'previous_full': 'Previous full readout', 'previous_attention_only': 'Previous attention-only',
          'previous_static_96': 'Previous static 96-head', 'qwen_native_strict': 'Qwen native (strict)',
          'qwen_native_completed_lax': 'Qwen native (complete lax)', 'ui_tars': 'UI-TARS-1.5-7B',
          'local_coarse_centers': 'Local: coarse-cell centers', 'local_fine_centers': 'Local: fine-cell centers',
          'local_generated_action_policy': 'Local: obey generated CLICK/WAIT'}
CHINESE = {'first': '首token空间图', 'uniform': '逐词等权融合', 'global': '全局词选择', 'local': '逐位置词选择（预先固定主方案）',
           'first_replica': '首token空间图，第二随机种子', 'local_replica': '逐位置词选择，第二随机种子',
           'previous_full': '上一轮完整读出', 'previous_attention_only': '上一轮仅注意力', 'previous_static_96': '上一轮静态96头',
           'qwen_native_strict': 'Qwen直接坐标，严格格式', 'qwen_native_completed_lax': 'Qwen直接坐标，完整宽松格式',
           'ui_tars': 'UI-TARS-1.5-7B', 'local_coarse_centers': '主方案：粗格中心', 'local_fine_centers': '主方案：细格中心',
           'local_generated_action_policy': '主方案：服从生成的CLICK/WAIT'}


def table(headers, rows):
    return '<table><thead><tr>'+''.join('<th>'+html.escape(str(x))+'</th>' for x in headers)+'</tr></thead><tbody>'+''.join(
        '<tr>'+''.join('<td>'+html.escape(str(x))+'</td>' for x in row)+'</tr>' for row in rows)+'</tbody></table>'


def examples(dest, rows, cases, group_name, condition):
    by = {(r['method'], r['sample_id']): r for r in rows}
    selected = sorted([c for c in cases if condition(by['local', c['sample_id']], by['first', c['sample_id']])], key=lambda c: key(c['sample_id']))[:4]
    if not selected:
        return None
    fig, axes = plt.subplots(len(selected), 1, figsize=(13, 4.1*len(selected)), squeeze=False)
    for ax, c in zip(axes[:, 0], selected):
        with Image.open(c['image']) as im:
            ax.imshow(im.convert('RGB'))
        w, h = c['image_size']; target = shape_from_geometry(c['target_geometry'])
        parts = target.geoms if target.geom_type == 'MultiPolygon' else [target]
        for polygon in parts:
            xy = np.asarray(polygon.exterior.coords); ax.plot(xy[:, 0]*w, xy[:, 1]*h, c='#00d975', lw=2)
        for method, color, marker in [('local', '#ef4444', 'x'), ('first', '#f59e0b', '+'), ('qwen_native_completed_lax', '#38bdf8', 'o')]:
            r = by[method, c['sample_id']]
            if r['point'] is not None:
                style = dict(facecolors='none', edgecolors=color) if marker == 'o' else dict(c=color)
                ax.scatter(*r['point'], marker=marker, s=80, linewidths=2, label=LABELS[method]+(' hit' if r['hit'] else ' miss'), **style)
        caption = c['instruction']+'\nName: '+by['local', c['sample_id']].get('target_name', 'unavailable')
        ax.set_title('\n'.join(textwrap.wrap(caption, 110)), fontsize=10)
        ax.legend(loc='upper center', bbox_to_anchor=(.5, -.015), ncol=3, fontsize=8, frameon=False); ax.axis('off')
    fig.suptitle('Green: valid target. Cases: first four by hashed sample ID within the declared outcome group.', fontsize=10)
    fig.tight_layout(h_pad=3, rect=(0, 0, 1, .975)); filename=group_name+'.png'; fig.savefig(dest/filename, dpi=140); plt.close(fig)
    return filename


def main():
    font = Path('C:/Windows/Fonts/msyh.ttc')
    if font.exists():
        font_manager.fontManager.addfont(str(font))
        plt.rcParams['font.family'] = font_manager.FontProperties(fname=str(font)).get_name()
    out = Path('runs/phrase_v4'); dest = out/'evaluation'
    summary = read_json(dest/'summary.json'); protocol = read_json(out/'protocol.json'); audit = read_json(out/'final_audit.json')
    assert audit['complete']
    rows = read_jsonl(dest/'predictions.jsonl'); cases = [c for c in read_jsonl(out/'manifest.jsonl') if c['experiment_split'] == 'test']
    by_case = {c['sample_id']: c for c in cases}; methods = summary['methods']
    order = list(LABELS)
    selected = [n for n in order if n not in ('local_coarse_centers', 'local_fine_centers', 'local_generated_action_policy')]
    fig, ax = plt.subplots(figsize=(11, 7)); values = [methods[n]['accuracy']*100 for n in selected]
    colors = ['#2563eb' if n == 'local' else '#0891b2' if n.startswith('qwen_native') else '#94a3b8' for n in selected]
    ax.barh([LABELS[n] for n in selected], values, color=colors); ax.invert_yaxis(); ax.set_xlim(0, max(values)+10)
    for i, (n, value) in enumerate(zip(selected, values)):
        ax.text(value+.5, i, f'{methods[n]["hits"]}/192 ({value:.1f}%)', va='center', fontsize=9)
    ax.set_xlabel('Point inside target region (%)'); ax.set_title('Fresh 192-image matched test, 262,144-pixel cap')
    fig.tight_layout(); fig.savefig(dest/'matched_scores.png', dpi=160); plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    histories = {}
    for name in ('first', 'uniform', 'global', 'local', 'first_replica', 'local_replica'):
        history = read_json(out/'models'/f'{name}_history.json'); histories[name] = history
        axes[0].plot([h['epoch'] for h in history['history']], [h['val']['accuracy']*100 for h in history['history']], label=LABELS[name], alpha=.85)
        axes[1].scatter(history['train_metrics']['accuracy']*100, history['validation']['accuracy']*100, label=LABELS[name], s=55)
    axes[0].set(xlabel='Epoch', ylabel='Validation point accuracy (%)', title='Validation-only epoch selection'); axes[0].legend(fontsize=7)
    axes[1].set(xlabel='Selected-checkpoint train accuracy (%)', ylabel='Validation accuracy (%)', title='Generalization at selected checkpoint')
    axes[1].legend(fontsize=7); fig.tight_layout(); fig.savefig(dest/'learning_curves.png', dpi=160); plt.close(fig)
    diagnostic = {}; local = [r for r in rows if r['method'] == 'local']; named = [r for r in local if 'token_weights_at_selected_cell' in r]
    multiple = [r for r in named if len(r['token_weights_at_selected_cell']) > 1]
    diagnostic['pooling'] = dict(named=len(named), multi_token=len(multiple),
        first_largest_multi=sum(r['selected_token_index'] == 0 for r in multiple),
        median_first_weight_multi=float(np.median([r['token_weights_at_selected_cell'][0] for r in multiple])),
        median_effective_tokens_multi=float(np.median([np.exp(-sum(w*np.log(max(w, 1e-12)) for w in r['token_weights_at_selected_cell'])) for r in multiple])))
    distances = []
    for r in local:
        if r['point'] is None:
            continue
        c = by_case[r['sample_id']]; w, h = c['image_size']
        target = scale(shape_from_geometry(c['target_geometry']), xfact=w, yfact=h, origin=(0, 0))
        distance = Point(*r['point']).distance(target); r['distance_to_valid_region_px'] = distance
        if not r['hit']:
            distances.append(distance)
    diagnostic['median_miss_distance_px'] = float(np.median(distances)) if distances else None
    diagnostic['hits_with_pixel_tolerance'] = {str(t): sum(r['point'] is not None and r['distance_to_valid_region_px'] <= t+1e-6 for r in local) for t in (0, 2, 5, 10, 20)}
    native = {r['sample_id']: r for r in rows if r['method'] == 'qwen_native_completed_lax'}
    diagnostic['native_only_failure_stages'] = dict(Counter(r['failure_stage'] for r in local if native[r['sample_id']]['hit'] and not r['hit']))
    write_json(dest/'readout_diagnostics.json', diagnostic)
    gains = examples(dest, rows, cases, 'local_wins_over_first', lambda a, b: a['hit'] and not b['hit'])
    regressions = examples(dest, rows, cases, 'local_losses_to_first', lambda a, b: b['hit'] and not a['hit'])
    comparison = next(c for c in summary['paired_comparisons'] if c['first'] == 'local' and c['second'] == 'first')
    if comparison['ci95_pp'][0] > 0:
        verdict = '主随机种子支持逐位置词选择优于首token空间图；还需结合第二种子和跨数据集表现判断稳定性。'
    elif comparison['delta_pp'] > 0:
        verdict = '主随机种子有数值提升，但配对区间包含零，尚不能确认逐位置词选择稳定优于首token空间图。'
    else:
        verdict = '这次结构没有证明逐位置词选择优于首token空间图。不能把失败解释成单token方案已经达到VLM定位上限。'
    replication = next(c for c in summary['paired_comparisons'] if c['first'] == 'local_replica')
    global_comparison = next(c for c in summary['paired_comparisons'] if c['first'] == 'local' and c['second'] == 'global')
    if comparison['delta_pp']*replication['delta_pp'] < 0:
        verdict += ' 两个训练随机种子的提升方向不一致，稳定性不足。'
    elif comparison['delta_pp'] > 0 and replication['delta_pp'] > 0:
        verdict += ' 两个训练随机种子的数值方向一致；这不是两批独立测试图，不能合并分母扩大显著性。'
    if global_comparison['ci95_pp'][0] <= 0 <= global_comparison['ci95_pp'][1]:
        verdict += ' 逐位置与全局词选择的配对区间包含零，尚无证据确认逐位置版本更好。'
    ds_names = sorted({c['dataset'] for c in cases})
    overall = [[CHINESE[n], f'{methods[n]["hits"]}/192', f'{methods[n]["accuracy"]*100:.1f}%', f'{methods[n]["macro_accuracy"]*100:.1f}%'] for n in order]
    breakdown = [[CHINESE[n]]+[f'{methods[n]["datasets"][ds]["hits"]}/{methods[n]["datasets"][ds]["n"]}' for ds in ds_names] for n in selected]
    paired = [[CHINESE[c['first']]+' − '+CHINESE[c['second']], f'{c["delta_pp"]:+.1f}', f'[{c["ci95_pp"][0]:+.1f}, {c["ci95_pp"][1]:+.1f}]', f'{c["wins"]}/{c["losses"]}'] for c in summary['paired_comparisons']]
    enabled = {'first': 443075, 'uniform': 443075, 'global': 443108, 'local': 577381}
    training = [[CHINESE[n], h['best_epoch'], f'{h["train_metrics"]["hits"]}/618', f'{h["validation"]["hits"]}/172',
                 str(h['head_parameters'])+' / '+str(enabled[h['variant']])] for n, h in histories.items()]
    stages = [[CHINESE[n], methods[n]['selected_coarse_overlap'], methods[n]['hits'], json.dumps(methods[n]['failure_stages'], ensure_ascii=False)] for n in ['first', 'uniform', 'global', 'local', 'previous_full', 'previous_attention_only']]
    body = '<h1>目标名称逐词空间融合实验</h1><p>2026-10-31。冻结Qwen3-VL-8B，六个预先登记读出，192张全新截图。</p>'
    body += '<p><strong>'+verdict+'</strong></p><h2>同题结果</h2>'+table(['方法', '命中', '整体准确率', '四集宏平均'], overall)
    body += '<p>区域内任意点均算正确；不是精确匹配单个标准坐标。全部缺失名称、WAIT、格式失败保留在分母。宽松解析只修复完整格式变体，不用GT修正坐标。</p>'
    body += '<img src="matched_scores.png"><h2>分集结果</h2>'+table(['方法']+ds_names, breakdown)
    body += '<p>OSWorld-G只剩24张从未使用过的像素截图，其余三集各56张。所有方法使用同一192题，另报告宏平均；不把本轮整体百分比直接与上一轮不同题目比较。它不是完整官方榜单，也不是应用级OOD测试。</p>'
    body += '<h2>结构与控制</h2><p>保留每个真实生成名称token的96头注意力，不在头打分前混词。四种方案共用逐词条件化头权重、完整名称结束状态、直接粗区域监督和同一个局部细化结构。首token控制只限制注意力空间图，仍可使用完整词组隐藏状态。全局选择给所有位置相同的词权重；逐位置选择允许不同位置使用不同词的证据。没有首token位置偏置、绝对XY通道或全局空间CNN。</p>'
    body += '<p>图像特征只能影响选词；粗区域分数仍由名称注意力产生。先选粗格，再在该格四个细格中选点，细化头不能把粗格移到别处。GT仅用于训练损失和离线评分，不进入模型forward。此轮同时改变了粗区域监督和细化约束，因此与上一轮的差异不是纯粹的多token因果效应；first/uniform/global/local之间才是较受控的比较。</p>'
    body += '<p>四种方案的物理参数数量相同，但启用分支不同：逐位置选择额外使用该位置的图像特征和局部选择器，其他三种不启用这条支路。因此local与global的差异不能完全分离空间选择与额外图像条件化两项因素。global对first是更直接的多token空间图融合对照；报告同时列出物理参数和启用分支参数，不能将未使用参数当成相同有效容量。</p>'
    body += '<h2>配对差异</h2>'+table(['比较', '差值（百分点）', '探索性95%区间', '赢/输题数'], paired)
    body += '<p>按截图配对、数据集分层bootstrap 5000次，未作多重比较校正。local是预先固定主方案，没有将测试最高分版本改称主方案。</p>'
    length_names = {'single_token': '名称仅一个token', 'multiple_tokens': '名称至少两个token', 'no_name': '无可用名称'}
    body += '<h2>收益与名称长度</h2>'+table(['名称组', '样本量']+[CHINESE[n] for n in ['first', 'uniform', 'global', 'local']],
        [[length_names[group], values['n']]+[values['methods'][n]['hits'] for n in ['first', 'uniform', 'global', 'local']]
         for group, values in summary['name_length_subsets'].items()])
    body += '<p>此为事后分组诊断。单token输入在推理时没有额外词可融合，但不同方案的训练过程仍可学到不同参数，因此单token组的分数差异不能被解释成推理时使用了多个词。</p>'
    body += '<h2>训练和泛化</h2>'+table(['方案', '选中epoch', '训练命中', '验证命中', '物理 / 启用分支参数'], training)+'<img src="learning_curves.png">'
    body += '<h2>定位环节</h2>'+table(['方案', '所选粗格与GT相交', '实际点命中', '互斥阶段统计'], stages)
    local_stages = methods['local']['failure_stages']; failures = 192-methods['local']['hits']
    if failures:
        body += '<p>主方案失败中，选错粗区域占'+f'{local_stages.get("wrong_coarse_region", 0)}/{failures}（{100*local_stages.get("wrong_coarse_region", 0)/failures:.1f}%）'+'。粗格可达后仍失败的数量为'+str(local_stages.get('correct_coarse_wrong_fine', 0)+local_stages.get('fine_overlap_offset_miss', 0))+'，无可用名称为'+str(local_stages.get('no_name', 0))+'。</p>'
    body += '<p>粗格相交仅是潜在可达性，不能当作点击成功。候选点oracle也需要GT选候选，不是可部署分数。</p><pre>'+html.escape(json.dumps(diagnostic, ensure_ascii=False, indent=2))+'</pre>'
    body += '<h2>逐词语义与空间图配对诊断</h2><p>保持最后的完整词组状态，旋转此前token的隐藏状态与注意力图。二者共同换序应不影响结果；只换隐藏状态会破坏逐词配对。此为冻结权重后的诊断，不参与选模型。</p>'
    body += table(['条件', '命中/192'], [['原始逐位置选择', methods['local']['hits']], ['词与图共同换序', methods['local_pair_order_control']['hits']], ['只换词状态、破坏词图配对', methods['local_query_map_misaligned']['hits']]])
    body += '<pre>'+html.escape(json.dumps(summary['primary_diagnostics']['query_map_alignment_intervention'], ensure_ascii=False, indent=2))+'</pre>'
    body += '<h2>还保留多少空间线索</h2><p>分别检查逐词学习后的头组合峰值、第15层均头峰值以及任意词任意头峰值，是否存在与目标相交的粗格。它们需要GT挑选词/头，不能当作部署准确率；大量头和词的随机覆盖也很高，因此另列相同峰值数量随机分配到网格的期望，它也不是现实GUI基线。</p><pre>'+html.escape(json.dumps(summary['primary_diagnostics']['spatial_evidence_oracles'], ensure_ascii=False, indent=2))+'</pre>'
    body += '<h2>本轮解释</h2><p>取消首token偏置后，172个多token名称中仅'+str(diagnostic['pooling']['first_largest_multi'])+'个在选中区域仍给首token最高权重，首token权重中位数为'+f'{diagnostic["pooling"]["median_first_weight_multi"]:.3f}'+'。因此已成功摆脱固定首token依赖，但这件事没有自动带来稳定的定位提升。主种子local救回first的16题，同时损失17题；第二种子为救回19题、损失8题，不能只展示较好的种子。</p>'
    body += '<p>33次粗区域错误中，某个名称token经过学习后的头组合，其峰值粗格已与GT相交，而融合后选到了别处。这支持继续研究词图选择与组合；它不意味着可部署系统已能得到107次正确点击，也不意味着所有失败都能靠选词解决。</p>'
    body += '<p>将前面的词状态与图错配后只少命中3题，且完整词组的最后状态保持不变，邻近状态也高度相关。这不足以证明词语义没有作用，但尚未显示强而稳定的逐词语义绑定能力。局部softmax可能允许不同候选区域各自选择有利词，产生多个假峰；这是结构假设，尚未被因果实验确认。下一轮更值得检验直接的词/头选择监督、完整描述的区域对比，以及同图多目标与相似控件负例，而不是无约束地继续扩大融合网络。</p>'
    body += '<p>原生坐标命中而主方案失败的46题中，28题选错粗区域，5题选错细格，13题细格相交但点落在外面。名称任务和坐标任务的提示词、后续计算不同，因此本轮没有测出冻结VLM内部定位信息的上限，也不能断言所有原生坐标能力都已存在于名称阶段缓存中。</p>'
    reference_path = dest/'score_reference_validation.json'
    if reference_path.exists():
        reference = read_json(reference_path)
        assert reference['point_hits']['original'] == histories['local']['validation']['hits']
        body += '<h2>补充：融合分数的参考水平</h2><p>这是在正式测试之后提出、仅在验证集运行的冻结权重诊断，不是新的独立测试结果。单个词的整张空间图加同一个常数，不会改变该词自己的空间分布；但逐位置权重乘原始分数时，会把不同词的常数偏移混成位置相关项。使用相同权重的全局平均对照保持不变。</p>'
        body += '<p>验证集155个多token样本中，对各词分别加空间恒定的+1/-1，局部融合有8个粗区域改变，全局权重对照为0。对词图分别减均值或转为空间log概率后，验证命中分别为53/172和52/172，原来为52/172。因此参考水平敏感性确实存在，但这个冻结后小修正没有解释主要差距；需要归一化后重新训练并另留独立测试才能判断真正收益。</p><pre>'+html.escape(json.dumps(reference, ensure_ascii=False, indent=2))+'</pre>'
    body += '<h2>逐例检查</h2><p>下例按指定结果组内sample ID哈希排序取前四例，未人工挑选。</p>'
    if gains:
        body += '<h3>逐位置选择正确、首token错误</h3><img src="'+gains+'">'
    if regressions:
        body += '<h3>首token正确、逐位置选择错误</h3><img src="'+regressions+'">'
    body += '<h2>使用范围与审计</h2><p>训练/验证沿用618/172条真实GUI监督，600条训练有可用名称，保持数据不变以检查读出结构。测试截图排除全部先前实验像素截图；所有六个权重冻结后才采集新测试特征。Qwen权重通过原固定SHA再次校验，UI-TARS保持官方权重及坐标换算。这里只测试正例静态定位，不提供阴性误点率、真实鼠标执行或agent完整任务成功率。</p><pre>'+html.escape(json.dumps(audit, ensure_ascii=False, indent=2))+'</pre>'
    document = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>逐词空间融合实验</title><style>body{font:16px/1.65 system-ui,sans-serif;max-width:1280px;margin:40px auto;padding:0 24px;color:#152034}table{border-collapse:collapse;font-size:14px;width:100%;margin:20px 0}td,th{border:1px solid #dbe2ea;padding:8px;text-align:left}th{background:#eff5ff}img{max-width:100%;height:auto;margin:16px 0}pre{white-space:pre-wrap;background:#f3f6fa;padding:16px;font-size:13px}h2{margin-top:40px}</style>'+body+'</html>'
    (dest/'report.html').write_text(document, encoding='utf-8')
    write_json(out/'run_handoff.json', dict(status='pipeline_complete_pending_review', primary='local', fresh_test=192,
               report=str(dest/'report.html'), summary=str(dest/'summary.json'), audit=str(out/'final_audit.json'),
               method_hits={n: methods[n]['hits'] for n in order},
               must_finish=['Inspect exported score, learning, win and regression PNGs', 'Update README with verified scores', 'Deliver conclusions and limits to the user']))
    print(verdict); print('REPORT '+str(dest/'report.html'))


if __name__ == '__main__':
    main()
