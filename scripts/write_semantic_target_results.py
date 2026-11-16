"""Write the final Chinese report only after the matched external baseline."""
import json
from pathlib import Path


def read(path):return json.loads(path.read_text(encoding='utf-8'))


def main():
    root=Path(__file__).resolve().parents[1];output=root/'runs/semantic_target_v2'
    report=read(output/'comparison/summary.json');diag=read(output/'diagnostics/summary.json')
    assert read(output/'ui_tars/summary.json')['complete']
    assert read(output/'ui_tars/final_audit.json')['complete']
    assert read(output/'final_audit.json')['complete']
    methods=report['methods'];n=report['test_n']
    names={
        'selected_prompt_prefill':'旧定位头：prefill',
        'v1_original_target_mean':'旧定位头：整段目标名称平均',
        'optimized':'开发集锁定方案：目标首token + 旧定位头',
        'optimized_layer15_mean':'目标首token + 第15层32头直接平均',
        'adapted_prefill':'开发集适配定位头：prefill',
        'adapted_target':'开发集适配定位头：目标首token',
        'qwen_native_strict':'Qwen3-VL-8B直接坐标：严格格式',
        'qwen_native_completed_lax':'Qwen3-VL-8B直接坐标：完整宽松格式',
        'ui_tars_1_5_7b':'UI-TARS-1.5-7B直接坐标（本机实测）'}
    lines=['# 目标语义token读出：优化与同题模型对照（2026-10-30）','',
        '**本轮复现了目标名称阶段优于prefill的收益，但新增时间策略和紧凑提示词没有超越已有目标词平均方案；简单重训空间线性读出也没有提升。当前方法仍明显落后于同一Qwen直接输出坐标。**','',
        '## 本轮改动','',
        '冻结Qwen3-VL-8B，让它真实生成`ACTION`和`TARGET`名称，不生成坐标。对每个目标subtoken先做其自身的KV-cache forward，再读取第15/28/36层的图像attention行，保留96个头的空间分布。标准名称、坐标和目标框均未输入模型。此方案仍生成动作类别和普通目标名称，非显式读出的部分是空间位置，不是完全不生成token。','',
        '当前实现先生成并检查完整目标名称，再按冻结策略回读缓存的首token注意力；“首token读出”不表示已经实现首词到达就执行点击。全程离线截图判分，没有实际调用鼠标。','',
        '在64张独立开发截图上比较原提示词/紧凑命名两种提示与六种时间读取策略。最终锁定原提示词的目标首token，开发命中7/64；紧凑提示词最高4/64。冻结记录写入时测试缓存数量为0。','',
        '主实验保留旧的96头Save线性定位器、262144像素预算和格子中心坐标。额外单列两个开发集适配的线性定位器，各97个系数/偏置（另拟合标准化统计）：prefill与目标首token使用相同监督与四折图级交叉验证，均选择C=0.001。只训练小读出，主干训练参数为0。该额外分支在测试采集期间加入，但在任何测试分数产生之前冻结，未加载测试特征；不把它混同为纯时点改动。','',
        '## 同批实测','',
        '从ScreenSpot-v2、ScreenSpot-Pro、OSWorld-G、VenusBench-GD各取16张开发、32张测试，共64开发/128测试。按图像内容SHA排重，并排除此前实验的全部截图；应用可以重叠。这是四集均衡抽样，不是完整官方榜单。','',
        '**正确标准是点击点落入标注框或多边形的有效区域，不要求与一个指定点坐标相等，也不按距离给部分分。** 无名称、拒绝、截断和解析失败全部留在分母。本轮全是正例定位，不报告负例误点率或真实电脑任务成功率。','',
        '| 方法 | 命中/128 | 准确率 |','|---|---:|---:|']
    for method,label in names.items():
        m=methods[method];lines.append(f"| {label} | {m['hits']}/{n} | {m['accuracy']:.1%} |")
    lines+=['','每集测试32题：','',
        '| 数据集 | prefill旧头 | 目标首token旧头 | 目标首token适配头 | Qwen完整宽松坐标 | UI-TARS本机 | 网格中心上限 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for ds in ('ScreenSpot-v2','ScreenSpot-Pro','OSWorld-G','VenusBench-GD'):
        vals=[methods[m]['datasets'][ds]['hits'] for m in ('selected_prompt_prefill','optimized','adapted_target','qwen_native_completed_lax','ui_tars_1_5_7b')]
        vals.append(report['grid_center_quantization_ceiling']['datasets'][ds])
        lines.append('| '+ds+' | '+' | '.join(f'{h}/32 ({h/32:.1%})' for h in vals)+' |')
    paired=report['paired_comparisons']['selected_prompt_prefill'];ci=paired['paired_stratified_bootstrap95']
    adapt=report['adapted_readout_paired_comparisons']['adapted_prefill']
    training=report['adapted_readout_training']['heads']
    lines+=['','## 可以支持的结论','',
        f"同一个旧头下，目标首token相对prefill新增{paired['gains']}题正确、丢失{paired['losses']}题，净增{paired['accuracy_difference']*100:.2f}个百分点；按数据集分层、以图为单位的配对bootstrap区间为[{ci[0]*100:.2f}, {ci[1]*100:.2f}]个百分点。这个结果支持修正读取时点。区间是本批样本的探索性估计，没有按应用聚类，不能当成完整benchmark或应用级泛化保证。",'',
        '首token与整段目标平均均为20/128，但两者各有5题独占命中。因此没有证据称首token普遍胜过整段，也不能将开发集选择包装成超越v1的新增收益。紧凑命名未胜出；仅压短名称并不能保证更好的空间指向。','',
        f"在同样用GUI开发标注重新拟合的两个小头中，目标首token相对prefill净增{adapt['accuracy_difference']*100:.2f}个百分点；但目标适配头本身只有19/128，未优于旧头20/128。该结果支持时点信号存在，但不支持当前97参数线性适配已经解决空间解码。",'',
        f"64张开发图中，受没有可命中网格中心与缺少名称影响，实际参与prefill拟合的只有{training['prefill']['fit_images_used']}张、目标阶段只有{training['target']['fit_images_used']}张；验证与测试没有删掉这些失败。有效监督很少可能限制了适配收益，因此不能据这一组小样本实验否定学习式读出本身。",'',
        '## 当前失败发生在哪一步','',
        '主方案128题分解（互斥）：','',
        '| 结果/失败阶段 | 数量 |','|---|---:|']
    bucket=diag['buckets']['optimized']
    for key,label in [('correct','命中'),('no_representable_grid_center','目标内不存在任何可输出网格中心'),('selected_wrong_cell','存在可命中中心，但选错且所选格子不与目标相交'),('selected_cell_overlaps_but_center_misses','存在可命中中心，但所选格子仅相交、其中心在目标外'),('no_target_name','未生成可用目标名称')]:
        lines.append(f'| {label} | {bucket.get(key,0)} |')
    att=diag['raw_layer15_attention'];available=diag['paired_attention_diagnostic_n']
    lines+=['',
        f"固定网格中心的理论上限仅{report['grid_center_quantization_ceiling']['hits']}/{n}（{report['grid_center_quantization_ceiling']['hits']/n:.1%}）。71题无论如何选格子都无法命中；这解释了为什么仅调整时间窗口或头权重很难追上直接坐标。剩余可达样本中仍有明显选错区域，不能把全部失败归因于几何量化。",'',
        f"在同样{available}条有目标名称的样本上，第15层原始头平均分配给图片的总attention从{att['prefill']['image_mass']:.2%}升至{att['target']['image_mass']:.2%}；图内分配到目标相交格的相对质量从{att['prefill']['target_overlap_relative_mass']:.2%}升至{att['target']['target_overlap_relative_mass']:.2%}，{diag['target_relative_mass_increased']}/{available}题增加。目标词阶段的聚焦现象确实存在，但相交格热量不是精确点击率。",'',
        f"另有{diag['any_individual_head_peak_hits']}/{n}题至少一个单独头的峰值可以命中，而主读出命中20题；其中23题主读出失败但个别头峰值正确。这是用标注做的事后oracle诊断，不是可部署算法成绩，提示当前头组合仍未充分利用已有空间信号。",'',
        '下一步应固定已验证的目标词读取阶段，再独立处理格内坐标/局部细化和更有表达力的空间读出。当前证据不支持继续把主要精力放在prefill门控阈值或缩短命名提示词上。','',
        '## 模型对照的口径','',
        'Qwen直接坐标的严格格式是完整`CLICK`和合法坐标；宽松诊断只容忍已完整结束的逗号、空格等格式变体，不根据标注修改坐标，也不接受token上限造成的未完成输出。严格可解析69/128，完整宽松可解析111/128；命中分别29与54，因此格式确实损失了一部分原生基线分数。','',
        f"UI-TARS-1.5-7B从ModelScope镜像下载，全部7片权重SHA与官方发布LFS SHA逐一一致；主干冻结、FP16，加载无缺失或形状不匹配的权重。采用作者单点定位提示词，按Qwen2.5-VL约定将实际处理图的绝对像素坐标映射回原图；{report['valid_coordinate_outputs']['ui_tars_1_5_7b']}/128题可解析。与Qwen使用相同128题及262144像素上限，但架构、原训练和提示词不同；这不是作者默认高分辨率协议的复现。坐标解析失败保留在完整分母。本机Pro子样本0/32，尚未通过分辨率消融等实验解释它与作者完整榜单成绩的差距。",'',
        '以下仅为作者发表的完整ScreenSpot-Pro参考，不能与上面32条Pro子样本或四集混合平均直接排名：','',
        '| 方法 | 作者报告的完整Pro成绩 | 与我们的差异 |','|---|---:|---|',
        '| [UI-TARS-1.5-7B](https://github.com/bytedance/UI-TARS) | 49.6% | GUI专门训练的坐标生成模型，作者原分辨率协议 |',
        '| [GUI-Actor-7B-LiteTrain](https://arxiv.org/html/2506.03143v1) | 22.9%；加Verifier为35.8% | 冻结Qwen2-VL主干，但训练约103M新增动作模块参数 |',
        '| [MVP中的Qwen3-VL-8B](https://arxiv.org/html/2512.08529v1) | 原生55.0%；MVP为65.3% | 多视图推理、不同分辨率和评测协议，本机未复现 |','',
        'UI-TARS表中61.6%属于另一个更大的UI-TARS-1.5，不能写成发布7B模型的成绩。机制最接近的[TAG](https://arxiv.org/html/2412.10840v1)也利用生成描述词与attention过滤，但其54.8%是原始ScreenSpot上的MiniCPM-V2.5结果，不是本轮v2/Pro，故不放入同题成绩表。','',
        '## 检查与复现','',
        '384份缓存的形状、概率、真实token位置及源码哈希全部检查通过。主方案128条预测、额外适配头256条预测全部重放一致；开发/测试与历史实验的截图内容无重叠。额外读出在任何测试分数前冻结，拟合读取测试特征数为0。新增时序策略/基线解析测试共22项通过。','',
        '重放现有评分和审计（使用已冻结权重）：','',
        '```powershell',
        '.venv/python.exe -X utf8 scripts/run_semantic_target.py --phase evaluate',
        '.venv/python.exe -X utf8 scripts/evaluate_adapted_target.py',
        '.venv/python.exe -X utf8 scripts/audit_semantic_target.py',
        '.venv/python.exe -X utf8 scripts/audit_ui_tars_baseline.py',
        '.venv/python.exe -X utf8 scripts/summarize_semantic_target.py',
        '.venv/python.exe -X utf8 scripts/write_semantic_target_results.py',
        '```','',
        '从新目录复现开发选择、补充头拟合和推理，按顺序运行；先拟合后测试，不在已评分目录重训：','',
        '```powershell',
        '$taskOutput = "runs/semantic_target_repro"',
        '.venv/python.exe -X utf8 scripts/run_semantic_target.py --output $taskOutput --phase dev',
        '.venv/python.exe -X utf8 scripts/fit_target_attention_readouts.py --output $taskOutput',
        '.venv/python.exe -X utf8 scripts/run_semantic_target.py --output $taskOutput --phase test',
        '.venv/python.exe -X utf8 scripts/evaluate_adapted_target.py --output $taskOutput',
        '.venv/python.exe -X utf8 scripts/run_ui_tars_baseline.py --output $taskOutput',
        '.venv/python.exe -X utf8 scripts/summarize_semantic_target.py --output $taskOutput',
        '```','',
        '协议见[SEMANTIC_TARGET_PROTOCOL.md](SEMANTIC_TARGET_PROTOCOL.md)。后续对区域选择、中心量化、命名偏差和读出损失的交叉诊断见[SEMANTIC_TARGET_FAILURE_ANALYSIS.md](SEMANTIC_TARGET_FAILURE_ANALYSIS.md)。原始数据、生成名称、冻结配置、模型来源、逐例预测、配对区间和完整审计位于`runs/semantic_target_v2/`；可视化为`comparison/comparison.png`和PDF。','']
    (root/'docs/SEMANTIC_TARGET_RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
    print('Wrote docs/SEMANTIC_TARGET_RESULTS.md')


if __name__=='__main__':main()
