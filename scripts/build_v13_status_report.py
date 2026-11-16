"""Build the human-readable V13 status report from sealed evaluation results.

This script only reads experiment artifacts. It does not run a model or alter
predictions, selection, metrics, or any sealed experiment file.
"""
from __future__ import annotations

import hashlib
import html
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "runs/multiregion_v13"
OUTPUT = ROOT / "reports/ClickAgent_V13_Status_2026-11-03.html"


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def esc(value):
    return html.escape(str(value))


def fmt_score(method):
    percent = (Decimal(method['hits']) * 100 / Decimal(method['n'])).quantize(Decimal('.01'), rounding=ROUND_HALF_UP)
    return f"{method['hits']}/{method['n']}<span class='percent'>{percent}%</span>"


def table(headers, rows, widths=None, row_classes=None):
    col = ""
    if widths:
        col = "<colgroup>" + "".join(f"<col style='width:{w}%'>" for w in widths) + "</colgroup>"
    body = []
    for i, row in enumerate(rows):
        cls = (row_classes or {}).get(i, "")
        body.append(f"<tr class='{cls}'>" + "".join(f"<td>{v}</td>" for v in row) + "</tr>")
    return ("<div class='table-wrap'><table>" + col + "<thead><tr>"
            + "".join(f"<th scope='col'>{esc(h)}</th>" for h in headers)
            + "</tr></thead><tbody>" + "".join(body) + "</tbody></table></div>")


def main():
    summary = read(RUN / "evaluation/test/summary.json")
    analysis = read(RUN / "evaluation/mechanism_analysis/results.json")
    native = read(RUN / "evaluation/mechanism_analysis/native_output_diagnostic.json")
    supplement = read(RUN / "evaluation/supplement/summary.json")
    completion = read(RUN / "completion.json")
    assert summary["complete"] and analysis["complete"] and completion["complete"]
    assert summary["n"] == 192 and supplement["n"] == 6
    for relative, expected in completion["artifact_sha256"].items():
        path = RUN / relative
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, path

    m = summary["methods"]
    s1 = "c3_top1_s1_loso__K16"
    s2 = "c3_top1_s2_loso__K16"
    assert (m[s1]["hits"], m[s2]["hits"], m["qwen_native_completed_lax"]["hits"]) == (55, 57, 40)
    chain = analysis["cohorts"]["test"]["failure_chains"]["c3_top1_s1_loso_top4_gui__K16"]
    assert list(chain["stages"].values()) == [81, 22, 10, 14, 65]

    main_rows = [
        ["原生 Qwen 坐标输出", "原图直接生成坐标", fmt_score(m["qwen_native_completed_lax"]), "共用参考", "不使用"],
        ["单局部 Top1", "普通验证集选模", fmt_score(m["c1_top1_s1_val__K16"]), fmt_score(m["c1_top1_s2_val__K16"]), "不使用"],
        ["三局部 Top1", "普通验证集选模", fmt_score(m["c3_top1_s1_val__K16"]), fmt_score(m["c3_top1_s2_val__K16"]), "不使用"],
        ["三局部 Top1 主方案", "评分器来源 CV 选模", fmt_score(m[s1]), fmt_score(m[s2]), "不使用"],
        ["三局部 Top4 目标", "来源 CV 后直接取 Top1", fmt_score(m["c3_top4_s1_loso__K16"]), fmt_score(m["c3_top4_s2_loso__K16"]), "不使用"],
        ["主方案 Top4 后接 GUI", "Top1 训练头预筛四点", fmt_score(m["c3_top1_s1_loso_top4_gui__K16"]), fmt_score(m["c3_top1_s2_loso_top4_gui__K16"]), "只验证短名单"],
        ["Top4 训练头后接 GUI", "专门学习保留正确点", fmt_score(m["c3_top4_s1_loso_top4_gui__K16"]), fmt_score(m["c3_top4_s2_loso_top4_gui__K16"]), "只验证短名单"],
        ["单局部完整池 GUI", "同一批候选逐点验证", fmt_score(m["gui_c1__K16"]), "共用参考", "完整候选池"],
        ["三局部完整池 GUI", "同一批候选逐点验证", fmt_score(m["gui_c3__K16"]), "共用参考", "完整候选池"],
        ["全图提议器 Top1", "上游原始排序参考", fmt_score(m["global_top1"]), "共用参考", "不使用"],
    ]
    main_table = table(["方法", "设置", "S1 预定主种子", "S2 稳健性种子", "GUI 验证器"],
                       main_rows, [24, 23, 18, 18, 17], {3: "main-row", 5: "main-row"})

    source_rows = []
    for label, name in [("原生 Qwen 坐标", "qwen_native_completed_lax"),
                        ("单局部内部头 S1", "c1_top1_s1_val__K16"),
                        ("三局部主内部头 S1", s1), ("三局部主内部头 S2", s2),
                        ("主头 Top4 后接 GUI S1", "c3_top1_s1_loso_top4_gui__K16"),
                        ("主头 Top4 后接 GUI S2", "c3_top1_s2_loso_top4_gui__K16")]:
        method = m[name]
        source_rows.append([label, fmt_score(method["by_source"]["screenspot_pro"]),
                            fmt_score(method["by_source"]["venusbench_gd"]), fmt_score(method)])
    source_table = table(["方法", "ScreenSpot Pro 96 例", "Venus 96 例", "合计 192 例"], source_rows, [40, 20, 20, 20])

    comparisons = [
        ("完整主改动 S1", s1, "c1_top1_s1_val__K16"),
        ("完整主改动 S2", s2, "c1_top1_s2_val__K16"),
        ("多区域 同普通选模 S1", "c3_top1_s1_val__K16", "c1_top1_s1_val__K16"),
        ("多区域 同普通选模 S2", "c3_top1_s2_val__K16", "c1_top1_s2_val__K16"),
        ("来源 CV 的额外增量 S1", s1, "c3_top1_s1_val__K16"),
        ("来源 CV 的额外增量 S2", s2, "c3_top1_s2_val__K16"),
        ("多区域 同 GUI 验证器", "gui_c3__K16", "gui_c1__K16"),
    ]
    paired = {(p["method"], p["reference"]): p for p in summary["paired_comparisons"]}
    # Decimal rounding of binary floating-point CI endpoints is for display only.
    def pp(x):
        return f"{x * 100 + (1e-9 if x >= 0 else -1e-9):.2f}"
    paired_rows = []
    for label, method, reference in comparisons:
        p = paired[method, reference]
        paired_rows.append([label, f"{m[method]['hits']} 对 {m[reference]['hits']}",
                            f"{p['wins']} / {p['losses']}", f"+{pp(p['delta'])}",
                            f"[{pp(p['ci95'][0])}, {pp(p['ci95'][1])}]"])
    paired_table = table(["对照", "命中数", "救回 / 损失", "增量 百分点", "配对 95% 区间"], paired_rows, [34, 15, 15, 15, 21])

    coverage_table = table(["诊断阶段", "单局部", "三局部", "含义"], [
        ["局部框覆盖目标", "79/192", "124/192", "至少一个裁剪与目标有正面积交集"],
        ["全部已解码点可达", "85/192", "111/192", "枚举现有提议器产生的点后用 GT 判断"],
        ["K16 候选池可达", "73/192", "89/192", "经过配额截取和整数点去重"],
        ["评分器 Top4 可达 S1", "59/192", "79/192", "正确点仍在前四个候选中"],
        ["Top4 后接 GUI 命中 S1", "49/192", "65/192", "实际选择的一个点命中目标"],
    ], [30, 16, 16, 38])

    loss_rows = [
        ["整个已解码点集中无正确点", "81", "42.19%", "搜索范围或现有表示及坐标解码尚未产出可用点"],
        ["正确点被配额截取丢掉", "22", "11.46%", "已有正确点，但全局与局部名额分配未保住"],
        ["正确点被评分 Top4 丢掉", "10", "5.21%", "K16 内有点，内部压缩仍有选择损失"],
        ["GUI 在短名单中选错", "14", "7.29%", "Top4 内有正确点，最后的语义核验失败"],
        ["最终成功", "65", "33.85%", "主种子 Top4 加 GUI 组合的实际点命中"],
    ]
    loss_table = table(["互斥归因", "题数", "占全部 192 例", "对应环节"], loss_rows, [31, 10, 18, 41])

    chart_rows = []
    for label, name in [("原生坐标", "qwen_native_completed_lax"),
                        ("单局部内部头 S1", "c1_top1_s1_val__K16"),
                        ("三局部主内部头 S1", s1),
                        ("主头 Top4 加 GUI S1", "c3_top1_s1_loso_top4_gui__K16"),
                        ("三局部完整池 GUI", "gui_c3__K16")]:
        v = m[name]
        chart_rows.append(f"<div class='bar-row'><span>{esc(label)}</span><div class='bar-track'><div class='bar' style='width:{v['accuracy'] / .5 * 100:.4f}%'></div></div><strong>{v['accuracy']:.2%}</strong></div>")
    chart = "".join(chart_rows)

    source_links = []
    for label, relative in [
        ("实验协议", "runs/multiregion_v13/protocol.json"),
        ("主测试指标和配对区间", "runs/multiregion_v13/evaluation/test/summary.json"),
        ("主实验完整报告", "runs/multiregion_v13/evaluation/report.html"),
        ("机制分析", "runs/multiregion_v13/evaluation/mechanism_analysis/report.html"),
        ("机制分析原始统计", "runs/multiregion_v13/evaluation/mechanism_analysis/results.json"),
        ("原生坐标输出诊断", "runs/multiregion_v13/evaluation/mechanism_analysis/native_output_diagnostic.json"),
        ("补充测试统计", "runs/multiregion_v13/evaluation/supplement/summary.json"),
        ("最终审计", "runs/multiregion_v13/final_audit.json"),
        ("完成状态及文件校验", "runs/multiregion_v13/completion.json"),
        ("当前方法的工程说明", "docs/CURRENT_METHOD.md"),
        ("多区域评分器实现", "src/clickagent/multiregion_scorer.py"),
        ("多区域候选构造", "src/clickagent/multiregion_proposals.py"),
    ]:
        path = ROOT / relative
        assert path.is_file(), path
        source_links.append(f"<li><a href='{esc(path.as_uri())}'>{esc(label)}</a><code>{esc(relative)}</code></li>")

    report = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ClickAgent V13 阶段研究汇报</title>
<style>
:root{--ink:#18202b;--muted:#536173;--blue:#315d85;--border:#d9d9d9;--pale:#f2f6fa}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:#edf0f3;color:var(--ink);font-family:'Segoe UI','Microsoft YaHei','DengXian',sans-serif;font-size:16px;line-height:1.85}
main{max-width:1080px;margin:32px auto;padding:52px 62px 56px;background:#fff}
h1,h2,h3{color:#000;line-height:1.45}h1{font-size:31px;letter-spacing:.025em;margin:0 0 10px}h2{font-size:23px;margin:43px 0 16px}h3{font-size:18px;margin:25px 0 10px}p{margin:0 0 14px}strong{font-weight:650}
.meta,.note,figcaption{color:var(--muted);font-size:14px}.meta{margin-bottom:28px}.lead{font-size:18px;line-height:1.9}.note{margin-top:12px}.eyebrow{font-size:14px;letter-spacing:.12em;color:var(--muted);margin-bottom:6px}
nav{margin:26px 0 32px;display:flex;gap:8px 24px;flex-wrap:wrap}nav a{font-size:14px}a{color:#235d8c;text-decoration:none}a:hover{text-decoration:underline}
ul,ol{padding-left:1.55em;margin:10px 0 20px}li{margin:7px 0}li p{margin:6px 0 10px}
.table-wrap{overflow-x:auto;margin:14px 0 20px}table{width:100%;border-collapse:collapse;table-layout:fixed;font-size:14px;line-height:1.65}th,td{border:1px solid var(--border);padding:11px 12px;vertical-align:middle;overflow-wrap:anywhere}th{background:#e4edf5;color:#000;font-weight:650;text-align:left}td:not(:first-child){text-align:center}tbody tr:nth-child(even){background:#f7f9fb}tr.main-row{font-weight:650;background:#edf4fa!important}.percent{display:block;color:var(--muted);font-size:13px;font-weight:400}
.bar-chart{margin:22px 0 14px}.bar-row{display:grid;grid-template-columns:185px 1fr 70px;gap:16px;align-items:center;margin:12px 0;font-size:14px}.bar-track{background:#eef1f4;height:19px}.bar{background:#496f93;height:100%}.bar-row strong{text-align:right}.axis{margin:8px 86px 0 201px;display:flex;justify-content:space-between;font-size:12px;color:var(--muted)}
.flow{margin:22px 0 18px;overflow-x:auto}.flow svg{width:100%;height:auto;min-width:700px;display:block}.flow text{font-family:'Segoe UI','Microsoft YaHei','DengXian',sans-serif;fill:#18202b}.flow .minor{fill:#536173}
.chain{font-size:21px;text-align:center;letter-spacing:.03em;margin:22px 0;line-height:1.8}code{font-family:Consolas,monospace;font-size:12px;background:#f5f6f8;overflow-wrap:anywhere}.sources{font-size:14px}.sources code{display:block;background:none;color:var(--muted);font-size:12px}.end{margin-top:34px;font-size:13px;color:var(--muted)}
@media(max-width:760px){main{margin:0;padding:30px 20px}body{font-size:15px}h1{font-size:26px}h2{font-size:21px}table{min-width:760px}.bar-row{grid-template-columns:138px 1fr 62px;gap:8px;font-size:12px}.axis{margin-left:146px;margin-right:70px}}
@media print{@page{size:letter;margin:17mm 16mm}body{background:#fff;font-size:10.5pt;line-height:1.75}main{margin:0;padding:0;max-width:none}h1{font-size:23pt}h2{font-size:16pt;margin-top:24pt;break-after:avoid}h3{font-size:12pt;break-after:avoid}p,li{orphans:3;widows:3}.meta,.note,figcaption{font-size:9pt}.lead{font-size:12pt}nav{display:none}.table-wrap{overflow:visible}table{font-size:9pt;min-width:0}th,td{padding:6pt 7pt}thead{display:table-header-group}tr{break-inside:avoid}.percent{font-size:8.5pt}.flow,.bar-chart{break-inside:avoid}.flow svg{min-width:0}.bar-row{font-size:9pt;grid-template-columns:150px 1fr 64px}a{color:#18202b}.sources{font-size:9pt}.end{font-size:8.5pt}*{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
</style></head><body><main>
<div class="eyebrow">研究进展与定位机制</div>
<h1>ClickAgent V13 阶段研究汇报</h1>
<p class="meta">2026 年 11 月 3 日　｜　Qwen3 VL 8B 冻结主干　｜　V13 已完成训练 评测与终审</p>
<p class="lead"><strong>我们已经把生成目标名称时的内部状态，做成了一条可训练的 GUI 定位路线。</strong>最新独立测试中，内部评分器命中 55/192 和 57/192，原生坐标对照命中 40/192；内部头预筛 Top4 后接 GUI 验证器，命中 65/192 和 67/192。当前最明确的进展来自多区域搜索增加正确候选，定位准确率仍有较大提升空间。</p>
<p>本轮验证的是静态截图上的目标定位。核心机制仍然是读取模型实际生成目标名称时的全部 token，而后通过内部特征产生和选择点击点。它仍需生成名称文本；当前没有校准好的动作权限门控，也没有据此完成通用电脑 Agent 的多步执行验证。</p>
<nav aria-label="文档目录"><a href="#method">当前方法</a><a href="#protocol">评测口径</a><a href="#scores">跑分对比</a><a href="#evidence">改动的作用</a><a href="#bottleneck">失败链条</a><a href="#cost">成本与适用范围</a><a href="#next">判断与下一步</a><a href="#sources">证据索引</a></nav>

<section id="method"><h2>1 当前方法的完整流程</h2>
<p>我们不再用早期的全局 CLICK 分类器决定点击位置。当前流程由冻结的视觉语言模型、已训练后冻结的空间提议器，以及一个本轮训练的轻量候选评分器组成。GUI 验证器是可选的末端选择模块。</p>
<figure class="flow"><svg viewBox="0 0 960 300" role="img" aria-labelledby="flowTitle flowDesc">
<title id="flowTitle">从目标名称内部状态到点击坐标</title><desc id="flowDesc">截图和指令进入冻结 Qwen，生成目标名称并逐 token 读取状态，提出最多三个局部裁剪，独立重新生成名称，合并全图与局部候选形成 K16 池，由内部评分器取 Top1，或预筛 Top4 后由 GUI 验证器选择。</desc>
<defs><marker id="arrow" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8" fill="#75899b"/></marker></defs>
<g fill="#f1f5f8" stroke="#b6c5d2"><rect x="3" y="20" width="210" height="92" rx="4"/><rect x="251" y="20" width="210" height="92" rx="4"/><rect x="499" y="20" width="210" height="92" rx="4"/><rect x="747" y="20" width="210" height="92" rx="4"/><rect x="499" y="182" width="210" height="92" rx="4"/><rect x="747" y="157" width="210" height="54" rx="4"/><rect x="747" y="235" width="210" height="54" rx="4"/></g>
<g stroke="#75899b" stroke-width="2" fill="none" marker-end="url(#arrow)"><path d="M213 66 H247"/><path d="M461 66 H495"/><path d="M709 66 H743"/><path d="M852 112 V137 H604 V178"/><path d="M709 207 H726 V184 H743"/><path d="M709 245 H726 V262 H743"/></g>
<g text-anchor="middle" font-size="17"><text x="108" y="49">截图与指令</text><text x="108" y="76">冻结 Qwen 生成目标名</text><text class="minor" x="108" y="98" font-size="13">模型实际生成的名称文本</text><text x="356" y="49">逐名称 token 采集</text><text x="356" y="76">96 头与隐藏状态</text><text class="minor" x="356" y="98" font-size="13">保留 token 和 head 维度</text><text x="604" y="49">全图提出最多三处裁剪</text><text x="604" y="76">局部重新生成和采集</text><text class="minor" x="604" y="98" font-size="13">各局部独立读取证据</text><text x="852" y="49">解码原图实际点击点</text><text x="852" y="76">配额合成 K16 候选池</text><text class="minor" x="852" y="98" font-size="13">候选不含原生文本坐标</text><text x="604" y="213">共享多视图评分器</text><text x="604" y="240">1 230 338 个可训练参数</text><text class="minor" x="604" y="262" font-size="13">只排名已有点 不移动坐标</text><text x="852" y="190">直接取 Top1 点击点</text><text x="852" y="267">Top4 后接 GUI 验证器</text></g></svg>
<figcaption>图 1　内部读出负责产生和排序坐标。两种末端路径分别报告，避免把外部验证器的能力归到内部头上。</figcaption></figure>
<ol>
<li><strong>在目标名称出现时采集状态。</strong>冻结 Qwen3-VL-8B-Instruct 接收截图和任务，生成 ACTION/TARGET 及目标名称。每个已经生成的名称 token 再送入带 KV cache 的前向，读取属于该 token 自身的状态。采集语言解码器第 15、28、36 层共 96 个注意力头，以及目标隐藏状态、图像位置隐藏状态和合并前视觉特征。图像位置隐藏状态来自预填充；关键的目标查询和注意力来自名称 token。</li>
<li><strong>从内部证据构造实际坐标。</strong>冻结的 V12 空间提议器从名称 token 与各头的空间峰值提出父格，结合目标语义和图像内容排序，再用父格内 2×2 子格选择及连续偏移解码，得到原图整数点。已经超出早期“平均热图取最亮格子中心”的机制。</li>
<li><strong>增加目标相关的局部重看。</strong>按全图粗候选排名选最多三个局部框，第一个框严格复用单局部方案，后续框与已有框的两两 IoU 不超过 0.25。每个局部重新生成名称、采集状态并解码候选；局部输入包含裁剪图和原指令。</li>
<li><strong>按固定预算合成候选池。</strong>K16 是最多 16 点的预算：一半名额给全图，另一半在局部视图间轮流分配。先截取名额，再按原图整数坐标去重，不补位。K4、K8、K16、K32 为嵌套池，原生 Qwen 生成的文本坐标始终不进入这些池。</li>
<li><strong>以候选点为条件读取证据并排名。</strong>1,230,338 参数的评分器对每个点查询全图和覆盖该点的局部空间邻域，结合目标隐藏状态、图像内容与逐 token/head 注意力。局部编码器共享权重，局部候选表示取最大聚合。评分器只重排已存在的点，无法修正池内所有坐标都错误的问题。</li>
<li><strong>输出一个点或交给验证器终选。</strong>内部路线直接取 Top1；组合路线保留内部 Top4，由冻结的 GUI-Actor-Verifier-2B 对标记候选的图像做 True/False log-odds 评分，取最大值。GUI 模块负责语义核验和选择，不生成新坐标。</li>
</ol>
<p><strong>训练与冻结的边界。</strong>Qwen 主干冻结，V12 上游提议器在本轮冻结，本轮只训练内部评分器；GUI 验证器使用外部预训练权重，本项目没有重训，也没有用它的分数作为评分器教师。单局部和三局部评分器参数量相同。</p>
</section>

<section id="protocol"><h2>2 数据与指标口径</h2>
<p>评分器沿用 873 条训练指令和 172 条验证指令，数据来自 AriaUI、Omniact、OS-Atlas 和 RICO WidgetCaption。训练候选使用按截图及近似图分组的五折 OOF 提议器，避免由见过本题的上游模型为本题产生训练候选；验证和部署使用完整训练集训练的提议器。</p>
<p>本轮三类训练配置为单局部 Top1、三局部 Top1、三局部 Top4，各固定两个初始化种子，共六个完整训练。三局部额外使用四个来源的评分器留一验证来选择公共 epoch，共八个评分器 CV 训练。两种目标都选第 7 轮，并应用于两个完整训练种子。主种子 S1 和稳健性种子 S2 在测试前固定，全部模型及策略封存后才对新测试样本运行模型前向。</p>
<p><strong>主测试共 192 例：</strong>ScreenSpot-Pro 96 例，Venus 96 例，严格排除历史同图及近似图。ScreenSpot-v2 在同样排重规则下仅剩 6 例，单独作为补充。V13 的来源构成与若干旧轮次的 64+64+64 队列不同，因此不将跨轮次分数直接相减。</p>
<p><strong>准确率按点击点是否落入任意有效目标框或多边形计算。</strong>不要求等于标注中心，不按距离连续给分，也不额外放宽若干像素的半径。空候选池、无输出及所有定位失败都保留在 192 的分母中。后文“可达”表示用 GT 检查候选集合里是否存在正确点，它只用于诊断，不能当作系统真实准确率。</p>
<p class="note">来源 CV 只用于评分器选模。上游提议器已见过这些来源，并用过原验证标签，因此这不是整个系统的未见领域留一评估。表中均为本地固定样本测试，不能作为完整公开榜单复现成绩。</p>
</section>

<section id="scores"><h2>3 同一主测试集上的跑分对比</h2>
<p>主结果按测试前确定的 S1 报告，S2 用于观察训练随机性的影响。表中 K16 均为候选上限；原生坐标直接在原图上生成，不受候选池限制。百分比对应整批 192 例。</p>
MAIN_TABLE
<p class="note">“共用参考”表示该模型或策略不随评分器训练种子变化，只评测一份，不是第二种子缺失。GUI 短名单成绩从同一完整候选池的固定逐点评分中取子集得到，尚未独立运行 Top4 端到端计时。</p>
<figure class="bar-chart" aria-label="主种子与同题基线准确率对比">CHART<div class="axis"><span>0%</span><span>25%</span><span>50%</span></div><figcaption>图 2　主种子的同题对比。横轴统一为准确率，50% 为显示范围上界。</figcaption></figure>
<p>内部主方案相对原生坐标 S1 多命中 15 例，S2 多命中 17 例；配对增量分别为 +7.81 和 +8.85 个百分点。这个优势是额外裁剪、经过监督的提议器与评分器共同带来的系统级结果，不能全部归因于内部坐标解码优于语言坐标解码。</p>
<h3>分来源结果</h3>
SOURCE_TABLE
<p>当前主方案在 Venus 上明显更容易成功；ScreenSpot-Pro 仍然困难。S1 到 S2 的变化也不均匀：主内部头在 Pro 上由 19 例降到 15 例，在 Venus 上由 36 例升到 42 例。只看总分会掩盖来源差异。</p>
<p><strong>其他模型的对照范围。</strong>本轮同题参考包含原生 Qwen 坐标和 GUI-Actor 验证器组合。UI-TARS 等模型没有在 V13 这批新题上重跑；历史轮次的成绩不能放入此表作为同题排名，完整公开榜单也不与这 192 例子集直接混比。</p>
</section>

<section id="evidence"><h2>4 哪些改动带来了提升</h2>
PAIRED_TABLE
<p class="note">区间为按来源分层、按截图簇重采样 2000 次的配对 95% 区间，条件是已训练模型固定；它不包含训练随机性。次要对照未作多重比较校正，区间跨零时不能宣称收益已确定。</p>
<p>两个种子的主比较都朝正方向变化，但 S2 的区间包含零，跨训练种子的稳定性仍需进一步验证。</p>
<h3>多区域搜索有最明确的支持</h3>
<p>多区域使局部裁剪覆盖目标的题数从 79 增至 124，K16 池内有正确点的题数从 73 增至 89，新增 19 例、损失 3 例。保持同一个 GUI 验证器，完整池的最终命中由 51 增至 65，净增 14 例，配对区间不跨零。这支持扩大空间搜索及相应多视图证据的价值，但尚未单独分离每一种融合算子的贡献。</p>
<h3>来源 CV 选模的收益仍不确定</h3>
<p>三局部条件下，来源 CV 比普通验证选模让内部 Top1 多命中 6 例和 5 例，但两个种子的差值区间都跨零。主比较同时改变了搜索和选模；不能把 S1 的全部 18 例净增都归因于搜索，也不能据此宣布泛化问题已经解决。</p>
<h3>专门 Top4 训练目标尚未获支持</h3>
<p>在同一三局部池、同一第 7 轮下，Top1 目标训练头的 Top4 在 S1、S2 中分别保留 79、79 例，专门 Top4 目标分别保留 79、78 例。接 GUI 后，前者分别命中 65、67 例，后者为 60、65 例。专门目标没有增加正确点保留，也没有带来最终成功收益。这个损失只奖励“至少一个正确点进前四”，没有学习排除会误导 GUI 的负例；这是一项机制上的不足，但当前对照仍不足以把全部差异归因于某一种干扰。结论限于本轮实现。</p>
<p>主头 Top4 加 GUI 的 S1 与完整池 GUI 都命中 65 例，但逐题有 6 例救回、6 例损失，差值区间为 [-3.65, 3.65] 个百分点。同总分不能视为完全等效或无损压缩。S2 普通验证选模组合的最高观测为 69/192；它属于已预定的次要对照，本报告不据此换主种子或主方案。</p>
</section>

<section id="bottleneck"><h2>5 定位失败主要发生在哪一步</h2>
COVERAGE_TABLE
<p class="chain">192 例 → 全部解码点可达 111 → K16 可达 89 → Top4 可达 79 → GUI 命中 65</p>
LOSS_TABLE
<p class="note">上表是主种子组合路线的互斥归因，题数相加为 192。111/192 只表示当前冻结提议器已解码坐标集合的可达性，不是原始注意力信息的上限，更不是冻结 Qwen 的能力上限。</p>
<h3>候选构造是当前最大的损失来源</h3>
<p>81 例在全部现有解码点中都没有正确点，另有 22 例在 K16 配额截取时丢掉所有正确点。二者共 103 例，占全部样本 53.65%。在这些题目上，单独改进 K16 池后的评分器或 GUI 验证器无法成功，因为正确点已经不存在。</p>
<p>增加裁剪有帮助，也带来大量无效视图。实际 561 次局部读取中，420 个局部框与目标分离；第一、第二、第三裁剪没有生成目标名称 token 的次数分别为 49、77、90，各有 187 个视图。尚需区分目标未在裁剪中、缺少全图上下文和模型理解失败，不能仅凭未生成名称就断言内部不存在定位信息。</p>
<h3>内部评分器仍有明显的选择损失</h3>
<p>直接内部 Top1 的 S1 失败 137 例，可拆成 5 例空池、98 例非空但无正确点、34 例池内有正确点却选错。它选对 55/89 个可达目标，同池 GUI 选对 65/89。可见独立内部头已经有作用，但当前仍没有可靠替代专用验证器。</p>
<p>K16 扩到 K32 后，候选可达仅由 89 增至 95，多覆盖 6 例。扩大最终名额可以缓解一部分损失，但仍需改进目标相关搜索、坐标候选的产生方式和配额分配，而不能只依靠增加 K。</p>
</section>

<section id="cost"><h2>6 计算成本和原生对照的解释</h2>
<p><strong>K16 上限相同，实际候选数和上游计算量不同。</strong>单局部总计 2600 个候选，平均每题 13.54 点；三局部总计 2911 个，平均 15.16 点。三局部需要更多 Qwen 裁剪读取及名称生成，每个视图图像像素上限为 262144。当前没有“与原生单图同计算量”的结论。</p>
<p>主头 Top4 总计保留 748 个待验证候选，约为完整三局部池 2911 个的 25.7%。这说明内部头有潜在的预筛价值，但它只是验证图像数量代理；本轮复用了完整池的缓存分数，尚未分别实测 Top4 与完整池端到端时延。</p>
<h3>原生坐标的低分并非主要由格式解析造成</h3>
<p>192 份原生输出全部正常 EOS 结束，没有截断。31 份是真实文本 WAIT，161 份含有效坐标；其中 46 份逗号形式 CLICK 已由正式 completed-lax 解析接纳，包含 17 个命中。坐标尺度与像素换算逐条检查一致。不能把严格解析元数据中的 77 次 wait 都当成语义拒答，也不能再次给宽松结果补加这 17 例。</p>
<p>原生提示词允许目标缺失、禁用、歧义或无法确定时 WAIT，内部评分器则在非空池中强制选点。事后仅看原生有有效坐标的 161 题，原生命中 40 例，内部 S1、S2 分别命中 50、51 例；剩余真实 WAIT 的 31 题，两种子分别另命中 5、6 例。收益并非全来自拒答差异，但此子集按原生行为划分，有选择偏差，不能代替原 192 分母或作为新独立结论。</p>
<h3>当前没有动作权限和真实执行成功率结论</h3>
<p>评分器训练的 null 表示“当前候选池没有正确点”，没有校准成“用户是否允许点击”。本轮没有成组的禁止点击、按钮禁用和目标缺失阴性样本，也没有真实应用中的鼠标操作及结果反馈，因此静态命中率不等于动作安全性、误点率或多步任务成功率。</p>
</section>

<section id="next"><h2>7 当前判断与下一步优先级</h2>
<p><strong>我们已验证内部状态能够支持有用的候选定位与预筛。</strong>在本轮固定的新题上，多区域路线带来正向结果；主内部头高于原生坐标对照，组合路线进一步提高命中。但内部独立准确率仍只有约 29%，组合约 34%，训练种子和来源差异仍然明显。</p>
<p>方法保留了最初观察到的现象：目标名称 token 的内部注意力提供空间证据。当前系统的效果还依赖目标语义、视觉内容、经过监督的提议器与评分器，不能把全部成绩解释为某个注意力头直接编码了最终动作。更早版本中“某个 token/head 存在正确峰值”的 GT 诊断，也不能直接移作 V13 的定位准确率。</p>
<ol>
<li><strong>先提高正确候选的产生与保留。</strong>优先研究有全图语义上下文的局部重看、目标相关的多区域选择，以及替代固定全局一半局部一半的自适应配额。分别记录裁剪覆盖、名称可用性、全部解码点可达和 K16 保留，判断收益发生在哪一步。</li>
<li><strong>再强化同池候选的语义区分。</strong>以真实困难负例学习“这个点是否满足当前指令”，重点观察已有正确点的样本能否选对。继续把内部 Top1 与 Top4 预筛作为两个用途评测，暂不将本轮专门 Top4 损失作为默认升级。</li>
<li><strong>用新的独立队列确认后再推进动作层。</strong>在验证集确定搜索、配额和训练策略，封存后测试；同时报告来源差异和两个固定种子。后续补上缺失、禁用、禁止动作样本，再校准拒绝策略，并实测真正执行 Top4 的端到端延迟。</li>
</ol>
<p>ScreenSpot-v2 的补充 6 例中，主内部 S1 命中 4 例，S2 命中 5 例，单局部两个种子、GUI 及原生参考均为 5 例；样本过少，不支持泛化外推，也未与主测试合并。</p>
</section>

<section id="sources"><h2>8 完成状态和证据索引</h2>
<p>V13 已完成预测、评测和终审。记录显示 98 项核心测试、17 项审计兼容回归通过；24 组原始状态到模型输入的重建完全一致，120 次固定 CPU 前向的落点及 Top4 顺序一致，7141 个不同样本与坐标组合的独立几何判分无差异。这些核验支持实现和指标的一致性，不替代统计上的泛化检验。</p>
<p class="note">原审计器遗漏了归档中的两个附带 prefill 数组，测试后增加了严格兼容适配器，核对六个归档键并只向评分器传入原四类特征。原失败日志保留，最终审计通过；没有因此改变模型、预测、选模或判分结果。</p>
<ul class="sources">SOURCE_LINKS</ul>
<p class="end">生成依据为已封存 V13 结果。此文档重新整理方法与证据，不运行模型、不调整测试策略、不改写历史成绩。</p>
</section>
</main></body></html>"""
    report = report.replace("MAIN_TABLE", main_table).replace("SOURCE_TABLE", source_table)
    report = report.replace("PAIRED_TABLE", paired_table).replace("COVERAGE_TABLE", coverage_table)
    report = report.replace("LOSS_TABLE", loss_table).replace("CHART", chart).replace("SOURCE_LINKS", "".join(source_links))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(report, encoding="utf-8")
    print(str(OUTPUT))
    print(f"bytes={OUTPUT.stat().st_size}; source checksums verified={len(completion['artifact_sha256'])}")


if __name__ == "__main__":
    main()
