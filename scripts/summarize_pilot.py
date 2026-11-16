"""Render the actual cached pilot results into repository documentation."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load(relative):
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def percent(value):
    return "未评估" if value is None else f"{100 * value:.1f}%"


def main():
    visual = load("runs/pilot/evaluation/metrics.json")
    no_image = load("runs/no_image/evaluation/metrics.json")
    run = load("runs/pilot/features/run.json")
    metadata = [json.loads(line) for line in (ROOT / "runs/pilot/features/metadata.jsonl").read_text(encoding="utf-8").splitlines()]
    test = [row for row in metadata if row["split"] == "test"]
    recognized = [row for row in test if row["generated_decision"] in {"click", "describe", "forbid"}]
    binary_correct = sum((row["generated_decision"] == "click") == (row["label"] == "click")
                         for row in recognized)
    direct_correct = sum(row["generated_decision"] == row["label"] for row in test)
    lines = [
        "# 首轮冻结 Qwen3-VL 门控实验结果（2026-10-28）", "",
        "已实际运行 Qwen3-VL-8B-Instruct；以下数字来自本项目缓存特征与独立测试集，不是模拟数据特征。界面图片为程序化生成的合成 GUI。", "",
        "24 张截图，每图同目标 click / describe / forbid 三条指令，共 72 条。按截图组隔离：train 16 图 / 48 条，val 4 图 / 12 条，test 4 图 / 12 条。三类在每个 split 配平。本轮是截图组留出，未做措辞家族留出。", "",
        f"模型共 {run['model_runtime']['parameter_count']:,} 个参数，trainable_parameter_count=0；eval 模式、FP16、双 Quadro P5000。读语言层 15 / 28 / 36（从 1 开始）。仅训练标准化 + L2 Logistic Regression；C 和门控阈值都由 val 选择。", "",
        "同一条自由生成轨迹读取 prefill（0 个生成 token）与 post_decision（模型实际生成完整类别词自身）。类别词是普通词表中的 CLICK / DESCRIBE / FORBID，没有新增或训练特殊 token，没有输入真值回答。两条件均有全部 72 条样本。", "",
        "## 三类意图分类", "",
        "| 输入与读取条件 | 注意力特征 | 隐藏状态 | 两者结合 |", "| --- | ---: | ---: | ---: |",
    ]
    for title, summary, condition in [
        ("有图，prefill", visual, "prefill"), ("有图，post_decision", visual, "post_decision"),
        ("无图，prefill", no_image, "prefill"),
    ]:
        results = {row["feature_set"]: row for row in summary["runs"] if row["condition"] == condition}
        lines.append(f"| {title} | " + " | ".join(percent(results[name]["test"]["accuracy"])
                                                  for name in ("attention", "hidden", "combined")) + " |")
    text = next(row for row in visual["runs"] if row["feature_set"] == "text_only")
    lines.extend([
        "", f"指令字符 TF-IDF 基线：{percent(text['test']['accuracy'])}（10/12）。直接解析模型生成类别：{direct_correct}/{len(test)}，即 {percent(direct_correct / len(test))}。",
        f"直接类别词的两个测试错误均为 forbid → describe；它们在是否点击二分类上都属于不点击。因此直接生成类别词的是否点击判断为 {binary_correct}/{len(test)}（类别词可识别 {len(recognized)}/{len(test)}）。不能把三类 10/12 误解为发生了两次错误点击。", "",
        "## 验证集阈值下的点击门控", "",
        "以下使用既定的 1% val 经验误触发预算。val 只有 8 条非点击指令，最小分辨率为 12.5%，所以 1% / 5% / 10% 在本轮对应相同阈值。阈值原样应用 test，没有根据 test 调整。", "",
        "| 条件 | 特征 | 测试误触发 / 非点击数 | 正确点击召回 | 触发比例 |",
        "| --- | --- | ---: | ---: | ---: |",
    ])
    for row in visual["runs"]:
        if row["feature_set"] == "text_only":
            continue
        gate = row["click_gates"][0]["test"]
        lines.append(f"| {row['condition']} | {row['feature_set']} | {gate['false_clicks']}/{gate['n_nonclick']} | {percent(gate['recall'])} | {percent(gate['coverage'])} |")
    lines.extend([
        "", "三类 argmax 分类正确，不保证另行选择的低概率门控阈值也没有误触发；这是不同的决策规则。上述差异只涉及 1 个样本，不能据此排出稳定的优劣次序。", "",
        "无图 attention 特征全零，分类为机会水平，校准后的常数门控拒绝所有点击。这不代表有用的控制能力。", "",
        "## 读取成本与解释", "",
        "有图测试集读取耗时中位数：prefill 约 1.53 秒，post_decision 约 2.28 秒；后者已生成 token 中位数为 3。这些是本机实验采集成本，含 hook、同步和 CPU 拷贝，未测真实桌面动作执行延迟。", "",
        "本轮支持冻结 VLM 的内部状态可以用于读出指令意图；还不能说明视觉注意力理解了当前界面能否执行操作，因为无图 hidden 也为 12/12。三组有图特征都已碰到这个小测试集的分类天花板，组合特征未显示额外准确率收益，生成类别后也未显示三类准确率收益。", "",
        "下一轮需要让标签真正依赖截图状态，例如同一句指令配合目标可见/不存在、启用/禁用、被遮挡/未遮挡的界面，再增加独立截图组与措辞留出。当前 4 个测试截图组不能支撑低误点概率或真实 agent 可用性的结论。", "",
        "## 复现与产物", "",
        "- `scripts/run_pilot.ps1`：同样配置生成数据、提取与评测；已有特征逐样本续用。",
        "- [有图报告](../runs/pilot/evaluation/report.md) / [完整指标](../runs/pilot/evaluation/metrics.json)",
        "- [无图报告](../runs/no_image/evaluation/report.md) / [完整指标](../runs/no_image/evaluation/metrics.json)",
        "- [文本基线](../runs/text_baseline/report.md)",
        "- `runs/environment.json`、`runs/pilot/features/run.json`：版本、GPU、模型身份与冻结状态。",
        "- 原始特征、真实生成文本、概率和小探针模型均已保存在对应 runs 目录。", "",
        "post_response 条件的代码和 token 归属测试已实现，本轮没有运行完整回答后的第三个读取条件。",
    ])
    target = ROOT / "docs" / "PILOT_RESULTS.md"
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
