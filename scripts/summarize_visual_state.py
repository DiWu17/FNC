"""Build a shareable report and static figures from the actual second-round runs."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def pct(value):
    return "—" if value is None else f"{value:.1%}"


def ci_text(value):
    if not value or value.get("ci95") is None:
        return "—"
    return f"{pct(value['point'])} [{pct(value['ci95'][0])}, {pct(value['ci95'][1])}]"


def gate_for(run, budget):
    return next(gate for gate in run["click_gates"] if np.isclose(gate["false_positive_budget"], budget))


def paired_stage_differences(predictions, seed=43, iterations=10000):
    """Conditional paired accuracy difference, resampling complete test groups."""
    with Path(predictions).open(encoding="utf-8-sig", newline="") as stream:
        rows = [row for row in csv.DictReader(stream) if row["split"] == "test"]
    differences = {}
    for feature in ("attention", "hidden", "combined"):
        pre = {row["sample_id"]: row for row in rows if row["condition"] == "prefill" and row["feature_set"] == feature}
        post = {row["sample_id"]: row for row in rows if row["condition"] == "post_decision" and row["feature_set"] == feature}
        if set(pre) != set(post):
            raise ValueError("Stage comparison requires identical test samples")
        grouped = {}
        group_labels = {}
        for sid in pre:
            a, b = pre[sid], post[sid]
            if (a["label"], a["group_id"]) != (b["label"], b["group_id"]):
                raise ValueError("Stage comparison facts differ")
            grouped.setdefault(a["group_id"], []).append(int(b["prediction"] == b["label"]) - int(a["prediction"] == a["label"]))
            group_labels.setdefault(a["group_id"], []).append(a["label"])
        if any(len(labels) != 3 or set(labels) != {"click", "disabled", "absent"} for labels in group_labels.values()):
            raise ValueError("Stage comparison requires complete state triplets")
        group_mean = np.asarray([np.mean(grouped[group]) for group in sorted(grouped)])
        if len(set(map(len, grouped.values()))) != 1:
            raise ValueError("This paired mean requires equal-sized complete groups")
        rng = np.random.default_rng(seed)
        draws = group_mean[rng.integers(0, len(group_mean), size=(iterations, len(group_mean)))].mean(axis=1)
        differences[feature] = {"post_minus_prefill_accuracy": float(group_mean.mean()),
                                "ci95": np.percentile(draws, [2.5, 97.5]).tolist(),
                                "n_groups": len(group_mean), "iterations": iterations, "seed": seed}
    return differences


def audit_comparison_inputs(run_dir, manifest):
    truth = {row["sample_id"]: (row["group_id"], row["label"]) for row in manifest if row["split"] == "test"}
    for subdir, permutation in (("", False), ("no_image", False), ("permutation", True)):
        with (run_dir / subdir / "evaluation/predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
            rows = [row for row in csv.DictReader(stream) if row["split"] == "test" and row["feature_set"] != "text_only"]
        comparisons = {}
        for row in rows:
            key = (row["condition"], row["feature_set"])
            facts = comparisons.setdefault(key, {})
            if row["sample_id"] in facts:
                raise ValueError("Duplicate prediction ID")
            facts[row["sample_id"]] = (row["group_id"], row["label"])
        if not comparisons:
            raise ValueError(f"No numeric test predictions in {subdir}")
        for facts in comparisons.values():
            if set(facts) != set(truth) or any(facts[sid][0] != truth[sid][0] for sid in truth):
                raise ValueError(f"Test sample/group mismatch in {subdir}")
            if not permutation and facts != truth:
                raise ValueError(f"Test label mismatch in {subdir}")


def hidden_state_diagnostics(run_dir, result):
    records = [json.loads(line) for line in (run_dir / "features/metadata.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    with (run_dir / "evaluation/predictions.csv").open(encoding="utf-8-sig", newline="") as stream:
        post = {row["sample_id"]: row for row in csv.DictReader(stream)
                if row["split"] == "test" and row["condition"] == "post_decision" and row["feature_set"] == "hidden"}
    wrong_click = [row for row in records if row["split"] == "test" and row["label"] == "disabled" and row["generated_decision"] == "click"]
    post_run = next(run for run in result["runs"] if run["condition"] == "post_decision" and run["feature_set"] == "hidden")
    threshold = gate_for(post_run, .01)["threshold"]
    return {"test_disabled_generated_as_click": len(wrong_click),
            "post_hidden_recognized_disabled_among_those": sum(post[row["sample_id"]]["prediction"] == "disabled" for row in wrong_click),
            "post_hidden_gate_rejected_among_those": sum(float(post[row["sample_id"]]["p_click"]) < threshold for row in wrong_click),
            "gate_val_false_positive_budget": .01, "gate_threshold": threshold,
            "sample_ids": [row["sample_id"] for row in wrong_click],
            "interpretation": "These samples share the generated decision CLICK, so state separation cannot be explained solely by reading that class word."}


def make_plot(main, no_image, permutation, path):
    # Keep Matplotlib's cache inside this workspace.
    os.environ.setdefault("MPLCONFIGDIR", str(path.parent.resolve() / ".mplconfig"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    features = ("attention", "hidden", "combined")
    sources = [("With image: prefill", main, "prefill", "#3182bd"),
               ("With image: post decision", main, "post_decision", "#31a354"),
               ("No image: prefill", no_image, "prefill", "#969696"),
               ("Random labels: prefill", permutation, "prefill", "#e69f00")]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), layout="constrained")
    xs = np.arange(3)
    for index, (name, source, condition, color) in enumerate(sources):
        selected = [next(run for run in source["runs"] if run["condition"] == condition and run["feature_set"] == feature) for feature in features]
        values = np.asarray([run["test"]["accuracy"] for run in selected])
        intervals = [run["group_bootstrap"]["accuracy"]["ci95"] for run in selected]
        error = np.asarray([[value - interval[0], interval[1] - value] for value, interval in zip(values, intervals)]).T
        axes[0].bar(xs + (index - 1.5) * .19, values, .18, label=name, color=color,
                    yerr=np.maximum(error, 0), capsize=2)
    axes[0].axhline(1 / 3, color="black", ls="--", lw=1, label="3-class chance")
    axes[0].set_xticks(xs, features)
    axes[0].set_ylim(0, 1.10)
    axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[0].set_title("Test state classification (group-bootstrap 95% CI)")
    axes[0].legend(fontsize=8, loc="upper left")
    markers = {"attention": "o", "hidden": "s", "combined": "^"}
    plotted_false_rates = []
    for condition, color in (("prefill", "#3182bd"), ("post_decision", "#31a354")):
        for feature in features:
            run = next(run for run in main["runs"] if run["condition"] == condition and run["feature_set"] == feature)
            gates = [gate_for(run, budget)["test"] for budget in (.01, .05, .10)]
            plotted_false_rates.extend(gate["false_click_rate"] for gate in gates)
            axes[1].plot([gate["false_click_rate"] for gate in gates], [gate["recall"] for gate in gates],
                         marker=markers[feature], color=color, alpha=.8, label=f"{condition}: {feature}")
    upper_false_rate = min(1.02, max(.10, max(plotted_false_rates) + .025))
    axes[1].set(xlim=(-.005, upper_false_rate), ylim=(-.02, 1.06), xlabel="Test false click rate (disabled + absent)",
                ylabel="Test recall for enabled Save", title="Val-only gates: 1%, 5%, 10% budgets")
    axes[1].xaxis.set_major_formatter(PercentFormatter(1))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(fontsize=8, loc="lower right")
    test_groups = main["runs"][0]["group_bootstrap"]["n_groups"]
    fig.suptitle(f"Frozen Qwen3-VL-8B | fixed instruction | {test_groups} held-out scenario triplets", fontsize=13)
    fig.savefig(path, dpi=180)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/visual_state_v2_matched"))
    parser.add_argument("--dataset", type=Path, default=Path("data/visual_state_v2_matched"))
    parser.add_argument("--output", type=Path, default=Path("docs/VISUAL_STATE_RESULTS.md"))
    args = parser.parse_args()
    result = read_json(args.run / "evaluation/metrics.json")
    no_image = read_json(args.run / "no_image/evaluation/metrics.json")
    permutation = read_json(args.run / "permutation/evaluation/metrics.json")
    data = read_json(args.dataset / "summary.json")
    runtime = read_json(args.run / "features/run.json")
    no_image_runtime = read_json(args.run / "no_image/features/run.json")
    constant_control = no_image_runtime["constant_input_control"]
    manifest_rows = [json.loads(line) for line in (args.dataset / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    audit_comparison_inputs(args.run, manifest_rows)
    splits = data["splits"]
    test_counts = splits["test"]["labels"]
    n_disabled, n_absent = test_counts["disabled"], test_counts["absent"]
    n_nonclick = n_disabled + n_absent
    n_val_negative = splits["val"]["labels"]["disabled"] + splits["val"]["labels"]["absent"]
    split_groups = " / ".join(str(splits[split]["num_groups"]) for split in ("train", "val", "test"))
    split_rows = " / ".join(str(splits[split]["num_samples"]) for split in ("train", "val", "test"))
    domain_counts = data["scenario_counts_by_split_domain"]["test"]
    differences = paired_stage_differences(args.run / "evaluation/predictions.csv")
    diagnostics = hidden_state_diagnostics(args.run, result)
    (args.run / "hidden_state_diagnostics.json").write_text(json.dumps(diagnostics, indent=2) + "\n", encoding="utf-8")
    (args.run / "paired_stage_differences.json").write_text(json.dumps(differences, indent=2) + "\n", encoding="utf-8")
    figure = args.run / "visual_state_results.png"
    make_plot(result, no_image, permutation, figure)
    lines = ["# 第二轮：固定指令的视觉状态门控", "",
             f"实际运行冻结 Qwen3-VL-8B-Instruct。{data['num_groups']} 个基础场景、{data['num_samples']} 张合成截图，用户指令全部为 **点击保存**。每组三张图只有目标槽位不同：保存可用、保存禁用、同外观非目标按钮替代保存。三张图均有六个按钮；标签及框只供审计，不输入模型。", "",
             f"train / val / test 为 {split_groups} 个完整场景组（{split_rows} 张）。测试有 {domain_counts['iid']} 个 IID 与 {domain_counts['ood']} 个 OOD 组；OOD 只留出布局，仍共享渲染器、字体和禁用外观。实验预定方法见 [协议](VISUAL_STATE_PROTOCOL.md)。", "",
             f"模型冻结：`trainable_parameter_count={runtime['model_runtime']['trainable_parameter_count']}`；FP16，语言层 15 / 28 / 36。完整两条件共同覆盖 {result['coverage']['n_common']}/{data['num_samples']} 条。post_decision 使用模型实际生成词，UNKNOWN 和错误预测均保留。", "",
             "## 分类与反事实配对", "",
             "| 读取条件 | 特征 | 测试 accuracy [组级95%CI] | Macro-F1 | Click AUROC | IID accuracy | OOD accuracy | 三联全对 | enabled分数同时高于两负态 |",
             "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for run in result["runs"]:
        domains = run["test_slices"]["axes"]["domain"]
        paired = run["paired_visual_state"]
        lines.append(f"| {run['condition']} | {run['feature_set']} | {ci_text(run['group_bootstrap']['accuracy'])} | {run['test']['macro_f1']:.4f} | {run['test']['click_auroc']:.4f} | {pct(domains['iid']['classification']['accuracy'])} | {pct(domains['ood']['classification']['accuracy'])} | {pct(paired['triplet_all_correct_rate'])} | {pct(paired['paired_flip']['both_positive_rate'])} |")
    hidden_pre = next(run for run in result["runs"] if run["condition"] == "prefill" and run["feature_set"] == "hidden")
    hidden_post = next(run for run in result["runs"] if run["condition"] == "post_decision" and run["feature_set"] == "hidden")
    pre_gate = gate_for(hidden_pre, .01)["test"]
    state_errors = "、".join(f"{state} {item['false_clicks']} 个" for state, item in pre_gate["false_click_rate_by_negative_state"].items())
    domain_errors = "、".join(f"{domain} {next(g for g in entry['click_gates'] if np.isclose(g['false_positive_budget'], .01))['false_clicks']} 个" for domain, entry in hidden_pre["test_slices"]["axes"]["domain"].items())
    lines += ["", "## 本轮最有解释力的观察", "",
              f"隐藏状态在生成前的三类正确率为 {pct(hidden_pre['test']['accuracy'])}，生成后为 {pct(hidden_post['test']['accuracy'])}；去图prefill对照的三组特征均为机会水平。两种读取时机在全部 {hidden_pre['paired_visual_state']['n_complete_triplets']} 个测试三联中，可用按钮的点击分数都高于同场景的禁用和不存在状态（两差均为正的比例：{pct(hidden_pre['paired_visual_state']['paired_flip']['both_positive_rate'])} / {pct(hidden_post['paired_visual_state']['paired_flip']['both_positive_rate'])}）。这比第一轮更支持冻结模型内部有可读的视觉状态信息，而非只区分指令措辞。", "",
              f"在模型原生输出把禁用按钮说成 CLICK 的 {diagnostics['test_disabled_generated_as_click']} 个测试错例中，post hidden 探针仍正确识别禁用 {diagnostics['post_hidden_recognized_disabled_among_those']} 个；验证集1%预算门控拒绝其中 {diagnostics['post_hidden_gate_rejected_among_those']} 个。这些样本已经生成相同的 CLICK，所以后读门控的效果不能仅由读回类别词解释。这里仍是当前合成数据和统一提示下的结果。", "",
              f"监督预算不同：探针使用 {splits['train']['num_samples']} 张标注训练图拟合，并用 {splits['val']['num_samples']} 张验证图选择C和阈值；直接类别输出是冻结模型在统一提示下的zero-shot决策。因此结果支持可学习读出的可用性，不是同等监督预算下击败原生解码，也不能等价为模型早已语义理解全部可执行条件。", "",
              f"场景内分数排序正确，不等于统一阈值跨场景可靠。生成前 hidden 在验证集1%预算下测试有 {pre_gate['false_clicks']} 个误放行，按负状态为 {state_errors}，按域为 {domain_errors}；整体三类高分并未消除分数校准和布局迁移问题。注意力与隐藏状态直接拼接也未稳定超过hidden单独使用，不能假定特征更多就会更好。", "",
              "## 验证集校准后，原样应用测试集的点击门控", "",
              f"验证集 {n_val_negative} 条非点击样本，1% / 5% / 10% 经验预算分别允许最多 " + " / ".join(str(int(np.floor(n_val_negative * budget))) for budget in (.01, .05, .1)) + " 个误放行。预算不等于测试集风险保证；下表全部报告，没有根据 test 选阈值。", "",
              f"| 条件 | 特征 | Val预算 | 正确点击召回 | 禁用误放行/{n_disabled} | 不存在误放行/{n_absent} | 总误放行/{n_nonclick} | 三联完整门控正确率 |",
              "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for run in result["runs"]:
        if run["feature_set"] == "text_only":
            continue
        for gate in run["click_gates"]:
            point = gate["test"]
            states = point["false_click_rate_by_negative_state"]
            paired_gate = next(item for item in run["paired_visual_state"]["click_gates"] if item["false_positive_budget"] == gate["false_positive_budget"])
            lines.append(f"| {run['condition']} | {run['feature_set']} | {pct(gate['false_positive_budget'])} | {pct(point['recall'])} | {states['disabled']['false_clicks']}/{n_disabled} | {states['absent']['false_clicks']}/{n_absent} | {point['false_clicks']}/{n_nonclick} | {pct(paired_gate['complete_triplet_correct_gate_rate'])} |")
    lines += ["", "总体FPR将disabled和absent合在一起，依赖这两种负态的比例。post hidden在1%验证预算下的1/120总体误放行，实际全部来自disabled，即禁用状态为1/60；不能把总体0.83%写成已对禁用按钮证明1%风险。"]
    lines += ["", "## 去图和随机标签对照", "", "| 对照 | 条件 | 特征 | 测试accuracy | Click AUROC |", "| --- | --- | --- | ---: | ---: |"]
    for name, source in (("无图", no_image), ("组内均匀随机标签", permutation)):
        for run in source["runs"]:
            lines.append(f"| {name} | {run['condition']} | {run['feature_set']} | {pct(run['test']['accuracy'])} | {run['test']['click_auroc']:.4f} |")
    lines += ["", f"去图后独立模型输入数为 {constant_control['unique_input_count']}；实际执行 {constant_control['actual_forward_count']} 次冻结模型前向，确认处理后的输入张量hash和特征bytes完全相同，再将真实特征复用于其余 {constant_control['feature_reused_rows']} 行。只有实际核验行记录了实测时延。这是常数输入的数值对照，完整清单的行数不能解释为同等数量的独立模型观测。"]
    direct = result["extraction_audit"]["direct_generation"]["paired_test"]
    direct_gate = direct["click_gate"]
    lines += ["", "## 直接生成与读取成本", "",
              f"直接解析模型自己的类别词：{direct['correct_count']}/{direct['n_evaluated']} 三类正确（{pct(direct['accuracy_including_missing'])}），未识别/UNKNOWN {direct['no_decision_count']} 条。只把明确 CLICK 当作点击时，召回 {pct(direct_gate['recall'])}，误放行 {direct_gate['false_clicks']}/{direct_gate['n_nonclick']}。后读探针须与该基线比较，不能把读回类别词当成额外的视觉理解。", "",
              "| 特征 | post − prefill accuracy | 配对组bootstrap95%CI |", "| --- | ---: | --- |"]
    for feature, difference in differences.items():
        lines.append(f"| {feature} | {difference['post_minus_prefill_accuracy']:+.1%} | [{difference['ci95'][0]:+.1%}, {difference['ci95'][1]:+.1%}] |")
    lines += ["", "上述差值使用固定探针，对相同测试场景组配对重采样 10000 次；没有重新训练或重新校准阈值。", "",
              "| 条件 | 测试读取秒数p50 | p95 | 已生成token p50 |", "| --- | ---: | ---: | ---: |"]
    for condition, detail in result["extraction_audit"]["readout_latency"].items():
        subset = detail["subsets"]["paired_test"]
        latency, tokens = subset["readout_elapsed_seconds"], subset["generated_tokens_at_readout"]
        lines.append(f"| {condition} | {latency['p50']:.3f} | {latency['p95']:.3f} | {tokens['p50']:.1f} |")
    lines += ["", "读取成本包含 hook、CUDA 同步及 CPU 拷贝；prefill 仍需完整视觉编码和语言前向，不能当作真实桌面执行延迟。", "",
              "## 可以怎样解释", "",
              "只有在有图分数明显高于去图和随机标签对照，且同场景状态变化能使点击分数正确变化时，才支持探针利用了与截图状态有关的信号。它仍可能利用共享绘制器的字体、对比度或禁用色板；布局留出不能排除这些捷径。", "",
              "这轮测试的是离线的是否放行点击，未验证点击位置、真实应用行为或多步任务成功。attention 使用的是统计摘要，低分不能否定完整注意力张量中存在信息；hidden/combined 的高分也不意味着原生热图本身就足以当门控。", "",
              f"测试只有 {splits['test']['num_groups']} 个独立场景；零错误的经验bootstrap区间可能为零宽，不能证明未来误点概率为零，也不能证明满足1%风险。一次随机标签对照不是显著性检验。", "",
              "## 复现与产物", "", "```powershell", ".\\scripts\\run_visual_state.ps1", ".\\.venv\\python.exe scripts/summarize_visual_state.py", "```", "",
              f"- 数据 manifest SHA256：`{data['manifest_sha256']}`。",
              f"- 提取 fingerprint：`{runtime['fingerprint']}`。",
              f"- 图表：[{figure.name}](../{figure.as_posix()})；原始分数、分域指标、分组区间、真实生成文本、小探针和逐样本概率保存在 `{args.run.as_posix()}`。",
              "- `evaluation/report.md` 是完整报告；`no_image/evaluation` 与 `permutation/evaluation` 分别为两个对照。",
              "- 初版空白槽位数据及17条中断缓存仅供审计，不进入上述正式结果。", ""]
    integrity = args.dataset / "image_integrity.json"
    if integrity.exists():
        lines.insert(-1, f"- 原始PNG完整性：{read_json(integrity)['image_count']} 张，聚合SHA256 `{read_json(integrity)['aggregate_sha256']}`；逐文件hash见数据目录的image_integrity.json。")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
