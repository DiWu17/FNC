"""Render the completed sealed v7 experiment; never fit or select on test.

Only report.html, report.md, report_integrity.json and assets/ are written.
Missing results, partial coverage, failed audits or fingerprint drift are errors,
not opportunities to fill in provisional numbers. All figures are CPU-only.
"""
from __future__ import annotations

import argparse
import base64
from collections import Counter, defaultdict
import hashlib
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


RULES = ("region", "point", "quality")
MODES = {"region": ("rank",), "point": ("rank", "combined", "quality"), "quality": ("combined",)}
BASELINES = ("v5_fixed", "v6_s1_fixed", "v6_s2_fixed", "phrase_local", "qwen_native_completed_lax", "ui_tars")
FAMILIES = ("flat_base", "static_base", "semantic_base", "semantic_expanded", "semantic_hard")
FAMILY_LABELS = {"flat_base": "Flat / base", "static_base": "Early static / base", "semantic_base": "Early semantic / base",
                 "semantic_expanded": "Early semantic / expanded", "semantic_hard": "Early semantic / grouped hard"}
RULE_LABELS = {"region": "区域优先", "point": "冻结点命中优先", "quality": "质量合成点命中优先"}
FACTOR_LABELS = {"point checkpoint selection": "实际点命中选模", "explicit early semantic condition": "显式逐头语义条件",
    "early versus flat architecture": "逐头条件化与原扁平结构", "more real multi-target data": "增加真实同图多目标数据",
    "same data: grouped hard training": "同数据的分组难例训练", "same checkpoint: point-quality score": "同一检查点加入点质量分数",
    "combined recipe": "组合方案", "combined versus previous v6": "组合方案对旧 v6",
    "combined versus Qwen native": "组合方案对原生 Qwen", "combined versus UI-TARS": "组合方案对 UI-TARS"}
STAGES = ("hit", "fixed_fine_miss_after_intersecting_parent", "ranking_miss_with_target_candidate",
          "no_target_peak_candidate", "no_name")
STAGE_LABELS = ("命中", "相交父格后的冻结点错误", "有目标候选但排序错误", "候选池无相交目标", "未生成有效名称")
COLORS = ("#2a9d8f", "#e9aa51", "#457b9d", "#dc6c62", "#a9adb5")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def rows(path):
    with Path(path).open(encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def method(job, rule="point", mode="rank"):
    return job + "__" + rule + "__" + mode


def chosen_mode(rule):
    return "combined" if rule == "quality" else "rank"


def family(job_name):
    return job_name.rsplit("_s", 1)[0]


def short(job_name):
    return FAMILY_LABELS.get(family(job_name), family(job_name)) + " / S" + job_name.rsplit("_s", 1)[-1]


def expected_methods(protocol):
    result = set(BASELINES) | {"v6_s1_ownfine", "v6_s2_ownfine"}
    for job in protocol["jobs"]:
        for rule in RULES:
            for mode in MODES[rule]:
                name = method(job["name"], rule, mode)
                result.update((name, name + "_ownfine", name + "_center"))
    return result


def load_bundle(out):
    """Fail closed before creating any output directories or provisional plots."""
    out = Path(out)
    protocol, audit, frozen = read(out / "protocol.json"), read(out / "final_audit.json"), read(out / "frozen_readouts.json")
    require(audit.get("complete") is True, "Final audit is incomplete; refusing to report")
    require(len(protocol["jobs"]) == 10 and len({j["name"] for j in protocol["jobs"]}) == 10, "Expected ten distinct training trajectories")
    require(len(frozen["models"]) == 30, "Expected all thirty validation-selected checkpoints")
    require(frozen["primary_model"] == protocol["primary"], "Primary must be fixed by the sealed protocol")
    require(audit["frozen_sha256"] == sha(out / "frozen_readouts.json"), "Frozen model index changed after audit")
    require(audit["protocol_sha256"] == sha(out / "protocol.json"), "Protocol changed after audit")
    require(audit["manifest_sha256"] == protocol["manifest_sha256"] == sha(out / "manifest.jsonl"), "Manifest changed after audit")
    data = read(out / "data_selection.json")
    require(protocol["data_selection_sha256"] == sha(out / "data_selection.json"), "Data selection changed")
    require(data["status"].startswith("ready"), "Data selection is not ready")
    cases = rows(out / "manifest.jsonl")
    require(len({c["sample_id"] for c in cases}) == len(cases), "Manifest has duplicate sample IDs")
    histories = {}
    for job in protocol["jobs"]:
        name = job["name"]
        history = read(out / "models" / f"{name}_history.json")
        require(history.get("complete") is True and history["job"] == job, f"Incomplete/mismatched trajectory: {name}")
        require([r["epoch"] for r in history["history"]] == list(range(protocol["epochs"] + 1)), f"Partial epoch coverage: {name}")
        for rule in RULES:
            checkpoint = history["checkpoints"][rule]
            sealed = frozen["models"][name + "__" + rule]
            require(checkpoint["sha256"] == sealed["sha256"] == sha(out / "models" / checkpoint["file"]), f"Checkpoint changed: {name}/{rule}")
            require(checkpoint["epoch"] == sealed["epoch"], f"Selected epoch changed: {name}/{rule}")
        histories[name] = history
    selection_audit = read(out / "data_selection_audit.json") if (out / "data_selection_audit.json").exists() else None
    if selection_audit is not None:
        require(selection_audit.get("status") == "pass" and all(selection_audit["checks"].values()), "Independent data-selection audit failed")
    summaries, predictions, lookup = {}, {}, {}
    methods = expected_methods(protocol)
    primary = protocol["primary"] + "__combined"
    for split, nkey in (("test", "fresh_test_queries"), ("diagnostic", "diagnostic_queries")):
        path = out / "evaluation" / split
        summary, predictions_split = read(path / "summary.json"), rows(path / "predictions.jsonl")
        subset = [c for c in cases if c["experiment_split"] == split]
        ids = {c["sample_id"] for c in subset}
        require(summary["n"] == len(ids) == protocol[nkey], f"Wrong {split} denominator")
        require(summary["primary"] == primary, f"Test-dependent primary in {split}")
        require(summary["test_is_fresh"] is (split == "test"), f"Incorrect cohort interpretation: {split}")
        require(summary["manifest_sha256"] == protocol["manifest_sha256"], f"Manifest mismatch: {split}")
        require(summary["frozen_sha256"] == audit["frozen_sha256"], f"Frozen index mismatch: {split}")
        require(summary["predictions_sha256"] == sha(path / "predictions.jsonl"), f"Predictions changed: {split}")
        require(summary["predictions_sha256"] == audit["scores"][split]["prediction_sha256"], f"Predictions differ from audited {split}")
        require(set(summary["methods"]) == methods, f"Missing or unexpected methods: {split}")
        by = {}
        grouped = defaultdict(list)
        for row in predictions_split:
            cell = row["method"], row["sample_id"]
            require(cell not in by and row["sample_id"] in ids and row["method"] in methods, f"Duplicate/unknown prediction: {cell}")
            require(isinstance(row["hit"], bool), f"Prediction hit must be boolean: {cell}")
            by[cell] = row
            grouped[row["method"]].append(row)
        require(len(predictions_split) == summary["prediction_count"] == len(ids) * len(methods), f"Incomplete predictions: {split}")
        for name in methods:
            group, metric = grouped[name], summary["methods"][name]
            require({r["sample_id"] for r in group} == ids, f"Incomplete case coverage: {split}/{name}")
            require(metric["n"] == len(ids) and metric["hits"] == sum(r["hit"] for r in group), f"Summary score mismatch: {split}/{name}")
            require(metric["points"] == sum(r["point"] is not None for r in group), f"Point coverage mismatch: {split}/{name}")
        require(len(summary["paired_comparisons"]) == 20, f"Expected all prespecified paired comparisons: {split}")
        for comparison in summary["paired_comparisons"]:
            first, second = comparison["first"], comparison["second"]
            require(first in methods and second in methods, f"Unknown paired comparison: {split}")
            low, high = comparison["ci95_pp"]
            require(np.isfinite([low, high, comparison["delta_pp"]]).all() and low <= high, f"Invalid paired interval: {split}")
            wins = sum(by[first,sid]["hit"] and not by[second,sid]["hit"] for sid in ids)
            losses = sum(by[second,sid]["hit"] and not by[first,sid]["hit"] for sid in ids)
            require((wins, losses) == (comparison["wins"], comparison["losses"]), f"Paired win/loss mismatch: {split}")
            require(abs(comparison["delta_pp"] - 100*(wins-losses)/len(ids)) < 1e-8, f"Paired delta mismatch: {split}")
        summaries[split], predictions[split], lookup[split] = summary, predictions_split, by
    require([(c["factor"],c["first"],c["second"]) for c in summaries["test"]["paired_comparisons"]] ==
            [(c["factor"],c["first"],c["second"]) for c in summaries["diagnostic"]["paired_comparisons"]], "Cohorts have different comparison plans")
    return dict(out=out, protocol=protocol, audit=audit, frozen=frozen, data=data, cases=cases,
                histories=histories, summaries=summaries, predictions=predictions, lookup=lookup, primary=primary,
                selection_audit=selection_audit)


def hit_text(metric):
    return f"{metric['hits']}/{metric['n']} ({metric['hits']/metric['n']:.1%})"


def count_text(value, n):
    return "不可评估" if value is None else f"{value}/{n}"


def failure_summary(predictions, name, dataset=None):
    selected = [r for r in predictions if r["method"] == name and (dataset is None or r["dataset"] == dataset)]
    require(bool(selected), f"No failure records for {name}/{dataset}")
    require(all(r.get("failure_stage") in STAGES for r in selected), f"Incomplete failure stages: {name}/{dataset}")
    stages = Counter(r["failure_stage"] for r in selected)
    require(stages["hit"] == sum(r["hit"] for r in selected), f"Failure hit labels disagree: {name}/{dataset}")
    require(all(r["point"] is not None or r["failure_stage"] == "no_name" for r in selected), f"Missing point not marked no_name: {name}/{dataset}")
    return dict(n=len(selected), hits=sum(r["hit"] for r in selected), stages=stages,
                selected_parent_overlap=sum(bool(r.get("selected_region_overlap", False)) for r in selected),
                parent_oracle=sum(bool(r.get("candidate_region_oracle", False)) for r in selected),
                frozen_point_oracle=sum(bool(r.get("frozen_fine_point_oracle", False)) for r in selected))


def checkpoint_table(bundle, split):
    summary, histories = bundle["summaries"][split], bundle["histories"]
    result = []
    for job in bundle["protocol"]["jobs"]:
        h = histories[job["name"]]
        for rule in RULES:
            selected, mode = h["checkpoints"][rule], chosen_mode(rule)
            name = method(job["name"], rule, mode)
            metric, val = summary["methods"][name], selected["validation"]
            train = selected["training"]
            result.append([short(job["name"]), str(job["seed"]), RULE_LABELS[rule], str(selected["epoch"]),
                f"{train['modes'][mode]['hits']}/{train['n']}", f"{val['modes'][mode]['hits']}/{val['n']}",
                hit_text(metric), count_text(metric.get("region_top1"), metric["n"])])
    return result


def table_html(headers, body):
    esc = lambda value: html.escape(str(value))
    return '<div class="table"><table><thead><tr>' + ''.join('<th>' + esc(v) + '</th>' for v in headers) + '</tr></thead><tbody>' + ''.join(
        '<tr>' + ''.join('<td>' + esc(v) + '</td>' for v in row) + '</tr>' for row in body) + '</tbody></table></div>'


def table_md(headers, body):
    esc = lambda value: str(value).replace('|', '\\|').replace('\n', ' ')
    return '\n'.join(['| ' + ' | '.join(map(esc, headers)) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(esc, row)) + ' |' for row in body])


def _save(fig, assets, name):
    paths = []
    for suffix in ("png", "pdf"):
        path = assets / f"{name}.{suffix}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def plot_results(bundle, assets):
    p, m = bundle["protocol"], bundle["summaries"]["test"]["methods"]
    paths = []
    n = bundle["summaries"]["test"]["n"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.5, 1]})
    positions = np.arange(len(FAMILIES))
    for i, seed in enumerate(("s1", "s2")):
        values = [m[method(f + "_" + seed)]["hits"] for f in FAMILIES]
        bars = axes[0].bar(positions + (i-.5)*.34, np.array(values)*100/n, .34, label=seed.upper(), color=("#258b83", "#7aaecb")[i])
        for bar, value in zip(bars, values):
            axes[0].annotate(str(value), (bar.get_x()+bar.get_width()/2, bar.get_height()), ha="center", xytext=(0, 3), textcoords="offset points", fontsize=8)
    for name, label, color in (("qwen_native_completed_lax", "Native Qwen", "#ad5e58"), ("ui_tars", "UI-TARS", "#8b7c5c")):
        axes[0].axhline(100*m[name]["hits"]/n, color=color, ls="--", label=label, lw=1.2)
    axes[0].set_xticks(positions, [FAMILY_LABELS[f].replace(" / ", "\n") for f in FAMILIES], fontsize=8)
    axes[0].set(ylabel="Point hits (%)", ylim=(0, 100), title="Fresh test: point-selected checkpoint, rank score")
    axes[0].legend(fontsize=8, loc="upper left")
    names = [bundle["primary"], bundle["primary"].replace("_s1__", "_s2__"), "v5_fixed", "v6_s1_fixed", "v6_s2_fixed", "phrase_local", "qwen_native_completed_lax", "ui_tars"]
    labels = ["Prespecified primary S1", "Same recipe S2", "Previous v5 / fixed", "Previous v6 S1 / fixed", "Previous v6 S2 / fixed", "Phrase local", "Native Qwen", "UI-TARS"]
    values = [m[name]["hits"] for name in names]
    bars = axes[1].barh(labels, np.array(values)*100/n, color=["#258b83", "#7aaecb"]+["#a6afb9"]*6)
    for bar, value in zip(bars, values):
        axes[1].text(min(96, bar.get_width()+1), bar.get_y()+bar.get_height()/2, f"{value}/{n}", va="center", fontsize=8)
    axes[1].invert_yaxis(); axes[1].set(xlim=(0, 100), xlabel="Point hits (%)", title="Same images; fixed primary, no test winner selection")
    fig.tight_layout(); paths += _save(fig, assets, "fresh_scores")
    fig, axes = plt.subplots(1, 2, figsize=(13, 7), sharey=True)
    jobs = p["jobs"]
    for ax, conditions, title in (
        (axes[0], [("region", "rank", "Region selection", "#8b9aae"), ("point", "rank", "Point selection", "#258b83"), ("quality", "combined", "Quality+point selection", "#d28a45")], "Three validation selectors / same training trajectory"),
        (axes[1], [("point", "rank", "Rank", "#258b83"), ("point", "combined", "Rank + log quality", "#d28a45"), ("point", "quality", "Quality only", "#7176ad")], "Same point-selected checkpoint / score ablation")):
        per_condition = [[100*m[method(j["name"], rule, mode)]["hits"]/n for j in jobs] for rule,mode,_,_ in conditions]
        for row_index in range(len(jobs)):
            values = [v[row_index] for v in per_condition]
            ax.plot([min(values),max(values)],[row_index,row_index],color="#b6bdc4",lw=1)
        for values, (_, _, label, color), marker, size in zip(per_condition, conditions, ("s","o","x"), (45,26,40)):
            ax.scatter(values, range(len(jobs)), marker=marker, s=size, label=label, color=color,
                       facecolors="none" if marker=="s" else color, linewidths=1.1)
        ax.set(title=title, xlabel="Fresh test point hits (%)", xlim=(0, 100))
        ax.grid(axis="x", alpha=.2); ax.legend(fontsize=8, loc="lower right")
    axes[0].set_yticks(range(len(jobs)), [short(j["name"]) for j in jobs], fontsize=8)
    axes[0].invert_yaxis(); fig.tight_layout(); paths += _save(fig, assets, "selectors_and_quality")
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for row, suffix in enumerate(("s1", "s2")):
        for f, color in zip(FAMILIES, plt.get_cmap("tab10").colors):
            job = f + "_" + suffix; h = bundle["histories"][job]
            for col, (mode, metric, rule, title) in enumerate((("rank", "region", "region", "Parent overlap"), ("rank", "hits", "point", "Frozen point / rank"), ("combined", "hits", "quality", "Frozen point / combined"))):
                ax = axes[row, col]; xs = [r["epoch"] for r in h["history"]]
                ys = [r["val"]["modes"][mode][metric] for r in h["history"]]
                ax.plot(xs, ys, label=FAMILY_LABELS[f], color=color, lw=1.2)
                epoch = h["checkpoints"][rule]["epoch"]
                ax.scatter([epoch], [ys[xs.index(epoch)]], color=color, s=28, marker="s" if col == 0 else "o" if col == 1 else "*")
                ax.set(title=f"{suffix.upper()}: {title}", ylim=(0, p["validation_queries"]), xlabel="Epoch", ylabel="Validation hits")
                ax.grid(alpha=.15)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=8, bbox_to_anchor=(.5, .005))
    fig.tight_layout(rect=(0,.075,1,1)); paths += _save(fig, assets, "validation_curves")
    names = [method("flat_base_"+s) for s in ("s1", "s2")] + [bundle["primary"], bundle["primary"].replace("_s1__", "_s2__"), "v6_s1_fixed", "v6_s2_fixed"]
    fig, ax = plt.subplots(figsize=(12, 5.5)); left = np.zeros(len(names))
    for stage, label, color in zip(STAGES, ("Hit", "Fixed-point miss after target parent", "Ranking miss", "No target candidate", "No valid name"), COLORS):
        values = [sum(r.get("failure_stage") == stage for r in bundle["predictions"]["test"] if r["method"] == name) for name in names]
        ax.barh(range(len(names)), np.array(values)*100/n, left=left*100/n, label=label, color=color)
        left += values
    require(np.all(left == n), "Failure decomposition must cover all fresh test cases")
    ax.set_yticks(range(len(names)), ["Flat S1 / point", "Flat S2 / point", "Primary S1", "Same recipe S2", "Previous v6 S1", "Previous v6 S2"])
    ax.invert_yaxis(); ax.set(xlim=(0, 100), xlabel="Fresh test cases (%)", title="Shared frozen fine points: complete failure decomposition")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.14), ncol=3, fontsize=8); fig.tight_layout()
    paths += _save(fig, assets, "failure_stages")
    primary_names=[bundle["primary"],bundle["primary"].replace("_s1__","_s2__")]
    slices=[(name,ds,failure_summary(bundle["predictions"]["test"],name,ds))
            for name in primary_names for ds in sorted(m[name]["datasets"])]
    fig,ax=plt.subplots(figsize=(12,max(4, .65*len(slices)+1.7)));left=np.zeros(len(slices))
    for stage,label,color in zip(STAGES,("Hit","Fixed-point miss after target parent","Ranking miss","No target candidate","No valid name"),COLORS):
        values=np.array([entry[2]["stages"][stage]/entry[2]["n"]*100 for entry in slices])
        ax.barh(range(len(slices)),values,left=left,label=label,color=color);left+=values
    ax.set_yticks(range(len(slices)),[ds+(" / S1" if "_s1__" in name else " / S2") for name,ds,_ in slices])
    ax.invert_yaxis();ax.set(xlim=(0,100),xlabel="Cases within each dataset (%)",title="Prespecified recipe: fresh-test failures by dataset")
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.14),ncol=3,fontsize=8);fig.tight_layout()
    paths+=_save(fig,assets,"failure_by_dataset")
    fig, axes = plt.subplots(1, 2, figsize=(13, 10), sharey=True)
    for ax, split, title in zip(axes, ("test", "diagnostic"), ("Fresh independent test", "Previously used diagnostic set")):
        comparisons = bundle["summaries"][split]["paired_comparisons"]
        for i, c in enumerate(comparisons):
            lo, hi = c["ci95_pp"]; ax.plot([lo, hi], [i, i], color="#7c8d9c", lw=1.5)
            ax.scatter(c["delta_pp"], i, color="#258b83" if "_s1__" in c["first"] else "#b57d40", s=25)
        ax.axvline(0, color="#555", ls="--", lw=1); ax.grid(axis="x", alpha=.15)
        ax.set(title=title, xlabel="Paired difference (percentage points)")
    labels = [c["factor"] + (" / S1" if "_s1__" in c["first"] else " / S2") for c in bundle["summaries"]["test"]["paired_comparisons"]]
    axes[0].set_yticks(range(len(labels)), labels, fontsize=8); axes[0].invert_yaxis()
    fig.tight_layout(); paths += _save(fig, assets, "paired_intervals")
    return paths


def plot_structure(assets):
    fig, ax = plt.subplots(figsize=(14, 5.2)); ax.set_axis_off()
    def box(x, y, text, width=.19, color="#e5f2ef"):
        ax.text(x, y, text, ha="center", va="center", fontsize=9, bbox=dict(boxstyle="round,pad=.6", facecolor=color, edgecolor="#81949b"), transform=ax.transAxes)
    def arrow(x0, y0, x1, y1):
        ax.annotate("", xy=(x1,y1), xytext=(x0,y0), xycoords="axes fraction", arrowprops=dict(arrowstyle="->", color="#53646d"))
    box(.12,.82,"Frozen Qwen cache\nEvery candidate / token / head\n6 attention channels")
    box(.39,.82,"Per-head compatibility\nQuery + phrase + image\nHead identity + mass / entropy")
    box(.68,.82,"4 soft head pools\nthen 4 soft token pools")
    box(.90,.82,"Base candidate score\n+ routed residual\n= rank score")
    arrow(.24,.82,.27,.82); arrow(.52,.82,.56,.82); arrow(.80,.82,.82,.82)
    box(.12,.46,"Unchanged base branch\nOriginal semantic ranker\nOwn fine loss is auxiliary",color="#e7edf5")
    box(.49,.46,"Detached point-quality branch\nQuery / image / rank scores\n13 frozen-point features",color="#fff0d9")
    box(.82,.46,"Rank score + log sigmoid(quality)\nSelect candidate\nUse SAME frozen v5 fine point",color="#fff0d9")
    arrow(.23,.46,.32,.46); arrow(.66,.46,.70,.46); arrow(.90,.73,.90,.55)
    box(.30,.13,"STATIC control: zero explicit query / phrase\nonly inside the new router; same parameters",color="#f0eafa")
    box(.75,.13,"Validation selects region / point / combined epochs\nPrimary fixed before fresh-test extraction",color="#e7edf5")
    arrow(.82,.36,.78,.22)
    fig.tight_layout(); return _save(fig, assets, "implementation")


def select_examples(bundle, limit=9):
    """Deterministic, explicitly post-hoc explanatory cases; no model changes."""
    require(0 <= limit <= 9, "At most nine examples are allowed")
    by = bundle["lookup"]["test"]; cases = sorted((c for c in bundle["cases"] if c["experiment_split"] == "test"), key=lambda c:c["sample_id"])
    primary = bundle["primary"]; chosen = []; used = set()
    criteria = [
        ("显式语义条件：改善", method("semantic_base_s1"), method("static_base_s1"), lambda a,b:a["hit"] and not b["hit"]),
        ("显式语义条件：退步", method("semantic_base_s1"), method("static_base_s1"), lambda a,b:not a["hit"] and b["hit"]),
        ("点命中选模：改善", method("flat_base_s1"), method("flat_base_s1","region"), lambda a,b:a["hit"] and not b["hit"]),
        ("同检查点质量分数：改善", method("semantic_hard_s1","point","combined"), method("semantic_hard_s1"), lambda a,b:a["hit"] and not b["hit"]),
        ("同检查点质量分数：退步", method("semantic_hard_s1","point","combined"), method("semantic_hard_s1"), lambda a,b:not a["hit"] and b["hit"]),
        ("同数据分组难例：改善", method("semantic_hard_s1"), method("semantic_expanded_s1"), lambda a,b:a["hit"] and not b["hit"]),
        ("有冻结点可命中但主模型失败", primary, "qwen_native_completed_lax", lambda a,b:not a["hit"] and a.get("frozen_fine_point_oracle") is True),
        ("主模型候选池未覆盖目标", primary, "qwen_native_completed_lax", lambda a,b:a.get("failure_stage")=="no_target_peak_candidate"),
        ("主模型失败、UI-TARS命中", primary, "ui_tars", lambda a,b:not a["hit"] and b["hit"]),
    ]
    for label, first, second, condition in criteria:
        if len(chosen) >= limit: break
        for case in cases:
            sid = case["sample_id"]
            if sid not in used and condition(by[first,sid], by[second,sid]):
                chosen.append((label, case, first, second)); used.add(sid); break
    return chosen


def plot_example(case, first, second, path):
    with Image.open(case["image"]) as opened:
        image = opened.convert("RGB")
    w,h = image.size; geometry = case["target_geometry"]
    polygons = ([[[x0,y0],[x1,y0],[x1,y1],[x0,y1],[x0,y0]] for x0,y0,x1,y1 in geometry["boxes"]]
                if geometry["type"]=="bbox" else [list(poly)+[poly[0]] for poly in geometry["polygons"]])
    vertices = np.concatenate([np.asarray(poly) for poly in polygons]); low,high=vertices.min(0),vertices.max(0)
    margin=np.maximum((high-low)*1.5, np.array([100/w,100/h]))
    bounds=[int(max(0,low[0]-margin[0])*w),int(max(0,low[1]-margin[1])*h),
            int(min(1,high[0]+margin[0])*w),int(min(1,high[1]+margin[1])*h)]
    overview=image.copy();overview.thumbnail((2100,1400))
    fig,axes=plt.subplots(1,2,figsize=(13,5.5),gridspec_kw={"width_ratios":[1.4,1]})
    axes[0].imshow(overview,extent=[0,w,h,0]);axes[1].imshow(image.crop(bounds),extent=[bounds[0],bounds[2],bounds[3],bounds[1]])
    for ax in axes:
        for poly in polygons:
            points=np.asarray(poly)*[w,h];ax.plot(points[:,0],points[:,1],color="#55e04b",lw=1.8)
        for pred,color,marker,label in ((first,"#ff783c","x","First method"),(second,"#1985e0","+","Second method")):
            for i,candidate in enumerate(pred.get("candidates",[])[:5]):
                x0,y0,x1,y1=candidate["region"]
                ax.add_patch(plt.Rectangle((x0*w,y0*h),(x1-x0)*w,(y1-y0)*h,fill=False,ec=color,lw=1.4 if i==0 else .6,alpha=.85 if i==0 else .3))
            if pred["point"] is not None:
                ax.scatter(*pred["point"],color=color,marker=marker,s=90,lw=2,label=label)
        ax.set_axis_off()
    axes[0].set(xlim=(0,w),ylim=(h,0),title="Target geometry, selected parent and top-5 candidates")
    axes[1].set(xlim=(bounds[0],bounds[2]),ylim=(bounds[3],bounds[1]),title="Original-pixel target crop")
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc="lower center",ncol=2,fontsize=9)
    fig.subplots_adjust(left=.01,right=.99,top=.92,bottom=.13,wspace=.03)
    fig.savefig(path,dpi=145);plt.close(fig)


def render_report(bundle, max_examples=9):
    out=bundle["out"];dest=out/"evaluation";assets=dest/"assets";assets.mkdir(parents=True,exist_ok=True)
    plots=plot_results(bundle,assets)+plot_structure(assets)
    examples=[]
    for i,(label,case,first,second) in enumerate(select_examples(bundle,max_examples)):
        path=assets/f"example_{i+1:02d}.png";lookup=bundle["lookup"]["test"]
        plot_example(case,lookup[first,case["sample_id"]],lookup[second,case["sample_id"]],path)
        plots.append(path);examples.append(dict(label=label,case=case,first=first,second=second,path=path))
    p=bundle["protocol"];test=bundle["summaries"]["test"];m=test["methods"];primary=bundle["primary"]
    replica=primary.replace("_s1__","_s2__")
    document=[];markdown=[]
    def heading(title,level=2):
        document.append(f"<h{level}>{html.escape(title)}</h{level}>");markdown.append("#"*level+" "+title)
    def paragraph(text,kind=None):
        document.append('<p'+(' class="note"' if kind else '')+'>'+html.escape(text)+'</p>');markdown.append(text)
    def table(headers,body):
        document.append(table_html(headers,body));markdown.append(table_md(headers,body))
    def figure(name,caption):
        path=assets/(name+".png")
        document.append('<figure><img src="data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode()+'"><figcaption>'+html.escape(caption)+'</figcaption></figure>')
        markdown.append(f"![{caption}](assets/{name}.png)\n\n{caption}")
    heading("逐头语义路由、实际点选模与同图难例：独立测试 v7",1)
    paragraph(f"封存协议预先指定的主模型 {primary} 在新独立测试上命中 {hit_text(m[primary])}；同方案第二种子 {replica} 为 {hit_text(m[replica])}。同图原生 Qwen 为 {hit_text(m['qwen_native_completed_lax'])}，UI-TARS 为 {hit_text(m['ui_tars'])}。本文保留全部两种子结果，没有按测试集选择主模型。")
    paragraph("主指标是静态正样本截图上的实际整数点是否落入有效目标几何，并非电脑点击执行、文件保存成功、负例拒绝或完整 agent 任务成功。主表使用新的独立测试；已经反复分析过的旧 192 题单列为探索性诊断。",True)
    heading("实验结构与受控变量")
    paragraph("读取冻结 Qwen 生成名称 token 自身的状态；候选是全部 token×96 原始头峰值的父格并集。每个候选保留每 token、每头的六通道证据。新增逐头分支先让 query、末 token phrase、候选视觉、头身份、图片注意力总量和熵进入兼容性评分，再做四路软头池化和四路软 token 池化，最后只给候选分数加残差。")
    paragraph("static 与 semantic 物理参数数相同；static 仅将新路由内显式 query/phrase 条件置零，旧 base 语义分支和原注意力场保留。因此 static 不是完全不含目标信息的模型。flat 为原 v6 扁平投影结构，参数量不同，不能当作相同参数量的唯一变量对照。")
    paragraph("点质量分支只接收 detached 的语义投影、候选分数和 13 项冻结点特征，以训练集上该固定点是否命中作 BCE 监督。它不反向更新 core，梯度裁剪分组独立；合成分数为 core score + log sigmoid(quality)。类别平衡 BCE 不产生校准的点击安全概率。")
    figure("implementation","结构示意：主计分统一使用旧 v5 冻结精定位；模型自己的 child/offset 仅作辅助训练和另列诊断。")
    table(["轨迹","seed","架构","训练池","训练配方","core参数","总参数"],[[short(j["name"]),j["seed"],j["architecture"],j["data"],j["recipe"],bundle["histories"][j["name"]]["core_parameters"],bundle["histories"][j["name"]]["parameters"]] for j in p["jobs"]])
    paragraph(f"每轨迹训练 {p['epochs']} epoch，每 epoch {p['training_queries_per_epoch']} 条 query、{p['updates_per_epoch']} 次更新、batch={p['batch_size']}。同一轨迹分别保存区域优先、固定点命中优先、质量合成点命中优先三个验证检查点；选模只读取旧验证集。random 与 grouped_hard 使用协议约定的共同 query 集合和一致候选移除决定，后者改变组内顺序并加同图其他目标的负例 margin loss。组合 recipe 不能拆解为分组顺序与 margin 项各自独立的因果收益。")
    heading("数据与隔离")
    d=bundle["data"]
    table(["数据部分","query数","解释"],[["原训练池",p["train_queries_base"],"保持原内容"],["真实新增训练 query",p["train_queries_extra"],"官方 train 指令与几何；不是合成改写"],["扩展训练池",p["train_queries_expanded"],"总更新预算保持相同"],["旧验证集",p["validation_queries"],"跨研究轮复用；全部 epoch 选择只看它"],["旧诊断集",p["diagnostic_queries"],"已用于前轮诊断，不作为新独立验证"],["新独立测试",p["fresh_test_queries"],"全部模型在封存后才提取新特征"]])
    paragraph(f"新增 query 来源：{json.dumps(d.get('extra_by_source',{}),ensure_ascii=False)}；同图互不重叠真实目标对共 {d.get('same_image_distinct_nonoverlap_pairs','未提供')} 对，其中包含新增 query 的 {d.get('same_image_pairs_involving_added_query','未提供')} 对。新测试按三公开来源各 {p['fresh_test_queries']//3} 张截图选择；exact RGB SHA 排除历史图像，并以 dHash256 Hamming≤{p['fresh_test_near_exclusion_radius']} 排除近重复。此哈希策略不保证发现一切语义近重复。")
    selection_audit=bundle.get("selection_audit")
    if selection_audit is not None:
        paragraph(f"独立数据审计 {sum(selection_audit['checks'].values())}/{len(selection_audit['checks'])} 项通过；扩展训练池有 {selection_audit['expanded_train_images']} 张独立截图，新增目标涉及 {selection_audit['extra_images']} 张图，其中新引入的训练图为 {selection_audit['new_train_images']} 张，其余为原训练图增加目标。实际最近 dHash 距离：{json.dumps(selection_audit['minimum_dhash_distances'],ensure_ascii=False)}。")
    heading("主表：新独立测试的全部十轨迹与三个验证选择器")
    paragraph("下表每一行都是协议允许的检查点，不按测试得分重排或挑胜者。训练池全量分母随数据条件不同；其训练得分不是相同数据分母的泛化比较。区域优先/点优先使用 rank，质量优先使用 combined；实际点列一律共同冻结 fine。")
    headers=["轨迹","seed","验证选择器","epoch","训练池固定点命中","验证固定点命中","新独立测试固定点命中","测试父格相交"]
    table(headers,checkpoint_table(bundle,"test"));figure("fresh_scores","两种子与同图基线均完整保留；左图固定使用点优先检查点的 rank 分数，右图展示预指定组合主方案。")
    heading("同一轨迹选模与同一检查点点质量消融")
    paragraph("三个验证选择器对应同一训练轨迹；同一 point 检查点的 rank、combined、quality-only 使用完全相同 core/质量权重与候选。不能把换 checkpoint 的收益误称为只加一个分数的收益。")
    table(["轨迹","rank固定点命中","同checkpoint combined","同checkpoint quality-only"],[[short(j["name"]),hit_text(m[method(j["name"])]),hit_text(m[method(j["name"],"point","combined")]),hit_text(m[method(j["name"],"point","quality")])] for j in p["jobs"]])
    figure("selectors_and_quality","左：同一训练轨迹的三种验证选模；右：同一 point 检查点改变候选计分。")
    heading("共同冻结精定位与其他坐标输出的区别")
    auxiliary=[]
    for j in p["jobs"]:
        for rule in RULES:
            name=method(j["name"],rule,chosen_mode(rule))
            auxiliary.append([short(j["name"]),RULE_LABELS[rule],hit_text(m[name]),hit_text(m[name+"_ownfine"]),hit_text(m[name+"_center"])])
    table(["轨迹","验证选择器","主：共同冻结fine","辅助：自身fine","辅助：父格中心"],auxiliary)
    paragraph("主分数隔离候选选择的变化：所有新模型、旧 v5/v6 的 fixed 比较均从同一份冻结 v5 点缓存取点。自身 fine 会受辅助训练影响，因此单独报告。原生 Qwen、UI-TARS 和历史 phrase-local 使用各自坐标解码，作为同图系统参考；它们不受冻结 v5 候选点 oracle 约束。")
    heading("同图系统基线与按来源切片")
    comparison_names=[primary,replica,"v5_fixed","v6_s1_fixed","v6_s2_fixed","phrase_local","qwen_native_completed_lax","ui_tars"]
    datasets=sorted(m[primary]["datasets"])
    table(["方法","全部测试","输出点覆盖"]+datasets,[[name,hit_text(m[name]),f"{m[name]['points']}/{m[name]['n']}"]+[f"{m[name]['datasets'][ds]['hits']}/{m[name]['datasets'][ds]['n']}" for ds in datasets] for name in comparison_names])
    heading("失败分解与候选点上限")
    failure_names=[method("flat_base_"+s) for s in ("s1","s2")]+[primary,replica,"v6_s1_fixed","v6_s2_fixed"]
    table(["方法"]+list(STAGE_LABELS)+["父格候选覆盖GT oracle","冻结点可命中GT oracle"],[[name]+[sum(r.get("failure_stage")==stage for r in bundle["predictions"]["test"] if r["method"]==name) for stage in STAGES]+[count_text(m[name].get("candidate_region_coverage"),test["n"]),count_text(m[name].get("frozen_fine_point_oracle"),test["n"])] for name in failure_names])
    paragraph("这两个 oracle 都用 GT 做事后诊断，不是可部署策略。父格相交不保证其冻结整数点命中；冻结点 oracle 表示候选列表里至少存在一个同一冻结 decoder 的点能命中，只限制本报告使用该点缓存的主方法，不代表 Qwen 能力上限。未生成有效名称、候选池没有目标、排序失败、选到相交父格但固定点错误均保留在完整测试分母。")
    recoverable=[]
    for name in (primary,replica):
        rows=[r for r in bundle['predictions']['test'] if r['method']==name]
        available=lambda r: bool(r.get('frozen_fine_point_oracle',False))
        recoverable.append([name,sum(r['hit'] for r in rows),
            sum(not r['hit'] and available(r) for r in rows),
            sum(available(r) and r.get('failure_stage')=='ranking_miss_with_target_candidate' for r in rows),
            sum(available(r) and r.get('failure_stage')=='fixed_fine_miss_after_intersecting_parent' for r in rows),
            sum(not available(r) for r in rows),
            sum(not available(r) and r.get('candidate_region_oracle',False) for r in rows)])
    table(['方法','已命中','换选现有点尚可救回','其中父格排错','其中相交父格后点错','现有点集合无解','无解但父格覆盖目标'],recoverable)
    paragraph(f"错误阶段与可恢复性是两种划分，不能把所有“有目标候选却排错”都算作仅改善排序能救回：有些样本的全部冻结候选点都落在目标外。主方案在本轮点集合中最多还能救回{recoverable[0][2]}题；另{recoverable[0][5]}题需要改变候选、精定位或名称提取，其中{recoverable[0][6]}题已有相交父格却没有可命中的冻结点。第二种子的可恢复缺口为{recoverable[1][2]}题。这些计数只针对当前固定点集合。")
    figure("failure_stages","主方法的全部失败均归入明确阶段，避免只统计 gate 放行或有目标候选的子集。")
    heading("主方案按来源的完整失败与覆盖",3)
    per_dataset=[]
    for name in (primary,replica):
        for ds in datasets:
            f=failure_summary(bundle["predictions"]["test"],name,ds)
            per_dataset.append(["S1" if "_s1__" in name else "S2",ds,f['n'],f['hits'],f['selected_parent_overlap'],
                f['parent_oracle'],f['frozen_point_oracle'],f['stages']['no_name'],f['stages']['no_target_peak_candidate'],
                f['stages']['ranking_miss_with_target_candidate'],f['stages']['fixed_fine_miss_after_intersecting_parent']])
    table(["种子","来源","完整分母","主点命中","所选父格相交","候选parent oracle","冻结point oracle","无名称","无目标候选","排序错","相交后固定点错"],per_dataset)
    paragraph("上表直接按每条预测记录汇总，不用总体比例推算子集。每来源64个样本内，hit与四类失败之和必须等于64；缺失名称也计入分母。parent oracle和固定点oracle衡量当前缓存能否表达正确落点，不能保证排序器实现这个上限，也不是VLM能力上限。")
    figure("failure_by_dataset","两种子分别显示三个来源的完整失败结构，避免总体均值掩盖仅某个来源有效。")
    base_checkpoint=bundle['histories']['flat_base_s1']['checkpoints']['point']
    expanded_checkpoint=bundle['histories']['semantic_expanded_s1']['checkpoints']['point']
    oracle_ranges=[]
    for label,metric in (("原训练池",base_checkpoint['training']),("扩展训练池",expanded_checkpoint['training']),
                          ("验证集",base_checkpoint['validation'])):
        oracle_ranges.append([label,metric['n'],metric['candidate_region_coverage'],metric['frozen_fine_point_oracle']])
    for split,label in (("diagnostic","旧探索性诊断"),("test","新独立测试")):
        metric=bundle['summaries'][split]['methods'][primary]
        oracle_ranges.append([label,metric['n'],metric.get('candidate_region_coverage','不可评估'),metric.get('frozen_fine_point_oracle','不可评估')])
    table(["范围（分母互不混用）","query数","候选父格相交GT oracle","共同冻结点GT oracle"],oracle_ranges)
    heading("配对差异与不确定性")
    paragraph("以下直接呈现预先计划的同图比较；正值支持前一方法。按数据集分层的 5,000 次配对 bootstrap，未校正多重比较；区间条件于已训练模型和既定数据选择，不覆盖训练随机性。两种子分别报告，不能用其中较好者替换预指定主种子。")
    for split,title in (("test","新独立测试"),("diagnostic","旧诊断集（探索性）")):
        heading(title+"：配对比较",3)
        table(["因素","前一方法","后一方法","差值pp","95% CI pp","赢/输"],[[FACTOR_LABELS.get(c["factor"],c["factor"]),c["first"],c["second"],f"{c['delta_pp']:+.2f}",f"[{c['ci95_pp'][0]:+.2f}, {c['ci95_pp'][1]:+.2f}]",f"{c['wins']}/{c['losses']}"] for c in bundle["summaries"][split]["paired_comparisons"]])
    figure("paired_intervals","同一计划比较分别展示新独立测试和旧诊断集；旧诊断集结果不作为独立确认。")
    heading("旧 192 题：单列的探索性复用结果")
    paragraph("这些截图已经参与上一轮诊断并启发了本轮设计，不能与新测试混合汇总为更大独立测试。检查点仍只由验证集选择，没有按旧诊断成绩改 checkpoint。")
    table(headers[:-2]+["旧诊断固定点命中","旧诊断父格相交"],checkpoint_table(bundle,"diagnostic"))
    heading("验证学习曲线与选中 epoch")
    figure("validation_curves","所有十条轨迹的验证曲线；标记由各自 selector 选中的 epoch。没有绘制用于调参的测试学习曲线。")
    routing_path=out/'evaluation/routing_diagnostic/summary.json'
    if routing_path.exists():
        routing=read(routing_path)
        require(routing['scope']=='validation_only_post_hoc_mechanism_diagnostic','Unexpected routing diagnostic scope')
        require(routing['source']['frozen_sha256']==sha(out/'frozen_readouts.json'),'Routing diagnostic checkpoint index mismatch')
        require(routing['source']['script_sha256']==sha(Path('scripts/diagnose_semantic_head_routing.py')),'Routing diagnostic source mismatch')
        heading("补充机制诊断：条件筛选是否实际发生")
        paragraph(f"只使用验证集按 sample_id 排序的前{routing['selection']['count']}条有效名称样本；六个 point 检查点共用此子集，没有读取新测试，也没有据此重新训练或选模。")
        diagnostic_rows=[]
        for name,cp in routing['checkpoints'].items():
            require(cp['checkpoint_sha256']==bundle['frozen']['models'][name]['sha256'],'Routing diagnostic model mismatch')
            stats=cp['summary']['case_equal_statistics'];outcomes=cp['summary']['outcomes']
            value=lambda key: stats[key]['mean']
            diagnostic_rows.append([name,f"{value('effective_heads'):.2f}/96",f"{value('head_entropy_normalized'):.4f}",
                f"{value('grouped_representation_group_cosine'):.4f}",f"{value('token_entropy_normalized'):.5f}",
                outcomes['explicit_condition_top1_changed'],f"{outcomes['rank_point_hit']} / {outcomes['condition_off_rank_point_hit']}"])
        table(['检查点','有效头数 exp(H)','头归一熵','池后组间cosine','token归一熵','去语义条件后首选改变','原命中 / 去条件命中'],diagnostic_rows)
        paragraph("当前头权重接近均匀、四组表示高度相似，显式语义条件对所选候选的影响很小；支持这个实现的条件筛选较弱。有效头数只是权重弥散程度，不能单独证明模块无用；候选表示仍有区别，也不能把整个分支称为常数。去条件是已训练模型的事后干预，不等于重新训练的独立消融。完整逐样本与残差诊断见 routing_diagnostic/summary.json 和 notes.md。")
    if examples:
        heading("事后案例：展示改善、退步与结构性失败")
        paragraph("以下使用 GT 和已完成预测挑选案例，只作事后解释，不用于训练、阈值或检查点选择。每一类别按 sample_id 稳定选首个且不重复，最多九张；没有符合条件的类别不伪造案例。绿线为完整目标几何；橙色是前一方法，蓝色是后一方法；粗框为所选父格，淡框为前五候选，点均为各方法实际主输出。")
        for item in examples:
            heading(item["label"],3);case=item["case"]
            paragraph(f"{case['sample_id']} · {case['dataset']} · {case['instruction']}；前一方法 {item['first']}，后一方法 {item['second']}。")
            figure(item["path"].stem,"左为全图，右为原始像素裁剪；裁剪仅用于报告展示，没有重送模型。")
    heading("审计、复现与限制")
    a=bundle["audit"]
    table(["审计项","结果"],[["轨迹 / checkpoint",f"{a['trajectories_verified']} / {a['checkpoints_verified']}"],["CPU/GPU主方法回放精确一致",a["fresh_primary_cpu_gpu_exact"]],["旧冻结文件未改",a["previous_frozen_artifacts_unchanged"]],["新测试特征在封存后提取",a["fresh_features_after_freeze"]],["backbone可训练参数",a["backbone_trainable_parameters"]],["测试通过数",a["tests_passed"]],["manifest SHA256",p["manifest_sha256"]],["frozen index SHA256",a["frozen_sha256"]]])
    numeric=a.get('fixed_points',{}).get('cpu_numeric_exceptions',[])
    if numeric:
        require(all(e['canonical_points_exact'] and (e.get('canonical_all_13_float16_features_exact',False) or
            e.get('canonical_all_13_float16_elements_reproduced',False)) for e in numeric),'Numeric exception lacks verified coordinate/descriptor evidence')
        paragraph(f"冻结fine的CPU复算发现{len(numeric)}个候选存在跨半像素取整差异。同一原cuda:0批次的固定五次运行中，全部整数坐标及坐标派生列每次均逐位一致。13维特征的每个缓存元素都实际复现，但这五次没有同一遍整批完全一致：两处descriptor在相邻FP16舍入边界上交错波动。直接记录的不一致float32值与匹配运行的量化区间共同约束边界；区间推断不冒充直接测量值。完整日志及其事后认证方式保留于numeric_evidence，不追加随机重试、不改缓存和预测。主模型CPU候选选择另行严格核验。这里有数值复现例外，不能称所有特征跨设备逐位一致。")
        paragraph("下表GPU连续坐标引用保留的先前单次独立复现，来源在审计事件中单列；当前正式五次记录的是整数点、坐标派生列和descriptor差异，没有直接记录全部连续坐标或所有匹配元素的float32值。")
        table(['样本 / 候选','CPU连续坐标 / 整数点','先前独立GPU连续坐标 / 缓存整数点','该点命中变化','该图point oracle变化'],
            [[e['sample_id']+' / '+str(e['candidate_index']),str([round(v,10) for v in e['cpu_continuous_point']])+' / '+str(e['cpu_point']),
              str([round(v,10) for v in e['canonical_gpu_continuous_point']])+' / '+str(e['stored_point']),e['hit_changed'],e['point_oracle_changed']] for e in numeric])
        table(['批次','候选数','五次13维差异元素数','整批完全复现的次数索引（零起始）','13列逐元素实际复现','运行间特征逐位稳定'],
            [[str(b['split'])+' / '+str(b['batch_start']),b['n_candidates'],
              str([r['float16_feature_mismatch_elements'] for r in b['runs']]),str(b['whole_batch_13_column_exact_witness_runs']),
              b.get('all_13_columns_elementwise_reproduced',b.get('witness_exact',False)),b['interrun_features_bitstable']] for b in a['fixed_points']['canonical_gpu_exception_batches']])
    for text in [
        "样本量：新测试只有192张、每来源64张；两种子不足以估计完整训练随机性，探索性多重比较不提供确定的显著性胜负。",
        "任务范围：只测静态正例指向，不测禁用/缺失/禁止点击、拒绝误放行、多步交互或实际 OS 执行。",
        "验证复用：旧验证集跨研究轮反复使用；新独立测试是本轮的独立确认，后续若据其改模型就必须再找新测试。",
        "训练范围：扩展池改变真实样本构成；相同600更新query预算不意味着相同训练池。分组与hard margin合成一个recipe，不能分别归因。",
        "结构范围：全部头先条件化再池化仍会压缩信息；四组可冗余或塌缩，没有额外多样性约束；没有增加原图裁剪重编码。",
        "固定fine：候选点oracle是本轮固定decoder的上限，不是注意力、VLM或自由坐标生成的普遍上限。",
        "质量分数：类平衡监督与logsigmoid合成是排序规则，不是校准风险概率；GT只进监督和审计。",
    ]: paragraph(text)
    paragraph("复现命令：powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_semantic_head_experiment.ps1。报告只读取封存协议、histories、两组summary/predictions、data_selection和final_audit；不重新选择模型，不训练，也不调用GPU。PNG/PDF图在assets，所有PNG内嵌于此HTML，便于单文件查看。")
    style="body{font:16px/1.7 system-ui,'Microsoft YaHei',sans-serif;max-width:1350px;margin:30px auto;padding:0 22px;color:#253442;background:#f6f8fa}h1,h2{line-height:1.35}h2{margin-top:38px}.note{background:#fff0cb;padding:14px;border-left:4px solid #d49b39}.table{overflow-x:auto}table{border-collapse:collapse;width:100%;background:white;font-size:14px}td,th{padding:8px 10px;border-bottom:1px solid #dce2e8;text-align:left;vertical-align:top}th{background:#eaf0f5}figure{margin:22px 0}img{max-width:100%;background:white}figcaption{font-size:14px;color:#596879}p{overflow-wrap:anywhere}"
    (dest/"report.html").write_text('<!doctype html><html lang="zh"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>逐头语义路由实验 v7</title><style>'+style+'</style></head><body>'+''.join(document)+'</body></html>',encoding="utf-8")
    (dest/"report.md").write_text('\n\n'.join(markdown)+'\n',encoding="utf-8")
    integrity=dict(report_source_sha256=sha(Path(__file__)),protocol_sha256=sha(out/"protocol.json"),
        manifest_sha256=p["manifest_sha256"],audit_sha256=sha(out/"final_audit.json"),
        primary=primary,test_winner_selection=False,trajectories=len(p["jobs"]),examples=len(examples),
        predictions_sha256={s:bundle["summaries"][s]["predictions_sha256"] for s in ("test","diagnostic")},
        report_sha256={name:sha(dest/name) for name in ("report.html","report.md")},
        assets_sha256={str(path.relative_to(dest)).replace('\\','/'):sha(path) for path in plots})
    if bundle.get("selection_audit") is not None:
        integrity['data_selection_audit_sha256']=sha(out/'data_selection_audit.json')
    if routing_path.exists():
        integrity['routing_diagnostic_sha256']=sha(routing_path)
    (dest/"report_integrity.json").write_text(json.dumps(integrity,ensure_ascii=False,indent=2)+'\n',encoding="utf-8")
    return integrity


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('runs/semantic_head_v7'))
    parser.add_argument('--max-examples',type=int,default=9)
    args=parser.parse_args(argv)
    require(0 <= args.max_examples <= 9,'--max-examples must be between 0 and 9')
    bundle=load_bundle(args.output);result=render_report(bundle,args.max_examples)
    print('REPORT '+str(args.output/'evaluation/report.html')+' '+json.dumps(dict(primary=result['primary'],examples=result['examples'])),flush=True)


if __name__=='__main__':main()
