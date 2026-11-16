"""Post-audit V13 mechanism diagnostics; never trains, predicts or selects a policy.

The CLI may read fresh results only after both cohorts are scored and the final
audit binds every consumed artifact. Pure helpers accept synthetic inputs for
tests. No model, raw attention values or hidden-state arrays are loaded.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import html
import json
import math
from pathlib import Path
import zipfile

import numpy as np
import shapely
from shapely.geometry import box

from clickagent.public_evaluation import point_in_geometry
from clickagent.spatial_geometry import shape_from_geometry


WORKSPACE = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = WORKSPACE / "runs/multiregion_v13"
COHORTS = ("test", "supplement")
BUDGETS = (16, 32)
CHAIN_STAGES = ("decoded_union_no_valid_point", "quota_excludes_all_valid_points",
                "score_top4_excludes_all_valid_points", "gui_misselection", "success")
LIMITS = {
    "raw_union": "GT诊断：当前冻结proposer已解码坐标的并集可达性；不是原始attention信息上限，也不是模型能力天花板。",
    "crops": "事后有效目标几何与crop的交集；空间覆盖不等于正确点已被解码。边界接触不计正面积覆盖。",
    "attribution": "分别呈现搜索、选模、Top4目标对照；整体主比较同时改变搜索与选模，不能全部归因于搜索。",
    "cv": "评分器来源留出CV只用于选epoch；上游已见来源及其验证标签，CV不是独立泛化成绩或全链路LODO。",
    "seeds": "S1固定主种子，S2固定稳健性；逐一报告，不按测试选种子、epoch、阈值或方法。",
    "cost": "Top4 GUI重用同池分数是候选验证数代理；不构成新的端到端时延实测。多crop增加上游计算。",
    "cohorts": "主测试192与补充6单列；新主测试来源构成改变，不能直接与历史64×3队列分数比较。",
}


def require(value, message):
    if not value:
        raise ValueError(message)


def path_for(value):
    p = Path(value)
    return p if p.is_absolute() else WORKSPACE / p


def canonical(value):
    return str(path_for(value).resolve()).casefold()


def sha(path):
    h = hashlib.sha256()
    with path_for(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(path_for(path).read_text(encoding="utf-8-sig"))


def key(sid):
    return hashlib.sha256(sid.encode()).hexdigest()[:24]


class Evidence:
    """Authenticate only consumed artifacts against the successful final audit."""
    def __init__(self, expected):
        self.expected = {}
        for name, digest in expected.items():
            k = canonical(name)
            require(k not in self.expected or self.expected[k] == digest, "Conflicting audited identity")
            self.expected[k] = digest
        self.used = {}
        self.states = {}

    def need(self, path):
        p = path_for(path).resolve(); k = canonical(p)
        require(k in self.expected, "Not bound by successful final audit: " + str(p))
        stat = p.stat(); state = (stat.st_size, stat.st_mtime_ns)
        if str(p) in self.used:
            require(self.states[str(p)] == state, "Artifact changed during analysis: " + str(p))
        else:
            digest = sha(p)
            require(digest == self.expected[k], "Audited artifact bytes changed: " + str(p))
            require((p.stat().st_size, p.stat().st_mtime_ns) == state, "Artifact changed while hashing")
            self.used[str(p)] = digest; self.states[str(p)] = state
        return p

    def json(self, path):
        return read(self.need(path))

    def rows(self, path):
        return [json.loads(line) for line in self.need(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]

    def unchanged(self):
        for path, state in self.states.items():
            stat = Path(path).stat()
            require((stat.st_size, stat.st_mtime_ns) == state, "Artifact changed before analysis commit")


def authorize_reads(root):
    """Check gates without opening summary/details/pipeline test artifacts first."""
    root = Path(root).resolve()
    needed = [root / "frozen.json", root / "final_audit.json"] + [
        root / "evaluation" / split / "summary.json" for split in COHORTS]
    require(all(p.is_file() for p in needed), "Need freeze, both evaluation summaries, and successful final_audit before analysis")
    frozen = read(root / "frozen.json"); audit = read(root / "final_audit.json")
    require(frozen.get("complete") is True and frozen.get("fresh_test_features_at_freeze") == 0,
            "Invalid pretest freeze")
    require(audit.get("complete") is True and audit.get("version") == 13, "Final audit did not pass")
    require(audit.get("frozen_sha256") == sha(root / "frozen.json"), "Final audit belongs to another freeze")
    ledger = Evidence(audit["input_sha256"])
    for split in COHORTS:
        p = root / "evaluation" / split / "summary.json"
        ledger.need(p)
        require(audit["evaluation"][split]["summary_sha256"] == sha(p), "Summary not bound to final audit")
    return ledger, audit


def point_hit(point, case):
    if point is None:
        return False
    w, h = case["image_size"]
    return bool(point_in_geometry([point[0] / w, point[1] / h], case["target_geometry"]))


def crop_relation(case, transform):
    """Intersect actual valid multibox/polygon union, never its bounding box."""
    shape = shape_from_geometry(case["target_geometry"])
    w, h = case["image_size"]; a = transform["box"]
    require(len(a) == 4 and all(math.isfinite(v) for v in a) and
            0 <= a[0] < a[2] <= w and 0 <= a[1] < a[3] <= h, "Invalid crop box")
    crop = box(a[0] / w, a[1] / h, a[2] / w, a[3] / h)
    intersection = shape.intersection(crop)
    relation = ("positive_area" if intersection.area > 0 else
                "boundary_contact" if not intersection.is_empty else "separated")
    return dict(relation=relation, box=a, target_area_coverage=float(intersection.area / shape.area),
        bbox_envelope_would_overclaim=bool(shape.envelope.intersection(crop).area > 0 and intersection.area == 0))


def decoded_union(pipeline, condition):
    """All archived decoded points, including low ranked; no new model output."""
    require(condition in ("c1", "c3"), "Unknown condition")
    points = list(pipeline["global_result"]["all_points"])
    views = pipeline["local_views"][:1 if condition == "c1" else 3]
    for view in views:
        transform = view["transform"]; left, top, _, _ = transform["box"]
        w, h = transform["original_size"]; scale = transform["scale"]
        require(scale > 0, "Invalid crop scale")
        for x, y in view["result"]["all_points"]:
            points.append([min(w - 1, max(0, int(round(x / scale + left)))),
                           min(h - 1, max(0, int(round(y / scale + top))))])
    result = []; seen = set()
    for point in points:
        require(len(point) == 2 and all(math.isfinite(v) and v == round(v) for v in point), "Noninteger decoded point")
        token = tuple(map(int, point))
        if token not in seen:
            seen.add(token); result.append(list(token))
    return result


def raw_token_count(path):
    """Read only the NPY header, avoiding loading target attention values."""
    with zipfile.ZipFile(path_for(path)) as archive, archive.open("target_attention.npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, _, _ = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, _, _ = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError("Unsupported raw attention NPY header")
    require(len(shape) == 3 and shape[1] == 96, "Expected archived [T,96,P] target attention")
    return int(shape[0])


def paired(a, b, ids=None, sources=None):
    require(len(a) == len(b) and len(a) > 0, "Need aligned nonempty paired vectors")
    n = len(a)
    result = dict(n=n, method_hits=sum(a), reference_hits=sum(b),
        wins=sum(bool(x) and not y for x, y in zip(a, b)),
        losses=sum(bool(y) and not x for x, y in zip(a, b)),
        both_hit=sum(bool(x and y) for x, y in zip(a, b)),
        both_miss=sum(not (x or y) for x, y in zip(a, b)), delta=(sum(a) - sum(b)) / n)
    if ids is not None:
        require(len(ids) == n, "Case IDs not aligned")
        result["gained_ids"] = [sid for sid, x, y in zip(ids, a, b) if x and not y]
        result["lost_ids"] = [sid for sid, x, y in zip(ids, a, b) if y and not x]
    if sources is not None:
        require(len(sources) == n, "Sources not aligned")
        result["by_source"] = {}
        for source in sorted(set(sources)):
            ix = [i for i, value in enumerate(sources) if value == source]
            result["by_source"][source] = paired([a[i] for i in ix], [b[i] for i in ix],
                [ids[i] for i in ix] if ids is not None else None)
    return result


def failure_stage(raw, quota, retained, final):
    require(not quota or raw, "Quota point missing from decoded union")
    require(not retained or quota, "Shortlist oracle outside quota oracle")
    require(not final or retained, "GUI success outside reachable shortlist")
    return ("decoded_union_no_valid_point" if not raw else "quota_excludes_all_valid_points" if not quota else
            "score_top4_excludes_all_valid_points" if not retained else "gui_misselection" if not final else "success")


def saved_interval(summary, name, reference, top4=False):
    for row in summary.get("paired_top4_oracle" if top4 else "paired_comparisons", []):
        if row["method"] == name and row["reference"] == reference:
            return row["ci95"]
        if row["method"] == reference and row["reference"] == name:
            return [-row["ci95"][1], -row["ci95"][0]]
    return None


def compare_methods(details, summary, name, reference):
    metrics = {}
    sources = [d["source"] for d in details]
    for metric, field in (("head_top1", "hit"), ("score_top4_retention", "top4_oracle")):
        values = [d["methods"] for d in details]
        a = [v[name][field] for v in values]; b = [v[reference][field] for v in values]
        row = paired(a, b, [d["sample_id"] for d in details], sources)
        row["ci95_from_scored_summary"] = saved_interval(summary, name, reference, top4=field == "top4_oracle")
        metrics[metric] = row
    gui_name, gui_reference = [v.replace("__K16", "_top4_gui__K16") for v in (name, reference)]
    metrics["top4_gui_final"] = paired([d["methods"][gui_name]["hit"] for d in details],
        [d["methods"][gui_reference]["hit"] for d in details], [d["sample_id"] for d in details], sources)
    metrics["top4_gui_final"]["ci95_from_scored_summary"] = saved_interval(summary, gui_name, gui_reference)
    return dict(method=name, reference=reference, metrics=metrics)


def analyze_cohort(cases, details, pipelines, summary, *, token_counts=None):
    """Pure, deterministic posthoc analysis over a fully scored cohort."""
    require(cases and len(cases) == len(details) == len(pipelines), "Cohort coverage mismatch")
    ids = [c["sample_id"] for c in cases]
    require(len(set(ids)) == len(ids), "Repeated case")
    detail_map = {d["sample_id"]: d for d in details}; pipe_map = {p["sample_id"]: p for p in pipelines}
    require(set(detail_map) == set(pipe_map) == set(ids), "Artifact/case mismatch")
    details = [detail_map[sid] for sid in ids]; pipelines = [pipe_map[sid] for sid in ids]
    require(summary["complete"] and summary["n"] == len(cases), "Invalid scored summary")
    token_counts = token_counts or {}
    per_case = []; chains = {}; crop_counts = Counter({k: 0 for k in
        ("positive_area", "boundary_contact", "separated", "bbox_envelope_overclaims")})
    region_counts = {}; references = []
    for case, detail, pipeline in zip(cases, details, pipelines):
        raw = {}; quotas = {}; crop_rows = []
        for region, view in enumerate(pipeline["local_views"]):
            relation = crop_relation(case, view["transform"])
            t = token_counts.get(view["feature_path"])
            item = dict(index=region, **relation, no_name=view["result"].get("failure") == "no_name",
                        target_token_count=t, t0=(t == 0) if t is not None else None,
                        decoded_points=len(view["result"]["all_points"]))
            crop_rows.append(item); crop_counts[item["relation"]] += 1
            crop_counts["bbox_envelope_overclaims"] += int(item["bbox_envelope_would_overclaim"])
            counter = region_counts.setdefault(str(region), Counter())
            counter.update(views=1, no_name=int(item["no_name"]), t0=int(t == 0), t_unknown=int(t is None),
                           no_name_nonzero_tokens=int(item["no_name"] and t is not None and t > 0),
                           named_but_zero_tokens=int(not item["no_name"] and t == 0))
        for condition in ("c1", "c3"):
            points = decoded_union(pipeline, condition); coords = {tuple(p) for p in points}
            hits = [point_hit(p, case) for p in points]
            raw[condition] = dict(oracle=any(hits), points=len(points), valid_points=sum(hits))
            quotas[condition] = {}
            for budget in BUDGETS:
                pool = detail["pools"][condition][str(budget)]
                require(all(tuple(p) in coords for p in pool["points"]), "Saved quota point absent from raw decoded union")
                oracle = any(point_hit(p, case) for p in pool["points"])
                require(oracle == pool["point_oracle"], "Scored quota oracle mismatch")
                quotas[condition][str(budget)] = dict(oracle=oracle, points=len(pool["points"]))
        chain_rows = {}
        for name, metric in detail["methods"].items():
            if not name.endswith("_top4_gui__K16"):
                continue
            condition = metric["condition"]
            parent = name.replace("_top4_gui__K16", "__K16")
            retained = bool(detail["methods"][parent]["top4_oracle"])
            stage = failure_stage(raw[condition]["oracle"], quotas[condition]["16"]["oracle"], retained, metric["hit"])
            chains.setdefault(name, Counter())[stage] += 1
            chain_rows[name] = stage
        per_case.append(dict(sample_id=case["sample_id"], source=detail.get("source", case.get("dataset")),
            crop_views=crop_rows, no_crop=not crop_rows, global_no_name=pipeline["global_result"].get("failure") == "no_name",
            raw_decoded_union=raw, quota=quotas, failure_chains=chain_rows))
    attribution = {"search_same_val_selector": [], "selection_same_c3_top1": [], "objective_same_c3_selector": []}
    for seed in ("s1", "s2"):
        def name(arm, policy="val"):
            return f"{arm}_{seed}_{policy}__K16"
        specifications = [("search_same_val_selector", name("c3_top1"), name("c1_top1")),
                          ("selection_same_c3_top1", name("c3_top1", "loso"), name("c3_top1"))]
        specifications.extend(("objective_same_c3_selector", name("c3_top4", policy), name("c3_top1", policy))
                              for policy in ("val", "loso"))
        for kind, method, reference in specifications:
            item = compare_methods(details, summary, method, reference)
            item["seed_role"] = "fixed_primary_s1" if seed == "s1" else "fixed_robustness_s2"
            attribution[kind].append(item)
        for arm, policy in (("c1_top1", "val"), ("c3_top1", "val"), ("c3_top1", "loso"),
                            ("c3_top4", "val"), ("c3_top4", "loso")):
            method = name(arm, policy); condition = arm[:2]
            for reference in ("qwen_native_completed_lax", f"gui_{condition}__K16"):
                references.append(dict(method=method, reference=reference,
                    comparison_scope="native_unrestricted_coordinates" if reference.startswith("qwen") else "same_quota16_point_pool",
                    **paired([d["methods"][method]["hit"] for d in details],
                             [d["methods"][reference]["hit"] for d in details], ids, [d["source"] for d in details])))
            shortlist = method.replace("__K16", "_top4_gui__K16")
            for reference in (f"gui_{condition}__K16", f"gui_{condition}__K4"):
                references.append(dict(method=shortlist, reference=reference,
                    comparison_scope="same_parent_quota16_pool" if reference.endswith("K16") else
                                     "verification_count_proxy_only_distinct_point_sets",
                    **paired([d["methods"][shortlist]["hit"] for d in details],
                             [d["methods"][reference]["hit"] for d in details], ids, [d["source"] for d in details])))
    coverage = {}
    for budget in BUDGETS:
        coverage[str(budget)] = paired([p["quota"]["c3"][str(budget)]["oracle"] for p in per_case],
            [p["quota"]["c1"][str(budget)]["oracle"] for p in per_case], ids, [d["source"] for d in details])
    raw_summary = {c: dict(n=len(cases), oracle=sum(p["raw_decoded_union"][c]["oracle"] for p in per_case),
        total_decoded_unique_points=sum(p["raw_decoded_union"][c]["points"] for p in per_case),
        interpretation=LIMITS["raw_union"]) for c in ("c1", "c3")}
    chain_summary = {}
    for name, counts in chains.items():
        require(sum(counts.values()) == len(cases), "Failure chain not a complete partition")
        stages = {s: int(counts[s]) for s in CHAIN_STAGES}
        chain_summary[name] = dict(n=len(cases), stages=stages,
            raw_union_oracle=len(cases)-stages[CHAIN_STAGES[0]],
            quota16_oracle=sum(stages[s] for s in CHAIN_STAGES[2:]),
            top4_oracle=stages["gui_misselection"]+stages["success"], gui_hits=stages["success"], by_source={})
        for source in sorted({p["source"] for p in per_case}):
            selected = [p for p in per_case if p["source"] == source]
            chain_summary[name]["by_source"][source] = dict(n=len(selected),
                stages={stage: sum(p["failure_chains"][name] == stage for p in selected) for stage in CHAIN_STAGES})
    return dict(n=len(cases), cohort=summary.get("cohort"), attribution=attribution,
        oracle_coverage_c3_minus_c1=coverage, raw_decoded_union=raw_summary,
        crop_geometry=dict(view_counts=dict(crop_counts), by_region={k: dict(v) for k, v in region_counts.items()},
            cases_with_no_crop=sum(p["no_crop"] for p in per_case),
            cases_global_no_name=sum(p["global_no_name"] for p in per_case),
            cases_new_positive_area=sum(any(v["relation"] == "positive_area" for v in p["crop_views"][1:]) and
                not (p["crop_views"] and p["crop_views"][0]["relation"] == "positive_area") for p in per_case)),
        failure_chains=chain_summary, same_pool_and_native_references=references,
        paired_intervals_source="Existing final scored summary; no new bootstrap/seed/policy selection", per_case=per_case)


def training_pool_changes(cases, indices, labels_for):
    """Compare fixed OOF/full pools with labels only as posthoc supervision."""
    ids = [c["sample_id"] for c in cases]
    require(set(indices) == {"c1", "c3"}, "Need both trainval conditions")
    maps = {c: {r["sample_id"]: r for r in index["cases"]} for c, index in indices.items()}
    require(all(set(v) == set(ids) for v in maps.values()), "Trainval index coverage mismatch")
    result = {}
    for budget in BUDGETS:
        counts = {c: [] for c in maps}
        for case in cases:
            for condition in maps:
                row = maps[condition][case["sample_id"]]
                y, pool = labels_for(row, budget)
                require(np.isin(y, [0, 1]).all(), "Nonbinary diagnostic labels")
                counts[condition].append(int(y[pool].sum()))
        a, b = counts["c3"], counts["c1"]
        result[str(budget)] = dict(query_oracle_change=paired([n > 0 for n in a], [n > 0 for n in b], ids),
            c1_positive_points=sum(b), c3_positive_points=sum(a),
            more_positive_points_queries=sum(x > y for x, y in zip(a, b)),
            fewer_positive_points_queries=sum(x < y for x, y in zip(a, b)),
            equal_positive_points_queries=sum(x == y for x, y in zip(a, b)))
    return result


def report_html(result):
    esc = html.escape
    def table(headers, rows):
        return "<table><tr>"+"".join("<th>"+esc(str(h))+"</th>" for h in headers)+"</tr>"+"".join(
            "<tr>"+"".join("<td>"+esc(str(v))+"</td>" for v in row)+"</tr>" for row in rows)+"</table>"
    body = ["<h1>V13 机制诊断</h1>", "<p>读取最终评分与审计后的固定记录；本脚本不运行模型，不生成新候选，不调超参数。</p>",
            "<ul>"+"".join("<li>"+esc(v)+"</li>" for v in LIMITS.values())+"</ul>"]
    for split, cohort in result["cohorts"].items():
        body.append(f"<h2>{esc(split)} · {cohort['n']}题，独立分母</h2>")
        for kind, items in cohort["attribution"].items():
            body.append("<h3>"+esc(kind)+"</h3>")
            rows = []
            for item in items:
                for metric, values in item["metrics"].items():
                    rows.append([item["method"], item["reference"], metric, "all", values["n"], values["method_hits"],
                        values["reference_hits"], f"{values['wins']}/{values['losses']}", f"{100*values['delta']:+.2f}",
                        values["ci95_from_scored_summary"]])
                    for source, local in values.get("by_source", {}).items():
                        rows.append([item["method"], item["reference"], metric, source, local["n"], local["method_hits"],
                            local["reference_hits"], f"{local['wins']}/{local['losses']}", f"{100*local['delta']:+.2f}",
                            "见正式评分分来源区间；本脚本未重采样"])
            body.append(table(["方法", "对照", "阶段", "来源", "N", "命中/保留", "对照", "赢/输", "差值pp", "既有95%区间(率)"], rows))
        body.append("<h3>固定配额的oracle增减：c3 − c1</h3>"+table(["K", "c3", "c1", "增加", "丢失", "净差pp"],
            [[k, v["method_hits"], v["reference_hits"], v["wins"], v["losses"], f"{100*v['delta']:+.2f}"]
             for k, v in cohort["oracle_coverage_c3_minus_c1"].items()]))
        body.append("<h3>当前解码坐标并集 → quota16 → scoreTop4 → GUI</h3>"+table(
            ["方法", "并集无正确点", "配额丢失", "Top4丢失", "GUI误选", "成功"],
            [[name]+[v["stages"][s] for s in CHAIN_STAGES] for name, v in cohort["failure_chains"].items()]))
        body.append("<p>"+esc(LIMITS["raw_union"])+"</p><h3>有效目标空间覆盖与local no-name/T0</h3><pre>"+
                    esc(json.dumps(cohort["crop_geometry"], ensure_ascii=False, indent=2))+"</pre>")
        body.append("<h3>同池GUI与原图native参照</h3>"+table(["方法", "参照", "命中", "参照命中", "赢/输", "差值pp"],
            [[v["method"], v["reference"], v["method_hits"], v["reference_hits"],
              f"{v['wins']}/{v['losses']}", f"{100*v['delta']:+.2f}"] for v in cohort["same_pool_and_native_references"]]))
    for title, field in (("训练OOF及验证池正例变化（事后GT诊断）", "trainval_pool_changes"),
                         ("训练前固定epoch及来源CV曲线", "fixed_model_selection")):
        body.append("<h2>"+title+"</h2><pre>"+esc(json.dumps(result[field], ensure_ascii=False, indent=2))+"</pre>")
    body.append('<p>逐题几何、失败链、输入SHA与计数见 <a href="results.json">results.json</a>。未新增独立显著性或因果结论。</p>')
    return ('<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>V13 机制分析</title>'
        '<style>body{font:15px system-ui;max-width:1550px;margin:30px auto;padding:0 22px;color:#243246}'
        'p,li{line-height:1.7}table{border-collapse:collapse;width:100%;font-size:12px}td,th{border:1px solid #d5dce3;'
        'padding:7px;text-align:left}th{background:#edf3f7}pre{background:#f5f7fa;padding:15px;overflow:auto}</style>'+
        "".join(body)+"</html>")


def run(root):
    root = Path(root).resolve(); evidence, audit = authorize_reads(root)
    for dependency in ("src/clickagent/public_evaluation.py", "src/clickagent/spatial_geometry.py"):
        evidence.need(dependency)
    selection = evidence.json(root / "models/selection.json")
    require(selection["complete"] and not selection["fresh_test_read"] and not selection["seed_selected_by_test"],
            "Model selection was not fixed before test")
    cohorts = {}
    for split in COHORTS:
        cases = evidence.rows(root / "data" / (split + ".jsonl"))
        summary = evidence.json(root / "evaluation" / split / "summary.json")
        details = evidence.json(root / "evaluation" / split / "details.json")
        pipelines = [evidence.json(root / "pipeline" / split / (key(c["sample_id"]) + ".json")) for c in cases]
        counts = {}
        for pipeline in pipelines:
            for view in pipeline["local_views"]:
                raw = view["feature_path"]
                counts[raw] = raw_token_count(evidence.need(raw))
        cohorts[split] = analyze_cohort(cases, details, pipelines, summary, token_counts=counts)
    trainval = {}
    def labels_for(row, budget):
        with np.load(evidence.need(row["path"]), allow_pickle=False) as archive:
            require(str(archive["sample_id"]) == row["sample_id"], "Cache sample mismatch")
            return archive["labels"].astype(np.int64), archive["pool_indices_" + str(budget)].astype(np.int64)
    for split in ("train", "val"):
        cases = evidence.rows(root / "data" / (split + ".jsonl"))
        indices = {c: evidence.json(root / "cache" / f"{split}_{c}_index.json") for c in ("c1", "c3")}
        require(all(v["complete"] for v in indices.values()), "Incomplete trainval cache")
        if split == "train":
            require(all(c["oof_provider"] == f"fold_{c['image_fold']}" for c in cases), "Train pool is not OOF")
        trainval[split] = dict(provider="OOF fivefold" if split == "train" else "full-D original validation",
                              budgets=training_pool_changes(cases, indices, labels_for))
    evidence.unchanged()
    result = dict(complete=True, version=13, purpose="post-audit mechanism analysis only", limits=LIMITS,
        cohorts=cohorts, trainval_pool_changes=trainval,
        fixed_model_selection={k: selection[k] for k in ("selections", "cross_validation")},
        model_training=False, model_execution=False, hyperparameter_selection=False, gpu_used=False,
        input_sha256=evidence.used, final_audit_sha256=sha(root / "final_audit.json"),
        final_audit_frozen_sha256=audit["frozen_sha256"], analysis_source_sha256=sha(__file__),
        analysis_dependencies={str(WORKSPACE / "src/clickagent/public_evaluation.py"): sha("src/clickagent/public_evaluation.py"),
            str(WORKSPACE / "src/clickagent/spatial_geometry.py"): sha("src/clickagent/spatial_geometry.py")},
        library_versions=dict(numpy=np.__version__, shapely=shapely.__version__))
    out = root / "evaluation/mechanism_analysis"; out.mkdir(parents=True, exist_ok=True)
    target = out / "results.json"; text = json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False)+"\n"
    if target.exists():
        require(read(target) == result, "Existing mechanism analysis differs; preserve it and use a separate version")
    else:
        temporary = target.with_suffix(".json.tmp"); temporary.write_text(text, encoding="utf-8"); temporary.replace(target)
    page = out / "report.html"; rendered = report_html(result)
    if page.exists():
        require(page.read_text(encoding="utf-8") == rendered, "Existing mechanism report differs")
    else:
        page.write_text(rendered, encoding="utf-8")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    args = parser.parse_args()
    result = run(args.root)
    print(json.dumps(dict(complete=result["complete"], model_execution=False,
        output=str(args.root / "evaluation/mechanism_analysis"))))


if __name__ == "__main__":
    main()
