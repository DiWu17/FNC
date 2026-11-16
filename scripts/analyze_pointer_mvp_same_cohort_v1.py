"""Analyze sealed same-cohort PointerBench predictions without running models.

All public-prediction SHA and V2 integrity checks run before this module opens
the PointerBench labels. This is retrospective same-sample evidence: V2 labels
were already analyzed and several public answers were previewed before sealing.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]

import numpy as np

import analyze_paper_scale_v2 as v2
import run_pointer_mvp_same_cohort_v1 as runner
from clickagent.public_evaluation import point_in_geometry

RUN = ROOT / "runs/pointer_mvp_same_cohort_v1"
V2 = ROOT / "runs/paper_scale_ablation_v2"
PUBLIC = V2 / "pointerbench/public.jsonl"
PROTOCOL = RUN / "protocol.json"
REGISTRATION = RUN / "analysis_registration.json"
COMPLETION = RUN / "public_completion.json"
LABELS = V2 / "pointerbench/labels.jsonl"
OUT = RUN / "analysis"
METHODS = ("mvp", "qwen")
SCALES = ("full", "3k")
SEEDS = (1, 2, 3)
COHORTS = ("pro", "sheets")
N = 1000
REPLICATES = 10_000
BOOTSTRAP_SEED = 20261112


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def jsonl(path: Path):
    with path.open("r", encoding="utf-8-sig") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_once(path: Path, payload, *, plain_text: bool = False) -> None:
    content = payload if plain_text else json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    if path.exists():
        require(path.read_text(encoding="utf-8") == content,
                "Existing immutable analysis output differs: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    require(not temporary.exists(), "Unresolved temporary output: " + str(temporary))
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def _point(point, label: str):
    if point is None:
        return None
    require(type(point) is list and len(point) == 2 and
            all(type(value) is int and 0 <= value < size
                for value, size in zip(point, (1024, 768))),
            "Invalid native integer point: " + label)
    return point


def _case_path(method: str, sample_id: str) -> Path:
    return RUN / "predictions" / method / "cases" / (
        hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:24] + ".json")


def public_identity(row: dict) -> str:
    return hashlib.sha256(json.dumps(row, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode("utf-8")).hexdigest()


def _verify_external_bundle(method: str, public_rows: list[dict],
                            receipt: dict, protocol_hash: str, public_hash: str):
    path = RUN / "predictions" / f"{method}.json"
    require(path.exists() and receipt["prediction_sha256"][method] == sha(path),
            "External prediction bundle SHA mismatch: " + method)
    bundle = read(path)
    require(bundle.get("complete") is True and bundle.get("method") == method and
            bundle.get("n") == N and len(bundle.get("rows", [])) == N and
            bundle.get("labels_opened", False) is False,
            "External prediction bundle incomplete: " + method)
    if "protocol_sha256" in bundle:
        require(bundle["protocol_sha256"] == protocol_hash,
                "External prediction protocol mismatch: " + method)
    if "public_source_sha256" in bundle:
        require(bundle["public_source_sha256"] == public_hash,
                "External prediction public-source mismatch: " + method)
    hashes = bundle.get("case_hashes")
    require(type(hashes) is dict and set(hashes) ==
            {row["sample_id"] for row in public_rows},
            "External prediction case-hash coverage mismatch: " + method)
    outputs = []
    for index, (public, row) in enumerate(zip(public_rows, bundle["rows"])):
        sid = public["sample_id"]
        require(type(row) is dict and row.get("sample_id") == sid and
                row.get("method") == method and
                row.get("image_sha256") == public["image_sha256"] and
                row.get("pixel_sha256") == public["pixel_sha256"] and
                row.get("suite") == public["suite"] and
                row.get("public_identity") == public_identity(public) and
                row.get("protocol_sha256") == protocol_hash and
                row.get("labels_opened") is False,
                f"External case order/image mismatch: {method}/{index}")
        point = _point(row.get("point"), method + "/" + sid)
        for field in ("elapsed_seconds", "backbone_forward_calls",
                      "processed_pixels"):
            require(field in row and type(row[field]) in (int, float) and
                    math.isfinite(row[field]) and row[field] >= 0,
                    "Missing or invalid cost counter: " + method + "/" + sid + "/" + field)
        peak = row.get("peak_allocated_bytes")
        require(type(peak) is list and len(peak) >= 1 and
                all(type(value) is int and value >= 0 for value in peak),
                "Missing or invalid GPU peak-memory counters: " + method + "/" + sid)
        case_path = _case_path(method, sid)
        require(case_path.exists() and sha(case_path) == hashes[sid],
                "External per-case SHA mismatch: " + method + "/" + sid)
        case = read(case_path)
        require(case == row,
                "External per-case/bundle mismatch: " + method + "/" + sid)
        outputs.append(row)
    return outputs, sha(path)


def gate():
    """Public-only verification; never opens PointerBench ground truth."""
    require(PROTOCOL.exists() and PUBLIC.exists() and REGISTRATION.exists(),
            "Protocol, public cohort, or analysis registration missing")
    protocol_hash, public_hash = sha(PROTOCOL), sha(PUBLIC)
    registration = read(REGISTRATION)
    require(registration.get("complete") is True and
            registration.get("analysis_script_sha256") == sha(Path(__file__)) and
            registration.get("runner_protocol_sha256") == protocol_hash and
            registration.get("public_source_sha256") == public_hash and
            registration.get("v2_pointer_completion_sha256") ==
            sha(V2 / "pointerbench/public_completion.json") and
            registration.get("primary_contrast") ==
            "Pro500: full_aligned_three_seed_mean minus MVP" and
            registration.get("bootstrap_replicates") == REPLICATES and
            registration.get("bootstrap_seed") == BOOTSTRAP_SEED and
            registration.get("bootstrap_unit") == "RGB image pixel_sha256" and
            registration.get("strict_blind") is False and
            registration.get("retrospective_same_sample") is True,
            "Analysis registration or frozen analysis script changed")
    for cohort in COHORTS:
        official_path = V2 / "pointerbench/source" / cohort / "eval.py"
        require(registration.get("official_scorer_sha256", {}).get(cohort) ==
                sha(official_path),
                "Analysis registration official scorer SHA changed: " + cohort)
    require(COMPLETION.exists(), "External public completion receipt missing")
    protocol, receipt = read(PROTOCOL), read(COMPLETION)
    require(runner.protocol(deep_model=False) == protocol,
            "Registered public/model-source integrity changed")
    require(protocol.get("complete") is True and receipt.get("complete") is True and
            receipt.get("n") == N and receipt.get("labels_opened") is False and
            receipt.get("protocol_sha256") == protocol_hash and
            receipt.get("public_source_sha256") == public_hash and
            set(receipt.get("prediction_sha256", {})) == set(METHODS),
            "New comparison protocol/public receipt mismatch")
    require(protocol.get("methods") == list(METHODS) and
            protocol.get("total_n") == N and
            protocol.get("primary") == "PointerBench Pro500" and
            protocol.get("secondary") == "PointerBench Sheets500" and
            protocol.get("bootstrap_replicates") == REPLICATES and
            protocol.get("bootstrap_seed") == BOOTSTRAP_SEED and
            protocol.get("bootstrap_unit") == "RGB image pixel_sha256" and
            protocol.get("primary_contrast") ==
            "full_aligned_three_seed_mean_vs_mvp_on_Pro500" and
            protocol.get("public_source_sha256") == public_hash,
            "Registered comparison/statistical plan changed")
    public_rows = jsonl(PUBLIC)
    require(len(public_rows) == N and
            [row["sample_id"] for row in public_rows] ==
            [row["sample_id"] for row in jsonl(V2 / "pointerbench/public.jsonl")],
            "New public cohort membership/order differs from sealed V2 PointerBench")
    require(sum(row["suite"] == "pro" for row in public_rows) == 500 and
            sum(row["suite"] == "sheets" for row in public_rows) == 500 and
            all(row["image_size"] == [1024, 768] for row in public_rows),
            "Expected Pro500 and Sheets500 at native 1024x768")
    # The V2 gate SHA-verifies every original per-case prediction and all three
    # completion receipts before returning, including the six aligned seeds here.
    v2_registration, v2_bundles, _ = v2.gate()
    provenance_path = V2 / "pointerbench/provenance.json"
    pointer_protocol = read(V2 / "pointerbench/evaluation_protocol_v2.json")
    require(sha(provenance_path) == pointer_protocol["provenance_sha256"],
            "Official PointerBench provenance SHA drift")
    provenance = read(provenance_path)
    for cohort in COHORTS:
        official_path = V2 / "pointerbench/source" / cohort / "eval.py"
        official_key = f"source\\{cohort}\\eval.py"
        require(sha(official_path) == provenance["source_sha256"][official_key],
                "Pinned official PointerBench scorer changed: " + cohort)
    for cohort in COHORTS:
        sealed_rows, sealed_cases, _ = v2_bundles[cohort]
        selected = [row for row in public_rows if row["suite"] == cohort]
        require(len(selected) == len(sealed_rows) == 500,
                "Cohort size differs from V2: " + cohort)
        for new, old in zip(selected, sealed_rows):
            require(all(new.get(field) == old.get(field) for field in old),
                    "Public case changed from V2: " + old["sample_id"])
            require(set(sealed_cases[old["sample_id"]]["methods"]) ==
                    set(v2_registration["model_hashes"]),
                    "V2 frozen model case missing: " + old["sample_id"])
    external = {method: _verify_external_bundle(
        method, public_rows, receipt, protocol_hash, public_hash)
        for method in METHODS}
    return protocol, public_rows, v2_bundles, external,


def open_labels_after_gate(protocol, public_rows):
    """Only call after gate() has verified all original and external seals."""
    expected = read(V2 / "pointerbench/evaluation_protocol_v2.json")[
        "labels_source_sha256_unopened"]
    require(sha(LABELS) == expected, "PointerBench label SHA changed")
    labels = jsonl(LABELS)
    require(len(labels) == N and
            {row["sample_id"] for row in labels} ==
            {row["sample_id"] for row in public_rows},
            "PointerBench public/label IDs differ")
    return {row["sample_id"]: row for row in labels}


def official_hit(point, bbox):
    if point is None:
        return False
    x0, y0, x1, y1 = bbox
    return min(x0, x1) <= point[0] <= max(x0, x1) and \
        min(y0, y1) <= point[1] <= max(y0, y1)


def load_official_scorer(cohort):
    """Load the released PointerBench evaluator without invoking its GT reader."""
    path = V2 / "pointerbench/source" / cohort / "eval.py"
    spec = importlib.util.spec_from_file_location("pointerbench_" + cohort + "_eval", path)
    require(spec is not None and spec.loader is not None,
            "Official PointerBench evaluator unavailable: " + cohort)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(callable(getattr(module, "_in_bbox", None)),
            "Official PointerBench point scorer missing: " + cohort)
    return module._in_bbox, sha(path)


def hit_with_parity(point, label, sample_id, scorer):
    native = bool(point is not None and scorer(tuple(point), label["bbox_pixels"]))
    require(native == official_hit(point, label["bbox_pixels"]),
            "Official/source native scorer parity failure: " + sample_id)
    normalized = bool(point is not None and point_in_geometry(
        [point[0] / 1024, point[1] / 768], label["target_geometry"]))
    require(native == normalized, "Official/V2 scorer parity failure: " + sample_id)
    return native


def paired_cluster_ci(left, right, rows, draws):
    delta = np.asarray(left, dtype=np.float64) - np.asarray(right, dtype=np.float64)
    by_rgb = defaultdict(list)
    for index, row in enumerate(rows):
        by_rgb[row["pixel_sha256"]].append(index)
    clusters = list(by_rgb.values())
    numer = np.asarray([delta[indices].sum() for indices in clusters])
    denom = np.asarray([len(indices) for indices in clusters])
    sampled = (draws @ numer) / (draws @ denom)
    return dict(delta=float(delta.mean()), ci95=np.quantile(
        sampled, [0.025, 0.975]).astype(float).tolist(),
        cluster_count=len(clusters))


def analyze_cohort(cohort, public_rows, v2_bundles, external_rows, labels):
    rows = [row for row in public_rows if row["suite"] == cohort]
    require(len(rows) == 500, "Cohort size drift: " + cohort)
    scorer, scorer_sha = load_official_scorer(cohort)
    by_external = {name: {row["sample_id"]: row for row in external_rows[name]}
                   for name in METHODS}
    sealed_cases = v2_bundles[cohort][1]
    hits = {}
    for method in METHODS:
        hits[method] = np.asarray([
            hit_with_parity(by_external[method][row["sample_id"]]["point"],
                            labels[row["sample_id"]], row["sample_id"], scorer)
            for row in rows], dtype=np.float64)
    for scale in SCALES:
        for seed in SEEDS:
            name = f"{scale}_aligned_s{seed}"
            hits[name] = np.asarray([
                hit_with_parity(sealed_cases[row["sample_id"]]["methods"][name]["point"],
                                labels[row["sample_id"]], row["sample_id"], scorer)
                for row in rows], dtype=np.float64)
        hits[f"{scale}_aligned_mean"] = np.mean(
            [hits[f"{scale}_aligned_s{seed}"] for seed in SEEDS], axis=0)
    clusters = len({row["pixel_sha256"] for row in rows})
    rng = np.random.default_rng(BOOTSTRAP_SEED + (0 if cohort == "pro" else 1))
    draws = rng.multinomial(clusters, np.full(clusters, 1 / clusters),
                            size=REPLICATES).astype(np.float32)
    contrasts = {}
    for scale in SCALES:
        for baseline in METHODS:
            ours = hits[f"{scale}_aligned_mean"]
            other = hits[baseline]
            contrast = paired_cluster_ci(ours, other, rows, draws)
            contrast["mean_seed_wins"] = int(np.sum(ours > other))
            contrast["mean_seed_losses"] = int(np.sum(ours < other))
            contrast["mean_seed_ties"] = int(np.sum(ours == other))
            contrast["seed_details"] = {}
            for seed in SEEDS:
                single = hits[f"{scale}_aligned_s{seed}"]
                contrast["seed_details"][str(seed)] = dict(
                    accuracy=float(np.mean(single)),
                    delta=float(np.mean(single - other)),
                    wins=int(np.sum((single == 1) & (other == 0))),
                    losses=int(np.sum((single == 0) & (other == 1))))
            contrasts[f"{scale}_vs_{baseline}"] = contrast
    baseline_costs = {}
    for method in METHODS:
        records = [by_external[method][row["sample_id"]] for row in rows]
        baseline_costs[method] = {
            "mean_seconds": float(np.mean([r["elapsed_seconds"] for r in records])),
            "median_seconds": float(np.median([r["elapsed_seconds"] for r in records])),
            "mean_forward_calls": float(np.mean([r["backbone_forward_calls"] for r in records])),
            "mean_processed_pixels": float(np.mean([r["processed_pixels"] for r in records])),
            "max_peak_allocated_bytes_per_gpu": [
                int(max(record["peak_allocated_bytes"][device] for record in records))
                for device in range(len(records[0]["peak_allocated_bytes"]))],
        }
    details = [dict(sample_id=row["sample_id"], pixel_sha256=row["pixel_sha256"],
                    hits={name: float(vector[index]) for name, vector in hits.items()})
               for index, row in enumerate(rows)]
    report = dict(complete=True, cohort=cohort, n=len(rows), rgb_clusters=clusters,
                  metric="Official native inclusive point-in-bbox, 1024x768",
                  official_scorer_sha256=scorer_sha,
                  scorer_parity_checked_points=len(rows) * (len(METHODS) +
                                                               len(SCALES) * len(SEEDS)),
                  accuracies={name: float(np.mean(values)) for name, values in hits.items()},
                  contrasts=contrasts, baseline_runtime=baseline_costs,
                  published_context_only={"mvp_8b": 0.708 if cohort == "pro" else 0.604,
                                          "qwen3_vl_8b": 0.658 if cohort == "pro" else 0.482},
                  retrospective_same_sample=True, strict_blind=False,
                  our_v2_runtime_comparable=False)
    return report, details


def render(results):
    lines = [
        "PointerBench same-cohort comparison (retrospective; not a new blind test)",
        "Same local Qwen3-VL-8B backbone; frozen V2 3k/full aligned predictions versus local MVP and native Qwen.",
        "Official native 1024x768 inclusive point-in-bbox; invalid or absent point counted wrong.",
        "Pro500 primary, Sheets500 secondary; all pre-fixed cases kept. RGB-image-cluster paired bootstrap, 10,000 draws; descriptive 95% CIs.",
        "V2 labels and results were analyzed before this comparison. At least seven PointerBench answers were previewed before V2 prediction sealing; exact count unknown. No strict blind or prospective superiority claim.",
        "Published PointerBench MVP-8B / Qwen3-VL-8B values (https://arxiv.org/html/2610.31274, Table 15) are context, not paired local scores or official reproduction.",
        "Runtime below covers only new baselines. V2 did not record matched per-case end-to-end timing; no direct speed claim.",
        "",
    ]
    for cohort in COHORTS:
        item = results[cohort]
        lines.append(f"{cohort.upper()} n=500, RGB clusters={item['rgb_clusters']}")
        for method in METHODS:
            context_key = "mvp_8b" if method == "mvp" else "qwen3_vl_8b"
            lines.append(f"{method}: {100*item['accuracies'][method]:.2f}% "
                         f"(published context {100*item['published_context_only'][context_key]:.1f}%)")
        for scale in SCALES:
            counts = [round(500 * item["accuracies"][f"{scale}_aligned_s{seed}"])
                      for seed in SEEDS]
            lines.append(f"{scale} aligned: seed hits {counts}; mean "
                         f"{100*item['accuracies'][f'{scale}_aligned_mean']:.2f}%")
            for baseline in METHODS:
                c = item["contrasts"][f"{scale}_vs_{baseline}"]
                lo, hi = c["ci95"]
                lines.append(f"  vs {baseline}: delta {100*c['delta']:+.2f} pp "
                             f"[95% {100*lo:+.2f}, {100*hi:+.2f}]; "
                             f"mean-score case wins/losses/ties "
                             f"{c['mean_seed_wins']}/{c['mean_seed_losses']}/{c['mean_seed_ties']}")
                for seed in SEEDS:
                    s = c["seed_details"][str(seed)]
                    lines.append(f"    seed{seed}: {100*s['accuracy']:.2f}%, "
                                 f"delta {100*s['delta']:+.2f} pp, "
                                 f"wins/losses {s['wins']}/{s['losses']}")
        for baseline in METHODS:
            cost = item["baseline_runtime"][baseline]
            lines.append(f"{baseline} runtime: mean/median "
                         f"{cost['mean_seconds']:.2f}/{cost['median_seconds']:.2f}s, "
                         f"mean forwards {cost['mean_forward_calls']:.2f}, "
                         f"mean processed pixels {cost['mean_processed_pixels']:.0f}")
        lines.append("")
    return "\n".join(lines) + "\n"


def analyze():
    protocol, public_rows, v2_bundles, external = gate()
    labels = open_labels_after_gate(protocol, public_rows)
    external_rows = {name: external[name][0] for name in METHODS}
    reports = {}
    for cohort in COHORTS:
        report, details = analyze_cohort(cohort, public_rows, v2_bundles,
                                         external_rows, labels)
        write_once(OUT / f"{cohort}.json", report)
        write_once(OUT / f"{cohort}_details.json", details)
        reports[cohort] = report
    write_once(OUT / "readout.txt", render(reports), plain_text=True)
    summary = dict(complete=True, protocol_sha256=sha(PROTOCOL),
                   analysis_registration_sha256=sha(REGISTRATION),
                   external_prediction_sha256={name: external[name][1] for name in METHODS},
                   v2_prediction_sha256={cohort: v2_bundles[cohort][2] for cohort in COHORTS},
                   labels_sha256=sha(LABELS),
                   result_sha256={cohort: sha(OUT / f"{cohort}.json") for cohort in COHORTS},
                   readout_sha256=sha(OUT / "readout.txt"),
                   retrospective_same_sample=True, strict_blind=False)
    write_once(OUT / "summary.json", summary)
    return summary


def self_test():
    box = [100, 50, 200, 150]
    assert official_hit([100, 50], box)
    assert official_hit([200, 150], box)
    assert not official_hit([201, 150], box)
    assert not official_hit(None, box)
    rows = [{"pixel_sha256": "a"}, {"pixel_sha256": "a"},
            {"pixel_sha256": "b"}]
    rng = np.random.default_rng(3)
    draws = rng.multinomial(2, [0.5, 0.5], size=100).astype(np.float32)
    result = paired_cluster_ci([1, 1, 0], [0, 0, 0], rows, draws)
    assert result["cluster_count"] == 2 and math.isclose(result["delta"], 2 / 3)
    print(json.dumps({"synthetic_self_test": "passed", "real_labels_opened": False}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=("self-test", "verify-gate", "analyze"))
    args = parser.parse_args()
    if args.stage == "self-test":
        self_test()
    elif args.stage == "verify-gate":
        protocol, rows, bundles, external = gate()
        print(json.dumps({"gate": "passed", "n": len(rows),
                          "external_sha256": {name: external[name][1]
                                                for name in METHODS},
                          "gt_opened_by_analysis": False}))
    else:
        print(json.dumps(analyze(), ensure_ascii=False))


if __name__ == "__main__":
    main()
