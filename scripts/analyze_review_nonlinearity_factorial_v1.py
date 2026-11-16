"""Registered post-seal analysis of the pooled-feature 2x2 scorer control."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import run_review_nonlinearity_factorial_v1 as worker
import analyze_review_w3_simple_readout_v1 as w3_analysis

OUT = worker.OUT
REG = OUT / "analysis_registration.json"
PUBLIC = w3_analysis.PUBLIC
N = w3_analysis.N
POINTER_LABELS = w3_analysis.POINTER_LABELS
MM_LABELS = w3_analysis.MM_LABELS
SCORER = w3_analysis.SCORER
POINTER_SCORER = w3_analysis.POINTER_SCORER
REPS = 10000
BOOT_SEED = 20261115


def write_once(path: Path, value) -> None:
    if path.exists():
        worker.require(worker.read(path) == value, "Immutable analysis output drift: " + str(path))
    else:
        worker.write(path, value)


def register() -> None:
    worker.gate()
    worker.require(not (OUT / "predictions/completion.json").exists(),
                   "Analysis registration must precede new prediction seal")
    source = {"public": {suite: worker.sha(path) for suite, path in PUBLIC.items()},
              "w3_bundle": {suite: worker.sha(worker.W3 / f"predictions/{suite}/bundle.json")
                            for suite in PUBLIC},
              "w3_summary": worker.sha(worker.W3 / "analysis/summary.json"),
              "pointer_scorer": {suite: worker.sha(path) for suite, path in POINTER_SCORER.items()},
              "mmbench_scorer": worker.sha(SCORER),
              "pointer_labels_expected_from_w3": worker.read(worker.W3 / "analysis_registration_v3.json")["label_sha256_unopened"]["pointer"],
              "mmbench_labels_expected_from_w3": worker.read(worker.W3 / "analysis_registration_v3.json")["label_sha256_unopened"]["mmbench"]}
    record = {
        "registered_at_utc": worker.now(),
        "analysis_script_sha256": worker.sha(Path(__file__)),
        "worker_script_sha256": worker.sha(Path(worker.__file__)),
        "worker_protocol_sha256": worker.sha(OUT / "protocol.json"),
        "source_sha256": source,
        "cohorts": {"pro": 500, "sheets": 500, "mmbench": 1000},
        "arms": list(worker.ALL_ARMS),
        "main_contrast": "pooled_mlp_all minus linear_all on Pro500",
        "secondary_contrasts": "Same architecture contrast on Sheets500 and MMBench1000; base32 architecture contrast, within-architecture normalization contrasts and interaction are descriptive",
        "metric": "Original K16 forced stable Top1 point-in-target; pinned official scorer real-annotation parity for all new and W3 anchor predictions",
        "interval": {"unit": "exact RGB pixel SHA cluster", "query_weighted": True,
                     "draws": REPS, "rng_seed": BOOT_SEED,
                     "primary": "percentile paired 95% CI, three-seed mean",
                     "across_three_cohorts": "Bonferroni 98.3333% paired CI for main contrast",
                     "other": "descriptive 95% paired CIs; conditional on three fixed seeds"},
        "label_access": "Never parse labels before all nine new models on all 2000 cases and W3 anchor pass SHA and same-feature gate",
        "limitations": ["Existing test labels/results had been analyzed before this retrospective control",
                        "At least seven PointerBench answers were previously visible",
                        "MMBench is selected project-history image-disjoint subset, not official full leaderboard",
                        "Archived original full_aligned model has pre-pooling state encoder and is external reference only"],
    }
    if REG.exists():
        record["registered_at_utc"] = worker.read(REG)["registered_at_utc"]
    write_once(REG, record)
    print(json.dumps({"stage": "analysis_registered", "registration_sha256": worker.sha(REG)}), flush=True)


def gate() -> dict:
    worker.verify_gate()
    worker.require(REG.exists(), "Analysis registration missing")
    reg = worker.read(REG)
    worker.require(reg["analysis_script_sha256"] == worker.sha(Path(__file__)) and
                   reg["worker_script_sha256"] == worker.sha(Path(worker.__file__)) and
                   reg["worker_protocol_sha256"] == worker.sha(OUT / "protocol.json"),
                   "Analysis source/protocol SHA drift")
    source = reg["source_sha256"]
    worker.require(source["w3_summary"] == worker.sha(worker.W3 / "analysis/summary.json") and
                   source["mmbench_scorer"] == worker.sha(SCORER),
                   "Registered source hash drift")
    for suite in PUBLIC:
        worker.require(source["public"][suite] == worker.sha(PUBLIC[suite]) and
                       source["w3_bundle"][suite] == worker.sha(worker.W3 / f"predictions/{suite}/bundle.json"),
                       "Public or W3 bundle SHA drift")
    for suite, path in POINTER_SCORER.items():
        worker.require(source["pointer_scorer"][suite] == worker.sha(path),
                       "Pinned PointerBench scorer drift")
    return reg


def load_labels_after_gate():
    source = worker.read(REG)["source_sha256"]
    worker.require(source["pointer_labels_expected_from_w3"] == worker.sha(POINTER_LABELS) and
                   source["mmbench_labels_expected_from_w3"] == worker.sha(MM_LABELS),
                   "Pinned label bytes drift after gate")
    pointer = [json.loads(line) for line in POINTER_LABELS.read_text(encoding="utf-8-sig").splitlines()
               if line.strip()]
    worker.require(len(pointer) == 1000, "PointerBench label count drift")
    pointer_by_id = {row["sample_id"]: row for row in pointer}
    worker.require(len(pointer_by_id) == 1000, "PointerBench label ID duplication")
    mm = worker.read(MM_LABELS)
    worker.require(isinstance(mm, list) and len(mm) == 3594, "MMBench annotation count drift")
    mm_by_index = {row["index"]: row for row in mm}
    worker.require(len(mm_by_index) == len(mm), "MMBench source index duplicate")
    return pointer_by_id, mm_by_index


def hits_for_suite(suite: str, pointer: dict, mm_labels: dict):
    import numpy as np
    from clickagent.public_evaluation import point_in_geometry
    from run_mmbench_gui_l2_v1 import official_hit, source_reference_hit
    rows = worker.read(PUBLIC[suite])
    if suite == "mmbench":
        native_scorer = source_reference_hit()
    else:
        spec = importlib.util.spec_from_file_location(f"factorial_{suite}_official", POINTER_SCORER[suite])
        worker.require(spec is not None and spec.loader is not None, "Pointer official scorer import failed")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        native_scorer = getattr(module, "_in_bbox", None)
        worker.require(callable(native_scorer), "Pointer official _in_bbox missing")
    names = [f"{arm}_s{s}" for arm in worker.ALL_ARMS for s in (1, 2, 3)]
    hits = {name: np.zeros(len(rows), np.float32) for name in names}
    parity_nonnull = 0
    for i, row in enumerate(rows):
        sid = row["sample_id"]
        key = hashlib.sha256(sid.encode()).hexdigest()[:24]
        new = worker.read(OUT / f"predictions/{suite}/cases/{key}.json")
        anchor = worker.read(worker.W3 / f"predictions/{suite}/cases/{key}.json")
        for name in names:
            point = (anchor["predictions"][name]["point"] if name.startswith("linear_all_")
                     else new["predictions"][name]["point"])
            if suite == "mmbench":
                source = mm_labels[row["source_index"]]
                worker.require(all(source[field] == row[field]
                                   for field in ("instruction", "image_size", "data_type", "platform", "grounding_type")),
                               "MMBench public/annotation identity drift")
                hit = official_hit(point, source["bbox"], row["image_size"])
                native = (bool(native_scorer(source["bbox"].copy(), tuple(row["image_size"]), tuple(point)))
                          if point is not None else False)
            else:
                width, height = row["image_size"]
                hit = point is not None and point_in_geometry(
                    (point[0] / width, point[1] / height), pointer[sid]["target_geometry"])
                native = (bool(native_scorer(tuple(point), pointer[sid]["bbox_pixels"]))
                          if point is not None else False)
            worker.require(bool(hit) == native,
                           "Pinned official scorer parity failure: " + suite + "/" + sid + "/" + name)
            hits[name][i] = bool(hit)
            parity_nonnull += point is not None
    return rows, hits, {"passed": True, "nonnull_points": parity_nonnull,
                        "all_points_checked": len(rows) * len(names)}


def summarize(rows: list, hits: dict) -> dict:
    import numpy as np
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["pixel_sha256"]].append(i)
    cluster = list(groups.values())
    rng = np.random.default_rng(BOOT_SEED)
    draw = rng.multinomial(len(cluster), np.full(len(cluster), 1 / len(cluster)),
                           size=REPS).astype(np.float32)
    den = draw @ np.asarray([len(ix) for ix in cluster], np.float32)
    by_arm = {arm: np.stack([hits[f"{arm}_s{s}"] for s in (1, 2, 3)])
              for arm in worker.ALL_ARMS}
    arms = {arm: {"seed_hits": x.sum(axis=1).astype(int).tolist(),
                  "seed_accuracy": x.mean(axis=1).astype(float).tolist(),
                  "mean_accuracy": float(x.mean())}
            for arm, x in by_arm.items()}

    def compare(a, b):
        diff = by_arm[a] - by_arm[b]
        per_case = diff.mean(axis=0)
        numer = np.asarray([per_case[ix].sum() for ix in cluster], np.float32)
        samples = (draw @ numer) / den
        return {"delta_pp": float(100 * per_case.mean()),
                "per_seed_delta_pp": (100 * diff.mean(axis=1)).astype(float).tolist(),
                "ci95_pp": (100 * np.quantile(samples, [0.025, 0.975])).astype(float).tolist(),
                "ci_familywise_98_33_pp": (100 * np.quantile(samples, [1/120, 1-1/120])).astype(float).tolist(),
                "wins_mean": float(((by_arm[a] == 1) & (by_arm[b] == 0)).sum(axis=1).mean()),
                "losses_mean": float(((by_arm[a] == 0) & (by_arm[b] == 1)).sum(axis=1).mean())}

    comparisons = {
        "mlp_minus_linear_allnorm": compare("pooled_mlp_all", "linear_all"),
        "mlp_minus_linear_base32norm": compare("pooled_mlp_base32", "linear_base32"),
        "linear_allnorm_minus_base32norm": compare("linear_all", "linear_base32"),
        "mlp_allnorm_minus_base32norm": compare("pooled_mlp_all", "pooled_mlp_base32"),
    }
    # Difference-in-differences: how much normalizing all features changes the
    # MLP-versus-linear gap. This is descriptive, not a prospectively powered claim.
    did = ((by_arm["pooled_mlp_all"] - by_arm["linear_all"]) -
           (by_arm["pooled_mlp_base32"] - by_arm["linear_base32"]))
    per_case = did.mean(axis=0)
    numer = np.asarray([per_case[ix].sum() for ix in cluster], np.float32)
    samples = (draw @ numer) / den
    comparisons["normalization_by_architecture_interaction"] = {
        "delta_pp": float(100 * per_case.mean()),
        "per_seed_delta_pp": (100 * did.mean(axis=1)).astype(float).tolist(),
        "ci95_pp": (100 * np.quantile(samples, [0.025, 0.975])).astype(float).tolist()}
    return {"n": len(rows), "rgb_clusters": len(cluster), "bootstrap_draws": REPS,
            "arms": arms, "contrasts": comparisons}


def analyze() -> None:
    reg = gate()
    pointer, mm_labels = load_labels_after_gate()
    result = {}
    for suite in PUBLIC:
        rows, hits, parity = hits_for_suite(suite, pointer, mm_labels)
        result[suite] = summarize(rows, hits)
        result[suite]["official_scorer_parity"] = parity
    output = {"complete": True, "analysis_registration_sha256": worker.sha(REG),
              "prediction_completion_sha256": worker.sha(OUT / "predictions/completion.json"),
              "cohorts": result, "retrospective": True,
              "archived_full_aligned_external_reference_sha256": reg["source_sha256"]["w3_summary"]}
    out = OUT / "analysis/summary.json"
    write_once(out, output)
    lines = ["Pooled feature architecture x normalization 2x2 (retrospective)",
             "All four arms read the same W3 1208-d pooled features and original K16.",
             "Original full_aligned token-prepooling MLP is external reference only.",
             "Three fixed final-step seeds, no test-based model selection."]
    for suite, item in result.items():
        lines += ["", f"{suite}: n={item['n']}, RGB screenshot clusters={item['rgb_clusters']}"]
        for arm, value in item["arms"].items():
            lines.append(f"  {arm}: seed_hits={value['seed_hits']}, mean={100*value['mean_accuracy']:.2f}%")
        for name, value in item["contrasts"].items():
            a, b = value["ci95_pp"]
            lines.append(f"  {name}: {value['delta_pp']:+.2f} pp, CI95 [{a:+.2f},{b:+.2f}]")
            if name == "mlp_minus_linear_allnorm":
                a, b = value["ci_familywise_98_33_pp"]
                lines.append(f"    three-cohort familywise CI98.33 [{a:+.2f},{b:+.2f}]")
    lines += ["", "Every nonnull new and anchor point passed pinned official scorer parity.",
              "All contrasts condition on three fixed trained seeds; clusters are exact screenshot RGB hashes.",
              "Retrospective: Pointer labels and at least seven preview answers were known earlier; MMBench is a selected project-history-disjoint subset.",
              "This test isolates nonlinearity after a shared pooled feature map; it does not test the original model's learned pre-pooling encoder in isolation."]
    text = "\n".join(lines) + "\n"
    readout = OUT / "analysis/readout.txt"
    if readout.exists():
        worker.require(readout.read_text(encoding="utf-8") == text, "Immutable readout drift")
    else:
        readout.parent.mkdir(parents=True, exist_ok=True)
        readout.write_text(text, encoding="utf-8")
    print(json.dumps({"stage": "analyzed", "summary_sha256": worker.sha(out),
                      "readout_sha256": worker.sha(readout)}), flush=True)


if __name__ == "__main__":
    worker.w3.low_priority()
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=("register", "verify-gate", "analyze"))
    stage = parser.parse_args().stage
    if stage == "register": register()
    elif stage == "verify-gate":
        gate()
        print(json.dumps({"stage": "verify_gate_pass", "labels_opened": False}), flush=True)
    else: analyze()
