"""Registered, label-gated analysis for same-candidate simple learned readouts.

The register and verify-gate stages inspect public metadata and SHA receipts
only. The analyze stage opens target geometry after all six new models' 2,000
predictions and the archived comparison predictions have passed the SHA gate.
"""
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
import run_review_w3_simple_readout_v1 as worker

OUT = worker.OUT
REG = OUT / "analysis_registration_v3.json"
SUPERSEDED_REG = OUT / "analysis_registration_v2.json"
AR = ROOT / "runs/paper_scale_ablation_v2/eval"
MM = ROOT / "runs/mmbench_gui_new_source_v1"
PUBLIC = {"pro": AR / "pro/public.json", "sheets": AR / "sheets/public.json",
          "mmbench": MM / "public.json"}
ARCHIVE = {"pro": AR / "pro/predictions.json", "sheets": AR / "sheets/predictions.json",
           "mmbench": MM / "mmbench/predictions.json"}
MM_ATTENTION = ROOT / "runs/mmbench_full_ablation_supplement_v1/predictions.json"
POINTER_LABELS = ROOT / "runs/paper_scale_ablation_v2/pointerbench/labels.jsonl"
MM_LABELS = ROOT / "data/mmbench_gui_l2_v1/source/L2_annotations.json"
SCORER = MM / "official_l2_evaluator.py"
POINTER_SCORER = {suite: ROOT / f"runs/paper_scale_ablation_v2/pointerbench/source/{suite}/eval.py"
                  for suite in ("pro", "sheets")}
BASELINES = {"pro": ("full_aligned", "full_attention_only"),
             "sheets": ("full_aligned", "full_attention_only"),
             "mmbench": ("full_aligned", "full_attention_only")}
KINDS = worker.MODEL_NAMES
N = {"pro": 500, "sheets": 500, "mmbench": 1000}
REPS = 10000
BOOT_SEED = 20261114


def write_once(path: Path, value) -> None:
    if path.exists():
        worker.require(worker.read(path) == value, "Immutable output drift: " + str(path))
    else:
        worker.write(path, value)


def text_once(path: Path, value: str) -> None:
    if path.exists():
        worker.require(path.read_text(encoding="utf-8") == value,
                       "Immutable text drift: " + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(value, encoding="utf-8")
        tmp.replace(path)


def register() -> None:
    protocol = worker.gate()
    worker.require(not (OUT / "predictions/completion.json").exists(),
                   "Analysis registration must precede new prediction completion")
    worker.require(SUPERSEDED_REG.exists(), "Retain the superseded pre-prediction registration")
    supplement = worker.read(MM_ATTENTION)
    worker.require(supplement["complete"] and supplement["n"] == 1000 and
                   supplement["labels_opened"] is False and
                   supplement["public_sha256"] == worker.sha(PUBLIC["mmbench"]) and
                   all(f"full_attention_only_s{s}" in supplement["model_hashes"] for s in (1, 2, 3)),
                   "MMBench full attention-only supplement is not matching sealed public data")
    mm_archive = worker.read(ARCHIVE["mmbench"])
    mm_public = worker.read(PUBLIC["mmbench"])
    matched = 0
    for row in mm_public:
        sid = row["sample_id"]
        key = hashlib.sha256(sid.encode()).hexdigest()[:24]
        sf = MM_ATTENTION.parent / "predictions/cases" / f"{key}.json"
        af = ARCHIVE["mmbench"].parent / "predictions/cases" / f"{key}.json"
        worker.require(supplement["case_hashes"][sid] == worker.sha(sf) and
                       mm_archive["case_hashes"][sid] == worker.sha(af),
                       "Unsealed MMBench supplementary/original case")
        srow, arow = worker.read(sf), worker.read(af)
        worker.require(srow["sample_id"] == arow["sample_id"] == sid and
                       srow["feature_sha256"] == arow["feature_sha256"] and
                       srow["K16_count"] == arow["K16_count"] and
                       all(f"full_attention_only_s{k}" in srow["methods"] for k in (1, 2, 3)),
                       "MMBench supplementary/original public feature mismatch")
        matched += 1
    worker.require(matched == 1000, "MMBench supplement matching count drift")
    for suite in PUBLIC:
        worker.require(len(worker.read(PUBLIC[suite])) == N[suite], "Public cohort drift")
        archive = worker.read(ARCHIVE[suite])
        worker.require(archive["complete"] and archive["n"] == N[suite] and
                       archive["labels_opened"] is False, "Archive not sealed")
        for baseline in BASELINES[suite]:
            for s in (1, 2, 3):
                source = supplement if suite == "mmbench" and baseline == "full_attention_only" else archive
                worker.require(f"{baseline}_s{s}" in source["model_hashes"],
                               "Missing archived comparator")
    rec = {
        "registered_at_utc": worker.now(),
        "analysis_script_sha256": worker.sha(__file__),
        "supersedes_registration_sha256": worker.sha(SUPERSEDED_REG),
        "amendment_reason": "Pre-prediction correction: verify every new and archived prediction point against pinned official PointerBench and MMBench scorers on real annotations; the superseded v2 registered only local scoring. The earlier v1-to-v2 MMBench attention supplement correction remains in force.",
        "worker_protocol_sha256": worker.sha(OUT / "protocol.json"),
        "worker_script_sha256": protocol["worker_sha256"],
        "public_sha256": {suite: worker.sha(PUBLIC[suite]) for suite in PUBLIC},
        "archived_prediction_bundle_sha256": {suite: worker.sha(ARCHIVE[suite]) for suite in ARCHIVE},
        "mmbench_attention_supplement_bundle_sha256": worker.sha(MM_ATTENTION),
        "mmbench_supplement_public_feature_k16_matched_count": matched,
        "mmbench_official_scorer_sha256": worker.sha(SCORER),
        "pointer_official_scorer_sha256": {suite: worker.sha(path) for suite, path in POINTER_SCORER.items()},
        "label_sha256_unopened": {"pointer": worker.sha(POINTER_LABELS),
                                   "mmbench": worker.sha(MM_LABELS)},
        "real_annotation_parity": "For every nonnull new and archived prediction, require pinned official native scorer equality to local normalized scorer; null predictions are incorrect in both. Fail analysis on any mismatch.",
        "prediction_gate": "All six new seed models, all 2,000 cases, per-case and bundle SHA before opening any target annotation",
        "comparator_matrix": {suite: list(BASELINES[suite]) for suite in BASELINES},
        "supplement_source": str(MM_ATTENTION.relative_to(ROOT)).replace("\\", "/"),
        "primary": "PointerBench Pro500",
        "secondary": ["PointerBench Sheets500", "MMBench-GUI selected L2 1000"],
        "metric": "Original K16 forced Top1, same public feature SHA and candidate count; normalized PointerBench geometry or pinned official MMBench bbox scorer",
        "contrasts": "Each new kind versus each available archived comparator, corresponding seed 1/2/3; and linear_all versus head_weight96",
        "statistics": {"unit": "exact RGB pixel SHA cluster", "query_weighted": True,
                       "draws": REPS, "rng_seed": BOOT_SEED, "interval": "percentile 95%",
                       "conditional_on_three_fixed_seeds": True,
                       "multiplicity": "descriptive, no confirmatory superiority claim"},
        "known_limitations": ["Retrospective same-cohort comparison; PointerBench labels were already analyzed by V2",
                              "Selected MMBench subset is not the official full leaderboard",
                              "At least seven PointerBench answers were visible before earlier V2 sealing"],
    }
    if REG.exists():
        rec["registered_at_utc"] = worker.read(REG)["registered_at_utc"]
    write_once(REG, rec)
    print(json.dumps({"stage": "registered", "analysis_registration_sha256": worker.sha(REG)}), flush=True)


def gate() -> tuple[dict, dict]:
    worker.gate()
    worker.require(REG.exists(), "Register analysis before gate")
    reg = worker.read(REG)
    worker.require(reg["analysis_script_sha256"] == worker.sha(__file__) and
                   reg["worker_protocol_sha256"] == worker.sha(OUT / "protocol.json") and
                   reg["supersedes_registration_sha256"] == worker.sha(SUPERSEDED_REG),
                   "Analysis source/protocol SHA drift")
    worker.verify_gate()
    worker.require(reg["mmbench_attention_supplement_bundle_sha256"] == worker.sha(MM_ATTENTION),
                   "MMBench attention supplement bundle SHA drift")
    supplement = worker.read(MM_ATTENTION)
    old_cases = {}
    for suite in PUBLIC:
        worker.require(reg["public_sha256"][suite] == worker.sha(PUBLIC[suite]) and
                       reg["archived_prediction_bundle_sha256"][suite] == worker.sha(ARCHIVE[suite]),
                       "Archived public or prediction SHA drift")
        public = worker.read(PUBLIC[suite])
        archive = worker.read(ARCHIVE[suite])
        worker.require(archive["n"] == len(public) == N[suite] and
                       len(archive["case_hashes"]) == N[suite], "Archive count drift")
        old_cases[suite] = {}
        for p in public:
            sid = p["sample_id"]
            key = hashlib.sha256(sid.encode()).hexdigest()[:24]
            path = ARCHIVE[suite].parent / "predictions/cases" / f"{key}.json"
            worker.require(archive["case_hashes"][sid] == worker.sha(path),
                           "Archived case SHA drift")
            row = worker.read(path)
            npath = OUT / f"predictions/{suite}/cases/{key}.json"
            new = worker.read(npath)
            worker.require(row["sample_id"] == sid == new["sample_id"] and
                           row["labels_opened"] is False and new["labels_opened"] is False and
                           row["feature_sha256"] == new["feature_sha256"] and
                           row["K16_count"] == new["K16_count"] and
                           all(f"{base}_s{s}" in row["methods"]
                               for base in BASELINES[suite] if base != "full_attention_only" or suite != "mmbench"
                               for s in (1, 2, 3)),
                           "Archived/new same-feature or method mismatch")
            if suite == "mmbench":
                extra_path = MM_ATTENTION.parent / "predictions/cases" / f"{key}.json"
                worker.require(supplement["case_hashes"][sid] == worker.sha(extra_path),
                               "MMBench attention supplement case SHA drift")
                extra = worker.read(extra_path)
                worker.require(extra["sample_id"] == sid and extra["labels_opened"] is False and
                               extra["feature_sha256"] == row["feature_sha256"] and
                               extra["K16_count"] == row["K16_count"] and
                               all(f"full_attention_only_s{s}" in extra["methods"] for s in (1, 2, 3)),
                               "MMBench attention supplement same-feature mismatch")
                row["methods"].update({f"full_attention_only_s{s}": extra["methods"][f"full_attention_only_s{s}"]
                                       for s in (1, 2, 3)})
            old_cases[suite][sid] = row
    worker.require(reg["mmbench_official_scorer_sha256"] == worker.sha(SCORER),
                   "Pinned official scorer drift")
    for suite, path in POINTER_SCORER.items():
        worker.require(reg["pointer_official_scorer_sha256"][suite] == worker.sha(path),
                       "Pinned PointerBench scorer drift: " + suite)
    return reg, old_cases


def labels_after_gate(reg: dict) -> tuple[dict, dict]:
    """Only call after gate() verifies every public new and archived prediction."""
    worker.require(reg["label_sha256_unopened"]["pointer"] == worker.sha(POINTER_LABELS) and
                   reg["label_sha256_unopened"]["mmbench"] == worker.sha(MM_LABELS),
                   "Pinned annotation SHA drift")
    pointer = [json.loads(line) for line in POINTER_LABELS.read_text(encoding="utf-8-sig").splitlines()
               if line.strip()]
    worker.require(len(pointer) == 1000, "Pointer labels count drift")
    pointer_by_id = {row["sample_id"]: row for row in pointer}
    worker.require(len(pointer_by_id) == len(pointer), "Pointer label ID duplication")
    mm = worker.read(MM_LABELS)
    worker.require(isinstance(mm, list) and len(mm) == 3594,
                   "MMBench annotation format drift")
    mm_by_index = {row["index"]: row for row in mm}
    worker.require(len(mm_by_index) == len(mm), "MMBench source index duplication")
    return pointer_by_id, mm_by_index


def hits_for_suite(suite: str, old_cases: dict, pointer: dict, mm_labels: dict) -> tuple:
    import numpy as np
    from clickagent.public_evaluation import point_in_geometry
    from run_mmbench_gui_l2_v1 import official_hit, source_reference_hit
    rows = worker.read(PUBLIC[suite])
    if suite == "mmbench":
        native_scorer = source_reference_hit()
    else:
        spec = importlib.util.spec_from_file_location(f"pointer_{suite}_official_eval", POINTER_SCORER[suite])
        worker.require(spec is not None and spec.loader is not None,
                       "Pinned PointerBench evaluator import failed")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        native_scorer = getattr(module, "_in_bbox", None)
        worker.require(callable(native_scorer), "Pinned PointerBench _in_bbox missing")
    keys = [f"{kind}_s{s}" for kind in KINDS for s in (1, 2, 3)]
    keys += [f"{base}_s{s}" for base in BASELINES[suite] for s in (1, 2, 3)]
    hits = {key: np.zeros(len(rows), dtype=np.float32) for key in keys}
    parity_nonnull = 0; null_count = 0
    for i, row in enumerate(rows):
        sid = row["sample_id"]
        key = hashlib.sha256(sid.encode()).hexdigest()[:24]
        new = worker.read(OUT / f"predictions/{suite}/cases/{key}.json")
        old = old_cases[suite][sid]
        for name in keys:
            point = (new["predictions"][name]["point"] if name in new["predictions"]
                     else old["methods"][name]["point"])
            if suite == "mmbench":
                source = mm_labels[row["source_index"]]
                worker.require(all(source[field] == row[field]
                                   for field in ("instruction", "image_size", "data_type", "platform", "grounding_type")),
                               "MMBench public/annotation identity drift")
                outcome = official_hit(point, source["bbox"], row["image_size"])
                native = (bool(native_scorer(source["bbox"].copy(), tuple(row["image_size"]), tuple(point)))
                          if point is not None else False)
            else:
                width, height = row["image_size"]
                outcome = point is not None and point_in_geometry(
                    (point[0] / width, point[1] / height), pointer[sid]["target_geometry"])
                native = (bool(native_scorer(tuple(point), pointer[sid]["bbox_pixels"]))
                          if point is not None else False)
            worker.require(bool(outcome) == native,
                           "Pinned official/local scorer parity failure: " + suite + "/" + sid + "/" + name)
            parity_nonnull += point is not None
            null_count += point is None
            hits[name][i] = bool(outcome)
    return rows, hits, {"nonnull_points": parity_nonnull, "null_points": null_count,
                        "all_points_checked": len(rows) * len(keys), "passed": True}


def summarize(suite: str, rows: list, hits: dict) -> dict:
    import numpy as np
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["pixel_sha256"]].append(i)
    cluster = list(groups.values())
    rng = np.random.default_rng(BOOT_SEED)
    draw = rng.multinomial(len(cluster), np.full(len(cluster), 1 / len(cluster)),
                           size=REPS).astype(np.float32)
    den = draw @ np.asarray([len(ix) for ix in cluster], np.float32)

    def arm(name):
        vals = np.stack([hits[f"{name}_s{s}"] for s in (1, 2, 3)])
        return {"seed_hits": vals.sum(axis=1).astype(int).tolist(),
                "seed_accuracy": vals.mean(axis=1).astype(float).tolist(),
                "mean_accuracy": float(vals.mean())}

    def compare(left, right):
        a = np.stack([hits[f"{left}_s{s}"] for s in (1, 2, 3)])
        b = np.stack([hits[f"{right}_s{s}"] for s in (1, 2, 3)])
        difference = a - b
        per_case = difference.mean(axis=0)
        numerator = np.asarray([per_case[ix].sum() for ix in cluster], np.float32)
        samples = (draw @ numerator) / den
        return {"delta_pp": round(float(per_case.mean() * 100), 4),
                "per_seed_delta_pp": (difference.mean(axis=1) * 100).astype(float).tolist(),
                "ci95_pp": (np.quantile(samples, [0.025, 0.975]) * 100).astype(float).tolist(),
                "wins_mean": float(((a == 1) & (b == 0)).sum(axis=1).mean()),
                "losses_mean": float(((a == 0) & (b == 1)).sum(axis=1).mean()),
                "bootstrap_clusters": len(cluster), "bootstrap_draws": REPS}

    arms = {name: arm(name) for name in (*KINDS, *BASELINES[suite])}
    contrasts = {f"{kind}_vs_{base}": compare(kind, base)
                 for kind in KINDS for base in BASELINES[suite]}
    contrasts["linear_all_vs_head_weight96"] = compare("linear_all", "head_weight96")
    return {"n": len(rows), "rgb_clusters": len(cluster), "arms": arms,
            "paired_contrasts": contrasts, "cohort": suite}


def analyze() -> None:
    reg, old_cases = gate()
    pointer, mm_labels = labels_after_gate(reg)
    outcomes = {}
    for suite in PUBLIC:
        rows, hits, parity = hits_for_suite(suite, old_cases, pointer, mm_labels)
        outcomes[suite] = summarize(suite, rows, hits)
        outcomes[suite]["official_scorer_real_annotation_parity"] = parity
    output = {"complete": True, "analysis_registration_sha256": worker.sha(REG),
              "prediction_completion_sha256": worker.sha(OUT / "predictions/completion.json"),
              "cohorts": outcomes, "retrospective_not_strict_blind": True}
    path = OUT / "analysis/summary.json"
    write_once(path, output)
    lines = ["W3 simple learned readout comparison (retrospective)",
             f"Analysis registration SHA256: {worker.sha(REG)}",
             "Original K16 forced Top1; same frozen feature and candidate pool per case.",
             "Per-seed and mean accuracy; paired differences below are percentage points with RGB-screenshot-cluster 95% intervals."]
    for suite, result in outcomes.items():
        lines += ["", f"{suite}: {result['n']} queries, {result['rgb_clusters']} RGB clusters"]
        for name, value in result["arms"].items():
            lines.append(f"  {name}: {value['seed_hits']} hits; mean {100*value['mean_accuracy']:.2f}%")
        for name, value in result["paired_contrasts"].items():
            lo, hi = value["ci95_pp"]
            lines.append(f"  {name}: {value['delta_pp']:+.2f} pp [{lo:+.2f}, {hi:+.2f}]")
    lines += ["", "Every nonnull new and archived prediction passed real-annotation parity against the pinned official scorer; null predictions were incorrect under both scorers.",
              "MMBench full attention-only comes from the previously sealed full-ablation supplement on the same 1000 public IDs/features/K16; its source was added in a pre-prediction registration amendment.",
              "This is a retrospective same-cohort comparison. PointerBench V2 labels were previously analyzed; at least seven answers were visible before V2 sealing. The selected MMBench subset is not the official complete leaderboard."]
    text_once(OUT / "analysis/readout.txt", "\n".join(lines) + "\n")
    print(json.dumps({"stage": "analyzed", "summary_sha256": worker.sha(path),
                      "readout_sha256": worker.sha(OUT / "analysis/readout.txt")}), flush=True)


if __name__ == "__main__":
    worker.low_priority()
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("register", "verify-gate", "analyze"))
    args = parser.parse_args()
    if args.stage == "register": register()
    elif args.stage == "verify-gate":
        gate()
        print(json.dumps({"stage": "verify_gate_pass", "labels_opened": False}), flush=True)
    else: analyze()
