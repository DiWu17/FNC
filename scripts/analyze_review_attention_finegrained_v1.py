"""Registered label-gated analysis for cached fine-grained attention ablations."""
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

import numpy as np
import run_review_attention_finegrained_v1 as worker
import score_review_attention_finegrained_v1 as scoring

OUT = worker.OUT
REG = OUT / "analysis_registration.json"
RUN_KIND = "cpu_v1"
LEGACY_REG = ROOT / "runs/review_controls_v1/w3/analysis_registration_v3.json"
POINTER_LABELS = ROOT / "runs/paper_scale_ablation_v2/pointerbench/labels.jsonl"
MM_LABELS = ROOT / "data/mmbench_gui_l2_v1/source/L2_annotations.json"
MM_SCORER = ROOT / "runs/mmbench_gui_new_source_v1/official_l2_evaluator.py"
POINTER_SCORER = {
    suite: ROOT / f"runs/paper_scale_ablation_v2/pointerbench/source/{suite}/eval.py"
    for suite in ("pro", "sheets")
}
REPS = 10000
BOOT_SEED = 20261115
REF = "full_attention_only"
SEEDS = (1, 2, 3)


def select_run(kind: str) -> None:
    global OUT, REG, RUN_KIND
    scoring.select_run(kind)
    RUN_KIND = kind
    OUT = scoring.OUT
    REG = OUT / "analysis_registration.json"


def write_once(path: Path, obj) -> None:
    if path.exists():
        worker.require(worker.read(path) == obj, "Immutable result drift: " + str(path))
    else:
        worker.write(path, obj)


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
    _, train_reg = scoring.training_gate()
    worker.require(scoring.REG.exists(), "Scoring must be registered first")
    scoring.gate()
    worker.require(not (OUT / "predictions/completion.json").exists(),
                   "Analysis registration must precede all new prediction seals")
    if not REG.exists():
        worker.require(not any((OUT / "predictions").glob("*/cases/*.json")),
                       "Analysis registration must precede every new prediction")
    old = worker.read(LEGACY_REG)
    worker.require(old["label_sha256_unopened"]["pointer"] and
                   old["label_sha256_unopened"]["mmbench"],
                   "Missing previously frozen label digests")
    rec = {
        "registered_at_utc": worker.now(),
        "run_kind": RUN_KIND,
        "train_protocol_sha256": worker.sha(train_reg),
        "analysis_script_sha256": worker.sha(__file__),
        "worker_protocol_sha256": worker.sha(worker.REG),
        "score_registration_sha256": worker.sha(scoring.REG),
        "scoring_script_sha256": worker.sha(scoring.__file__),
        "legacy_label_digest_registration_sha256": worker.sha(LEGACY_REG),
        "label_sha256_expected_without_opening": old["label_sha256_unopened"],
        "mmbench_official_scorer_sha256": worker.sha(MM_SCORER),
        "pointer_official_scorer_sha256": {s: worker.sha(p) for s, p in POINTER_SCORER.items()},
        "prediction_gate": "All 18 new models x all 2000 cases and per-case/bundle SHA receipts must seal before opening bbox/target geometry",
        "cohorts": scoring.EXPECTED_N,
        "reference": REF,
        "variants": list(worker.VARIANTS),
        "metric": "Original cached K16 forced Top1, null or empty candidate pool incorrect, official native scorer parity for every nonnull point",
        "primary": "PointerBench Pro500",
        "secondary": ["PointerBench Sheets500", "selected MMBench-GUI L2 1000"],
        "statistics": {"unit": "RGB pixel SHA screenshot cluster", "query_weighted": True,
                       "draws": REPS, "rng_seed": BOOT_SEED,
                       "interval": "percentile paired 95%",
                       "contrasts": "Each of six new variants minus frozen full_attention_only, matched training seed",
                       "multiplicity": "descriptive only; no post hoc best-arm selection or superiority claim"},
        "known_limitations": ["Retrospective same-cohort comparison; PointerBench labels were previously analyzed",
                              "At least seven PointerBench answers were visible before earlier V2 sealing",
                              "Selected MMBench subset is not the full official leaderboard"],
    }
    if REG.exists():
        rec["registered_at_utc"] = worker.read(REG)["registered_at_utc"]
    write_once(REG, rec)
    print(json.dumps({"stage": "analysis_registered", "sha256": worker.sha(REG)}), flush=True)


def registration_gate() -> dict:
    worker.require(REG.exists(), "Analysis registration missing")
    reg = worker.read(REG)
    _, train_reg = scoring.training_gate()
    scoring.gate()
    worker.require(reg["run_kind"] == RUN_KIND and
                   reg["train_protocol_sha256"] == worker.sha(train_reg) and
                   reg["analysis_script_sha256"] == worker.sha(__file__) and
                   reg["worker_protocol_sha256"] == worker.sha(worker.REG) and
                   reg["score_registration_sha256"] == worker.sha(scoring.REG) and
                   reg["scoring_script_sha256"] == worker.sha(scoring.__file__) and
                   reg["legacy_label_digest_registration_sha256"] == worker.sha(LEGACY_REG),
                   "Analysis protocol or source drift")
    worker.require(reg["mmbench_official_scorer_sha256"] == worker.sha(MM_SCORER),
                   "Official MMBench scorer drift")
    for suite, path in POINTER_SCORER.items():
        worker.require(reg["pointer_official_scorer_sha256"][suite] == worker.sha(path),
                       "Official Pointer scorer drift: " + suite)
    return reg


def gate() -> dict:
    reg = registration_gate()
    scoring.verify_gate()  # Public prediction receipt only; never opens target geometry.
    return reg


def labels_after_gate(reg: dict) -> tuple[dict, dict]:
    expected = reg["label_sha256_expected_without_opening"]
    worker.require(worker.sha(POINTER_LABELS) == expected["pointer"] and
                   worker.sha(MM_LABELS) == expected["mmbench"],
                   "Pinned label SHA drift")
    pointer = [json.loads(line) for line in POINTER_LABELS.read_text(encoding="utf-8-sig").splitlines()
               if line.strip()]
    worker.require(len(pointer) == 1000, "Pointer label count drift")
    by_id = {row["sample_id"]: row for row in pointer}
    worker.require(len(by_id) == len(pointer), "Pointer label duplicate IDs")
    mm = worker.read(MM_LABELS)
    worker.require(isinstance(mm, list) and len(mm) == 3594,
                   "MMBench label list drift")
    by_index = {row["index"]: row for row in mm}
    worker.require(len(by_index) == len(mm), "MMBench label duplicate index")
    return by_id, by_index


def scored_points(suite: str, pointer: dict, mm: dict) -> tuple[list, dict, dict]:
    from clickagent.public_evaluation import point_in_geometry
    from run_mmbench_gui_l2_v1 import official_hit, source_reference_hit

    rows = worker.read(scoring.PUBLIC[suite])
    names = [f"{kind}_s{s}" for kind in (REF, *worker.VARIANTS) for s in SEEDS]
    hits = {name: np.zeros(len(rows), np.float32) for name in names}
    if suite == "mmbench":
        native = source_reference_hit()
    else:
        spec = importlib.util.spec_from_file_location(f"attention_pointer_{suite}", POINTER_SCORER[suite])
        worker.require(spec is not None and spec.loader is not None,
                       "Cannot load Pointer official scorer")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        native = getattr(module, "_in_bbox", None)
        worker.require(callable(native), "Pointer official scorer function missing")
    parity = {"nonnull_points": 0, "null_points": 0, "all_points_checked": 0, "passed": True}
    for j, row in enumerate(rows):
        sid = row["sample_id"]
        key = scoring.case_key(sid)
        new = worker.read(OUT / f"predictions/{suite}/cases/{key}.json")
        archived = scoring.old_case(suite, sid)
        for name in names:
            point = (archived["methods"][name]["point"] if name.startswith(REF)
                     else new["predictions"][name]["point"])
            if suite == "mmbench":
                source = mm[row["source_index"]]
                worker.require(all(source[k] == row[k] for k in
                                   ("instruction", "image_size", "data_type", "platform", "grounding_type")),
                               "MMBench public/label identity drift")
                local_hit = official_hit(point, source["bbox"], row["image_size"])
                native_hit = (bool(native(source["bbox"].copy(), tuple(row["image_size"]), tuple(point)))
                              if point is not None else False)
            else:
                label = pointer[sid]
                w, h = row["image_size"]
                local_hit = (bool(point_in_geometry((point[0] / w, point[1] / h),
                                                    label["target_geometry"]))
                             if point is not None else False)
                native_hit = (bool(native(tuple(point), label["bbox_pixels"]))
                              if point is not None else False)
            worker.require(bool(local_hit) == bool(native_hit),
                           "Real-label official/local scorer parity drift: " + suite + "/" + sid + "/" + name)
            hits[name][j] = bool(local_hit)
            parity["all_points_checked"] += 1
            parity["nonnull_points" if point is not None else "null_points"] += 1
    return rows, hits, parity


def summary(suite: str, rows: list, hits: dict) -> dict:
    groups = defaultdict(list)
    for j, row in enumerate(rows):
        groups[row["pixel_sha256"]].append(j)
    clusters = list(groups.values())
    names = (REF, *worker.VARIANTS)
    arms = {}
    for name in names:
        array = np.stack([hits[f"{name}_s{s}"] for s in SEEDS])
        arms[name] = {"seed_hits": array.sum(axis=1).astype(int).tolist(),
                      "seed_accuracy_pct": (100 * array.mean(axis=1)).tolist(),
                      "mean_accuracy_pct": float(100 * array.mean())}
    rng = np.random.default_rng(BOOT_SEED)
    draw = rng.multinomial(len(clusters), np.full(len(clusters), 1 / len(clusters)),
                           size=REPS).astype(np.float32)
    lengths = np.array([len(ix) for ix in clusters], np.float32)
    den = draw @ lengths
    deltas = []
    seed_deltas = []
    for kind in worker.VARIANTS:
        diff = np.stack([hits[f"{kind}_s{s}"] - hits[f"{REF}_s{s}"] for s in SEEDS])
        deltas.append(diff.mean(axis=0))
        seed_deltas.append((100 * diff.mean(axis=1)).tolist())
    numerators = np.array([[d[ix].sum() for d in deltas] for ix in clusters], np.float32)
    draws = (draw @ numerators) / den[:, None]
    contrasts = {}
    for k, kind in enumerate(worker.VARIANTS):
        contrasts[f"{kind}_vs_{REF}"] = {
            "delta_pp": float(100 * deltas[k].mean()),
            "per_seed_delta_pp": seed_deltas[k],
            "ci95_pp": (100 * np.quantile(draws[:, k], [0.025, 0.975])).tolist(),
            "bootstrap_clusters": len(clusters), "bootstrap_draws": REPS,
        }
    return {"cohort": suite, "n": len(rows), "rgb_clusters": len(clusters),
            "arms": arms, "paired_contrasts": contrasts}


def markdown(outcomes: dict, reg: dict) -> str:
    lines = ["# Fine-grained attention evidence ablation", "",
             f"Selected training run: `{RUN_KIND}`; training protocol SHA256: `{reg['train_protocol_sha256']}`.", "",
             "Cached original K16, forced Top1, 13,376 AdamW steps, three fixed seeds. All six variants use the unchanged 22,050-parameter readout with hidden-state channels zeroed. Parent model inference and candidate generation were not rerun.",
             "", "Each cell reports S1/S2/S3 correct counts and the three-seed mean accuracy. The frozen attention-only readout is the matched reference.", ""]
    for suite, title in (("pro", "PointerBench Pro500"), ("sheets", "PointerBench Sheets500"),
                         ("mmbench", "MMBench-GUI selected L2 1000")):
        result = outcomes[suite]
        lines += [f"## {title}", "", "| Readout input | S1 | S2 | S3 | Mean % | Δ vs attention only (pp), paired 95% CI |",
                  "| --- | ---: | ---: | ---: | ---: | ---: |"]
        for name in (REF, *worker.VARIANTS):
            arm = result["arms"][name]
            count = arm["seed_hits"]
            if name == REF:
                delta = "reference"
            else:
                cmp = result["paired_contrasts"][f"{name}_vs_{REF}"]
                lo, hi = cmp["ci95_pp"]
                delta = f"{cmp['delta_pp']:+.2f} [{lo:+.2f}, {hi:+.2f}]"
            lines.append(f"| {name.replace('_', ' ')} | {count[0]} | {count[1]} | {count[2]} | {arm['mean_accuracy_pct']:.2f} | {delta} |")
        lines += ["", f"Fixed denominator: {result['n']} queries; {result['rgb_clusters']} RGB screenshot clusters. All nonnull predicted points passed official/local scorer parity.", ""]
    lines += ["## Interpretation and limits", "",
              "Paired intervals resample screenshot clusters and average the three fixed seed contrasts. The six contrasts are descriptive; no arm is selected from test results. Differences isolate which cached attention channels are available to an unchanged readout, not the effect of a new backbone or candidate pool.",
              "", "This is a retrospective same-cohort analysis. PointerBench V2 labels had already been analyzed, and at least seven answers had been visible before that earlier sealing. The selected MMBench subset is not its complete leaderboard. The experiment tests one backbone and one candidate proposer.",
              "", f"Analysis registration SHA256: `{worker.sha(REG)}`. Prediction seal SHA256: `{worker.sha(OUT / 'predictions/completion.json')}`.", ""]
    return "\n".join(lines)


def analyze() -> None:
    reg = gate()
    pointer, mm = labels_after_gate(reg)
    outcomes = {}
    for suite in scoring.EXPECTED_N:
        rows, hits, parity = scored_points(suite, pointer, mm)
        outcomes[suite] = summary(suite, rows, hits)
        outcomes[suite]["official_scorer_real_annotation_parity"] = parity
    result = {"complete": True, "analysis_registration_sha256": worker.sha(REG),
              "prediction_completion_sha256": worker.sha(OUT / "predictions/completion.json"),
              "cohorts": outcomes, "retrospective_not_strict_blind": True}
    write_once(OUT / "analysis/summary.json", result)
    text_once(OUT / "analysis/readout.txt", markdown(outcomes, reg))
    text_once(ROOT / f"paper/GSAIE/REVIEW_ATTENTION_FINEGRAINED_{RUN_KIND.upper()}_RESULTS.md",
              markdown(outcomes, reg))
    print(json.dumps({"stage": "analyzed", "summary_sha256": worker.sha(OUT / "analysis/summary.json"),
                      "readout_sha256": worker.sha(OUT / "analysis/readout.txt")}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", choices=tuple(scoring.RUN_ROOTS), default="cpu_v1")
    parser.add_argument("--stage", required=True, choices=("register", "verify-gate", "analyze"))
    args = parser.parse_args()
    select_run(args.run_root)
    if args.stage == "register":
        register()
    elif args.stage == "verify-gate":
        gate()
        print(json.dumps({"stage": "verify_gate_pass", "labels_opened": False}), flush=True)
    else:
        analyze()


if __name__ == "__main__":
    worker.low_priority()
    main()
