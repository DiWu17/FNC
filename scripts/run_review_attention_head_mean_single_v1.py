"""One-seed cached head-mean readout: registered prediction then label-gated analysis."""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np
import torch
from clickagent.evidence_retention import RetentionModel
import run_review_attention_finegrained_v1 as fine
import score_review_attention_finegrained_v1 as common

TRAIN = ROOT / "runs/review_attention_finegrained_gpu_v3"
OUT = TRAIN / "single_arm_eval"
TRAIN_REG = TRAIN / "protocol.json"
TRAIN_WORKER = ROOT / "scripts/run_review_attention_finegrained_gpu_v3.py"
SCORE_REG = OUT / "score_registration.json"
ANALYSIS_REG = OUT / "analysis_registration.json"
LEGACY_REG = ROOT / "runs/review_controls_v1/w3/analysis_registration_v3.json"
POINTER_LABELS = ROOT / "runs/paper_scale_ablation_v2/pointerbench/labels.jsonl"
MM_LABELS = ROOT / "data/mmbench_gui_l2_v1/source/L2_annotations.json"
MM_SCORER = ROOT / "runs/mmbench_gui_new_source_v1/official_l2_evaluator.py"
POINTER_SCORER = {
    suite: ROOT / f"runs/paper_scale_ablation_v2/pointerbench/source/{suite}/eval.py"
    for suite in ("pro", "sheets")
}
ARM = "head_mean_all96_s1"
REF = [f"full_attention_only_s{i}" for i in (1, 2, 3)]
COHORTS = {"pro": 500, "sheets": 500, "mmbench": 1000}


def require(ok: bool, message: str) -> None:
    fine.require(ok, message)


def sha(path: str | Path) -> str:
    return fine.sha(path)


def read(path: str | Path):
    return fine.read(path)


def write_once(path: Path, data) -> None:
    if path.exists():
        require(read(path) == data, "Immutable receipt drift: " + str(path))
    else:
        fine.write(path, data)


def text_once(path: Path, content: str) -> None:
    if path.exists():
        require(path.read_text(encoding="utf-8") == content,
                "Immutable text result drift: " + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)


def status(phase: str, **fields) -> None:
    data = dict(phase=phase, at_utc=fine.now(), **fields)
    fine.write(OUT / "status.json", data)
    print(json.dumps(data), flush=True)


def training_gate() -> tuple[dict, dict, dict]:
    parent = fine.gate()
    p = read(TRAIN_REG)
    require(p["parent_protocol_sha256"] == sha(fine.REG) and
            p["parent_worker_sha256"] == sha(fine.__file__) and
            p["gpu_worker_sha256"] == sha(TRAIN_WORKER),
            "Frozen single-arm GPU V3 script/science drift")
    expected_training = dict(parent["training"], seeds=[20261311])
    require(p["training"] == expected_training and
            p["evaluation"] == parent["evaluation"] and
            p["source_sha256"] == parent["source_sha256"],
            "Single-arm GPU V3 training/source/evaluation drift")
    variant = p.get("variant", p.get("variants"))
    seed = p.get("seed", p.get("seeds"))
    require(variant in ("head_mean_all96", ["head_mean_all96"]) and
            seed in (20261311, [20261311]),
            "GPU V3 must be the one prespecified head-mean seed")
    cpath = TRAIN / "models/completion.json"
    completion = read(cpath)
    receipt_path = TRAIN / f"models/{ARM}.json"
    require(completion["complete"] is True and completion["count"] == 1 and
            completion["protocol_sha256"] == sha(TRAIN_REG) and
            completion["models"] == {ARM: sha(receipt_path)},
            "Single-arm model registry/count/SHA drift")
    model = read(receipt_path)
    require(model["complete"] is True and model["name"] == ARM and
            model["seed"] == 20261311 and model["steps"] == fine.STEPS and
            model["fitting_queries"] == 12886 and model["empty_skipped"] == 487 and
            model["parameter_count"] == 22050 and
            model["protocol_sha256"] == sha(TRAIN_REG) and
            Path(model["checkpoint_path"]).resolve() ==
                (TRAIN / f"models/{ARM}.pt").resolve() and
            model["checkpoint_sha256"] == sha(model["checkpoint_path"]),
            "Single-arm checkpoint/receipt drift")
    base_seed = read(fine.V18 / "models/full_s1.json")
    require(model["initial_state_sha256"] == base_seed["initial_state_sha256"],
            "Single-arm initial state did not match frozen seed 1")
    return p, completion, model


def source_sha() -> dict:
    return {"public": {s: sha(p) for s, p in common.PUBLIC.items()},
            "features": {s: sha(p) for s, p in common.FEATURES.items()},
            "archived_attention": {s: sha(p) for s, p in common.ARCHIVED_BUNDLES.items()}}


def register() -> None:
    p, completion, model = training_gate()
    if not SCORE_REG.exists() or not ANALYSIS_REG.exists():
        require(not any((OUT / "predictions").glob("*/cases/*.json")) and
                not (OUT / "predictions/completion.json").exists(),
                "Register before any one-arm predictions")
    score = dict(registered_at_utc=fine.now(),
                 purpose="Single prespecified head-mean S1 cached K16 prediction",
                 script_sha256=sha(__file__), common_scoring_sha256=sha(common.__file__),
                 train_protocol_sha256=sha(TRAIN_REG),
                 train_worker_sha256=sha(TRAIN_WORKER),
                 model_completion_sha256=sha(TRAIN / "models/completion.json"),
                 model_receipt_sha256=completion["models"][ARM],
                 checkpoint_sha256=model["checkpoint_sha256"],
                 frozen_reference_models=p.get("reference_models", fine.reference_models()),
                 source_sha256=source_sha(), cohorts=COHORTS,
                 method=ARM, seed=20261311,
                 candidate_rule="Original cached K16, sorted union indices, stable forced Top1",
                 reference_parity="Re-score archived full_attention_only S1/S2/S3, require identical point/index on every case",
                 labels_opened=False, no_backbone_or_proposer_call=True)
    if SCORE_REG.exists():
        score["registered_at_utc"] = read(SCORE_REG)["registered_at_utc"]
    write_once(SCORE_REG, score)
    old = read(LEGACY_REG)
    analysis = dict(registered_at_utc=fine.now(),
                    score_registration_sha256=sha(SCORE_REG),
                    analysis_script_sha256=sha(__file__),
                    legacy_label_digest_registration_sha256=sha(LEGACY_REG),
                    label_sha256_expected_without_opening=old["label_sha256_unopened"],
                    official_pointer_scorer_sha256={s: sha(p) for s, p in POINTER_SCORER.items()},
                    official_mmbench_scorer_sha256=sha(MM_SCORER),
                    gate="All 2000 one-arm predictions and per-case/bundle SHA sealed before opening any target geometry",
                    metric="Fixed denominator, original K16 forced Top1; null predictions wrong; native official scorer parity for every nonnull point",
                    comparisons="One new seed S1 versus archived attention-only S1; archived S1/S2/S3 mean shown only as historical context",
                    inference="Descriptive trend only; no three-seed new-arm statistics or superiority claim",
                    limitations=["Retrospective cohorts; PointerBench labels had been analyzed earlier",
                                 "At least seven PointerBench answers had been visible before earlier sealing",
                                 "Selected MMBench 1000 is not the full official leaderboard"])
    if ANALYSIS_REG.exists():
        analysis["registered_at_utc"] = read(ANALYSIS_REG)["registered_at_utc"]
    write_once(ANALYSIS_REG, analysis)
    status("registered", score_registration_sha256=sha(SCORE_REG),
           analysis_registration_sha256=sha(ANALYSIS_REG), labels_opened=False)


def registration_gate() -> tuple[dict, dict]:
    _, completion, model = training_gate()
    require(SCORE_REG.exists() and ANALYSIS_REG.exists(),
            "Both score and analysis must register before prediction")
    s, a = read(SCORE_REG), read(ANALYSIS_REG)
    require(s["script_sha256"] == sha(__file__) and
            s["common_scoring_sha256"] == sha(common.__file__) and
            s["train_protocol_sha256"] == sha(TRAIN_REG) and
            s["train_worker_sha256"] == sha(TRAIN_WORKER) and
            s["model_completion_sha256"] == sha(TRAIN / "models/completion.json") and
            s["model_receipt_sha256"] == completion["models"][ARM] and
            s["checkpoint_sha256"] == model["checkpoint_sha256"] and
            s["frozen_reference_models"] == fine.reference_models() and
            s["source_sha256"] == source_sha() and s["cohorts"] == COHORTS and
            s["method"] == ARM and s["seed"] == 20261311 and
            s["labels_opened"] is False,
            "Single-arm frozen scoring registration drift")
    require(a["score_registration_sha256"] == sha(SCORE_REG) and
            a["analysis_script_sha256"] == sha(__file__) and
            a["legacy_label_digest_registration_sha256"] == sha(LEGACY_REG) and
            a["label_sha256_expected_without_opening"] ==
                read(LEGACY_REG)["label_sha256_unopened"] and
            a["official_pointer_scorer_sha256"] ==
                {suite: sha(path) for suite, path in POINTER_SCORER.items()} and
            a["official_mmbench_scorer_sha256"] == sha(MM_SCORER),
            "Single-arm frozen analysis registration drift")
    return s, a


def start_gate(*, allow_create: bool) -> str:
    s, a = registration_gate()
    path = OUT / "predictions/start.json"
    value = dict(score_registration_sha256=sha(SCORE_REG),
                 analysis_registration_sha256=sha(ANALYSIS_REG),
                 model_checkpoint_sha256=s["checkpoint_sha256"],
                 labels_opened=False)
    if path.exists():
        require(read(path) == value, "Prediction-start receipt drift")
    elif allow_create:
        write_once(path, value)
    else:
        raise ValueError("Prediction-start receipt absent")
    return sha(path)


def load_models(score_reg: dict) -> tuple[dict, torch.Tensor, torch.Tensor]:
    receipt = read(TRAIN / f"models/{ARM}.json")
    saved = torch.load(receipt["checkpoint_path"], map_location="cpu", weights_only=True)
    require(saved["seed"] == 20261311 and
            saved["mean"].shape == (32,) and saved["std"].shape == (32,),
            "Single-arm checkpoint contents drift")
    models = {}
    model = RetentionModel().cpu().eval()
    model.load_state_dict(saved["state_dict"])
    models[ARM] = model
    for name in REF:
        item = score_reg["frozen_reference_models"][name]
        require(sha(item["path"]) == item["sha256"],
                "Attention-only reference checkpoint drift")
        frozen = torch.load(item["path"], map_location="cpu", weights_only=True)
        require(torch.equal(frozen["mean"].float(), saved["mean"].float()) and
                torch.equal(frozen["std"].float(), saved["std"].float()),
                "Training-only scaler differs from archived attention-only")
        ref = RetentionModel().cpu().eval()
        ref.load_state_dict(frozen["state_dict"])
        models[name] = ref
    return models, saved["mean"].float(), saved["std"].float()


def score_suite(suite: str) -> None:
    fine.low_priority(); torch.set_num_threads(2)
    require(not torch.cuda.is_available(), "Hide CUDA for single-arm CPU scoring")
    score_reg, _ = registration_gate()
    start_sha = start_gate(allow_create=True)
    require(suite in COHORTS, "Unknown cohort")
    rows, entries = read(common.PUBLIC[suite]), read(common.FEATURES[suite])["cases"]
    require(len(rows) == len(entries) == COHORTS[suite] and
            [r["sample_id"] for r in rows] == [e["sample_id"] for e in entries],
            "Public/feature cohort order drift")
    models, mean, std = load_models(score_reg)
    case_hashes = {}
    for j, (row, entry) in enumerate(zip(rows, entries), 1):
        sid = row["sample_id"]
        path = OUT / f"predictions/{suite}/cases/{common.case_key(sid)}.json"
        if path.exists():
            case = read(path)
            require(case["complete"] and case["sample_id"] == sid and
                    case["feature_sha256"] == entry["sha256"] and
                    case["checkpoint_sha256"] == score_reg["checkpoint_sha256"] and
                    case["labels_opened"] is False,
                    "Existing one-arm prediction case drift")
        else:
            pred, pool_n, union_n = common.score_one(entry, suite, models, mean, std)
            require(set(pred) == {ARM}, "Only the single registered arm may be scored")
            case = dict(complete=True, suite=suite, sample_id=sid,
                        feature_sha256=entry["sha256"], K16_count=pool_n,
                        candidate_count=union_n, checkpoint_sha256=score_reg["checkpoint_sha256"],
                        prediction=pred[ARM], archived_attention_only_parity=True,
                        labels_opened=False)
            write_once(path, case)
        case_hashes[sid] = sha(path)
        if j % 25 == 0 or j == len(rows):
            status("scoring", suite=suite, completed=j, total=len(rows), labels_opened=False)
    bundle = dict(complete=True, suite=suite, n=len(rows), case_hashes=case_hashes,
                  public_sha256=score_reg["source_sha256"]["public"][suite],
                  feature_index_sha256=score_reg["source_sha256"]["features"][suite],
                  prediction_start_sha256=start_sha,
                  score_registration_sha256=sha(SCORE_REG),
                  analysis_registration_sha256=sha(ANALYSIS_REG),
                  checkpoint_sha256=score_reg["checkpoint_sha256"], labels_opened=False)
    path = OUT / f"predictions/{suite}/bundle.json"
    write_once(path, bundle)
    status("suite_sealed", suite=suite, bundle_sha256=sha(path), labels_opened=False)


def verify_gate() -> None:
    score_reg, _ = registration_gate()
    start_sha = start_gate(allow_create=False)
    for suite, n in COHORTS.items():
        rows, entries = read(common.PUBLIC[suite]), read(common.FEATURES[suite])["cases"]
        bundle_path = OUT / f"predictions/{suite}/bundle.json"
        bundle = read(bundle_path)
        require(bundle["complete"] and bundle["n"] == n and
                bundle["labels_opened"] is False and
                bundle["prediction_start_sha256"] == start_sha and
                bundle["score_registration_sha256"] == sha(SCORE_REG) and
                bundle["analysis_registration_sha256"] == sha(ANALYSIS_REG) and
                bundle["checkpoint_sha256"] == score_reg["checkpoint_sha256"] and
                bundle["public_sha256"] == score_reg["source_sha256"]["public"][suite] and
                bundle["feature_index_sha256"] == score_reg["source_sha256"]["features"][suite] and
                len(bundle["case_hashes"]) == n,
                "One-arm bundle count/identity drift: " + suite)
        require([r["sample_id"] for r in rows] == [e["sample_id"] for e in entries],
                "Public/feature order drift: " + suite)
        for row, entry in zip(rows, entries):
            sid = row["sample_id"]
            path = OUT / f"predictions/{suite}/cases/{common.case_key(sid)}.json"
            require(sha(path) == bundle["case_hashes"][sid],
                    "One-arm case SHA drift: " + sid)
            case = read(path)
            points, _, _, pool, _, _ = common.read_cached_case(entry)
            require(case["complete"] and case["sample_id"] == sid and
                    case["feature_sha256"] == entry["sha256"] and
                    case["checkpoint_sha256"] == score_reg["checkpoint_sha256"] and
                    case["K16_count"] == len(pool) and
                    case["candidate_count"] == len(points) and
                    case["archived_attention_only_parity"] is True and
                    case["labels_opened"] is False,
                    "One-arm prediction case identity drift: " + sid)
            ix, point = case["prediction"]["index"], case["prediction"]["point"]
            require((ix is None and point is None and not len(pool)) or
                    (ix is not None and ix in pool and point == points[ix].tolist()),
                    "One-arm prediction left original K16: " + sid)
            archived = common.old_case(suite, sid)
            require(archived["feature_sha256"] == entry["sha256"] and
                    archived["K16_count"] == len(pool) and
                    all(name in archived["methods"] for name in REF),
                    "Archived attention-only comparison identity drift")
    completion = dict(complete=True, labels_opened=False, cohorts=COHORTS,
                      score_registration_sha256=sha(SCORE_REG),
                      analysis_registration_sha256=sha(ANALYSIS_REG),
                      prediction_start_sha256=start_sha,
                      bundle_sha256={s: sha(OUT / f"predictions/{s}/bundle.json") for s in COHORTS})
    write_once(OUT / "predictions/completion.json", completion)
    status("prediction_gate_pass", prediction_completion_sha256=sha(OUT / "predictions/completion.json"),
           labels_opened=False)


def labels_after_gate(analysis_reg: dict) -> tuple[dict, dict]:
    expected = analysis_reg["label_sha256_expected_without_opening"]
    require(sha(POINTER_LABELS) == expected["pointer"] and
            sha(MM_LABELS) == expected["mmbench"],
            "Pinned test label SHA drift")
    pointer = [json.loads(line) for line in POINTER_LABELS.read_text(encoding="utf-8-sig").splitlines()
               if line.strip()]
    require(len(pointer) == 1000, "Pointer label count drift")
    pmap = {row["sample_id"]: row for row in pointer}
    require(len(pmap) == 1000, "Pointer duplicate label ID")
    mm = read(MM_LABELS)
    require(isinstance(mm, list) and len(mm) == 3594,
            "MMBench label list drift")
    mmap = {row["index"]: row for row in mm}
    require(len(mmap) == len(mm), "MMBench duplicate source index")
    return pmap, mmap


def evaluate_suite(suite: str, pointer: dict, mm: dict) -> dict:
    from clickagent.public_evaluation import point_in_geometry
    from run_mmbench_gui_l2_v1 import official_hit, source_reference_hit

    rows = read(common.PUBLIC[suite])
    if suite == "mmbench":
        native = source_reference_hit()
    else:
        spec = importlib.util.spec_from_file_location(f"single_pointer_{suite}", POINTER_SCORER[suite])
        require(spec is not None and spec.loader is not None,
                "Official Pointer scorer import failed")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        native = getattr(module, "_in_bbox", None)
        require(callable(native), "Official Pointer _in_bbox missing")
    hits = {name: 0 for name in (ARM, *REF)}
    paired_wins = paired_losses = 0
    checked = nulls = 0
    for row in rows:
        sid = row["sample_id"]
        case = read(OUT / f"predictions/{suite}/cases/{common.case_key(sid)}.json")
        archive = common.old_case(suite, sid)
        outcomes = {}
        for name in (ARM, *REF):
            point = (case["prediction"]["point"] if name == ARM else
                     archive["methods"][name]["point"])
            if suite == "mmbench":
                source = mm[row["source_index"]]
                require(all(source[k] == row[k] for k in
                            ("instruction", "image_size", "data_type", "platform", "grounding_type")),
                        "MMBench public/annotation identity drift")
                local_hit = official_hit(point, source["bbox"], row["image_size"])
                native_hit = (bool(native(source["bbox"].copy(), tuple(row["image_size"]), tuple(point)))
                              if point is not None else False)
            else:
                w, h = row["image_size"]
                label = pointer[sid]
                local_hit = (bool(point_in_geometry((point[0]/w, point[1]/h),
                                                    label["target_geometry"]))
                             if point is not None else False)
                native_hit = (bool(native(tuple(point), label["bbox_pixels"]))
                              if point is not None else False)
            require(bool(local_hit) == bool(native_hit),
                    "Official/local real-label parity drift: " + suite + "/" + sid + "/" + name)
            outcomes[name] = int(bool(local_hit))
            hits[name] += outcomes[name]
            checked += 1
            nulls += point is None
        paired_wins += outcomes[ARM] == 1 and outcomes[REF[0]] == 0
        paired_losses += outcomes[ARM] == 0 and outcomes[REF[0]] == 1
    n = len(rows)
    result = {"n": n, "hits": hits,
              "accuracy_pct": {name: 100*count/n for name, count in hits.items()},
              "historical_attention_only_three_seed_mean_pct":
                  100*sum(hits[name] for name in REF)/(3*n),
              "single_seed_delta_vs_attention_only_s1_pp":
                  100*(hits[ARM]-hits[REF[0]])/n,
              "single_seed_delta_vs_historical_three_seed_mean_pp":
                  100*(hits[ARM]-sum(hits[name] for name in REF)/3)/n,
              "paired_s1_wins": int(paired_wins), "paired_s1_losses": int(paired_losses),
              "official_scorer_real_annotation_parity":
                  {"passed": True, "all_points_checked": checked, "null_points": nulls,
                   "nonnull_points": checked-nulls}}
    return result


def report(result: dict) -> str:
    lines = ["# Single-seed head-averaged attention diagnostic", "",
             "One prespecified `head_mean_all96_s1` model (seed 20261311) trained with the original 13,376-step schedule. All test predictions used cached original K16 candidates and were SHA sealed before labels were opened. The numbers below are a descriptive single-seed diagnostic.", "",
             "| Cohort | N | Head mean S1 hits | Head mean S1 % | Archived attention-only S1 hits | Archived attention-only S1 % | Δ S1 (pp) | Archived attention-only 3-seed mean % |",
             "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for suite, title in (("pro", "Pro500"), ("sheets", "Sheets500"), ("mmbench", "MMB1000")):
        r = result["cohorts"][suite]
        lines.append(f"| {title} | {r['n']} | {r['hits'][ARM]} | {r['accuracy_pct'][ARM]:.2f} | {r['hits'][REF[0]]} | {r['accuracy_pct'][REF[0]]:.2f} | {r['single_seed_delta_vs_attention_only_s1_pp']:+.2f} | {r['historical_attention_only_three_seed_mean_pct']:.2f} |")
    lines += ["", "The historical attention-only three-seed mean is context, not a matched three-seed estimate for the new one-seed model. No confidence interval, significance test, or claim about the full head-mean distribution is made.",
              "", "The original K16 pool, model architecture and training data were fixed. This experiment collapses 96 head values to three metric means per candidate/token; it does not isolate layer diversity, which requires a per-layer mean control.",
              "", "Retrospective cohorts: PointerBench labels were analyzed earlier and at least seven answers were visible before earlier sealing; selected MMBench L2 1000 is not the full official leaderboard.",
              "", f"Score registration SHA256: `{sha(SCORE_REG)}`; analysis registration SHA256: `{sha(ANALYSIS_REG)}`; prediction seal SHA256: `{sha(OUT / 'predictions/completion.json')}`.", ""]
    return "\n".join(lines)


def analyze() -> None:
    verify_gate()
    analysis_reg = read(ANALYSIS_REG)
    pointer, mm = labels_after_gate(analysis_reg)
    cohorts = {suite: evaluate_suite(suite, pointer, mm) for suite in COHORTS}
    result = dict(complete=True, method=ARM, seed=20261311,
                  training_protocol_sha256=sha(TRAIN_REG),
                  score_registration_sha256=sha(SCORE_REG),
                  analysis_registration_sha256=sha(ANALYSIS_REG),
                  prediction_completion_sha256=sha(OUT / "predictions/completion.json"),
                  cohorts=cohorts, descriptive_single_seed_only=True)
    write_once(OUT / "analysis/summary.json", result)
    content = report(result)
    text_once(OUT / "analysis/readout.txt", content)
    text_once(ROOT / "paper/GSAIE/REVIEW_HEAD_MEAN_SINGLE_SEED_RESULTS.md", content)
    status("analyzed", summary_sha256=sha(OUT / "analysis/summary.json"),
           readout_sha256=sha(OUT / "analysis/readout.txt"), labels_opened=True)


def selftest() -> None:
    common.selftest()
    print(json.dumps({"stage": "single_arm_eval_selftest_pass", "model_required": ARM,
                      "test_labels_opened": False}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=("selftest", "register", "score-pro", "score-sheets",
                                 "score-mmbench", "verify-gate", "analyze"))
    args = parser.parse_args()
    if args.stage == "selftest": selftest()
    elif args.stage == "register": register()
    elif args.stage.startswith("score-"): score_suite(args.stage.removeprefix("score-"))
    elif args.stage == "verify-gate": verify_gate()
    else: analyze()


if __name__ == "__main__":
    main()
