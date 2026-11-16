"""Label-free K16 scoring of cached 288-channel attention ablation models."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from functools import lru_cache

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np
import torch
from clickagent.evidence_retention import RetentionModel
import run_review_attention_finegrained_v1 as worker

OUT = worker.OUT
REG = OUT / "score_registration.json"
RUN_KIND = "cpu_v1"
RUN_ROOTS = {"cpu_v1": worker.OUT,
             "gpu_v2": ROOT / "runs/review_attention_finegrained_gpu_v2"}
PUBLIC = {
    "pro": ROOT / "runs/paper_scale_ablation_v2/eval/pro/public.json",
    "sheets": ROOT / "runs/paper_scale_ablation_v2/eval/sheets/public.json",
    "mmbench": ROOT / "runs/mmbench_gui_new_source_v1/public.json",
}
FEATURES = {
    "pro": ROOT / "runs/paper_scale_ablation_v2/eval/pro/features/index.json",
    "sheets": ROOT / "runs/paper_scale_ablation_v2/eval/sheets/features/index.json",
    "mmbench": ROOT / "runs/mmbench_gui_new_source_v1/mmbench/features/index.json",
}
ARCHIVED_BUNDLES = {
    "pro": ROOT / "runs/paper_scale_ablation_v2/eval/pro/predictions.json",
    "sheets": ROOT / "runs/paper_scale_ablation_v2/eval/sheets/predictions.json",
    "mmbench": ROOT / "runs/mmbench_full_ablation_supplement_v1/predictions.json",
}
ARCHIVED_CASES = {
    "pro": ROOT / "runs/paper_scale_ablation_v2/eval/pro/predictions/cases",
    "sheets": ROOT / "runs/paper_scale_ablation_v2/eval/sheets/predictions/cases",
    "mmbench": ROOT / "runs/mmbench_full_ablation_supplement_v1/predictions/cases",
}
EXPECTED_N = {"pro": 500, "sheets": 500, "mmbench": 1000}


def select_run(kind: str) -> None:
    global OUT, REG, RUN_KIND
    worker.require(kind in RUN_ROOTS, "Unknown training run")
    RUN_KIND = kind
    OUT = RUN_ROOTS[kind]
    REG = OUT / "score_registration.json"


def training_gate() -> tuple[dict, Path]:
    scientific = worker.gate()
    train_reg = OUT / "protocol.json"
    if RUN_KIND == "cpu_v1":
        worker.require(OUT == worker.OUT and train_reg == worker.REG,
                       "CPU V1 root drift")
    else:
        import run_review_attention_finegrained_gpu_v2 as gpu
        worker.require(OUT == gpu.OUT and train_reg == gpu.REG,
                       "GPU V2 root drift")
        p = gpu.gate()
        worker.require(p["parent_protocol_sha256"] == worker.sha(worker.REG) and
                       p["variants"] == list(worker.VARIANTS) and
                       p["training"] == scientific["training"] and
                       p["evaluation"] == scientific["evaluation"],
                       "GPU V2 scientific settings drift")
        choice = OUT / "execution_choice.json"
        worker.require(choice.exists() and gpu.read(choice)["selected_execution"] == "gpu_v2_all18",
                       "GPU V2 was not selected by training-only benchmark")
        completion = OUT / "models/completion.json"
        if completion.exists():
            worker.require(gpu.read(completion)["gpu_choice_sha256"] == worker.sha(choice),
                           "GPU V2 model/choice receipt drift")
    return scientific, train_reg


def score_status(phase: str, **fields) -> None:
    value = dict(phase=phase, at_utc=worker.now(), run_kind=RUN_KIND, **fields)
    worker.write(OUT / "score_status.json", value)
    print(json.dumps(value), flush=True)


def frozen_models(train_reg: Path) -> dict:
    completion = worker.read(OUT / "models/completion.json")
    worker.require(completion["complete"] and completion["count"] == 18 and
                   completion["protocol_sha256"] == worker.sha(train_reg),
                   "All 18 models must be frozen")
    names = {f"{variant}_s{i}" for variant in worker.VARIANTS for i in (1, 2, 3)}
    worker.require(set(completion["models"]) == names,
                   "Frozen 18-model registry names drift")
    hashes = {}
    for variant in worker.VARIANTS:
        for i in (1, 2, 3):
            name = f"{variant}_s{i}"
            receipt_path = OUT / f"models/{name}.json"
            worker.require(completion["models"][name] == worker.sha(receipt_path),
                           "Frozen model receipt SHA drift: " + name)
            model = worker.read(receipt_path)
            worker.require(model["complete"] and
                           model["checkpoint_sha256"] == worker.sha(model["checkpoint_path"]) and
                           model["steps"] == worker.STEPS and
                           model["protocol_sha256"] == worker.sha(train_reg) and
                           model["name"] == name and
                           model["seed"] == worker.SEEDS[i-1] and
                           Path(model["checkpoint_path"]).resolve() ==
                               (OUT / f"models/{name}.pt").resolve(),
                           "Frozen model checkpoint/step/seed drift: " + name)
            hashes[name] = model["checkpoint_sha256"]
    return hashes


def case_key(sample_id: str) -> str:
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:24]


def register() -> None:
    p, train_reg = training_gate()
    if not REG.exists():
        worker.require(not (OUT / "predictions/completion.json").exists() and
                       not any((OUT / "predictions").glob("*/cases/*.json")),
                       "Scoring registration must precede every new prediction")
    model_hashes = frozen_models(train_reg)
    sources = {"public": {k: worker.sha(v) for k, v in PUBLIC.items()},
               "features": {k: worker.sha(v) for k, v in FEATURES.items()},
               "archived_bundles": {k: worker.sha(v) for k, v in ARCHIVED_BUNDLES.items()}}
    expected = dict(registered_at_utc=worker.now(), run_kind=RUN_KIND,
                    train_protocol_sha256=worker.sha(train_reg),
                    worker_protocol_sha256=worker.sha(worker.REG),
                    worker_script_sha256=worker.sha(worker.__file__),
                    scoring_script_sha256=worker.sha(__file__),
                    model_completion_sha256=worker.sha(OUT / "models/completion.json"),
                    model_hashes=model_hashes, reference_models=p["reference_models"],
                    source_sha256=sources, cohorts=EXPECTED_N,
                    candidate_rule="original cached K16, sorted union indices, stable Top1",
                    reference_parity="Re-score frozen full_attention_only S1/S2/S3 on each K16 and require identical archived point/index before sealing new predictions",
                    labels_may_open=False, no_backbone_or_proposer_call=True)
    if REG.exists():
        old = worker.read(REG)
        expected["registered_at_utc"] = old["registered_at_utc"]
        worker.require(old == expected, "Frozen score registration drift")
    else:
        worker.write(REG, expected)
    print(json.dumps({"stage": "registered", "score_registration_sha256": worker.sha(REG)}), flush=True)


def gate() -> dict:
    _, train_reg = training_gate()
    worker.require(REG.exists(), "Register score protocol before prediction")
    p = worker.read(REG)
    worker.require(p["run_kind"] == RUN_KIND and
                   p["train_protocol_sha256"] == worker.sha(train_reg) and
                   p["worker_protocol_sha256"] == worker.sha(worker.REG) and
                   p["worker_script_sha256"] == worker.sha(worker.__file__) and
                   p["scoring_script_sha256"] == worker.sha(__file__) and
                   p["model_completion_sha256"] == worker.sha(OUT / "models/completion.json"),
                   "Scoring source/model drift")
    worker.require(p["source_sha256"] == {
        "public": {k: worker.sha(v) for k, v in PUBLIC.items()},
        "features": {k: worker.sha(v) for k, v in FEATURES.items()},
        "archived_bundles": {k: worker.sha(v) for k, v in ARCHIVED_BUNDLES.items()}},
        "Cached public/features/archive drift")
    worker.require(p["model_hashes"] == frozen_models(train_reg),
                   "18 registered model checkpoint/receipt SHA drift")
    return p


def analysis_ready() -> str:
    # Import only at runtime to avoid a module cycle; this checks registration
    # metadata and code/scorer hashes, without invoking the prediction gate or labels.
    import analyze_review_attention_finegrained_v1 as analysis
    analysis.select_run(RUN_KIND)
    analysis.registration_gate()
    return worker.sha(analysis.REG)


def prediction_start(score_reg: dict, *, allow_create: bool) -> str:
    analysis_sha = analysis_ready()
    path = OUT / "predictions/start.json"
    value = dict(run_kind=RUN_KIND,
                 score_registration_sha256=worker.sha(REG),
                 analysis_registration_sha256=analysis_sha,
                 model_hashes=score_reg["model_hashes"],
                 labels_opened=False)
    if path.exists():
        worker.require(worker.read(path) == value,
                       "Prediction start registration drift")
    elif allow_create:
        worker.write(path, value)
    else:
        raise ValueError("Prediction start receipt missing before sealed cases")
    return worker.sha(path)


def load_models(p: dict):
    models = {}
    reference_mean = reference_std = None
    for name, digest in p["model_hashes"].items():
        receipt = worker.read(OUT / f"models/{name}.json")
        worker.require(worker.sha(receipt["checkpoint_path"]) == digest,
                       "Checkpoint drift: " + name)
        saved = torch.load(receipt["checkpoint_path"], map_location="cpu", weights_only=True)
        model = RetentionModel().cpu().eval()
        model.load_state_dict(saved["state_dict"])
        mean, std = saved["mean"].float(), saved["std"].float()
        if reference_mean is None:
            reference_mean, reference_std = mean, std
        else:
            worker.require(torch.equal(mean, reference_mean) and torch.equal(std, reference_std),
                           "Different training base32 scaler")
        models[name] = model
    for i in (1, 2, 3):
        name = f"full_attention_only_s{i}"
        entry = p["reference_models"][name]
        worker.require(worker.sha(entry["path"]) == entry["sha256"],
                       "Archived reference checkpoint drift")
        saved = torch.load(entry["path"], map_location="cpu", weights_only=True)
        worker.require(torch.equal(saved["mean"].float(), reference_mean) and
                       torch.equal(saved["std"].float(), reference_std),
                       "Archived attention-only scaler differs")
        model = RetentionModel().cpu().eval()
        model.load_state_dict(saved["state_dict"])
        models[name] = model
    return models, reference_mean, reference_std


def read_cached_case(entry: dict):
    for field, digest in (("path", "sha256"), ("state_path", "state_sha256"),
                          ("mask_path", "mask_sha256"), ("pipeline_path", "pipeline_sha256")):
        worker.require(worker.sha(entry[field]) == entry[digest], "Cached feature drift: " + field)
    with np.load(entry["path"], allow_pickle=False) as z:
        points = z["points"].copy()
        base = z["base"].copy()
        active = z["active"].copy()
        pool = np.sort(z["baseline"].copy())
    worker.require(points.shape == (entry["n"], 2) and base.shape == (entry["n"], 32)
                   and active.shape == (entry["n"], 4) and len(pool) <= 16 and
                   len(np.unique(pool)) == len(pool) and
                   bool(((pool >= 0) & (pool < len(points))).all()),
                   "Cached K16/base/point shapes drift")
    state = np.load(entry["state_path"], mmap_mode="r", allow_pickle=False)
    mask = np.load(entry["mask_path"], mmap_mode="r", allow_pickle=False)
    worker.require(state.shape[0] == len(points) and state.shape[1] == 4 and
                   state.shape[-1] == 294 and mask.shape == state.shape[:-1],
                   "Cached state/mask shape drift")
    return points, base, active, pool, state, mask


@lru_cache(maxsize=3)
def archived_case_hashes(suite: str) -> dict:
    return worker.read(ARCHIVED_BUNDLES[suite])["case_hashes"]


def old_case(suite: str, sample_id: str):
    path = ARCHIVED_CASES[suite] / (case_key(sample_id) + ".json")
    worker.require(worker.sha(path) == archived_case_hashes(suite)[sample_id],
                   "Archived prediction case SHA drift")
    return worker.read(path)


@torch.inference_mode()
def score_one(entry: dict, suite: str, models: dict, mean: torch.Tensor,
              std: torch.Tensor):
    points, base, active, pool, state, mask = read_cached_case(entry)
    archive = old_case(suite, entry["sample_id"])
    worker.require(archive["sample_id"] == entry["sample_id"] and
                   archive["feature_sha256"] == entry["sha256"],
                   "Archived same-feature identity drift")
    if len(pool):
        # Match the original NumPy normalization in prior.tensors exactly.
        args = dict(base=torch.from_numpy((base[pool] - mean.numpy()) / std.numpy()),
                    active=torch.from_numpy(active[pool].copy()),
                    state=torch.from_numpy(np.asarray(state[pool]).copy()),
                    mask=torch.from_numpy(np.asarray(mask[pool]).copy()))
        inputs = {variant: worker.transformed(args, variant) for variant in worker.VARIANTS}
        inputs["attention_only_identity"] = worker.transformed(args, "attention_only_identity")
    result = {}
    for name, model in models.items():
        if len(pool):
            variant = ("attention_only_identity" if name.startswith("full_attention_only") else
                       name.rsplit("_s", 1)[0])
            score = model(**inputs[variant]).float().numpy()
            worker.require(score.shape == (len(pool),) and np.isfinite(score).all(),
                           "Nonfinite or wrong score shape: " + name)
            ix = int(pool[np.argsort(-score, kind="stable")[0]])
            point = points[ix].tolist()
        else:
            ix, point = None, None
        if name.startswith("full_attention_only"):
            old = archive["methods"][name]
            worker.require(old["index"] == ix and old["point"] == point,
                           "Archived attention-only K16 score parity failure: " +
                           suite + "/" + entry["sample_id"] + "/" + name)
        else:
            result[name] = {"index": ix, "point": point}
    return result, len(pool), len(points)


def selftest() -> None:
    worker.low_priority(); torch.set_num_threads(2)
    worker.require(not torch.cuda.is_available(), "Hide CUDA for CPU selftest")
    worker.require(worker.REG.exists(), "Register worker first")
    p = worker.read(worker.REG)
    refs = {}
    for i in (1, 2, 3):
        name = f"full_attention_only_s{i}"
        saved = torch.load(p["reference_models"][name]["path"], map_location="cpu",
                           weights_only=True)
        model = RetentionModel().cpu().eval()
        model.load_state_dict(saved["state_dict"])
        refs[name] = model
    saved = torch.load(p["reference_models"]["full_attention_only_s1"]["path"],
                       map_location="cpu", weights_only=True)
    checks = {}
    for suite in EXPECTED_N:
        entry = worker.read(FEATURES[suite])["cases"][0]
        _, count, union_n = score_one(entry, suite, refs, saved["mean"].float(),
                                       saved["std"].float())
        worker.require(0 <= count <= 16 and union_n >= count,
                       "Selftest K16 size invalid: " + suite)
        checks[suite] = {"K16": count, "union": union_n, "archived_seed_parity": True}
    print(json.dumps({"stage": "score_selftest_pass", "cohorts": checks}), flush=True)


def score(suite: str) -> None:
    worker.low_priority(); torch.set_num_threads(2)
    worker.require(not torch.cuda.is_available(), "Hide CUDA while MVP uses GPUs")
    p = gate()
    start_sha = prediction_start(p, allow_create=True)
    worker.require(suite in EXPECTED_N, "Unknown suite")
    rows = worker.read(PUBLIC[suite]); entries = worker.read(FEATURES[suite])["cases"]
    worker.require(len(rows) == len(entries) == EXPECTED_N[suite] and
                   [r["sample_id"] for r in rows] == [e["sample_id"] for e in entries],
                   "Public/feature row order drift")
    models, mean, std = load_models(p)
    case_hashes = {}
    for j, (row, entry) in enumerate(zip(rows, entries), 1):
        path = OUT / f"predictions/{suite}/cases/{case_key(row['sample_id'])}.json"
        if path.exists():
            saved = worker.read(path)
            worker.require(saved["complete"] and saved["sample_id"] == row["sample_id"] and
                           saved["feature_sha256"] == entry["sha256"] and
                           saved["model_hashes"] == p["model_hashes"] and
                           saved["labels_opened"] is False,
                           "Existing case prediction drift")
        else:
            prediction, k, n = score_one(entry, suite, models, mean, std)
            saved = dict(complete=True, sample_id=row["sample_id"], suite=suite,
                         feature_sha256=entry["sha256"], K16_count=k,
                         candidate_count=n, model_hashes=p["model_hashes"],
                         predictions=prediction, archived_attention_only_parity=True,
                         labels_opened=False)
            worker.write(path, saved)
        case_hashes[row["sample_id"]] = worker.sha(path)
        if j % 25 == 0 or j == len(rows):
            score_status("scoring", suite=suite, completed=j, total=len(rows))
    bundle = dict(complete=True, suite=suite, n=len(rows), case_hashes=case_hashes,
                  public_sha256=p["source_sha256"]["public"][suite],
                  feature_index_sha256=p["source_sha256"]["features"][suite],
                  score_registration_sha256=worker.sha(REG),
                  prediction_start_sha256=start_sha,
                  analysis_registration_sha256=analysis_ready(),
                  model_hashes=p["model_hashes"], labels_opened=False)
    path = OUT / f"predictions/{suite}/bundle.json"
    if path.exists():
        worker.require(worker.read(path) == bundle, "Existing bundle drift")
    else:
        worker.write(path, bundle)
    print(json.dumps({"stage": "suite_sealed", "suite": suite,
                      "bundle_sha256": worker.sha(path)}), flush=True)


def verify_gate() -> None:
    p = gate()
    start_sha = prediction_start(p, allow_create=False)
    for suite, n in EXPECTED_N.items():
        rows = worker.read(PUBLIC[suite]); entries = worker.read(FEATURES[suite])["cases"]
        bundle_path = OUT / f"predictions/{suite}/bundle.json"
        worker.require(bundle_path.exists(), "Prediction bundle missing: " + suite)
        bundle = worker.read(bundle_path)
        worker.require(bundle["complete"] and bundle["n"] == n and
                       bundle["model_hashes"] == p["model_hashes"] and
                       bundle["score_registration_sha256"] == worker.sha(REG) and
                       bundle["prediction_start_sha256"] == start_sha and
                       bundle["analysis_registration_sha256"] == analysis_ready() and
                       bundle["public_sha256"] == p["source_sha256"]["public"][suite] and
                       bundle["feature_index_sha256"] == p["source_sha256"]["features"][suite] and
                       bundle["labels_opened"] is False and
                       len(bundle["case_hashes"]) == n,
                       "Prediction bundle identity/count drift: " + suite)
        worker.require([r["sample_id"] for r in rows] == [e["sample_id"] for e in entries],
                       "Public/feature order drift")
        for row, entry in zip(rows, entries):
            sid = row["sample_id"]
            path = OUT / f"predictions/{suite}/cases/{case_key(sid)}.json"
            worker.require(worker.sha(path) == bundle["case_hashes"][sid],
                           "New case SHA drift: " + sid)
            case = worker.read(path)
            worker.require(case["complete"] and case["sample_id"] == sid and
                           case["feature_sha256"] == entry["sha256"] and
                           case["model_hashes"] == p["model_hashes"] and
                           case["archived_attention_only_parity"] is True and
                           case["labels_opened"] is False and
                           set(case["predictions"]) == set(p["model_hashes"]),
                           "New case identity/method drift: " + sid)
            with np.load(entry["path"], allow_pickle=False) as z:
                pts, pool = z["points"], z["baseline"]
            worker.require(case["K16_count"] == len(pool) and
                           case["candidate_count"] == len(pts), "Case K16 count drift")
            for pred in case["predictions"].values():
                ix, point = pred["index"], pred["point"]
                worker.require((ix is None and point is None and not len(pool)) or
                               (ix is not None and ix in pool and point == pts[ix].tolist()),
                               "Predicted point not in original K16: " + sid)
            old = old_case(suite, sid)
            worker.require(old["feature_sha256"] == entry["sha256"],
                           "Archived same-feature drift")
    output = dict(complete=True, labels_opened=False, cohorts=EXPECTED_N,
                  score_registration_sha256=worker.sha(REG),
                  prediction_start_sha256=start_sha,
                  analysis_registration_sha256=analysis_ready(),
                  bundle_sha256={suite: worker.sha(OUT / f"predictions/{suite}/bundle.json")
                                 for suite in EXPECTED_N})
    path = OUT / "predictions/completion.json"
    if path.exists():
        worker.require(worker.read(path) == output, "Global prediction seal drift")
    else:
        worker.write(path, output)
    print(json.dumps({"stage": "verify_gate_pass", "labels_opened": False,
                      "global_seal_sha256": worker.sha(path)}), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", choices=tuple(RUN_ROOTS), default="cpu_v1")
    parser.add_argument("--stage", required=True,
                        choices=("selftest", "register", "score-pro", "score-sheets",
                                 "score-mmbench", "verify-gate"))
    args = parser.parse_args()
    select_run(args.run_root)
    if args.stage == "selftest": selftest()
    elif args.stage == "register": register()
    elif args.stage.startswith("score-"): score(args.stage.removeprefix("score-"))
    else: verify_gate()


if __name__ == "__main__":
    main()
