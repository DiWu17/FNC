"""Five one-seed cached attention variants with sealed predictions before labels."""
from __future__ import annotations

import argparse
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
import run_review_attention_head_mean_single_v1 as v3

TRAIN = ROOT / "runs/review_attention_finegrained_gpu_v4"
OUT = TRAIN / "five_arm_eval"
TRAIN_REG = TRAIN / "protocol.json"
TRAIN_WORKER = ROOT / "scripts/run_review_attention_finegrained_gpu_v4.py"
SCORE_REG = OUT / "score_registration.json"
ANALYSIS_REG = OUT / "analysis_registration.json"
LEGACY_REG = v3.LEGACY_REG
ARMS = ("head_mean_per_layer", "cell_only", "peak_only", "neighborhood_only", "cell_peak")
NAMES = tuple(f"{arm}_s1" for arm in ARMS)
ATTN_REF = "full_attention_only_s1"
HEAD_REF = "head_mean_all96_s1"
COHORTS = {"pro": 500, "sheets": 500, "mmbench": 1000}
V4_PROTOCOL_SHA256 = "2d9c7ad9eba12205a1d468b5e3fb8e3b7d52f1d2dc3a4295ce5763eddc9fa853"


def require(ok: bool, message: str) -> None:
    fine.require(ok, message)


def sha(path: str | Path) -> str:
    return fine.sha(path)


def read(path: str | Path):
    return fine.read(path)


def write_once(path: Path, value) -> None:
    if path.exists():
        require(read(path) == value, "Immutable receipt drift: " + str(path))
    else:
        fine.write(path, value)


def text_once(path: Path, value: str) -> None:
    if path.exists():
        require(path.read_text(encoding="utf-8") == value,
                "Immutable report drift: " + str(path))
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(value, encoding="utf-8")
        tmp.replace(path)


def status(phase: str, **fields) -> None:
    obj = dict(phase=phase, at_utc=fine.now(), **fields)
    fine.write(OUT / "status.json", obj)
    print(json.dumps(obj), flush=True)


def v3_seal_gate() -> tuple[dict, dict]:
    v3.registration_gate()  # Metadata/checkpoints only, never opens labels.
    completion_path = v3.OUT / "predictions/completion.json"
    completion = read(completion_path)
    require(completion["complete"] and completion["labels_opened"] is False and
            completion["cohorts"] == COHORTS and
            completion["score_registration_sha256"] == sha(v3.SCORE_REG) and
            completion["analysis_registration_sha256"] == sha(v3.ANALYSIS_REG),
            "V3 head-mean prediction seal drift")
    bundles = {}
    for suite, n in COHORTS.items():
        path = v3.OUT / f"predictions/{suite}/bundle.json"
        require(completion["bundle_sha256"][suite] == sha(path),
                "V3 head-mean bundle SHA drift: " + suite)
        b = read(path)
        require(b["complete"] and b["labels_opened"] is False and
                b["n"] == n and len(b["case_hashes"]) == n,
                "V3 head-mean bundle count drift: " + suite)
        bundles[suite] = b
    return completion, bundles


def training_gate() -> tuple[dict, dict, dict]:
    parent = fine.gate()
    p = read(TRAIN_REG)
    expected_training = dict(parent["training"], seeds=[20261311])
    require(sha(TRAIN_REG) == V4_PROTOCOL_SHA256 and
            p["parent_protocol_sha256"] == sha(fine.REG) and
            p["parent_worker_sha256"] == sha(fine.__file__) and
            p["worker_sha256"] == sha(TRAIN_WORKER) and
            p["source_sha256"] == parent["source_sha256"] and
            p["training"] == expected_training and
            p["evaluation"] == parent["evaluation"] and
            p["variants"] == list(ARMS) and
            p["model_order"] == list(NAMES) and p["seed"] == 20261311 and
            p["v3_reference"]["protocol_sha256"] == sha(v3.TRAIN_REG) and
            p["v3_reference"]["worker_sha256"] == sha(v3.TRAIN_WORKER) and
            p["v3_reference"]["model_completion_sha256"] ==
                sha(v3.TRAIN / "models/completion.json") and
            p["v3_reference"]["model_receipt_sha256"] ==
                sha(v3.TRAIN / f"models/{HEAD_REF}.json") and
            p["v3_reference"]["checkpoint_sha256"] ==
                read(v3.SCORE_REG)["checkpoint_sha256"],
            "Frozen V4 source/schedule/arm drift")
    completion = read(TRAIN / "models/completion.json")
    require(completion["complete"] and completion["count"] == 5 and
            completion["protocol_sha256"] == sha(TRAIN_REG) and
            set(completion["models"]) == set(NAMES),
            "Five model completion count/identity drift")
    original_seed = read(fine.V18 / "models/full_s1.json")["initial_state_sha256"]
    receipts = {}
    for name in NAMES:
        path = TRAIN / f"models/{name}.json"
        require(completion["models"][name] == sha(path),
                "V4 model receipt SHA drift: " + name)
        r = read(path)
        require(r["complete"] and r["name"] == name and
                r["seed"] == 20261311 and r["steps"] == fine.STEPS and
                r["fitting_queries"] == 12886 and r["empty_skipped"] == 487 and
                r["parameter_count"] == 22050 and
                r["initial_state_sha256"] == original_seed and
                r["protocol_sha256"] == sha(TRAIN_REG) and
                Path(r["checkpoint_path"]).resolve() == (TRAIN / f"models/{name}.pt").resolve() and
                r["checkpoint_sha256"] == sha(r["checkpoint_path"]),
                "V4 model checkpoint/schedule drift: " + name)
        receipts[name] = r
    return p, completion, receipts


def source_sha() -> dict:
    return {"public": {s: sha(p) for s, p in common.PUBLIC.items()},
            "features": {s: sha(p) for s, p in common.FEATURES.items()},
            "archived_attention": {s: sha(p) for s, p in common.ARCHIVED_BUNDLES.items()}}


def register() -> None:
    p, model_completion, receipts = training_gate()
    v3_completion, _ = v3_seal_gate()
    if not SCORE_REG.exists() or not ANALYSIS_REG.exists():
        require(not any((OUT / "predictions").glob("*/cases/*.json")) and
                not (OUT / "predictions/completion.json").exists(),
                "Both registrations must precede first V4 prediction")
    score = dict(registered_at_utc=fine.now(), purpose="Five prespecified attention input variants, one seed each",
                 script_sha256=sha(__file__), common_scoring_sha256=sha(common.__file__),
                 v3_single_eval_script_sha256=sha(v3.__file__),
                 train_protocol_sha256=sha(TRAIN_REG), train_worker_sha256=sha(TRAIN_WORKER),
                 model_completion_sha256=sha(TRAIN / "models/completion.json"),
                 model_receipt_sha256=model_completion["models"],
                 model_checkpoint_sha256={n: receipts[n]["checkpoint_sha256"] for n in NAMES},
                 v3_prediction_completion_sha256=sha(v3.OUT / "predictions/completion.json"),
                 v3_model_checkpoint_sha256=read(v3.SCORE_REG)["checkpoint_sha256"],
                 old_attention_checkpoint_sha256=fine.reference_models()[ATTN_REF]["sha256"],
                 source_sha256=source_sha(), cohorts=COHORTS,
                 arms=list(ARMS), seed=20261311,
                 candidate_rule="Original cached K16, sorted union indices, stable forced Top1",
                 labels_opened=False, no_backbone_or_proposer_call=True)
    if SCORE_REG.exists():
        score["registered_at_utc"] = read(SCORE_REG)["registered_at_utc"]
    write_once(SCORE_REG, score)
    legacy = read(LEGACY_REG)
    analysis = dict(registered_at_utc=fine.now(), score_registration_sha256=sha(SCORE_REG),
                    analysis_script_sha256=sha(__file__),
                    legacy_label_digest_registration_sha256=sha(LEGACY_REG),
                    label_sha256_expected_without_opening=legacy["label_sha256_unopened"],
                    official_pointer_scorer_sha256={s: sha(path) for s, path in v3.POINTER_SCORER.items()},
                    official_mmbench_scorer_sha256=sha(v3.MM_SCORER),
                    gate="All 2000 cases x five new arms, per-case/bundle/global SHA before target geometry",
                    metric="Original K16 forced Top1, null incorrect, pinned official scorer pointwise parity",
                    comparisons="Each V4 S1 arm versus archived attention-only S1 and sealed V3 head-mean S1 on each fixed cohort",
                    inference="Descriptive single-seed trend only; no confidence interval, significance, best-arm selection or three-seed claim",
                    limitations=["Retrospective cohorts, PointerBench labels analyzed earlier",
                                 "At least seven PointerBench answers visible before prior V2 sealing",
                                 "Selected MMBench 1000 is not full official leaderboard"])
    if ANALYSIS_REG.exists():
        analysis["registered_at_utc"] = read(ANALYSIS_REG)["registered_at_utc"]
    write_once(ANALYSIS_REG, analysis)
    status("registered", score_registration_sha256=sha(SCORE_REG),
           analysis_registration_sha256=sha(ANALYSIS_REG),
           v3_prediction_completion_sha256=score["v3_prediction_completion_sha256"],
           labels_opened=False)


def registration_gate() -> tuple[dict, dict, dict]:
    _, model_completion, receipts = training_gate()
    v3_seal_gate()
    require(SCORE_REG.exists() and ANALYSIS_REG.exists(),
            "Register score and analysis before V4 prediction")
    s, a = read(SCORE_REG), read(ANALYSIS_REG)
    require(s["script_sha256"] == sha(__file__) and
            s["common_scoring_sha256"] == sha(common.__file__) and
            s["v3_single_eval_script_sha256"] == sha(v3.__file__) and
            s["train_protocol_sha256"] == sha(TRAIN_REG) and
            s["train_worker_sha256"] == sha(TRAIN_WORKER) and
            s["model_completion_sha256"] == sha(TRAIN / "models/completion.json") and
            s["model_receipt_sha256"] == model_completion["models"] and
            s["model_checkpoint_sha256"] ==
                {n: receipts[n]["checkpoint_sha256"] for n in NAMES} and
            s["v3_prediction_completion_sha256"] == sha(v3.OUT / "predictions/completion.json") and
            s["v3_model_checkpoint_sha256"] == read(v3.SCORE_REG)["checkpoint_sha256"] and
            s["old_attention_checkpoint_sha256"] == fine.reference_models()[ATTN_REF]["sha256"] and
            s["source_sha256"] == source_sha() and s["cohorts"] == COHORTS and
            s["arms"] == list(ARMS) and s["seed"] == 20261311 and
            s["labels_opened"] is False,
            "Frozen five-arm scoring registration drift")
    require(a["score_registration_sha256"] == sha(SCORE_REG) and
            a["analysis_script_sha256"] == sha(__file__) and
            a["legacy_label_digest_registration_sha256"] == sha(LEGACY_REG) and
            a["label_sha256_expected_without_opening"] ==
                read(LEGACY_REG)["label_sha256_unopened"] and
            a["official_pointer_scorer_sha256"] ==
                {suite: sha(path) for suite, path in v3.POINTER_SCORER.items()} and
            a["official_mmbench_scorer_sha256"] == sha(v3.MM_SCORER),
            "Frozen five-arm analysis registration drift")
    return s, a, receipts


def start_gate(*, allow_create: bool) -> str:
    s, _, _ = registration_gate()
    path = OUT / "predictions/start.json"
    value = dict(score_registration_sha256=sha(SCORE_REG),
                 analysis_registration_sha256=sha(ANALYSIS_REG),
                 model_checkpoint_sha256=s["model_checkpoint_sha256"],
                 v3_prediction_completion_sha256=s["v3_prediction_completion_sha256"],
                 labels_opened=False)
    if path.exists():
        require(read(path) == value, "V4 prediction-start drift")
    elif allow_create:
        write_once(path, value)
    else:
        raise ValueError("V4 prediction-start receipt missing")
    return sha(path)


def v3_case(suite: str, sid: str, bundles: dict) -> dict:
    path = v3.OUT / f"predictions/{suite}/cases/{common.case_key(sid)}.json"
    require(sha(path) == bundles[suite]["case_hashes"][sid],
            "V3 reference case SHA drift: " + sid)
    case = read(path)
    require(case["complete"] and case["sample_id"] == sid and
            case["checkpoint_sha256"] == read(v3.SCORE_REG)["checkpoint_sha256"] and
            case["labels_opened"] is False,
            "V3 reference case identity drift")
    return case


def load_models(s: dict, receipts: dict) -> tuple[dict, torch.Tensor, torch.Tensor]:
    models = {}
    mean = std = None
    for name in NAMES:
        saved = torch.load(receipts[name]["checkpoint_path"], map_location="cpu", weights_only=True)
        require(saved["seed"] == 20261311 and
                saved["mean"].shape == (32,) and saved["std"].shape == (32,),
                "V4 checkpoint scaler/seed drift: " + name)
        if mean is None:
            mean, std = saved["mean"].float(), saved["std"].float()
        else:
            require(torch.equal(mean, saved["mean"].float()) and
                    torch.equal(std, saved["std"].float()),
                    "V4 training-only scalers differ")
        model = RetentionModel().cpu().eval()
        model.load_state_dict(saved["state_dict"])
        models[name] = model
    ref_path = fine.reference_models()[ATTN_REF]["path"]
    frozen = torch.load(ref_path, map_location="cpu", weights_only=True)
    require(torch.equal(mean, frozen["mean"].float()) and
            torch.equal(std, frozen["std"].float()),
            "V4 scaler differs from archived attention-only S1")
    ref = RetentionModel().cpu().eval()
    ref.load_state_dict(frozen["state_dict"])
    models[ATTN_REF] = ref
    return models, mean, std


def score_suite(suite: str) -> None:
    fine.low_priority(); torch.set_num_threads(2)
    require(not torch.cuda.is_available(), "Hide CUDA for CPU-only cached V4 scoring")
    s, _, receipts = registration_gate()
    start_sha = start_gate(allow_create=True)
    require(suite in COHORTS, "Unknown V4 cohort")
    _, v3_bundles = v3_seal_gate()
    rows, entries = read(common.PUBLIC[suite]), read(common.FEATURES[suite])["cases"]
    require(len(rows) == len(entries) == COHORTS[suite] and
            [r["sample_id"] for r in rows] == [e["sample_id"] for e in entries],
            "V4 public/feature order drift")
    models, mean, std = load_models(s, receipts)
    hashes = {}
    for j, (row, entry) in enumerate(zip(rows, entries), 1):
        sid = row["sample_id"]
        path = OUT / f"predictions/{suite}/cases/{common.case_key(sid)}.json"
        v3_ref = v3_case(suite, sid, v3_bundles)
        require(v3_ref["feature_sha256"] == entry["sha256"],
                "V3 and V4 cached feature differs: " + sid)
        if path.exists():
            case = read(path)
            require(case["complete"] and case["sample_id"] == sid and
                    case["feature_sha256"] == entry["sha256"] and
                    case["model_checkpoint_sha256"] == s["model_checkpoint_sha256"] and
                    case["v3_reference_case_sha256"] == v3_bundles[suite]["case_hashes"][sid] and
                    case["labels_opened"] is False,
                    "Existing V4 prediction case drift")
        else:
            pred, pool_n, union_n = common.score_one(entry, suite, models, mean, std)
            require(set(pred) == set(NAMES), "Unexpected V4 prediction names")
            require(v3_ref["K16_count"] == pool_n and
                    v3_ref["candidate_count"] == union_n,
                    "V3/V4 original K16/union count mismatch")
            case = dict(complete=True, suite=suite, sample_id=sid,
                        feature_sha256=entry["sha256"], K16_count=pool_n,
                        candidate_count=union_n,
                        model_checkpoint_sha256=s["model_checkpoint_sha256"],
                        v3_reference_case_sha256=v3_bundles[suite]["case_hashes"][sid],
                        predictions=pred, archived_attention_only_parity=True,
                        labels_opened=False)
            write_once(path, case)
        hashes[sid] = sha(path)
        if j % 25 == 0 or j == len(rows):
            status("scoring", suite=suite, completed=j, total=len(rows), labels_opened=False)
    bundle = dict(complete=True, suite=suite, n=len(rows), case_hashes=hashes,
                  public_sha256=s["source_sha256"]["public"][suite],
                  feature_index_sha256=s["source_sha256"]["features"][suite],
                  prediction_start_sha256=start_sha,
                  score_registration_sha256=sha(SCORE_REG),
                  analysis_registration_sha256=sha(ANALYSIS_REG),
                  model_checkpoint_sha256=s["model_checkpoint_sha256"],
                  v3_reference_bundle_sha256=sha(v3.OUT / f"predictions/{suite}/bundle.json"),
                  labels_opened=False)
    path = OUT / f"predictions/{suite}/bundle.json"
    write_once(path, bundle)
    status("suite_sealed", suite=suite, bundle_sha256=sha(path), labels_opened=False)


def verify_gate() -> None:
    s, _, _ = registration_gate()
    start_sha = start_gate(allow_create=False)
    _, v3_bundles = v3_seal_gate()
    for suite, n in COHORTS.items():
        rows, entries = read(common.PUBLIC[suite]), read(common.FEATURES[suite])["cases"]
        bundle_path = OUT / f"predictions/{suite}/bundle.json"
        bundle = read(bundle_path)
        require(bundle["complete"] and bundle["n"] == n and
                bundle["labels_opened"] is False and
                bundle["prediction_start_sha256"] == start_sha and
                bundle["score_registration_sha256"] == sha(SCORE_REG) and
                bundle["analysis_registration_sha256"] == sha(ANALYSIS_REG) and
                bundle["model_checkpoint_sha256"] == s["model_checkpoint_sha256"] and
                bundle["public_sha256"] == s["source_sha256"]["public"][suite] and
                bundle["feature_index_sha256"] == s["source_sha256"]["features"][suite] and
                bundle["v3_reference_bundle_sha256"] ==
                    sha(v3.OUT / f"predictions/{suite}/bundle.json") and
                len(bundle["case_hashes"]) == n,
                "V4 bundle count/SHA drift: " + suite)
        require([r["sample_id"] for r in rows] == [e["sample_id"] for e in entries],
                "V4 cohort order drift: " + suite)
        for row, entry in zip(rows, entries):
            sid = row["sample_id"]
            path = OUT / f"predictions/{suite}/cases/{common.case_key(sid)}.json"
            require(sha(path) == bundle["case_hashes"][sid], "V4 case SHA drift: " + sid)
            case = read(path)
            pts, _, _, pool, _, _ = common.read_cached_case(entry)
            ref = v3_case(suite, sid, v3_bundles)
            archived = common.old_case(suite, sid)
            require(case["complete"] and case["sample_id"] == sid and
                    case["feature_sha256"] == entry["sha256"] and
                    case["model_checkpoint_sha256"] == s["model_checkpoint_sha256"] and
                    case["v3_reference_case_sha256"] == v3_bundles[suite]["case_hashes"][sid] and
                    case["K16_count"] == ref["K16_count"] == archived["K16_count"] == len(pool) and
                    case["candidate_count"] == ref["candidate_count"] == len(pts) and
                    archived["feature_sha256"] == ref["feature_sha256"] == entry["sha256"] and
                    case["archived_attention_only_parity"] is True and
                    case["labels_opened"] is False and set(case["predictions"]) == set(NAMES),
                    "V4 case/reference feature/K16 drift: " + sid)
            for pred in (*case["predictions"].values(), ref["prediction"],
                         archived["methods"][ATTN_REF]):
                ix, point = pred["index"], pred["point"]
                require((ix is None and point is None and not len(pool)) or
                        (ix is not None and ix in pool and point == pts[ix].tolist()),
                        "V4 or reference point outside original K16: " + sid)
    completion = dict(complete=True, labels_opened=False, cohorts=COHORTS,
                      score_registration_sha256=sha(SCORE_REG),
                      analysis_registration_sha256=sha(ANALYSIS_REG),
                      prediction_start_sha256=start_sha,
                      bundle_sha256={s: sha(OUT / f"predictions/{s}/bundle.json") for s in COHORTS})
    write_once(OUT / "predictions/completion.json", completion)
    status("prediction_gate_pass", completion_sha256=sha(OUT / "predictions/completion.json"),
           labels_opened=False)


def evaluate_suite(suite: str, pointer: dict, mm: dict, v3_bundles: dict) -> dict:
    from clickagent.public_evaluation import point_in_geometry
    from run_mmbench_gui_l2_v1 import official_hit, source_reference_hit

    rows = read(common.PUBLIC[suite])
    if suite == "mmbench":
        native = source_reference_hit()
    else:
        spec = importlib.util.spec_from_file_location(f"five_arm_pointer_{suite}", v3.POINTER_SCORER[suite])
        require(spec is not None and spec.loader is not None, "Official Pointer scorer import failed")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        native = getattr(module, "_in_bbox", None)
        require(callable(native), "Official Pointer _in_bbox missing")
    names = (*NAMES, HEAD_REF, ATTN_REF)
    hits = {name: 0 for name in names}
    wins = {name: 0 for name in NAMES}
    losses = {name: 0 for name in NAMES}
    checked = nulls = 0
    for row in rows:
        sid = row["sample_id"]
        case = read(OUT / f"predictions/{suite}/cases/{common.case_key(sid)}.json")
        v3ref = v3_case(suite, sid, v3_bundles)
        archived = common.old_case(suite, sid)
        outcome = {}
        for name in names:
            point = (case["predictions"][name]["point"] if name in NAMES else
                     v3ref["prediction"]["point"] if name == HEAD_REF else
                     archived["methods"][ATTN_REF]["point"])
            if suite == "mmbench":
                source = mm[row["source_index"]]
                require(all(source[k] == row[k] for k in
                            ("instruction", "image_size", "data_type", "platform", "grounding_type")),
                        "MMBench public/annotation identity drift")
                local = official_hit(point, source["bbox"], row["image_size"])
                official = (bool(native(source["bbox"].copy(), tuple(row["image_size"]), tuple(point)))
                            if point is not None else False)
            else:
                w, h = row["image_size"]
                label = pointer[sid]
                local = (bool(point_in_geometry((point[0]/w, point[1]/h), label["target_geometry"]))
                         if point is not None else False)
                official = (bool(native(tuple(point), label["bbox_pixels"]))
                            if point is not None else False)
            require(bool(local) == bool(official),
                    "V4 official/local scorer parity drift: " + suite + "/" + sid + "/" + name)
            outcome[name] = int(bool(local))
            hits[name] += outcome[name]
            checked += 1
            nulls += point is None
        for name in NAMES:
            wins[name] += outcome[name] == 1 and outcome[ATTN_REF] == 0
            losses[name] += outcome[name] == 0 and outcome[ATTN_REF] == 1
    n = len(rows)
    return dict(n=n, hits=hits,
                accuracy_pct={name: 100*hits[name]/n for name in names},
                delta_vs_attention_only_s1_pp={name: 100*(hits[name]-hits[ATTN_REF])/n for name in NAMES},
                delta_vs_head_mean_all96_s1_pp={name: 100*(hits[name]-hits[HEAD_REF])/n for name in NAMES},
                paired_wins_vs_attention_only_s1=wins,
                paired_losses_vs_attention_only_s1=losses,
                official_scorer_parity=dict(passed=True, all_points_checked=checked,
                                            nonnull_points=checked-nulls, null_points=nulls))


def report(result: dict) -> str:
    lines = ["# Five one-seed attention feature ablations", "",
             "All five variants use the same original K16 cached candidate pool, full training union, 13,376 updates, 22,050-parameter readout, and seed 20261311. Predictions for all 2,000 queries were SHA sealed before target annotations were read. These are descriptive single-seed trends, not three-seed estimates or significance tests.", ""]
    for suite, title in (("pro", "PointerBench Pro500"), ("sheets", "PointerBench Sheets500"),
                         ("mmbench", "MMBench-GUI selected L2 1000")):
        x = result["cohorts"][suite]
        lines += [f"## {title}", "",
                  "| Variant | Correct | Accuracy % | Δ vs attention-only S1 (pp) | Δ vs 96-head mean S1 (pp) |",
                  "| --- | ---: | ---: | ---: | ---: |"]
        for name in (ATTN_REF, HEAD_REF, *NAMES):
            d0 = "reference" if name == ATTN_REF else (
                f"{x['accuracy_pct'][name]-x['accuracy_pct'][ATTN_REF]:+.2f}")
            d1 = "reference" if name == HEAD_REF else (
                f"{x['accuracy_pct'][name]-x['accuracy_pct'][HEAD_REF]:+.2f}")
            lines.append(f"| {name.replace('_', ' ')} | {x['hits'][name]}/{x['n']} | {x['accuracy_pct'][name]:.2f} | {d0} | {d1} |")
        lines += ["", f"All {x['official_scorer_parity']['nonnull_points']} nonnull predicted points (five new arms plus two references) passed official/local scorer parity.", ""]
    lines += ["## Interpretation limits", "",
              "The 96-head global mean removes head and layer identity; the per-layer mean retains layer identity but removes within-layer head identity. The single-metric variants retain unchanged base metadata, so they show conditional sufficiency. This one-seed retrospective diagnostic does not establish stable rank ordering or a three-seed effect; no arm was chosen from test results.",
              "", "PointerBench labels had been analyzed earlier, and at least seven answers were visible before earlier V2 sealing. Selected MMBench L2 1000 is not its full official leaderboard.",
              "", f"Score registration SHA256: `{sha(SCORE_REG)}`; analysis registration SHA256: `{sha(ANALYSIS_REG)}`; V4 prediction seal SHA256: `{sha(OUT / 'predictions/completion.json')}`.", ""]
    return "\n".join(lines)


def analyze() -> None:
    verify_gate()
    a = read(ANALYSIS_REG)
    pointer, mm = v3.labels_after_gate(a)  # Called only after all five-arm predictions seal.
    _, bundles = v3_seal_gate()
    cohorts = {suite: evaluate_suite(suite, pointer, mm, bundles) for suite in COHORTS}
    result = dict(complete=True, method_names=list(NAMES), seed=20261311,
                  training_protocol_sha256=sha(TRAIN_REG),
                  score_registration_sha256=sha(SCORE_REG),
                  analysis_registration_sha256=sha(ANALYSIS_REG),
                  prediction_completion_sha256=sha(OUT / "predictions/completion.json"),
                  v3_prediction_completion_sha256=sha(v3.OUT / "predictions/completion.json"),
                  cohorts=cohorts, descriptive_single_seed_only=True)
    write_once(OUT / "analysis/summary.json", result)
    content = report(result)
    text_once(OUT / "analysis/readout.txt", content)
    text_once(ROOT / "paper/GSAIE/REVIEW_ATTENTION_FIVE_ARM_SINGLE_SEED_RESULTS.md", content)
    status("analyzed", summary_sha256=sha(OUT / "analysis/summary.json"),
           readout_sha256=sha(OUT / "analysis/readout.txt"), labels_opened=True)


def selftest() -> None:
    fine.low_priority(); torch.set_num_threads(2)
    require(not torch.cuda.is_available(), "Hide CUDA for V4 CPU selftest")
    common.selftest()
    _, bundles = v3_seal_gate()
    for suite in COHORTS:
        entry = read(common.FEATURES[suite])["cases"][0]
        ref = v3_case(suite, entry["sample_id"], bundles)
        require(ref["feature_sha256"] == entry["sha256"],
                "V3/V4 first cached feature mismatch")
    print(json.dumps({"stage": "five_arm_eval_selftest_pass", "variants": list(ARMS),
                      "v3_reference_sealed": True, "test_labels_opened": False}), flush=True)


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
