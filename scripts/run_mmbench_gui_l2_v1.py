"""Frozen ClickAgent 3k/full evaluation on history-clean MMBench-GUI L2.

The annotation file with ground-truth boxes is parsed only by ``analyze`` after
``verify-gate`` validates 1,000 sealed label-free predictions. All stages are
idempotent; public extraction resumes from authenticated per-case caches.
"""
from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import copy
from datetime import datetime, timezone
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import numpy as np

import run_paper_scale_eval_v1 as engine
import run_quality_readout_v15_3k as prior
_BASE_MODELS = engine.models

RUN = ROOT / "runs/mmbench_gui_new_source_v1"
COHORT = "mmbench"
ANNOTATIONS = ROOT / "data/mmbench_gui_l2_v1/source/L2_annotations.json"
AUDIT = RUN / "audit.json"
AUDIT_CORRECTION = RUN / "audit_correction_receipt.json"
PUBLIC = RUN / "public.json"
PROTOCOL = RUN / "protocol.json"
STATUS = RUN / "status.json"
PRED = RUN / COHORT / "predictions.json"
PUBLIC_COMPLETION = RUN / "public_completion.json"
FREEZE = ROOT / "research/paper/method_freeze_v1.json"
PREPARER = ROOT / "scripts/prepare_mmbench_gui_l2_v1.py"
DOWNLOADER = ROOT / "scripts/download_mmbench_gui_l2_v1.py"
SCORER = RUN / "official_l2_evaluator.py"
SCORER_SHA = "2ce501eefc9babcd9f22091f3b52d04894b0f981c96cdfb44e20a8d67aa2d09d"
SCORER_COMMIT = "b67cae38a89916304cba2368ce957d5664857cab"
MODEL_NAMES = tuple(f"{scale}_aligned_s{s}" for scale in ("3k", "full") for s in (1, 2, 3))
PUBLIC_CASE_KEYS = set(engine.CASE_KEYS) | {"source_index", "grounding_type", "data_type", "platform", "dhash256"}


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_once(path: Path, obj: object) -> None:
    content = (json.dumps(obj, sort_keys=True, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    if path.exists():
        assert path.read_bytes() == content, f"Frozen output differs: {path}"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)


def status(phase: str, **fields) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(dict(phase=phase, at_utc=datetime.now(timezone.utc).isoformat(), **fields),
                                 ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(phase=phase, **fields), ensure_ascii=False), flush=True)


def six_models() -> dict:
    table = _BASE_MODELS(False)
    selected = {name: table[name] for name in MODEL_NAMES}
    freeze = read(FREEZE)
    for s in (1, 2, 3):
        assert selected[f"3k_aligned_s{s}"]["sha256"] == freeze["scorers"][f"expanded_s{s}"]["sha256"]
    return selected


def verify_public() -> tuple[dict, list[dict]]:
    spec = read(PROTOCOL)
    audit = read(AUDIT)
    rows = read(PUBLIC)
    assert spec["complete"] and spec["labels_opened_before_predictions"] is False
    assert spec["source_annotation_sha256"] == audit["source_annotation_sha256"]
    assert spec["source_archive_sha256"] == audit["source_archive_sha256"]
    assert spec["audit_sha256"] == sha(AUDIT) and spec["public_sha256"] == sha(PUBLIC)
    assert spec["audit_correction_sha256"] == sha(AUDIT_CORRECTION)
    assert spec["preparer_sha256"] == sha(PREPARER) and spec["downloader_sha256"] == sha(DOWNLOADER)
    assert spec["official_scorer_sha256"] == sha(SCORER) == SCORER_SHA
    assert spec["runner_sha256"] == sha(Path(__file__))
    assert spec["engine_sha256"] == sha(Path(engine.__file__))
    assert spec["prior_sha256"] == sha(Path(prior.__file__))
    assert spec["method_freeze_sha256"] == sha(FREEZE)
    assert spec["model_hashes"] == {name: entry["sha256"] for name, entry in six_models().items()}
    assert len(rows) == 1000 and len({r["sample_id"] for r in rows}) == 1000
    assert Counter(r["grounding_type"] for r in rows) == {"basic": 500, "advanced": 500}
    assert all(set(r) == PUBLIC_CASE_KEYS for r in rows)
    assert all(r["dataset"] == "MMBench-GUI-L2" for r in rows)
    for row in rows:
        assert sha(Path(row["image"])) == row["image_sha256"]
    return spec, [{key: row[key] for key in engine.CASE_KEYS} for row in rows]


def register() -> None:
    audit = read(AUDIT)
    rows = read(PUBLIC)
    assert audit["complete"] and audit["selected_images"] == 500 and audit["selected_queries"] == 1000
    assert audit["selected_history_exact"] == audit["selected_history_near_le4"] == 0
    assert audit["labels_examined_by_selection"] is False and audit["model_forward_calls"] == 0
    assert audit["new_source_scope"].startswith("Project-history-unseen")
    assert sha(PUBLIC) == audit["public_sha256"]
    assert sha(SCORER) == SCORER_SHA
    correction = read(AUDIT_CORRECTION)
    assert correction["complete"] and correction["final_audit_sha256"] == sha(AUDIT)
    assert correction["public_sha256_unchanged"] == sha(PUBLIC)
    table = six_models()
    freeze = read(FREEZE)
    policy = freeze["inference_policy"]
    assert policy["max_pixels_per_view"] == 262144 and "K16" in policy["candidate_pool"]
    spec = dict(complete=True, purpose="Project-history-unseen cross-source static point-grounding test",
                source="MMBench-GUI L2 official public annotation and offline images",
                source_annotation_sha256=audit["source_annotation_sha256"],
                source_archive_sha256=audit["source_archive_sha256"],
                source_hf_revision=audit["source_hf_revision"],
                audit_sha256=sha(AUDIT), public_sha256=sha(PUBLIC),
                audit_correction_sha256=sha(AUDIT_CORRECTION),
                preparer_sha256=sha(PREPARER), downloader_sha256=sha(DOWNLOADER),
                official_scorer_sha256=SCORER_SHA, official_scorer_commit=SCORER_COMMIT,
                official_scorer_url=f"https://github.com/open-compass/MMBench-GUI/blob/{SCORER_COMMIT}/benchmarks/l2_element_grounding.py",
                sample_ids=[r["sample_id"] for r in rows], image_count=500, query_count=1000,
                basic_count=500, advanced_count=500, image_paired_not_target_paired=True,
                model_hashes={name: entry["sha256"] for name, entry in table.items()},
                method_freeze_sha256=sha(FREEZE), runner_sha256=sha(Path(__file__)),
                engine_sha256=sha(Path(engine.__file__)), prior_sha256=sha(Path(prior.__file__)),
                backbone_identity_sha256=freeze["qwen_identity_sha256"],
                proposer=policy["proposer"], prompt_and_candidate_rule=policy["decision"],
                candidate_pool=policy["candidate_pool"], views=policy["views"],
                max_pixels_per_view=262144, fixed_seeds=freeze["training_policy"]["fixed_seeds"],
                metric="Official MMBench-GUI L2 inclusive point-in-normalized-bbox; invalid/missing point incorrect",
                reports=("overall", "basic", "advanced", "platform", "icon_text", "full_minus_3k_cluster_interval"),
                cluster_unit="source RGB image (two queries)", bootstrap_replicates=10000,
                bootstrap_seed=20261112, labels_opened_before_predictions=False,
                no_model_or_sample_selection_from_test_results=True,
                no_archived_feature_reuse=True, foundation_model_training_overlap_unknown=True,
                new_source_scope=audit["new_source_scope"],
                history_lineage_note=audit["history_lineage_note"],
                strict_blind_claim=False)
    write_once(PROTOCOL, spec)
    verify_public()
    status("registered", query_count=1000, protocol_sha256=sha(PROTOCOL))


def bind_engine() -> None:
    engine.OUT = RUN
    engine.archived_sources = lambda: ({}, {})
    engine.models = lambda reference_only=False: {name: _BASE_MODELS(False)[name] for name in MODEL_NAMES}
    engine.verify_public = lambda cohort: (verify_public() if cohort == COHORT else (_ for _ in ()).throw(ValueError(cohort)))


def extract() -> None:
    verify_public()
    bind_engine()
    status("extracting", events=str(RUN / "events.jsonl"))
    engine.extract_public(COHORT)
    index = read(RUN / COHORT / "features/index.json")
    assert index["complete"] and index["n"] == 1000 and not index["labels_present"]
    assert index["reused_v16"] == index["reused_v18"] == 0
    assert all(entry.get("public_feature_reuse") is None for entry in index["cases"])
    status("extracted", features_index_sha256=sha(RUN / COHORT / "features/index.json"))


def score() -> None:
    verify_public()
    bind_engine()
    status("scoring")
    engine.score_public(COHORT, "cpu")
    assert PRED.exists() and read(PRED)["n"] == 1000
    status("predictions_sealed", predictions_sha256=sha(PRED))


def official_hit(point: list[int] | None, bbox: list[float], size: list[int]) -> bool:
    if point is None or len(point) != 2 or not all(isinstance(v, (int, float)) for v in point):
        return False
    if not (len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox)):
        return False
    # The official scorer tests normalization on the *original* bbox, then
    # maps -1 to 0 in either the normalized or pixel-coordinate branch.
    is_normalized = all(0 <= v <= 1 for v in bbox)
    safe = [0 if value == -1 else value for value in bbox]
    box = safe if is_normalized else [safe[0] / size[0], safe[1] / size[1],
                                      safe[2] / size[0], safe[3] / size[1]]
    if any(v < 0 or v > 1 for v in box):
        raise ValueError("Official bbox normalization produced out-of-range coordinates")
    x, y = point
    # Match the official evaluator: normalize both point components if either >1.
    if x > 1 or y > 1:
        x, y = x / size[0], y / size[1]
    return bool(box[0] <= x <= box[2] and box[1] <= y <= box[3])


@lru_cache(maxsize=1)
def source_reference_hit():
    """Compile the exact normalization/match AST from the pinned official file.

    Its evaluator class needs the full VLMEvalKit stack, so extract only the
    mathematical nodes for label-free synthetic parity checks.
    """
    assert sha(SCORER) == SCORER_SHA
    module = ast.parse(SCORER.read_text(encoding="utf-8"))
    evaluators = [node for node in ast.walk(module) if isinstance(node, ast.FunctionDef) and node.name == "evaluate"]
    assert len(evaluators) == 1
    nodes = list(ast.walk(evaluators[0]))

    def take(kind, line):
        found = [node for node in nodes if isinstance(node, kind) and node.lineno == line]
        assert len(found) == 1, f"Official scorer AST changed at line {line}"
        return copy.deepcopy(found[0])

    body = [take(ast.FunctionDef, 181), take(ast.FunctionDef, 188),
            take(ast.If, 194), take(ast.If, 210), take(ast.If, 224),
            take(ast.Assign, 230), ast.Return(value=ast.Name(id="match", ctx=ast.Load()))]
    args = ast.arguments(posonlyargs=[], args=[ast.arg(arg=name) for name in ("bbox", "img_size", "click_point")],
                         vararg=None, kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    fn = ast.FunctionDef(name="_reference_hit", args=args, body=body, decorator_list=[], returns=None, type_comment=None)
    mini = ast.fix_missing_locations(ast.Module(body=[fn], type_ignores=[]))
    scope = {}
    exec(compile(mini, str(SCORER), "exec"), scope)
    return scope["_reference_hit"]


def scorer_self_check() -> None:
    ref = source_reference_hit()
    cases = [([25, 25], [0.25, 0.25, 0.5, 0.5], [100, 100]),
             ([50, 50], [0.25, 0.25, 0.5, 0.5], [100, 100]),
             ([51, 50], [0.25, 0.25, 0.5, 0.5], [100, 100]),
             ([1, 1], [0.9, 0.9, 1, 1], [100, 100]),
             ([25, 25], [25, 25, 50, 50], [100, 100]),
             ([0, 25], [-1, 0.2, 0.4, 0.5], [100, 100]),
             ([1, 25], [-1, 20, 40, 50], [100, 100]),
             ([0, 0], [-1, -1, 1, 1], [100, 100])]
    for point, bbox, size in cases:
        assert official_hit(point, bbox, size) == bool(ref(bbox.copy(), tuple(size), tuple(point)))
    assert official_hit(None, [0.25, 0.25, 0.5, 0.5], [100, 100]) is False


def verify_gate() -> dict:
    spec, cases = verify_public()
    scorer_self_check()
    feature_index_path = RUN / COHORT / "features/index.json"
    feature_index = read(feature_index_path)
    assert feature_index["complete"] and feature_index["n"] == 1000
    assert feature_index["public_sha256"] == spec["public_sha256"]
    assert feature_index["labels_present"] is False and feature_index["ground_truth_in_model_input"] is False
    # The engine's `fresh` counter is per invocation, so a resumed run may
    # have fewer than 1,000 fresh calls in its final invocation. The sealed
    # entries and zero reuse counters prove the requested policy instead.
    assert feature_index["reused_v16"] == feature_index["reused_v18"] == 0
    assert all(entry.get("public_feature_reuse") is None for entry in feature_index["cases"])
    pred = read(PRED)
    assert pred["complete"] and pred["n"] == 1000 and pred["public_sha256"] == spec["public_sha256"]
    assert pred["model_hashes"] == spec["model_hashes"] and pred["labels_opened"] is False
    assert set(pred["case_hashes"]) == {r["sample_id"] for r in cases}
    assert len(feature_index["cases"]) == 1000
    for row, entry in zip(cases, feature_index["cases"]):
        engine.authenticate_feature(entry, row)
        prediction_path = RUN / COHORT / f"predictions/cases/{engine.key(row['sample_id'])}.json"
        assert sha(prediction_path) == pred["case_hashes"][row["sample_id"]]
        result = read(prediction_path)
        assert result["complete"] and result["labels_opened"] is False
        assert result["sample_id"] == row["sample_id"] and result["model_hashes"] == spec["model_hashes"]
        assert set(result["methods"]) == set(MODEL_NAMES)
    receipt = dict(complete=True, protocol_sha256=sha(PROTOCOL), audit_sha256=sha(AUDIT),
                   public_sha256=sha(PUBLIC), feature_index_sha256=sha(feature_index_path),
                   predictions_sha256=sha(PRED), case_count=1000,
                   case_hashes_sha256=hashlib.sha256(json.dumps(pred["case_hashes"], sort_keys=True).encode()).hexdigest(),
                   labels_opened=False, all_public_predictions_sealed=True,
                   official_scorer_synthetic_parity_passed=True)
    write_once(PUBLIC_COMPLETION, receipt)
    status("public_gate_verified", completion_sha256=sha(PUBLIC_COMPLETION))
    return receipt


def bootstrap_diff(rows: list[dict], hits: dict[str, np.ndarray]) -> dict:
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["pixel_sha256"]].append(i)
    assert len(groups) == 500 and all(len(ix) == 2 for ix in groups.values())
    clusters = list(groups.values())
    full = np.asarray([hits[f"full_aligned_s{s}"] for s in (1, 2, 3)], np.float32).mean(axis=0)
    small = np.asarray([hits[f"3k_aligned_s{s}"] for s in (1, 2, 3)], np.float32).mean(axis=0)
    diff = full - small
    numerators = np.asarray([diff[ix].sum() for ix in clusters], np.float32)
    rng = np.random.default_rng(20261112)
    draws = rng.multinomial(len(clusters), np.full(len(clusters), 1 / len(clusters)), size=10000).astype(np.float32)
    sampled = (draws @ numerators) / (draws.sum(axis=1) * 2)
    return dict(delta=float(diff.mean()), ci95=np.quantile(sampled, [0.025, 0.975]).tolist(),
                bootstrap_unit="RGB image", bootstrap_replicates=10000,
                conditional_on_fixed_three_seeds=True)


def analyze() -> None:
    gate = verify_gate()
    assert gate["all_public_predictions_sealed"]
    assert sha(ANNOTATIONS) == read(PROTOCOL)["source_annotation_sha256"]
    source = read(ANNOTATIONS)  # First opening target bbox after all predictions are sealed.
    by_index = {row["index"]: row for row in source}
    public = read(PUBLIC)
    assert len(source) == 3594 and len(by_index) == len(source)
    reference_hit = source_reference_hit()
    parity_checked_nonnull = 0
    missing_points = 0
    hits = {name: np.zeros(1000, dtype=np.int8) for name in MODEL_NAMES}
    details = []
    for i, row in enumerate(public):
        official = by_index[row["source_index"]]
        assert all(official[field] == row[field] for field in ("instruction", "image_size", "data_type", "platform", "grounding_type"))
        assert official["app_name"] == row["application"]
        bbox = official["bbox"]
        assert len(bbox) == 4 and all(isinstance(x, (int, float)) for x in bbox)
        case_path = RUN / COHORT / f"predictions/cases/{engine.key(row['sample_id'])}.json"
        prediction = read(case_path)
        outcomes = {}
        for name in MODEL_NAMES:
            point = prediction["methods"][name]["point"]
            outcome = official_hit(point, bbox, row["image_size"])
            if point is None:
                missing_points += 1
            else:
                reference = bool(reference_hit(bbox.copy(), tuple(row["image_size"]), tuple(point)))
                assert outcome == reference, f"Official scorer parity failed: {row['sample_id']} {name}"
                parity_checked_nonnull += 1
            hits[name][i] = int(outcome)
            outcomes[name] = int(outcome)
        details.append(dict(sample_id=row["sample_id"], source_index=row["source_index"],
                            pixel_sha256=row["pixel_sha256"], platform=row["platform"],
                            grounding_type=row["grounding_type"], data_type=row["data_type"],
                            hits=outcomes))
    summary = {}
    for scale in ("3k", "full"):
        arm = {}
        for subset, ix in (("all", list(range(1000))),
                           ("basic", [i for i, row in enumerate(public) if row["grounding_type"] == "basic"]),
                           ("advanced", [i for i, row in enumerate(public) if row["grounding_type"] == "advanced"])):
            arm[subset] = dict(n=len(ix), seed_hits=[int(hits[f"{scale}_aligned_s{s}"][ix].sum()) for s in (1, 2, 3)],
                               mean_accuracy=float(np.mean([hits[f"{scale}_aligned_s{s}"][ix].mean() for s in (1, 2, 3)])))
        arm["platform"] = {}
        for platform in sorted({r["platform"] for r in public}):
            ix = [i for i, row in enumerate(public) if row["platform"] == platform]
            arm["platform"][platform] = dict(n=len(ix), mean_accuracy=float(np.mean([
                hits[f"{scale}_aligned_s{s}"][ix].mean() for s in (1, 2, 3)])))
        arm["data_type"] = {}
        for kind in ("icon", "text"):
            ix = [i for i, row in enumerate(public) if row["data_type"] == kind]
            arm["data_type"][kind] = dict(n=len(ix), mean_accuracy=float(np.mean([
                hits[f"{scale}_aligned_s{s}"][ix].mean() for s in (1, 2, 3)])))
        summary[scale] = arm
    summary["full_minus_3k"] = bootstrap_diff(public, hits)
    analysis = RUN / "analysis"
    analysis.mkdir(parents=True, exist_ok=True)
    write_once(analysis / "details.json", details)
    write_once(analysis / "summary.json", dict(complete=True, source="MMBench-GUI-L2",
                                               protocol_sha256=sha(PROTOCOL), gate_sha256=sha(PUBLIC_COMPLETION),
                                               label_sha256=sha(ANNOTATIONS), n=1000, images=500, results=summary,
                                               official_scorer_real_annotation_parity_nonnull=parity_checked_nonnull,
                                               missing_points=missing_points,
                                               official_scorer_real_annotation_parity_passed=True,
                                               label_access_after_prediction_seal=True))
    text = ["MMBench-GUI L2 history-clean fixed cohort: 500 images, 1,000 queries (500 basic, 500 advanced).",
            "Project-history exact/near overlaps excluded before prediction; public source can still be in backbone pretraining.",
            "Each image contributes one basic and one advanced instruction, not guaranteed to target the same element.",
            "Frozen original-K16 3k/full aligned readouts, three fixed seeds; official inclusive point-in-bbox metric."]
    for scale in ("3k", "full"):
        for subset in ("all", "basic", "advanced"):
            value = summary[scale][subset]
            text.append(f"{scale} {subset}: {value['mean_accuracy']*100:.2f}% (seed hits {value['seed_hits']}, n={value['n']} each)")
    delta = summary["full_minus_3k"]
    text.append(f"full minus 3k: {delta['delta']*100:+.2f} pp, descriptive 95% image-cluster CI [{delta['ci95'][0]*100:+.2f}, {delta['ci95'][1]*100:+.2f}] pp.")
    text.append("Platform and icon/text strata: analysis/summary.json. This selected subset is not the official full MMBench-GUI leaderboard.")
    (analysis / "readout.txt").write_text("\n".join(text) + "\n", encoding="utf-8")
    status("complete", summary_sha256=sha(analysis / "summary.json"), readout=str(analysis / "readout.txt"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True,
                        choices=("register", "extract", "score", "verify-gate", "analyze", "all"))
    args = parser.parse_args()
    if args.stage in ("register", "all"):
        register()
    if args.stage in ("extract", "all"):
        extract()
    if args.stage in ("score", "all"):
        score()
    if args.stage in ("verify-gate", "all"):
        verify_gate()
    if args.stage in ("analyze", "all"):
        analyze()


if __name__ == "__main__":
    main()
