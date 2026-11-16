"""Retrospective full-readout ablation on the fixed MMBench-GUI L2 cohort.

This supplements, but never rewrites, mmbench_gui_new_source_v1. The five new
readout arms are predicted from the original label-free feature cache. All
their predictions are sealed before this script opens target boxes in analyze.
The parent cohort's labels have already been analyzed, so this is not a new
blind evaluation or an independent test of model selection.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import numpy as np

import run_mmbench_gui_l2_v1 as parent
import run_paper_scale_eval_v1 as engine

RUN = ROOT / "runs/mmbench_full_ablation_supplement_v1"
PROTOCOL = RUN / "protocol.json"
STATUS = RUN / "status.json"
PRED = RUN / "predictions.json"
GATE = RUN / "public_completion.json"
PARENT_RUN = ROOT / "runs/mmbench_gui_new_source_v1"
PARENT_FEATURES = PARENT_RUN / "mmbench/features/index.json"
PARENT_PRED = PARENT_RUN / "mmbench/predictions.json"
PARENT_GATE = PARENT_RUN / "public_completion.json"
PARENT_PROTOCOL = PARENT_RUN / "protocol.json"
PARENT_PUBLIC = PARENT_RUN / "public.json"
MODEL_REGISTRY = ROOT / "runs/paper_scale_ablation_v1/models/frozen.json"
FREEZE = ROOT / "research/paper/method_freeze_v1.json"
ANNOTATIONS = ROOT / "data/mmbench_gui_l2_v1/source/L2_annotations.json"
ARMS = ("no_evidence", "spatial_shuffle", "attention_only", "hidden_only", "token_mean")
ALL_ARMS = ("aligned",) + ARMS
SEED_IDS = (1, 2, 3)
MODEL_NAMES = tuple(f"full_{a}_s{s}" for a in ARMS for s in SEED_IDS)
BOOTSTRAP_SEED = 20261112
BOOTSTRAP_REPLICATES = 10000


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_once(path: Path, obj) -> None:
    data = (json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")
    if path.exists():
        assert path.read_bytes() == data, f"Frozen output differs: {path}"
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)


def set_status(phase: str, **kwargs) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    value = dict(phase=phase, at_utc=datetime.now(timezone.utc).isoformat(), **kwargs)
    STATUS.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(value, ensure_ascii=False), flush=True)


def fixed_sources() -> tuple[dict, list[dict], list[dict], dict]:
    parent_spec, cases = parent.verify_public()
    old_gate = read(PARENT_GATE)
    assert old_gate["complete"] and old_gate["all_public_predictions_sealed"]
    assert old_gate["predictions_sha256"] == sha(PARENT_PRED)
    assert old_gate["feature_index_sha256"] == sha(PARENT_FEATURES)
    assert old_gate["protocol_sha256"] == sha(PARENT_PROTOCOL)
    assert old_gate["public_sha256"] == sha(PARENT_PUBLIC)
    index = read(PARENT_FEATURES)
    assert index["complete"] and index["n"] == 1000 and not index["labels_present"]
    assert index["ground_truth_in_model_input"] is False
    assert index["public_sha256"] == parent_spec["public_sha256"]
    assert len(cases) == len(index["cases"]) == 1000
    assert [r["sample_id"] for r in cases] == [r["sample_id"] for r in index["cases"]]
    public = read(PARENT_PUBLIC)
    assert Counter(r["grounding_type"] for r in public) == {"basic": 500, "advanced": 500}
    assert len({r["pixel_sha256"] for r in cases}) == 500
    assert parent_spec["candidate_pool"].startswith("original V15 actual K16")
    assert parent_spec["max_pixels_per_view"] == 262144
    return parent_spec, cases, index["cases"], old_gate


def fixed_models() -> dict:
    # The engine authenticates every checkpoint against the original frozen
    # registry. Only the predeclared five arms are returned for this scoring.
    table = engine.models(False)
    registry = read(MODEL_REGISTRY)
    assert registry["complete"]
    assert set(MODEL_NAMES) <= set(registry["models"])
    assert all(table[n]["sha256"] == registry["models"][n]["sha256"] for n in MODEL_NAMES)
    return {n: table[n] for n in MODEL_NAMES}


def register() -> None:
    parent_spec, cases, _, old_gate = fixed_sources()
    models = fixed_models()
    assert sha(ANNOTATIONS) == parent_spec["source_annotation_sha256"]  # Hash only: no labels opened here.
    spec = dict(
        complete=True,
        purpose="Retrospective full six-arm MMBench-GUI L2 supplement",
        retrospective=True,
        strict_blind_claim=False,
        reason="The parent aligned-only MMBench evaluation had already opened and analyzed labels before this supplement was registered.",
        parent_protocol_sha256=sha(PARENT_PROTOCOL),
        parent_public_sha256=sha(PARENT_PUBLIC),
        parent_feature_index_sha256=sha(PARENT_FEATURES),
        parent_predictions_sha256=sha(PARENT_PRED),
        parent_completion_sha256=sha(PARENT_GATE),
        parent_completion_old_labels_opened_at_seal=old_gate["labels_opened"],
        source_annotation_sha256=parent_spec["source_annotation_sha256"],
        official_scorer_sha256=parent_spec["official_scorer_sha256"],
        sample_ids=[r["sample_id"] for r in cases],
        image_count=500, query_count=1000, basic_count=500, advanced_count=500,
        model_registry_sha256=sha(MODEL_REGISTRY),
        model_hashes={n: models[n]["sha256"] for n in MODEL_NAMES},
        old_full_aligned_model_hashes={f"full_aligned_s{s}": parent_spec["model_hashes"][f"full_aligned_s{s}"] for s in SEED_IDS},
        method_freeze_sha256=sha(FREEZE),
        runner_sha256=sha(Path(__file__)),
        parent_runner_sha256=sha(Path(parent.__file__)),
        engine_sha256=sha(Path(engine.__file__)),
        candidate_pool="Original V15 actual K16; canonical union-index-sorted K16, up to 16 distinct integer points",
        choice="Forced stable score argmax over K16; empty pool is incorrect",
        max_pixels_per_view=262144,
        fixed_training_seeds=[20261311, 20261312, 20261313],
        metric="Pinned official MMBench-GUI L2 inclusive point-in-normalized-bbox; invalid/missing point incorrect",
        reports=["all", "basic", "advanced"],
        fixed_comparisons=[f"full_aligned_vs_{arm}" for arm in ARMS],
        bootstrap_unit="source RGB image, two queries per image",
        bootstrap_seed=BOOTSTRAP_SEED,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        intervals=["descriptive 95% percentile", "familywise Bonferroni 99% percentile for five comparisons"],
        selection="All five arms and all three seeds are reported; no result-dependent model choice",
        labels_opened_by_this_supplement_before_new_prediction_seal=False,
        only_existing_label_free_features=True,
        no_visual_backbone_or_proposer_rerun=True,
    )
    write_once(PROTOCOL, spec)
    set_status("registered", protocol_sha256=sha(PROTOCOL))


def verify_registration() -> tuple[dict, list[dict], list[dict]]:
    spec = read(PROTOCOL)
    parent_spec, cases, entries, _ = fixed_sources()
    assert spec["complete"] and spec["retrospective"] and not spec["strict_blind_claim"]
    for name, path in (("parent_protocol_sha256", PARENT_PROTOCOL),
                       ("parent_public_sha256", PARENT_PUBLIC),
                       ("parent_feature_index_sha256", PARENT_FEATURES),
                       ("parent_predictions_sha256", PARENT_PRED),
                       ("parent_completion_sha256", PARENT_GATE),
                       ("model_registry_sha256", MODEL_REGISTRY),
                       ("method_freeze_sha256", FREEZE),
                       ("runner_sha256", Path(__file__)),
                       ("parent_runner_sha256", Path(parent.__file__)),
                       ("engine_sha256", Path(engine.__file__))):
        assert spec[name] == sha(path), f"Frozen source changed: {path}"
    assert spec["sample_ids"] == [r["sample_id"] for r in cases]
    assert spec["source_annotation_sha256"] == parent_spec["source_annotation_sha256"]
    assert spec["model_hashes"] == {n: e["sha256"] for n, e in fixed_models().items()}
    assert spec["old_full_aligned_model_hashes"] == {
        f"full_aligned_s{s}": parent_spec["model_hashes"][f"full_aligned_s{s}"] for s in SEED_IDS}
    return spec, cases, entries


def score() -> None:
    spec, cases, entries = verify_registration()
    assert not GATE.exists() or PRED.exists()
    model_table = engine.load_model_table(fixed_models(), "cpu")
    started = time.perf_counter()
    hashes = {}
    set_status("scoring", completed=0, total=1000)
    for i, (row, entry) in enumerate(zip(cases, entries), 1):
        path = RUN / "predictions/cases" / f"{engine.key(row['sample_id'])}.json"
        if path.exists():
            item = read(path)
            assert item["complete"] and item["sample_id"] == row["sample_id"]
            assert item["feature_sha256"] == entry["sha256"]
            assert item["model_hashes"] == spec["model_hashes"]
            assert item["labels_opened"] is False
        else:
            record = engine.read_feature(entry, row)
            choices = engine.score_record(record, model_table, "cpu")
            item = dict(complete=True, sample_id=row["sample_id"],
                        feature_sha256=entry["sha256"],
                        model_hashes=spec["model_hashes"],
                        candidate_count=record["n"], K16_count=len(record["baseline"]),
                        methods=choices, labels_opened=False)
            write_once(path, item)
        hashes[row["sample_id"]] = sha(path)
        if i % 50 == 0 or i == len(cases):
            set_status("scoring", completed=i, total=len(cases),
                       elapsed_seconds=round(time.perf_counter() - started, 1))
    write_once(PRED, dict(complete=True, n=1000, cohort="mmbench", public_sha256=sha(PARENT_PUBLIC),
                          protocol_sha256=sha(PROTOCOL), model_hashes=spec["model_hashes"],
                          case_hashes=hashes, labels_opened=False))
    set_status("predictions_sealed", predictions_sha256=sha(PRED))


def verify_gate() -> dict:
    spec, cases, entries = verify_registration()
    parent.scorer_self_check()  # Synthetic parity only; no benchmark labels.
    pred = read(PRED)
    assert pred["complete"] and pred["n"] == 1000 and not pred["labels_opened"]
    assert pred["protocol_sha256"] == sha(PROTOCOL)
    assert pred["public_sha256"] == sha(PARENT_PUBLIC)
    assert pred["model_hashes"] == spec["model_hashes"]
    assert set(pred["case_hashes"]) == {r["sample_id"] for r in cases}
    old = read(PARENT_PRED)
    assert old["complete"] and old["n"] == 1000 and old["model_hashes"] == read(PARENT_PROTOCOL)["model_hashes"]
    for row, entry in zip(cases, entries):
        engine.authenticate_feature(entry, row)
        sid = row["sample_id"]
        path = RUN / "predictions/cases" / f"{engine.key(sid)}.json"
        assert sha(path) == pred["case_hashes"][sid]
        item = read(path)
        assert item["complete"] and item["sample_id"] == sid and not item["labels_opened"]
        assert item["feature_sha256"] == entry["sha256"]
        assert item["model_hashes"] == spec["model_hashes"]
        assert set(item["methods"]) == set(MODEL_NAMES)
        old_path = PARENT_RUN / "mmbench/predictions/cases" / f"{engine.key(sid)}.json"
        assert sha(old_path) == old["case_hashes"][sid]
        old_item = read(old_path)
        assert old_item["sample_id"] == sid and not old_item["labels_opened"]
        assert all(f"full_aligned_s{s}" in old_item["methods"] for s in SEED_IDS)
        assert item["candidate_count"] == old_item["candidate_count"]
        assert item["K16_count"] == old_item["K16_count"]
    receipt = dict(complete=True, protocol_sha256=sha(PROTOCOL),
                   parent_completion_sha256=sha(PARENT_GATE),
                   feature_index_sha256=sha(PARENT_FEATURES),
                   predictions_sha256=sha(PRED), case_count=1000,
                   labels_opened_by_supplement_before_seal=False,
                   retrospective_parent_labels_already_seen=True,
                   all_new_predictions_sealed=True,
                   official_scorer_synthetic_parity_passed=True)
    write_once(GATE, receipt)
    set_status("gate_verified", completion_sha256=sha(GATE))
    return receipt


def paired_interval(rows: list[dict], hits: dict[str, np.ndarray], arm: str, draws: np.ndarray) -> dict:
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row["pixel_sha256"]].append(i)
    clusters = list(groups.values())
    assert len(clusters) == 500 and all(len(indices) == 2 for indices in clusters)
    a = np.stack([hits[f"full_aligned_s{s}"] for s in SEED_IDS]).astype(np.float32)
    b = np.stack([hits[f"full_{arm}_s{s}"] for s in SEED_IDS]).astype(np.float32)
    difference = (a - b).mean(axis=0)
    numerators = np.asarray([difference[ix].sum() for ix in clusters], np.float32)
    sampled = (draws @ numerators) / (draws.sum(axis=1) * 2)
    return dict(delta=float(difference.mean()),
                per_seed_delta=(a - b).mean(axis=1).tolist(),
                ci95=np.quantile(sampled, [0.025, 0.975]).tolist(),
                ci99_familywise_bonferroni=np.quantile(sampled, [0.005, 0.995]).tolist(),
                bootstrap_unit="source RGB image", bootstrap_replicates=BOOTSTRAP_REPLICATES,
                conditional_on_fixed_three_seeds=True)


def analyze() -> None:
    receipt = verify_gate()
    assert receipt["all_new_predictions_sealed"]
    spec = read(PROTOCOL)
    assert sha(ANNOTATIONS) == spec["source_annotation_sha256"]
    source = read(ANNOTATIONS)  # Target boxes first opened by this script after the gate.
    by_index = {r["index"]: r for r in source}
    assert len(source) == 3594 and len(by_index) == 3594
    public = read(PARENT_PUBLIC)
    reference = parent.source_reference_hit()
    hits = {f"full_{a}_s{s}": np.zeros(1000, dtype=np.int8) for a in ALL_ARMS for s in SEED_IDS}
    details = []
    parity_count = 0
    missing_points = 0
    for i, row in enumerate(public):
        label = by_index[row["source_index"]]
        assert all(label[k] == row[k] for k in ("instruction", "image_size", "data_type", "platform", "grounding_type"))
        assert label["app_name"] == row["application"]
        new_item = read(RUN / "predictions/cases" / f"{engine.key(row['sample_id'])}.json")
        old_item = read(PARENT_RUN / "mmbench/predictions/cases" / f"{engine.key(row['sample_id'])}.json")
        methods = {**new_item["methods"], **{f"full_aligned_s{s}": old_item["methods"][f"full_aligned_s{s}"] for s in SEED_IDS}}
        outcomes = {}
        for name, item in methods.items():
            point = item["point"]
            hit = parent.official_hit(point, label["bbox"], row["image_size"])
            if point is None:
                missing_points += 1
            else:
                assert hit == bool(reference(label["bbox"].copy(), tuple(row["image_size"]), tuple(point)))
                parity_count += 1
            hits[name][i] = int(hit)
            outcomes[name] = int(hit)
        details.append(dict(sample_id=row["sample_id"], source_index=row["source_index"],
                            pixel_sha256=row["pixel_sha256"], grounding_type=row["grounding_type"],
                            data_type=row["data_type"], platform=row["platform"], hits=outcomes))
    old_summary = read(PARENT_RUN / "analysis/summary.json")
    result = {}
    for arm in ALL_ARMS:
        sub = {}
        for subset, ix in (("all", list(range(1000))),
                           ("basic", [i for i, r in enumerate(public) if r["grounding_type"] == "basic"]),
                           ("advanced", [i for i, r in enumerate(public) if r["grounding_type"] == "advanced"])):
            seed_hits = [int(hits[f"full_{arm}_s{s}"][ix].sum()) for s in SEED_IDS]
            sub[subset] = dict(n=len(ix), seed_hits=seed_hits,
                               mean_hits=float(np.mean(seed_hits)),
                               mean_accuracy=float(np.mean(seed_hits) / len(ix)))
        result[arm] = sub
    assert result["aligned"]["all"]["seed_hits"] == old_summary["results"]["full"]["all"]["seed_hits"]
    assert result["aligned"]["basic"]["seed_hits"] == old_summary["results"]["full"]["basic"]["seed_hits"]
    assert result["aligned"]["advanced"]["seed_hits"] == old_summary["results"]["full"]["advanced"]["seed_hits"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.multinomial(500, np.full(500, 1 / 500), size=BOOTSTRAP_REPLICATES).astype(np.float32)
    comparisons = {f"aligned_vs_{arm}": paired_interval(public, hits, arm, draws) for arm in ARMS}
    analysis_dir = RUN / "analysis"
    write_once(analysis_dir / "details.json", details)
    write_once(analysis_dir / "summary.json", dict(
        complete=True, retrospective=True, strict_blind_claim=False,
        protocol_sha256=sha(PROTOCOL), completion_sha256=sha(GATE),
        old_aligned_predictions_sha256=sha(PARENT_PRED),
        label_sha256=sha(ANNOTATIONS), n=1000, images=500,
        arms=result, comparisons=comparisons,
        official_scorer_real_annotation_parity_nonnull=parity_count,
        missing_points=missing_points,
        official_scorer_real_annotation_parity_passed=True,
        new_label_access_after_new_prediction_seal=True))
    lines = [
        "MMBench-GUI L2 full-readout six-arm supplement; fixed 500 RGB images / 1,000 queries.",
        "Retrospective supplement: parent aligned-only results and labels were already seen before registration.",
        "All five new arms x three seeds used existing label-free features; new predictions sealed before this script read labels.",
        "Original K16, forced stable Top1; empty candidate pool counted incorrect; pinned official inclusive point-in-bbox scorer.",
    ]
    for arm in ALL_ARMS:
        for subset in ("all", "basic", "advanced"):
            value = result[arm][subset]
            lines.append(f"full {arm} {subset}: {value['mean_accuracy']*100:.2f}% "
                         f"(seed hits {value['seed_hits']}, n={value['n']} each)")
    for arm in ARMS:
        value = comparisons[f"aligned_vs_{arm}"]
        lines.append(f"full aligned - {arm}: {value['delta']*100:+.2f} pp, "
                     f"descriptive 95% RGB-image-cluster CI "
                     f"[{value['ci95'][0]*100:+.2f}, {value['ci95'][1]*100:+.2f}] pp; "
                     f"five-way Bonferroni 99% CI "
                     f"[{value['ci99_familywise_bonferroni'][0]*100:+.2f}, "
                     f"{value['ci99_familywise_bonferroni'][1]*100:+.2f}] pp")
    lines.append("Intervals condition on the three fixed training seeds; this project-history-clean cohort is not an official full-L2 leaderboard score.")
    (analysis_dir / "readout.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    set_status("complete", summary_sha256=sha(analysis_dir / "summary.json"),
               readout=str(analysis_dir / "readout.txt"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=("register", "score", "verify-gate", "analyze"))
    args = parser.parse_args()
    {"register": register, "score": score, "verify-gate": verify_gate, "analyze": analyze}[args.stage]()


if __name__ == "__main__":
    main()
