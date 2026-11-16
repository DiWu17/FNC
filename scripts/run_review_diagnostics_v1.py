"""CPU-only, retrospective review diagnostics over sealed ClickAgent caches.

Stages: register -> predict -> verify-gate -> analyze.  The first three stages
never open target labels.  This script does not alter the frozen K16 pipeline;
K=4/8/16/32 uses the archived C3 pools and the three fixed full-aligned fits.
"""
from __future__ import annotations

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import numpy as np
import torch
import run_paper_scale_eval_v1 as engine
from clickagent.public_evaluation import point_in_geometry

OUT = ROOT / "runs/review_controls_v1/diagnostics"
V2 = ROOT / "runs/paper_scale_ablation_v2"
MM = ROOT / "runs/mmbench_gui_new_source_v1"
MVP = ROOT / "runs/pointer_mvp_same_cohort_v1"
REGISTRY = ROOT / "runs/paper_scale_ablation_v1/models/frozen.json"
COHORTS = ("pro", "sheets", "mmbench", "full")
COUNTS = {"pro": 500, "sheets": 500, "mmbench": 1000, "full": 1581}
KS = (4, 8, 16, 32)
SEEDS = (1, 2, 3)
PREVIEW_IDS = {"pro": ("pbp_0000", "pbp_0001", "pbp_0005"),
               "sheets": ("pbs_0000", "pbs_0001", "pbs_0002", "pbs_0003")}
BOOT_SEED = 20261114
BOOT_REPLICATES = 2000


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.strip()]


def encode(obj):
    return (json.dumps(obj, indent=2, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def write_once(path, obj):
    path = Path(path)
    data = encode(obj)
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError("Frozen output differs: " + str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp")
    if temp.exists():
        temp.unlink()
    with temp.open("xb") as stream:
        stream.write(data)
    os.replace(temp, path)


def status(phase, **kwargs):
    p = OUT / "status.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(encode(dict(phase=phase, at_utc=now(), **kwargs)))
    print(json.dumps(dict(phase=phase, **kwargs), ensure_ascii=False), flush=True)


def low_priority():
    # Windows BELOW_NORMAL_PRIORITY_CLASS.  No CUDA context can be created.
    if os.name == "nt":
        import ctypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        kernel.SetPriorityClass.restype = ctypes.c_int
        if not kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000):
            raise OSError("Could not set below-normal process priority")
    torch.set_num_threads(2)


def cohort_paths(cohort):
    folder = MM / "mmbench" if cohort == "mmbench" else V2 / "eval" / cohort
    public = MM / "public.json" if cohort == "mmbench" else folder / "public.json"
    return folder, public


def model_table():
    registry = read(REGISTRY)
    if registry.get("complete") is not True:
        raise ValueError("Frozen registry incomplete")
    result = {}
    for s in SEEDS:
        name = f"full_aligned_s{s}"
        entry = registry["models"][name]
        if entry["variant"] != "aligned" or sha(entry["path"]) != entry["sha256"]:
            raise ValueError("Frozen full model changed: " + name)
        result[name] = entry
    return result


def source_hashes():
    paths = [REGISTRY, V2 / "protocol.json", V2 / "pointerbench/evaluation_protocol_v2.json",
             V2 / "pointerbench/preseal_label_access_addendum_v2.json",
             MM / "protocol.json", MM / "public_completion.json",
             V2 / "public_completion.json", V2 / "pointerbench/public_completion.json",
             MVP / "public_completion.json", MVP / "predictions/mvp.json",
             ROOT / "scripts/run_paper_scale_eval_v1.py",
             ROOT / "scripts/run_evidence_retention_v14.py",
             ROOT / "src/clickagent/evidence_retention.py",
             ROOT / "src/clickagent/contribution_ablation.py"]
    for cohort in COHORTS:
        folder, public = cohort_paths(cohort)
        paths += [folder / "features/index.json", public, folder / "predictions.json"]
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path) for path in paths}


def manifest():
    cohorts = {}
    for cohort in COHORTS:
        folder, public = cohort_paths(cohort)
        rows, index = read(public), read(folder / "features/index.json")
        if len(rows) != COUNTS[cohort] or index["n"] != COUNTS[cohort] or \
           index["complete"] is not True or index["labels_present"] is not False:
            raise ValueError("Public feature cohort changed: " + cohort)
        entries = index["cases"]
        if [row["sample_id"] for row in rows] != [e["sample_id"] for e in entries]:
            raise ValueError("Public feature order differs: " + cohort)
        cohorts[cohort] = [{"sample_id": row["sample_id"], "pixel_sha256": row["pixel_sha256"],
                            "image_size": row["image_size"], "feature_sha256": e["sha256"],
                            "state_sha256": e["state_sha256"], "pipeline_sha256": e["pipeline_sha256"]}
                           for row, e in zip(rows, entries)]
    return {"complete": True, "labels_present": False, "cohorts": cohorts,
            "counts": COUNTS, "K": list(KS)}


def register():
    low_priority()
    table = model_table()
    sources = source_hashes()
    man = manifest()
    write_once(OUT / "public_manifest.json", man)
    pointer_protocol = read(V2 / "pointerbench/evaluation_protocol_v2.json")
    v2_protocol = read(V2 / "protocol.json")
    mm_protocol = read(MM / "protocol.json")
    labels = {
        "pointer": pointer_protocol["labels_source_sha256_unopened"],
        "full": v2_protocol["official_manifest_sha256_from_preparation_audit"],
        "mmbench": mm_protocol["source_annotation_sha256"],
    }
    expected = {"complete": True, "registered_at_utc": None,
                "purpose": "Retrospective full-aligned K-pool and exposure diagnostics on sealed public caches",
                "worker_sha256": sha(__file__), "public_manifest_sha256": sha(OUT / "public_manifest.json"),
                "source_hashes": sources,
                "model_sha256": {name: e["sha256"] for name, e in table.items()},
                "cohorts": list(COHORTS), "denominators": COUNTS, "K": list(KS),
                "K_definition": "V2 archived C3 pools at K4/8/16/32; score the union of these pools with the same fixed full-aligned S1/S2/S3 weights and training-only scalers; stable union-index Top1",
                "prediction_gate": "All 3581 K-pool predictions, per-case and suite SHA, before opening labels",
                "labels_sha256_unopened": labels,
                "label_access_before_predictions": False,
                "preview_ids": PREVIEW_IDS,
                "preview_sensitivity": "Drop exactly known seven preseal preview IDs from their original Pro/Sheets cohort, retaining every other query; report aligned vs no_evidence/spatial_shuffle/local MVP",
                "uncertainty": "Descriptive 95pct RGB-image cluster paired bootstrap, 2000 replicates, seed 20261114, conditional on three fixed fits",
                "bootstrap_seed": BOOT_SEED, "bootstrap_replicates": BOOT_REPLICATES,
                "interpretation": "K sweep and preview sensitivity are posthoc; SSP1581 is historically exposed descriptive. No training or model/prompt/candidate rule change."}
    p = OUT / "protocol.json"
    if p.exists():
        expected["registered_at_utc"] = read(p)["registered_at_utc"]
        if read(p) != expected:
            raise ValueError("Frozen protocol drift")
    else:
        expected["registered_at_utc"] = now()
        write_once(p, expected)
    status("registered", protocol_sha256=sha(p), n=sum(COUNTS.values()))


def gate():
    spec = read(OUT / "protocol.json")
    if spec.get("complete") is not True or spec["worker_sha256"] != sha(__file__) or \
       spec["source_hashes"] != source_hashes() or spec["model_sha256"] != \
       {name: e["sha256"] for name, e in model_table().items()} or \
       spec["label_access_before_predictions"] is not False:
        raise ValueError("Frozen diagnostic protocol drift")
    man = read(OUT / "public_manifest.json")
    if sha(OUT / "public_manifest.json") != spec["public_manifest_sha256"] or man != manifest():
        raise ValueError("Frozen diagnostic public manifest drift")
    return spec, man


def key(sample_id):
    return hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:24]


def case_path(cohort, sample_id):
    return OUT / cohort / "cases" / f"{key(sample_id)}.json"


def archived_case(cohort, sample_id):
    folder, _ = cohort_paths(cohort)
    return read(folder / "predictions/cases" / f"{engine.key(sample_id)}.json")


def score_case(cohort, public, entry, loaded, spec):
    record = engine.read_feature(entry, public)
    pipe = read(entry["pipeline_path"])
    if sha(entry["pipeline_path"]) != entry["pipeline_sha256"] or pipe["sample_id"] != public["sample_id"]:
        raise ValueError("Pipeline changed: " + public["sample_id"])
    pools = pipe["pools"]["c3"]
    point_to_ix = {tuple(map(int, p)): i for i, p in enumerate(record["points"])}
    if len(point_to_ix) != record["n"] or set(pools) != {str(k) for k in KS}:
        raise ValueError("Candidate union or C3 pool changed")
    ix_by_k = {}
    for k in KS:
        pts = [tuple(row["point"]) for row in pools[str(k)]]
        if len(pts) > k or len(set(pts)) != len(pts) or any(point not in point_to_ix for point in pts):
            raise ValueError("Invalid K pool: " + public["sample_id"])
        ix_by_k[str(k)] = [point_to_ix[point] for point in pts]
    if ix_by_k["16"] != record["baseline"].tolist():
        raise ValueError("K16 differs from frozen original baseline: " + public["sample_id"])
    ix_all = sorted({i for indices in ix_by_k.values() for i in indices})
    map_all = {old: new for new, old in enumerate(ix_all)}
    methods = {name: {} for name in loaded}
    if ix_all:
        sub = dict(record, n=len(ix_all), points=record["points"][ix_all], base=record["base"][ix_all],
                   active=record["active"][ix_all], state=record["state"][ix_all], mask=record["mask"][ix_all])
        with torch.inference_mode():
            for name, (model_entry, model, mean, std) in loaded.items():
                args, lengths = engine.prior.tensors([sub], mean, std, "aligned", "cpu")
                args = engine.transform_inputs(args, lengths, [record["sample_id"]], model_entry["variant"])
                score = model(**args).float().cpu().numpy()
                if score.shape != (len(ix_all),) or not np.isfinite(score).all():
                    raise ValueError("Invalid readout score vector")
                for k in KS:
                    pool = np.asarray(sorted(ix_by_k[str(k)]), dtype=np.int64)
                    pick = int(pool[np.argsort(-score[[map_all[int(i)] for i in pool]], kind="stable")[0]]) if len(pool) else None
                    methods[name][str(k)] = record["points"][pick].tolist() if pick is not None else None
    else:
        for name in loaded:
            for k in KS:
                methods[name][str(k)] = None
    old = archived_case(cohort, public["sample_id"])
    if any(methods[name]["16"] != old["methods"][name]["point"] for name in loaded):
        raise ValueError("K16 frozen prediction parity failed: " + public["sample_id"])
    return {"complete": True, "labels_opened": False, "sample_id": public["sample_id"],
            "cohort": cohort, "pixel_sha256": public["pixel_sha256"], "image_size": public["image_size"],
            "feature_sha256": entry["sha256"], "pipeline_sha256": entry["pipeline_sha256"],
            "model_sha256": spec["model_sha256"],
            "pool_points": {k: [list(map(int, record["points"][i])) for i in indices] for k, indices in ix_by_k.items()},
            "methods": methods, "union_n": int(record["n"]), "K16_parity": True}


def validate_case(saved, cohort, public, entry, spec):
    if saved.get("complete") is not True or saved.get("labels_opened") is not False or \
       saved.get("cohort") != cohort or saved.get("sample_id") != public["sample_id"] or \
       saved.get("pixel_sha256") != public["pixel_sha256"] or \
       saved.get("feature_sha256") != entry["sha256"] or \
       saved.get("pipeline_sha256") != entry["pipeline_sha256"] or \
       saved.get("model_sha256") != spec["model_sha256"] or saved.get("K16_parity") is not True:
        raise ValueError("Invalid cached diagnostic case: " + public["sample_id"])
    if set(saved["pool_points"]) != {str(k) for k in KS} or set(saved["methods"]) != set(spec["model_sha256"]):
        raise ValueError("Missing diagnostic predictions: " + public["sample_id"])
    width, height = public["image_size"]
    for pts in saved["pool_points"].values():
        for p in pts:
            if len(p) != 2 or not (0 <= p[0] < width and 0 <= p[1] < height):
                raise ValueError("Out of bounds pool point")
    for methods in saved["methods"].values():
        for k, p in methods.items():
            if p is not None and (len(p) != 2 or not (0 <= p[0] < width and 0 <= p[1] < height)):
                raise ValueError("Out of bounds predicted point")
            if p is not None and p not in saved["pool_points"][k]:
                raise ValueError("Predicted point outside registered K pool")


def smoke():
    """Read one label-free case per cohort before the protocol is frozen."""
    low_priority()
    table = model_table()
    loaded = engine.load_model_table(table, "cpu")
    spec = {"model_sha256": {name: entry["sha256"] for name, entry in table.items()}}
    for cohort in COHORTS:
        folder, public_path = cohort_paths(cohort)
        public = read(public_path)[0]
        entry = read(folder / "features/index.json")["cases"][0]
        saved = score_case(cohort, public, entry, loaded, spec)
        validate_case(saved, cohort, public, entry, spec)
        print(json.dumps({"cohort": cohort, "sample_id": public["sample_id"],
                          "K16_parity": saved["K16_parity"],
                          "pool_sizes": {k: len(v) for k, v in saved["pool_points"].items()}},
                         ensure_ascii=False), flush=True)


def predict():
    low_priority()
    spec, _ = gate()
    table = model_table()
    loaded = engine.load_model_table(table, "cpu")
    total = sum(COUNTS.values())
    done = 0
    started = time.perf_counter()
    try:
        for cohort in COHORTS:
            folder, public_path = cohort_paths(cohort)
            public, index = read(public_path), read(folder / "features/index.json")
            for public_row, entry in zip(public, index["cases"]):
                p = case_path(cohort, public_row["sample_id"])
                if p.exists():
                    validate_case(read(p), cohort, public_row, entry, spec)
                else:
                    saved = score_case(cohort, public_row, entry, loaded, spec)
                    validate_case(saved, cohort, public_row, entry, spec)
                    write_once(p, saved)
                done += 1
                if done % 25 == 0 or done == total:
                    status("predicting", completed=done, total=total, cohort=cohort,
                           elapsed_seconds=round(time.perf_counter() - started, 1))
            seal_cohort(cohort, spec, public, index["cases"])
        seal_all(spec)
    except Exception:
        error = {"at_utc": now(), "completed": done, "traceback": traceback.format_exc()}
        write_once(OUT / "failures" / (now().replace(":", "-") + ".json"), error)
        status("failed", completed=done, total=total)
        raise


def seal_cohort(cohort, spec, public, entries):
    hashes = {}
    for row, entry in zip(public, entries):
        p = case_path(cohort, row["sample_id"])
        validate_case(read(p), cohort, row, entry, spec)
        hashes[row["sample_id"]] = sha(p)
    write_once(OUT / cohort / "completion.json",
               {"complete": True, "cohort": cohort, "n": COUNTS[cohort], "labels_opened": False,
                "protocol_sha256": sha(OUT / "protocol.json"), "case_sha256": hashes})


def seal_all(spec):
    receipts = {}
    for cohort in COHORTS:
        p = OUT / cohort / "completion.json"
        rec = read(p)
        if rec["complete"] is not True or rec["n"] != COUNTS[cohort] or \
           rec["protocol_sha256"] != sha(OUT / "protocol.json"):
            raise ValueError("Incomplete diagnostic cohort: " + cohort)
        receipts[cohort] = sha(p)
    write_once(OUT / "public_completion.json",
               {"complete": True, "n": sum(COUNTS.values()), "labels_opened": False,
                "protocol_sha256": sha(OUT / "protocol.json"), "cohort_receipt_sha256": receipts,
                "K16_prediction_parity_all": True})
    status("predictions_sealed", n=sum(COUNTS.values()))


def verify_gate():
    spec, man = gate()
    receipt = read(OUT / "public_completion.json")
    if receipt["complete"] is not True or receipt["n"] != sum(COUNTS.values()) or \
       receipt["labels_opened"] is not False or receipt["protocol_sha256"] != sha(OUT / "protocol.json"):
        raise ValueError("Global prediction seal invalid")
    for cohort in COHORTS:
        folder, public_path = cohort_paths(cohort)
        public, index = read(public_path), read(folder / "features/index.json")
        p = OUT / cohort / "completion.json"
        if sha(p) != receipt["cohort_receipt_sha256"][cohort]:
            raise ValueError("Cohort seal SHA changed: " + cohort)
        rec = read(p)
        if rec["n"] != COUNTS[cohort] or len(rec["case_sha256"]) != COUNTS[cohort]:
            raise ValueError("Cohort denominator changed: " + cohort)
        for row, entry in zip(public, index["cases"]):
            cp = case_path(cohort, row["sample_id"])
            if sha(cp) != rec["case_sha256"][row["sample_id"]]:
                raise ValueError("Case seal changed: " + row["sample_id"])
            validate_case(read(cp), cohort, row, entry, spec)
    return spec, man


def paired_ci(diff, rows, seed):
    clusters = defaultdict(list)
    for ix, row in enumerate(rows):
        clusters[row["pixel_sha256"]].append(ix)
    groups = list(clusters.values())
    numerator = np.asarray([np.asarray(diff)[g].sum() for g in groups], np.float32)
    denominator = np.asarray([len(g) for g in groups], np.float32)
    draws = np.random.default_rng(seed).multinomial(len(groups), np.full(len(groups), 1 / len(groups)),
                                                    size=BOOT_REPLICATES).astype(np.float32)
    values = (draws @ numerator) / (draws @ denominator)
    return [float(v * 100) for v in np.quantile(values, [.025, .975])]


def analyze():
    low_priority()
    spec, man = verify_gate()
    # First access to target annotations begins here, after all 3,581 predictions seal.
    full_labels_path = ROOT / "data/public_v1/manifests/screenspot_pro.jsonl"
    pointer_labels_path = V2 / "pointerbench/labels.jsonl"
    mm_labels_path = ROOT / "data/mmbench_gui_l2_v1/source/L2_annotations.json"
    if sha(full_labels_path) != spec["labels_sha256_unopened"]["full"] or \
       sha(pointer_labels_path) != spec["labels_sha256_unopened"]["pointer"] or \
       sha(mm_labels_path) != spec["labels_sha256_unopened"]["mmbench"]:
        raise ValueError("Pinned label source SHA changed")
    full_labels = {row["sample_id"]: row["target_geometry"] for row in jsonl(full_labels_path)}
    pointer_labels = {row["sample_id"]: row["target_geometry"] for row in jsonl(pointer_labels_path)}
    mm_labels = {row["index"]: row for row in read(mm_labels_path)}
    import run_mmbench_gui_l2_v1 as mmengine

    summaries = {}
    preview = {}
    readout = ["Reviewer diagnostics V1: frozen full-aligned S1-S3 on archived C3 K4/8/16/32 pools.",
               "Posthoc, retrospective; fixed 262144 view budget; predictions sealed before this analysis opened labels.",
               "K16 point parity with archived V2/MMBench predictions required on every case."]
    for cohort in COHORTS:
        folder, public_path = cohort_paths(cohort)
        public = read(public_path)
        if cohort == "full":
            labels = full_labels
        elif cohort in ("pro", "sheets"):
            labels = pointer_labels
        else:
            labels = mm_labels
        if cohort != "mmbench" and set(row["sample_id"] for row in public) - set(labels):
            raise ValueError("Missing cohort labels: " + cohort)

        def hit(row, point):
            if point is None:
                return False
            if cohort == "mmbench":
                official = labels[row["source_index"]]
                if any(official[field] != row[field] for field in ("instruction", "image_size", "data_type", "platform", "grounding_type")):
                    raise ValueError("MMBench public/label identity mismatch")
                return bool(mmengine.official_hit(point, official["bbox"], row["image_size"]))
            size = row["image_size"]
            return bool(point_in_geometry((point[0] / size[0], point[1] / size[1]), labels[row["sample_id"]]))

        rows = [read(case_path(cohort, row["sample_id"])) for row in public]
        table = {}
        for k in KS:
            oracle = np.asarray([any(hit(pub, p) for p in pred["pool_points"][str(k)])
                                 for pub, pred in zip(public, rows)], np.float32)
            by_seed = np.asarray([[hit(pub, pred["methods"][f"full_aligned_s{s}"][str(k)])
                                   for pub, pred in zip(public, rows)] for s in SEEDS], np.float32)
            if np.any(by_seed > oracle[None, :]):
                raise ValueError("Prediction outside oracle coverage")
            krec = {"n": len(public), "oracle_reachable": int(oracle.sum()),
                    "oracle_percent": float(oracle.mean() * 100),
                    "seed_correct": [int(x) for x in by_seed.sum(1)],
                    "mean_accuracy_percent": float(by_seed.mean() * 100),
                    "mean_reachable_selection_percent": float(by_seed.sum() / (oracle.sum() * 3) * 100) if oracle.sum() else None,
                    "empty_pool": int(sum(not pred["pool_points"][str(k)] for pred in rows))}
            if k != 16:
                baseline = np.asarray([[hit(pub, pred["methods"][f"full_aligned_s{s}"]["16"])
                                        for pub, pred in zip(public, rows)] for s in SEEDS], np.float32)
                diff = (by_seed - baseline).mean(0)
                krec["minus_K16_pp"] = float(diff.mean() * 100)
                krec["minus_K16_ci95_pp"] = paired_ci(diff, public, BOOT_SEED + k)
            table[str(k)] = krec
        summaries[cohort] = table
        readout.append(f"{cohort.upper()} n={len(public)}")
        for k in KS:
            r = table[str(k)]
            delta = (f"; vs K16 {r['minus_K16_pp']:+.2f} pp, 95% cluster CI "
                     f"[{r['minus_K16_ci95_pp'][0]:+.2f}, {r['minus_K16_ci95_pp'][1]:+.2f}]"
                     if k != 16 else "")
            readout.append(f"  K{k}: oracle {r['oracle_reachable']}/{r['n']}={r['oracle_percent']:.2f}%; "
                           f"correct S1/S2/S3 {r['seed_correct']}; mean {r['mean_accuracy_percent']:.2f}%; "
                           f"selection|reachable {r['mean_reachable_selection_percent']:.2f}%{delta}")

        if cohort in PREVIEW_IDS:
            known = set(PREVIEW_IDS[cohort])
            if not known <= {row["sample_id"] for row in public}:
                raise ValueError("Known preview ID missing")
            mvp_rows = {row["sample_id"]: row for row in read(MVP / "predictions/mvp.json")["rows"]}
            kept = [i for i, row in enumerate(public) if row["sample_id"] not in known]
            conditions = ("full_aligned", "full_no_evidence", "full_spatial_shuffle")
            variants = {}
            archived = [archived_case(cohort, pub["sample_id"]) for pub in public]
            for cond in conditions:
                hits = np.asarray([[hit(pub, old["methods"][f"{cond}_s{s}"]["point"])
                                    for pub, old in zip(public, archived)] for s in SEEDS], np.float32)
                variants[cond] = {"all_seed_correct": [int(x) for x in hits.sum(1)],
                                  "kept_seed_correct": [int(x) for x in hits[:, kept].sum(1)],
                                  "all_mean_accuracy_percent": float(hits.mean() * 100),
                                  "kept_mean_accuracy_percent": float(hits[:, kept].mean() * 100)}
            mvp_hits = np.asarray([hit(pub, mvp_rows[pub["sample_id"]]["point"]) for pub in public], np.float32)
            variants["local_mvp"] = {"all_correct": int(mvp_hits.sum()),
                                      "kept_correct": int(mvp_hits[kept].sum()),
                                      "all_accuracy_percent": float(mvp_hits.mean() * 100),
                                      "kept_accuracy_percent": float(mvp_hits[kept].mean() * 100)}
            preview[cohort] = {"known_preview_ids": list(PREVIEW_IDS[cohort]), "original_n": len(public),
                               "kept_n": len(kept), "methods": variants,
                               "known_seven_exclusion_is_not_exhaustive": True}
            readout.append(f"  Known-preview exclusion: {len(known)} removed; {len(kept)} retained")
            for cond, metrics in variants.items():
                readout.append(f"    {cond}: {metrics['all_accuracy_percent'] if cond == 'local_mvp' else metrics['all_mean_accuracy_percent']:.2f}% "
                               f"-> {metrics['kept_accuracy_percent'] if cond == 'local_mvp' else metrics['kept_mean_accuracy_percent']:.2f}%")
    result = {"complete": True, "at_utc": now(), "protocol_sha256": sha(OUT / "protocol.json"),
              "public_completion_sha256": sha(OUT / "public_completion.json"),
              "label_source_sha256": spec["labels_sha256_unopened"],
              "cohorts": summaries, "known_preview_exclusion": preview,
              "interpretation": "Posthoc K sweep of already frozen scores; SSP historically exposed; removal of seven known previews does not certify all preview exposure absent."}
    write_once(OUT / "analysis/results.json", result)
    readout.extend(["", "Interpretation: K curves change only eligible C3 pool size, not views, weights, or feature extraction.",
                    "Known-preview sensitivity removes exactly seven identified cases; total preseal preview count is unknown.",
                    "SSP1581 is historical descriptive, not independent confirmation."])
    text = "\n".join(readout) + "\n"
    rp = OUT / "analysis/readout.txt"
    rp.parent.mkdir(parents=True, exist_ok=True)
    if rp.exists() and rp.read_text(encoding="utf-8") != text:
        raise ValueError("Frozen readout changed")
    rp.write_text(text, encoding="utf-8")
    write_once(OUT / "analysis/completion.json",
               {"complete": True, "protocol_sha256": sha(OUT / "protocol.json"),
                "predictions_sha256": sha(OUT / "public_completion.json"),
                "results_sha256": sha(OUT / "analysis/results.json"), "readout_sha256": sha(rp)})
    status("analyzed", cohorts=list(COHORTS), results_sha256=sha(OUT / "analysis/results.json"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("smoke", "register", "predict", "verify-gate", "analyze"))
    args = parser.parse_args()
    if args.stage == "smoke":
        smoke()
    elif args.stage == "register":
        register()
    elif args.stage == "predict":
        predict()
    elif args.stage == "verify-gate":
        verify_gate()
        status("gate_verified", n=sum(COUNTS.values()))
    else:
        analyze()
