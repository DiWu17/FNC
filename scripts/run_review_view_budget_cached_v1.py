"""Frozen-readout view-prefix diagnostics on archived, label-free GUI features.

This program never invokes Qwen, the proposer, or new coordinate generation.
It only masks archived evidence and reselects archived per-view proposals.
"""
from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import sys
from datetime import datetime, timezone

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import audit_view_budget_cache_v1 as audit
from clickagent.evidence_retention import RetentionModel

OUT = ROOT / "runs/review_view_budget_cached_v1"
PROTOCOL = OUT / "protocol.json"
STATUS = OUT / "status.json"
COMPLETE = OUT / "predictions/completion.json"
PUBLIC = {
    "pro": ROOT / "runs/paper_scale_ablation_v2/eval/pro/public.json",
    "sheets": ROOT / "runs/paper_scale_ablation_v2/eval/sheets/public.json",
    "mmbench": ROOT / "runs/mmbench_gui_new_source_v1/public.json",
}
ARCHIVED = {
    "pro": ROOT / "runs/paper_scale_ablation_v2/eval/pro/predictions",
    "sheets": ROOT / "runs/paper_scale_ablation_v2/eval/sheets/predictions",
    "mmbench": ROOT / "runs/mmbench_gui_new_source_v1/mmbench/predictions",
}
MM_ABLATION = ROOT / "runs/mmbench_full_ablation_supplement_v1/predictions"
REGISTRY = ROOT / "runs/paper_scale_ablation_v1/models/frozen.json"
W3_PROTOCOL = ROOT / "runs/review_controls_v1/w3/protocol.json"
MVP = ROOT / "runs/pointer_mvp_same_cohort_v1/predictions/mvp.json"
METHODS = [f"{kind}_s{s}" for kind in ("aligned", "no_evidence") for s in (1, 2, 3)]
VIEWS = (0, 1, 2, 3)


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    return audit.sha(path)


def read(path):
    return audit.read(path)


def write_once(path, value):
    if path.exists():
        require(read(path) == value, "Immutable output drift: " + str(path))
    else:
        audit.write(path, value)


def status(phase, **kwargs):
    audit.write(STATUS, {"phase": phase, "at_utc": datetime.now(timezone.utc).isoformat(), **kwargs})


def low_priority():
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["OMP_NUM_THREADS"] = "2"
    torch.set_num_threads(2)
    if os.name == "nt":
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = ctypes.c_void_p
        kernel.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
        kernel.SetPriorityClass(kernel.GetCurrentProcess(), 0x4000)


def source_record():
    registry = read(REGISTRY)["models"]
    models = {}
    for name in METHODS:
        entry = registry[f"full_{name}"]
        require(sha(entry["path"]) == entry["sha256"], "Frozen model SHA drift: " + name)
        models[name] = {"path": entry["path"], "sha256": entry["sha256"],
                        "seed": entry["seed"], "variant": entry["variant"]}
    paths = {**{f"feature_index_{k}": p for k, p in audit.PUBLIC.items()},
             **{f"public_{k}": p for k, p in PUBLIC.items()},
             "train_old": audit.TRAIN["old_train"], "train_new": audit.TRAIN["new_train"],
             "model_registry": REGISTRY, "w3_protocol": W3_PROTOCOL,
             "mvp_predictions": MVP, "audit_script": ROOT / "scripts/audit_view_budget_cache_v1.py"}
    return models, {k: sha(p) for k, p in paths.items()}


def register():
    require(not PROTOCOL.exists() and not COMPLETE.exists(), "Registration must precede new predictions")
    audit_receipt = OUT / "audit_all_test.json"
    require(audit_receipt.exists(), "Full 2,000-case label-free cache audit required")
    checked = read(audit_receipt)
    require(checked["complete"] and checked["label_access"] is False and
            all(checked["reports"][k]["n"] == n for k, n in (("pro", 500), ("sheets", 500), ("mmbench", 1000))),
            "Full audit incomplete")
    models, sources = source_record()
    value = {
        "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Inference-time cached view-prefix diagnostic; trained readouts remain frozen",
        "worker_sha256": sha(__file__), "audit_receipt_sha256": sha(audit_receipt),
        "source_sha256": sources, "models": models,
        "training": {"query_count": 13373, "nonempty": 12886, "steps": 13376, "batch_queries": 8,
                     "optimizer": "AdamW lr2e-4 wd0.02 clip1", "seeds": [20261311, 20261312, 20261313],
                     "status": "reuse exact frozen full aligned and no_evidence checkpoints; no view-specific retraining"},
        "tests": {"pro": 500, "sheets": 500, "mmbench": 1000},
        "feature_policy": "Existing full-union candidate coordinates and per-token 294-d cached states only; no Qwen/proposer calls",
        "A": "Hold archived C3 K16 fixed; zero later-view base32 source fields, recompute source/view counts, zero later-view state and mask. Future-view candidate coordinate remains by design.",
        "B": "Reassemble selection only from cached global plus first m local result lists. m0 uses global Top8 primary and Top16 sensitivity; m>=1 uses original C3 8-global+8-local quota. No new coordinate is generated. Local crop plans are the archived prefix of the global-only plan.",
        "candidate_budget": "Original K16 forced stable Top1; m0 Top8 primary is smaller due original 8+8 source quota",
        "counterfactual_scope": "Inference-time ablation on frozen full-view-trained readouts; not retrained limited-view systems and not a new end-to-end execution",
        "prediction_gate": "All 2000 cases and six frozen models at each m and setting must be SHA-sealed before opening target labels",
        "analysis_primary": "Pro500 A/B accuracy progression and coverage at m0..3; Sheets/MMBench secondary; screenshot-cluster paired intervals descriptive",
        "mvp_view_count_limit": "Original local MVP stored forwards and pixels but dropped per-case inference_count; only protocol maximum three distinct views is directly guaranteed",
        "our_forward_count_rule": "For each archived raw view with own_token_forward=true: one prefill plus one forward per generated_token_id; reconstructed from sealed raw metadata, not a live model call",
        "known_exposure": "PointerBench labels were previously analyzed and >=7 answers previewed; MMB subset historically image-disjoint, not full official denominator",
        "labels_opened": False,
    }
    write_once(PROTOCOL, value)
    print(json.dumps({"stage": "registered", "protocol_sha256": sha(PROTOCOL)}), flush=True)


def verify_protocol():
    require(PROTOCOL.exists(), "Register first")
    p = read(PROTOCOL)
    require(p["worker_sha256"] == sha(__file__), "Worker script SHA drift")
    require(p["audit_receipt_sha256"] == sha(OUT / "audit_all_test.json"), "Cache audit receipt drift")
    models, sources = source_record()
    require(p["models"] == models and p["source_sha256"] == sources, "Frozen sources drift")
    return p


def models_load(protocol):
    models = {}
    for name, info in protocol["models"].items():
        saved = torch.load(info["path"], map_location="cpu", weights_only=True)
        model = RetentionModel().eval()
        model.load_state_dict(saved["state_dict"])
        models[name] = (model, np.asarray(saved["mean"], np.float32), np.asarray(saved["std"], np.float32))
    return models


def pools_for_case(p, lookup, baseline):
    pools = {}
    for m in VIEWS:
        pools[f"A_m{m}"] = np.asarray(baseline, np.int64)
        if m == 0:
            for k in (8, 16):
                pools[f"B_m0_top{k}"] = np.asarray([lookup[tuple(item["point"])] for item in
                    audit.pool_prefix(p, 0, global_top=k)], np.int64)
        else:
            pools[f"B_m{m}"] = np.asarray([lookup[tuple(item["point"])] for item in audit.pool_prefix(p, m)], np.int64)
    return pools


def archive_case(suite, sid):
    key = hashlib.sha256(sid.encode()).hexdigest()[:24]
    original = read(ARCHIVED[suite] / "cases" / f"{key}.json")
    extra = read(MM_ABLATION / "cases" / f"{key}.json") if suite == "mmbench" else None
    return original, extra


def score_one(entry, public, suite, models):
    require(entry["sample_id"] == public["sample_id"], "Public/feature sample order drift")
    for field, hash_field in (("path", "sha256"), ("state_path", "state_sha256"),
                              ("mask_path", "mask_sha256"), ("pipeline_path", "pipeline_sha256")):
        require(sha(entry[field]) == entry[hash_field], "Feature source SHA drift: " + entry["sample_id"])
    p = read(entry["pipeline_path"])
    raw_views = [p["global_feature_path"]] + [v["feature_path"] for v in p["local_views"]]
    per_view_forwards = []
    for raw in raw_views:
        with np.load(raw, allow_pickle=False) as z:
            meta = json.loads(z["metadata"].item())
        require(meta.get("own_token_forward") is True and
                isinstance(meta.get("generated_token_ids"), list),
                "Raw view lacks auditable own-token forward sequence")
        per_view_forwards.append(1 + len(meta["generated_token_ids"]))
    with np.load(entry["path"], allow_pickle=False) as z:
        points = z["points"].copy(); base = z["base"].copy(); active = z["active"].copy(); baseline = z["baseline"].copy()
    state = np.load(entry["state_path"], mmap_mode="r", allow_pickle=False)
    mask = np.load(entry["mask_path"], mmap_mode="r", allow_pickle=False)
    lookup = {tuple(x): i for i, x in enumerate(points.tolist())}
    require(len(lookup) == len(points), "Full union duplicate coordinates")
    require(audit.pool_prefix(p, 3) == p["pools"]["c3"]["16"], "m3 pool source drift")
    require(np.array_equal(baseline, [lookup[tuple(v["point"])] for v in p["pools"]["c3"]["16"]]),
            "m3 K16 mismatch")
    pools = pools_for_case(p, lookup, baseline)
    predictions = {}; masked = {}
    with torch.inference_mode():
        for m in VIEWS:
            ms = [name for name in pools if name.endswith(f"m{m}") or (m == 0 and name.startswith("B_m0_"))]
            indices = np.unique(np.concatenate([pools[k] for k in ms])).astype(np.int64)
            b = audit.masked_base(base, p, m)[indices]
            a = active[indices].copy(); a[:, m + 1:] = False
            s = np.asarray(state[indices]).copy(); q = np.asarray(mask[indices]).copy()
            s[:, m + 1:] = 0; q[:, m + 1:] = False
            masked[m] = (indices, b, a, s, q)
            for method, (model, mean, std) in models.items():
                x = torch.from_numpy((b - mean) / std)
                token = torch.from_numpy(np.zeros_like(s) if method.startswith("no_evidence") else s)
                score = model(base=x, active=torch.from_numpy(a), state=token,
                              mask=torch.from_numpy(q)).numpy()
                require(np.isfinite(score).all(), "Nonfinite score")
                mapping = {int(ix): float(value) for ix, value in zip(indices, score)}
                for pool_name in ms:
                    # Preserve the proposal order: archived forced Top1 uses
                    # first-in-pool tie-breaking, not sorted union indices.
                    pool = pools[pool_name]
                    selected = int(pool[np.argmax([mapping[int(ix)] for ix in pool])]) if len(pool) else None
                    predictions[f"{pool_name}_{method}"] = {
                        "index": selected, "point": points[selected].tolist() if selected is not None else None,
                    }
    original, extra = archive_case(suite, entry["sample_id"])
    require(original["feature_sha256"] == entry["sha256"], "Archive feature identity drift")
    for method in METHODS:
        old = original["methods"].get(f"full_{method}")
        if old is None and extra is not None:
            old = extra["methods"].get(f"full_{method}")
        require(old is not None, "Missing archived reference: " + method)
        require(predictions[f"A_m3_{method}"]["point"] == old["point"] and
                predictions[f"B_m3_{method}"]["point"] == old["point"],
                "m3 prediction parity failed: " + method)
    output = {"sample_id": entry["sample_id"], "cohort": suite, "feature_sha256": entry["sha256"],
              "pipeline_sha256": entry["pipeline_sha256"], "local_views": len(p["local_views"]),
              "actual_view_count": len(raw_views), "backbone_forward_calls_reconstructed": sum(per_view_forwards),
              "per_view_backbone_forwards_reconstructed": per_view_forwards,
              "pool_points": {name: points[ix].tolist() for name, ix in pools.items()},
              "predictions": predictions, "labels_opened": False}
    return output


def predict():
    protocol = verify_protocol()
    models = models_load(protocol)
    all_hashes = {}
    for suite, index_path in audit.PUBLIC.items():
        entries = audit.entries(index_path)
        public = read(PUBLIC[suite])
        require(len(entries) == len(public) == protocol["tests"][suite], "Cohort size drift")
        hashes = {}
        for i, (entry, row) in enumerate(zip(entries, public), 1):
            sid = entry["sample_id"]
            key = hashlib.sha256(sid.encode()).hexdigest()[:24]
            target = OUT / "predictions" / suite / "cases" / f"{key}.json"
            if target.exists():
                value = read(target)
                require(value["sample_id"] == sid and value["feature_sha256"] == entry["sha256"] and
                        value["pipeline_sha256"] == entry["pipeline_sha256"] and
                        value["labels_opened"] is False, "Existing case invalid")
            else:
                value = score_one(entry, row, suite, models)
                audit.write(target, value)
            hashes[sid] = sha(target)
            if i % 50 == 0:
                status("predicting", cohort=suite, completed=i, total=len(entries), labels_opened=False)
        all_hashes[suite] = hashes
        write_once(OUT / "predictions" / suite / "completion.json", {
            "complete": True, "cohort": suite, "n": len(entries), "case_hashes": hashes,
            "protocol_sha256": sha(PROTOCOL), "labels_opened": False})
    write_once(COMPLETE, {"complete": True, "n": sum(len(h) for h in all_hashes.values()),
                          "cohorts": {k: {"n": len(v), "completion_sha256": sha(OUT / "predictions" / k / "completion.json")}
                                      for k, v in all_hashes.items()},
                          "protocol_sha256": sha(PROTOCOL), "labels_opened": False})
    status("predictions_sealed", n=2000, completion_sha256=sha(COMPLETE), labels_opened=False)


def verify_gate():
    protocol = verify_protocol()
    require(COMPLETE.exists(), "All predictions must seal first")
    full = read(COMPLETE)
    require(full["complete"] and full["n"] == 2000 and full["protocol_sha256"] == sha(PROTOCOL)
            and full["labels_opened"] is False, "Prediction seal invalid")
    for suite, index_path in audit.PUBLIC.items():
        entries = audit.entries(index_path)
        seal_path = OUT / "predictions" / suite / "completion.json"
        require(full["cohorts"][suite]["completion_sha256"] == sha(seal_path), "Cohort receipt SHA drift")
        receipt = read(seal_path)
        require(receipt["n"] == len(entries) == protocol["tests"][suite], "Cohort denominator drift")
        for e in entries:
            sid = e["sample_id"]
            key = hashlib.sha256(sid.encode()).hexdigest()[:24]
            case = OUT / "predictions" / suite / "cases" / f"{key}.json"
            require(receipt["case_hashes"][sid] == sha(case), "Prediction case SHA drift")
            row = read(case)
            require(row["sample_id"] == sid and row["feature_sha256"] == e["sha256"] and
                    row["labels_opened"] is False and len(row["predictions"]) == 54,
                    "Prediction case content drift")
    return {"complete": True, "n": 2000, "protocol_sha256": sha(PROTOCOL),
            "prediction_completion_sha256": sha(COMPLETE), "labels_opened": False}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=("register", "predict", "verify-gate"))
    args = parser.parse_args()
    low_priority()
    if args.stage == "register":
        register()
    elif args.stage == "predict":
        predict()
    else:
        value = verify_gate()
        write_once(OUT / "verify_gate.json", value)
        print(json.dumps(value), flush=True)


if __name__ == "__main__":
    main()
