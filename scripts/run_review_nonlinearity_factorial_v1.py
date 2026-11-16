"""CPU-only pooled-feature 2x2 scorer control; no evaluation labels here.

The archived W3 linear_all arm is reused byte-for-byte. New arms share its
1208-dimensional pooled feature cache. Nothing in this worker opens test labels.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
for _key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_key] = "2"

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts"), str(ROOT / "src")]
import run_review_w3_simple_readout_v1 as w3

OUT = ROOT / "runs/review_nonlinearity_factorial_v1"
W3 = w3.OUT
ARMS = ("linear_base32", "pooled_mlp_all", "pooled_mlp_base32")
ALL_ARMS = ("linear_all", *ARMS)
SEEDS = w3.SEEDS
STEPS = w3.STEPS
DIM = w3.FULL_DIM
EXPECTED_TOTAL = 2_862_369
MLP_HIDDEN = (18, 14)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read(path: Path):
    return w3.read(path)


def write(path: Path, value) -> None:
    w3.write(path, value)


def sha(path: Path) -> str:
    return w3.sha(path)


def require(ok: bool, message: str) -> None:
    w3.require(ok, message)


def sources() -> dict[str, str]:
    files = [W3 / "protocol.json", W3 / "cache/completion.json",
             W3 / "cache/normalizers.json", W3 / "models/completion.json",
             W3 / "predictions/completion.json"]
    files += [W3 / f"models/linear_all_s{i}.pt" for i in (1, 2, 3)]
    files += [W3 / f"models/linear_all_s{i}.json" for i in (1, 2, 3)]
    files += [W3 / f"predictions/{suite}/bundle.json" for suite in w3.EVAL]
    files += list(w3.EVAL.values())
    return {str(p.relative_to(ROOT)).replace("\\", "/"): sha(p) for p in files}


def verify_training_cache_bytes() -> None:
    receipt = read(W3 / "cache/completion.json")
    require(receipt["complete"] and receipt["candidate_n"] == EXPECTED_TOTAL and
            receipt["fit_query_n"] == 12886, "W3 pooled cache receipt drift")
    for name in ("full.npy", "labels.npy", "offsets.npy", "normalizers.json"):
        require(sha(W3 / "cache" / name) == receipt["artifact_sha256"][name],
                "W3 pooled training byte SHA drift: " + name)


def register() -> None:
    verify_training_cache_bytes()
    require(read(W3 / "predictions/completion.json")["complete"], "W3 linear anchor unsealed")
    normalized = read(W3 / "cache/normalizers.json")["linear_all"]
    require(len(normalized["mean"]) == len(normalized["std"]) == DIM,
            "W3 normalizer dimension drift")
    record = {
        "registered_at_utc": now(),
        "purpose": "Retrospective architecture-by-normalization control on one shared pooled candidate representation",
        "worker_sha256": sha(Path(__file__)),
        "source_sha256": sources(),
        "cached_training_features_sha256": read(W3 / "cache/completion.json")["artifact_sha256"],
        "input": "Same W3 pooled 1208-dim base32+1176 feature for all four cells",
        "archived_anchor": "linear_all_s1/s2/s3 W3 model weights, normalizer and sealed predictions unchanged",
        "arms": {
            "linear_all": {"architecture": "affine+null", "normalization": "all1208", "source": "archived W3"},
            "linear_base32": {"architecture": "affine+null", "normalization": "base32_only", "source": "new"},
            "pooled_mlp_all": {"architecture": "1208-18-GELU-14-GELU-1+null", "normalization": "all1208", "source": "new"},
            "pooled_mlp_base32": {"architecture": "1208-18-GELU-14-GELU-1+null", "normalization": "base32_only", "source": "new"},
        },
        "normalization": "Training-candidate W3 normalizer, all dims for all1208; for base32_only apply its first32 mean/std and leave remaining1176 mean0/std1",
        "train_queries": 13373, "nonempty_train_queries": 12886,
        "train_candidates": EXPECTED_TOTAL,
        "training": {"seeds": list(SEEDS), "steps": STEPS, "batch_queries": 8,
                     "optimizer": "AdamW lr=2e-4 weight_decay=.02, gradient_clip=1",
                     "loss": "Original listwise multi-positive/null plus 0.25 balanced BCE",
                     "checkpoint": "fixed final step; no test or validation selection",
                     "device": "CPU, below-normal priority, max2 threads"},
        "test": {"pro": 500, "sheets": 500, "mmbench": 1000,
                 "candidate_rule": "archived original K16 forced stable Top1",
                 "per_view_pixels": 262144},
        "main_comparison": "pooled_mlp_all minus archived linear_all on Pro500, three fixed seed mean, paired RGB screenshot cluster bootstrap 10000 draws, percentile 95% CI",
        "secondary": "Same contrast Sheets500 and MMBench1000; 98.33% familywise CIs across three cohorts; base32 architecture contrasts, normalization effects and factorial interaction descriptive",
        "original_full_aligned_role": "External reference only; it learns token encodings before pooling and is not a factorial MLP cell",
        "labels_may_open_before_new_predictions_sealed": False,
        "retrospective_not_strict_blind": True,
        "mainline_1m_untouched": True,
    }
    path = OUT / "protocol.json"
    if path.exists():
        old = read(path)
        record["registered_at_utc"] = old["registered_at_utc"]
        require(record == old, "Frozen factorial protocol drift")
    else:
        write(path, record)
    print(json.dumps({"stage": "registered", "protocol_sha256": sha(path)}), flush=True)


def gate() -> dict:
    path = OUT / "protocol.json"
    require(path.exists(), "Register protocol before training")
    p = read(path)
    require(p["worker_sha256"] == sha(Path(__file__)) and p["source_sha256"] == sources(),
            "Frozen source or worker SHA drift")
    require(p["cached_training_features_sha256"] == read(W3 / "cache/completion.json")["artifact_sha256"],
            "Pooled training cache receipt drift")
    return p


def stats(kind: str):
    import numpy as np
    original = read(W3 / "cache/normalizers.json")["linear_all"]
    mean = np.asarray(original["mean"], np.float32)
    std = np.asarray(original["std"], np.float32)
    require(mean.shape == std.shape == (DIM,) and bool((std > 0).all()),
            "Invalid W3 normalizer")
    if kind.endswith("base32"):
        mean[32:] = 0.0
        std[32:] = 1.0
    return mean, std


def model_factory(kind: str):
    import torch
    import torch.nn as nn
    if kind.startswith("linear"):
        return w3.model_factory(DIM)
    class PooledMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(nn.Linear(DIM, MLP_HIDDEN[0]), nn.GELU(),
                                     nn.Linear(MLP_HIDDEN[0], MLP_HIDDEN[1]), nn.GELU(),
                                     nn.Linear(MLP_HIDDEN[1], 1))
            self.null = nn.Parameter(torch.zeros(()))
        def forward(self, x):
            return self.net(x).squeeze(-1)
    return PooledMLP()


def smoke() -> None:
    import numpy as np
    import torch
    torch.set_num_threads(2)
    require(not torch.cuda.is_available(), "CPU only")
    x = np.load(W3 / "cache/full.npy", mmap_mode="r")
    y = np.load(W3 / "cache/labels.npy", mmap_mode="r")
    offsets = np.load(W3 / "cache/offsets.npy", allow_pickle=False)
    require(x.shape == (EXPECTED_TOTAL, DIM) and y.shape == (EXPECTED_TOTAL,) and
            offsets.shape == (12887,) and offsets[-1] == EXPECTED_TOTAL,
            "Shared pooled training matrix drift")
    expected_params = {"linear_base32": 1210, "pooled_mlp_all": 22044,
                       "pooled_mlp_base32": 22044}
    result = {}
    for kind in ARMS:
        mean, std = stats(kind)
        model = model_factory(kind)
        n = sum(p.numel() for p in model.parameters())
        require(n == expected_params[kind], "Parameter count drift")
        a = (np.asarray(x[0:8], np.float32) - mean) / std
        score = model(torch.from_numpy(a))
        require(score.shape == (8,) and bool(torch.isfinite(score).all()),
                "Smoke model score invalid")
        result[kind] = n
    print(json.dumps({"stage": "smoke_pass", "parameter_count": result,
                      "candidate_count": len(x), "labels_opened": False}), flush=True)


def train_one(kind: str, seed: int) -> None:
    import numpy as np
    import torch
    from clickagent.evidence_retention import retention_loss
    gate(); torch.set_num_threads(2)
    require(not torch.cuda.is_available(), "CPU-only experiment")
    cache = W3 / "cache"
    x = np.load(cache / "full.npy", mmap_mode="r")
    y = np.load(cache / "labels.npy", mmap_mode="r")
    offsets = np.load(cache / "offsets.npy", allow_pickle=False)
    require(x.shape == (EXPECTED_TOTAL, DIM) and len(y) == EXPECTED_TOTAL and
            len(offsets) == 12887 and offsets[-1] == EXPECTED_TOTAL,
            "Shared training cache drift")
    mean, std = stats(kind)
    tag = f"{kind}_s{SEEDS.index(seed)+1}"
    model_path = OUT / f"models/{tag}.pt"
    receipt_path = OUT / f"models/{tag}.json"
    if receipt_path.exists():
        rec = read(receipt_path)
        require(rec["complete"] and rec["checkpoint_sha256"] == sha(model_path) and
                rec["protocol_sha256"] == sha(OUT / "protocol.json"),
                "Existing model receipt drift")
        return
    torch.manual_seed(seed); np.random.seed(seed)
    model = model_factory(kind)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.02)
    resume = OUT / f"models/{tag}.resume.pt"
    step, epoch, position, elapsed = 0, 1, 0, 0.0
    final_recent_loss_mean = None
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=True)
        require(state["protocol_sha256"] == sha(OUT / "protocol.json") and
                state["w3_cache_completion_sha256"] == sha(cache / "completion.json"),
                "Resume identity drift")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        step, epoch, position, elapsed = (state[k] for k in ("step", "epoch", "position", "elapsed"))
        torch.set_rng_state(state["torch_rng"])
        final_recent_loss_mean = state["recent_loss_mean"]
    started = time.perf_counter(); losses = []
    while step < STEPS:
        if position == len(offsets) - 1:
            epoch += 1; position = 0
        order = np.random.default_rng(seed + epoch * 101).permutation(len(offsets) - 1)
        ix = order[position:position+8]
        position += len(ix)
        slices = [slice(int(offsets[i]), int(offsets[i+1])) for i in ix]
        lengths = [s.stop - s.start for s in slices]
        xb = np.concatenate([np.asarray(x[s], np.float32) for s in slices], axis=0)
        yb = np.concatenate([np.asarray(y[s], np.float32) for s in slices], axis=0)
        xb = (xb - mean) / std
        xt = torch.from_numpy(xb); yt = torch.from_numpy(yb)
        optimizer.zero_grad(set_to_none=True)
        score = model(xt)
        loss = torch.stack([retention_loss(s, labels, model.null)
                            for s, labels in zip(score.split(lengths), yt.split(lengths))]).mean()
        require(bool(torch.isfinite(loss)), "Nonfinite train loss")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); step += 1
        losses.append(float(loss.detach()))
        if step % 250 == 0 or step == STEPS:
            elapsed_now = elapsed + time.perf_counter() - started
            final_recent_loss_mean = round(float(np.mean(losses)), 5)
            temp = resume.with_suffix(".pt.tmp")
            temp.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": step, "epoch": epoch, "position": position,
                        "elapsed": elapsed_now, "torch_rng": torch.get_rng_state(),
                        "recent_loss_mean": final_recent_loss_mean,
                        "protocol_sha256": sha(OUT / "protocol.json"),
                        "w3_cache_completion_sha256": sha(cache / "completion.json")}, temp)
            os.replace(temp, resume)
            state = {"phase": "training", "at_utc": now(), "model": tag,
                     "completed_steps": step, "total_steps": STEPS,
                     "elapsed_seconds": round(elapsed_now, 1),
                     "recent_loss_mean": final_recent_loss_mean}
            losses = []
            write(OUT / "status.json", state)
            if step % 1000 == 0 or step == STEPS:
                print(json.dumps(state), flush=True)
    temp = model_path.with_suffix(".pt.tmp")
    torch.save({"state_dict": model.state_dict(), "arm": kind, "seed": seed,
                "mean": torch.tensor(mean), "std": torch.tensor(std),
                "protocol_sha256": sha(OUT / "protocol.json")}, temp)
    os.replace(temp, model_path)
    write(receipt_path, {"complete": True, "arm": kind, "seed": seed,
                         "parameter_count": sum(p.numel() for p in model.parameters()),
                         "checkpoint_sha256": sha(model_path), "checkpoint_path": str(model_path),
                         "steps": STEPS, "fit_queries": 12886,
                         "train_candidates": EXPECTED_TOTAL,
                         "final_recent_loss_mean": final_recent_loss_mean,
                         "protocol_sha256": sha(OUT / "protocol.json")})


def train() -> None:
    gate()
    verify_training_cache_bytes()
    for kind in ARMS:
        for seed in SEEDS:
            train_one(kind, seed)
    receipts = {f"{kind}_s{i}": sha(OUT / f"models/{kind}_s{i}.json")
                for kind in ARMS for i in (1, 2, 3)}
    write(OUT / "models/completion.json", {"complete": True,
          "protocol_sha256": sha(OUT / "protocol.json"), "receipts_sha256": receipts})
    write(OUT / "status.json", {"phase": "models_frozen", "at_utc": now(), "models": 9})


def predict() -> None:
    import numpy as np
    import torch
    gate(); torch.set_num_threads(2)
    require(not torch.cuda.is_available(), "CPU-only experiment")
    frozen = read(OUT / "models/completion.json")
    require(frozen["complete"] and frozen["protocol_sha256"] == sha(OUT / "protocol.json"),
            "New models not sealed")
    models = {}
    for kind in ARMS:
        for i in (1, 2, 3):
            tag = f"{kind}_s{i}"
            rec = read(OUT / f"models/{tag}.json")
            require(frozen["receipts_sha256"][tag] == sha(OUT / f"models/{tag}.json") and
                    rec["checkpoint_sha256"] == sha(OUT / f"models/{tag}.pt"),
                    "New model receipt drift")
            saved = torch.load(OUT / f"models/{tag}.pt", map_location="cpu", weights_only=True)
            model = model_factory(kind); model.load_state_dict(saved["state_dict"]); model.eval()
            models[tag] = (model, saved["mean"].numpy(), saved["std"].numpy())
    for suite, ix_path in w3.EVAL.items():
        ix = read(ix_path)
        w3_bundle = read(W3 / f"predictions/{suite}/bundle.json")
        case_hashes = {}
        for j, entry in enumerate(ix["cases"], 1):
            sid = entry["sample_id"]
            key = hashlib.sha256(sid.encode()).hexdigest()[:24]
            old_path = W3 / f"predictions/{suite}/cases/{key}.json"
            require(w3_bundle["case_hashes"][sid] == sha(old_path),
                    "W3 anchor per-case SHA drift")
            old = read(old_path)
            path = OUT / f"predictions/{suite}/cases/{key}.json"
            if path.exists():
                row = read(path)
                require(row["sample_id"] == sid and row["feature_sha256"] == entry["sha256"] and
                        row["anchor_prediction_sha256"] == sha(old_path) and
                        row["model_hashes"] == {k: sha(OUT / f"models/{k}.pt") for k in models} and
                        row["labels_opened"] is False, "Cached new prediction drift")
            else:
                full, _, _, points, pool = w3.pooled(entry, train=False)
                require(len(pool) == old["K16_count"] and
                        old["feature_sha256"] == entry["sha256"],
                        "Shared eval feature/K16 drift")
                predictions = {}
                for tag, (model, mean, std) in models.items():
                    if len(pool):
                        with torch.inference_mode():
                            score = model(torch.from_numpy((full - mean) / std)).numpy()
                        canonical = np.sort(pool)
                        chosen = int(canonical[np.argsort(-score[canonical], kind="stable")[0]])
                        predictions[tag] = {"index": chosen, "point": points[chosen].tolist()}
                    else:
                        predictions[tag] = {"index": None, "point": None}
                row = {"sample_id": sid, "suite": suite, "feature_sha256": entry["sha256"],
                       "state_sha256": entry["state_sha256"], "mask_sha256": entry["mask_sha256"],
                       "K16_count": len(pool), "n": entry["n"],
                       "predictions": predictions,
                       "anchor_prediction_sha256": sha(old_path),
                       "model_hashes": {k: sha(OUT / f"models/{k}.pt") for k in models},
                       "labels_opened": False, "ground_truth_in_model_input": False}
                write(path, row)
            case_hashes[sid] = sha(path)
            if j % 100 == 0 or j == len(ix["cases"]):
                write(OUT / "status.json", {"phase": "predicting", "at_utc": now(),
                                            "suite": suite, "completed": j,
                                            "total": len(ix["cases"])})
        bundle = OUT / f"predictions/{suite}/bundle.json"
        expected = {"complete": True, "suite": suite, "n": len(ix["cases"]),
                    "public_index_sha256": sha(ix_path),
                    "protocol_sha256": sha(OUT / "protocol.json"),
                    "case_hashes": case_hashes, "labels_opened": False}
        if bundle.exists():
            require(read(bundle) == expected, "Existing bundle drift")
        else:
            write(bundle, expected)
    write(OUT / "predictions/completion.json", {"complete": True,
          "protocol_sha256": sha(OUT / "protocol.json"),
          "bundle_sha256": {suite: sha(OUT / f"predictions/{suite}/bundle.json") for suite in w3.EVAL},
          "labels_opened": False})
    write(OUT / "status.json", {"phase": "predictions_sealed", "at_utc": now(),
                                "queries": 2000, "new_models": 9})


def verify_gate() -> None:
    gate()
    receipt = read(OUT / "predictions/completion.json")
    require(receipt["complete"] and receipt["labels_opened"] is False and
            receipt["protocol_sha256"] == sha(OUT / "protocol.json"),
            "New prediction completion invalid")
    for suite, ix_path in w3.EVAL.items():
        bundle_path = OUT / f"predictions/{suite}/bundle.json"
        require(receipt["bundle_sha256"][suite] == sha(bundle_path),
                "New prediction bundle SHA drift")
        bundle = read(bundle_path); ix = read(ix_path)
        require(bundle["n"] == len(ix["cases"]) and
                bundle["public_index_sha256"] == sha(ix_path),
                "New prediction cohort drift")
        old_bundle = read(W3 / f"predictions/{suite}/bundle.json")
        for entry in ix["cases"]:
            sid = entry["sample_id"]
            key = hashlib.sha256(sid.encode()).hexdigest()[:24]
            path = OUT / f"predictions/{suite}/cases/{key}.json"
            old_path = W3 / f"predictions/{suite}/cases/{key}.json"
            require(bundle["case_hashes"][sid] == sha(path) and
                    old_bundle["case_hashes"][sid] == sha(old_path),
                    "New/anchor case SHA drift")
            row = read(path); old = read(old_path)
            require(row["sample_id"] == sid == old["sample_id"] and
                    row["feature_sha256"] == old["feature_sha256"] == entry["sha256"] and
                    row["K16_count"] == old["K16_count"] <= 16 and
                    row["anchor_prediction_sha256"] == sha(old_path) and
                    len(row["predictions"]) == 9 and row["labels_opened"] is False,
                    "New/anchor same-feature gate failed")
    print(json.dumps({"stage": "verify_gate_pass", "queries": 2000,
                      "new_models": 9, "labels_opened": False}), flush=True)


if __name__ == "__main__":
    w3.low_priority()
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=("register", "smoke", "train", "predict", "verify-gate"))
    args = parser.parse_args()
    try:
        if args.stage == "register": register()
        elif args.stage == "smoke": smoke()
        elif args.stage == "train": train()
        elif args.stage == "predict": predict()
        else: verify_gate()
    except Exception:
        if args.stage != "smoke":
            write(OUT / f"failures/{now().replace(':', '-')}.json",
                  {"stage": args.stage, "at_utc": now(), "traceback": traceback.format_exc()})
        raise
