"""CPU-only, same-cache simple learned readouts for the ClickAgent W3 review.

No VLM forward, GPU allocation, or evaluation-label access occurs here.  The
worker trains on the sealed full-data union, predicts on three sealed public
cohorts, and seals all predictions before a separate analysis may read labels.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time
import traceback

# Respect the concurrently running 1M-pixel VLM inference.
for _name in ("CUDA_VISIBLE_DEVICES",):
    os.environ[_name] = "-1"
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
OUT = ROOT / "runs/review_controls_v1/w3"
OLD = ROOT / "runs/full_scaling_v18/data/old_train_index.json"
NEW = ROOT / "runs/full_scaling_v18/features/new_train/index.json"
EVAL = {
    "pro": ROOT / "runs/paper_scale_ablation_v2/eval/pro/features/index.json",
    "sheets": ROOT / "runs/paper_scale_ablation_v2/eval/sheets/features/index.json",
    "mmbench": ROOT / "runs/mmbench_gui_new_source_v1/mmbench/features/index.json",
}
REGISTRY = ROOT / "runs/paper_scale_ablation_v1/models/frozen.json"
SEEDS = (20261311, 20261312, 20261313)
STEPS = 13376
MODEL_NAMES = ("linear_all", "head_weight96")
FULL_DIM = 32 + 4 * 294
HEAD_DIM = 32 + 96


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def low_priority() -> None:
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.SetPriorityClass.restype = ctypes.c_int
        # BELOW_NORMAL_PRIORITY_CLASS; leave the live VLM run favored by Windows.
        ok = kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x4000)
        require(bool(ok), "Could not set below-normal CPU priority")


def source_hashes() -> dict:
    return {str(path.relative_to(ROOT)).replace("\\", "/"): sha(path)
            for path in (OLD, NEW, *EVAL.values(), REGISTRY)}


def entries() -> list[dict]:
    result = read(OLD) + read(NEW)["cases"]
    require(len(result) == 13373, "Full training query count drift")
    require(len({e["sample_id"] for e in result}) == len(result), "Duplicate train ID")
    require(sum(bool(e["n"]) for e in result) == 12886, "Fitting query count drift")
    return result


def register() -> None:
    x = source_hashes()
    for suite, n in (("pro", 500), ("sheets", 500), ("mmbench", 1000)):
        ix = read(EVAL[suite])
        require(ix.get("complete") and ix["n"] == n and ix.get("labels_present") is False,
                f"Unsealed or labeled public index: {suite}")
    entries()
    expected = {
        "registered_at_utc": None,
        "purpose": "Retrospective same-candidate, same-training-query simple learned scorer controls for W3",
        "worker_sha256": sha(__file__), "source_sha256": x,
        "training_queries": 13373, "fitting_queries": 12886,
        "training_candidate_policy": "Full original decoded union; skip 487 empty records, as full aligned",
        "evaluation_cohorts": {"pro": 500, "sheets": 500, "mmbench": 1000},
        "evaluation_candidate_policy": "Frozen original K16; forced stable Top1 over sorted union indices",
        "feature_recipe": (
            "Use the original candidate base32 and generated-name-token state294/mask; "
            "fixed token max and log-mean-exp, then view max and mean, producing 4x294; "
            "linear_all uses base32+1176, head_weight96 uses base32 plus the equally averaged "
            "three attention measurements and four pool channels per head. "
            "Compute all normalizers from training candidates only."),
        "models": {"linear_all": {"input_dim": FULL_DIM, "form": "single affine candidate score and learned null"},
                   "head_weight96": {"input_dim": HEAD_DIM,
                                     "form": "single learned coefficient per attention head plus affine base32 and null"}},
        "training": {"seeds": list(SEEDS), "steps": STEPS, "batch_queries": 8,
                     "optimizer": "AdamW lr=2e-4 weight_decay=.02, gradient_clip=1",
                     "loss": "Original listwise multi-positive/null plus 0.25 balanced BCE",
                     "checkpoint": "fixed final step, no validation/test checkpoint selection",
                     "device": "cpu", "max_torch_threads": 2,
                     "training_feature_storage": "float16 memory map; float32 computation"},
        "prediction_order": ["pro", "sheets", "mmbench"],
        "analysis_plan": {
            "gate": "All six model predictions for every case in all three cohorts must pass case/bundle SHA verification before any ground-truth parsing",
            "metric": "Original frozen K16, forced stable Top1; PointerBench inclusive normalized point-in-target-geometry and MMBench-GUI pinned official inclusive normalized point-in-bbox",
            "comparison": "Each new model versus archived full_aligned and full_attention_only, corresponding seed indices 1/2/3, on the same cases and candidate pools; no checkpoint or model selection from test results",
            "primary_cohort": "PointerBench Pro500",
            "secondary_cohorts": ["PointerBench Sheets500", "MMBench-GUI selected L2 1000"],
            "summary": "Report per-seed hit counts and accuracies, three-fixed-seed mean, paired deltas and per-seed deltas",
            "interval": "Query-weighted paired nonparametric RGB-screenshot-cluster bootstrap, 10000 draws, fixed RNG seed 20261114, percentile 95% interval, conditional on three fixed seeds",
            "families": "W3 contrasts descriptive; no significance or superiority claim from overlapping 95% intervals",
        },
        "labels_may_open_before_prediction_seal": False,
        "current_1m_experiment_untouched": True,
    }
    path = OUT / "protocol.json"
    if path.exists():
        old = read(path)
        expected["registered_at_utc"] = old["registered_at_utc"]
        require(old == expected, "Frozen W3 protocol drift")
    else:
        expected["registered_at_utc"] = now()
        write(path, expected)
    print(json.dumps({"stage": "registered", "protocol_sha256": sha(path),
                      "training_queries": 13373, "evaluation_queries": 2000}), flush=True)


def gate() -> dict:
    path = OUT / "protocol.json"
    require(path.exists(), "Register before training or prediction")
    p = read(path)
    require(p["worker_sha256"] == sha(__file__) and p["source_sha256"] == source_hashes()
            and p["labels_may_open_before_prediction_seal"] is False,
            "Worker/source/protocol SHA drift")
    return p


def pooled(entry: dict, *, train: bool) -> tuple:
    import numpy as np
    for file_field, hash_field in (("path", "sha256"), ("state_path", "state_sha256"),
                                   ("mask_path", "mask_sha256")):
        require(sha(entry[file_field]) == entry[hash_field], "Source feature SHA drift: " + entry["sample_id"])
    if train:
        require(sha(entry["labels_path"]) == entry["labels_sha256"],
                "Source training label SHA drift: " + entry["sample_id"])
    with np.load(entry["path"], allow_pickle=False) as z:
        base = z["base"].astype(np.float32)
        points = z["points"].copy()
        baseline = z["baseline"].copy()
    state = np.load(entry["state_path"], mmap_mode="r", allow_pickle=False)
    mask = np.load(entry["mask_path"], mmap_mode="r", allow_pickle=False)
    n = int(entry["n"])
    require(base.shape == (n, 32) and state.shape[0] == n and state.shape[1] == 4 and
            state.shape[3] == 294 and mask.shape == state.shape[:3] and
            points.shape == (n, 2), "Feature shape drift")
    require(np.isfinite(base).all() and np.isfinite(state).all(), "Non-finite source features")
    if n:
        value = np.asarray(state, np.float32)
        valid = np.asarray(mask, bool)
        present = valid.any(axis=2)
        counts = valid.sum(axis=2).clip(1)
        maximum = np.where(valid[..., None], value, -np.inf).max(axis=2)
        maximum = np.where(present[..., None], maximum, 0.0)
        shifted = np.where(valid[..., None], value - maximum[:, :, None, :], -np.inf)
        exp_sum = np.exp(shifted).sum(axis=2)
        exp_sum = np.where(present[..., None], exp_sum, 1.0)
        lme = maximum + np.log(exp_sum / counts[..., None])
        lme = np.where(present[..., None], lme, 0.0)
        per_view = np.concatenate((maximum, lme), axis=-1)
        vmax = np.where(present[..., None], per_view, -np.inf).max(axis=1)
        vmax = np.where(present.any(axis=1)[:, None], vmax, 0.0)
        vmean = (per_view * present[..., None]).sum(axis=1) / present.sum(axis=1).clip(1)[:, None]
        full = np.concatenate((base, vmax, vmean), axis=-1).astype(np.float32)
        attn = full[:, 32:].reshape(n, 4, 294)[:, :, :288].reshape(n, 4, 3, 96)
        heads = np.concatenate((base, attn.mean(axis=(1, 2))), axis=-1).astype(np.float32)
    else:
        full = np.empty((0, FULL_DIM), dtype=np.float32)
        heads = np.empty((0, HEAD_DIM), dtype=np.float32)
    require(full.shape == (n, FULL_DIM) and heads.shape == (n, HEAD_DIM) and
            np.isfinite(full).all() and np.isfinite(heads).all(), "Invalid pooled features")
    labels = np.load(entry["labels_path"], allow_pickle=False) if train else None
    if train:
        require(labels.shape == (n,) and np.isfinite(labels).all(), "Training label shape drift")
    return full, heads, labels, points, baseline


def smoke() -> None:
    import numpy as np
    import torch
    torch.set_num_threads(2)
    assert not torch.cuda.is_available()
    sample = [e for e in entries() if e["n"]][:8]
    train_rows = []
    for e in sample:
        full, heads, labels, _, pool = pooled(e, train=True)
        require(len(pool) <= 16 and len(labels) == len(full) == len(heads), "Smoke train mismatch")
        train_rows.append((full, heads, labels))
    for suite in EVAL:
        for e in read(EVAL[suite])["cases"][:2]:
            full, heads, labels, points, pool = pooled(e, train=False)
            require(labels is None and len(pool) <= 16 and len(points) == len(full),
                    "Smoke public mismatch")
    from clickagent.evidence_retention import retention_loss
    durations = {}
    for kind, column in (("linear_all", 0), ("head_weight96", 1)):
        model = model_factory(FULL_DIM if column == 0 else HEAD_DIM)
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.02)
        xb = [torch.from_numpy(row[column]) for row in train_rows]
        yb = [torch.from_numpy(row[2]) for row in train_rows]
        t0 = time.perf_counter()
        for _ in range(40):
            opt.zero_grad(set_to_none=True)
            loss = torch.stack([retention_loss(model(x), y, model.null)
                                for x, y in zip(xb, yb)]).mean()
            require(bool(torch.isfinite(loss)), "Smoke loss non-finite")
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        durations[kind] = round((time.perf_counter() - t0) / 40, 4)
    print(json.dumps({"stage": "smoke_pass", "train_cases": len(sample),
                      "public_cases": 6, "full_dim": FULL_DIM, "head_dim": HEAD_DIM,
                      "seconds_per_train_step_sample_batch": durations}), flush=True)


def prepare() -> None:
    import numpy as np
    p = gate()
    fit = [e for e in entries() if e["n"]]
    total = sum(int(e["n"]) for e in fit)
    require(len(fit) == 12886 and total > 0, "Training size invalid")
    cache = OUT / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    complete = cache / "completion.json"
    if complete.exists():
        receipt = read(complete)
        require(receipt["protocol_sha256"] == sha(OUT / "protocol.json"), "Training cache protocol drift")
        for name, expected in receipt["artifact_sha256"].items():
            require(sha(cache / name) == expected, "Training cache SHA drift: " + name)
        return
    progress_path = cache / "progress.json"
    if progress_path.exists():
        progress = read(progress_path)
        require(progress["protocol_sha256"] == sha(OUT / "protocol.json") and
                progress["source_sha256"] == p["source_sha256"] and
                progress["total_queries"] == len(fit) and progress["total_candidates"] == total,
                "Partial training cache identity drift")
        full_map = np.lib.format.open_memmap(cache / "full.npy", mode="r+")
        head_map = np.lib.format.open_memmap(cache / "heads.npy", mode="r+")
        label_map = np.lib.format.open_memmap(cache / "labels.npy", mode="r+")
        require(full_map.shape == (total, FULL_DIM) and full_map.dtype == np.float16 and
                head_map.shape == (total, HEAD_DIM) and head_map.dtype == np.float16 and
                label_map.shape == (total,) and label_map.dtype == np.float32,
                "Partial training cache shape drift")
        start_i = int(progress["completed_queries"])
        cursor = int(progress["completed_candidates"])
        require(0 <= start_i <= len(fit) and cursor == sum(int(e["n"]) for e in fit[:start_i]),
                "Partial training cursor drift")
        offsets = np.zeros(len(fit) + 1, dtype=np.int64)
        offsets[:start_i+1] = progress["offset_prefix"]
        full_sum = np.asarray(progress["full_sum"], np.float64)
        full_sq = np.asarray(progress["full_sq"], np.float64)
        head_sum = np.asarray(progress["head_sum"], np.float64)
        head_sq = np.asarray(progress["head_sq"], np.float64)
    else:
        for name in ("full.npy", "heads.npy", "labels.npy", "offsets.npy", "normalizers.json"):
            require(not (cache / name).exists(), "Unregistered partial cache: " + name)
        full_map = np.lib.format.open_memmap(cache / "full.npy", mode="w+", dtype=np.float16,
                                             shape=(total, FULL_DIM))
        head_map = np.lib.format.open_memmap(cache / "heads.npy", mode="w+", dtype=np.float16,
                                             shape=(total, HEAD_DIM))
        label_map = np.lib.format.open_memmap(cache / "labels.npy", mode="w+", dtype=np.float32,
                                              shape=(total,))
        offsets = np.zeros(len(fit) + 1, dtype=np.int64)
        full_sum, full_sq = np.zeros(FULL_DIM, np.float64), np.zeros(FULL_DIM, np.float64)
        head_sum, head_sq = np.zeros(HEAD_DIM, np.float64), np.zeros(HEAD_DIM, np.float64)
        cursor = 0; start_i = 0
    started = time.perf_counter()
    for i in range(start_i + 1, len(fit) + 1):
        e = fit[i - 1]
        full, heads, labels, _, _ = pooled(e, train=True)
        n = len(full)
        full16, head16 = full.astype(np.float16), heads.astype(np.float16)
        require(np.isfinite(full16).all() and np.isfinite(head16).all(), "Pool overflow in float16 cache")
        full_map[cursor:cursor+n] = full16
        head_map[cursor:cursor+n] = head16
        label_map[cursor:cursor+n] = labels
        f64, h64 = full16.astype(np.float64), head16.astype(np.float64)
        full_sum += f64.sum(axis=0); full_sq += (f64 * f64).sum(axis=0)
        head_sum += h64.sum(axis=0); head_sq += (h64 * h64).sum(axis=0)
        cursor += n; offsets[i] = cursor
        if i % 100 == 0 or i == len(fit):
            full_map.flush(); head_map.flush(); label_map.flush()
            write(progress_path, {"protocol_sha256": sha(OUT / "protocol.json"),
                                  "source_sha256": p["source_sha256"],
                                  "total_queries": len(fit), "total_candidates": total,
                                  "completed_queries": i, "completed_candidates": cursor,
                                  "offset_prefix": offsets[:i+1].tolist(),
                                  "full_sum": full_sum.tolist(), "full_sq": full_sq.tolist(),
                                  "head_sum": head_sum.tolist(), "head_sq": head_sq.tolist()})
            write(OUT / "status.json", {"phase": "preparing", "at_utc": now(), "completed": i,
                                        "total": len(fit), "candidates": cursor,
                                        "elapsed_seconds": round(time.perf_counter() - started, 1)})
            if i % 500 == 0 or i == len(fit):
                print(json.dumps({"phase": "preparing", "completed": i, "total": len(fit),
                                  "elapsed_seconds": round(time.perf_counter() - started, 1)}), flush=True)
    require(cursor == total, "Candidate cursor drift")
    del full_map, head_map, label_map
    np.save(cache / "offsets.npy", offsets, allow_pickle=False)
    def normalizer(sums, sq):
        mean = sums / total
        std = np.sqrt(np.maximum(sq / total - mean * mean, 0.0))
        std[std < 1e-5] = 1.0
        return {"mean": mean.astype(np.float32).tolist(), "std": std.astype(np.float32).tolist()}
    write(cache / "normalizers.json", {"linear_all": normalizer(full_sum, full_sq),
                                        "head_weight96": normalizer(head_sum, head_sq),
                                        "candidate_n": total, "fit_query_n": len(fit)})
    artifacts = ["full.npy", "heads.npy", "labels.npy", "offsets.npy", "normalizers.json"]
    write(complete, {"complete": True, "protocol_sha256": sha(OUT / "protocol.json"),
                     "artifact_sha256": {name: sha(cache / name) for name in artifacts},
                     "candidate_n": total, "fit_query_n": len(fit)})
    progress_path.unlink()
    write(OUT / "status.json", {"phase": "prepared", "at_utc": now(),
                                "queries": len(fit), "candidates": total})


def model_factory(dim: int):
    import torch
    import torch.nn as nn
    class SimpleReadout(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(dim, 1)
            self.null = nn.Parameter(torch.zeros(()))
        def forward(self, x):
            return self.linear(x).squeeze(-1)
    return SimpleReadout()


def fit_one(kind: str, seed: int) -> None:
    import numpy as np
    import torch
    from clickagent.evidence_retention import retention_loss
    p = gate(); torch.set_num_threads(2)
    require(not torch.cuda.is_available(), "CPU-only experiment; CUDA must be invisible")
    cache = OUT / "cache"
    receipt = read(cache / "completion.json")
    require(receipt["complete"] and receipt["protocol_sha256"] == sha(OUT / "protocol.json"),
            "Training cache incomplete")
    dim = FULL_DIM if kind == "linear_all" else HEAD_DIM
    x = np.load(cache / ("full.npy" if kind == "linear_all" else "heads.npy"), mmap_mode="r")
    y = np.load(cache / "labels.npy", mmap_mode="r")
    offsets = np.load(cache / "offsets.npy", allow_pickle=False)
    fit_n = len(offsets) - 1
    stats = read(cache / "normalizers.json")[kind]
    mean, std = np.asarray(stats["mean"], np.float32), np.asarray(stats["std"], np.float32)
    require(x.shape[1] == dim and len(x) == len(y) == int(offsets[-1]) and fit_n == 12886,
            "Training matrix shape drift")
    tag = f"{kind}_s{SEEDS.index(seed)+1}"
    final = OUT / f"models/{tag}.pt"
    final_receipt = OUT / f"models/{tag}.json"
    if final_receipt.exists():
        saved = read(final_receipt)
        require(saved["complete"] and saved["checkpoint_sha256"] == sha(final),
                "Existing model receipt mismatch")
        return
    torch.manual_seed(seed); np.random.seed(seed)
    model = model_factory(dim)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=.02)
    resume = OUT / f"models/{tag}.resume.pt"
    step, epoch, position, elapsed = 0, 1, 0, 0.0
    if resume.exists():
        state = torch.load(resume, map_location="cpu", weights_only=True)
        require(state["protocol_sha256"] == sha(OUT / "protocol.json") and
                state["cache_completion_sha256"] == sha(cache / "completion.json"),
                "Resume identity drift")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        step, epoch, position, elapsed = state["step"], state["epoch"], state["position"], state["elapsed"]
        torch.set_rng_state(state["torch_rng"])
    start = time.perf_counter(); losses = []
    while step < STEPS:
        if position == fit_n:
            epoch += 1; position = 0
        order = np.random.default_rng(seed + epoch * 101).permutation(fit_n)
        ix = order[position:position+8]
        position += len(ix)
        slices = [slice(int(offsets[i]), int(offsets[i+1])) for i in ix]
        lengths = [s.stop - s.start for s in slices]
        xb = np.concatenate([np.asarray(x[s], np.float32) for s in slices], axis=0)
        yb = np.concatenate([np.asarray(y[s], np.float32) for s in slices], axis=0)
        xb = (xb - mean) / std
        xt = torch.from_numpy(xb); yt = torch.from_numpy(yb)
        optimizer.zero_grad(set_to_none=True)
        scores = model(xt)
        loss = torch.stack([retention_loss(s, labels, model.null)
                            for s, labels in zip(scores.split(lengths), yt.split(lengths))]).mean()
        require(bool(torch.isfinite(loss)), "Nonfinite loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step(); step += 1
        losses.append(float(loss.detach()))
        if step % 250 == 0 or step == STEPS:
            elapsed_now = elapsed + time.perf_counter() - start
            temp = resume.with_suffix(".pt.tmp")
            temp.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "step": step, "epoch": epoch, "position": position,
                        "elapsed": elapsed_now, "torch_rng": torch.get_rng_state(),
                        "protocol_sha256": sha(OUT / "protocol.json"),
                        "cache_completion_sha256": sha(cache / "completion.json")}, temp)
            os.replace(temp, resume)
            state = {"phase": "training", "at_utc": now(), "model": tag,
                     "completed_steps": step, "total_steps": STEPS,
                     "elapsed_seconds": round(elapsed_now, 1),
                     "mean_recent_loss": round(float(np.mean(losses)), 5)}
            losses = []
            write(OUT / "status.json", state)
            if step % 1000 == 0 or step == STEPS:
                print(json.dumps(state), flush=True)
    temp = final.with_suffix(".pt.tmp")
    torch.save({"state_dict": model.state_dict(), "kind": kind, "seed": seed,
                "mean": torch.tensor(mean), "std": torch.tensor(std),
                "protocol_sha256": sha(OUT / "protocol.json")}, temp)
    os.replace(temp, final)
    write(final_receipt, {"complete": True, "model": tag, "kind": kind, "seed": seed,
                          "parameter_count": sum(z.numel() for z in model.parameters()),
                          "checkpoint_sha256": sha(final), "checkpoint_path": str(final),
                          "steps": STEPS, "training_queries": 13373, "fitting_queries": fit_n,
                          "candidate_n": len(x), "protocol_sha256": sha(OUT / "protocol.json")})


def train() -> None:
    gate()
    for kind in MODEL_NAMES:
        for seed in SEEDS:
            fit_one(kind, seed)
    receipts = {f"{kind}_s{i}": sha(OUT / f"models/{kind}_s{i}.json")
                for kind in MODEL_NAMES for i in (1, 2, 3)}
    write(OUT / "models/completion.json", {"complete": True,
          "protocol_sha256": sha(OUT / "protocol.json"), "receipts_sha256": receipts})
    write(OUT / "status.json", {"phase": "models_frozen", "at_utc": now(), "models": 6})


def predict() -> None:
    import numpy as np
    import torch
    p = gate(); torch.set_num_threads(2)
    require(not torch.cuda.is_available(), "CPU-only experiment; CUDA must be invisible")
    frozen = read(OUT / "models/completion.json")
    require(frozen["complete"] and frozen["protocol_sha256"] == sha(OUT / "protocol.json"),
            "Models not frozen")
    loaded = {}
    for kind in MODEL_NAMES:
        dim = FULL_DIM if kind == "linear_all" else HEAD_DIM
        for i in (1, 2, 3):
            tag = f"{kind}_s{i}"
            receipt = read(OUT / f"models/{tag}.json")
            require(frozen["receipts_sha256"][tag] == sha(OUT / f"models/{tag}.json") and
                    receipt["checkpoint_sha256"] == sha(receipt["checkpoint_path"]),
                    "Model receipt drift")
            saved = torch.load(receipt["checkpoint_path"], map_location="cpu", weights_only=True)
            model = model_factory(dim); model.load_state_dict(saved["state_dict"]); model.eval()
            loaded[tag] = (model, saved["mean"].numpy(), saved["std"].numpy())
    for suite, ix_path in EVAL.items():
        ix = read(ix_path)
        case_hashes = {}
        for j, e in enumerate(ix["cases"], 1):
            path = OUT / f"predictions/{suite}/cases/{hashlib.sha256(e['sample_id'].encode()).hexdigest()[:24]}.json"
            if path.exists():
                row = read(path)
                require(row["sample_id"] == e["sample_id"] and row["feature_sha256"] == e["sha256"] and
                        row["model_hashes"] == {k: sha(OUT / f"models/{k}.pt") for k in loaded} and
                        row["labels_opened"] is False, "Cached prediction drift")
            else:
                full, heads, _, points, pool = pooled(e, train=False)
                preds = {}
                for tag, (model, mean, std) in loaded.items():
                    features = full if tag.startswith("linear_all") else heads
                    if len(pool):
                        with torch.inference_mode():
                            scores = model(torch.from_numpy((features - mean) / std)).numpy()
                        canonical = np.sort(pool)
                        chosen = int(canonical[np.argsort(-scores[canonical], kind="stable")[0]])
                        preds[tag] = {"index": chosen, "point": points[chosen].tolist()}
                    else:
                        preds[tag] = {"index": None, "point": None}
                row = {"sample_id": e["sample_id"], "suite": suite,
                       "feature_sha256": e["sha256"], "state_sha256": e["state_sha256"],
                       "mask_sha256": e["mask_sha256"], "predictions": preds,
                       "n": e["n"], "K16_count": len(pool),
                       "model_hashes": {k: sha(OUT / f"models/{k}.pt") for k in loaded},
                       "labels_opened": False, "ground_truth_in_model_input": False}
                write(path, row)
            case_hashes[e["sample_id"]] = sha(path)
            if j % 100 == 0 or j == len(ix["cases"]):
                state = {"phase": "predicting", "at_utc": now(), "suite": suite,
                         "completed": j, "total": len(ix["cases"])}
                write(OUT / "status.json", state)
                if j % 500 == 0 or j == len(ix["cases"]):
                    print(json.dumps(state), flush=True)
        bundle = OUT / f"predictions/{suite}/bundle.json"
        expected = {"complete": True, "suite": suite, "n": len(ix["cases"]),
                    "public_index_sha256": sha(ix_path),
                    "protocol_sha256": sha(OUT / "protocol.json"),
                    "case_hashes": case_hashes, "labels_opened": False}
        if bundle.exists():
            require(read(bundle) == expected, "Prediction bundle drift")
        else:
            write(bundle, expected)
    write(OUT / "predictions/completion.json", {"complete": True,
          "protocol_sha256": sha(OUT / "protocol.json"),
          "bundles_sha256": {suite: sha(OUT / f"predictions/{suite}/bundle.json") for suite in EVAL},
          "labels_opened": False})
    write(OUT / "status.json", {"phase": "predictions_sealed", "at_utc": now(),
                                "queries": 2000, "models": 6})


def verify_gate() -> None:
    gate()
    receipt = read(OUT / "predictions/completion.json")
    require(receipt["complete"] and receipt["labels_opened"] is False and
            receipt["protocol_sha256"] == sha(OUT / "protocol.json"),
            "Public prediction completion invalid")
    for suite, ix_path in EVAL.items():
        bundle_path = OUT / f"predictions/{suite}/bundle.json"
        require(receipt["bundles_sha256"][suite] == sha(bundle_path), "Prediction bundle SHA drift")
        bundle = read(bundle_path)
        ix = read(ix_path)
        require(bundle["n"] == len(ix["cases"]) and bundle["public_index_sha256"] == sha(ix_path),
                "Bundle cohort drift")
        for e in ix["cases"]:
            path = OUT / f"predictions/{suite}/cases/{hashlib.sha256(e['sample_id'].encode()).hexdigest()[:24]}.json"
            require(bundle["case_hashes"][e["sample_id"]] == sha(path), "Case prediction SHA drift")
            row = read(path)
            require(row["sample_id"] == e["sample_id"] and row["labels_opened"] is False and
                    row["K16_count"] <= 16 and len(row["predictions"]) == 6,
                    "Invalid public prediction row")
    print(json.dumps({"stage": "verify_gate_pass", "public_predictions": 2000,
                      "models": 6, "labels_opened": False}), flush=True)


if __name__ == "__main__":
    low_priority()
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=("smoke", "register", "prepare", "train", "predict", "verify-gate"))
    args = parser.parse_args()
    try:
        if args.stage == "smoke": smoke()
        elif args.stage == "register": register()
        elif args.stage == "prepare": prepare()
        elif args.stage == "train": train()
        elif args.stage == "predict": predict()
        elif args.stage == "verify-gate": verify_gate()
    except Exception:
        if args.stage != "smoke":
            write(OUT / f"failures/{now().replace(':', '-')}.json",
                  {"stage": args.stage, "at_utc": now(), "traceback": traceback.format_exc()})
        raise
