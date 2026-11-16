"""Cached-feature fine-grained attention ablations; no backbone or proposer calls.

Only the 294-channel state input transform changes. The unchanged RetentionModel,
full training union, optimizer, loss, seed, step and K16 evaluation conventions are
inherited from the sealed full-data ablation workflow.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np
import torch
from clickagent.contribution_ablation import transform_inputs as old_transform
from clickagent.evidence_retention import RetentionModel, retention_loss
import run_quality_readout_v15_3k as prior

OUT = ROOT / "runs/review_attention_finegrained_v1"
V18 = ROOT / "runs/full_scaling_v18"
REG = OUT / "protocol.json"
SEEDS = (20261311, 20261312, 20261313)
STEPS = 13376
VARIANTS = ("head_mean_all96", "head_mean_per_layer", "cell_only", "peak_only",
            "neighborhood_only", "cell_peak")
ORIGINAL_TENSORS = prior.tensors
SOURCE_FILES = (
    "paper/GSAIE/DRAFT_NOTES.txt",
    "runs/full_scaling_v18/data/old_train_index.json",
    "runs/full_scaling_v18/data/new_train.jsonl",
    "runs/full_scaling_v18/features/new_train/index.json",
    "runs/full_scaling_v18/models/frozen.json",
    "runs/paper_scale_ablation_v1/models/frozen.json",
    "runs/paper_scale_ablation_v2/eval/pro/features/index.json",
    "runs/paper_scale_ablation_v2/eval/sheets/features/index.json",
    "runs/mmbench_gui_new_source_v1/mmbench/features/index.json",
    "src/clickagent/evidence_retention.py",
    "src/clickagent/model_features.py",
    "src/clickagent/contribution_ablation.py",
    "scripts/run_quality_readout_v15_3k.py",
    "scripts/run_evidence_retention_v14.py",
)


def require(ok: bool, msg: str) -> None:
    if not ok:
        raise ValueError(msg)


def sha(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
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


def low_priority() -> None:
    if os.name == "nt":
        import ctypes
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel32.GetCurrentProcess.argtypes = []
        kernel32.GetCurrentProcess.restype = ctypes.c_void_p
        kernel32.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        kernel32.SetPriorityClass.restype = ctypes.c_int
        require(bool(kernel32.SetPriorityClass(kernel32.GetCurrentProcess(), 0x4000)),
                "Could not set below-normal CPU priority")


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def status(phase: str, **kwargs) -> None:
    value = dict(phase=phase, at_utc=now(), pid=os.getpid(), **kwargs)
    write(OUT / "status.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def source_hashes() -> dict[str, str]:
    return {name: sha(ROOT / name) for name in SOURCE_FILES}


def training_entries() -> list[dict]:
    entries = read(V18 / "data/old_train_index.json") + read(
        V18 / "features/new_train/index.json")["cases"]
    require(len(entries) == 13373 and len({e["sample_id"] for e in entries}) == 13373,
            "Full training query/order drift")
    require(sum(int(e["n"] > 0) for e in entries) == 12886,
            "Nonempty training query count drift")
    return entries


def reference_models() -> dict:
    frozen = read(ROOT / "runs/paper_scale_ablation_v1/models/frozen.json")
    require(frozen["complete"], "Full ablation model registry incomplete")
    result = {}
    for s in (1, 2, 3):
        name = f"full_attention_only_s{s}"
        e = frozen["models"][name]
        require(sha(e["path"]) == e["sha256"], "Attention-only checkpoint drift")
        result[name] = {"path": e["path"], "sha256": e["sha256"]}
    return result


def register() -> None:
    entries = training_entries()
    expected = dict(
        registered_at_utc=now(),
        purpose="Retrospective six-arm fine-grained 288-channel attention input ablation",
        worker_sha256=sha(__file__), source_sha256=source_hashes(),
        train_query_order_sha256=hashlib.sha256(
            json.dumps([e["sample_id"] for e in entries], separators=(",", ":"),
                       ensure_ascii=False).encode("utf-8")).hexdigest(),
        training_queries=13373, fitting_queries=12886, skipped_empty_queries=487,
        reference_models=reference_models(),
        variants=list(VARIANTS),
        channels=dict(cell=[0, 96], peak=[96, 192], neighborhood=[192, 288], hidden=[288, 294],
                      layers={"15": [0, 32], "28": [32, 64], "36": [64, 96]}),
        transform="Zero hidden6 in every arm, then mask/mean attention channels only; same transform at train and test, unchanged base32/active/mask/candidates/order",
        attention_reference="frozen full_attention_only S1/S2/S3 on the same K16 test cases",
        model="Unchanged RetentionModel 22050 parameters; same training-only base32 scaler",
        training=dict(seeds=list(SEEDS), steps=STEPS, batch_queries=8, optimizer="AdamW",
                      learning_rate=0.0002, weight_decay=0.02, grad_clip=1.0,
                      loss="listwise multi-positive/null + 0.25 balanced BCE",
                      checkpoint="fixed final step, no validation/test model selection"),
        evaluation=dict(cohorts={"pro": 500, "sheets": 500, "mmbench": 1000},
                        candidate_rule="original frozen K16 stable forced Top1",
                        per_view_pixels=262144, null_or_empty="wrong but retained in denominator"),
        integrity="All 18 checkpoints SHA-sealed; all 2000x18 label-free predictions SHA-sealed before target geometry parsed; official scorer pointwise parity",
        analysis="Report S1/S2/S3 hits, three-seed mean, paired screenshot-cluster intervals versus frozen attention_only; descriptive multiple comparisons, no best-arm selection",
        retrospective=True, strict_blind=False, no_backbone_or_candidate_regeneration=True,
        optional_layer_only_not_in_this_protocol=True,
    )
    if REG.exists():
        old = read(REG)
        expected["registered_at_utc"] = old["registered_at_utc"]
        require(old == expected, "Frozen attention ablation protocol drift")
    else:
        write(REG, expected)
    print(json.dumps({"stage": "registered", "protocol_sha256": sha(REG),
                      "variants": list(VARIANTS)}), flush=True)


def gate() -> dict:
    require(REG.exists(), "Register before work")
    p = read(REG)
    require(p["worker_sha256"] == sha(__file__) and p["source_sha256"] == source_hashes(),
            "Frozen source/script drift")
    require(p["reference_models"] == reference_models(), "Frozen reference drift")
    require(p["variants"] == list(VARIANTS) and p["training"]["steps"] == STEPS,
            "Protocol arm/schedule drift")
    return p


def transformed(inputs: dict, variant: str) -> dict:
    require(variant in VARIANTS or variant == "attention_only_identity", "Unknown variant")
    state = inputs["state"]
    require(state.ndim == 4 and state.shape[-1] == 294, "State channel shape drift")
    result = state.clone()
    result[..., 288:294] = 0
    if variant == "head_mean_all96":
        for start in (0, 96, 192):
            block = result[..., start:start+96]
            result[..., start:start+96] = block.float().mean(-1, keepdim=True).to(block.dtype)
    elif variant == "head_mean_per_layer":
        for start in (0, 96, 192):
            for h in (0, 32, 64):
                block = result[..., start+h:start+h+32]
                result[..., start+h:start+h+32] = block.float().mean(-1, keepdim=True).to(block.dtype)
    elif variant == "cell_only":
        result[..., 96:288] = 0
    elif variant == "peak_only":
        result[..., 0:96] = 0
        result[..., 192:288] = 0
    elif variant == "neighborhood_only":
        result[..., 0:192] = 0
    elif variant == "cell_peak":
        result[..., 192:288] = 0
    return dict(inputs, state=result)


def build_inputs(records, mean, std, variant: str, device: str):
    args, lengths = ORIGINAL_TENSORS(records, mean, std, "aligned", device)
    return transformed(args, variant), lengths


@contextmanager
def train_adapter(variant: str):
    saved = prior.OUT, prior.status, prior.tensors
    prior.OUT = OUT
    prior.status = status
    prior.tensors = lambda records, mean, std, ignored, device: build_inputs(
        records, mean, std, variant, device)
    try:
        yield
    finally:
        prior.OUT, prior.status, prior.tensors = saved


def check_head_order() -> list[str]:
    # This is archived public raw metadata only; no model call or annotation.
    index = read(ROOT / "runs/paper_scale_ablation_v2/eval/pro/features/index.json")
    pipe = read(index["cases"][0]["pipeline_path"])
    with np.load(pipe["global_feature_path"], allow_pickle=False) as raw:
        meta = json.loads(str(raw["metadata"]))
    heads = meta["head_order"]
    expected = [f"layer_{layer}.head_{head}" for layer in (15, 28, 36) for head in range(32)]
    require(heads == expected, "Archived 96-head layer order differs")
    return heads


def selftest() -> None:
    low_priority(); torch.set_num_threads(2)
    require(not torch.cuda.is_available(), "Hide CUDA for CPU selftest")
    check_head_order()
    entries = [e for e in training_entries() if e["n"] > 0][:16]
    records = prior.base.load_records(entries)
    mean = np.concatenate([r["base"] for r in records]).mean(0)
    std = np.concatenate([r["base"] for r in records]).std(0)
    std[std < 1e-5] = 1
    args, lengths = ORIGINAL_TENSORS(records[:2], mean, std, "aligned", "cpu")
    identity = transformed(args, "attention_only_identity")
    reference = old_transform(args, lengths, [r["sample_id"] for r in records[:2]],
                              "attention_only")
    require(torch.equal(identity["state"], reference["state"]),
            "Attention-only input transform parity failed")
    state = args["state"]
    for name in VARIANTS:
        changed = transformed(args, name)
        require(torch.equal(changed["base"], args["base"]) and
                torch.equal(changed["mask"], args["mask"]) and
                torch.equal(changed["active"], args["active"]),
                "Non-state inputs changed: " + name)
        require(bool((changed["state"][..., 288:] == 0).all()), "Hidden6 retained: " + name)
        require(bool(torch.isfinite(changed["state"]).all()), "Nonfinite variant: " + name)
        if name == "head_mean_all96":
            for start in (0, 96, 192):
                require(bool((changed["state"][..., start:start+96] ==
                              changed["state"][..., start:start+1]).all()),
                        "Head mean not broadcast: " + name)
        if name in ("cell_only", "peak_only", "neighborhood_only", "cell_peak"):
            ranges = {"cell_only": [(96, 288)], "peak_only": [(0, 96), (192, 288)],
                      "neighborhood_only": [(0, 192)], "cell_peak": [(192, 288)]}[name]
            for a, b in ranges:
                require(bool((changed["state"][..., a:b] == 0).all()),
                        "Removed channel retained: " + name)
        model = RetentionModel().cpu()
        scores = model(**changed)
        targets = torch.from_numpy(np.concatenate([r["labels"] for r in records[:2]])).float()
        loss = torch.stack([retention_loss(s, y, model.null)
                            for s, y in zip(scores.split(lengths), targets.split(lengths))]).mean()
        require(bool(torch.isfinite(loss)), "Nonfinite smoke loss: " + name)
        loss.backward()
    receipts = read(V18 / "models/frozen.json")["models"]
    for i, seed in enumerate(SEEDS, 1):
        torch.manual_seed(seed)
        model = RetentionModel()
        digest = prior.digest({k: hashlib.sha256(v.detach().cpu().numpy().tobytes()).hexdigest()
                               for k, v in model.state_dict().items()})
        reference_receipt = read(V18 / f"models/full_s{i}.json")
        require(digest == reference_receipt["initial_state_sha256"],
                "Seed initial state differs from frozen aligned: " + str(seed))
    write(OUT / "smoke.json", dict(complete=True, at_utc=now(),
                                   code_sha256=sha(__file__), records=len(records),
                                   variants=list(VARIANTS), layer_head_order=check_head_order(),
                                   attention_only_input_parity=True,
                                   seed_initial_state_parity=True,
                                   labels_scope="training labels only; no test labels opened",
                                   no_backbone_or_proposer_call=True))
    print(json.dumps({"stage": "selftest_pass", "variants": len(VARIANTS),
                      "training_records": len(records)}), flush=True)


def bench250() -> None:
    gate(); low_priority(); torch.set_num_threads(2)
    require(not torch.cuda.is_available(), "Hide CUDA for CPU benchmark")
    target = OUT / "bench250"
    target.mkdir(parents=True, exist_ok=True)
    proto = target / "protocol.json"
    value = dict(parent_protocol_sha256=sha(REG), code_sha256=sha(__file__),
                 variant="cell_peak", seed=SEEDS[0], steps=250,
                 full_training_union=True, device="cpu", no_test_labels=True)
    if proto.exists():
        require(read(proto) == value, "Benchmark protocol drift")
    else:
        write(proto, value)
    entries = training_entries()
    status("benchmark_loading", records=len(entries))
    started = time.perf_counter()
    records = prior.base.load_records(entries)
    loading = time.perf_counter() - started
    with train_adapter("cell_peak"):
        prior.OUT = target
        prior.status = lambda phase, **kw: status("bench_" + phase, **kw)
        prior.train_one("cell_peak_bench", records, SEEDS[0], 250, "cpu")
    receipt = read(target / "models/cell_peak_bench_s1.json")
    write(target / "completion.json", dict(complete=True, protocol_sha256=sha(proto),
                                           load_seconds=loading, train_seconds=receipt["history"][-1]["elapsed_seconds"],
                                           full_union_records=len(records),
                                           checkpoint_sha256=receipt["checkpoint_sha256"]))
    status("benchmark_complete", load_seconds=round(loading, 1),
           train_seconds=round(receipt["history"][-1]["elapsed_seconds"], 1))


def train() -> None:
    gate(); low_priority(); torch.set_num_threads(2)
    require(not torch.cuda.is_available(), "Hide CUDA while MVP uses both GPUs")
    entries = training_entries()
    status("loading_train", queries=len(entries))
    records = prior.base.load_records(entries)
    require(len(records) == 13373 and sum(r["n"] > 0 for r in records) == 12886,
            "Full training cache count drift")
    refs = [read(V18 / f"models/full_s{s}.json") for s in (1, 2, 3)]
    for variant in VARIANTS:
        for i, seed in enumerate(SEEDS, 1):
            with train_adapter(variant):
                prior.train_one(variant, records, seed, STEPS, "cpu")
            receipt = read(OUT / f"models/{variant}_s{i}.json")
            require(receipt["steps"] == STEPS and receipt["fitting_queries"] == 12886 and
                    receipt["empty_skipped"] == 487 and receipt["parameter_count"] == 22050 and
                    receipt["initial_state_sha256"] == refs[i-1]["initial_state_sha256"] and
                    receipt["protocol_sha256"] == sha(REG) and
                    receipt["checkpoint_sha256"] == sha(receipt["checkpoint_path"]),
                    "Frozen arm/seed/schedule mismatch")
            status("model_complete", variant=variant, seed_index=i,
                   models_completed=VARIANTS.index(variant)*3+i, models_total=len(VARIANTS)*3)
    files = {f"{variant}_s{i}": sha(OUT / f"models/{variant}_s{i}.json")
             for variant in VARIANTS for i in (1, 2, 3)}
    write(OUT / "models/completion.json", dict(complete=True, models=files,
                                               protocol_sha256=sha(REG), count=len(files)))
    status("models_frozen", models=len(files))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=("selftest", "register", "bench250", "train"))
    args = parser.parse_args()
    if args.stage == "selftest": selftest()
    elif args.stage == "register": register()
    elif args.stage == "bench250": bench250()
    else: train()


if __name__ == "__main__":
    main()
