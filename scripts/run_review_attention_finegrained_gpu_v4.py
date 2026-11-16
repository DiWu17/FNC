"""Five remaining S1 cached attention ablations, isolated from sealed V3.

This runner never calls the VLM or proposer and never opens test annotations.
It reuses the frozen V1 channel transforms and original full-data trainer.
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

os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import numpy as np
import torch
from clickagent.evidence_retention import RetentionModel, retention_loss
import run_quality_readout_v15_3k as prior
import run_review_attention_finegrained_v1 as fine
import run_review_attention_finegrained_gpu_v3 as v3

OUT = ROOT / "runs/review_attention_finegrained_gpu_v4"
REG = OUT / "protocol.json"
PARENT = ROOT / "runs/review_attention_finegrained_v1/protocol.json"
V3 = ROOT / "runs/review_attention_finegrained_gpu_v3"
SEED = 20261311
STEPS = 13376
DEVICE = "cuda:0"
VARIANTS = ("head_mean_per_layer", "cell_only", "peak_only",
            "neighborhood_only", "cell_peak")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def sha(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read(path: str | Path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                   encoding="utf-8")
    os.replace(tmp, path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def status(phase: str, **fields) -> None:
    record = dict(phase=phase, at_utc=now(), pid=os.getpid(), **fields)
    write(OUT / "status.json", record)
    print(json.dumps(record, ensure_ascii=False), flush=True)


def v3_reference() -> dict:
    v3.gate()
    frozen = read(V3 / "models/completion.json")
    receipt_path = V3 / "models/head_mean_all96_s1.json"
    receipt = read(receipt_path)
    require(frozen.get("complete") is True and frozen.get("count") == 1 and
            frozen.get("protocol_sha256") == sha(V3 / "protocol.json") and
            frozen["models"] == {"head_mean_all96_s1": sha(receipt_path)} and
            receipt.get("complete") is True and receipt.get("steps") == STEPS and
            receipt.get("seed") == SEED and
            receipt.get("checkpoint_sha256") == sha(receipt["checkpoint_path"]),
            "Frozen V3 S1 reference drift")
    return dict(protocol_sha256=sha(V3 / "protocol.json"),
                worker_sha256=sha(Path(v3.__file__)),
                model_completion_sha256=sha(V3 / "models/completion.json"),
                model_receipt_sha256=sha(receipt_path),
                checkpoint_sha256=receipt["checkpoint_sha256"])


def registered_value() -> dict:
    parent = read(PARENT)
    require(tuple(parent["variants"]) == ("head_mean_all96", *VARIANTS) and
            parent["training"]["seeds"] == [20261311, 20261312, 20261313] and
            parent["training"]["steps"] == STEPS,
            "Parent attention protocol variant/schedule drift")
    return dict(
        registered_at_utc=now(),
        purpose="Five prespecified remaining fine-grained attention variants, S1 only",
        parent_protocol_sha256=sha(PARENT),
        parent_worker_sha256=sha(Path(fine.__file__)),
        original_trainer_sha256=sha(Path(prior.__file__)),
        worker_sha256=sha(__file__),
        source_sha256=fine.source_hashes(),
        v3_reference=v3_reference(),
        mvp_pause_receipt_sha256=sha(V3 / "mvp_pause_receipt.json"),
        cpu_parking_receipt_sha256=sha(V3 / "cpu_v1_parked.json"),
        variants=list(VARIANTS), model_order=[f"{v}_s1" for v in VARIANTS],
        seed=SEED, training_queries=13373, fitting_queries=12886,
        skipped_empty_queries=487,
        train_query_order_sha256=parent["train_query_order_sha256"],
        training=dict(parent["training"], seeds=[SEED]),
        channel_transform="Frozen V1 attention-only channel mask, applied identically at train and test",
        model="Unchanged RetentionModel 22050 parameters; same training-only base32 scaler",
        device=DEVICE, expected_gpu_name="Quadro P5000", host_torch_threads=2,
        cublas_workspace_config=":4096:8", serial_model_order=True,
        checkpoint="Each fixed final-step model and receipt SHA-sealed before next arm; no test selection",
        evaluation=parent["evaluation"],
        planned_scoring="Separate pre-prediction registration; five models x 2000 cached original-K16 label-free predictions sealed before any target geometry is read",
        no_backbone_or_proposer_call=True, test_labels_opened_by_runner=False,
        retrospective=True, strict_blind=False,
    )


def register() -> None:
    record = registered_value()
    if REG.exists():
        old = read(REG)
        record["registered_at_utc"] = old["registered_at_utc"]
        require(old == record, "Frozen V4 protocol drift")
    else:
        write(REG, record)
    status("registered", protocol_sha256=sha(REG), models=len(VARIANTS),
           gpu_touched=False, labels_opened=False)


def gate() -> dict:
    require(REG.exists(), "Register V4 before training")
    old = read(REG)
    expected = registered_value()
    expected["registered_at_utc"] = old["registered_at_utc"]
    require(old == expected, "V4 worker/source/protocol drift")
    require(read(PARENT)["worker_sha256"] == sha(Path(fine.__file__)),
            "Frozen V1 worker drift")
    v3.release_gate()
    v3.parked_gate()
    return old


def training_records():
    entries = fine.training_entries()
    require(len(entries) == 13373 and sum(e["n"] > 0 for e in entries) == 12886,
            "Frozen full-union index drift")
    start = time.perf_counter()
    records = prior.base.load_records(entries)
    elapsed = time.perf_counter() - start
    require(len(records) == 13373 and sum(r["n"] > 0 for r in records) == 12886,
            "Loaded full-union records drift")
    return records, elapsed


def selftest() -> None:
    """CPU-only transform/loss smoke test on four training records."""
    registered_value()  # Checks all frozen source identities, no CUDA calls.
    torch.set_num_threads(2)
    entries = [entry for entry in fine.training_entries() if entry["n"] > 0][:4]
    require(len(entries) == 4, "Insufficient nonempty training smoke cases")
    records = prior.base.load_records(entries)
    base = np.concatenate([r["base"] for r in records])
    mean, std = base.mean(0), base.std(0)
    std[std < 1e-5] = 1
    args, lengths = fine.ORIGINAL_TENSORS(records[:2], mean, std,
                                          "aligned", "cpu")
    for variant in VARIANTS:
        changed = fine.transformed(args, variant)
        require(torch.equal(changed["base"], args["base"]) and
                torch.equal(changed["mask"], args["mask"]) and
                torch.equal(changed["active"], args["active"]) and
                bool((changed["state"][..., 288:] == 0).all()),
                "Non-evidence inputs or hidden6 changed: " + variant)
        model = RetentionModel().cpu()
        scores = model(**changed)
        labels = torch.from_numpy(np.concatenate([r["labels"] for r in records[:2]])).float()
        loss = torch.stack([retention_loss(s, y, model.null)
                            for s, y in zip(scores.split(lengths),
                                            labels.split(lengths))]).mean()
        require(bool(torch.isfinite(loss)), "Nonfinite smoke loss: " + variant)
        loss.backward()
    write(OUT / "smoke.json",
          dict(complete=True, at_utc=now(), code_sha256=sha(__file__),
               parent_protocol_sha256=sha(PARENT),
               training_records=len(records), variants=list(VARIANTS),
               no_gpu_calls=True, test_labels_opened=False))
    print(json.dumps({"stage": "selftest_pass", "variants": len(VARIANTS),
                      "training_records": len(records), "gpu_touched": False}),
          flush=True)


def train() -> None:
    gate()  # Before CUDA context or any model construction.
    require(torch.cuda.is_available() and torch.cuda.device_count() > 0,
            "CUDA unavailable")
    require(torch.cuda.get_device_name(0) == "Quadro P5000",
            "Unexpected CUDA device 0")
    torch.set_num_threads(2)
    fine.OUT = OUT  # This independent process cannot mutate frozen V1/V3.
    status("loading_train", queries=13373, fitting_queries=12886,
           models_total=len(VARIANTS), labels_opened=False)
    records, loading_seconds = training_records()  # Exactly once per process.
    reference = read(fine.V18 / "models/full_s1.json")
    receipts = {}
    for index, variant in enumerate(VARIANTS, 1):
        with fine.train_adapter(variant):
            prior.train_one(variant, records, SEED, STEPS, DEVICE)
        path = OUT / f"models/{variant}_s1.json"
        receipt = read(path)
        require(receipt.get("complete") is True and
                receipt.get("name") == f"{variant}_s1" and
                receipt.get("seed") == SEED and receipt.get("steps") == STEPS and
                receipt.get("queries") == 13373 and
                receipt.get("fitting_queries") == 12886 and
                receipt.get("empty_skipped") == 487 and
                receipt.get("parameter_count") == 22050 and
                receipt.get("initial_state_sha256") ==
                reference["initial_state_sha256"] and
                receipt.get("protocol_sha256") == sha(REG) and
                receipt.get("checkpoint_sha256") == sha(receipt["checkpoint_path"]),
                "V4 fixed-final checkpoint drift: " + variant)
        receipts[f"{variant}_s1"] = sha(path)
        status("model_complete", variant=variant, seed_index=1,
               models_completed=index, models_total=len(VARIANTS),
               checkpoint_sha256=receipt["checkpoint_sha256"],
               labels_opened=False)
    completion = dict(complete=True, protocol_sha256=sha(REG), count=len(VARIANTS),
                      models=receipts, records_loaded_once=True,
                      labels_opened=False)
    path = OUT / "models/completion.json"
    if path.exists():
        require(read(path) == completion, "V4 model completion drift")
    else:
        write(path, completion)
    status("models_frozen", models=len(VARIANTS),
           model_completion_sha256=sha(path), loading_seconds=round(loading_seconds, 1),
           labels_opened=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=("selftest", "register", "train"))
    args = parser.parse_args()
    if args.stage == "selftest":
        selftest()
    elif args.stage == "register":
        register()
    else:
        train()


if __name__ == "__main__":
    main()
