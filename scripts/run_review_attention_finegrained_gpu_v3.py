"""One-arm GPU diagnostic after user-paused MVP adaptive inference.

This runner imports the V1 channel transform and the original trainer.  It
changes only the execution device and output directory.  It never reads test
annotations, regenerates candidates, or invokes the VLM.
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

import torch
import run_quality_readout_v15_3k as prior
import run_review_attention_finegrained_v1 as fine

OUT = ROOT / "runs/review_attention_finegrained_gpu_v3"
REG = OUT / "protocol.json"
PARENT = ROOT / "runs/review_attention_finegrained_v1/protocol.json"
CPU_BENCH = ROOT / "runs/review_attention_finegrained_v1/bench250/completion.json"
MVP = ROOT / "runs/review_mvp_resolution_v1"
V2 = ROOT / "runs/review_attention_finegrained_gpu_v2"
RELEASE = OUT / "mvp_pause_receipt.json"
PARKED = OUT / "cpu_v1_parked.json"
DEVICE = "cuda:0"
ARM = "head_mean_all96"
SEED = 20261311
BENCH_STEPS = 250


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
    value = dict(phase=phase, at_utc=now(), pid=os.getpid(), **fields)
    write(OUT / "status.json", value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def registered_value() -> dict:
    parent = read(PARENT)
    cpu_bench = read(CPU_BENCH)
    require(ARM in parent["variants"] and SEED in parent["training"]["seeds"],
            "Parent arm/seed drift")
    require(parent["training"]["steps"] == fine.STEPS and
            parent["training"]["seeds"] == list(fine.SEEDS),
            "Parent training schedule drift")
    require(cpu_bench["complete"] and cpu_bench["train_seconds"] > 0,
            "CPU benchmark incomplete")
    return dict(
        registered_at_utc=now(), purpose="One-seed head-averaged attention diagnostic after resumable MVP pause",
        supersedes_v2_protocol_sha256=sha(V2 / "protocol.json"),
        parent_protocol_sha256=sha(PARENT),
        parent_worker_sha256=sha(Path(fine.__file__)),
        original_trainer_sha256=sha(Path(prior.__file__)),
        gpu_worker_sha256=sha(__file__),
        release_helper_sha256=sha(ROOT / "scripts/record_review_attention_gpu_release_v3.py"),
        mvp_runner_sha256=sha(ROOT / "scripts/run_review_mvp_resolution_v1.py"),
        source_sha256=fine.source_hashes(),
        cpu_benchmark_completion_sha256=sha(CPU_BENCH),
        cpu_benchmark_train_seconds=cpu_bench["train_seconds"],
        variants=[ARM], seeds=[SEED],
        training_queries=parent["training_queries"],
        fitting_queries=parent["fitting_queries"],
        train_query_order_sha256=parent["train_query_order_sha256"],
        training=dict(parent["training"], seeds=[SEED]),
        channel_transform="Byte-identical frozen V1 transform at train and test",
        device=DEVICE, expected_gpu_name="Quadro P5000",
        host_torch_threads=2, cublas_workspace_config=":4096:8",
        serial_models=True,
        benchmark=dict(variant=ARM, seed=SEED, steps=BENCH_STEPS,
                       full_training_union=True, purpose="time measurement only; no model/arm/device selection"),
        execution_order=[
            "MVP fixed 500 predictions sealed; adaptive partial per-case cache validated and SHA-sealed; observe MVP PTY process exit and GPU release",
            "Observe a valid CPU V1 250-step resume checkpoint and process exit; verify original parking_receipt.json",
            "Write V3 mvp_pause_receipt.json and cpu_v1_parked.json before GPU load",
            "Single GPU V3 process loads full training union once and times a separate 250-step same-arm benchmark",
            "Regardless of benchmark timing, train head_mean_all96 seed20261311 from initialization to fixed step13376",
            "Seal the one checkpoint; then independently register label-free fixed-K16 scoring",
        ],
        training_scope="One arm and one seed only; no CPU/GPU checkpoint mixing or model selection",
        evaluation=parent["evaluation"],
        test_labels_opened_by_this_runner=False,
        no_backbone_or_proposer_call=True,
        retrospective=True, strict_blind=False,
    )


def register() -> None:
    value = registered_value()
    if REG.exists():
        old = read(REG)
        value["registered_at_utc"] = old["registered_at_utc"]
        require(old == value, "Frozen GPU execution protocol drift")
    else:
        write(REG, value)
    status("registered", protocol_sha256=sha(REG), variants=1,
           seeds=1, gpu_touched=False)


def gate() -> dict:
    require(REG.exists(), "Register GPU execution before benchmark")
    value = read(REG)
    expected = registered_value()
    expected["registered_at_utc"] = value["registered_at_utc"]
    require(value == expected, "Frozen GPU execution protocol/source drift")
    require(read(PARENT)["worker_sha256"] == sha(Path(fine.__file__)),
            "Frozen V1 worker drift")
    return value


def release_gate() -> None:
    """Verify the paused and resumable MVP handoff before touching CUDA."""
    require(RELEASE.exists(), "Missing explicit MVP pause/GPU release receipt")
    r = read(RELEASE)
    require(r.get("mvp_process_exited_observed") is True and
            r.get("no_active_mvp_gpu_process_observed") is True and
            r.get("fixed_complete") is True and
            r.get("adaptive_bundle_absent") is True and
            r.get("labels_opened") is False,
            "MVP GPU release has not been observed")
    paths = {"protocol_sha256": MVP / "protocol.json",
             "status_sha256": MVP / "status.json",
             "fixed_bundle_sha256": MVP / "predictions/fixed.json",
             "mvp_user_pause_receipt_sha256": MVP / "paused_by_user_v1.json",
             "highpixel_completion_sha256": ROOT / "runs/high_pixel_full_inference_v1/public_completion.json"}
    for field, path in paths.items():
        require(path.exists() and r.get(field) == sha(path),
                "MVP pause source changed: " + field)
    require(not (MVP / "predictions/adaptive.json").exists(),
            "Adaptive MVP method was sealed after pause")
    cases = MVP / "predictions/adaptive/cases"
    listed = r.get("adaptive_case_sha256")
    require(isinstance(listed, dict) and len(listed) == r.get("adaptive_cache_count") and
            len(listed) < 500 and
            set(listed) == {p.name for p in cases.glob("*.json")},
            "Adaptive MVP cache inventory drift")
    for name, digest in listed.items():
        require(sha(cases / name) == digest,
                "Adaptive MVP cached case SHA drift: " + name)


def parked_gate() -> None:
    require(PARKED.exists(), "Missing CPU V1 safe-stop receipt")
    r = read(PARKED)
    require(r.get("cpu_v1_process_exited_observed") is True and
            r.get("cpu_v1_exact_resume_available") is True and
            r.get("test_labels_opened") is False,
            "CPU V1 has not been safely parked")
    receipt = ROOT / "runs/review_attention_finegrained_v1/parking_receipt.json"
    require(receipt.exists() and r.get("cpu_parking_receipt_sha256") == sha(receipt),
            "CPU V1 original parking receipt drift")
    artifact = Path(r.get("resumable_artifact_path", ""))
    models = (ROOT / "runs/review_attention_finegrained_v1/models").resolve()
    require(artifact.resolve().is_relative_to(models) and artifact.exists() and
            r.get("resumable_artifact_sha256") == sha(artifact),
            "CPU V1 resume/completed-model artifact drift")


def gpu_gate() -> str:
    release_gate()
    parked_gate()
    require(torch.cuda.is_available() and torch.cuda.device_count() > 0,
            "CUDA is unavailable")
    name = torch.cuda.get_device_name(0)
    require(name == "Quadro P5000", "Unexpected GPU 0: " + name)
    return name


def load_training():
    entries = fine.training_entries()
    require(len(entries) == 13373 and sum(e["n"] > 0 for e in entries) == 12886,
            "Full cached training union drift")
    started = time.perf_counter()
    records = prior.base.load_records(entries)
    seconds = time.perf_counter() - started
    require(len(records) == 13373 and sum(r["n"] > 0 for r in records) == 12886,
            "Loaded training records drift")
    return records, seconds


def bench250(records=None, loading: float | None = None) -> None:
    gate()
    gpu_name = gpu_gate()
    torch.set_num_threads(2)
    fine.OUT = OUT  # In this independent process only; V1 files are untouched.
    target = OUT / "bench250"
    subprotocol = target / "protocol.json"
    value = dict(parent_protocol_sha256=sha(REG), mvp_pause_receipt_sha256=sha(RELEASE),
                 worker_sha256=sha(__file__), variant=ARM,
                 seed=SEED, steps=BENCH_STEPS, device=DEVICE,
                 full_training_union=True, no_test_labels=True)
    if subprotocol.exists():
        require(read(subprotocol) == value, "GPU benchmark protocol drift")
    else:
        write(subprotocol, value)
    completion = target / "completion.json"
    if completion.exists():
        cached = read(completion)
        receipt = read(target / f"models/{ARM}_bench_s1.json")
        require(cached["complete"] is True and
                cached["protocol_sha256"] == sha(subprotocol) and
                cached["checkpoint_sha256"] == sha(receipt["checkpoint_path"]) and
                cached["no_test_labels_opened"] is True,
                "Existing GPU benchmark drift")
        status("benchmark_already_complete",
               estimated_train_hours=round(cached["estimated_one_model_gpu_train_hours"], 2),
               labels_opened=False)
        return
    if records is None:
        status("benchmark_loading", records=13373, gpu=gpu_name, labels_opened=False)
        records, loading = load_training()
    else:
        require(len(records) == 13373 and loading is not None,
                "Shared loaded training union missing")
        status("benchmark_training", records=len(records), gpu=gpu_name,
               labels_opened=False)
    with fine.train_adapter(ARM):
        prior.OUT = target
        prior.status = lambda phase, **kw: status("bench_" + phase, **kw)
        prior.train_one(f"{ARM}_bench", records, SEED, BENCH_STEPS,
                        DEVICE)
    receipt = read(target / f"models/{ARM}_bench_s1.json")
    require(receipt["steps"] == BENCH_STEPS and
            receipt["checkpoint_sha256"] == sha(receipt["checkpoint_path"]) and
            receipt["protocol_sha256"] == sha(subprotocol),
            "GPU benchmark checkpoint drift")
    elapsed = receipt["history"][-1]["elapsed_seconds"]
    cpu_elapsed = read(CPU_BENCH)["train_seconds"]
    result = dict(complete=True, at_utc=now(), protocol_sha256=sha(subprotocol),
                  mvp_pause_receipt_sha256=sha(RELEASE), checkpoint_sha256=receipt["checkpoint_sha256"],
                  training_queries=len(records), fitting_queries=12886,
                  load_seconds=loading, train_seconds=elapsed,
                  cpu_reference_train_seconds=cpu_elapsed,
                  descriptive_cpu_cell_peak_ratio=cpu_elapsed / elapsed,
                  estimated_one_model_gpu_train_hours=fine.STEPS * elapsed / BENCH_STEPS / 3600,
                  no_test_labels_opened=True)
    write(completion, result)
    status("benchmark_complete",
           estimated_train_hours=round(result["estimated_one_model_gpu_train_hours"], 2),
           labels_opened=False)


def train(records=None, loading: float | None = None) -> None:
    gate()
    gpu_name = gpu_gate()
    bench = OUT / "bench250/completion.json"
    require(bench.exists() and read(bench).get("complete") is True and
            read(bench)["protocol_sha256"] == sha(OUT / "bench250/protocol.json"),
            "Complete training-only timing benchmark first")
    torch.set_num_threads(2)
    fine.OUT = OUT
    if records is None:
        status("loading_train", queries=13373, gpu=gpu_name, labels_opened=False)
        records, loading = load_training()  # Standalone resume path.
    else:
        require(len(records) == 13373 and loading is not None,
                "Shared loaded training union missing")
        status("training_after_benchmark", queries=len(records), gpu=gpu_name,
               labels_opened=False)
    reference = read(fine.V18 / "models/full_s1.json")
    with fine.train_adapter(ARM):
        prior.train_one(ARM, records, SEED, fine.STEPS, DEVICE)
    receipt = read(OUT / f"models/{ARM}_s1.json")
    require(receipt["complete"] is True and receipt["name"] == f"{ARM}_s1" and
            receipt["steps"] == fine.STEPS and receipt["seed"] == SEED and
            receipt["fitting_queries"] == 12886 and
            receipt["empty_skipped"] == 487 and
            receipt["parameter_count"] == 22050 and
            receipt["initial_state_sha256"] == reference["initial_state_sha256"] and
            receipt["protocol_sha256"] == sha(REG) and
            receipt["checkpoint_sha256"] == sha(receipt["checkpoint_path"]),
            "Frozen GPU one-arm model/schedule drift")
    status("model_complete", variant=ARM, seed_index=1,
           models_completed=1, models_total=1, labels_opened=False)
    files = {f"{ARM}_s1": sha(OUT / f"models/{ARM}_s1.json")}
    write(OUT / "models/completion.json",
          dict(complete=True, at_utc=now(), protocol_sha256=sha(REG),
               benchmark_completion_sha256=sha(bench), records_loaded_once=True,
               loading_seconds=loading, count=len(files), models=files,
               no_test_labels_opened=True))
    status("models_frozen", models=1, labels_opened=False)


def auto() -> None:
    """One cache load for training-only timing, then the fixed single-arm fit."""
    gate()
    gpu_name = gpu_gate()
    torch.set_num_threads(2)
    fine.OUT = OUT
    status("loading_auto", queries=13373, gpu=gpu_name, labels_opened=False)
    records, loading = load_training()
    bench250(records, loading)
    train(records, loading)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=("register", "bench250", "train", "auto"))
    args = parser.parse_args()
    if args.stage == "register":
        register()
    elif args.stage == "bench250":
        bench250()
    elif args.stage == "auto":
        auto()
    else:
        train()


if __name__ == "__main__":
    main()
