"""Isolated GPU execution of the frozen cached attention ablation.

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

OUT = ROOT / "runs/review_attention_finegrained_gpu_v2"
REG = OUT / "protocol.json"
PARENT = ROOT / "runs/review_attention_finegrained_v1/protocol.json"
CPU_BENCH = ROOT / "runs/review_attention_finegrained_v1/bench250/completion.json"
MVP = ROOT / "runs/review_mvp_resolution_v1"
RELEASE = OUT / "gpu_release.json"
PARKED = OUT / "cpu_v1_parked.json"
CHOICE = OUT / "execution_choice.json"
DEVICE = "cuda:0"
BENCH_STEPS = 250
MIN_SPEEDUP = 1.25


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
    require(parent["variants"] == list(fine.VARIANTS), "Parent arm drift")
    require(parent["training"]["steps"] == fine.STEPS and
            parent["training"]["seeds"] == list(fine.SEEDS),
            "Parent training schedule drift")
    require(cpu_bench["complete"] and cpu_bench["train_seconds"] > 0,
            "CPU benchmark incomplete")
    return dict(
        registered_at_utc=now(), purpose="Training-only GPU throughput test and optional isolated full six-arm execution",
        parent_protocol_sha256=sha(PARENT),
        parent_worker_sha256=sha(Path(fine.__file__)),
        original_trainer_sha256=sha(Path(prior.__file__)),
        gpu_worker_sha256=sha(__file__),
        source_sha256=fine.source_hashes(),
        cpu_benchmark_completion_sha256=sha(CPU_BENCH),
        cpu_benchmark_train_seconds=cpu_bench["train_seconds"],
        variants=list(fine.VARIANTS), seeds=list(fine.SEEDS),
        training_queries=parent["training_queries"],
        fitting_queries=parent["fitting_queries"],
        train_query_order_sha256=parent["train_query_order_sha256"],
        training=parent["training"],
        channel_transform="Byte-identical frozen V1 transform at train and test",
        device=DEVICE, expected_gpu_name="Quadro P5000",
        host_torch_threads=2, cublas_workspace_config=":4096:8",
        serial_models=True,
        benchmark=dict(variant="cell_peak", seed=fine.SEEDS[0], steps=BENCH_STEPS,
                       full_training_union=True, min_speedup_for_choice=MIN_SPEEDUP,
                       decision_uses="training-only elapsed seconds, never test labels or scores"),
        execution_order=[
            "MVP fixed and adaptive predictions sealed; observe MVP GPU process exit",
            "Observe a valid CPU V1 resume checkpoint or completed-model receipt, park CPU V1 process and SHA-seal it",
            "Write gpu_release.json and cpu_v1_parked.json before GPU load",
            "Single GPU V2 process loads full training union once, benchmarks 250 steps in bench250/",
            "If benchmark speedup is below threshold or GPU fails, resume untouched CPU V1",
            "If threshold passes, GPU V2 trains all 18 arms from initialization to fixed final step using the same loaded records",
            "Seal all model checkpoints; only then launch a separately gated label-free prediction stage",
        ],
        train_if_chosen="All 18 arms trained from initialization to fixed final step on one GPU; no CPU/GPU checkpoint mixing",
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
    status("registered", protocol_sha256=sha(REG), variants=len(fine.VARIANTS),
           seeds=len(fine.SEEDS), gpu_touched=False)


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
    """An independent agent must first observe that MVP has exited."""
    require(RELEASE.exists(), "Missing explicit MVP GPU release receipt")
    r = read(RELEASE)
    require(r.get("mvp_process_exited_observed") is True and
            r.get("no_active_mvp_gpu_process_observed") is True,
            "MVP GPU release has not been observed")
    for name in ("public_completion.json", "predictions/fixed.json",
                 "predictions/adaptive.json"):
        path = MVP / name
        require(path.exists() and r.get(name) == sha(path),
                "MVP prediction completion changed: " + name)


def parked_gate() -> None:
    require(PARKED.exists(), "Missing CPU V1 safe-stop receipt")
    r = read(PARKED)
    require(r.get("cpu_v1_process_exited_observed") is True and
            r.get("cpu_v1_exact_resume_available") is True and
            r.get("test_labels_opened") is False,
            "CPU V1 has not been safely parked")
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
    value = dict(parent_protocol_sha256=sha(REG), gpu_release_sha256=sha(RELEASE),
                 worker_sha256=sha(__file__), variant="cell_peak",
                 seed=fine.SEEDS[0], steps=BENCH_STEPS, device=DEVICE,
                 full_training_union=True, no_test_labels=True)
    if subprotocol.exists():
        require(read(subprotocol) == value, "GPU benchmark protocol drift")
    else:
        write(subprotocol, value)
    completion = target / "completion.json"
    if completion.exists():
        cached = read(completion)
        receipt = read(target / "models/cell_peak_bench_s1.json")
        require(cached["complete"] is True and
                cached["protocol_sha256"] == sha(subprotocol) and
                cached["checkpoint_sha256"] == sha(receipt["checkpoint_path"]) and
                cached["no_test_labels_opened"] is True,
                "Existing GPU benchmark drift")
        status("benchmark_already_complete", speedup=round(cached["speedup"], 3),
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
    with fine.train_adapter("cell_peak"):
        prior.OUT = target
        prior.status = lambda phase, **kw: status("bench_" + phase, **kw)
        prior.train_one("cell_peak_bench", records, fine.SEEDS[0], BENCH_STEPS,
                        DEVICE)
    receipt = read(target / "models/cell_peak_bench_s1.json")
    require(receipt["steps"] == BENCH_STEPS and
            receipt["checkpoint_sha256"] == sha(receipt["checkpoint_path"]) and
            receipt["protocol_sha256"] == sha(subprotocol),
            "GPU benchmark checkpoint drift")
    elapsed = receipt["history"][-1]["elapsed_seconds"]
    cpu_elapsed = read(CPU_BENCH)["train_seconds"]
    result = dict(complete=True, at_utc=now(), protocol_sha256=sha(subprotocol),
                  gpu_release_sha256=sha(RELEASE), checkpoint_sha256=receipt["checkpoint_sha256"],
                  training_queries=len(records), fitting_queries=12886,
                  load_seconds=loading, train_seconds=elapsed,
                  cpu_reference_train_seconds=cpu_elapsed,
                  speedup=cpu_elapsed / elapsed,
                  estimated_18_model_gpu_train_hours=18 * fine.STEPS * elapsed / BENCH_STEPS / 3600,
                  no_test_labels_opened=True)
    write(completion, result)
    status("benchmark_complete", speedup=round(result["speedup"], 3),
           estimated_train_hours=round(result["estimated_18_model_gpu_train_hours"], 2),
           labels_opened=False)


def choose_gpu() -> bool:
    gate()
    completion = OUT / "bench250/completion.json"
    require(completion.exists() and read(completion)["complete"],
            "Benchmark must finish before device decision")
    bench = read(completion)
    require(bench["protocol_sha256"] == sha(OUT / "bench250/protocol.json") and
            bench["no_test_labels_opened"] is True,
            "Benchmark identity drift")
    if bench["speedup"] < MIN_SPEEDUP:
        status("gpu_not_chosen_resume_cpu_v1",
               measured_speedup=round(bench["speedup"], 3),
               preregistered_min_speedup=MIN_SPEEDUP, labels_opened=False)
        return False
    value = dict(selected_execution="gpu_v2_all18",
                 decision_basis="training-only runtime; no test labels or accuracies",
                 preregistered_min_speedup=MIN_SPEEDUP,
                 measured_speedup=bench["speedup"],
                 gpu_benchmark_completion_sha256=sha(completion),
                 cpu_benchmark_completion_sha256=sha(CPU_BENCH),
                 selected_at_utc=now(), test_labels_opened=False)
    if CHOICE.exists():
        old = read(CHOICE)
        value["selected_at_utc"] = old["selected_at_utc"]
        require(old == value, "GPU execution choice drift")
    else:
        write(CHOICE, value)
    status("gpu_chosen", measured_speedup=round(bench["speedup"], 3),
           labels_opened=False)
    return True


def train(records=None, loading: float | None = None) -> None:
    gate()
    gpu_name = gpu_gate()
    require(CHOICE.exists() and read(CHOICE)["selected_execution"] == "gpu_v2_all18",
            "Complete benchmark-based GPU choice before full training")
    require(read(CHOICE)["gpu_benchmark_completion_sha256"] ==
            sha(OUT / "bench250/completion.json"), "GPU choice/benchmark drift")
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
    refs = [read(fine.V18 / f"models/full_s{s}.json") for s in (1, 2, 3)]
    for variant in fine.VARIANTS:
        for i, seed in enumerate(fine.SEEDS, 1):
            with fine.train_adapter(variant):
                prior.train_one(variant, records, seed, fine.STEPS, DEVICE)
            receipt = read(OUT / f"models/{variant}_s{i}.json")
            require(receipt["steps"] == fine.STEPS and
                    receipt["seed"] == seed and
                    receipt["fitting_queries"] == 12886 and
                    receipt["empty_skipped"] == 487 and
                    receipt["parameter_count"] == 22050 and
                    receipt["initial_state_sha256"] == refs[i-1]["initial_state_sha256"] and
                    receipt["protocol_sha256"] == sha(REG) and
                    receipt["checkpoint_sha256"] == sha(receipt["checkpoint_path"]),
                    "Frozen GPU model/arm/schedule drift")
            status("model_complete", variant=variant, seed_index=i,
                   models_completed=fine.VARIANTS.index(variant)*3+i,
                   models_total=len(fine.VARIANTS)*3, labels_opened=False)
    files = {f"{variant}_s{i}": sha(OUT / f"models/{variant}_s{i}.json")
             for variant in fine.VARIANTS for i in (1, 2, 3)}
    write(OUT / "models/completion.json",
          dict(complete=True, at_utc=now(), protocol_sha256=sha(REG),
               gpu_choice_sha256=sha(CHOICE), records_loaded_once=True,
               loading_seconds=loading, count=len(files), models=files,
               no_test_labels_opened=True))
    status("models_frozen", models=len(files), labels_opened=False)


def auto() -> None:
    """One cache load for benchmark and, if faster, all 18 final-step fits."""
    gate()
    gpu_name = gpu_gate()
    torch.set_num_threads(2)
    fine.OUT = OUT
    status("loading_auto", queries=13373, gpu=gpu_name, labels_opened=False)
    records, loading = load_training()
    bench250(records, loading)
    if choose_gpu():
        train(records, loading)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True,
                        choices=("register", "bench250", "choose-gpu", "train", "auto"))
    args = parser.parse_args()
    if args.stage == "register":
        register()
    elif args.stage == "bench250":
        bench250()
    elif args.stage == "choose-gpu":
        choose_gpu()
    elif args.stage == "auto":
        auto()
    else:
        train()


if __name__ == "__main__":
    main()
