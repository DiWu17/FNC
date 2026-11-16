"""Write read-only execution handoff receipts after both processes exit.

The caller must first observe MVP GPU process exit and safely interrupt the
CPU V1 trainer after a persisted resume checkpoint.  This script never stops
either process or reads test annotations.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import torch
from clickagent.evidence_retention import RetentionModel
import run_quality_readout_v15_3k as prior
import run_review_attention_finegrained_v1 as fine
import run_review_mvp_resolution_v1 as mvp_runner

OUT = ROOT / "runs/review_attention_finegrained_gpu_v2"
MVP = ROOT / "runs/review_mvp_resolution_v1"
V1 = ROOT / "runs/review_attention_finegrained_v1"


def sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_once(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        old = read(path)
        old.pop("recorded_at_utc", None)
        copy = dict(value)
        copy.pop("recorded_at_utc", None)
        if old != copy:
            raise ValueError(f"Existing handoff receipt drift: {path}")
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8")
    os.replace(temp, path)


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def entry_identity() -> str:
    """Match the exact record identity used by the original trainer."""
    entries = fine.training_entries()
    keys = ("sample_id", "sha256", "state_sha256", "labels_sha256")
    return prior.digest([{key: entry[key] for key in keys} for entry in entries])


def model_state_valid(state: object) -> bool:
    if not isinstance(state, dict):
        return False
    expected = RetentionModel().state_dict()
    return (set(state) == set(expected) and
            all(isinstance(state[name], torch.Tensor) and
                state[name].shape == parameter.shape and
                state[name].dtype == parameter.dtype and
                bool(torch.isfinite(state[name]).all())
                for name, parameter in expected.items()))


def validate_mvp_public(public: dict, hashes: dict[str, str],
                        protocol_sha: str, highpixel_sha: str) -> None:
    expected = dict(complete=True, n=500, methods=["fixed", "adaptive"],
                    prediction_sha256=hashes,
                    protocol_sha256=protocol_sha,
                    highpixel_completion_sha256=highpixel_sha,
                    labels_opened=False)
    require(public == expected,
            "MVP public completion schema, method SHA, or high-pixel seal drift")


def mvp_public_gate() -> dict[str, str]:
    """Replay the frozen runner's read-only gates; never call verify_gate()."""
    mvp_runner.highpixel_sealed()
    hashes = {method: mvp_runner.verify_method(method)
              for method in mvp_runner.METHODS}
    public = read(MVP / "public_completion.json")
    validate_mvp_public(public, hashes, sha(mvp_runner.PROTOCOL),
                        sha(mvp_runner.HPR / "public_completion.json"))
    return hashes


def validate_artifact(artifact: Path, models: Path, status: dict,
                      protocol: dict, identity: str) -> dict:
    """Accept only the active V1 resume or just-completed final model."""
    artifact = artifact.resolve()
    models = models.resolve()
    require(artifact.is_relative_to(models) and artifact.parent == models and
            artifact.is_file(), "Artifact must be a direct V1 models/ file")
    variants = tuple(protocol["variants"])
    seeds = tuple(protocol["training"]["seeds"])
    steps = int(protocol["training"]["steps"])
    require(steps == 13376 and len(seeds) == 3 and len(variants) == 6,
            "Frozen V1 variant/schedule drift")
    match = re.fullmatch(r"([a-z0-9_]+)_s([123])(\.resume\.pt|\.json)", artifact.name)
    require(match is not None, "Artifact must be a legal variant/seed resume or receipt")
    variant, seed_index, extension = match.group(1), int(match.group(2)), match.group(3)
    require(variant in variants, "Artifact variant is not frozen")
    tag = f"{variant}_s{seed_index}"
    seed = seeds[seed_index - 1]
    protocol_sha = sha(V1 / "protocol.json")

    if extension == ".resume.pt":
        require(status.get("phase") == "training" and status.get("model") == tag and
                status.get("total_steps") == steps,
                "Resume checkpoint is not the active V1 model")
        payload = torch.load(artifact, map_location="cpu", weights_only=True)
        require(isinstance(payload, dict) and
                payload.get("protocol_sha256") == protocol_sha and
                payload.get("entry_identity") == identity,
                "Resume protocol or training-entry identity drift")
        step = payload.get("step")
        require(type(step) is int and 0 < step < steps and step % 250 == 0 and
                status.get("step") == step,
                "Resume step must be the active 250-update boundary")
        epoch, position = payload.get("epoch"), payload.get("position")
        require(type(epoch) is int and epoch >= 1 and type(position) is int and
                0 <= position <= protocol["fitting_queries"],
                "Invalid resume epoch or query cursor")
        require(model_state_valid(payload.get("state_dict")),
                "Missing or invalid resume model state")
        optimizer = payload.get("optimizer")
        require(isinstance(optimizer, dict) and
                isinstance(optimizer.get("state"), dict) and optimizer["state"] and
                isinstance(optimizer.get("param_groups"), list) and
                len(optimizer["param_groups"]) == 1,
                "Missing AdamW optimizer state")
        group = optimizer["param_groups"][0]
        require(group.get("lr") == 0.0002 and group.get("weight_decay") == 0.02 and
                tuple(group.get("betas", ())) == (0.9, 0.999) and
                group.get("eps") == 1e-8,
                "Resume optimizer hyperparameters drift")
        rng = payload.get("torch_rng")
        require(isinstance(rng, torch.Tensor) and rng.dtype == torch.uint8 and
                rng.ndim == 1 and rng.numel() > 0 and payload.get("cuda_rng") == [],
                "Missing CPU RNG or unexpected CUDA RNG")
        # Validate that the original trainer can actually reload both states.
        reload_model = RetentionModel().cpu()
        reload_model.load_state_dict(payload["state_dict"])
        reload_optimizer = torch.optim.AdamW(reload_model.parameters(),
                                             lr=0.0002, weight_decay=0.02)
        reload_optimizer.load_state_dict(optimizer)
        torch.Generator(device="cpu").set_state(rng)
        history = payload.get("history")
        require(isinstance(history, list) and len(history) == step // 250 and
                all(type(h.get("step")) is int and h["step"] == 250 * (i + 1)
                    for i, h in enumerate(history)) and
                history[-1]["step"] == step,
                "Resume history does not match step")
        require(type(payload.get("elapsed_seconds")) in (float, int) and
                payload["elapsed_seconds"] >= 0,
                "Invalid elapsed training time")
        return dict(kind="resume", name=tag, variant=variant, seed=seed,
                    step=step, epoch=epoch, position=position,
                    entry_identity_sha256=identity)

    require(status.get("phase") == "model_complete" and
            status.get("variant") == variant and
            status.get("seed_index") == seed_index,
            "Completed receipt is not the latest V1 model")
    receipt = read(artifact)
    require(receipt.get("complete") is True and receipt.get("name") == tag and
            receipt.get("seed") == seed and receipt.get("steps") == steps and
            receipt.get("queries") == protocol["training_queries"] and
            receipt.get("fitting_queries") == protocol["fitting_queries"] and
            receipt.get("empty_skipped") == protocol["skipped_empty_queries"] and
            receipt.get("parameter_count") == 22050 and
            receipt.get("protocol_sha256") == protocol_sha,
            "Completed-model receipt does not match frozen V1 training")
    checkpoint = Path(receipt.get("checkpoint_path", "")).resolve()
    require(checkpoint == models / f"{tag}.pt" and checkpoint.is_file() and
            receipt.get("checkpoint_sha256") == sha(checkpoint),
            "Completed-model checkpoint SHA or path drift")
    reference = read(ROOT / f"runs/full_scaling_v18/models/full_s{seed_index}.json")
    require(receipt.get("initial_state_sha256") == reference["initial_state_sha256"],
            "Completed-model seed initialization drift")
    history = receipt.get("history")
    require(isinstance(history, list) and history and history[-1].get("step") == steps,
            "Completed model lacks final-step history")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    require(isinstance(saved, dict) and model_state_valid(saved.get("state_dict")) and
            saved.get("variant") == "aligned" and saved.get("seed") == seed and
            isinstance(saved.get("mean"), torch.Tensor) and
            isinstance(saved.get("std"), torch.Tensor) and
            tuple(saved["mean"].shape) == (32,) and
            tuple(saved["std"].shape) == (32,),
            "Completed-model payload is invalid")
    return dict(kind="completed_model", name=tag, variant=variant, seed=seed,
                step=steps, checkpoint_sha256=sha(checkpoint))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mvp-exited-observed", action="store_true", required=True)
    parser.add_argument("--cpu-exited-observed", action="store_true", required=True)
    parser.add_argument("--resume-artifact", required=True)
    args = parser.parse_args()
    artifact = Path(args.resume_artifact).resolve()
    models = (V1 / "models").resolve()
    if not (V1 / "status.json").exists():
        raise ValueError("CPU V1 status missing")
    protocol = fine.gate()
    status = read(V1 / "status.json")
    validated = validate_artifact(artifact, models, status, protocol,
                                  entry_identity())
    mvp_public_gate()
    names = ("public_completion.json", "predictions/fixed.json",
             "predictions/adaptive.json")
    common = dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
                  receipt_writer_sha256=sha(Path(__file__)))
    release = dict(common, mvp_process_exited_observed=True,
                   no_active_mvp_gpu_process_observed=True,
                   **{name: sha(MVP / name) for name in names})
    parked = dict(common, cpu_v1_process_exited_observed=True,
                  cpu_v1_exact_resume_available=True,
                  test_labels_opened=False,
                  v1_protocol_sha256=sha(V1 / "protocol.json"),
                  v1_status_sha256=sha(V1 / "status.json"),
                  resumable_artifact_path=str(artifact),
                  resumable_artifact_sha256=sha(artifact),
                  validated_artifact=validated,
                  recovery="If GPU benchmark is not chosen or fails, resume CPU V1 with the frozen script and parameters")
    write_once(OUT / "gpu_release.json", release)
    write_once(OUT / "cpu_v1_parked.json", parked)
    print(json.dumps({"gpu_release_sha256": sha(OUT / "gpu_release.json"),
                      "cpu_v1_parked_sha256": sha(OUT / "cpu_v1_parked.json")}),
          flush=True)


if __name__ == "__main__":
    main()
