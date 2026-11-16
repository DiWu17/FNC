"""Synthetic, label-free negative tests for the V1 handoff receipt gate."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

import torch
from clickagent.evidence_retention import RetentionModel
import record_review_attention_gpu_release_v2 as helper


def rejects(fn, label: str) -> None:
    try:
        fn()
    except (ValueError, RuntimeError, KeyError, TypeError):
        print("rejected:", label)
    else:
        raise AssertionError("Accepted invalid artifact: " + label)


def main() -> None:
    protocol = helper.read(helper.V1 / "protocol.json")
    tag = "cell_only_s1"
    status = dict(phase="training", model=tag, step=250, total_steps=13376)
    identity = "synthetic_entry_identity"
    model = RetentionModel().cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0002, weight_decay=0.02)
    loss = sum((p.float() ** 2).sum() for p in model.parameters())
    loss.backward()
    optimizer.step()
    base = dict(protocol_sha256=helper.sha(helper.V1 / "protocol.json"),
                entry_identity=identity, step=250, epoch=1, position=2000,
                elapsed_seconds=10.0,
                history=[dict(step=250, epoch=1, position=2000,
                              train_loss=1.0, elapsed_seconds=10.0)],
                state_dict=model.state_dict(), optimizer=optimizer.state_dict(),
                torch_rng=torch.get_rng_state(), cuda_rng=[])
    helper.OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="helper_test_", dir=helper.OUT) as temp:
        directory = Path(temp) / "models"
        directory.mkdir()
        artifact = directory / f"{tag}.resume.pt"

        def check(payload, path=artifact, current=status):
            torch.save(payload, path)
            return helper.validate_artifact(path, directory, current, protocol,
                                            identity)

        valid = check(base)
        assert valid["kind"] == "resume" and valid["step"] == 250
        rejects(lambda: check(dict(base, protocol_sha256="wrong")),
                "wrong protocol SHA")
        rejects(lambda: check(dict(base, entry_identity="wrong")),
                "wrong training entry identity")
        rejects(lambda: check(dict(base, step=251)), "non-boundary step")
        rejects(lambda: check(dict(base, optimizer={})), "missing optimizer")
        rejects(lambda: check(dict(base, torch_rng=None)), "missing RNG")
        rejects(lambda: check(base, current=dict(status, model="peak_only_s1")),
                "wrong active model")
        arbitrary = directory / "arbitrary.pt"
        rejects(lambda: check(base, path=arbitrary), "arbitrary PT in models")
        receipt = directory / f"{tag}.json"
        receipt.write_text('{"complete":true,"name":"cell_only_s1"}',
                           encoding="utf-8")
        completed_status = dict(phase="model_complete", variant="cell_only",
                                seed_index=1)
        rejects(lambda: helper.validate_artifact(receipt, directory,
                                                  completed_status, protocol,
                                                  identity),
                "incomplete model receipt")
        checkpoint = directory / f"{tag}.pt"
        seed = protocol["training"]["seeds"][0]
        torch.save(dict(state_dict=model.state_dict(),
                        mean=torch.zeros(32), std=torch.ones(32),
                        variant="aligned", seed=seed), checkpoint)
        reference = helper.read(ROOT / "runs/full_scaling_v18/models/full_s1.json")
        completed = dict(complete=True, name=tag, seed=seed, steps=13376,
                         queries=13373, fitting_queries=12886,
                         empty_skipped=487, parameter_count=22050,
                         protocol_sha256=helper.sha(helper.V1 / "protocol.json"),
                         checkpoint_path=str(checkpoint),
                         checkpoint_sha256=helper.sha(checkpoint),
                         initial_state_sha256=reference["initial_state_sha256"],
                         history=[dict(step=13376)])
        receipt.write_text(json.dumps(completed), encoding="utf-8")
        valid_receipt = helper.validate_artifact(receipt, directory,
                                                 completed_status, protocol,
                                                 identity)
        assert valid_receipt["kind"] == "completed_model"
        completed["checkpoint_sha256"] = "wrong"
        receipt.write_text(json.dumps(completed), encoding="utf-8")
        rejects(lambda: helper.validate_artifact(receipt, directory,
                                                  completed_status, protocol,
                                                  identity),
                "completed checkpoint SHA drift")
    hashes = {"fixed": "f" * 64, "adaptive": "a" * 64}
    mvp_public = dict(complete=True, n=500, methods=["fixed", "adaptive"],
                      prediction_sha256=hashes,
                      protocol_sha256="p" * 64,
                      highpixel_completion_sha256="h" * 64,
                      labels_opened=False)
    helper.validate_mvp_public(mvp_public, hashes, "p" * 64, "h" * 64)
    rejects(lambda: helper.validate_mvp_public(dict(mvp_public, n=499), hashes,
                                               "p" * 64, "h" * 64),
            "MVP wrong fixed denominator")
    rejects(lambda: helper.validate_mvp_public(
                dict(mvp_public, methods=["fixed"]), hashes,
                "p" * 64, "h" * 64), "MVP missing arm")
    rejects(lambda: helper.validate_mvp_public(
                dict(mvp_public, prediction_sha256={"fixed": "bad", "adaptive": "a" * 64}),
                hashes, "p" * 64, "h" * 64), "MVP prediction bundle SHA drift")
    rejects(lambda: helper.validate_mvp_public(
                dict(mvp_public, protocol_sha256="wrong"), hashes,
                "p" * 64, "h" * 64), "MVP protocol SHA drift")
    rejects(lambda: helper.validate_mvp_public(
                dict(mvp_public, highpixel_completion_sha256="wrong"), hashes,
                "p" * 64, "h" * 64), "ClickAgent 1M seal SHA drift")
    rejects(lambda: helper.validate_mvp_public(
                dict(mvp_public, labels_opened=True), hashes,
                "p" * 64, "h" * 64), "MVP labels-opened flag")
    # Exercise the read-only call path without reading live MVP predictions.
    with tempfile.TemporaryDirectory(prefix="mvp_gate_test_", dir=helper.OUT) as temp:
        fake = Path(temp)
        (fake / "highpixel").mkdir()
        (fake / "protocol.json").write_text("frozen", encoding="utf-8")
        (fake / "highpixel/public_completion.json").write_text(
            "sealed", encoding="utf-8")
        fake_public = dict(mvp_public,
                           protocol_sha256=helper.sha(fake / "protocol.json"),
                           highpixel_completion_sha256=helper.sha(
                               fake / "highpixel/public_completion.json"))
        (fake / "public_completion.json").write_text(json.dumps(fake_public),
                                                       encoding="utf-8")
        saved = (helper.MVP, helper.mvp_runner.PROTOCOL,
                 helper.mvp_runner.HPR, helper.mvp_runner.highpixel_sealed,
                 helper.mvp_runner.verify_method, helper.mvp_runner.verify_gate)
        calls = []
        try:
            helper.MVP = fake
            helper.mvp_runner.PROTOCOL = fake / "protocol.json"
            helper.mvp_runner.HPR = fake / "highpixel"
            helper.mvp_runner.highpixel_sealed = lambda: calls.append("highpixel")
            helper.mvp_runner.verify_method = lambda method: (
                calls.append(method), hashes[method])[1]
            helper.mvp_runner.verify_gate = lambda: (_ for _ in ()).throw(
                AssertionError("write-capable verify_gate called"))
            assert helper.mvp_public_gate() == hashes
            assert calls == ["highpixel", "fixed", "adaptive"]
            fake_public["prediction_sha256"] = dict(hashes, fixed="bad")
            (fake / "public_completion.json").write_text(json.dumps(fake_public),
                                                           encoding="utf-8")
            rejects(helper.mvp_public_gate, "read-only MVP gate bad method SHA")
        finally:
            (helper.MVP, helper.mvp_runner.PROTOCOL,
             helper.mvp_runner.HPR, helper.mvp_runner.highpixel_sealed,
             helper.mvp_runner.verify_method, helper.mvp_runner.verify_gate) = saved
    print("synthetic release-helper gate tests passed")


if __name__ == "__main__":
    main()
