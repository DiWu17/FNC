"""Synthetic, label-free checks of the paused-MVP handoff inventory."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import record_review_attention_gpu_release_v3 as helper


def reject(fn, reason: str) -> None:
    try:
        fn()
    except (ValueError, KeyError):
        print("rejected:", reason)
    else:
        raise AssertionError("Accepted invalid paused-MVP inventory: " + reason)


def main() -> None:
    helper.OUT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="v3_pause_test_", dir=helper.OUT) as temp:
        fake = Path(temp)
        (fake / "predictions/adaptive/cases").mkdir(parents=True)
        (fake / "highpixel").mkdir()
        (fake / "protocol.json").write_text("frozen", encoding="utf-8")
        (fake / "highpixel/public_completion.json").write_text(
            "sealed", encoding="utf-8")
        (fake / "adaptive_bugfix_addendum_v1.json").write_text(
            "bugfix", encoding="utf-8")
        (fake / "status.json").write_text(json.dumps(dict(
            phase="loading_model", method="adaptive", labels_opened=False)),
            encoding="utf-8")
        rows = [dict(sample_id="case0"), dict(sample_id="case1")]
        case = fake / "predictions/adaptive/cases/case0.json"
        case.write_text(json.dumps(dict(sample_id="case0", method="adaptive")),
                        encoding="utf-8")
        pause = dict(protocol_sha256=helper.sha(fake / "protocol.json"),
                     runner_sha256=helper.sha(Path(helper.mvp_runner.__file__)),
                     adaptive_bugfix_addendum_sha256=helper.sha(
                         fake / "adaptive_bugfix_addendum_v1.json"),
                     fixed=dict(completed=500, prediction_bundle_sha256="f" * 64,
                                labels_opened=False),
                     adaptive=dict(completed=1, total=500,
                                   case_sha256={"case0": helper.sha(case)},
                                   paused_in_phase="loading_model", labels_opened=False),
                     execution=dict(pty_exited=True, gpu0_memory_after_exit_mib=28),
                     analysis=dict(all_prediction_gate_run=False,
                                   test_labels_opened=False, accuracy_computed=False))
        (fake / "paused_by_user_v1.json").write_text(json.dumps(pause),
                                                      encoding="utf-8")
        saved = (helper.MVP, helper.mvp_runner.PROTOCOL, helper.mvp_runner.HPR,
                 helper.mvp_runner.highpixel_sealed, helper.mvp_runner.verify_method,
                 helper.mvp_runner.pro_rows, helper.mvp_runner.case_path,
                 helper.mvp_runner.check_case, helper.mvp_runner.verify_gate)
        calls = []
        try:
            helper.MVP = fake
            helper.mvp_runner.PROTOCOL = fake / "protocol.json"
            helper.mvp_runner.HPR = fake / "highpixel"
            helper.mvp_runner.highpixel_sealed = lambda: calls.append("highpixel")
            helper.mvp_runner.verify_method = lambda method: (
                calls.append(method), "f" * 64)[1]
            helper.mvp_runner.pro_rows = lambda: rows
            helper.mvp_runner.case_path = lambda method, sid: (
                fake / f"predictions/{method}/cases/{sid}.json")
            helper.mvp_runner.check_case = lambda record, public, method, digest: (
                calls.append("case"),
                helper.require(record["sample_id"] == public["sample_id"] and
                               record["method"] == method, "Wrong cached case"))
            helper.mvp_runner.verify_gate = lambda: (_ for _ in ()).throw(
                AssertionError("write-capable verify_gate called"))
            result = helper.mvp_pause_inventory()
            assert result["fixed_complete"] is True
            assert result["adaptive_cache_count"] == 1
            assert result["adaptive_case_sha256"] == {"case0.json": helper.sha(case)}
            assert calls == ["highpixel", "fixed", "case"]
            (fake / "predictions/adaptive/cases/unknown.json").write_text("{}",
                                                                            encoding="utf-8")
            reject(helper.mvp_pause_inventory, "unknown adaptive cache file")
            (fake / "predictions/adaptive/cases/unknown.json").unlink()
            (fake / "predictions/adaptive.json").write_text("{}", encoding="utf-8")
            reject(helper.mvp_pause_inventory, "adaptive already sealed")
            (fake / "predictions/adaptive.json").unlink()
            (fake / "status.json").write_text(json.dumps(dict(
                phase="loading_model", method="adaptive", labels_opened=True)),
                encoding="utf-8")
            reject(helper.mvp_pause_inventory, "labels-opened status")
            (fake / "status.json").write_text(json.dumps(dict(
                phase="loading_model", method="adaptive", labels_opened=False)),
                encoding="utf-8")
            pause["adaptive"]["case_sha256"]["case0"] = "wrong"
            (fake / "paused_by_user_v1.json").write_text(json.dumps(pause),
                                                          encoding="utf-8")
            reject(helper.mvp_pause_inventory, "user pause receipt cached-case SHA drift")
        finally:
            (helper.MVP, helper.mvp_runner.PROTOCOL, helper.mvp_runner.HPR,
             helper.mvp_runner.highpixel_sealed, helper.mvp_runner.verify_method,
             helper.mvp_runner.pro_rows, helper.mvp_runner.case_path,
             helper.mvp_runner.check_case, helper.mvp_runner.verify_gate) = saved
    print("paused-MVP handoff inventory tests passed")


if __name__ == "__main__":
    main()
