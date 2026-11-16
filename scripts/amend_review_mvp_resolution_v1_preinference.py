"""Transparent pre-inference correction of the registered MVP 1M controls.

This narrow amendment changes failure receipts and timing disclosure only.
It refuses to run after any GPU runtime or test prediction has appeared.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_review_mvp_resolution_v1 as runner

OUT = ROOT / "runs/review_mvp_resolution_v1"
PROTOCOL = OUT / "protocol.json"
OLD_SHA = "9587ab34572d5d54b5223b5acc976e8760dfb8c933ffe38a50ee1f86d27dbdfd"
BACKUP = OUT / "protocol_before_preinference_amendment.json"
ADDENDUM = OUT / "preinference_amendment.json"


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + ".tmp." + str(os.getpid()))
    temp.write_bytes(data)
    os.replace(temp, path)


def main() -> None:
    assert not ADDENDUM.exists() and not BACKUP.exists(), "Amendment already recorded"
    assert not (OUT / "predictions").exists(), "Prediction file exists; no pre-inference amendment allowed"
    assert not (OUT / "runtime").exists(), "GPU runtime file exists; no pre-inference amendment allowed"
    assert not (OUT / "public_completion.json").exists(), "Completion exists"
    status = json.loads((OUT / "status.json").read_text(encoding="utf-8"))
    assert status["phase"] == "registered", "GPU stage may have begun"
    original_bytes = PROTOCOL.read_bytes()
    assert sha_bytes(original_bytes) == OLD_SHA, "Prior frozen protocol SHA changed"
    before = json.loads(original_bytes)
    assert before["pixel_cap_per_view"] == 1_048_576 and before["n"] == 500
    assert before["methods"] == ["fixed", "adaptive"]
    after = json.loads(original_bytes)
    after["source_hashes"] = runner.source_hashes()
    after["runtime"]["timer"] = (
        "GPU-synchronized around one public inference, excluding PNG decode, image load, and file I/O; "
        "ClickAgent 1M pilot.one_call includes PNG decode, so raw latency is descriptive rather than a matched speed comparison")
    after["start_gate"] = (
        "Both MVP modes refuse GPU model load until all 2581 high-pixel ClickAgent public predictions are SHA sealed "
        "and status is complete; the operator must also confirm the ClickAgent 1M GPU process has exited before invoking either mode")
    after["limitations"].append(
        "MVP timer excludes PNG decode while ClickAgent 1M timer includes it; no matched end-to-end latency claim")
    assert after["fixed"] == before["fixed"] and after["adaptive"] == before["adaptive"]
    assert after["model_shards"] == before["model_shards"]
    assert after["public_source_sha256"] == before["public_source_sha256"]
    assert after["highpixel_protocol_sha256"] == before["highpixel_protocol_sha256"]
    changed = {key for key in after if before[key] != after[key]}
    assert changed == {"source_hashes", "runtime", "start_gate", "limitations"}, changed
    content = (json.dumps(after, ensure_ascii=False, indent=2) + "\n").encode("utf-8")
    addendum = {
        "complete": True,
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "phase": "before_any_gpu_model_load_or_test_prediction",
        "previous_protocol_sha256": OLD_SHA,
        "corrected_protocol_sha256": sha_bytes(content),
        "previous_runner_sha256": before["source_hashes"]["scripts/run_review_mvp_resolution_v1.py"],
        "corrected_runner_sha256": after["source_hashes"]["scripts/run_review_mvp_resolution_v1.py"],
        "previous_analyzer_sha256": before["source_hashes"]["scripts/analyze_review_mvp_resolution_v1.py"],
        "corrected_analyzer_sha256": after["source_hashes"]["scripts/analyze_review_mvp_resolution_v1.py"],
        "changed_protocol_fields": sorted(changed),
        "science_unchanged": "Methods, model, 1048576-pixel budget, 500 public cases, crop geometry, warmup IDs, metric, and analysis contrasts unchanged.",
        "amendments": [
            "Capture failures during model load and validation-image warmup, in addition to inference failures.",
            "Require ClickAgent 1M completion status before GPU load; operator must confirm its GPU process exited.",
            "State that MVP timer excludes PNG decode whereas ClickAgent 1M timer includes it; do not claim matched speed.",
        ],
        "prior_status_phase": status["phase"],
        "prediction_directory_absent": True,
        "runtime_directory_absent": True,
        "labels_opened": False,
    }
    write_atomic(BACKUP, original_bytes)
    write_atomic(ADDENDUM, (json.dumps(addendum, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    write_atomic(PROTOCOL, content)
    assert runner.sha(PROTOCOL) == addendum["corrected_protocol_sha256"]
    assert runner.gate()[0] == after
    runner.status("registered", protocol_sha256=runner.sha(PROTOCOL),
                  amendment_sha256=runner.sha(ADDENDUM), n=500, labels_opened=False)
    print(json.dumps({"amended_before_inference": True,
                      "protocol_sha256": runner.sha(PROTOCOL),
                      "addendum_sha256": runner.sha(ADDENDUM)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
