"""Record verified V37 TRAIN mechanism result without changing the adopted method."""
import json
from pathlib import Path

import run_detail_context_v37 as run


def main():
    out = run.OUT
    final = run.read(out / "final_completion.json")
    assert final["complete"]
    for name, key in (("completion.json", "initial_completion_sha256"),
                      ("archive_index.json", "archive_index_sha256"),
                      ("independent_audit.json", "audit_sha256"),
                      ("final_readout.txt", "final_readout_sha256")):
        assert run.sha(out / name) == final[key]
    audit = run.read(out / "independent_audit.json")
    assert audit["complete"] and audit["reserve_no_forward"] and audit["adopted_models_unchanged"]
    failed = run.read(run.ROOT / "runs/detail_context_v36/status.json")
    assert failed["phase"] == "failed" and "max_pixels=262144" in failed["error"]
    assert not list((run.ROOT / "runs/detail_context_v36/raw").glob("*/*.npz"))
    result = run.read(out / "results.json")
    primary = result["comparisons"]["primary_context_with_detail_restored"]
    assert primary["a_hits"] == 52 and primary["b_hits"] == 59
    registry_path = run.ROOT / "CURRENT_RESEARCH.json"
    registry = run.read(registry_path)
    assert registry["run_id"] == "quality_readout_v15_3k"
    assert registry["active_experiment"]["run_id"] == "paired_scale_v35"
    assert registry["active_experiment"]["status"] == "complete"
    assert not any(entry["run_id"] == "detail_context_v37" for entry in registry["completed_experiments"])
    for model in registry["models"].values():
        assert run.sha(model["path"]) == model["sha256"]
    decision_path = run.ROOT / "research/decisions/20261111_detail_context_v37_closeout.json"
    decision = {
        "decision": "Close V37 as TRAIN-only mechanism evidence; retain adopted V15_3k mainline and three fixed seeds. Do not promote 1280 high-budget observation from reused TRAIN outcomes.",
        "at_utc": run.now(),
        "run_id": "detail_context_v37",
        "cohort": "112 previously used TRAIN screenshots, 7 sources x16; no DEV labels or reserve forward",
        "primary": primary,
        "arms_hits": {arm: data["hits"] for arm, data in result["arms"].items()},
        "interpretation": "Higher budget plus wider view improves native local point on TRAIN, but high-budget width-only contrast is inconclusive; detail and observation remain bottlenecks, and best arm still misses53/112. No independently confirmed MVP advantage.",
        "cost": {arm: {"mean_processed_pixels": data["mean_pixels"], "mean_local_seconds": data["mean_seconds"]} for arm, data in result["arms"].items()},
        "v36_preflight": "Budget guard rejected max_pixels=1048576 before model load; no real outputs. V37 isolated experimental loader bypasses that guard without changing public protocol implementation.",
        "next": "TRAIN-only simple source-held-out content selector on actual old/new points; freeze before one reserve confirmation.",
        "artifact_sha256": {name: run.sha(out / name) for name in
                            ("protocol.json", "final_completion.json", "final_readout.txt", "interpretation.txt",
                             "results.json", "mechanism.json", "independent_audit.json", "archive_index.json")},
        "mainline_unchanged": True,
    }
    decision_path.parent.mkdir(parents=True, exist_ok=True)
    decision_path.write_text(json.dumps(decision, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    entry = {"run_id": "detail_context_v37", "status": "complete", "training_only": True,
             "protocol_path": str(out / "protocol.json"), "report_path": str(out / "final_readout.txt"),
             "interpretation_path": str(out / "interpretation.txt"), "completion_path": str(out / "final_completion.json"),
             "decision_path": str(decision_path), "decision_sha256": run.sha(decision_path),
             "completed_at_utc": decision["at_utc"], "mainline_unchanged": True,
             "reserve_no_forward": True}
    registry["completed_experiments"].append(entry)
    registry["active_experiment"] = {"run_id": "detail_context_v37", "status": "complete",
                                      "phase": "independent_closeout_verified", "protocol_path": entry["protocol_path"],
                                      "results_path": entry["report_path"], "interpretation_path": entry["interpretation_path"],
                                      "completion_path": entry["completion_path"], "decision_path": entry["decision_path"],
                                      "mainline_unchanged": True}
    registry_path.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"registered": entry, "adopted": registry["run_id"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
