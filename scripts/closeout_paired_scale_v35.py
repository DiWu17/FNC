"""Independent decision/statistics replay after all V35 observations finish."""
from collections import defaultdict
import json
from pathlib import Path
import numpy as np
import run_paired_scale_v35 as run


def main():
    out = run.OUT
    initial = run.read(out / "completion.json")
    assert initial["complete"]
    for name, expected in initial["artifact_sha256"].items():
        assert run.sha(out / name) == expected
    spec = run.verify()
    cases = run.read(out / "data/public.json")
    labels = run.read(out / "data/train_labels.json")
    rows = run.read(out / "predictions.json")["rows"]
    predictions = {row["sample_id"]: row for row in rows}
    result = run.read(out / "results.json")
    reserve = {row["sample_id"] for row in run.read(run.ROOT / "runs/independent_diagnostic_v32/data/reserve_public.json")}
    assert not reserve.intersection(predictions) and len(cases) == len(rows) == len(predictions) == 112
    coordinate_replays = point_judgments = token_alignment_checks = 0
    native_hits = {arm: [] for arm in run.ARMS}
    archive_hashes = {}
    conditional = defaultdict(lambda: defaultdict(int))
    for case in cases:
        sid = case["sample_id"]
        label = labels[sid]
        old_fraction = run.geometry.fraction(case["transforms"]["640"]["box"], case["image_size"], label)
        for arm in run.ARMS:
            path = out / "raw" / arm / (run.key(sid) + ".npz")
            archive_hashes[str(path.relative_to(out))] = run.sha(path)
            with np.load(path, allow_pickle=False) as archive:
                meta = json.loads(str(archive["metadata"]))
                tokens = len(meta["generated_token_ids"])
                assert archive["all_attention"].shape[0] == archive["all_hidden"].shape[0] == tokens
                assert len(meta["generated_prefixes"]) == tokens
                assert not tokens or meta["generated_prefixes"][-1] == meta["generated_text"]
                token_alignment_checks += 1
            transform = case["transforms"][arm]
            assert meta["transform"] == transform and meta["image_size"] == transform["crop_size"]
            assert meta["instruction"] == case["instruction"] and meta["system_prompt"] == run.LOCAL_PROMPT
            assert meta["ground_truth_in_model_input"] is False and meta["all_backbone_parameters_frozen"]
            assert meta["counter"]["backbone_forward_calls"] == 1 + tokens
            assert meta["counter"]["processed_pixels"] <= 262144
            if meta["point"] is None:
                point = None
            else:
                local = np.asarray(meta["point"], float) * (np.asarray(meta["image_size"]) - 1) / 1000
                world = local / 2 + transform["box"][:2]
                point = np.clip(np.rint(world), [0, 0], np.asarray(case["image_size"]) - 1).astype(int).tolist()
            assert point == meta["world_point"] == predictions[sid]["arms"][arm]["point"]
            coordinate_replays += 1
            if point is None:
                hit = False
            else:
                x, y = point[0] / case["image_size"][0], point[1] / case["image_size"][1]
                hit = any(x0 - 1e-12 <= x <= x1 + 1e-12 and y0 - 1e-12 <= y <= y1 + 1e-12 for x0, y0, x1, y1 in label["boxes"])
            assert hit == predictions[sid]["arms"][arm]["hit"]
            point_judgments += 1
            native_hits[arm].append(hit)
            frac = run.geometry.replay_fraction(transform["box"], case["image_size"], label)
            assert abs(frac - predictions[sid]["arms"][arm]["GT_fraction"]) < 1e-12
            if old_fraction == 0:
                conditional[arm]["old_main_crop_miss_n"] += 1
                conditional[arm]["old_main_crop_miss_hits"] += hit
                conditional[arm]["old_main_crop_miss_geometrically_covered"] += frac > 0
            if frac >= .99:
                conditional[arm]["full_coverage_n"] += 1
                conditional[arm]["full_coverage_failures"] += not hit
                conditional[arm]["full_coverage_abstentions"] += point is None
    a, b = [np.asarray(native_hits[arm], bool) for arm in run.ARMS]
    assert int(a.sum()) == result["arms"]["640"]["hits"]
    assert int(b.sum()) == result["arms"]["1280"]["hits"]
    assert int((~a & b).sum()) == result["repairs"]
    assert int((a & ~b).sum()) == result["harms"]
    # Reproduce the registered bootstrap using integer success differences.
    groups = defaultdict(list)
    for index, row in enumerate(rows):
        groups[row["source"]].append(index)
    differences = b.astype(int) - a.astype(int)
    rng = np.random.default_rng(20261135)
    draws = np.empty(10000)
    for iteration in range(10000):
        indices = np.concatenate([rng.choice(group, len(group), replace=True) for group in groups.values()])
        draws[iteration] = 100 * differences[indices].sum() / len(indices)
    np.testing.assert_allclose(np.quantile(draws, [.025, .975]), result["paired_CI95_pp"], atol=1e-12, rtol=0)
    np.testing.assert_allclose(100 * differences.mean(), result["delta_pp"], atol=1e-12, rtol=0)
    orders = defaultdict(list)
    log_text = "\n".join(path.read_text(encoding="utf-8") for path in sorted(out.glob("stdout*.log")))
    for line in log_text.splitlines():
        if not line.startswith("{"):
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("phase") == "train_observations" and event.get("current_case") is not None:
            orders[event["current_case"]].append(event["arm"])
    assert len(orders) == 112
    for index, case in enumerate(cases, 1):
        assert orders[index] == case["arm_order"]
    assert len(list((out / "raw").glob("*/*.npz"))) == 224
    assert len(list((out / "runtime/warmup").glob("*.npz"))) == 2
    for path in sorted((out / "runtime/warmup").glob("*.npz")):
        archive_hashes[str(path.relative_to(out))] = run.sha(path)
    run.write(out / "raw_archive_index.json", archive_hashes)
    for model in spec["adopted_models"].values():
        assert run.sha(model["path"]) == model["sha256"]
    if (out / "resume01_amendment.json").exists():
        amendment = run.read(out / "resume01_amendment.json")
        for path, digest in amendment["pre_resume_archive_sha256"].items():
            assert run.sha(path) == digest, path
        assert run.sha(run.ROOT / "scripts/resume_paired_scale_v35.py") == amendment["wrapper_sha256"]
    audit = {"complete": True, "at_utc": run.now(), "local_coordinate_replays": coordinate_replays, "independent_point_judgments": point_judgments, "real_token_alignment_checks": token_alignment_checks, "paired_order_replays": len(orders), "bootstrap_draws_replayed": 10000, "frozen_input_hashes": len(spec["hashes"]), "raw_archives_hashed": len(archive_hashes), "raw_archive_index_sha256": run.sha(out / "raw_archive_index.json"), "conditional_geometry": {arm: dict(counts) for arm, counts in conditional.items()}, "reserve_no_forward": True, "mainline_unchanged": True}
    run.write(out / "independent_final_audit.json", audit)
    final_text = (out / "readout.txt").read_text(encoding="utf-8")
    final_text += "\n独立收尾：224次坐标/点标签/真实token对齐及112次成对执行顺序重放一致，10000次预定bootstrap区间复算一致。\n"
    final_text += "完整目标可见仍失败的诊断：" + str(audit["conditional_geometry"]) + "\n"
    (out / "final_readout.txt").write_text(final_text, encoding="utf-8")
    run.write(out / "final_completion.json", {"complete": True, "at_utc": run.now(), "initial_completion_sha256": run.sha(out / "completion.json"), "independent_audit_sha256": run.sha(out / "independent_final_audit.json"), "final_readout_sha256": run.sha(out / "final_readout.txt"), "raw_archive_index_sha256": run.sha(out / "raw_archive_index.json")})
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
