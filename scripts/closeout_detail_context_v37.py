"""Independent archive and statistic replay for the TRAIN-only V37 study."""
from collections import defaultdict
import json
from pathlib import Path

import numpy as np
import run_detail_context_v37 as run


def main():
    spec = run.verify()
    out = run.OUT
    initial = run.read(out / "completion.json")
    assert initial["complete"]
    for name, digest in initial["artifact_sha256"].items():
        assert run.sha(out / name) == digest
    cases = run.read(run.OLD / "data/public.json")
    labels = run.read(run.OLD / "data/train_labels.json")
    old_index = run.read(run.OLD / "raw_archive_index.json")
    recorded = run.read(out / "predictions.json")["rows"]
    assert len(cases) == len(recorded) == 112
    assert [c["sample_id"] for c in cases] == [r["sample_id"] for r in recorded]
    reserve_ids = {r["sample_id"] for r in run.read(run.ROOT / "runs/independent_diagnostic_v32/data/reserve_public.json")}
    assert not reserve_ids.intersection(r["sample_id"] for r in recorded)
    hashes = {}
    judgments = 0
    token_alignments = 0
    coords = 0
    identical_rgb_checks = 0
    for case, row in zip(cases, recorded):
        sid = case["sample_id"]
        assert row["source"] == case["source"]
        for arm in ("640_low", "1280_low", *run.ARMS):
            side = arm.split("_")[0]
            if arm.endswith("low"):
                path = run.OLD / "raw" / side / (run.key(sid) + ".npz")
                relative = str(path.relative_to(run.OLD)).replace("\\", "/")
                expected = old_index.get(relative) or old_index.get(str(path.relative_to(run.OLD)))
                assert expected == run.sha(path)
            else:
                path = out / "raw" / arm / (run.key(sid) + ".npz")
                hashes[str(path.relative_to(out))] = run.sha(path)
            with np.load(path, allow_pickle=False) as archive:
                meta = json.loads(str(archive["metadata"]))
                n_tokens = len(meta["generated_token_ids"])
                assert archive["all_attention"].shape[0] == archive["all_hidden"].shape[0] == n_tokens
                assert len(meta["generated_prefixes"]) == n_tokens
                assert not n_tokens or meta["generated_prefixes"][-1] == meta["generated_text"]
                token_alignments += 1
            transform = case["transforms"][side]
            assert meta["transform"] == transform and meta["image_size"] == transform["crop_size"]
            assert meta["ground_truth_in_model_input"] is False and meta["all_backbone_parameters_frozen"]
            assert meta["counter"]["backbone_forward_calls"] == 1 + n_tokens
            cap = run.HIGH_PIXELS if arm.endswith("hi") else 262144
            assert 0 < meta["counter"]["processed_pixels"] <= cap
            if arm.endswith("hi"):
                old_path = run.OLD / "raw" / side / (run.key(sid) + ".npz")
                old_meta = run.metadata(old_path)
                assert meta["local_RGB_sha256"] == old_meta["local_RGB_sha256"]
                identical_rgb_checks += 1
            if meta["point"] is None:
                point = None
            else:
                local = np.asarray(meta["point"], float) * (np.asarray(meta["image_size"]) - 1) / 1000
                world = local / transform["scale"] + np.asarray(transform["box"][:2])
                point = np.clip(np.rint(world), [0, 0], np.asarray(case["image_size"]) - 1).astype(int).tolist()
            assert meta["world_point"] == point == row["arms"][arm]["point"]
            coords += 1
            hit = point is not None and any(
                x0 - 1e-9 <= point[0] / case["image_size"][0] <= x1 + 1e-9
                and y0 - 1e-9 <= point[1] / case["image_size"][1] <= y1 + 1e-9
                for x0, y0, x1, y1 in labels[sid]["boxes"])
            assert hit == row["arms"][arm]["hit"]
            judgments += 1
    assert len(hashes) == 224
    assert len(list((out / "raw").glob("*/*.npz"))) == 224
    for path in sorted((out / "runtime/warmup").glob("*.npz")):
        hashes[str(path.relative_to(out))] = run.sha(path)
    assert len(hashes) == 226
    result = run.read(out / "results.json")
    groups = defaultdict(list)
    for i, row in enumerate(recorded):
        groups[row["source"]].append(i)
    for name, comp in result["comparisons"].items():
        a = np.asarray([r["arms"][comp["a"]]["hit"] for r in recorded], bool)
        b = np.asarray([r["arms"][comp["b"]]["hit"] for r in recorded], bool)
        assert int(a.sum()) == comp["a_hits"] and int(b.sum()) == comp["b_hits"]
        assert int((~a & b).sum()) == comp["repairs"] and int((a & ~b).sum()) == comp["harms"]
        diff = b.astype(int) - a.astype(int)
        assert abs(100 * diff.mean() - comp["delta_pp"]) < 1e-12
        rng = np.random.default_rng(comp["seed"])
        draws = np.asarray([100 * diff[np.concatenate([rng.choice(indices, len(indices), replace=True)
                           for indices in groups.values()])].mean() for _ in range(10000)])
        np.testing.assert_allclose(np.quantile(draws, [.025, .975]), comp["paired_CI95_pp"], atol=1e-12, rtol=0)
    for model in spec["adopted_models"].values():
        assert run.sha(model["path"]) == model["sha256"]
    run.write(out / "archive_index.json", hashes)
    audit = {"complete": True, "at_utc": run.now(), "coordinate_replays": coords,
             "independent_point_judgments": judgments, "token_alignment_checks": token_alignments,
             "identical_crop_RGB_checks": identical_rgb_checks,
             "bootstrap_comparisons_replayed": len(result["comparisons"]), "bootstrap_draws_per_comparison": 10000,
             "new_archives_hashed": len(hashes), "old_archive_hashes_checked": 224,
             "frozen_inputs_checked": len(spec["input_sha256"]), "reserve_no_forward": True,
             "adopted_models_unchanged": True}
    run.write(out / "independent_audit.json", audit)
    final_text = (out / "readout.txt").read_text(encoding="utf-8")
    assert final_text.startswith("V36 TRAIN-only")
    final_text = final_text.replace("V36 TRAIN-only", "V37 TRAIN-only", 1)
    final_text += "原始readout.txt的V36标题属于复制实验脚本时的文字笔误；该文件及已封存的实验源码保留，最终报告仅更正标题，不改结果。\n"
    final_text += "\n独立复核：448次坐标、点标签和真实token对齐；5项预定配对区间各10000次重放；新226归档SHA及旧224归档SHA检查一致。\n"
    (out / "final_readout.txt").write_text(final_text, encoding="utf-8")
    run.write(out / "final_completion.json", {"complete": True, "at_utc": run.now(),
              "initial_completion_sha256": run.sha(out / "completion.json"),
              "archive_index_sha256": run.sha(out / "archive_index.json"),
              "audit_sha256": run.sha(out / "independent_audit.json"),
              "final_readout_sha256": run.sha(out / "final_readout.txt")})
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
