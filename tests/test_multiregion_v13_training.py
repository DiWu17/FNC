"""CPU-only V13 contracts using synthetic records, never real test outputs."""
from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch


@pytest.fixture
def modules(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / "scripts"))
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield (importlib.import_module("train_multiregion_v13"),
           importlib.import_module("cache_multiregion_v13"))
    torch.set_num_threads(old)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


class ForbiddenLabels:
    def __getitem__(self, key):
        raise AssertionError("A held-out training label was accessed")


class TinyScorer(torch.nn.Module):
    """One parameter lets the actual optimizer/backward loop run on CPU."""
    def __init__(self, config):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.2))

    def forward(self, x):
        return {"scores": self.weight * x, "null_scores": self.weight.reshape(1)}


def record(labels):
    return dict(labels=labels, x=np.asarray([1., -1.], np.float32),
                pool_indices={str(k): np.asarray([0, 1], np.int64) for k in (4, 8, 16, 32)})


def test_cv_excludes_source_and_crossing_image_components_before_label_access(modules, tmp_path, monkeypatch):
    train, _ = modules
    monkeypatch.setattr(train, "OUT", tmp_path)
    write(tmp_path / "protocol.json", {"optimization": dict(epochs=1, learning_rate=.001,
          weight_decay=0., accumulation=2, gradient_clip=1.)})
    train_rows = [dict(sample_id="held", dataset="a", image_fold_group="shared"),
                  dict(sample_id="crossing", dataset="b", image_fold_group="shared"),
                  dict(sample_id="allowed", dataset="b", image_fold_group="other")]
    val_rows = [dict(sample_id="held-val", dataset="a"), dict(sample_id="other-val", dataset="b")]
    data = [record(ForbiddenLabels()), record(ForbiddenLabels()), record(np.asarray([1., 0.]))]
    validation = [record(np.asarray([1., 0.])), record(ForbiddenLabels())]
    monkeypatch.setattr(train, "cases", lambda split: train_rows if split == "train" else val_rows)
    monkeypatch.setattr(train, "dataset", lambda split, condition: data if split == "train" else validation)
    monkeypatch.setattr(train, "sources", lambda: [])
    monkeypatch.setattr(train, "MultiRegionScorer", TinyScorer)
    supplied_inputs = []
    def subset(rec, indices, visual, device):
        supplied_inputs.append(tuple(rec))
        return dict(x=torch.as_tensor(rec["x"][indices], device=device))
    monkeypatch.setattr(train, "subset", subset)
    for split in ("train", "val"):
        write(tmp_path / "cache" / f"{split}_c3_index.json", {})
    train.train_one("c3_top1", "s1", "cpu", held_source="a")
    result = train.read(tmp_path / "cross_validation/c3_top1_held0_s1.json")
    assert result["train_sample_ids"] == ["allowed"]
    assert result["validation_sample_ids"] == ["held-val"]
    assert result["excluded_cross_source_component_ids"] == ["crossing"]
    assert result["held_source_labels_used_for_gradient"] is False
    assert result["history"][1]["queries"] == 1
    assert supplied_inputs  # The fake scorer accepts only x, never labels/GT.


def test_cv_uses_macro_source_rate_and_transfers_epoch_to_both_full_seeds(modules, tmp_path, monkeypatch):
    train, _ = modules
    monkeypatch.setattr(train, "OUT", tmp_path)
    write(tmp_path / "protocol.json", {"optimization": {"epochs": 2}})
    # Epoch1 wins macro Top1 but epoch2 wins pooled Top1; Top4 wins at epoch2.
    for objective in ("top1", "top4"):
        for fold, n in enumerate((1000, 1, 1, 1)):
            history = []
            for epoch in range(3):
                top1 = 0 if epoch == 0 else (int(fold != 0) if epoch == 1 else (1000 if fold == 0 else 0))
                top4 = 0 if epoch == 0 else ((500 if fold == 0 else 0) if epoch == 1 else int(fold != 0))
                history.append(dict(epoch=epoch, validation=dict(n=n, hits=top1,
                               top4_hits=top4, loss=1.)))
            write(tmp_path / "cross_validation" / f"c3_{objective}_held{fold}_s1.json",
                  dict(complete=True, held_source_labels_used_for_gradient=False, history=history))
    for arm in train.ARMS:
        for suffix in train.SEEDS:
            history = []
            for epoch in range(3):
                path = tmp_path / "training" / f"{arm}_{suffix}_epoch_{epoch:02}.pt"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(f"synthetic:{arm}:{suffix}:{epoch}".encode())
                # Give S2 a different independent val optimum to expose accidental transfer.
                hits = 3 if epoch == (1 if suffix == "s1" else 2) else 1
                history.append(dict(epoch=epoch, checkpoint_path=str(path), checkpoint_sha256=train.sha256(path),
                    validation=dict(n=4, hits=hits, top4_hits=hits, loss=1.)))
            write(tmp_path / "training" / f"{arm}_{suffix}.json", dict(complete=True, history=history,
                  initial_state_sha256=f"shared-initial-{suffix}"))
    train.select()
    selected = train.read(tmp_path / "models/selection.json")
    assert selected["cross_validation"]["top1"]["selected_epoch"] == 1
    assert selected["cross_validation"]["top4"]["selected_epoch"] == 2
    for suffix in train.SEEDS:
        assert selected["selections"][f"c3_top1_{suffix}_loso"]["epoch"] == 1
        assert selected["selections"][f"c3_top4_{suffix}_loso"]["epoch"] == 2
    assert selected["selections"]["c3_top1_s2_val"]["epoch"] == 2
    assert selected["fresh_test_read"] is False


def fake_cache_environment(cache, tmp_path, monkeypatch, split="test"):
    monkeypatch.setattr(cache, "OUT", tmp_path)
    monkeypatch.setattr(cache, "assert_sealed", lambda: {"complete": True})
    raw = tmp_path / "raw.npz"
    raw.write_bytes(b"raw-v1")
    case = dict(sample_id="synthetic-cache", image_size=[10, 10], image_sha256="synthetic-image-sha",
                instruction="Click save",
                target_geometry={"type": "bbox", "boxes": [[0, 0, .5, .5]]})
    pool = {str(k): [dict(point=[2, 3])] for k in cache.BUDGETS}
    pipeline_path = tmp_path / "pipeline" / split / (cache.key(case["sample_id"]) + ".json")
    write(pipeline_path, dict(global_feature_path=str(raw), global_feature_sha256=cache.sha256(raw),
          local_views=[], pools={"c1": pool, "c3": pool}))
    monkeypatch.setattr(cache, "cases", lambda selected_split: [case])
    monkeypatch.setattr(cache, "raw_arrays", lambda path: ({"fingerprint": Path(path).read_bytes()}, {}))
    monkeypatch.setattr(cache, "build_multiregion_inputs", lambda *args: {"query_hidden": np.asarray([1.])})
    return raw, case


def test_test_cache_never_derives_or_stores_labels(modules, tmp_path, monkeypatch):
    _, cache = modules
    _, case = fake_cache_environment(cache, tmp_path, monkeypatch)
    case["target_geometry"] = object()  # Any accidental annotation use must fail.
    def forbidden(*args):
        raise AssertionError("Test annotations reached cache generation")
    monkeypatch.setattr(cache, "point_in_geometry", forbidden)
    cache.cache("test")
    for condition in ("c1", "c3"):
        index = cache.read(tmp_path / "cache" / f"test_{condition}_index.json")
        assert index["complete"] and index["labels_present"] is False
        with np.load(index["cases"][0]["path"], allow_pickle=False) as archive:
            assert "labels" not in archive.files
            assert str(archive["sample_id"]) == case["sample_id"]


def test_cache_resume_rejects_raw_bytes_different_from_pipeline_hash(modules, tmp_path, monkeypatch):
    _, cache = modules
    raw, _ = fake_cache_environment(cache, tmp_path, monkeypatch)
    cache.cache("test")
    raw.write_bytes(b"raw-v2")
    with pytest.raises((AssertionError, ValueError), match=".*"):
        cache.cache("test")


@pytest.mark.parametrize("field", ["labels", "pool_indices_16", "sample_id", "cache_fingerprint"])
def test_cache_resume_rejects_tampered_cached_record(modules, tmp_path, monkeypatch, field):
    _, cache = modules
    fake_cache_environment(cache, tmp_path, monkeypatch, split="train")
    cache.cache("train")
    index = cache.read(tmp_path / "cache/train_c1_index.json")
    path = Path(index["cases"][0]["path"])
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files}
    arrays[field] = np.asarray({"labels": [0.], "pool_indices_16": [99],
                               "sample_id": "other-case", "cache_fingerprint": "other-input"}[field])
    cache.save_npz(path, arrays)
    with pytest.raises((AssertionError, ValueError)):
        cache.cache("train")


def test_full_arms_and_seeds_have_identical_real_schedule_but_seeded_initialization(modules, tmp_path, monkeypatch):
    train, _ = modules
    rows = [dict(sample_id=f"sample-{i}", dataset="a", image_fold_group=f"group-{i}") for i in range(5)]
    data = [record(np.asarray([1., 0.])) for _ in rows]
    monkeypatch.setattr(train, "cases", lambda split: rows)
    monkeypatch.setattr(train, "dataset", lambda split, condition: data)
    monkeypatch.setattr(train, "sources", lambda: [])
    traces = {}
    current = [None]
    class RandomScorer(TinyScorer):
        def __init__(self, config):
            super().__init__(config)
            with torch.no_grad():
                self.weight.copy_(torch.randn(()))
        def forward(self, x):
            if self.training:
                traces[current[0]].append((x.tolist(), torch.initial_seed()))
            return super().forward(x)
    monkeypatch.setattr(train, "MultiRegionScorer", RandomScorer)
    monkeypatch.setattr(train, "subset", lambda rec, indices, visual, device:
                        {"x": torch.as_tensor(rec["x"][indices], device=device)})
    monkeypatch.setattr(train, "save_multiregion_scorer", lambda path, model, metadata:
                        torch.save(dict(state=model.state_dict(), metadata=metadata), path))
    outputs = {}
    for suffix in train.SEEDS:
        for arm in train.ARMS:
            out = tmp_path / f"{arm}_{suffix}"
            monkeypatch.setattr(train, "OUT", out)
            write(out / "protocol.json", {"optimization": dict(epochs=2, learning_rate=.001,
                  weight_decay=0., accumulation=2, gradient_clip=1.)})
            condition = arm.split("_")[0]
            for split in ("train", "val"):
                write(out / "cache" / f"{split}_{condition}_index.json", {})
            current[0] = f"{arm}_{suffix}"
            traces[current[0]] = []
            train.train_one(arm, suffix, "cpu")
            outputs[(arm, suffix)] = train.read(out / "training" / f"{arm}_{suffix}.json")
    # Query order/budgets are logged; actual subset permutation/dropout seed are observed.
    assert len({tuple(r["schedule_sha256"] for r in result["history"]) for result in outputs.values()}) == 1
    assert all(trace == next(iter(traces.values())) for trace in traces.values())
    assert len(next(iter(traces.values()))) == 10
    for suffix in train.SEEDS:
        assert len({outputs[(arm, suffix)]["initial_state_sha256"] for arm in train.ARMS}) == 1
    assert outputs[(train.ARMS[0], "s1")]["initial_state_sha256"] != outputs[(train.ARMS[0], "s2")]["initial_state_sha256"]


def test_evaluation_counts_unavailable_queries_without_feeding_labels(modules, monkeypatch):
    train, _ = modules
    inputs_seen = []
    def subset(rec, indices, visual, device):
        inputs_seen.append(dict(x=torch.tensor([1., -1.])))
        return inputs_seen[-1]
    monkeypatch.setattr(train, "subset", subset)
    scorer = TinyScorer(None)
    result = train.evaluate(scorer, [None, record(np.asarray([1., 0.]))], [0, 1], "top4", "cpu")
    assert result["n"] == 2 and result["named"] == 1
    assert result["hits"] == result["top4_hits"] == 1
    assert list(inputs_seen[0]) == ["x"]

