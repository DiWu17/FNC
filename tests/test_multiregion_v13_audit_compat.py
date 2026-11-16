import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import audit_multiregion_v13_compat as adapter


FOUR = ('target_attention', 'target_hidden', 'image_hidden', 'vision_hidden')
SIX = FOUR + ('prefill_attention', 'prefill_hidden')


def raw_archive(tmp_path, *, keys=SIX, metadata_extra=None):
    case = dict(sample_id='synthetic', image_sha256='synthetic-image', image_size=[64, 64], instruction='click target')
    meta = dict(sample_id=case['sample_id'], image_sha256=case['image_sha256'], image_size=case['image_size'],
        own_token_forward=True, all_backbone_parameters_frozen=True, ground_truth_in_model_input=False,
        instruction_sha256=hashlib.sha256(case['instruction'].encode()).hexdigest(), grid_shape=[1, 1])
    meta.update(metadata_extra or {})
    values = {name: np.asarray([i], np.float32) for i, name in enumerate(keys)}
    values['metadata'] = np.asarray(json.dumps(meta))
    path = tmp_path / 'raw.npz'
    with path.open('wb') as stream:
        np.savez_compressed(stream, **values)
    return case, path, values


def invoke(case, path, load=True):
    return adapter.base._raw_metadata(path, case, None, adapter.base.Ledger(), adapter.sha256(path), load_arrays=load)


def test_exact_six_array_archive_returns_exact_four_consumer_arrays(tmp_path):
    case, path, original = raw_archive(tmp_path)
    original_reader = adapter.base._raw_metadata
    with adapter.archive_schema_adapter(FOUR, SIX):
        arrays, meta = invoke(case, path)
        assert tuple(arrays) == FOUR and not set(arrays) & adapter.ARCHIVE_ONLY
        for name in FOUR:
            np.testing.assert_array_equal(arrays[name], original[name])
        assert meta['ground_truth_in_model_input'] is False
        assert invoke(case, path, False)[0] is None
    assert adapter.base.RAW_KEYS == adapter.ORIGINAL_FOUR
    assert adapter.base._raw_metadata is original_reader


@pytest.mark.parametrize('keys', [SIX[:-1], SIX + ('teacher_click',), FOUR])
def test_missing_prefill_or_extra_teacher_archive_is_rejected(tmp_path, keys):
    case, path, _ = raw_archive(tmp_path, keys=keys)
    with adapter.archive_schema_adapter(FOUR, SIX):
        with pytest.raises(ValueError, match='unexpected arrays'):
            invoke(case, path)


@pytest.mark.parametrize('extra', [{'labels': [1]}, {'target_geometry': {'boxes': []}},
                                  {'ground_truth_in_model_input': True}, {'own_token_forward': False}])
def test_original_metadata_and_gt_checks_still_reject(tmp_path, extra):
    case, path, _ = raw_archive(tmp_path, metadata_extra=extra)
    with adapter.archive_schema_adapter(FOUR, SIX):
        with pytest.raises(ValueError):
            invoke(case, path)


@pytest.mark.parametrize('consumer,producer', [
    (SIX, SIX),
    (FOUR, FOUR + ('prefill_attention', 'teacher_score')),
    (FOUR, SIX + ('teacher_point',)),
    (FOUR, SIX[:-1]),
    (FOUR, FOUR + ('prefill_attention', 'prefill_attention')),
])
def test_wrong_canonical_contract_or_difference_is_rejected(consumer, producer):
    with pytest.raises(ValueError):
        adapter.validate_schemas(consumer, producer)


def test_ast_reads_actual_sealed_producer6_and_consumer4_without_model_execution():
    consumer = adapter.consumer_schema(adapter.CONSUMER_SOURCE.read_text(encoding='utf-8'))
    producer = adapter.producer_schema(adapter.PRODUCER_SOURCE.read_text(encoding='utf-8'))
    assert consumer == FOUR and set(producer) == set(SIX)
    assert adapter.validate_schemas(consumer, producer) == frozenset(SIX)
    expansion = "class ConditionedFeatureRuntime:\n def infer_spatial_features(self):\n  arrays={**other}\n"
    with pytest.raises(ValueError):
        adapter.producer_schema(expansion)


def test_score_tolerance_or_other_original_global_change_is_rejected(monkeypatch):
    original = adapter.base.REPLAY_RTOL
    monkeypatch.setattr(adapter.base, 'REPLAY_RTOL', original)
    with pytest.raises(ValueError, match='outside the two schema'):
        with adapter.archive_schema_adapter(FOUR, SIX):
            adapter.base.REPLAY_RTOL = original * 2
    assert adapter.base.RAW_KEYS == adapter.ORIGINAL_FOUR


def test_already_changed_original_constants_are_rejected(monkeypatch):
    monkeypatch.setattr(adapter.base, 'REPLAY_ATOL', 1e-3)
    with pytest.raises(ValueError, match='constants were changed'):
        with adapter.archive_schema_adapter(FOUR, SIX):
            pass


def test_wrapper_delegates_original_full_audit_and_records_honest_posthoc_identity(tmp_path, monkeypatch):
    frozen = dict(complete=True, fresh_test_features_at_freeze=0, files={str(path): adapter.sha256(path)
        for path in (Path(adapter.base.__file__), adapter.CONSUMER_SOURCE, adapter.PRODUCER_SOURCE)})
    (tmp_path / 'frozen.json').write_text(json.dumps(frozen), encoding='utf-8')
    calls = []
    def original_full_audit(root):
        calls.append(root)
        assert adapter.base.RAW_KEYS == set(SIX)
        return dict(complete=True, cpu_replay=dict(actual_cpu_forwards=120), input_sha256={})
    monkeypatch.setattr(adapter.base, 'run_audit', original_full_audit)
    result = adapter.run_compat(tmp_path)
    assert calls == [tmp_path] and result['cpu_replay']['actual_cpu_forwards'] == 120
    proof = result['schema_adapter']
    assert proof['created_after_test'] and proof['adapter_was_pretest_sealed'] is False
    assert set(proof['consumer_input_four_keys']) == set(FOUR)
    assert set(proof['effective_archive_keys']) == set(SIX)
    assert proof['canonical_producer_source_sha256'] == adapter.sha256(adapter.PRODUCER_SOURCE)
    assert result['input_sha256'][str(Path(adapter.__file__).resolve())] == adapter.sha256(adapter.__file__)
    assert adapter.base.RAW_KEYS == adapter.ORIGINAL_FOUR
