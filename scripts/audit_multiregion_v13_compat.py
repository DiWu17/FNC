"""Posthoc archive-schema repair around the byte-identical sealed V13 auditor.

The frozen producer archives six arrays; the frozen proposer/scorer consumes
four. Accept exactly the producer's six fields, retain every original metadata
check, and return exactly the consumer's four arrays to replay. No candidate,
sample, score tolerance, model, threshold, checkpoint or metric is changed.
This adapter was created after testing; it is not represented as pretest sealed.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

import audit_multiregion_v13 as base
from clickagent.action_benchmark import sha256, write_json
from multiregion_v13_protocol import OUT


CONSUMER_SOURCE = Path('src/clickagent/crossfit_proposer.py')
PRODUCER_SOURCE = Path('src/clickagent/conditioned_features.py')
TEST_SOURCE = Path('tests/test_multiregion_v13_audit_compat.py')
ORIGINAL_FOUR = frozenset(('target_attention', 'target_hidden', 'image_hidden', 'vision_hidden'))
ARCHIVE_ONLY = frozenset(('prefill_attention', 'prefill_hidden'))


def consumer_schema(source):
    tree = ast.parse(source)
    values = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
              and any(isinstance(target, ast.Name) and target.id == 'RAW_ARRAY_KEYS' for target in node.targets)]
    base.check(len(values) == 1, 'Expected one sealed consumer RAW_ARRAY_KEYS declaration')
    return values[0]


def producer_schema(source):
    tree = ast.parse(source)
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'ConditionedFeatureRuntime']
    base.check(len(classes) == 1, 'Expected one sealed conditioned producer class')
    methods = [node for node in classes[0].body if isinstance(node, ast.FunctionDef) and node.name == 'infer_spatial_features']
    base.check(len(methods) == 1, 'Expected one sealed spatial feature producer')
    assignments = [node for node in ast.walk(methods[0]) if isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id == 'arrays' for target in node.targets)]
    base.check(len(assignments) == 1 and isinstance(assignments[0].value, ast.Dict),
               'Expected one explicit sealed producer arrays dictionary')
    keys = assignments[0].value.keys
    base.check(all(isinstance(k, ast.Constant) and isinstance(k.value, str) for k in keys),
               'Producer schema must use explicit string keys without dictionary expansion')
    return tuple(k.value for k in keys)


def validate_schemas(consumer, producer):
    base.check(isinstance(consumer, (tuple, list)) and len(consumer) == 4 and
               len(set(consumer)) == 4 and set(consumer) == ORIGINAL_FOUR,
               'Sealed consumer contract is not the original exact four fields')
    base.check(isinstance(producer, (tuple, list)) and len(producer) == 6 and len(set(producer)) == 6 and
               set(producer) == ORIGINAL_FOUR | ARCHIVE_ONLY,
               'Sealed producer differs by more than the two archived prefill fields')
    return frozenset(producer)


@contextmanager
def archive_schema_adapter(consumer, producer):
    """Only two controlled in-memory substitutions; restore on every exit."""
    effective = validate_schemas(consumer, producer)
    base.check(base.RAW_KEYS == ORIGINAL_FOUR, 'Original auditor raw schema was already changed')
    base.check((base.REPLAY_PER_SOURCE, base.REPLAY_ATOL, base.REPLAY_RTOL, base.PARAMETERS) ==
               (6, 1e-4, 1e-5, 1230338), 'Original replay sample/tolerance/model constants were changed')
    old_keys, original_metadata = base.RAW_KEYS, base._raw_metadata
    protected = {name: value for name, value in vars(base).items() if name not in ('RAW_KEYS', '_raw_metadata')}
    @wraps(original_metadata)
    def consume_four(*args, **kwargs):
        arrays, metadata = original_metadata(*args, **kwargs)
        if arrays is not None:
            base.check(set(arrays) == effective, 'Original archive reader did not return the exact producer schema')
            arrays = {name: arrays[name] for name in consumer}
            base.check(set(arrays) == ORIGINAL_FOUR, 'Prefill archive fields reached replay model inputs')
        return arrays, metadata
    base.RAW_KEYS = set(effective)
    base._raw_metadata = consume_four
    try:
        yield
    finally:
        base.RAW_KEYS, base._raw_metadata = old_keys, original_metadata
        current = {name: value for name, value in vars(base).items() if name not in ('RAW_KEYS', '_raw_metadata')}
        base.check(current.keys() == protected.keys() and all(current[k] is v for k, v in protected.items()),
                   'Unexpected original auditor global change outside the two schema substitutions')


def _sealed_source(path, frozen):
    expected = [digest for name, digest in frozen['files'].items() if Path(name).resolve() == Path(path).resolve()]
    base.check(len(expected) == 1 and sha256(path) == expected[0], 'Source is not byte-identical to pretest seal: ' + str(path))
    return expected[0]


def run_compat(root):
    root = Path(root); frozen_path = root / 'frozen.json'; frozen = base.read(frozen_path)
    base.check(frozen['complete'] and frozen['fresh_test_features_at_freeze'] == 0, 'Missing pretest V13 freeze')
    paths = [Path(base.__file__), CONSUMER_SOURCE, PRODUCER_SOURCE, Path(__file__), TEST_SOURCE, frozen_path]
    original_sha = _sealed_source(Path(base.__file__), frozen)
    consumer_sha = _sealed_source(CONSUMER_SOURCE, frozen)
    producer_sha = _sealed_source(PRODUCER_SOURCE, frozen)
    # Do not retroactively claim this new entry point was sealed before testing.
    base.check(not any(Path(p).resolve() == Path(__file__).resolve() for p in frozen['files']),
               'This posthoc adapter must not be represented as an original frozen source')
    consumer = consumer_schema(CONSUMER_SOURCE.read_text(encoding='utf-8-sig'))
    producer = producer_schema(PRODUCER_SOURCE.read_text(encoding='utf-8-sig'))
    validate_schemas(consumer, producer)
    failure_log = root / 'logs/final_audit.log'
    if failure_log.exists():
        paths.append(failure_log)
    before = {str(p.resolve()): sha256(p) for p in paths}
    with archive_schema_adapter(consumer, producer):
        result = base.run_audit(root)
    after = {str(p.resolve()): sha256(p) for p in paths}
    base.check(before == after, 'Adapter/original/producer/consumer/test/frozen/log changed during audit')
    base.check(result['complete'] and result['cpu_replay']['actual_cpu_forwards'] == 120,
               'Original full final audit did not complete all 120 forwards')
    result['schema_adapter'] = dict(
        kind='posthoc_strict_archive_schema_compatibility', created_after_test=True,
        adapter_was_pretest_sealed=False,
        reason='Original auditor confused four consumed feature arrays with the sealed producer six-array archive; the additional fields are frozen prefill diagnostics, not labels.',
        original_auditor_sha256=original_sha,
        canonical_producer_source=str(PRODUCER_SOURCE), canonical_producer_source_sha256=producer_sha,
        canonical_producer_class='ConditionedFeatureRuntime.infer_spatial_features',
        canonical_consumer_source=str(CONSUMER_SOURCE), canonical_consumer_source_sha256=consumer_sha,
        original_auditor_four_keys=sorted(ORIGINAL_FOUR), producer_archive_six_keys=list(producer),
        effective_archive_keys=sorted(set(producer)), consumer_input_four_keys=list(consumer),
        archive_only_prefill_keys=sorted(ARCHIVE_ONLY),
        filtering='Original reader validates exact six-field archive and all original metadata/GT checks; only consumer four are returned to every replay builder.',
        exact_archive_key_check_preserved=True, metadata_and_gt_checks_unchanged=True,
        other_auditor_logic_unchanged=True, score_tolerances_unchanged=True, fixed_samples_unchanged=True,
        adapter_source_sha256=before[str(Path(__file__).resolve())],
        regression_tests_sha256=before[str(TEST_SOURCE.resolve())],
        source_identities_unchanged_during_execution=True,
        preserved_original_failure_log_sha256=before.get(str(failure_log.resolve())))
    result['input_sha256'].update(before)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root', type=Path, default=OUT)
    args = parser.parse_args()
    import torch
    torch.set_num_threads(4)
    result = run_compat(args.root)
    write_json(args.root / 'final_audit.json', result)
    print('COMPLETE original final audit with strict producer6/consumer4 archive compatibility; actual CPU forwards=120', flush=True)


if __name__ == '__main__':
    main()
