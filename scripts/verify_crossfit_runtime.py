"""V12 CPU-only full-byte model and frozen feature-runtime authentication.

This does not instantiate a model or open any prediction. The sole allowed
configuration difference is the offload storage directory.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
OUT = ROOT / 'runs/crossfit_candidates_v12/runtime_identity.json'
LEGACY = ROOT / 'runs/candidate_scorer_v11/frozen.json'
QWEN_IDENTITY = ROOT / 'data/conditioned_v3/qwen_weight_identity.json'
GUI_IDENTITY = ROOT / 'models/GUI-Actor-Verifier-2B/clickagent_download_verified.json'
RUNTIME_INPUT = ROOT / 'runs/candidate_v2/identity.json'
LEGACY_RUNTIME = ROOT / 'runs/candidate_scorer_v11/trainval/runtime.json'
PACKAGES = ('torch', 'transformers', 'accelerate', 'numpy', 'Pillow')
CORE_SOURCES = (
    'src/clickagent/conditioned_features.py',
    'src/clickagent/context_verifier.py',
    'src/clickagent/target_token.py',
    'src/clickagent/model_features.py',
    'src/clickagent/action_model.py',
    'src/clickagent/point_aligned_inference.py',
    'scripts/run_point_aligned_inference.py',
    'scripts/cache_candidate_scorer.py',
    'scripts/extract_candidate_scorer.py',
    'scripts/run_gui_verifier.py',
    'src/clickagent/model_baselines.py',
)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest_object(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def normalized_path(path):
    return os.path.normcase(os.path.abspath(path))


def full_sha(path):
    """Read the complete file and detect replacement during hashing."""
    path = Path(path).resolve()
    before = path.stat()
    digest = hashlib.sha256()
    count = 0
    last_report = 0
    started = time.monotonic()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
            count += len(block)
            if count - last_report >= 1024**3:
                print(json.dumps(dict(stage='full_byte_hash', file=path.name, bytes_read=count,
                                      total_bytes=before.st_size)), flush=True)
                last_report = count
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), ('Changed during hashing', path)
    assert count == before.st_size
    return dict(path=str(path), sha256=digest.hexdigest(), bytes=count,
                mtime_ns=after.st_mtime_ns, seconds=round(time.monotonic()-started, 6), full_bytes_read=True)


def legacy_references(path, seal):
    target = normalized_path(path)
    hits = []
    for location, inventory in (
        ('files', seal['files']),
        ('external_input_evidence.files', seal.get('external_input_evidence', {}).get('files', {})),
    ):
        for original, expected in inventory.items():
            if normalized_path(original) == target:
                hits.append(dict(location=location, key=original, sha256=expected))
    return hits


def expected_in_legacy(path, seal):
    hits = legacy_references(path, seal)
    assert hits, ('Not authenticated by the V11 seal', str(path))
    assert len({h['sha256'] for h in hits}) == 1, ('Conflicting old identities', str(path))
    return hits[0]['sha256'], hits


def authenticate_small(path, seal):
    expected, references = expected_in_legacy(path, seal)
    actual = full_sha(path)
    assert actual['sha256'] == expected, ('V11 identity mismatch', str(path), expected, actual['sha256'])
    return dict(actual, expected_sha256=expected, legacy_references=references, matches=True)


def verify_model_files(model_root, inventory, seal, model_name):
    model_root = Path(model_root).resolve()
    expected_weights = {name for name in inventory if name.endswith(('.safetensors', '.bin'))}
    actual_weights = {str(p.relative_to(model_root)).replace('\\', '/')
                      for p in model_root.rglob('*') if p.is_file() and p.suffix in ('.safetensors', '.bin')}
    assert actual_weights == expected_weights, ('Uninventoried or missing weights', model_name,
                                                sorted(actual_weights ^ expected_weights))
    records = []
    for relative, entry in sorted(inventory.items()):
        path = (model_root / relative).resolve()
        assert path.is_relative_to(model_root)
        expected_bytes = entry.get('bytes', entry.get('size'))
        assert path.stat().st_size == expected_bytes, ('Model file size', path)
        actual = full_sha(path)
        assert actual['sha256'] == entry['sha256'], ('Model bytes changed', path, actual['sha256'])
        actual.update(model=model_name, relative_path=relative, expected_sha256=entry['sha256'],
                      expected_bytes=expected_bytes, matches=True,
                      kind='weight' if relative in expected_weights else 'config_tokenizer_or_identity')
        # Qwen shards were checked during the V11 freeze and are recorded in
        # external evidence even when omitted from the cheap test-time seal.
        references = legacy_references(path, seal)
        if references:
            old_expected, references = expected_in_legacy(path, seal)
            assert old_expected == actual['sha256'], ('Legacy model-file mismatch', path)
        actual['legacy_direct_references'] = references
        records.append(actual)
        print(json.dumps(dict(stage='verified_file', model=model_name, file=relative,
                              bytes=actual['bytes'], sha256=actual['sha256'])), flush=True)
    return records


def runtime_contract():
    # Importing definitions does not create ContextFeatureRuntime, a model, or
    # any GPU tensor. Import late so the code snapshot is checked in one window.
    module_paths = [ROOT/'scripts/extract_candidate_scorer.py', ROOT/'scripts/extract_crossfit_candidates.py']
    before = {str(p): full_sha(p)['sha256'] for p in module_paths}
    from extract_candidate_scorer import runtime_config as legacy_config
    from extract_crossfit_candidates import runtime_config as current_config
    old_full, new_full = asdict(legacy_config()), asdict(current_config())
    old, new = dict(old_full), dict(new_full)
    old_storage, new_storage = old.pop('offload_folder'), new.pop('offload_folder')
    differences = {k: dict(legacy=old.get(k), current=new.get(k)) for k in sorted(old.keys() | new.keys())
                   if old.get(k) != new.get(k)}
    assert old == new, ('Runtime configuration mismatch beyond offload_folder', differences)
    after = {str(p): full_sha(p)['sha256'] for p in module_paths}
    assert before == after, 'Extraction code changed while comparing runtime configurations'
    # JSON object keys are strings, including GPU IDs in max_memory; normalize
    # only after asserting exact equality of the two untouched Python configs.
    contract = json.loads(json.dumps(new))
    return dict(runtime_contract=contract, runtime_contract_sha256=digest_object(contract),
                runtime_contract_serialization="SHA256(json.dumps(JSON_roundtrip(config_without_offload_folder),sort_keys=True,separators=(',',':')).encode())",
                runtime_config_exactly_equal_except=['offload_folder'],
                excluded_storage=dict(legacy=old_storage, current=new_storage),
                extractor_source_sha256=after)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    os.chdir(ROOT)
    output = args.output.resolve()
    assert output.is_relative_to((ROOT/'runs/crossfit_candidates_v12').resolve())
    # The experiment orchestrator may invoke the default command again. An
    # existing artifact always means read-only verification, never replacement.
    audit_only = args.audit_only or output.exists()
    seal = read(LEGACY)
    assert seal['complete'] and seal['fresh_test_features_at_freeze'] == 0
    legacy_hash = full_sha(LEGACY)['sha256']
    verifier_hash = full_sha(__file__)['sha256']
    qwen_id_record = authenticate_small(QWEN_IDENTITY, seal)
    gui_id_record = authenticate_small(GUI_IDENTITY, seal)
    runtime_input_record = authenticate_small(RUNTIME_INPUT, seal)
    legacy_runtime_record = full_sha(LEGACY_RUNTIME)
    legacy_versions = read(LEGACY_RUNTIME)['library_versions']
    current_versions = {name: importlib.metadata.version(name) for name in PACKAGES}
    assert {name: legacy_versions[name] for name in PACKAGES} == current_versions, (
        'Library versions differ from the designated V11 trainval runtime', legacy_versions, current_versions)
    old_runtime_references = legacy_references(LEGACY_RUNTIME, seal)
    if old_runtime_references:
        assert expected_in_legacy(LEGACY_RUNTIME, seal)[0] == legacy_runtime_record['sha256']
    legacy_runtime_record.update(legacy_references=old_runtime_references,
        listed_in_legacy_seal=bool(old_runtime_references),
        reference_scope='designated V11 trainval runtime record; not historically seal-pinned unless references are listed')
    core = {relative: authenticate_small(ROOT/relative, seal) for relative in CORE_SOURCES}
    qwen = read(QWEN_IDENTITY)
    gui = read(GUI_IDENTITY)
    assert gui['complete']
    qwen_files = verify_model_files(qwen['model_path'], qwen['files'], seal, 'Qwen3-VL-8B-Instruct')
    gui_files = verify_model_files(GUI_IDENTITY.parent, {r['file']: r for r in gui['files']},
                                   seal, 'GUI-Actor-Verifier-2B')
    contract = runtime_contract()
    assert Path(contract['runtime_contract']['model_path']).resolve() == Path(qwen['model_path']).resolve()
    assert full_sha(LEGACY)['sha256'] == legacy_hash
    for relative, entry in core.items():
        assert full_sha(ROOT/relative)['sha256'] == entry['sha256'], ('Core source changed during verification', relative)
    assert full_sha(QWEN_IDENTITY)['sha256'] == qwen_id_record['sha256']
    assert full_sha(GUI_IDENTITY)['sha256'] == gui_id_record['sha256']
    assert full_sha(RUNTIME_INPUT)['sha256'] == runtime_input_record['sha256']
    assert full_sha(LEGACY_RUNTIME)['sha256'] == legacy_runtime_record['sha256']
    assert full_sha(__file__)['sha256'] == verifier_hash, 'Verifier source changed during verification'
    result = dict(complete=True, created_utc=datetime.now(timezone.utc).isoformat(),
        legacy_frozen_path=str(LEGACY), legacy_frozen_sha256=legacy_hash,
        qwen_identity_path=str(QWEN_IDENTITY), qwen_identity_sha256=qwen_id_record['sha256'],
        gui_identity_path=str(GUI_IDENTITY), gui_identity_sha256=gui_id_record['sha256'],
        qwen_identity_authentication=qwen_id_record, gui_identity_authentication=gui_id_record,
        runtime_config_input_identity=runtime_input_record,
        legacy_library_versions_reference=legacy_runtime_record,
        library_versions=current_versions, legacy_library_versions={name: legacy_versions[name] for name in PACKAGES},
        library_versions_equal=True,
        actual_model_files=qwen_files+gui_files,
        actual_weight_files=[r for r in qwen_files+gui_files if r['kind'] == 'weight'],
        core_source_identity=core, full_model_bytes_read=sum(r['bytes'] for r in qwen_files+gui_files),
        no_model_execution=True, no_predictions_read=True,
        verifier_source_path=str(Path(__file__).resolve()), verifier_source_sha256=verifier_hash,
        package_versions=current_versions,
        **contract)
    if audit_only:
        prior = read(output)
        for key in ('legacy_frozen_sha256', 'qwen_identity_sha256', 'gui_identity_sha256',
                    'runtime_contract_sha256', 'runtime_contract', 'extractor_source_sha256',
                    'library_versions', 'legacy_library_versions', 'library_versions_equal'):
            assert prior[key] == result[key], key
        for key in ('runtime_config_input_identity', 'legacy_library_versions_reference'):
            assert prior[key]['sha256'] == result[key]['sha256'], key
        expected = {r['path']: (r['sha256'], r['bytes']) for r in prior['actual_model_files']}
        actual = {r['path']: (r['sha256'], r['bytes']) for r in result['actual_model_files']}
        assert expected == actual
        print(json.dumps(dict(runtime_identity_audit_complete=True, runtime_identity_sha256=full_sha(output)['sha256'])), flush=True)
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)+'\n', encoding='utf-8')
        temporary.replace(output)
        print(json.dumps(dict(complete=True, output=str(output), runtime_identity_sha256=full_sha(output)['sha256'],
            runtime_contract_sha256=result['runtime_contract_sha256'], full_model_bytes_read=result['full_model_bytes_read'])), flush=True)


if __name__ == '__main__':
    main()
