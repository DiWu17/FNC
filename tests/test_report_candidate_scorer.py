"""Synthetic report fixtures only; no real test labels/results are accessed."""
import copy
import importlib.util
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest


SCRIPT = Path(__file__).resolve().parents[1]/'scripts/report_candidate_scorer.py'
spec = importlib.util.spec_from_file_location('report_v11', SCRIPT)
report = importlib.util.module_from_spec(spec); spec.loader.exec_module(report)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf8')


def duration():
    return dict(n=192, total_seconds=19.2, mean_seconds=.1, median_seconds=.09, p95_seconds=.17)


@pytest.fixture
def fixture(tmp_path):
    out = tmp_path/'synthetic_report_fixture'; out.mkdir()
    def cohort(split):
        return [dict(sample_id=f'synthetic:{split}:{i:03}', dataset=('screenspot_v2', 'screenspot_pro', 'venusbench_gd')[i//64],
            experiment_split=split, image=str(out/'synthetic.png'), image_size=[1000, 600], instruction='Synthetic fixture target',
            target_geometry=dict(type='bbox', boxes=[[.08, .08, .13, .13]])) for i in range(192)]
    fresh, old = cohort('test'), cohort('diagnostic')
    for path, cases in ((out/'manifest.jsonl', fresh), (out/'diagnostic_manifest.jsonl', old)):
        path.write_text(''.join(json.dumps(c)+'\n' for c in cases), encoding='utf8')
    selection = dict(complete=True, counts=dict(train=873, val=172, test=192))
    write(out/'data_selection.json', selection)
    protocol = dict(primary_budget=16, budgets=[4, 8, 16, 32], variants=list(report.VARIANTS),
        manifest_sha256=report.sha(out/'manifest.jsonl'), diagnostic_manifest_sha256=report.sha(out/'diagnostic_manifest.jsonl'),
        data_selection_sha256=report.sha(out/'data_selection.json'),
        optimization=dict(epochs=16, learning_rate=.0002, weight_decay=.02, accumulation=8, gradient_clip=1))
    write(out/'protocol.json', protocol)
    write(out/'data_selection_audit.json', dict(complete=True, manifest_sha256=protocol['manifest_sha256'],
        all_cohort_unique_files_rgb_verified=1080, historical_unresolved=0,
        cross_split_overlap={'train__test':dict(left_unique_images=568, right_unique_images=192, exact_pixel_overlaps=0, dhash_le_4_pairs=0)}))
    models = {}
    for variant in report.VARIANTS:
        for seed in report.SEEDS:
            name = variant+'_'+seed; path = out/'models'/(name+'.pt'); path.parent.mkdir(exist_ok=True)
            path.write_text('SYNTHETIC_REPORT_FIXTURE_NO_MODEL '+name)
            history = [dict(epoch=i, validation=dict(n=172, named=169, hits=20+i, loss=2-i*.03)) for i in range(17)]
            selected = history[-1]
            value = dict(complete=True, checkpoint_sha256=report.sha(path), protocol_sha256=report.sha(out/'protocol.json'),
                history=history, selected=selected, selected_validation_replay=dict(selected['validation'], oracle=90),
                selected_training_evaluation=dict(n=873, hits=380, loss=1.1, oracle=500),
                variant=variant, seed=20261131 if seed == 's1' else 20261132, parameters=1230338)
            write(path.with_suffix('.json'), value)
            models[name] = dict(file=str(path), sha256=report.sha(path), selected=selected)
    frozen = dict(complete=True, fresh_test_features_at_freeze=0, primary='internal_full_s1', primary_budget=16, models=models,
        files={str(p):report.sha(p) for p in (out/'models').glob('*')})
    write(out/'frozen.json', frozen)
    evidence = {}
    for split, cases in (('test', fresh), ('diagnostic', old)):
        details = []
        for i, case in enumerate(cases):
            pools = {}
            for budget in report.BUDGETS:
                points = [] if i%17 == 0 else [[10+j*10, 20] for j in range(budget)]
                hits = [j == 1 and i%5 != 0 for j in range(len(points))]
                pools[str(budget)] = dict(points=points, point_hits=hits, point_oracle=any(hits), n_points=len(points))
            methods = {}
            for mi, name in enumerate(sorted(report.expected_methods())):
                budget = int(name.rsplit('_K', 1)[1]) if '_K' in name else None
                if budget:
                    p = pools[str(budget)]; selected = 1 if i%(3+mi%3) != 0 else 0
                    point = p['points'][selected] if p['points'] else None
                    hit = p['point_hits'][selected] if p['points'] else False
                    row = dict(point=point, hit=hit, pool_budget=budget, pool_oracle=p['point_oracle'], n_points=p['n_points'])
                    if name.startswith(tuple(v+'_' for v in report.VARIANTS)):
                        rejected = bool(point is not None and i%19 == 0)
                        scores = [1. if j == selected else -.5 for j in range(len(p['points']))]
                        row.update(scores=scores, selected_index=selected if point else None,
                            null_score=(2 if rejected else -1) if point else None, null_would_reject=rejected,
                            optional_null_point=None if rejected else point, optional_null_hit=hit and not rejected,
                            elapsed_cached_head_seconds=.1)
                else:
                    row = dict(point=[100, 100], hit=True if name == 'ui_tars' else i%3 == 0, pool_budget=None)
                methods[name] = row
            details.append(dict(sample_id=case['sample_id'], dataset=case['dataset'], target_name='Synthetic target', pools=pools, methods=methods))
        metrics = {}
        for name in sorted(report.expected_methods()):
            values = [d['methods'][name] for d in details]; hits = sum(v['hit'] for v in values); budget = values[0]['pool_budget']
            oracle = sum(v['pool_oracle'] for v in values) if budget else None
            value = dict(n=192, hits=hits, accuracy=hits/192, ci95=[max(0, hits/192-.1), min(1, hits/192+.1)],
                points=sum(v['point'] is not None for v in values), pool_budget=budget, pool_available=oracle,
                conditional_selection_accuracy=hits/oracle if oracle else None,
                empty_pools=sum(v['n_points'] == 0 for v in values) if budget else None,
                nonempty_pool_without_correct_point=sum(v['n_points'] > 0 and not v['pool_oracle'] for v in values) if budget else None,
                selector_missed_correct_point=sum(v['pool_oracle'] and not v['hit'] for v in values) if budget else None,
                datasets={d:dict(n=64, hits=sum(v['methods'][name]['hit'] for v in details if v['dataset'] == d),
                                 accuracy=sum(v['methods'][name]['hit'] for v in details if v['dataset'] == d)/64)
                          for d in ('screenspot_v2', 'screenspot_pro', 'venusbench_gd')})
            if 'optional_null_hit' in values[0]:
                value['optional_null_policy'] = dict(hits=sum(v['optional_null_hit'] for v in values),
                    accuracy=sum(v['optional_null_hit'] for v in values)/192,
                    learned_rejections=sum(v['null_would_reject'] for v in values),
                    points=sum(v['optional_null_point'] is not None for v in values))
            metrics[name] = value
        pairs = [(report.PRIMARY, n) for n in sorted(metrics) if n != report.PRIMARY]
        pairs += [(f'{v}_{s}_K{k}', f'{v}_{s}_K16') for v in report.VARIANTS for s in report.SEEDS for k in (4, 8, 32)]
        comparisons = []
        for method, reference in dict.fromkeys(pairs):
            diff = np.array([int(d['methods'][method]['hit'])-int(d['methods'][reference]['hit']) for d in details])
            comparisons.append(dict(method=method, reference=reference, n=192, wins=int((diff > 0).sum()),
                losses=int((diff < 0).sum()), ties=int((diff == 0).sum()), delta=float(diff.mean()),
                ci95=[max(-1, float(diff.mean())-.1), min(1, float(diff.mean())+.1)]))
        pools = {str(k):dict(n=192, oracle=sum(d['pools'][str(k)]['point_oracle'] for d in details),
                    empty=sum(d['pools'][str(k)]['n_points'] == 0 for d in details),
                    total_points=sum(d['pools'][str(k)]['n_points'] for d in details), candidate_size_counts={str(k):180,'0':12}) for k in report.BUDGETS}
        timing = dict(cached_heads={n:duration() for n in metrics if n.startswith(tuple(v+'_' for v in report.VARIANTS))},
            frozen_qwen_extraction=dict(global_view=duration(), local_view=duration(), reused=split == 'diagnostic'),
            extra_visual_encoding=duration(), native=duration(),
            gui_reference=dict(reused_scores=split == 'diagnostic', batch_calls=32, measured_batches=duration(), scope='synthetic batch'))
        summary = dict(complete=True, split=split, n=192, primary=report.PRIMARY, robustness=report.ROBUSTNESS,
            exploratory_reused_images=split == 'diagnostic', selection_used_this_cohort=False, methods=metrics, pools=pools,
            paired_comparisons=comparisons, timing=timing, frozen_sha256=report.sha(out/'frozen.json'), protocol_sha256=report.sha(out/'protocol.json'))
        folder = out/'evaluation'/split
        write(folder/'summary.json', summary); write(folder/'details.json', details)
        predicted = [dict(sample_id=d['sample_id'], dataset=d['dataset'], method=n, **v) for d in details for n, v in d['methods'].items()]
        (folder/'predictions.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in predicted), encoding='utf8')
        for name in ('summary.json', 'details.json', 'predictions.jsonl'):
            evidence[str(folder/name)] = report.sha(folder/name)
    auditor = out/'synthetic_auditor.txt'; auditor.write_text('SYNTHETIC FIXTURE')
    audit = dict(complete=True, frozen_sha256=report.sha(out/'frozen.json'), protocol_sha256=report.sha(out/'protocol.json'),
        frozen_files_checked=20, training_files_checked=16, external_files_checked=3,
        source_sha256={str(auditor):report.sha(auditor)}, evaluation_sha256=evidence,
        splits={s:dict(n=192, methods=38, predictions_checked=192*38, replay=dict(complete=True, integer_points_exact=True)) for s in ('test', 'diagnostic')})
    write(out/'final_audit.json', audit)
    return out


def test_complete_fixture_keeps_prespecified_primary_even_when_reference_is_better(fixture):
    bundle = report.load_bundle(fixture)
    assert bundle['summaries']['test']['primary'] == report.PRIMARY
    assert len(bundle['histories']) == 8
    assert bundle['summaries']['test']['methods']['ui_tars']['hits'] == 192
    assert len(report.key_comparisons(bundle['summaries']['test'])) == 11


def test_incomplete_audit_refuses_before_reading_fresh_details(fixture):
    audit = report.read(fixture/'final_audit.json'); audit['complete'] = False; write(fixture/'final_audit.json', audit)
    (fixture/'evaluation/test/details.json').unlink()
    with pytest.raises(ValueError, match='Complete two-cohort'):
        report.load_bundle(fixture)
    assert not (fixture/'evaluation/assets').exists()


def test_changed_prediction_and_incomplete_history_are_rejected(fixture):
    path = fixture/'evaluation/test/predictions.jsonl'; path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError, match='Audited evaluation changed'):
        report.load_bundle(fixture)
    model = fixture/'models/internal_full_s1.json'; value = report.read(model); value['history'].pop(); write(model, value)
    with pytest.raises(ValueError, match='Training history changed after freeze'):
        report.load_bundle(fixture)


def test_missing_results_do_not_create_placeholder_report(tmp_path):
    missing = tmp_path/'unrun_experiment'
    with pytest.raises(FileNotFoundError):
        report.main(['--output', str(missing)])
    assert not missing.exists()


def test_full_synthetic_report_embeds_images_and_produces_static_assets(fixture):
    bundle = report.load_bundle(fixture)
    result = report.render_report(bundle, max_examples=0)
    text = (fixture/'evaluation/report.html').read_text(encoding='utf8')
    assert '内部候选评分器与专用视觉验证器' in text
    assert 'data:image/png;base64,' in text and '<img src="http' not in text
    assert '873' in text and '旧v9 192题' in text and '没有按测试分数更换主模型' in text
    assert len(result['assets_sha256']) == 18
    assert result['primary'] == report.PRIMARY and result['test_model_selection'] is False
    assert Path(result['report']).is_absolute()


def test_overlay_uses_report_only_geometry_and_deterministic_selection(fixture):
    bundle = report.load_bundle(fixture)
    chosen = report.select_examples(bundle, 6)
    assert len(chosen) <= 6
    assert len({v['case']['sample_id'] for v in chosen}) == len(chosen)
    assert [x['case']['sample_id'] for x in chosen] == [x['case']['sample_id'] for x in report.select_examples(bundle, 6)]
    example = copy.deepcopy(chosen[0]); path = fixture/'synthetic.png'; Image.new('RGB', (1000, 600), '#dddddd').save(path)
    example['case']['target_geometry'] = dict(type='bbox', boxes=[[.08,.08,.13,.13],[.7,.7,.8,.8]])
    destination = fixture/'synthetic_overlay.png'; report.plot_example(example, destination)
    assert destination.stat().st_size > 1000
