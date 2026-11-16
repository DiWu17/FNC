"""Label-free one-case GPU resource preflight for higher-resolution MVP.

This is a new diagnostic run, not a revision to the sealed PointerBench
comparison. It reads only the public ScreenSpot-Pro image/instruction manifest,
records no coordinates or accuracy, and keeps the prior MVP model/runtime code.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
RUN = ROOT / 'runs/mvp_pixel_preflight_v1'
SSP_PUBLIC = ROOT / 'runs/paper_scale_ablation_v2/eval/full/public.json'
POINTER_PUBLIC = ROOT / 'runs/paper_scale_ablation_v2/pointerbench/public.jsonl'
SSP_ID = 'screenspot_pro:000615:macos_common_macos_64'
POINTER_ID = 'pbp_0000'
BUDGETS = (1048576, 4096 * 2160)


def write(name, value):
    RUN.mkdir(parents=True, exist_ok=True)
    path = RUN / name
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temp.replace(path)
    print(json.dumps({'file': str(path), **value}, ensure_ascii=False), flush=True)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    import torch
    from PIL import Image
    from clickagent.advantage_runtimes import configure_paths, MVPAdapter

    torch.set_num_threads(4)
    ssp = json.loads(SSP_PUBLIC.read_text(encoding='utf-8-sig'))
    ssp_row = next(x for x in ssp if x['sample_id'] == SSP_ID)
    pointer = [json.loads(line) for line in POINTER_PUBLIC.read_text(encoding='utf-8-sig').splitlines() if line]
    pointer_row = next(x for x in pointer if x['sample_id'] == POINTER_ID)
    cases = [('pointerbench_pro', pointer_row), ('screenspot_pro_stress', ssp_row)]
    assert pointer_row['dataset'] == 'PointerBench' and pointer_row['image_size'] == [1024, 768]
    assert ssp_row['dataset'] == 'ScreenSpot-Pro' and ssp_row['image_size'] == [6016, 3384]
    for _, row in cases:
        assert digest(row['image']) == row['image_sha256']
    assert torch.cuda.device_count() == 2, 'Expected existing two-GPU MVP placement'
    device_names = [torch.cuda.get_device_name(i) for i in range(2)]
    assert all(name == 'Quadro P5000' for name in device_names)

    protocol = dict(
        purpose='Resource preflight only; no labels, point correctness, or model selection',
        public_manifests=[dict(path=str(p), sha256=digest(p)) for p in (POINTER_PUBLIC, SSP_PUBLIC)],
        cases=[dict(case_type=kind, sample_id=row['sample_id'],
                    image_sha256=row['image_sha256'], image_size=row['image_size'])
               for kind, row in cases],
        budgets_per_view_pixels=list(BUDGETS),
        model='Qwen3-VL-8B-Instruct with original MVPAdapter',
        runtime='FP16/SDPA/balanced two GPUs/max 12 GiB per GPU/batch 1',
        max_subimages=2, ground_truth_opened=False,
        frozen_comparison_untouched=True, devices=device_names,
    )
    write('protocol.json', protocol)

    configure_paths('mvp')
    started_load = time.perf_counter()
    runtime = MVPAdapter(RUN / 'runtime', max_pixels=BUDGETS[0])
    load_seconds = time.perf_counter() - started_load
    assert runtime.metadata['max_subimages'] == 2
    write('status.json', dict(phase='loaded_model', load_seconds=load_seconds,
                              device_map=runtime.metadata['device_map'],
                              updated_utc=datetime.now(timezone.utc).isoformat()))

    records = []
    stop = False
    for kind, row in cases:
        if stop:
            break
        with Image.open(row['image']) as source:
            image = source.convert('RGB')
        for budget in BUDGETS:
            runtime.processor.image_processor.max_pixels = budget
            grids = []

            def capture_grid(module, args, kwargs):
                grid = kwargs.get('image_grid_thw')
                if grid is not None and kwargs.get('pixel_values') is not None:
                    grid = grid.detach().cpu().tolist()
                    grids.extend(grid)

            hook = runtime.model.register_forward_pre_hook(capture_grid, with_kwargs=True)
            for i in range(2):
                torch.cuda.empty_cache()
                torch.cuda.synchronize(i)
                torch.cuda.reset_peak_memory_stats(i)
            before = runtime.counter.snapshot()
            write('status.json', dict(phase='running', max_pixels=budget,
                                      case_type=kind, sample_id=row['sample_id'],
                                      completed_budgets=len(records),
                                      updated_utc=datetime.now(timezone.utc).isoformat()))
            started = time.perf_counter()
            try:
                with torch.inference_mode():
                    result = runtime.infer(image, row['instruction'])
                for i in range(2):
                    torch.cuda.synchronize(i)
                outcome = 'success'
                error = None
                output_failure = result.get('output_failure')
                # No target label or generated coordinate is serialized.
            except BaseException as exc:
                outcome = 'failed'
                error = type(exc).__name__ + ': ' + str(exc)[-2500:]
                output_failure = None
            finally:
                hook.remove()
            after = runtime.counter.snapshot()
            rec = dict(case_type=kind, sample_id=row['sample_id'],
                       original_image_size=row['image_size'],
                       max_pixels=budget, outcome=outcome, error=error,
                       output_failure=output_failure,
                       elapsed_seconds=time.perf_counter() - started,
                       image_grid_thw=grids,
                       actual_prefill_pixels=[int(g[1] * g[2] * 16**2) for g in grids],
                       backbone_forward_calls=after['backbone_forward_calls']-before['backbone_forward_calls'],
                       processed_pixels=after['processed_pixels']-before['processed_pixels'],
                       peak_allocated_bytes=[torch.cuda.max_memory_allocated(i) for i in range(2)],
                       peak_reserved_bytes=[torch.cuda.max_memory_reserved(i) for i in range(2)],
                       ground_truth_opened=False)
            records.append(rec)
            write('results.json', dict(complete=False, load_seconds=load_seconds,
                                       records=records, ground_truth_opened=False))
            if outcome != 'success':
                stop = True
                break
        image.close()
    write('results.json', dict(complete=True, load_seconds=load_seconds,
                               records=records, ground_truth_opened=False))
    write('status.json', dict(phase='complete', completed_budgets=len(records),
                              updated_utc=datetime.now(timezone.utc).isoformat()))


if __name__ == '__main__':
    try:
        if sys.argv[1:] == ['--fresh-author']:
            RUN = ROOT / 'runs/mvp_pixel_preflight_author_v1'
            BUDGETS = (4096 * 2160,)
        else:
            assert not sys.argv[1:], 'Only --fresh-author is supported'
        main()
    except BaseException:
        write('status.json', dict(phase='failed', error=traceback.format_exc()[-5000:],
                                  updated_utc=datetime.now(timezone.utc).isoformat()))
        raise
