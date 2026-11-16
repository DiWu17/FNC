"""CPU-only audit of Qwen3-VL image pixel sizing under MVP's public input path."""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from clickagent.advantage_runtimes import configure_paths

configure_paths('mvp')
from PIL import Image
from qwen_vl_utils import process_vision_info, smart_resize
from transformers import AutoProcessor


def main():
    path = ROOT / 'runs/paper_scale_ablation_v2/pointerbench/public.jsonl'
    row = json.loads(path.open(encoding='utf-8-sig').readline())
    with Image.open(row['image']) as opened:
        source = opened.convert('RGB')
    probe = source.crop((0, 0, 1024, 728)).resize((2048, 1456))
    records = []
    for budget in (1048576, 8847360):
        p = AutoProcessor.from_pretrained(ROOT / 'models/Qwen3-VL-8B-Instruct',
                                          min_pixels=3136, max_pixels=budget,
                                          local_files_only=True)
        h, w = smart_resize(probe.height, probe.width,
                            factor=p.image_processor.patch_size * p.image_processor.merge_size,
                            min_pixels=p.image_processor.min_pixels,
                            max_pixels=p.image_processor.max_pixels)
        resized = probe.resize((w, h))
        msg = [dict(role='user', content=[dict(type='image', image=resized),
                                          dict(type='text', text=row['instruction'])])]
        image_inputs, _ = process_vision_info(msg)
        prepared = p.image_processor(images=image_inputs, return_tensors='pt')
        grid = prepared['image_grid_thw'].tolist()
        records.append(dict(max_pixels=budget, processor_size=p.image_processor.size,
                            processor_max_pixels=p.image_processor.max_pixels,
                            manual_resize=[w, h], after_qwen_utils=list(image_inputs[0].size),
                            processor_grid=grid,
                            processor_pixels=sum(g[1] * g[2] * p.image_processor.patch_size**2
                                                 for g in grid)))
    dest = ROOT / 'runs/mvp_pixel_preflight_v1/processor_resize_audit.json'
    dest.write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(records, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()
