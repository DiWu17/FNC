"""Frozen Qwen3-VL adaptation of TAG, extracted from the validated timing pilot.

TAG prompt, per-token top-10 heads by image attention mass, token averaging,
threshold 0.5 and official connected regions. Qwen native visual raster replaces
MiniCPM resampler projection. No learned readout, candidates or labels.
"""
from __future__ import annotations
import gc
import importlib.util
from pathlib import Path
import re
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'third_party/env_v17/common'), str(ROOT / 'third_party/TAG')]
import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from core_utils.tag import AttentionDrivenGrounding


def token_span(tokenizer, ids, start, end):
    """Align a decoded character span to actual emitted token IDs, not re-encoding."""
    bounds = [0] + [len(tokenizer.decode(ids[:i], skip_special_tokens=False)) for i in range(1, len(ids) + 1)]
    return [i for i in range(len(ids)) if bounds[i] < end and bounds[i + 1] > start]


class TagQwenRuntime:
    def __init__(self):
        spec = importlib.util.spec_from_file_location('tag_qwen_prompt', ROOT / 'third_party/TAG/eval_mm/screenspot/prompt_tag.py')
        self.prompts = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.prompts)
        path = ROOT / 'models/Qwen3-VL-8B-Instruct'
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            path, dtype=torch.float16, device_map='balanced', max_memory={0: '12GiB', 1: '12GiB'},
            attn_implementation='eager', local_files_only=True).eval().requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(path, min_pixels=16384, max_pixels=262144, local_files_only=True)
        self.device = self.model.get_input_embeddings().weight.device
        self.tag = AttentionDrivenGrounding(self.processor.tokenizer, topk_layer_head=10, filter_thresold=0.5, vis_flag=False)

    @staticmethod
    def sync():
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)

    def infer(self, row):
        gc.collect()
        self.sync()
        for d in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(d)
        started = time.perf_counter()
        with Image.open(row['image']) as im:
            image = im.convert('RGB')
        prompt = self.prompts.query_prompt.format(query_text=row['instruction'], interested_key=self.prompts.interested_key)
        messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': prompt}]}]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors='pt').to(self.device)
        input_ids = inputs['input_ids'][0].tolist()
        input_length = len(input_ids)
        image_positions = [i for i, token in enumerate(input_ids) if token == self.model.config.image_token_id]
        grid = inputs['image_grid_thw'][0].tolist()
        gh, gw = (grid[1] // 2, grid[2] // 2)
        assert grid[0] == 1 and len(image_positions) == gh * gw
        processed_pixels = grid[1] * grid[2] * 16 * 16
        self.sync()
        prepared = time.perf_counter()
        with torch.inference_mode():
            out = self.model.generate(**inputs, max_new_tokens=1024, do_sample=False, use_cache=True, output_attentions=True, return_dict_in_generate=True)
        self.sync()
        generated_at = time.perf_counter()
        ids = out.sequences[0, input_length:].tolist()
        response = self.processor.tokenizer.decode(ids, skip_special_tokens=False)
        match = re.search('"element_content_or_description"\\s*:\\s*"((?:\\\\.|[^"\\\\])*)"', response)
        selected = token_span(self.processor.tokenizer, ids, *match.span(1)) if match else []
        own_steps = [j + 1 for j in selected if j + 1 < len(out.attentions)]
        fallback = not own_steps
        if fallback:
            decoded = self.processor.tokenizer.decode(input_ids, skip_special_tokens=False)
            start = decoded.find(row['instruction'])
            if start < 0:
                raise RuntimeError('Cannot align instruction fallback')
            prompt_indices = token_span(self.processor.tokenizer, input_ids, start, start + len(row['instruction']))
            rows_to_read = [(out.attentions[0], j) for j in prompt_indices]
        else:
            rows_to_read = [(out.attentions[j], -1) for j in own_steps]
        maps = []
        for layers, query_position in rows_to_read:
            if len(layers) != 36:
                raise RuntimeError(f'Incomplete layer capture: {len(layers)}')
            per_layer = []
            for attention in layers:
                if attention is None or attention.shape[1] != 32:
                    raise RuntimeError('Attention output missing or wrong head count')
                idx = torch.tensor(image_positions, device=attention.device)
                per_layer.append(attention[0, :, query_position, :].index_select(-1, idx).detach().float().cpu().numpy())
            heads = np.concatenate(per_layer, axis=0)
            best = np.argsort(-heads.sum(-1), kind='stable')[:10]
            maps.append(heads[best].mean(0))
        if not maps:
            raise RuntimeError('No description or instruction maps')
        grid_map = np.stack(maps).mean(0).reshape(gh, gw).T
        peak = float(grid_map.max())
        point = None
        if peak > 0:
            expanded = np.kron(grid_map / peak, np.ones((32, 32)))
            mask = expanded * (expanded >= 0.5)
            _, boxes = self.tag.find_connected_regions_with_confidence(mask)
            boxes = self.tag.filter_bbox(boxes, 0.5)
            if boxes:
                box, confidence = max(boxes, key=lambda b: (b[1], (b[0][2] - b[0][0]) * (b[0][3] - b[0][1])))
                point = [round((box[0] + box[2]) / 2 * image.width / (gw * 32)), round((box[1] + box[3]) / 2 * image.height / (gh * 32))]
        self.sync()
        elapsed = time.perf_counter() - started
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        result = {'sample_id': row['sample_id'], 'suite': row['suite'], 'elapsed_seconds': elapsed, 'preprocess_seconds': prepared - started, 'generation_seconds': generated_at - prepared, 'postprocess_seconds': time.perf_counter() - generated_at, 'generated_tokens': len(ids), 'terminated_by_eos': bool(ids and ids[-1] in eos), 'selected_description_tokens': len(own_steps), 'instruction_fallback': fallback, 'processed_pixels': processed_pixels, 'point': point, 'generated_text': response, 'grid_hw': [gh, gw], 'labels_opened': False, 'peak_allocated_bytes': [torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count())]}
        del out, inputs, maps, rows_to_read
        return result
