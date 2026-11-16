"""Future GPU extraction of every original user-instruction prefill token.

This is a separate, not-yet-launched experiment from w2_name_token_control.py.
Use ``audit-cpu`` before registration. GPU stages require a frozen protocol and
an explicit launch release file; they are not invoked by importing this module.
"""
from __future__ import annotations

import os
os.environ.setdefault('PYTHONDONTWRITEBYTECODE', '1')
for _name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(_name, '2')

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]

import numpy as np
from PIL import Image

from clickagent.evidence_retention import STATE_DIM, require, state_features
from clickagent.target_token import TARGET_NAME_PROMPT

OUT = ROOT / 'runs/review_controls_v1/instruction_future'
V18 = ROOT / 'runs/full_scaling_v18'
V2 = ROOT / 'runs/paper_scale_ablation_v2'
MMB = ROOT / 'runs/mmbench_gui_new_source_v1'
RUNTIME = ROOT / 'runs/crossfit_candidates_v12/runtime_identity.json'
SOURCES = {
    'old_train': (ROOT / 'runs/multiregion_v13/data/train.jsonl', V18 / 'data/old_train_index.json'),
    'new_train': (V18 / 'data/new_train.jsonl', V18 / 'features/new_train/index.json'),
    'pro': (V2 / 'eval/pro/public.json', V2 / 'eval/pro/features/index.json'),
    'sheets': (V2 / 'eval/sheets/public.json', V2 / 'eval/sheets/features/index.json'),
    'mmbench': (MMB / 'public.json', MMB / 'mmbench/features/index.json'),
}
EXPECTED = {'old_train': 873, 'new_train': 12500, 'pro': 500, 'sheets': 500, 'mmbench': 1000}
PILOT_MANIFEST = OUT / 'pilot_manifest.json'
PILOT_CPU_AUDIT = OUT / 'pilot_cpu_audit.json'
PILOT_PROTOCOL = OUT / 'pilot_protocol.json'
PILOT_THRESHOLDS = dict(multi_vs_original_last_max_abs=1e-5,
                        forced_4d_vs_original_last_max_abs=1e-5,
                        attention_max_abs=5e-4,
                        hidden_max_abs=2e-2,
                        image_hidden_max_abs=2e-2)


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def digest(value):
    content = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(',', ':')).encode('utf-8')
    return hashlib.sha256(content).hexdigest()


def low_priority():
    if os.name == 'nt':
        import ctypes
        k = ctypes.windll.kernel32
        k.GetCurrentProcess.restype = ctypes.c_void_p
        k.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint)
        k.SetPriorityClass.restype = ctypes.c_int
        require(bool(k.SetPriorityClass(k.GetCurrentProcess(), 0x00004000)), 'Cannot set below-normal priority')


def public_rows(cohort):
    manifest, index = SOURCES[cohort]
    if manifest.suffix == '.jsonl':
        rows = [json.loads(line) for line in manifest.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    else:
        rows = read(manifest)
    obj = read(index)
    cases = obj if isinstance(obj, list) else obj['cases']
    require(len(rows) == len(cases) == EXPECTED[cohort], 'Public/index count mismatch')
    require(all(r['sample_id'] == e['sample_id'] for r, e in zip(rows, cases)), 'Public/index order mismatch')
    return rows, cases


def raw_views(entry):
    require(sha(entry['pipeline_path']) == entry['pipeline_sha256'], 'Frozen pipeline changed')
    pipe = read(entry['pipeline_path'])
    views = [dict(feature_path=pipe['global_feature_path'], feature_sha256=pipe['global_feature_sha256'],
                  transform=None)] + pipe['local_views']
    require(1 <= len(views) <= 4, 'Frozen view count changed')
    return views


def view_image(case, transform):
    require(sha(case['image']) == case['image_sha256'], 'Source image bytes changed')
    with Image.open(case['image']) as source:
        image = source.convert('RGB')
    require(list(image.size) == case['image_size'], 'Source dimensions changed')
    if transform is not None:
        crop = image.crop(tuple(transform['box'])).resize(tuple(transform['crop_size']), Image.Resampling.LANCZOS)
        image.close()
        image = crop
    return image


def prompt_inputs(processor, image, instruction):
    messages = [{'role': 'system', 'content': TARGET_NAME_PROMPT},
                {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': instruction}]}]
    prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[prompt], images=[image], return_tensors='pt', padding=False)
    return inputs


def text_positions(processor, inputs, instruction, image_token_id):
    """Find the literal instruction's token IDs after the last expanded image token.

    Ambiguous or changed tokenization is a hard error, never a case exclusion.
    """
    ids = inputs['input_ids'][0].tolist()
    needle = processor.tokenizer(instruction, add_special_tokens=False)['input_ids']
    images = [i for i, token in enumerate(ids) if token == image_token_id]
    require(images and needle, 'Image/instruction tokens missing')
    candidates = [i for i in range(images[-1] + 1, len(ids) - len(needle) + 1)
                  if ids[i:i + len(needle)] == needle]
    require(len(candidates) == 1, 'Instruction token subsequence absent or ambiguous')
    start = candidates[0]
    return list(range(start, start + len(needle))), images


def reconstruct_selected_rows(module, hidden_states, position_embeddings, key_states,
                              token_positions, attention_mask=None):
    """Qwen3-VL SDPA-equivalent causal QK rows for selected prefill tokens.

    The denominator includes every permitted KV token, not only image keys.
    Query rows are chunked by caller to bound memory. Input/output are torch.
    """
    import torch
    positions = torch.as_tensor(token_positions, device=hidden_states.device, dtype=torch.long)
    batch, total, _ = hidden_states.shape
    require(batch == 1 and positions.ndim == 1 and positions.numel() > 0, 'Query selection shape')
    require(int(positions.min()) >= 0 and int(positions.max()) < total, 'Query outside prefill')
    head_dim = module.head_dim
    selected = hidden_states.index_select(1, positions)
    query = module.q_norm(module.q_proj(selected).view(batch, len(positions), -1, head_dim)).transpose(1, 2)
    cos, sin = position_embeddings
    cos = cos.index_select(1, positions).unsqueeze(1)
    sin = sin.index_select(1, positions).unsqueeze(1)
    first, second = query[..., :head_dim // 2], query[..., head_dim // 2:]
    query = query * cos + torch.cat((-second, first), dim=-1) * sin
    kv_heads = key_states.shape[1]
    groups = module.num_key_value_groups
    require(query.shape[1] == kv_heads * groups, 'Grouped-query configuration changed')
    grouped = query.reshape(batch, kv_heads, groups, len(positions), head_dim).float()
    keys = key_states.to(query.device).float()
    scores = torch.einsum('bkgtd,bkld->bkgtl', grouped, keys).reshape(
        batch, kv_heads * groups, len(positions), keys.shape[-2]) * module.scaling
    if attention_mask is not None:
        if attention_mask.ndim == 4:
            row_mask = attention_mask.index_select(-2, positions)[..., :scores.shape[-1]]
        elif attention_mask.ndim == 2:
            row_mask = attention_mask[:, None, None, :scores.shape[-1]]
        else:
            raise ValueError('Unsupported Qwen attention mask rank')
        row_mask = row_mask.to(scores.device)
        scores = scores.masked_fill(~row_mask, -torch.inf) if row_mask.dtype == torch.bool else scores + row_mask.float()
    # Explicit causal mask guards against 2D padding-only attention masks.
    causal = torch.arange(scores.shape[-1], device=scores.device)[None, :] <= positions[:, None]
    scores = scores.masked_fill(~causal[None, None], -torch.inf)
    return torch.softmax(scores, dim=-1, dtype=torch.float32)


class InstructionRecorder:
    """Wrap the frozen original recorder; capture instruction and last-prefill rows."""
    def __init__(self, model, layers):
        from clickagent.conditioned_features import SpatialContentRecorder

        class Recorder(SpatialContentRecorder):
            def begin_instruction(self, image_positions, grid_shape, instruction_positions):
                self.instruction_positions = np.asarray(instruction_positions, dtype=np.int64)
                self.instruction_attention = {}
                self.instruction_hidden = {}
                self.last_via_multi = {}
                self.last_via_forced_4d = {}
                self.mask_specs = {}
                self.begin(image_positions, grid_shape)

            def _attention_hook(self, layer):
                parent = super()._attention_hook(layer)
                def hook(module, args, kwargs, output):
                    import torch
                    parent(module, args, kwargs, output)
                    hidden = kwargs.get('hidden_states', args[0] if args else None)
                    if hidden.shape[1] <= 1:
                        return
                    embedding = kwargs.get('position_embeddings', args[1] if len(args) > 1 else None)
                    cache = kwargs.get('past_key_values', kwargs.get('past_key_value'))
                    require(cache is not None and hasattr(cache, 'layers') and embedding is not None,
                            'Qwen prefill DynamicCache/position embeddings absent')
                    keys = cache.layers[module.layer_idx].keys
                    require(keys is not None, 'Qwen prefill KV keys absent')
                    image_ix = torch.as_tensor(self.image_positions, device=hidden.device, dtype=torch.long)
                    mask = kwargs.get('attention_mask')
                    self.mask_specs[layer] = (None if mask is None else
                        dict(shape=list(mask.shape), dtype=str(mask.dtype), rank=int(mask.ndim)))
                    last = reconstruct_selected_rows(module, hidden, embedding, keys,
                                                     [hidden.shape[1] - 1], mask)
                    self.last_via_multi[layer] = last[0].index_select(-1, image_ix)[:, 0].detach().cpu().numpy().copy()
                    # Exercise the 4D query-row branch with *real* Qwen Q/RoPE/KV.
                    # The last prompt query can attend every prefill key here.
                    length = hidden.shape[1]
                    forced_4d = torch.tril(torch.ones((length, length), dtype=torch.bool,
                                                        device=hidden.device))[None, None]
                    last_4d = reconstruct_selected_rows(module, hidden, embedding, keys,
                                                        [length - 1], forced_4d)
                    self.last_via_forced_4d[layer] = (
                        last_4d[0].index_select(-1, image_ix)[:, 0].detach().cpu().numpy().copy())
                    chunks = []
                    for start in range(0, len(self.instruction_positions), 16):
                        positions = self.instruction_positions[start:start + 16]
                        row = reconstruct_selected_rows(module, hidden, embedding, keys,
                                                        positions, kwargs.get('attention_mask'))
                        chunks.append(row[0].index_select(-1, image_ix).permute(1, 0, 2).detach().cpu().numpy())
                    self.instruction_attention[layer] = np.concatenate(chunks).copy()
                return hook

            def _hidden_hook(self, layer):
                parent = super()._hidden_hook(layer)
                def hook(module, args, output):
                    import torch
                    parent(module, args, output)
                    hidden = output[0] if isinstance(output, (tuple, list)) else output
                    if hidden.shape[1] <= 1:
                        return
                    ix = torch.as_tensor(self.instruction_positions, device=hidden.device, dtype=torch.long)
                    self.instruction_hidden[layer] = hidden[0].index_select(0, ix).detach().to(dtype=torch.float16).cpu().numpy().copy()
                return hook

        self.inner = Recorder(model, layers)
        self.layers = layers

    def __enter__(self):
        self.inner.__enter__()
        return self

    def __exit__(self, *exc):
        self.inner.__exit__(*exc)

    def begin(self, images, grid, instruction):
        self.inner.begin_instruction(images, grid, instruction)

    def snapshot(self):
        base = self.inner.snapshot()
        attention = np.concatenate([self.inner.instruction_attention[layer] for layer in self.layers], axis=1)
        hidden = np.concatenate([self.inner.instruction_hidden[layer] for layer in self.layers], axis=1)
        last_via_multi = np.concatenate([self.inner.last_via_multi[layer] for layer in self.layers], axis=0)
        last_via_forced_4d = np.concatenate([self.inner.last_via_forced_4d[layer] for layer in self.layers], axis=0)
        return dict(base=base, instruction_attention=attention, instruction_hidden=hidden,
                    image_hidden=self.inner.image_hidden, multi_last_attention=last_via_multi,
                    forced_4d_last_attention=last_via_forced_4d,
                    attention_mask_specs=self.inner.mask_specs)


def audit_cpu(limit=1):
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    low_priority()
    import torch
    from transformers import AutoProcessor
    require(not torch.cuda.is_available(), 'CPU audit may not see CUDA')
    contract = read(RUNTIME)['runtime_contract']
    require(contract['max_pixels'] == 262144 and contract['layers'] == [15, 28, 36], 'Model budget/layers changed')
    processor = AutoProcessor.from_pretrained(contract['model_path'], min_pixels=contract['min_pixels'],
                                              max_pixels=contract['max_pixels'], local_files_only=True)
    image_token_id = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    checks = []
    for cohort in SOURCES:
        rows, index = public_rows(cohort)
        for case, entry in list(zip(rows, index))[:limit]:
            for view_ix, view in enumerate(raw_views(entry)[:2]):
                require(sha(view['feature_path']) == view['feature_sha256'], 'Archived raw NPZ changed')
                with np.load(view['feature_path'], allow_pickle=False) as z:
                    meta = json.loads(str(z['metadata']))
                image = view_image(case, view['transform'])
                try:
                    inputs = prompt_inputs(processor, image, case['instruction'])
                    token_ix, image_ix = text_positions(processor, inputs, case['instruction'], image_token_id)
                    grid = inputs['image_grid_thw'][0].tolist()
                finally:
                    image.close()
                require(meta['system_prompt'] == TARGET_NAME_PROMPT and meta['input_tokens'] == inputs['input_ids'].shape[-1],
                        'Prompt/prefill length does not match archived forward')
                require(meta['grid_shape'] == [grid[1] // 2, grid[2] // 2] and
                        meta['image_size'] == list(view['transform']['crop_size'] if view_ix else case['image_size']),
                        'Frozen image grid/view size changed')
                require(token_ix[0] > image_ix[-1] and token_ix[-1] < len(inputs['input_ids'][0]) - 1,
                        'User instruction placement changed')
                checks.append(dict(cohort=cohort, sample_id=case['sample_id'], view=view_ix,
                                   instruction_tokens=len(token_ix), instruction_start=token_ix[0],
                                   image_tokens=len(image_ix), input_tokens=int(inputs['input_ids'].shape[-1])))
    receipt = dict(complete=True, no_gpu=True, no_labels_opened=True, checks=checks,
                   runtime_identity_sha256=sha(RUNTIME), script_sha256=sha(__file__), at_utc=now())
    write(OUT / 'cpu_audit.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


def build_pilot_manifest():
    """Select two public cases per cohort, each with global + first local view."""
    selected = []
    for cohort in SOURCES:
        cases, index = public_rows(cohort)
        chosen = 0
        for source_position, (case, entry) in enumerate(zip(cases, index)):
            views = raw_views(entry)
            if len(views) < 2:
                continue
            # Never copy a GT field into the pilot manifest.
            public = {k: case[k] for k in ('sample_id', 'image', 'image_size',
                                            'image_sha256', 'instruction')}
            require(sha(public['image']) == public['image_sha256'], 'Pilot screenshot bytes changed')
            require(entry['n'] >= 0 and sha(entry['path']) == entry['sha256'] and
                    sha(entry['state_path']) == entry['state_sha256'] and
                    sha(entry['mask_path']) == entry['mask_sha256'],
                    'Original candidate/state cache changed')
            with np.load(entry['path'], allow_pickle=False) as z:
                points, baseline, active = (z[k].copy() for k in ('points', 'baseline', 'active'))
            old_mask = np.load(entry['mask_path'], allow_pickle=False)
            old_state = np.load(entry['state_path'], mmap_mode='r', allow_pickle=False)
            require(points.shape == (entry['n'], 2) and active.shape == (entry['n'], 4) and
                    old_mask.shape[:2] == (entry['n'], 4) and old_state.shape[:2] == (entry['n'], 4),
                    'Original candidate/view arrays changed')
            require(len(baseline) <= 16 and len(set(baseline.tolist())) == len(baseline) and
                    all(0 <= ix < len(points) for ix in baseline), 'Original K16 changed')
            for view_index in (0, 1):
                view = views[view_index]
                require(sha(view['feature_path']) == view['feature_sha256'], 'Original raw NPZ SHA changed')
                with np.load(view['feature_path'], allow_pickle=False) as z:
                    meta = json.loads(str(z['metadata']))
                    arrays = {k: z[k].copy() for k in ('target_attention', 'target_hidden', 'image_hidden')}
                    require(z['prefill_attention'].shape[0] == 96 and
                            z['prefill_hidden'].shape == (12288,), 'Archived prefill dimensions changed')
                # Historic raw NPZs record sample/image/prompt identity, but not
                # a separate instruction hash; the pinned public manifest supplies it.
                require(meta['sample_id'] == public['sample_id'] and
                        meta['image_sha256'] == public['image_sha256'] and
                        meta.get('transform') == view['transform'] and
                        meta['system_prompt'] == TARGET_NAME_PROMPT,
                        'Archived prompt/image/transform identity changed')
                present = old_mask[:, view_index].any(-1)
                original_size = public['image_size']
                rebuilt, valid = state_features(arrays, meta, points, original_size, view['transform'])
                require(np.array_equal(valid, present) or not present.any(), 'Original view mask changed')
                original = np.asarray(old_state[:, view_index, :rebuilt.shape[1]], np.float16)
                require(original.shape == rebuilt.shape and np.array_equal(original, rebuilt),
                        'Original name-token candidate state does not reconstruct exactly')
                if view_index == 1:
                    require(view['transform']['original_size'] == public['image_size'] and
                            view['transform']['ground_truth_used'] is False,
                            'Local crop differs from archived public crop')
                selected.append(dict(
                    cohort=cohort, sample_id=public['sample_id'], source_position=source_position,
                    public=public, source_entry_sha256=digest(entry),
                    candidate_path=entry['path'], candidate_sha256=entry['sha256'],
                    original_state_path=entry['state_path'], original_state_sha256=entry['state_sha256'],
                    original_mask_path=entry['mask_path'], original_mask_sha256=entry['mask_sha256'],
                    pipeline_path=entry['pipeline_path'], pipeline_sha256=entry['pipeline_sha256'],
                    view_index=view_index, view=view, candidate_count=int(entry['n']),
                    K16_count=int(len(baseline)), original_target_tokens=int(rebuilt.shape[1]),
                    original_name_state_exact_parity=True, original_valid_candidate_views=int(present.sum()),
                    archived_input_tokens=int(meta['input_tokens']), archived_grid_shape=meta['grid_shape']))
            chosen += 1
            if chosen == 2:
                break
        require(chosen == 2, 'Need two cases with a local view in cohort ' + cohort)
    require(len(selected) == 20 and len({(v['cohort'], v['sample_id'], v['view_index']) for v in selected}) == 20,
            'Pilot manifest must contain exactly 20 unique views')
    return dict(complete=True, public_only=True, labels_opened=False,
                selection_rule='For each of five frozen cohorts, first two index-ordered cases with >=1 local view; global and first local view per case.',
                views=selected, runtime_identity_sha256=sha(RUNTIME),
                source_sha256={str(path): sha(path) for pair in SOURCES.values() for path in pair})


def audit_pilot_cpu(manifest):
    """Check the exact 20 registered views with the original local processor."""
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    low_priority()
    import torch
    from transformers import AutoProcessor
    require(not torch.cuda.is_available(), 'CPU pilot audit may not access CUDA')
    contract = read(RUNTIME)['runtime_contract']
    require(contract['max_pixels'] == 262144 and contract['layers'] == [15, 28, 36],
            'Original visual budget/layers changed')
    processor = AutoProcessor.from_pretrained(contract['model_path'], min_pixels=contract['min_pixels'],
                                              max_pixels=contract['max_pixels'], local_files_only=True)
    image_token_id = processor.tokenizer.convert_tokens_to_ids('<|image_pad|>')
    checks = []
    for item in manifest['views']:
        case, view = item['public'], item['view']
        image = view_image(case, view['transform'])
        try:
            inputs = prompt_inputs(processor, image, case['instruction'])
            instruction_positions, image_positions = text_positions(
                processor, inputs, case['instruction'], image_token_id)
            grid = inputs['image_grid_thw'][0].tolist()
        finally:
            image.close()
        require(int(inputs['input_ids'].shape[-1]) == item['archived_input_tokens'] and
                [grid[1] // 2, grid[2] // 2] == item['archived_grid_shape'] and
                instruction_positions[0] > image_positions[-1] and
                instruction_positions[-1] < int(inputs['input_ids'].shape[-1]) - 1,
                'Pilot token span or archived prompt/grid parity failed')
        checks.append(dict(cohort=item['cohort'], sample_id=item['sample_id'],
                           view_index=item['view_index'], instruction_tokens=len(instruction_positions),
                           instruction_start=instruction_positions[0], image_tokens=len(image_positions),
                           prompt_tokens=int(inputs['input_ids'].shape[-1])))
    require(len(checks) == 20, 'CPU pilot audit did not cover all views')
    return dict(complete=True, no_gpu=True, no_labels_opened=True,
                manifest_sha256=sha(PILOT_MANIFEST), script_sha256=sha(__file__),
                runtime_identity_sha256=sha(RUNTIME), checks=checks, at_utc=now())


def register_pilot():
    """CPU-only immutable registration; never inspect evaluation target labels."""
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    low_priority()
    manifest = build_pilot_manifest()
    if PILOT_MANIFEST.exists():
        require(read(PILOT_MANIFEST) == manifest, '20-view pilot manifest changed')
    else:
        write(PILOT_MANIFEST, manifest)
    audit = audit_pilot_cpu(manifest)
    if PILOT_CPU_AUDIT.exists():
        old = read(PILOT_CPU_AUDIT)
        audit['at_utc'] = old['at_utc']
        require(old == audit, '20-view CPU pilot audit changed')
    else:
        write(PILOT_CPU_AUDIT, audit)
    sources = [Path(__file__), ROOT / 'src/clickagent/evidence_retention.py',
               ROOT / 'src/clickagent/conditioned_features.py', ROOT / 'src/clickagent/action_model.py',
               ROOT / 'src/clickagent/model_features.py', ROOT / 'src/clickagent/target_token.py',
               ROOT / 'src/clickagent/public_model.py']
    protocol = dict(registered=True, registered_at_utc=now(),
                    purpose='Exactly 20 label-free real-Qwen prefills to validate multi-row instruction QK capture; no full extraction permitted.',
                    script_sha256=sha(__file__), source_sha256={str(path): sha(path) for path in sources},
                    runtime_identity_sha256=sha(RUNTIME), pilot_manifest_sha256=sha(PILOT_MANIFEST),
                    pilot_cpu_audit_sha256=sha(PILOT_CPU_AUDIT),
                    public_manifest_and_index_sha256=manifest['source_sha256'],
                    view_count=20, visual_pixel_budget=262144, layers=[15, 28, 36],
                    thresholds=PILOT_THRESHOLDS,
                    checks=['exact prompt/input/grid identity', 'exact original candidate K16 and name-state replay',
                            'selected multi-row last prompt versus original last-row 96xP pointwise',
                            'real Qwen Q/RoPE/KV with forced 4D causal mask versus original last-row',
                            'archived last-prefill attention/hidden/image-content replay',
                            'finite instruction token states and derived candidate evidence'],
                    release='Requires pilot_launch_release.json with matching protocol SHA and explicit statement that the 1M run is complete and its process exited.',
                    labels_opened_before_or_during_pilot=False,
                    after_pilot='Seal all 20 per-view artifacts and SHA receipt; inspect numerical parity and freeze a separate full-extraction protocol. The full-extract entry remains disabled here.')
    if PILOT_PROTOCOL.exists():
        old = read(PILOT_PROTOCOL)
        protocol['registered_at_utc'] = old['registered_at_utc']
        require(old == protocol, 'Pilot protocol changed after registration')
    else:
        write(PILOT_PROTOCOL, protocol)
    print(json.dumps(dict(pilot_registered=True, views=20, manifest_sha256=sha(PILOT_MANIFEST),
                          cpu_audit_sha256=sha(PILOT_CPU_AUDIT), protocol_sha256=sha(PILOT_PROTOCOL),
                          labels_opened=False, gpu_used=False), ensure_ascii=False), flush=True)


def gpu_runtime():
    import torch
    from clickagent.model_features import ExtractionConfig
    from clickagent.conditioned_features import ConditionedFeatureRuntime
    require(torch.cuda.is_available(), 'GPU stage requested without CUDA')
    release_path = OUT / 'pilot_launch_release.json'
    protocol_path = OUT / 'pilot_protocol.json'
    require(release_path.exists() and protocol_path.exists(), 'No registered pilot or explicit GPU launch release')
    release, protocol = read(release_path), read(protocol_path)
    require(release['scope'] == '20_view_pilot_only' and
            release['one_million_run_complete_and_process_exited'] is True and
            release['pilot_protocol_sha256'] == sha(protocol_path), 'GPU pilot release invalid')
    require(protocol['registered'] and protocol['script_sha256'] == sha(__file__) and
            protocol['runtime_identity_sha256'] == sha(RUNTIME), 'Future protocol absent or changed')
    contract = read(RUNTIME)['runtime_contract']
    for weight in read(RUNTIME)['actual_model_files']:
        stat = Path(weight['path']).stat()
        require(stat.st_size == weight['bytes'] and stat.st_mtime_ns == weight['mtime_ns'],
                'Frozen Qwen weight inventory changed')
    cfg = dict(contract)
    cfg['layers'] = tuple(cfg['layers'])
    cfg['conditions'] = tuple(cfg['conditions'])
    cfg['max_memory'] = {int(k) if str(k).isdigit() else k: v for k, v in cfg['max_memory'].items()}
    cfg['offload_folder'] = str(OUT / 'offload')
    return ConditionedFeatureRuntime(ExtractionConfig(**cfg))


def extract_one_view(runtime, case, view, points, source_present):
    import torch
    require(sha(view['feature_path']) == view['feature_sha256'], 'Archived raw NPZ changed')
    with np.load(view['feature_path'], allow_pickle=False) as z:
        old_meta = json.loads(str(z['metadata']))
        old_prefill_attention = z['prefill_attention'].copy()
        old_prefill_hidden = z['prefill_hidden'].copy()
        old_image_hidden = z['image_hidden'].copy()
    image = view_image(case, view['transform'])
    try:
        with runtime._lock, torch.inference_mode():
            inputs, image_positions, grid = runtime._prepare_image(image, case['instruction'], TARGET_NAME_PROMPT)
            positions, _ = text_positions(runtime.processor, inputs, case['instruction'], runtime.model.config.image_token_id)
            with InstructionRecorder(runtime.model, runtime.config.layers) as recorder:
                recorder.begin(image_positions, grid, positions)
                result = runtime.model(**inputs, use_cache=True, logits_to_keep=1,
                                       output_attentions=False, output_hidden_states=False, return_dict=True)
                capture = recorder.snapshot()
                del result
    finally:
        image.close()
    require(old_meta['input_tokens'] == int(inputs['input_ids'].shape[-1]) and
            old_meta['system_prompt'] == TARGET_NAME_PROMPT and old_meta['grid_shape'] == list(grid),
            'Archived prefill contract changed')
    # Exact floating-point equality is intentionally checked in the pilot first;
    # the final tolerance must be registered before cohort extraction.
    old = capture['base']
    parity = dict(attention_max_abs=float(np.max(np.abs(old['raw_image_attention'] - old_prefill_attention))),
                  multi_vs_original_last_max_abs=float(np.max(np.abs(capture['multi_last_attention'] -
                                                                 old['raw_image_attention']))),
                  forced_4d_vs_original_last_max_abs=float(np.max(np.abs(capture['forced_4d_last_attention'] -
                                                                     old['raw_image_attention']))),
                  hidden_max_abs=float(np.max(np.abs(old['hidden'].astype(np.float16).astype(np.float32) -
                                                    old_prefill_hidden.astype(np.float32)))),
                  image_hidden_max_abs=float(np.max(np.abs(capture['image_hidden'].astype(np.float32) -
                                                          old_image_hidden.astype(np.float32)))))
    parity['attention_mask_specs'] = capture['attention_mask_specs']
    meta = dict(grid_shape=list(grid), image_size=list(case['image_size']))
    f, valid = state_features(dict(target_attention=capture['instruction_attention'],
                                   target_hidden=capture['instruction_hidden'],
                                   image_hidden=old_image_hidden.copy()), meta, points,
                              case['image_size'], view['transform'])
    require(np.array_equal(valid, source_present) or not source_present.any(), 'Candidate/view mask changed')
    return f, parity, positions, capture


def verify_pilot_registration():
    require(PILOT_PROTOCOL.exists() and PILOT_MANIFEST.exists() and PILOT_CPU_AUDIT.exists(),
            '20-view pilot is not registered')
    protocol, manifest, audit = read(PILOT_PROTOCOL), read(PILOT_MANIFEST), read(PILOT_CPU_AUDIT)
    require(protocol['registered'] and protocol['script_sha256'] == sha(__file__) and
            protocol['pilot_manifest_sha256'] == sha(PILOT_MANIFEST) and
            protocol['pilot_cpu_audit_sha256'] == sha(PILOT_CPU_AUDIT) and
            protocol['runtime_identity_sha256'] == sha(RUNTIME) and
            protocol['view_count'] == len(manifest['views']) == len(audit['checks']) == 20 and
            protocol['thresholds'] == PILOT_THRESHOLDS and audit['no_gpu'] and audit['no_labels_opened'],
            'Pilot protocol, CPU audit or script changed')
    for path, expected in protocol['source_sha256'].items():
        require(sha(path) == expected, 'Pilot scientific source changed: ' + path)
    for path, expected in protocol['public_manifest_and_index_sha256'].items():
        require(sha(path) == expected, 'Pilot public manifest/index changed: ' + path)
    for item in manifest['views']:
        for key in ('candidate', 'original_state', 'original_mask', 'pipeline'):
            require(sha(item[f'{key}_path']) == item[f'{key}_sha256'],
                    'Pilot selected original file changed: ' + key)
        view = item['view']
        require(sha(view['feature_path']) == view['feature_sha256'] and
                sha(item['public']['image']) == item['public']['image_sha256'] and
                item['original_name_state_exact_parity'] is True,
                'Pilot selected raw view/image/state provenance changed')
    return protocol, manifest


def pilot_key(item):
    text = f"{item['cohort']}::{item['sample_id']}::{item['view_index']}"
    return hashlib.sha256(text.encode()).hexdigest()[:24]


def save_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as f:
        np.savez_compressed(f, **arrays)
    os.replace(temporary, path)


def record_pilot_failure(index, item, error, parity=None):
    fail = dict(complete=False, stage='pilot-gpu', view_order=index,
                cohort=None if item is None else item['cohort'],
                sample_id=None if item is None else item['sample_id'],
                view_index=None if item is None else item['view_index'],
                error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc(),
                measured_parity=parity,
                pilot_protocol_sha256=sha(PILOT_PROTOCOL) if PILOT_PROTOCOL.exists() else None,
                no_labels_opened=True, at_utc=now())
    key = hashlib.sha256(f'{fail["at_utc"]}:{index}:{fail["sample_id"]}'.encode()).hexdigest()[:16]
    write(OUT / 'pilot/failures' / f'{key}.json', fail)
    write(OUT / 'pilot/status.json', dict(stage='failed', failure_receipt=f'pilot/failures/{key}.json',
                                          completed_before_failure=index, at_utc=now()))


def pilot_gpu():
    """Only the registered 20-view parity pilot; no training or full extraction."""
    protocol, manifest = verify_pilot_registration()
    runtime = None
    current_item = None
    current_index = 0
    parity = None
    try:
        runtime = gpu_runtime()
        case_hashes = {}
        array_hashes = {}
        numerical = []
        for current_index, current_item in enumerate(manifest['views'], 1):
            key = pilot_key(current_item)
            result_path = OUT / 'pilot/cases' / f'{key}.json'
            array_path = OUT / 'pilot/arrays' / f'{key}.npz'
            if result_path.exists():
                result = read(result_path)
                require(result['complete'] and result['parity_passed'] and
                        result['pilot_protocol_sha256'] == sha(PILOT_PROTOCOL) and
                        result['input_identity'] == digest(current_item) and
                        result['array_sha256'] == sha(array_path),
                        'Existing pilot case is incomplete or drifted')
            else:
                with np.load(current_item['candidate_path'], allow_pickle=False) as z:
                    points, baseline = z['points'].copy(), z['baseline'].copy()
                mask = np.load(current_item['original_mask_path'], allow_pickle=False)
                require(points.shape == (current_item['candidate_count'], 2) and
                        len(baseline) == current_item['K16_count'], 'Pilot K16/base changed')
                valid = mask[:, current_item['view_index']].any(-1)
                started = time.perf_counter()
                state, parity, positions, capture = extract_one_view(
                    runtime, current_item['public'], current_item['view'], points, valid)
                require(state.shape == (len(points), len(positions), STATE_DIM) and
                        capture['instruction_attention'].shape == (len(positions), 96,
                            int(np.prod(current_item['archived_grid_shape']))) and
                        capture['instruction_hidden'].shape == (len(positions), 12288) and
                        np.isfinite(state).all(), 'Captured instruction state shape/finite check failed')
                require(set(map(int, capture['attention_mask_specs'])) == {15, 28, 36},
                        'Selected-layer attention-mask diagnostics incomplete')
                for spec in capture['attention_mask_specs'].values():
                    require(spec is None or spec['rank'] in (2, 4),
                            'Actual Qwen attention mask rank unsupported')
                for metric, bound in PILOT_THRESHOLDS.items():
                    require(np.isfinite(parity[metric]) and parity[metric] <= bound,
                            f'Pilot parity {metric}={parity[metric]} exceeds frozen {bound}')
                save_npz(array_path,
                         instruction_attention=capture['instruction_attention'].astype(np.float32),
                         instruction_hidden=capture['instruction_hidden'].astype(np.float16),
                         derived_candidate_state=state.astype(np.float16),
                         original_last_attention=capture['base']['raw_image_attention'].astype(np.float32),
                         selected_multi_last_attention=capture['multi_last_attention'].astype(np.float32),
                         forced_4d_last_attention=capture['forced_4d_last_attention'].astype(np.float32))
                result = dict(complete=True, parity_passed=True,
                              cohort=current_item['cohort'], sample_id=current_item['sample_id'],
                              view_index=current_item['view_index'], input_identity=digest(current_item),
                              pilot_protocol_sha256=sha(PILOT_PROTOCOL),
                              array_path=str(array_path), array_sha256=sha(array_path),
                              instruction_token_positions=list(map(int, positions)),
                              K16_count=current_item['K16_count'],
                              original_name_state_exact_parity=True,
                              measured_parity=parity, elapsed_seconds=round(time.perf_counter() - started, 4),
                              labels_opened=False)
                write(result_path, result)
            case_hashes[key] = sha(result_path)
            array_hashes[key] = result['array_sha256']
            numerical.append(result['measured_parity'])
            write(OUT / 'pilot/status.json', dict(stage='running', completed=current_index, total=20,
                                                  latest_cohort=current_item['cohort'],
                                                  latest_sample_id=current_item['sample_id'],
                                                  at_utc=now()))
        summary = {metric: max(float(item[metric]) for item in numerical) for metric in PILOT_THRESHOLDS}
        completion = dict(complete=True, views=20, parity_passed=True, labels_opened=False,
                          pilot_protocol_sha256=sha(PILOT_PROTOCOL),
                          pilot_manifest_sha256=sha(PILOT_MANIFEST),
                          gpu_release_sha256=sha(OUT / 'pilot_launch_release.json'),
                          case_hashes=case_hashes, array_hashes=array_hashes,
                          max_absolute_parity_error=summary, at_utc=now(),
                          full_extraction_authorized=False)
        write(OUT / 'pilot/completion.json', completion)
        write(OUT / 'pilot/status.json', dict(stage='complete', completed=20, total=20,
                                              completion_sha256=sha(OUT / 'pilot/completion.json'), at_utc=now()))
        print(json.dumps(dict(pilot_complete=True, views=20, completion_sha256=sha(OUT / 'pilot/completion.json'),
                              max_absolute_parity_error=summary), ensure_ascii=False), flush=True)
    except BaseException as error:
        record_pilot_failure(current_index, current_item, error, parity)
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--stage', required=True, choices=('audit-cpu', 'register-pilot', 'pilot-gpu', 'extract-gpu'))
    p.add_argument('--cohort', choices=tuple(SOURCES))
    p.add_argument('--limit', type=int, default=1)
    args = p.parse_args()
    if args.stage == 'audit-cpu':
        require(1 <= args.limit <= 20, 'CPU audit limit is 1..20')
        audit_cpu(args.limit)
    elif args.stage == 'register-pilot':
        register_pilot()
    elif args.stage == 'pilot-gpu':
        pilot_gpu()
    else:
        raise RuntimeError('Full extraction remains disabled; the 20-view GPU pilot must pass before a separate protocol is registered')


if __name__ == '__main__':
    main()
