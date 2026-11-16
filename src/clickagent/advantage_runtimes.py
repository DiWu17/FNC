"""Public-image/instruction adapters. No adapter accepts target annotations."""
from __future__ import annotations

from contextlib import contextmanager, redirect_stdout, redirect_stderr
from dataclasses import replace
import importlib
import ast
import io
import json
import math
from pathlib import Path
import re
import sys
import types

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def configure_paths(method):
    common = ROOT / 'third_party/env_v17/common'
    paths = [common]
    if method == 'tag':
        paths[:0] = [ROOT / 'third_party/env_v17/tag', ROOT / 'third_party/TAG']
    elif method.startswith('aima'):
        paths[:0] = [ROOT / 'third_party/env_v17/aima', ROOT / 'third_party/GUI-AIMA/src']
    elif method in ['mvp', 'mvp_single']:
        paths[:0] = [ROOT / 'third_party/env_v17/mvp', ROOT / 'third_party/MVP']
    sys.path[:0] = [str(p) for p in paths]


class RuntimeCounter:
    def __init__(self):
        self.forward_calls, self.processed_pixels = 0, 0
        self.handles = []

    def attach(self, model, patch_size):
        def before(module, args, kwargs):
            self.forward_calls += 1
            grid = kwargs.get('image_grid_thw')
            if grid is not None and kwargs.get('pixel_values') is not None:
                self.processed_pixels += int((grid[:, 1] * grid[:, 2]).sum()) * patch_size ** 2
        self.handles.append(model.register_forward_pre_hook(before, with_kwargs=True))

    def snapshot(self):
        return dict(backbone_forward_calls=self.forward_calls, processed_pixels=self.processed_pixels)


class OurRuntime:
    def __init__(self, folder, max_pixels=262144, seeds=(1, 2, 3)):
        if max_pixels != 262144:
            raise ValueError('Adopted method image budget must remain fixed')
        import torch
        sys.path.insert(0, str(ROOT / 'scripts'))
        import run_quality_readout_v15 as base
        import run_contribution_v16 as contribution
        from .conditioned_features import ConditionedFeatureRuntime
        from .evidence_retention import RetentionModel
        legacy = base.Extractor()
        config = replace(legacy.config, offload_folder=str(Path(folder) / 'offload'))
        self.proposer, self.base, self.contribution = legacy.proposer.requires_grad_(False), base, contribution
        self.runtime = ConditionedFeatureRuntime(config)
        self.models = []
        for i in seeds:
            p = ROOT / f'runs/quality_readout_v15_3k/models/expanded_s{i}.pt'
            saved = torch.load(p, map_location='cpu', weights_only=True)
            model = RetentionModel().to('cuda:0').eval().requires_grad_(False)
            model.load_state_dict(saved['state_dict'])
            self.models.append((model, saved))
        self.counter = RuntimeCounter()
        self.counter.attach(self.runtime.model, 16)
        self.metadata = dict(method='our_full', mainline='quality_readout_v15_3k',
                             feature_storage='In-memory; same extraction/proposal/state/readout mathematics',
                             model_metadata=self.runtime.runtime_metadata, scorer_device='cuda:0',
                             scorer_seeds=list(seeds), primary_timing_seed='S1', max_pixels=max_pixels,
                             readout_parameters_per_model=22050, proposer_parameters=611919)

    def infer(self, image, instruction):
        import torch
        from .evidence_retention import STATE_DIM
        image = image.convert('RGB')
        size = list(image.size)
        view_data = []
        arrays, meta = self.runtime.infer_spatial_features(image, instruction)
        view_data.append((arrays, meta, None))
        global_result = self.base.predict_raw(self.proposer, arrays, meta, size, topk=64)
        local = []
        for crop in self.base.plan_multiregion_crops(global_result, size):
            transform = crop['transform']
            with image.crop(transform['box']).resize(tuple(transform['crop_size']), Image.Resampling.LANCZOS) as cropped:
                arrays, meta = self.runtime.infer_spatial_features(cropped, instruction)
            decoded = self.base.predict_raw(self.proposer, arrays, meta, transform['crop_size'], topk=64)
            view_data.append((arrays, meta, transform))
            local.append(dict(region_index=crop['region_index'], result=decoded, transform=transform))
        pipe = dict(global_result=global_result, local_views=local,
                    pools=self.base.build_c1_c3_pools(global_result, local))
        b = self.base.build_base(pipe, size)
        if getattr(self, 'proposal_only', False):
            return dict(point=None, candidates=b['points'][np.sort(b['baseline'])].tolist(),
                        views=len(view_data), ground_truth_in_model_input=False)
        features, masks = [], []
        for arrays, meta, transform in view_data:
            f, m = self.base.state_features(arrays, meta, b['points'], size, transform)
            features.append(f)
            masks.append(m)
        tmax = max(1, *(f.shape[1] for f in features))
        state = np.zeros((len(b['points']), 4, tmax, STATE_DIM), np.float16)
        mask = np.zeros((len(b['points']), 4, tmax), bool)
        for j, (f, m) in enumerate(zip(features, masks)):
            state[:, j, :f.shape[1]] = f
            mask[:, j, :f.shape[1]] = m[:, None]
        record = dict(b, sample_id='public_runtime', n=len(b['points']), state=state, mask=mask)
        points = []
        with torch.inference_mode():
            for model, saved in self.models:
                if len(b['points']):
                    inputs, _ = self.contribution.build_inputs([record], saved['mean'].numpy(), saved['std'].numpy(), 'aligned', 'cuda:0')
                    scores = model(**inputs).float().cpu().numpy()
                    pool = np.sort(b['baseline'])
                    order = pool[np.argsort(-scores[pool], kind='stable')]
                    point = b['points'][order[0]].tolist() if len(order) else None
                else:
                    point = None
                points.append(point)
        # Raw public-only intermediates support development parity; no geometry is used.
        self.last_record = record
        self.last_pipe = pipe
        return dict(point=points[0], seed_points=points, views=len(view_data),
                    candidates=b['points'][b['baseline']].tolist(),
                    ground_truth_in_model_input=False)


class OurGUIAdapter(OurRuntime):
    def __init__(self, folder, max_pixels=262144):
        super().__init__(folder, max_pixels, seeds=())
        from run_gui_verifier import VerifierRuntime
        self.verifier = VerifierRuntime()
        self.proposal_only = True
        self.counter.attach(self.verifier.model, 14)
        self.metadata.update(method='our_candidates_plus_GUI_verifier', verifier=self.verifier.metadata,
                             primary_timing_seed=None,
                             note='Independently regenerate all VLM views and proposals for each timed call; canonicalK16, verifier batch4')

    def infer(self, image, instruction):
        proposal = super().infer(image, instruction)
        points, rows = proposal['candidates'], []
        for start in range(0, len(points), 4):
            rows.extend(self.verifier.score(image, instruction, points[start:start + 4], 'crop'))
        point = max(rows, key=lambda r: r['log_odds'])['point'] if rows else None
        return dict(point=point, candidates=points, views=proposal['views'], verifier_scores=rows,
                    ground_truth_in_model_input=False)


class UITarsAdapter:
    def __init__(self, folder, max_pixels=262144):
        from .model_baselines import UITarsRuntime
        self.runtime = UITarsRuntime(ROOT / 'models/UI-TARS-1.5-7B', max_pixels=max_pixels)
        self.counter = RuntimeCounter()
        self.counter.attach(self.runtime.model, 14)
        self.metadata = dict(self.runtime.metadata, method='ui_tars', model_source='Official verified local release')

    def infer(self, image, instruction):
        return self.runtime.infer(image, instruction)


COORDINATE_PROMPT = ('Locate the element described by the task in this screenshot. '
                     'Return only one point as (x, y), with both coordinates normalized to the range 0 to 1000. Task: ')


class QwenCoordinateAdapter:
    def __init__(self, folder, max_pixels=262144):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        path = ROOT / 'models/Qwen3-VL-8B-Instruct'
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(path, dtype=torch.float16,
            device_map='balanced', max_memory={0: '12GiB', 1: '12GiB'},
            attn_implementation='sdpa', local_files_only=True).eval().requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(path, min_pixels=16384, max_pixels=max_pixels, local_files_only=True)
        self.counter = RuntimeCounter()
        self.counter.attach(self.model, 16)
        self.metadata = dict(method='qwen_explicit_coordinate', max_pixels=max_pixels, dtype='float16',
                             prompt=COORDINATE_PROMPT, prompt_scope='Explicit normalized-point prompt adaptation; not a publisher score reproduction',
                             device_map={k: str(v) for k, v in self.model.hf_device_map.items()}, max_new_tokens=64)

    def infer(self, image, instruction):
        import torch
        messages = [dict(role='user', content=[dict(type='image'), dict(type='text', text=COORDINATE_PROMPT + instruction)])]
        text = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.processor(text=[text], images=[image], return_tensors='pt').to(self.model.get_input_embeddings().weight.device)
        with torch.inference_mode():
            ids = self.model.generate(**inputs, max_new_tokens=64, do_sample=False, use_cache=True)[0, inputs['input_ids'].shape[-1]:]
        text = self.processor.tokenizer.decode(ids, skip_special_tokens=True).strip()
        eos = self.model.generation_config.eos_token_id
        eos = eos if isinstance(eos, list) else [eos]
        completed = bool(len(ids) and int(ids[-1]) in eos)
        match = re.fullmatch(r'\(?\s*([0-9.]+)\s*,\s*([0-9.]+)\s*\)?', text)
        point = None
        if match and completed:
            xy = [float(v) for v in match.groups()]
            if all(math.isfinite(v) and 0 <= v <= 1000 for v in xy):
                point = [round(v * (s - 1) / 1000) for v, s in zip(xy, image.size)]
        return dict(point=point, generated_text=text, generated_tokens=len(ids), terminated_by_eos=completed,
                    ground_truth_in_model_input=False)


class AimaAdapter:
    def __init__(self, folder, max_pixels=262144):
        import torch
        from transformers import AutoProcessor
        # The official eager model imports one training-only logging function.
        # Supply only that logger; no modeling or attention arithmetic is changed.
        trainer = types.ModuleType('gui_aima.trainer')
        trainer.rank0_print = print
        sys.modules['gui_aima.trainer'] = trainer
        from gui_aima.modeling_qwen25vl_eager import Qwen2_5_VLForConditionalGenerationWithPointer
        from gui_aima.inference_eager import inference
        path = ROOT / 'models/GUI-AIMA-3B'
        receipt = read(path / 'clickagent_download_verified_v17.json')
        assert receipt['complete'] and receipt['all_files_verified']
        self.processor = AutoProcessor.from_pretrained(path, max_pixels=max_pixels, local_files_only=True)
        self.model, loading = Qwen2_5_VLForConditionalGenerationWithPointer.from_pretrained(
            path, torch_dtype=torch.float16, device_map='cuda:0', attn_implementation='eager',
            local_files_only=True, output_loading_info=True)
        if loading['missing_keys'] or loading['mismatched_keys'] or loading['unexpected_keys'] or loading['error_msgs']:
            raise RuntimeError(f'AIMA checkpoint loading mismatch: {loading}')
        self.model.eval().requires_grad_(False)
        self.inference = inference
        self.counter = RuntimeCounter()
        self.counter.attach(self.model, 14)
        self.metadata = dict(method='GUI-AIMA-3B', max_pixels=max_pixels, dtype='float16',
                             attention='Official eager path; P5000 does not support publisher FlashAttention/BF16',
                             official_revision=receipt['official_revision'], loading_info=loading,
                             compatibility='Only training logger shim; official eager modeling and inference functions unchanged')
        if getattr(self, 'two_stage', False):
            self.metadata.update(two_stage=True, zoom_scale=3., crop_rule='Official616^2/5760000 area rule')

    def infer(self, image, instruction):
        import torch
        conversation = [dict(role='system', content=[dict(type='text', text=(
            'You are a GUI agent. Given a screenshot of the current GUI and a human instruction, your task is to locate the screen element that corresponds to the instruction. You should output a PyAutoGUI action that performs a click on the correct position. To indicate the click location, we will use some special tokens, which is used to refer to a visual patch later. For example, you can output: pyautogui.click(<your_special_token_here>).'))]),
            dict(role='user', content=[dict(type='image', image=image), dict(type='text', text=instruction)])]
        with torch.inference_mode():
            r = self.inference(conversation, self.model, self.processor.tokenizer, self.processor, use_placeholder=True, topk=3)
        xy = r['topk_points'][0] if r.get('topk_points') else None
        if xy and getattr(self, 'two_stage', False):
            source = (ROOT / 'third_party/GUI-AIMA/eval/visualization_utils.py').read_text(encoding='utf-8')
            node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'crop_subimage')
            namespace = {}
            exec(compile(ast.Module(body=[node], type_ignores=[]), 'official_crop_subimage', 'exec'), namespace)
            size = int(math.sqrt(image.width * image.height * 616 ** 2 / 5760000) / 28) * 28
            box = namespace['crop_subimage'](image.width, image.height, int(xy[0] * image.width), int(xy[1] * image.height), size)
            cropped = image.crop(box)
            enlarged = cropped.resize((cropped.width * 3, cropped.height * 3), Image.Resampling.BICUBIC)
            conversation[-1]['content'][0]['image'] = enlarged
            with torch.inference_mode():
                r2 = self.inference(conversation, self.model, self.processor.tokenizer, self.processor, use_placeholder=True, topk=3)
            xy2 = r2['topk_points'][0] if r2.get('topk_points') else None
            if xy2:
                xy = [(int(xy2[0] * enlarged.width) / 3 + box[0]) / image.width,
                      (int(xy2[1] * enlarged.height) / 3 + box[1]) / image.height]
            else:
                xy = None
        point = [round(x * s) for x, s in zip(xy, image.size)] if xy else None
        if point is not None and not all(0 <= v < s for v, s in zip(point, image.size)):
            point = None
        return dict(point=point, topk_normalized=r.get('topk_points'), output_text=r.get('output_text'),
                    ground_truth_in_model_input=False)


class AimaZoomAdapter(AimaAdapter):
    two_stage = True


def mvp_failed_output(captured_states, diagnostics):
    """Distinguish absent generated-token attention from swallowed reference errors."""
    missing_query = captured_states.get('query_states') is None
    expected_missing_token = (missing_query and 'get_top_attention_regions' in diagnostics
                              and 'query_states.shape' in diagnostics
                              and "'NoneType' object has no attribute 'shape'" in diagnostics)
    resource_error = any(term in diagnostics.lower() for term in (
        'out of memory', 'not enough memory', 'allocation on device',
        'paging file is too small', 'cublas_status_alloc_failed'))
    if resource_error:
        raise RuntimeError('MVP reference resource failure (out of memory): ' + diagnostics[-5000:])
    if expected_missing_token:
        return dict(point=None, output_failure='No comma-token attention captured',
                    reference_diagnostic=diagnostics[-3000:], ground_truth_in_model_input=False)
    raise RuntimeError('MVP reference failed: ' + diagnostics[-5000:])


class MVPAdapter:
    def __init__(self, folder, max_pixels=262144, single=False):
        import torch
        from transformers import AutoProcessor
        from qwen3_vl import Qwen3VLForConditionalGeneration, Qwen3VLConfig
        import mvp_sspro_qwen3vl as official
        path = ROOT / 'models/Qwen3-VL-8B-Instruct'
        config = Qwen3VLConfig.from_pretrained(path)
        config.target_token_id, config.target_layer_idx = ',', 24
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(path, config=config,
            torch_dtype=torch.float16, device_map='balanced', max_memory={0: '12GiB', 1: '12GiB'},
            attn_implementation='sdpa', local_files_only=True).eval().requires_grad_(False)
        self.processor = AutoProcessor.from_pretrained(path, min_pixels=3136, max_pixels=max_pixels, local_files_only=True)
        self.official, self.single = official, single
        self.device = self.model.get_input_embeddings().weight.device
        self.placeholder = Path(folder) / 'mvp_image_placeholder'
        self.placeholder.parent.mkdir(parents=True, exist_ok=True)
        self.placeholder.touch(exist_ok=True)
        self.counter = RuntimeCounter()
        self.counter.attach(self.model, 16)
        self.metadata = dict(method='Qwen3VL-MVP-single' if single else 'Qwen3VL-MVP', max_pixels=max_pixels,
            dtype='float16', attention_layer=24, max_subimages=0 if single else 2, consistency_threshold=14,
            implementation='Original repo functions, crop planning, grouping and coordinate conversion unchanged',
            image_interface='In-memory Image.open adapter; dummy bbox used only by reference diagnostic logging, no true labels',
            device_map={k: str(v) for k, v in self.model.hf_device_map.items()})

    def infer(self, image, instruction):
        import torch
        stdout, stderr = io.StringIO(), io.StringIO()
        self.model.clear_captured_states()
        with torch.inference_mode(), redirect_stdout(stdout), redirect_stderr(stderr):
            if self.single:
                xy, text = self.official.process_single_subimage(image, instruction, self.processor, self.model, self.device)
                r = dict(final_prediction=dict(point=xy), full_image_prediction=dict(output=text))
            else:
                original_open = Image.open
                def memory_open(path, *args, **kwargs):
                    if str(path) == str(self.placeholder):
                        return image.copy()
                    return original_open(path, *args, **kwargs)
                Image.open = memory_open
                try:
                    public = dict(img_filename=str(self.placeholder), instruction=instruction,
                                  bbox=[0, 0, 0, 0], id='public_only')
                    r = self.official.process_single_image(public, self.model, self.processor, '', self.device,
                                                           max_inferences=2, consistency_threshold=14)
                finally:
                    Image.open = original_open
        if r is None:
            return mvp_failed_output(self.model.captured_states, stderr.getvalue() + stdout.getvalue())
        point = r['final_prediction']['point'] if r.get('final_prediction') else None
        if point is not None:
            point = [int(v) for v in point]
            if not all(0 <= v < s for v, s in zip(point, image.size)):
                point = None
        return dict(point=point, reference_result=r, ground_truth_in_model_input=False)


class TAGAdapter:
    def __init__(self, folder, max_pixels=262144):
        import torch
        from MiniCPM_Llama3_V_2_5.modeling_minicpmv import MiniCPMV, PreTrainedTokenizerFastWrapper
        from core_utils.tag import AttentionDrivenGrounding
        prompt_path = ROOT / 'third_party/TAG/eval_mm/screenspot/prompt_tag.py'
        spec = importlib.util.spec_from_file_location('tag_screenspot_prompt_v17', prompt_path)
        prompts = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(prompts)
        eval_source = (ROOT / 'third_party/TAG/eval_mm/screenspot/eval_TAG.py').read_text(encoding='utf-8')
        resize_node = next(n for n in ast.parse(eval_source).body if isinstance(n, ast.FunctionDef) and n.name == 'resize_image')
        resize_namespace = {'Image': Image}
        exec(compile(ast.Module(body=[resize_node], type_ignores=[]), 'official_TAG_resize_image', 'exec'), resize_namespace)
        self.author_resize = resize_namespace['resize_image']
        path = ROOT / 'models/MiniCPM-Llama3-V-2_5'
        if not (path / 'clickagent_download_verified_v17.json').exists():
            path = ROOT / 'models/MiniCPM-Llama3-V-2_5_modelscope_staging'
        receipt = read(path / 'clickagent_download_verified_v17.json')
        assert receipt['complete'] and receipt['all_files_verified']
        layers = int(read(path / 'config.json')['num_hidden_layers'])
        device_map = {'llm.model.embed_tokens': 0, 'llm.model.norm': 0, 'llm.lm_head': 0,
                      'vpm': 0, 'resampler': 0}
        device_map.update({f'llm.model.layers.{i}': 0 if i < layers // 2 else 1 for i in range(layers)})
        self.model, loading = MiniCPMV.from_pretrained(path, torch_dtype=torch.float16,
            device_map=device_map,
            local_files_only=True, output_loading_info=True)
        if loading['missing_keys'] or loading['mismatched_keys'] or loading['error_msgs']:
            raise RuntimeError(f'TAG checkpoint mismatch: {loading}')
        self.model.eval().requires_grad_(False)
        self.tokenizer = PreTrainedTokenizerFastWrapper.from_pretrained(path, local_files_only=True)
        self.tag = AttentionDrivenGrounding(self.tokenizer, filter_thresold=.5, topk_layer_head=10, vis_flag=False)
        self.prompts, self.max_pixels = prompts, max_pixels
        self.counter = RuntimeCounter()
        self.counter.attach(self.model.llm, 14)
        self.metadata = dict(method='TAG-original-MiniCPM-Llama3-V-2_5', max_public_image_pixels=max_pixels,
            official_revision=receipt['official_revision'], dtype='float16', threshold=.5, topk_layer_head=10,
            native_slicing=True, default_max_new_tokens=1024, loading_info=loading,
            device_placement='Vision/resampler/input embeddings/final norm/output logits colocated onGPU0; language layers split equally across twoGPUs',
            note='Original TAG functions; prompt fallback when no description token is parsed; no true geometry enters TAG')

    def infer(self, image, instruction):
        import torch
        w, h = image.size
        image = self.author_resize(image, image.size)
        scale = min(1., math.sqrt(self.max_pixels / (image.width * image.height)))
        if scale < 1:
            image = image.resize((max(1, int(image.width * scale)), max(1, int(image.height * scale))), Image.Resampling.BICUBIC)
        prompt = self.prompts.query_prompt.format(query_text=instruction, interested_key=self.prompts.interested_key)
        with torch.inference_mode():
            response, attention = self.model.chat(image, [dict(role='user', content=prompt)], self.tokenizer,
                query_text_attention=True, sampling=False, temperature=0., num_beams=1,
                output_attentions=True, return_dict_in_generate=True, output_attentions_generation=True)
            self.counter.processed_pixels += sum(v.width * v.height for v in attention['slice_images'])
            generated_indices = self.prompts.find_element_content_indices(response, self.tokenizer,
                                                                           interested_key=self.prompts.interested_key)
            attention.pop('generated_text', None)
            self_attentions = attention.pop('self_attentions')
            attention['original_image'] = image
            if generated_indices:
                attention.update(self_attentions=self_attentions[1:], interested_query_idx=generated_indices,
                                 input_ids=[self.tokenizer.encode(response)], use_gen_attn=True)
            else:
                attention['self_attentions'] = [self_attentions[0]]
            self.tag.save_path = str(ROOT / 'runs/advantage_validation_v17/tag_unused_visualization')
            result = self.tag.forward(**attention)
        boxes = result['final_bbox']
        point = None
        if boxes:
            box, confidence = max(boxes, key=lambda v: (v[1], (v[0][2] - v[0][0]) * (v[0][3] - v[0][1])))
            point = [round((box[0] + box[2]) / 2 * w / image.width), round((box[1] + box[3]) / 2 * h / image.height)]
            if not (0 <= point[0] < w and 0 <= point[1] < h):
                point = None
        return dict(point=point, response=response, generated_description_tokens=len(generated_indices),
                    input_image_size=list(image.size), final_bbox=boxes, ground_truth_in_model_input=False)


def create_runtime(method, folder, max_pixels=262144):
    configure_paths(method)
    classes = {'ours': OurRuntime, 'ours_gui': OurGUIAdapter, 'ui_tars': UITarsAdapter, 'qwen': QwenCoordinateAdapter,
               'aima': AimaAdapter, 'aima_zoom': AimaZoomAdapter, 'tag': TAGAdapter}
    if method in ['mvp', 'mvp_single']:
        return MVPAdapter(folder, max_pixels, single=method == 'mvp_single')
    if method == 'ours_s1':
        return OurRuntime(folder, max_pixels, seeds=(1,))
    return classes[method](folder, max_pixels)
