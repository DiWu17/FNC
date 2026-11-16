"""Independent frozen model point baselines; no model output is executed."""
from __future__ import annotations

import ast
import re
import time


# Official UI-TARS README_v1 single-point grounding prompt (19 words).
UI_TARS_POINT_PROMPT = "Output only the coordinate of one point in your response. What element matches the following task: "
UI_TARS_PROMPT_SOURCE = "https://github.com/bytedance/UI-TARS/blob/main/README_v1.md"
UI_TARS_COORDINATE_SOURCE = "https://github.com/bytedance/UI-TARS/blob/main/codes/ui_tars/action_parser.py"


def parse_ui_tars_point(text, resized_size, original_size):
    """Qwen2.5-VL UI-TARS uses processed-image absolute pixels, not 0..1000.

    Accept a complete point or the publisher's click wrappers. No regex search
    for arbitrary numbers in prose, no eval, no label-based repair or clamping.
    """
    text=text.strip()
    if re.search(r"(?:\A|\n)Action:\s*",text):
        text=re.split(r"(?:\A|\n)Action:\s*",text)[-1].strip()
    if re.match(r"(?:click|left_double|right_single)\s*\(",text):
        try:
            node=ast.parse(text,mode="eval").body
            if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name) or node.args:
                return None
            kwargs={k.arg:ast.literal_eval(k.value) for k in node.keywords}
            if set(kwargs) not in ({"point"},{"start_box"}):return None
            text=next(iter(kwargs.values()))
            if not isinstance(text,str):return None
        except (ValueError,SyntaxError,TypeError):return None
    for token in ("<point>","</point>","<|box_start|>","<|box_end|>"):
        text=text.replace(token,"")
    text=text.strip().strip("()[]").strip()
    if not re.fullmatch(r"[0-9.]+(?:\s*,\s*|\s+)[0-9.]+(?:(?:\s*,\s*|\s+)[0-9.]+(?:\s*,\s*|\s+)[0-9.]+)?",text):
        return None
    try:values=[float(s) for s in re.split(r"\s*,\s*|\s+",text)]
    except ValueError:return None
    x,y=values[:2]
    if len(values)==4:x,y=(x+values[2])/2,(y+values[3])/2
    rw,rh=resized_size;w,h=original_size
    if not (0<=x<rw and 0<=y<rh):return None
    point=[round(x*w/rw),round(y*h/rh)]
    return point if 0<=point[0]<w and 0<=point[1]<h else None


class UITarsRuntime:
    def __init__(self,model_path,max_pixels=262144):
        import torch
        from transformers import AutoProcessor,Qwen2_5_VLForConditionalGeneration
        self.model,loading=Qwen2_5_VLForConditionalGeneration.from_pretrained(str(model_path),dtype=torch.float16,
            device_map="balanced",max_memory={0:"13GiB",1:"13GiB","cpu":"24GiB"},
            attn_implementation="sdpa",low_cpu_mem_usage=True,local_files_only=True,output_loading_info=True)
        if loading['missing_keys'] or loading['mismatched_keys'] or loading['error_msgs']:
            raise ValueError(f'Incomplete UI-TARS checkpoint loading: {loading}')
        if any('inv_freq' not in name for name in loading['unexpected_keys']):
            raise ValueError(f'Unexpected UI-TARS parameter keys: {loading}')
        self.model.eval().requires_grad_(False)
        self.processor=AutoProcessor.from_pretrained(str(model_path),min_pixels=16384,max_pixels=max_pixels,local_files_only=True)
        self.device=self.model.get_input_embeddings().weight.device
        eos=self.model.generation_config.eos_token_id
        self.eos=set(eos if isinstance(eos,list) else [eos])
        self.metadata={"model_class":type(self.model).__name__,"max_pixels":max_pixels,"dtype":"float16",
            "parameters":sum(p.numel() for p in self.model.parameters()),
            "trainable_parameters":sum(p.numel() for p in self.model.parameters() if p.requires_grad),
            "coordinate_contract":"absolute pixels in actual processed screenshot",
            "checkpoint_loading_info":loading,
            "prompt_source":UI_TARS_PROMPT_SOURCE,"coordinate_source":UI_TARS_COORDINATE_SOURCE}

    def infer(self,image,instruction):
        import torch
        from .model_features import _sync_cuda
        messages=[{"role":"user","content":[{"type":"image"},{"type":"text","text":UI_TARS_POINT_PROMPT+instruction}]}]
        text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True)
        _sync_cuda();started=time.perf_counter()
        inputs=self.processor(text=[text],images=[image.convert("RGB")],return_tensors="pt",padding=False).to(self.device)
        grid=inputs["image_grid_thw"][0]
        patch=self.model.config.vision_config.patch_size
        resized=[int(grid[2])*patch,int(grid[1])*patch]
        with torch.inference_mode():
            ids=self.model.generate(**inputs,max_new_tokens=64,do_sample=False,use_cache=True)[0,inputs['input_ids'].shape[-1]:].tolist()
        generated=self.processor.tokenizer.decode(ids,skip_special_tokens=True)
        completed=bool(ids and ids[-1] in self.eos)
        point=parse_ui_tars_point(generated,resized,image.size) if completed else None
        _sync_cuda()
        return {"generated_text":generated,"generated_token_ids":ids,"terminated_by_eos":completed,
            "point":point,"parse_valid":point is not None,"processed_size":resized,"image_size":list(image.size),
            "coordinate_contract":self.metadata['coordinate_contract'],"elapsed_seconds":time.perf_counter()-started,
            "generated_tokens":len(ids),"ground_truth_in_model_input":False}
