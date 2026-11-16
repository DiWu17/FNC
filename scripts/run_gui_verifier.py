"""Official trained verifier, same candidate points: matched crop and full view."""
import argparse
import json
import math
from pathlib import Path
import time

import torch
from PIL import Image,ImageDraw
from transformers import AutoProcessor,Qwen2VLForConditionalGeneration

from clickagent.action_benchmark import sha256,write_json
from clickagent.point_aligned_inference import marked_crop,unique_points
from point_aligned_data import read
from run_target_token_experiment import key,read_jsonl

OUT=Path('runs/context_verifier_v9');MODEL=Path('models/GUI-Actor-Verifier-2B')
PROMPT="Please observe the screenshot and exame whether the hollow red circle accurately placed on the intended position in the image: '{}'. Answer True or False."


def label_ids(tokenizer,label):
    result=[]
    for value in (label,' '+label):
        ids=tokenizer.encode(value,add_special_tokens=False)
        if len(ids)==1:result.append(ids[0])
    assert result
    return sorted(set(result))


class VerifierRuntime:
    def __init__(self):
        verified=read(MODEL/'clickagent_download_verified.json');assert verified['complete']
        for entry in verified['files']:assert sha256(MODEL/entry['file'])==entry['sha256']
        self.model,loading=Qwen2VLForConditionalGeneration.from_pretrained(str(MODEL),dtype=torch.float16,
            device_map={'':'cuda:0'},attn_implementation='sdpa',local_files_only=True,output_loading_info=True)
        assert not loading['missing_keys'] and not loading['mismatched_keys'] and not loading['error_msgs'],loading
        assert all('inv_freq' in k for k in loading['unexpected_keys']),loading
        self.model.eval().requires_grad_(False)
        self.processor=AutoProcessor.from_pretrained(str(MODEL),min_pixels=16384,max_pixels=262144,local_files_only=True)
        self.processor.tokenizer.padding_side='left'
        self.ids={label:label_ids(self.processor.tokenizer,label) for label in ('True','False')}
        self.metadata=dict(model=verified['official_model'],revision=verified['official_revision'],max_pixels=262144,
            dtype='float16',trainable_parameters=0,remote_code_executed=False,loading=loading,
            prompt=PROMPT,official_marker_full=True,matched_v8_marker_crop=True,
            score='logsumexp True variants minus logsumexp False variants')

    @torch.inference_mode()
    def score(self,image,instruction,points,mode):
        if not points:return []
        views=[];transforms=[];texts=[]
        for point in points:
            if mode=='crop':view,transform=marked_crop(image,point)
            else:
                view=image.copy();x,y=point;radius=math.ceil(8*image.height/1000*1.2);width=math.ceil(4*image.height/1000*1.2)
                ImageDraw.Draw(view).ellipse((x-radius,y-radius,x+radius,y+radius),outline='red',width=width)
                transform=dict(point=point,box=[0,0,image.width,image.height],marker_radius=radius,marker_width=width)
            messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':PROMPT.format(instruction)}]}]
            texts.append(self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True))
            views.append(view);transforms.append(transform)
        torch.cuda.synchronize();started=time.perf_counter()
        inputs=self.processor(text=texts,images=views,padding=True,return_tensors='pt').to('cuda:0')
        outputs=self.model(**inputs,use_cache=False,logits_to_keep=1,return_dict=True)
        logits=outputs.logits[:,-1].float().cpu();del outputs
        yes=torch.logsumexp(logits[:,self.ids['True']],-1);no=torch.logsumexp(logits[:,self.ids['False']],-1)
        masses=torch.exp(torch.logaddexp(yes,no)-torch.logsumexp(logits,-1));top=logits.argmax(-1)
        torch.cuda.synchronize();elapsed=time.perf_counter()-started
        return [dict(point=point,log_odds=float(y-n),true_logit=float(y),false_logit=float(n),
            label_mass=float(mass),preferred_token=self.processor.tokenizer.decode([int(t)]),
            transform=transform,image_grid_thw=grid.tolist(),mode=mode,elapsed_seconds_per_point=elapsed/len(points),
            batch_elapsed_seconds=elapsed,batch_size=len(points),ground_truth_in_model_input=False)
            for point,y,n,mass,t,transform,grid in zip(points,yes,no,masses,top,transforms,inputs['image_grid_thw'].cpu())]


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['validation','test'],required=True);args=p.parse_args()
    torch.set_num_threads(4);root=OUT/'validation_smoke' if args.stage=='validation' else OUT
    if args.stage=='test':
        from seal_context_verifier import identity
        assert read(OUT/'inference_identity.json')==identity()
    cases=[c for c in read_jsonl(OUT/'manifest.jsonl') if c['experiment_split']==('val' if args.stage=='validation' else 'test')]
    if args.stage=='validation':cases=sorted(cases,key=lambda c:key(c['sample_id']))[:12]
    folder=root/'gui_verifier';folder.mkdir(exist_ok=True)
    pending=[c for c in cases if not (folder/f'{key(c["sample_id"])}.json').exists()]
    if pending:
        runtime=VerifierRuntime();write_json(folder/'runtime.json',runtime.metadata)
        for i,c in enumerate(pending,1):
            sid=key(c['sample_id']);pipeline=read(root/'pipeline'/f'{sid}.json')
            union=unique_points(*pipeline['pools'].values());points=[v['point'] for v in union]
            with Image.open(c['image']) as im:image=im.convert('RGB')
            result={mode:runtime.score(image,c['instruction'],points,mode) for mode in ['crop','full']}
            write_json(folder/f'{sid}.json',dict(sample_id=c['sample_id'],scores=result,ground_truth_in_model_input=False))
            print(json.dumps(dict(stage=args.stage,i=i,n=len(pending),points=len(points))),flush=True)
    write_json(folder/'completion.json',dict(complete=all((folder/f'{key(c["sample_id"])}.json').exists() for c in cases),n=len(cases)))


if __name__=='__main__':main()
