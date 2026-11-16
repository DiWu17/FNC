"""Matched readouts of frozen full-instruction and local spatial sequences."""
from functools import lru_cache
import math
import numpy as np
import torch
from torch import nn
from .bottleneck_v19 import BottleneckModel

DIM=64
PROJECTION_SEED=20262201

@lru_cache(maxsize=4)
def projection(d):
    rng=np.random.default_rng(PROJECTION_SEED+int(d))
    return (2*rng.integers(0,2,(d,DIM),dtype=np.int8)-1).astype(np.float32)

def project(a):
    a=np.asarray(a,np.float32)
    return (a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-8))@projection(a.shape[-1])

def spatial_sequence(arrays,meta,points,size,transform=None):
    """Ordered3x3 fine vision cells plus central coarse decoder cell;no GT."""
    gh,gw=map(int,meta['grid_shape']);image=np.asarray(arrays['image_hidden']);vision=np.asarray(arrays['vision_hidden'])
    assert image.shape==(gh*gw,4096) and vision.shape==(4*gh*gw,1152)
    assert meta['fine_grid_shape']==[2*gh,2*gw]
    bounds=np.asarray([0,0,*size] if transform is None else transform['box'],float)
    xy=(np.asarray(points).reshape(-1,2)-bounds[:2])/(bounds[2:]-bounds[:2])
    valid=((xy>=0)&(xy<1)).all(-1);fine=np.floor(xy*[2*gw,2*gh]).astype(int)
    vf=project(vision).reshape(2*gh,2*gw,DIM);coarse=project(image).reshape(gh,gw,DIM)
    features=np.zeros((len(xy),10,DIM),np.float16);mask=np.zeros((len(xy),10),bool)
    j=0
    for dy in [-1,0,1]:
        for dx in [-1,0,1]:
            x,y=fine[:,0]+dx,fine[:,1]+dy;ok=valid&(x>=0)&(x<2*gw)&(y>=0)&(y<2*gh)
            features[ok,j]=vf[y[ok],x[ok]];mask[:,j]=ok;j+=1
    cell=np.floor(xy*[gw,gh]).astype(int);features[valid,9]=coarse[cell[valid,1],cell[valid,0]];mask[:,9]=valid
    assert np.isfinite(features).all()
    return features,mask

def pooled_inputs(text,tm,spatial,sm,text_pool=False,space_pool=False):
    """Keep sequence lengths,validity,roles and positions;collapse content only."""
    if text_pool:
        mean=(text*tm[...,None]).sum(1)/tm.sum(1).clamp_min(1)[:,None]
        text=mean[:,None].expand_as(text)*tm[...,None]
    if space_pool:
        fine=sm[...,:9]
        mean=(spatial[...,:9,:]*fine[...,None]).sum(-2)/fine.sum(-1).clamp_min(1)[...,None]
        spatial=torch.cat((mean[...,None,:].expand_as(spatial[...,:9,:])*fine[...,None],spatial[...,9:,:]),-2)
    return text,spatial

def safe_attention(values,mask,query):
    valid=mask.any(-1);safe=mask.clone();safe[...,0]|=~valid
    logits=(values*query[...,None,:]).sum(-1)/math.sqrt(values.shape[-1])
    weights=logits.masked_fill(~safe,-torch.inf).softmax(-1)
    return (weights[...,None]*values).sum(-2)*valid[...,None]

class StructureReadout(nn.Module):
    def __init__(self,width=32):
        super().__init__();self.width=width
        # First initialization is exactly the existing width-matched core.
        self.core=BottleneckModel(width=width)
        self.text_in=nn.Sequential(nn.Linear(DIM,width),nn.LayerNorm(width),nn.GELU())
        self.text_gru=nn.GRU(width,width,batch_first=True)
        self.visual=nn.Sequential(nn.Linear(DIM,width),nn.LayerNorm(width),nn.GELU())
        self.position=nn.Linear(3,width,bias=False);self.view=nn.Embedding(4,width)
        self.readout=nn.Sequential(nn.Linear(width*4,width*2),nn.GELU(),nn.Linear(width*2,1))
        nn.init.zeros_(self.readout[-1].weight);nn.init.zeros_(self.readout[-1].bias)
        locations=[[x,y,0] for y in [-1,0,1] for x in [-1,0,1]]+[[0,0,1]]
        self.register_buffer('locations',torch.tensor(locations,dtype=torch.float32))

    @property
    def null(self):return self.core.null

    def forward(self,base,state,mask,text,text_mask,spatial,spatial_mask,candidate_case,condition='ordered',ablation=None):
        assert condition in ['ordered','pooled']
        text_pool=condition=='pooled' or ablation in ['text_pooled','both_pooled']
        space_pool=condition=='pooled' or ablation in ['space_pooled','both_pooled']
        text,spatial=pooled_inputs(text.float(),text_mask,spatial.float(),spatial_mask,text_pool,space_pool)
        tq,_=self.text_gru(self.text_in(text));q=(tq*text_mask[...,None]).sum(1)/text_mask.sum(1).clamp_min(1)[:,None]
        cq=q[candidate_case]
        image=self.visual(spatial)+self.position(self.locations)[None,None]+self.view.weight[None,:,None]
        per_view=safe_attention(image,spatial_mask,cq[:,None].expand(-1,4,-1))
        context=safe_attention(per_view,spatial_mask.any(-1),cq)
        extra=self.readout(torch.cat((context,cq,context*cq,self.core.base_encoder(base)),-1)).squeeze(-1)
        return self.core(base=base,state=state,mask=mask)+extra
