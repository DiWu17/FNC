"""Frozen global-image point features and an independently trained utility head.

All points, including crop-origin points, use the SAME full-image state. Neither
feature construction nor inference accepts a target annotation.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .attention_field_ranker import build_attention_field


class FrozenPointEncoder:
    def __init__(self,readout):
        self.readout=readout.cpu().eval().requires_grad_(False)

    @torch.inference_mode()
    def prepare(self,arrays,grid_shape,image_size):
        model=self.readout;gh,gw=map(int,grid_shape);p=gh*gw
        a=np.asarray(arrays['target_attention'],np.float32)
        if not len(a):return None
        qraw=F.layer_norm(torch.from_numpy(np.asarray(arrays['target_hidden'],np.float32)).reshape(-1,3,4096),[4096])
        query=model.query(qraw.flatten(1))+model.shared_semantic(qraw[:,0])
        image=model.shared_semantic(F.layer_norm(torch.from_numpy(np.asarray(arrays['image_hidden'],np.float32)),[4096]))
        fine=np.asarray(arrays['vision_hidden']).reshape(gh,2,gw,2,1152).transpose(0,2,1,3,4).reshape(p,4,1152).copy()
        vision=model.vision(F.layer_norm(torch.from_numpy(fine.astype(np.float32)),[1152]))
        image=image+vision.mean(1)
        field=build_attention_field(a,np.arange(p)).astype(np.float32)
        spectra=np.concatenate((field.mean(1).reshape(p,-1),field.max(1).reshape(p,-1)),-1)
        return dict(query_last=query[-1].numpy(),query_mean=query.mean(0).numpy(),image=image.numpy(),vision=vision.numpy(),
            spectra=spectra,grid_shape=(gh,gw),image_size=tuple(image_size))

    def features(self,state,points):
        points=np.asarray(points,dtype=np.float32).reshape(-1,2)
        if state is None:
            if len(points):raise ValueError('points without name states')
            return np.empty((0,0),np.float32)
        gh,gw=state['grid_shape'];w,h=state['image_size'];xy=points/np.asarray([w,h],np.float32)
        grid=xy*np.asarray([gw,gh],np.float32);cell=np.floor(grid).astype(int)
        cell=np.clip(cell,[0,0],[gw-1,gh-1]);fraction=grid-cell;parent=cell[:,1]*gw+cell[:,0]
        childxy=np.floor(fraction*2).astype(int).clip(0,1);child=childxy[:,1]*2+childxy[:,0]
        image=state['image'][parent];vision=state['vision'][parent];selected=vision[np.arange(len(points)),child]
        query=np.broadcast_to(state['query_last'],image.shape);mean=np.broadcast_to(state['query_mean'],image.shape)
        geometry=np.concatenate((xy,fraction,fraction*2-childxy,
            np.tile([1/gw,1/gh],(len(points),1)),np.sin(xy*np.pi),np.cos(xy*np.pi)),-1)
        feature=np.concatenate((query,mean,image,vision.reshape(len(points),-1),selected,
            image*query,selected*query,state['spectra'][parent],geometry),-1).astype(np.float32)
        assert np.isfinite(feature).all()
        return feature


class PointUtilityCritic(nn.Module):
    def __init__(self,mean,std,width=128):
        super().__init__()
        self.register_buffer('mean',torch.as_tensor(mean,dtype=torch.float32))
        self.register_buffer('std',torch.as_tensor(std,dtype=torch.float32).clamp_min(.05))
        self.width=width
        self.network=nn.Sequential(nn.Linear(len(self.mean),width),nn.GELU(),nn.Dropout(.15),
            nn.Linear(width,64),nn.GELU(),nn.Linear(64,1))

    def forward(self,features):
        return self.network(((features.float()-self.mean)/self.std).clamp(-8,8)).squeeze(-1)


def point_utility_loss(scores,labels,lengths):
    cursor=0;parts=[]
    for n in lengths:
        s=scores[cursor:cursor+n];y=labels[cursor:cursor+n].bool();cursor+=n
        bce=F.binary_cross_entropy_with_logits(s,y.float(),reduction='none')
        balanced=.5*bce[y].mean()+.5*bce[~y].mean() if y.any() and (~y).any() else bce.mean()
        rank=torch.logsumexp(s,0)-torch.logsumexp(s[y],0) if y.any() else s.sum()*0
        if y.any() and (~y).any():
            negatives=s[~y].topk(min(4,int((~y).sum()))).values
            pair=F.softplus(.5+negatives[None]-s[y,None]).mean()
        else:pair=s.sum()*0
        parts.append(rank+balanced+.25*pair)
    assert cursor==len(scores)
    return torch.stack(parts).mean()


def load_point_critic(path):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    model=PointUtilityCritic(saved['state_dict']['mean'],saved['state_dict']['std'],saved['width'])
    model.load_state_dict(saved['state_dict']);return model.eval(),saved['metadata']
