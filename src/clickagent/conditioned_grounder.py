"""Target-conditioned spatial readout over frozen language/vision features."""
from __future__ import annotations

from dataclasses import asdict,dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .spatial_geometry import decode_candidates


@dataclass
class SpatialConfig:
    image_dim:int=4096
    vision_dim:int=1152
    query_dim:int=12288
    query_layers:int=3
    heads:int=96
    width:int=128
    variant:str='full'

    def __post_init__(self):
        if self.variant not in ('full','attention_only','no_image','no_mass','no_query','no_attention','static_96'):
            raise ValueError('unknown spatial ablation')
        if self.query_dim%self.query_layers:raise ValueError('query layer dimensions disagree')


class ConditionedSpatialGrounder(nn.Module):
    """Learn temporal pooling, semantic head combination and local offsets.

    Visual positions remain in their native grid. Attention mass is retained
    both as global head context and as absolute raw-attention spatial channels.
    No labels, target boxes, dataset IDs or output coordinates enter forward.
    """
    def __init__(self,config:SpatialConfig,initial_head_weights=None):
        super().__init__();self.config=config;c=config;d=c.width
        init=np.zeros(c.heads,np.float32);init[:min(32,c.heads)]=1/min(32,c.heads)
        if initial_head_weights is not None:
            init=np.asarray(initial_head_weights,np.float32)
            if init.shape!=(c.heads,):raise ValueError('initial head weights disagree')
        self.base_head_weights=nn.Parameter(torch.from_numpy(init.copy()))
        # Same enlarged training set, original static first-token geometry.
        # A common scalar bias would cancel in spatial softmax, so exactly H
        # weights suffice. No hidden state, mass or image content enters it.
        if c.variant=='static_96':return
        self.query_projection=nn.Sequential(nn.Linear(c.query_dim,d),nn.GELU(),nn.LayerNorm(d))
        self.image_projection=nn.Sequential(nn.Linear(c.image_dim,d),nn.GELU(),nn.LayerNorm(d))
        self.vision_projection=nn.Sequential(nn.Linear(c.vision_dim,64),nn.GELU(),nn.LayerNorm(64))
        self.token_score=nn.Linear(d,1)
        nn.init.zeros_(self.token_score.weight);nn.init.zeros_(self.token_score.bias)
        self.first_token_bias=nn.Parameter(torch.tensor(2.))
        self.mass_projection=nn.Sequential(nn.Linear(c.heads,d),nn.GELU(),nn.LayerNorm(d))
        self.context=nn.Sequential(nn.Linear(d*2,d),nn.GELU(),nn.LayerNorm(d))
        self.head_delta=nn.Sequential(nn.Linear(d,d),nn.GELU(),nn.Linear(d,c.heads))
        nn.init.zeros_(self.head_delta[-1].weight);nn.init.zeros_(self.head_delta[-1].bias)
        self.film=nn.Linear(d,d*2)
        self.attention_projection=nn.Sequential(nn.Conv2d(c.heads*2,64,1),nn.GELU())
        self.spatial=nn.Sequential(nn.Conv2d(d+128+3,d,1),nn.GroupNorm(8,d),nn.GELU(),
            nn.Conv2d(d,d,3,padding=1),nn.GroupNorm(8,d),nn.GELU(),
            nn.Conv2d(d,d,3,padding=1),nn.GroupNorm(8,d),nn.GELU())
        self.score_delta=nn.Conv2d(d,1,1);self.offset_head=nn.Conv2d(d,2,1)
        for layer in (self.score_delta,self.offset_head):
            nn.init.zeros_(layer.weight);nn.init.zeros_(layer.bias)

    def forward(self,attention,query_hidden,image_hidden,vision_hidden,token_mask,grid_shape):
        c=self.config;gh,gw=map(int,grid_shape)
        # [batch, target tokens, heads, image positions]
        b,t,h,p=attention.shape
        if (h,p)!=(c.heads,gh*gw) or query_hidden.shape!=(b,t,c.query_dim) or image_hidden.shape!=(b,p,c.image_dim) or vision_hidden.shape!=(b,p*4,c.vision_dim):
            raise ValueError('unaligned target/image feature shapes')
        if not bool(token_mask.any(-1).all()):raise ValueError('empty target phrase has no spatial readout')
        if c.variant=='static_96':
            first=attention[:,0].float();mass=first.sum(-1)
            features=torch.log1p(p*first/mass.unsqueeze(-1).clamp_min(1e-12))
            logits=torch.einsum('h,bhp->bp',self.base_head_weights,features).reshape(b,gh,gw)
            weights=attention.new_zeros((b,t));weights[:,0]=1
            return {'logits':logits,'grid_shape':[gh,gw],'offsets':attention.new_full((b,2,gh,gw),.5),
                'token_weights':weights,'head_weights':self.base_head_weights.unsqueeze(0).expand(b,-1),
                'head_mass':mass,'static_coarse_control':True}
        use_query=c.variant not in ('no_query','attention_only')
        use_image=c.variant not in ('no_image','attention_only')
        use_mass=c.variant not in ('no_mass','attention_only','no_attention')
        if use_query:
            q=F.layer_norm(query_hidden.float().reshape(b,t,c.query_layers,-1),[c.query_dim//c.query_layers]).reshape(b,t,c.query_dim)
            q=self.query_projection(q)
        else:q=attention.new_zeros((b,t,c.width))
        token_logits=self.token_score(q).squeeze(-1)
        position_bias=torch.zeros_like(token_logits);position_bias[:,0]=self.first_token_bias
        token_logits=(token_logits+position_bias).masked_fill(~token_mask,-torch.inf)
        token_weights=token_logits.softmax(-1)
        pooled=torch.einsum('bt,bthp->bhp',token_weights,attention.float())
        mass=pooled.sum(-1)
        relative=pooled/mass.unsqueeze(-1).clamp_min(1e-12)
        relative_features=torch.log1p(p*relative)
        if c.variant=='no_attention':relative_features=torch.zeros_like(relative_features)
        absolute_features=torch.log1p(p*pooled) if use_mass else torch.zeros_like(relative_features)
        mass_context=self.mass_projection(torch.log(mass.clamp_min(1e-8))) if use_mass else q.new_zeros((b,c.width))
        q_context=torch.einsum('bt,btd->bd',token_weights,q)
        context=self.context(torch.cat((q_context,mass_context),-1))
        head_weights=self.base_head_weights.unsqueeze(0)+self.head_delta(context)
        prior=torch.einsum('bh,bhp->bp',head_weights,relative_features).reshape(b,1,gh,gw)
        if use_image:image=self.image_projection(F.layer_norm(image_hidden.float(),[c.image_dim]))
        else:image=q.new_zeros((b,p,c.width))
        fh,fw=gh*2,gw*2
        if use_image:vision=self.vision_projection(F.layer_norm(vision_hidden.float(),[c.vision_dim]))
        else:vision=q.new_zeros((b,p*4,64))
        vision=vision.transpose(1,2).reshape(b,64,fh,fw)
        scale,shift=self.film(context).chunk(2,-1)
        conditioned=image*(1+scale.tanh().unsqueeze(1))+shift.unsqueeze(1)
        similarity=(image*q_context.unsqueeze(1)).sum(-1,keepdim=True)/c.width**.5
        attention_map=self.attention_projection(torch.cat((relative_features,absolute_features),1).reshape(b,h*2,gh,gw))
        up=lambda value:F.interpolate(value,size=(fh,fw),mode='nearest')
        yy,xx=torch.meshgrid((torch.arange(fh,device=attention.device)+.5)/fh,
            (torch.arange(fw,device=attention.device)+.5)/fw,indexing='ij')
        coords=torch.stack((xx,yy)).unsqueeze(0).expand(b,-1,-1,-1)
        features=torch.cat((up(conditioned.transpose(1,2).reshape(b,c.width,gh,gw)),up(attention_map),vision,
            up(similarity.transpose(1,2).reshape(b,1,gh,gw)),coords),1)
        spatial=self.spatial(features)
        return {'logits':(up(prior)+self.score_delta(spatial)).squeeze(1),'grid_shape':[fh,fw],
            'offsets':self.offset_head(spatial).sigmoid(),'token_weights':token_weights,
            'head_weights':head_weights,'head_mass':mass}


def spatial_loss(output,weights,offsets,offset_weight=2.):
    if output.get('static_coarse_control',False):
        b,gh,gw=output['logits'].shape
        # Summing the four fine intersection areas exactly recovers the coarse
        # positive-area label. Tiny targets remain training examples.
        weights=weights.reshape(b,gh,2,gw,2).sum((2,4))
        region=-(weights*F.log_softmax(output['logits'].flatten(1),-1).reshape_as(weights)).sum((-1,-2)).mean()
        return region,{'region_loss':float(region.detach()),'offset_loss':0.}
    if not torch.allclose(weights.sum((-1,-2)),torch.ones_like(weights[:,0,0]),atol=1e-5):
        raise ValueError('spatial labels must sum to one per image')
    region=-(weights*F.log_softmax(output['logits'].flatten(1),-1).reshape_as(weights)).sum((-1,-2)).mean()
    coordinate=(F.smooth_l1_loss(output['offsets'],offsets,reduction='none').sum(1)*weights).sum((-1,-2)).mean()
    return region+offset_weight*coordinate,{'region_loss':float(region.detach()),'offset_loss':float(coordinate.detach())}


def make_batch(records,device='cpu',labels=True):
    if not records:raise ValueError('empty batch')
    grid=records[0]['metadata']['grid_shape']
    if any(r['metadata']['grid_shape']!=grid for r in records):raise ValueError('batch requires one native grid shape')
    b=len(records);t=max(len(r['target_attention']) for r in records)
    if t<1:raise ValueError('empty target is a retained inference failure, not a training batch')
    h,p=records[0]['target_attention'].shape[1:];q=records[0]['target_hidden'].shape[-1]
    attention=np.zeros((b,t,h,p),np.float32);hidden=np.zeros((b,t,q),np.float16);mask=np.zeros((b,t),bool)
    for i,r in enumerate(records):
        n=len(r['target_attention'])
        if not n:raise ValueError('missing target in batch')
        attention[i,:n]=r['target_attention'];hidden[i,:n]=r['target_hidden'];mask[i,:n]=True
    tensor=lambda a:torch.from_numpy(np.asarray(a)).to(device)
    inputs={'attention':tensor(attention),'query_hidden':tensor(hidden),'image_hidden':tensor(np.stack([r['image_hidden'] for r in records])),
        'vision_hidden':tensor(np.stack([r['vision_hidden'] for r in records])),
        'token_mask':tensor(mask),'grid_shape':grid}
    if not labels:return inputs,None
    target={'weights':tensor(np.stack([r['label_weights'] for r in records])),
        'offsets':tensor(np.stack([r['label_offsets'] for r in records]))}
    return inputs,target


def save_spatial_checkpoint(path,model,metadata):
    torch.save({'config':asdict(model.config),'state_dict':model.state_dict(),'metadata':metadata},path)


def load_spatial_checkpoint(path,device='cpu'):
    checkpoint=torch.load(path,map_location='cpu',weights_only=True)
    model=ConditionedSpatialGrounder(SpatialConfig(**checkpoint['config']))
    model.load_state_dict(checkpoint['state_dict'],strict=True)
    return model.to(device).eval(),checkpoint['metadata']


@torch.inference_mode()
def predict_spatial(model,record,device='cpu',center_only=False,top_k=5):
    meta=record['metadata']
    if not meta['target_available'] or not len(record['target_attention']):
        return {'point':None,'candidates':[],'readout_available':False}
    inputs,_=make_batch([record],device,labels=False);output=model(**inputs)
    candidates=decode_candidates(output['logits'][0].cpu().numpy(),output['offsets'][0].cpu().numpy(),
        output['grid_shape'],meta['image_size'],top_k,center_only)
    return {'point':candidates[0]['point'],'candidates':candidates,'readout_available':True,
        'token_weights':output['token_weights'][0].cpu().tolist(),
        'head_mass':output['head_mass'][0].cpu().tolist(),
        'feature_grid_shape':meta['grid_shape'],'proposal_grid_shape':output['grid_shape'],
        'note':'Spatial scores are not calibrated action-execution confidence.'}
