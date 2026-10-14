"""Legal evidence -> spatial map -> seven actions; independent UAV parameters.

No recurrent neural state, raw-entity decoder bypass, dropout or oracle input.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from .belief import BeliefSpec, rasterize
from .stacked import AgentLinear, AgentMLP, StackedMAPPO
from .bsed import pointwise, candidate_geometry
from envs.config import ACTION_VALUES


def norm(x):
    return F.layer_norm(x, (x.shape[-1],))


class EvidenceBlock(nn.Module):
    def __init__(self, agents, dim=128, heads=4):
        super().__init__()
        self.dim, self.heads = dim, heads
        self.q = AgentLinear(agents, dim, dim, 1.)
        self.k = AgentLinear(agents, dim, dim, 1.)
        self.v = AgentLinear(agents, dim, dim, 1.)
        self.out = AgentLinear(agents, dim, dim, 1.)
        self.geometry = AgentLinear(agents, 3, heads, .01)
        self.age = AgentLinear(agents, 1, heads, .01)
        self.ff1 = AgentLinear(agents, dim, 2*dim, math.sqrt(2))
        self.ff2 = AgentLinear(agents, 2*dim, dim, 1.)

    def forward(self, maps, entities, positions, ages, valid, centers):
        n,b,j,d = maps.shape
        h = self.heads
        def split(layer, x):
            return pointwise(layer, x).reshape(n,b,-1,h,d//h).transpose(-3,-2)
        q,k,v = split(self.q,norm(maps)), split(self.k,entities), split(self.v,entities)
        delta = centers[None,None,:,None,:]-positions.unsqueeze(-3)
        geometry = torch.cat((delta,delta.square().sum(-1,keepdim=True)), -1)
        bias = 2*pointwise(self.geometry,geometry).tanh().permute(0,1,4,2,3)
        bias = bias + 2*pointwise(self.age,ages[...,None]).tanh().permute(0,1,3,2).unsqueeze(-2)
        bias = bias.masked_fill(~valid[:, :, None, None, :], float('-inf'))
        context = F.scaled_dot_product_attention(q,k,v,attn_mask=bias,dropout_p=0.)
        context = context.transpose(-3,-2).reshape(n,b,j,d)
        maps = maps + pointwise(self.out,context)
        return maps + pointwise(self.ff2,F.gelu(pointwise(self.ff1,norm(maps))))


class LocalMapConv(nn.Module):
    """Depthwise 3x3 spatial coherence with independent agent/channel kernels."""
    def __init__(self, agents, dim):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(agents,dim,1,3,3))
        self.bias = nn.Parameter(torch.zeros(agents,dim))
        nn.init.normal_(self.weight,std=.05)

    def forward(self, x):
        n,b,j,d = x.shape
        size = math.isqrt(j)
        image = x.permute(1,0,3,2).reshape(b,n*d,size,size)
        y = F.conv2d(image,self.weight.reshape(n*d,1,3,3),self.bias.reshape(n*d),padding=1,groups=n*d)
        return x + F.gelu(y.reshape(b,n,d,j).permute(1,0,3,2))


class MBDTActor(nn.Module):
    map_size, dim = 16, 128
    chunk_rows = 256
    heads, depth = 4, 2
    checkpoint_blocks = True
    block_type = EvidenceBlock

    def __init__(self, agents, hidden, spec):
        super().__init__()
        self.spec = spec
        d = self.dim
        self.anchor = AgentLinear(agents,11,d,1.)
        self.own_map = AgentLinear(agents,9,d,1.)
        # Each type includes explicit world position. History ages remain legal.
        self.types = nn.ModuleList([AgentLinear(agents,k,d,1.) for k in (11,10,7,8,6)])
        self.blocks = nn.ModuleList([self.block_type(agents,d,self.heads) for _ in range(self.depth)])
        self.local = LocalMapConv(agents,d)
        self.correction = AgentLinear(agents,d,3,.01)
        self.keys = AgentLinear(agents,d+5,d,1.)
        self.values = AgentLinear(agents,d+5,d,1.)
        self.query = AgentLinear(agents,18,d,1.)
        self.path_bias = AgentLinear(agents,3,1,.01)
        self.score = AgentMLP(agents,d+18,1,hidden,.01)
        centers=(torch.arange(self.map_size)+.5)/self.map_size
        yy,xx=torch.meshgrid(centers,centers,indexing='ij')
        self.register_buffer('centers',torch.stack((xx,yy),-1).reshape(-1,2))

    def sanitize(self, x):
        """Absent evidence cannot influence either raster or neural tokens."""
        s=self.spec
        own=x[...,:9]
        parts=[own]; records=[]; cursor=9
        for count,width,position_offset in ((s.agents-1,8,3),(s.targets,5,0),(s.threats,4,0)):
            raw=x[...,cursor:cursor+count*width].reshape(*x.shape[:2],count,width)
            cursor+=count*width
            valid=x[...,cursor:cursor+count]>0
            cursor+=count
            raw=torch.where(valid[...,None],raw,torch.zeros_like(raw))
            relative=raw[...,position_offset:position_offset+2]*s.length_scale/s.world_size
            f,r=relative.unbind(-1)
            positions=torch.stack((own[...,3,None]+f*own[...,5,None]+r*own[...,6,None],
                                   own[...,4,None]+f*own[...,6,None]-r*own[...,5,None]),-1)
            records.append((raw,positions,valid))
            parts.extend((raw.flatten(-2),valid.to(x.dtype)))
        if cursor != s.base_dim:raise ValueError('observation schema differs')
        history=x[...,cursor:cursor+s.targets*6].reshape(*x.shape[:2],s.targets,6)
        known=history[...,5]>0
        history=torch.where(known[...,None],history,torch.zeros_like(history))
        parts.extend((history.flatten(-2),x[...,cursor+s.targets*6:]))
        return torch.cat(parts,-1),records,history,known

    def encode_map(self, x, fixed=None, sanitized=None):
        # Reuse already sanitised evidence in the dual-scale forward path.
        x,records,history,known = sanitized if sanitized is not None else self.sanitize(x)
        n,b=x.shape[:2]; s=self.spec; own=x[...,:9]
        if fixed is None:fixed=rasterize(x.transpose(0,1),s).transpose(0,1)
        # Generalized spatial resolution: the established d128 actor remains bitwise
        # identical at 16x16, while the Lite actor can use 8x8 tokens.
        if 32 % self.map_size:raise ValueError('spatial grid must divide 32')
        anchor=F.avg_pool2d(fixed.reshape(n*b,9,32,32),32//self.map_size).reshape(n,b,9,self.map_size**2).transpose(-1,-2)
        centers=self.centers[None,None].expand(n,b,-1,-1)
        maps=norm(pointwise(self.anchor,torch.cat((anchor,centers),-1))+
                  self.own_map(own).unsqueeze(-2)).tanh()
        mate,target,threat=records
        raw=[own.unsqueeze(-2),mate[0],target[0],history,threat[0]]
        positions=[own[...,None,3:5],mate[1],target[1],history[...,:2],threat[1]]
        valid=[torch.ones(n,b,1,device=x.device,dtype=torch.bool),mate[2],target[2],known,threat[2]]
        entities=torch.cat([norm(pointwise(layer,torch.cat((r,p),-1))).tanh()
                            for layer,r,p in zip(self.types,raw,positions)],-2)
        pos=torch.cat(positions,-2); mask=torch.cat(valid,-1)
        ages=torch.cat([torch.zeros_like(v,dtype=x.dtype) if i!=3 else history[...,4]
                        for i,v in enumerate(valid)],-1)
        for i,block in enumerate(self.blocks):
            if torch.is_grad_enabled() and self.checkpoint_blocks:
                maps=checkpoint(block,maps,entities,pos,ages,mask,self.centers,use_reentrant=False)
            else:maps=block(maps,entities,pos,ages,mask,self.centers)
            if i==0:maps=self.local(maps)
        maps=norm(maps)
        delta=pointwise(self.correction,maps).tanh()
        evidence=(anchor[...,0]+.15*delta[...,0]).clamp_min(0)
        threat_map=(anchor[...,5]+.15*delta[...,1]).clamp(0,1)
        observed=anchor[...,7]
        # Fixed legal freshness is an anchor, not a ground-truth confidence label.
        reliability=(anchor[...,8]+.15*delta[...,2]).clamp(0,1)
        channels=torch.stack((evidence,threat_map,observed,1-observed,reliability),-1)
        return maps,channels

    def action_geometry(self,own):
        return candidate_geometry(own,self.spec)

    def action_controls(self,own):
        return own.new_tensor(ACTION_VALUES)

    def decode_map(self, maps, channels, own):
        motion,endpoint=self.action_geometry(own)
        n,b=own.shape[:2]
        controls=self.action_controls(own)[None,None,:,None].expand(n,b,-1,-1)
        action=torch.cat((own.unsqueeze(-2).expand(n,b,7,9),controls,motion),-1)
        tokens=torch.cat((maps,channels),-1)
        q=pointwise(self.query,action)
        k,v=pointwise(self.keys,tokens),pointwise(self.values,tokens)
        delta=self.centers[None,None,None]-endpoint.unsqueeze(-2)
        geometry=torch.cat((delta,delta.square().sum(-1,keepdim=True)),-1)
        bias=2*pointwise(self.path_bias,geometry).tanh().squeeze(-1)
        # Fuse score scaling, additive geometry bias, softmax, and V reduction.
        # Batch-by-agent axes stay independent; there is one logical SDPA head.
        n,b,choices,dim = q.shape
        tokens = k.shape[-2]
        context=F.scaled_dot_product_attention(
            q.reshape(n*b,1,choices,dim),
            k.reshape(n*b,1,tokens,dim),
            v.reshape(n*b,1,tokens,dim),
            attn_mask=bias.reshape(n*b,1,choices,tokens),
            dropout_p=0.0).reshape(n,b,choices,dim)
        return pointwise(self.score,torch.cat((action,context),-1)).squeeze(-1)

    def _forward(self,x):
        maps,channels=self.encode_map(x)
        return self.decode_map(maps,channels,x[...,:9])

    def forward(self,x):
        # Rematerialize independent row chunks, then form ONE full PPO loss.
        # No optimizer steps, gradient clipping or RNG draws occur per chunk.
        if torch.is_grad_enabled() and x.shape[1]>self.chunk_rows:
            return torch.cat([checkpoint(self._forward,x[:,start:start+self.chunk_rows],use_reentrant=False)
                              for start in range(0,x.shape[1],self.chunk_rows)],dim=1)
        return self._forward(x)


class MBDTMAPPO(StackedMAPPO):
    actor_type = MBDTActor
    def __init__(self,agents,obs_dim,state_dim,hidden=1024,value_normalization=False,belief_spec=None):
        if belief_spec is None:raise ValueError('MBDT requires legal history')
        spec=BeliefSpec(**belief_spec)
        if spec.dim!=obs_dim or spec.grid!=32:raise ValueError('MBDT requires matching 32x32 history')
        super().__init__(agents,spec.base_dim,state_dim,hidden,value_normalization=value_normalization)
        self.actor=self.actor_type(agents,hidden,spec)
