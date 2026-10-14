"""Research-only MBDT variants: device-safe public geometry, one remat level.

No changes to observations, PPO batch, action set or independent parameters.
The d128 variant has an identical persistent state_dict to frozen MBDT v1.
"""
import math
import torch
from torch.nn import functional as F
from torch.nn.attention import sdpa_kernel,SDPBackend
from .mbdt import MBDTActor, MBDTMAPPO, EvidenceBlock, norm
from .bsed import pointwise
from .belief import rasterize
from .stacked import AgentLinear
from envs.config import ACTION_VALUES,TYPE_NAMES,TYPE_PARAMS


class SplitKLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,weight,bias):
        ctx.save_for_backward(x,weight)
        return torch.bmm(x,weight)+bias

    @staticmethod
    def backward(ctx,grad):
        x,weight=ctx.saved_tensors
        n,rows,inputs=x.shape;outputs=weight.shape[-1]
        gx=torch.bmm(grad,weight.transpose(1,2)) if ctx.needs_input_grad[0] else None
        # Parallelize the very long K reduction, retaining independent UAV axes.
        # These are gradient partials, not policy minibatches/optimizer steps.
        splits=16 if rows%16==0 else 1
        left=x.reshape(n*splits,rows//splits,inputs).transpose(1,2)
        right=grad.reshape(n*splits,rows//splits,outputs)
        gw=torch.bmm(left,right).reshape(n,splits,inputs,outputs).sum(1)
        gb=grad.sum(1,keepdim=True)
        return gx,gw,gb


class ParallelGradientLinear(AgentLinear):
    def __init__(self,original):
        torch.nn.Module.__init__(self)
        # Preserve parameter objects, names and initialization without RNG draws.
        self.weight=original.weight;self.bias=original.bias

    def forward(self,x):
        if x.device.type=='cuda' and torch.is_grad_enabled() and x.shape[1]>=4096:
            return SplitKLinearFunction.apply(x,self.weight,self.bias)
        return super().forward(x)


def parallelize_long_gradients(parent):
    for name,child in list(parent.named_children()):
        if isinstance(child,AgentLinear):
            setattr(parent,name,ParallelGradientLinear(child))
        else:parallelize_long_gradients(child)


class PerformanceEvidenceBlock(EvidenceBlock):
    """Flatten independent batch axes for 4D SDPA dispatch, never attend across them."""
    def forward(self,maps,entities,positions,ages,valid,centers):
        n,b,j,d=maps.shape;h=self.heads
        def split(layer,x):
            return pointwise(layer,x).reshape(n,b,-1,h,d//h).transpose(-3,-2)
        q,k,v=split(self.q,norm(maps)),split(self.k,entities),split(self.v,entities)
        delta=centers[None,None,:,None,:]-positions.unsqueeze(-3)
        geometry=torch.cat((delta,delta.square().sum(-1,keepdim=True)),-1)
        bias=2*pointwise(self.geometry,geometry).tanh().permute(0,1,4,2,3)
        bias=bias+2*pointwise(self.age,ages[...,None]).tanh().permute(0,1,3,2).unsqueeze(-2)
        bias=bias.masked_fill(~valid[:,:,None,None,:],float('-inf'))
        def attend():
            return F.scaled_dot_product_attention(q.reshape(n*b,h,j,d//h),
                k.reshape(n*b,h,-1,d//h),v.reshape(n*b,h,-1,d//h),
                attn_mask=bias.contiguous().reshape(n*b,h,j,-1),dropout_p=0.)
        if maps.device.type=='cpu':
            # Retain v1's CPU math reference. CPU 4D otherwise selects a different
            # flash kernel whose near-zero bias gradients affect Adam rounding.
            with sdpa_kernel(SDPBackend.MATH):context=attend()
        else:context=attend()
        context=context.reshape(n,b,h,j,d//h).transpose(-3,-2).reshape(n,b,j,d)
        maps=maps+pointwise(self.out,context)
        return maps+pointwise(self.ff2,F.gelu(pointwise(self.ff1,norm(maps))))


class PerformanceActor(MBDTActor):
    # Avoid nested block+whole-actor recomputation. Full PPO loss remains intact.
    checkpoint_blocks = False
    rematerialize = False
    block_type = PerformanceEvidenceBlock

    def forward(self,x):
        # Speed-first full-batch actor: no Python chunks or activation recomputation.
        return self._forward(x)

    def __init__(self,agents,hidden,spec):
        super().__init__(agents,hidden,spec)
        parallelize_long_gradients(self)
        for name,value in (
            ('public_controls',ACTION_VALUES.tolist()),
            ('public_speeds',[TYPE_PARAMS[t]['speed'] for t in TYPE_NAMES]),
            ('public_turns',[math.radians(TYPE_PARAMS[t]['max_turn_deg_s']) for t in TYPE_NAMES])):
            self.register_buffer(name,torch.tensor(value,dtype=torch.float64),persistent=False)

    def action_controls(self,own):
        return self.public_controls.to(dtype=own.dtype)

    def action_geometry(self,own):
        # Preserve original operation order: turn first, then move at each step.
        controls=self.action_controls(own)
        speeds=self.public_speeds.to(dtype=own.dtype)
        turns=self.public_turns.to(dtype=own.dtype)
        speed=(own[...,:3]*speeds).sum(-1)/self.spec.world_size
        turn=(own[...,:3]*turns).sum(-1)
        yaw=torch.atan2(own[...,5],own[...,6])
        # Vectorize all 16 public trajectory integration steps into one tensor
        # program, then sample legal positions at steps 1, 4, 8 and 16.
        ticks=torch.arange(1,17,device=own.device,dtype=own.dtype)
        angle=(yaw[...,None,None]+
               ticks*self.spec.dt*turn[...,None,None]*controls[None,None,:,None])
        delta=speed[...,None,None,None]*self.spec.dt*torch.stack(
            (angle.sin(),angle.cos()),-1)
        displacement=delta.cumsum(dim=-2)
        offsets=displacement[..., (0,3,7,15), :].reshape(*own.shape[:2],7,8)
        location=own[...,None,3:5]+displacement[...,-1,:]
        return offsets,location


class Actor192(PerformanceActor):
    dim,heads,depth=192,6,3


class Actor256x3(PerformanceActor):
    dim,heads,depth=256,8,3
    # Full 1024-row loss retained; 256-row remat exceeded 4090 capacity.
    chunk_rows=128


class Actor256x4(PerformanceActor):
    dim,heads,depth=256,8,4
    chunk_rows=128


class Performance128(MBDTMAPPO):
    actor_type=PerformanceActor


class Performance192(MBDTMAPPO):
    actor_type=Actor192


class Performance256x3(MBDTMAPPO):
    actor_type=Actor256x3


class Performance256x4(MBDTMAPPO):
    actor_type=Actor256x4



class Lite64Actor(PerformanceActor):
    """Low-cost 8x8 spatial evidence, two 96D attention blocks and legal action scoring.

    Unlike simple pooling of the final actions, this keeps the evidence/motion
    bottleneck intact. Use model hidden=256 for both candidate score and critic.
    """
    map_size, dim, heads, depth = 8, 96, 4, 2


class MBDTLite64(MBDTMAPPO):
    """Research-only compact architecture; not compatible with full-model weights."""
    actor_type = Lite64Actor



class DualScaleActor(PerformanceActor):
    """Evidence-global 8x8 and path-local exact 32x32 legal raster, per-agent.

    Keep d128 and two evidence blocks for safe partial teacher transfer.
    Only 7 candidate physical trajectories query 32x32 high-resolution evidence.
    """
    map_size,dim,heads,depth = 8,128,4,2

    def __init__(self,agents,hidden,spec):
        super().__init__(agents,hidden,spec)
        # 4 physically simulated path coordinates * 9 public raster channels.
        self.path_local = AgentLinear(agents,4*9,32,math.sqrt(2))
        self.path_local_out = AgentLinear(agents,32,1,1.)
        # Initially acts like the coarse-only model, then learns refinements.
        with torch.no_grad():
            self.path_local_out.weight.zero_()
            self.path_local_out.bias.zero_()

    def _forward(self,x):
        # Reuse exactly one raster construction between global and local paths.
        # Sanitize hidden/invalid packets before any raster operations.
        # Otherwise masked history can induce NaN before its known-mask multiplies.
        cleaned=self.sanitize(x)
        legal=cleaned[0]
        fixed=rasterize(legal.transpose(0,1),self.spec).transpose(0,1)
        maps,channels=self.encode_map(legal,fixed=fixed,sanitized=cleaned)
        own=x[...,:9]
        logits=self.decode_map(maps,channels,own)
        motion,_=self.action_geometry(own)
        n,b=x.shape[:2]
        trajectories=own[...,None,None,3:5]+motion.reshape(n,b,7,4,2)
        # grid_sample align_corners=False matches 32x32 cell centers exactly.
        grid=(trajectories*2-1).reshape(n*b,7,4,2)
        local=F.grid_sample(fixed.reshape(n*b,9,32,32),grid,
                            mode='bilinear',padding_mode='zeros',align_corners=False)
        features=local.permute(0,2,3,1).reshape(n,b,7,36)
        correction=pointwise(self.path_local_out,
                             torch.tanh(pointwise(self.path_local,features))).squeeze(-1)
        return logits+correction


class MBDTDualScale(MBDTMAPPO):
    actor_type=DualScaleActor
