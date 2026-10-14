"""Independent per-agent grouped CNN, retaining the original global critic."""
import math
from contextlib import contextmanager
import torch
from torch import nn
from torch.nn import functional as F
from .stacked import StackedMAPPO,AgentLinear,AgentMLP
from .belief import BeliefSpec,rasterize


class AgentConv(nn.Module):
    def __init__(self,agents,cin,cout):
        super().__init__()
        self.agents,self.cout=agents,cout
        self.weight=nn.Parameter(torch.empty(agents,cout,cin,3,3))
        self.bias=nn.Parameter(torch.zeros(agents,cout))
        for w in self.weight:nn.init.kaiming_uniform_(w,a=math.sqrt(5))

    def forward(self,x):
        # x [batch, agent, channel, height, width]; groups keep policies separate.
        B,N,C,H,W=x.shape
        y=F.conv2d(x.reshape(B,N*C,H,W),self.weight.flatten(0,1),
                   self.bias.flatten(),stride=2,padding=1,groups=N)
        return y.reshape(B,N,self.cout,*y.shape[-2:])


class CognitiveActor(nn.Module):
    def __init__(self,agents,hidden,spec):
        super().__init__();self.spec=spec
        self.convs=nn.ModuleList([AgentConv(agents,9,16),AgentConv(agents,16,32),AgentConv(agents,32,32)])
        self.projection=AgentLinear(agents,32*4*4,64,math.sqrt(2))
        self.head=AgentMLP(agents,spec.base_dim+spec.targets*6+64,7,hidden,.01)
        self._cached_input = None
        self._cached_maps = None

    @contextmanager
    def cache_raster(self, obs):
        """Reuse observation-only maps within one immutable full-batch update.

        Cache no learned embeddings: every epoch still runs the complete CNN.
        Keep a live tensor reference and version, so allocator reuse or mutation
        cannot silently pair a different observation with a stale map.
        """
        if self._cached_input is not None:
            raise RuntimeError('nested raster cache')
        if obs.requires_grad:
            raise ValueError('raster cache requires constant observations')
        self._cached_input = obs.transpose(0, 1)
        self._cached_version = obs._version
        try:
            self._cached_maps = rasterize(obs, self.spec)
            yield
        finally:
            self._cached_input = self._cached_maps = None

    def forward(self,x):
        # Existing independent model contract is [agent,batch,feature].
        cached = self._cached_input
        if (cached is not None and x.data_ptr() == cached.data_ptr()
                and x.shape == cached.shape and x.stride() == cached.stride()
                and x._version == self._cached_version):
            spatial = self._cached_maps
        else:
            spatial=rasterize(x.transpose(0,1),self.spec)
        for conv in self.convs:spatial=F.relu(conv(spatial))
        B,N,C,H,W=spatial.shape
        # Explicit adaptive bins have the same means, with a deterministic
        # backward. CUDA adaptive_avg_pool2d uses atomic gradient additions.
        spatial=deterministic_pool4(spatial).reshape(B,N,-1).transpose(0,1)
        embedding=torch.tanh(self.projection(spatial))
        return self.head(torch.cat((x[...,:self.spec.base_dim+self.spec.targets*6],embedding),-1))


class CognitiveMAPPO(StackedMAPPO):
    def __init__(self,agents,obs_dim,state_dim,hidden=256,value_normalization=False,belief_spec=None):
        if belief_spec is None:raise ValueError('cognitive model needs public belief schema')
        spec=BeliefSpec(**belief_spec)
        if spec.dim!=obs_dim:raise ValueError('actor input schema differs')
        # Process-local backend choice, including convolution backward.
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark=False
        # Avoid allocating/initializing an unused augmented-input MLP.
        super().__init__(agents,spec.base_dim,state_dim,hidden,value_normalization=value_normalization)
        self.actor=CognitiveActor(agents,hidden,spec)


def deterministic_pool4(x):
    h,w=x.shape[-2:]
    if h == 4 and w == 4:
        # Every adaptive bin is exactly one cell; identity also preserves the
        # original deterministic backward without 16 means and stack kernels.
        return x
    rows=[]
    for i in range(4):
        columns=[]
        for j in range(4):
            patch=x[...,i*h//4:((i+1)*h+3)//4,j*w//4:((j+1)*w+3)//4]
            columns.append(patch.mean(dim=(-2,-1)))
        rows.append(torch.stack(columns,-1))
    return torch.stack(rows,-2)


class HistoricalMAPPO(StackedMAPPO):
    def __init__(self,agents,obs_dim,state_dim,hidden=256,value_normalization=False,belief_spec=None):
        if belief_spec is None:raise ValueError('history model needs public belief schema')
        spec=BeliefSpec(**belief_spec)
        if spec.dim!=obs_dim:raise ValueError('actor input schema differs')
        # Give A/B/C exactly the same initial critic for the same seed. Changing
        # actor input size otherwise changes how many random values precede it.
        super().__init__(agents,spec.base_dim,state_dim,hidden,value_normalization=value_normalization)
        self.actor=AgentMLP(agents,obs_dim,7,hidden,.01)
