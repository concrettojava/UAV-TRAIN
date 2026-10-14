"""CPU numerical equivalence for speed-first fused model operators."""
import math
import torch
from torch.nn import functional as F

from envs import BatchedUAVEnv, EnvConfig
from models.belief import belief_spec
from models.mbdt_perf import DualScaleActor


def test_fused_candidate_attention_matches_explicit_softmax():
    torch.manual_seed(1234)
    agents, rows, choices, tokens, dim = 8, 3, 7, 64, 128
    q = torch.randn(agents, rows, choices, dim)
    k = torch.randn(agents, rows, tokens, dim)
    v = torch.randn(agents, rows, tokens, dim)
    bias = torch.randn(agents, rows, choices, tokens) * .1
    explicit = (q @ k.transpose(-1,-2) / math.sqrt(dim) + bias).softmax(-1) @ v
    fused = F.scaled_dot_product_attention(
        q.reshape(agents*rows,1,choices,dim),
        k.reshape(agents*rows,1,tokens,dim),
        v.reshape(agents*rows,1,tokens,dim),
        attn_mask=bias.reshape(agents*rows,1,choices,tokens),
        dropout_p=0.).reshape(agents,rows,choices,dim)
    torch.testing.assert_close(fused, explicit, atol=3e-6, rtol=3e-5)


def test_vectorized_trajectory_matches_stepwise_reference():
    torch.manual_seed(5678)
    cfg = EnvConfig()
    env = BatchedUAVEnv(cfg, 1)
    spec = belief_spec(cfg, env.obs_dim_flat, 32)
    actor = DualScaleActor(cfg.n_uavs, 64, spec)
    own = torch.randn(cfg.n_uavs, 3, 9)
    own[..., :3] = 0.
    own[..., 0] = 1.
    own[..., 3:5] = torch.rand(cfg.n_uavs, 3, 2)
    own[..., 5:7] = torch.randn(cfg.n_uavs,3,2)
    got_paths,got_endpoint = actor.action_geometry(own)
    controls = actor.action_controls(own)
    speeds = actor.public_speeds.to(dtype=own.dtype)
    turns = actor.public_turns.to(dtype=own.dtype)
    speed = (own[...,:3]*speeds).sum(-1) / spec.world_size
    turn = (own[...,:3]*turns).sum(-1)
    yaw = torch.atan2(own[...,5],own[...,6])
    offsets = []
    origin = own[...,3:5].unsqueeze(-2)
    location = origin.expand(*own.shape[:2],7,2)
    for step in range(1,17):
        angle = yaw.unsqueeze(-1) + step*spec.dt*turn.unsqueeze(-1)*controls
        delta = speed[...,None,None]*spec.dt*torch.stack((angle.sin(),angle.cos()),-1)
        location = location + delta
        if step in (1,4,8,16):
            offsets.append(location-origin)
    old_paths = torch.cat(offsets,-1)
    torch.testing.assert_close(got_paths,old_paths,atol=1e-6,rtol=1e-5)
    torch.testing.assert_close(got_endpoint,location,atol=1e-6,rtol=1e-5)
