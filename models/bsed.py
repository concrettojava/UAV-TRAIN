"""BSED: age-conditioned track fields and action-aligned spatial decoding.

Inputs are replayable legal observations/history. No neural recurrent state,
truth labels, action filters, or privileged critic inputs enter this actor.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .belief import BeliefSpec, rasterize
from .cognitive import AgentConv, CognitiveMAPPO, deterministic_pool4
from .stacked import StackedMAPPO, AgentLinear, AgentMLP
from envs.config import ACTION_VALUES, TYPE_NAMES, TYPE_PARAMS


def pointwise(layer, x):
    """Apply independent AgentLinear to arbitrary agent-first spatial axes."""
    shape = x.shape
    return layer(x.reshape(shape[0], -1, shape[-1])).reshape(*shape[:-1], -1)


class TrackField(nn.Module):
    def __init__(self, spec, mode='learned'):
        super().__init__()
        if mode not in ('learned', 'cv', 'stationary', 'cv_residual'):
            raise ValueError('unknown track-field mode')
        self.spec = spec
        self.mode = mode
        self.condition = AgentMLP(spec.agents, 15, 4, 32, .01)
        # Retain the same draws/state layout so all downstream and critic
        # initial weights match the learned model exactly for the same seed.
        if mode not in ('learned', 'cv_residual'):
            self.condition.requires_grad_(False)
        if mode == 'cv_residual':
            # Keep all initialization draws; only the field's output is zero.
            # Initially identical to CV. Upstream field gradients start at zero.
            with torch.no_grad():
                self.condition.layers[-1].weight.zero_()
                self.condition.layers[-1].bias.zero_()

    def forward(self, x):
        s = self.spec
        slots = x[..., s.base_dim:s.base_dim+s.targets*6].reshape(*x.shape[:2], s.targets, 6)
        own = x[..., :9]
        context = own.unsqueeze(-2).expand(*slots.shape[:-1], 9)
        age, known = slots[..., 4].clamp_min(0), slots[..., 5]
        elapsed = age * s.max_steps * s.dt
        # The last observed heading is uncertain. The original learned mode
        # bounds drift by public speed/age; CV residual allows up to twice it
        # as a task field, not a physically calibrated position prediction.
        # All modes keep fresh positions fixed.
        base = .5 / s.grid
        radius = s.target_speed * elapsed / s.world_size
        if self.mode == 'learned':
            controls = pointwise(self.condition, torch.cat((slots, context), -1))
            drift = torch.sigmoid(controls[..., 0]) * s.target_speed * elapsed / s.world_size
            longitudinal = base + radius * (.25 + .75 * torch.sigmoid(controls[..., 1]))
            lateral = base + radius * (.25 + .75 * torch.sigmoid(controls[..., 2]))
            confidence = known * torch.exp(-(1 + 4 * torch.sigmoid(controls[..., 3])) * age)
        elif self.mode == 'cv_residual':
            controls = pointwise(self.condition, torch.cat((slots, context), -1)).tanh()
            drift = radius * (1 + controls[..., 0])
            longitudinal = base + radius * (1 + .75 * controls[..., 1])
            lateral = base + radius * (1 + .75 * controls[..., 2])
            confidence = known * torch.exp(-(3 + 2 * controls[..., 3]) * age)
        else:
            drift = radius if self.mode == 'cv' else torch.zeros_like(radius)
            longitudinal = lateral = base + radius
            confidence = known * torch.exp(-3 * age)
        mean = slots[..., :2] + drift.unsqueeze(-1) * slots[..., 2:4]
        centers = (torch.arange(s.grid, device=x.device, dtype=x.dtype) + .5) / s.grid
        yy, xx = torch.meshgrid(centers, centers, indexing='ij')
        dx, dy = xx-mean[..., 0, None, None], yy-mean[..., 1, None, None]
        hs, hc = slots[..., 2, None, None], slots[..., 3, None, None]
        along, across = dx*hs+dy*hc, dx*hc-dy*hs
        energy = -.5 * ((along/longitudinal[..., None, None]).square()
                         + (across/lateral[..., None, None]).square())
        mass = energy.flatten(-2).softmax(-1).reshape_as(energy) * confidence[..., None, None]
        target = mass.sum(-3)
        fresh = (mass*(age == 0)[..., None, None]).sum(-3)
        uncertainty = (mass*((longitudinal+lateral)*s.grid/2)[..., None, None]).sum(-3)
        age_field = (mass*age[..., None, None]).sum(-3)
        return torch.stack((target, fresh, uncertainty, (mass*hs).sum(-3),
                            (mass*hc).sum(-3), age_field), -3)


def candidate_geometry(own, spec):
    """[N,B,9] -> seven held-control public kinematic rollouts/features.

    One-step dynamics exactly match the environment. Longer horizons (4,8,16)
    are hypothetical own-control paths, not future battlefield predictions.
    Coordinates outside the arena are kept so geometry can express boundaries.
    """
    controls = own.new_tensor(ACTION_VALUES)
    speeds = own.new_tensor([TYPE_PARAMS[t]['speed'] for t in TYPE_NAMES])
    turns = own.new_tensor([math.radians(TYPE_PARAMS[t]['max_turn_deg_s']) for t in TYPE_NAMES])
    speed = (own[..., :3]*speeds).sum(-1) / spec.world_size
    turn = (own[..., :3]*turns).sum(-1)
    yaw = torch.atan2(own[..., 5], own[..., 6])
    offsets = []
    location = own[..., 3:5].unsqueeze(-2).expand(*own.shape[:2], 7, 2)
    for step in range(1, 17):
        angle = yaw.unsqueeze(-1) + step*spec.dt*turn.unsqueeze(-1)*controls
        delta = speed[..., None, None]*spec.dt*torch.stack((angle.sin(), angle.cos()), -1)
        location = location + delta
        if step in (1, 4, 8, 16):
            offsets.append(location-own[..., 3:5].unsqueeze(-2))
    features = torch.cat(offsets, -1)
    return features, location


class BSEDActor(nn.Module):
    def __init__(self, agents, hidden, spec, learned_encoder=True, spatial_decoder=True, field_mode='learned', target_vector=True):
        super().__init__()
        self.spec, self.spatial_decoder = spec, spatial_decoder
        self.target_vector = target_vector
        self.field = TrackField(spec, field_mode) if learned_encoder else None
        self.convs = nn.ModuleList([AgentConv(agents, 10, 16), AgentConv(agents, 16, 32), AgentConv(agents, 32, 48)])
        self.global_projection = AgentLinear(agents, 48*16, 128, math.sqrt(2))
        self.residual = AgentMLP(agents, spec.base_dim+spec.targets*6+128, 7, hidden, .01)
        if spatial_decoder:
            self.keys = AgentLinear(agents, 96, 32, math.sqrt(2))
            self.values = AgentLinear(agents, 96, 64, math.sqrt(2))
            self.query = AgentMLP(agents, 17, 36, 64, math.sqrt(2))
            self.score = AgentMLP(agents, 81, 1, 64, .01)
        centers = (torch.arange(8)+.5)/8
        yy, xx = torch.meshgrid(centers, centers, indexing='ij')
        self.register_buffer('centers', torch.stack((xx, yy), -1).reshape(64, 2))

    def situation(self, x):
        # Fixed context fields use the existing audited legal rasterizer.
        fixed = rasterize(x.transpose(0, 1), self.spec).transpose(0, 1)
        if self.field is not None:
            target = self.field(x)
        else:
            target = torch.cat((fixed[..., :5, :, :], torch.zeros_like(fixed[..., :1, :, :])), -3)
        return torch.cat((target, fixed[..., 5:, :, :]), -3)

    def compact(self, x):
        """Direct vector path; map generation always receives original x."""
        s=self.spec
        compact=x[..., :s.base_dim+s.targets*6]
        if self.target_vector:return compact
        compact=compact.clone()
        start=9+(s.agents-1)*9  # own, mate features and mate masks
        compact[...,start:start+s.targets*6]=0  # current features + masks
        compact[...,s.base_dim:]=0  # anonymous history slots
        return compact

    def forward(self, x):
        spatial = self.situation(x).transpose(0, 1)
        scales = []
        for conv in self.convs:
            spatial = F.relu(conv(spatial))
            scales.append(spatial)
        pooled = deterministic_pool4(spatial).flatten(-3).transpose(0, 1)
        global_features = torch.tanh(self.global_projection(pooled))
        logits = self.residual(torch.cat((self.compact(x), global_features), -1))
        if not self.spatial_decoder:
            return logits
        # Keep local and global scales at 8x8. Deterministic average pooling
        # and nearest upsampling avoid CUDA grid_sample atomic backward.
        B, N = x.shape[1], x.shape[0]
        fine = F.avg_pool2d(scales[0].reshape(B*N, 16, *scales[0].shape[-2:]), 2)
        middle = scales[1].reshape(B*N, 32, *scales[1].shape[-2:])
        coarse = F.interpolate(scales[2].reshape(B*N, 48, *scales[2].shape[-2:]), size=(8, 8), mode='nearest')
        tokens = torch.cat((fine, middle, coarse), 1).reshape(B, N, 96, 64).permute(1, 0, 3, 2)
        motion, endpoint = candidate_geometry(x[..., :9], self.spec)
        own = x[..., :9].unsqueeze(-2).expand(N, B, 7, 9)
        query = pointwise(self.query, torch.cat((own, motion), -1))
        keys, values = pointwise(self.keys, tokens), pointwise(self.values, tokens)
        content = torch.einsum('nbak,nbsk->nbas', query[..., :32], keys)/math.sqrt(32)
        delta = self.centers[None, None, None, :, :]-endpoint.unsqueeze(-2)
        hs, hc = x[..., 5, None, None], x[..., 6, None, None]
        forward = delta[..., 0]*hs+delta[..., 1]*hc
        right = delta[..., 0]*hc-delta[..., 1]*hs
        geometry = torch.stack((forward, right, delta.square().sum(-1), forward.square()), -1)
        alignment = (geometry*query[..., 32:].unsqueeze(-2)).sum(-1)
        attention = (content+alignment).softmax(-1)
        context = torch.einsum('nbas,nbsk->nbak', attention, values)
        return logits+pointwise(self.score, torch.cat((own, motion, context), -1)).squeeze(-1)


class BSEDMAPPO(StackedMAPPO):
    learned_encoder = True
    spatial_decoder = True
    field_mode = 'learned'
    target_vector = True

    def __init__(self, agents, obs_dim, state_dim, hidden=1024, value_normalization=False, belief_spec=None):
        if belief_spec is None:
            raise ValueError('BSED requires legal history schema')
        spec = BeliefSpec(**belief_spec)
        if spec.dim != obs_dim or spec.grid != 32:
            raise ValueError('BSED v1 requires matching 32x32 history schema')
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        super().__init__(agents, spec.base_dim, state_dim, hidden, value_normalization=value_normalization)
        self.actor = BSEDActor(agents, hidden, spec, self.learned_encoder, self.spatial_decoder, self.field_mode, self.target_vector)


class BSEDFixedEncoder(BSEDMAPPO):
    learned_encoder = False


class BSEDGlobalDecoder(BSEDMAPPO):
    spatial_decoder = False


class BSEDCVEncoder(BSEDMAPPO):
    field_mode = 'cv'


class BSEDStationaryEncoder(BSEDMAPPO):
    field_mode = 'stationary'


class BSEDNoTargetVector(BSEDMAPPO):
    """No explicit target vector bypass; legal maps/global map pooling remain."""
    target_vector = False


class BSEDCVResidualEncoder(BSEDMAPPO):
    """CV-initialized task field with bounded learned corrections, not a tracker."""
    field_mode = 'cv_residual'


class BSEDCapacityCNN(CognitiveMAPPO):
    """Original CNN with actor count matched to BSED v1, critic unchanged.

    Only the original actor MLP width changes; no new spatial mechanisms.
    The exact count is checked against the actual BSED actor by tests.
    """
    def __init__(self, agents, obs_dim, state_dim, hidden=1024, value_normalization=False, belief_spec=None):
        super().__init__(agents,obs_dim,state_dim,hidden,value_normalization,belief_spec)
        spec=self.actor.spec
        compact=spec.base_dim+spec.targets*6
        # BSED's field/CNN/projection/query/key/value/score have 146537
        # parameters per UAV; its global residual has a 2-layer hidden MLP.
        target=hidden*hidden+(compact+128+9)*hidden+7+146537
        original_fixed=sum(p.numel() for p in self.actor.convs.parameters())//agents
        original_fixed+=sum(p.numel() for p in self.actor.projection.parameters())//agents
        coefficient=compact+64+9
        root=(-coefficient+math.sqrt(coefficient**2+4*(target-original_fixed-7)))/2
        width=max(1,round(root))
        self.actor.head=AgentMLP(agents,compact+64,7,width,.01)
        self.actor.capacity_width=width
