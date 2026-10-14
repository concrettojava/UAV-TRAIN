"""Preallocated CPU rollout storage and boundary-aware GAE."""
from __future__ import annotations

import numpy as np
import torch

REWARD, LOGP, VALUE, ACTIVE, BOOTSTRAP, TRACE, OVERRIDE = range(7)


def compute_gae(rewards, values, next_values, bootstrap, trace, active,
                gamma: float, lam: float):
    """Bootstrap and trace masks differ at artificial time-limit truncations."""
    advantages = torch.empty_like(rewards)
    tail = torch.zeros_like(rewards[0])
    for t in range(rewards.shape[0] - 1, -1, -1):
        delta = rewards[t] + gamma * bootstrap[t] * next_values[t] - values[t]
        tail = (delta + gamma * lam * trace[t] * tail) * active[t]
        advantages[t] = tail
    return advantages, advantages + values


class RolloutBuffer:
    def __init__(self, steps: int, envs: int, agents: int, obs_dim: int,
                 state_dim: int, pin_memory: bool):
        self.shape = (steps, envs, agents)
        self.valid_steps = steps
        # Single float allocation and a separate int action allocation. Pinned
        # storage is used only when CUDA is actually available.
        size = steps * envs * (agents * (obs_dim + 7) + state_dim)
        self.storage = torch.empty(size, dtype=torch.float32, pin_memory=pin_memory)
        cursor = steps * envs * agents * obs_dim
        self.obs = self.storage[:cursor].view(steps, envs, agents, obs_dim).numpy()
        count = steps * envs * state_dim
        self.state = self.storage[cursor:cursor + count].view(steps, envs, state_dim).numpy()
        cursor += count
        self.data = self.storage[cursor:].view(steps, envs, agents, 7).numpy()
        self.actions_storage = torch.empty(self.shape, dtype=torch.int64, pin_memory=pin_memory)
        self.actions = self.actions_storage.numpy()

    def batch(self, device, last_values, gamma: float, lam: float):
        steps, envs, agents = self.shape
        used = self.valid_steps
        if not 1 <= used <= steps:
            raise ValueError('invalid collected trajectory length')
        packed = self.storage.to(device, non_blocking=True)
        od = self.obs.shape[-1]
        sd = self.state.shape[-1]
        cursor = steps * envs * agents * od
        obs = packed[:cursor].view(steps, envs, agents, od)[:used].reshape(used * envs, agents, od)
        count = steps * envs * sd
        state = packed[cursor:cursor + count].view(steps, envs, sd)[:used].reshape(used * envs, sd)
        d = packed[cursor + count:].view(steps, envs, agents, 7)[:used]
        values = d[..., VALUE]
        next_values = torch.cat([values[1:], last_values.unsqueeze(0)], dim=0)
        # At truncation, next episode's initial value must be replaced by final value.
        next_values = torch.where((d[..., TRACE] == 0) & (d[..., BOOTSTRAP] > 0),
                                  d[..., OVERRIDE], next_values)
        adv, returns = compute_gae(d[..., REWARD], values, next_values,
                                  d[..., BOOTSTRAP], d[..., TRACE], d[..., ACTIVE], gamma, lam)
        active = d[..., ACTIVE]
        counts = active.sum((0, 1)).clamp_min(1)
        means = (adv * active).sum((0, 1)) / counts
        variance = ((adv - means).square() * active).sum((0, 1)) / counts
        adv = (adv - means) / (variance.sqrt() + 1e-8)
        batch = {
            "obs": obs, "state": state,
            "actions": self.actions_storage.to(device, non_blocking=True)[:used].reshape(-1, agents),
            "logp": d[..., LOGP].reshape(-1, agents),
            "values": values.reshape(-1, agents),
            "active": active.reshape(-1, agents),
            "advantages": adv.reshape(-1, agents),
            "returns": returns.reshape(-1, agents),
        }
        # A finished task waits for its peers; padded env-time rows are not samples.
        rows = batch['active'].bool().any(-1)
        return {key:value[rows] for key,value in batch.items()}
