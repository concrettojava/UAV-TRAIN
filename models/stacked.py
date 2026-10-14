"""Independent agent MLPs evaluated together; no parameter sharing."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.distributions import Categorical
from .value_norm import ValueNorm


class AgentLinear(nn.Module):
    def __init__(self, agents: int, inputs: int, outputs: int, gain: float):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(agents, inputs, outputs))
        self.bias = nn.Parameter(torch.zeros(agents, 1, outputs))
        for weight in self.weight:
            nn.init.orthogonal_(weight, gain=gain)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.bmm(x, self.weight) + self.bias


class AgentMLP(nn.Module):
    def __init__(self, agents: int, inputs: int, outputs: int, hidden: int, gain: float):
        super().__init__()
        self.layers = nn.Sequential(
            AgentLinear(agents, inputs, hidden, math.sqrt(2)), nn.Tanh(),
            AgentLinear(agents, hidden, hidden, math.sqrt(2)), nn.Tanh(),
            AgentLinear(agents, hidden, outputs, gain),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class StackedMAPPO(nn.Module):
    def __init__(self, agents: int, obs_dim: int, state_dim: int,
                 hidden: int = 256, actions: int = 7, value_normalization: bool = False):
        super().__init__()
        self.agents = agents
        self.actor = AgentMLP(agents, obs_dim, actions, hidden, 0.01)
        self.critic = AgentMLP(agents, state_dim, 1, hidden, 1.0)
        self.value_norm = ValueNorm(agents) if value_normalization else None

    def distribution(self, obs: torch.Tensor) -> Categorical:
        # external tensors [batch, agent, feature]; bmm [agent, batch, feature]
        # Keep PPO log probabilities, ratios and KL in FP32 under BF16 autocast.
        return Categorical(logits=self.actor(obs.transpose(0, 1)).transpose(0, 1).float())

    def normalized_values(self, state: torch.Tensor) -> torch.Tensor:
        x = state.unsqueeze(0).expand(self.agents, -1, -1)
        return self.critic(x).squeeze(-1).transpose(0, 1).float()

    def values(self, state: torch.Tensor) -> torch.Tensor:
        values = self.normalized_values(state)
        return self.value_norm.denormalize(values) if self.value_norm is not None else values

    @torch.no_grad()
    def act(self, obs: torch.Tensor, state: torch.Tensor, deterministic: bool = False):
        dist = self.distribution(obs)
        actions = dist.logits.argmax(-1) if deterministic else dist.sample()
        return actions, dist.log_prob(actions), self.values(state)
