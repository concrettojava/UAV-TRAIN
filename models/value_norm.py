"""Masked per-agent running target statistics with PopArt output preservation."""
import torch
from torch import nn


class ValueNorm(nn.Module):
    def __init__(self, agents):
        super().__init__()
        for key in ("count", "mean", "m2"):
            self.register_buffer(key, torch.zeros(agents, dtype=torch.float64))

    def scale(self):
        variance = self.m2 / self.count.clamp_min(1)
        return torch.where(self.count > 0, variance.clamp_min(1e-4).sqrt(),
                           torch.ones_like(variance)).float()

    def normalize(self, values):
        return (values - self.mean.float()) / self.scale()

    def denormalize(self, values):
        return values * self.scale() + self.mean.float()

    @torch.no_grad()
    def update(self, targets, active, output):
        # Parallel Welford merge; dead samples never influence statistics.
        mask = active.double()
        values = targets.double()
        n = mask.sum(0)
        mean = (values * mask).sum(0) / n.clamp_min(1)
        m2 = ((values - mean).square() * mask).sum(0)
        old_mean, old_scale = self.mean.float().clone(), self.scale()
        total = self.count + n
        delta = mean - self.mean
        self.mean.add_(delta * n / total.clamp_min(1))
        self.m2.add_(m2 + delta.square() * self.count * n / total.clamp_min(1))
        self.count.copy_(total)
        new_scale = self.scale()
        # Preserve every raw V(s), not only predictions at sampled states.
        ratio = (old_scale / new_scale).view(-1, 1, 1)
        output.weight.mul_(ratio)
        output.bias.mul_(ratio).add_(((old_mean - self.mean.float()) / new_scale).view(-1, 1, 1))
