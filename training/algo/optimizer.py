"""Adam with separate clocks/moments for stacked independent agents."""
import torch


class AgentAdam(torch.optim.Optimizer):
    def __init__(self, params, lr):
        super().__init__(params, dict(lr=lr, betas=(0.9, 0.999), eps=1e-8))

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        for group in self.param_groups:
            if 'agent_lr' in group:
                group['agent_lr'] = group['agent_lr'].to(group['params'][0].device)
        # PyTorch's generic optimizer loader treats a key named "step" as a
        # CPU scalar unless capturable/fused is set. Here it is an agent vector
        # used in device arithmetic, so all state must follow the parameter.
        for parameter, state in self.state.items():
            for key, value in state.items():
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(parameter.device)

    @torch.no_grad()
    def step(self, active):
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                state = self.state[p]
                shape = (p.shape[0],) + (1,) * (p.ndim - 1)
                mask = active.view(shape)
                if not state:
                    state["step"] = torch.zeros(shape, device=p.device)
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                state["step"].add_(mask)
                m, v = state["exp_avg"], state["exp_avg_sq"]
                m.copy_(torch.where(mask, beta1 * m + (1 - beta1) * p.grad, m))
                v.copy_(torch.where(mask, beta2 * v + (1 - beta2) * p.grad.square(), v))
                clock = state["step"].clamp_min(1)
                update = (m / (1 - beta1 ** clock)) / ((v / (1 - beta2 ** clock)).sqrt() + group["eps"])
                if 'agent_lr' in group:
                    p.add_(-update * mask * group['agent_lr'].view(shape))
                else:
                    p.add_(update * mask, alpha=-group["lr"])
