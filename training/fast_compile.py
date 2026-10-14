"""Independent fixed-shape fast actor inference and compiled PPO kernels."""
import torch


def configure_fast_execution(trainer, *, enabled=True, precision="tf32"):
    if precision not in ("fp32", "tf32", "bf16"):
        raise ValueError("precision must be fp32, tf32 or bf16")
    if trainer.device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.allow_tf32 = precision != "fp32"
        torch.set_float32_matmul_precision("high" if precision != "fp32" else "highest")
    if not enabled:
        return
    if trainer.device.type != "cuda":
        raise RuntimeError("fast compiled execution requires CUDA")
    actor = trainer.model.actor
    original = actor._forward
    # One shared parameter set, but independent inference and training graphs.
    # The inference graph has a fixed 8-env row shape and may use CUDA graphs.
    trainer.fast_actor_infer = torch.compile(
        original, backend="inductor", mode="reduce-overhead", dynamic=False)
    actor._forward = torch.compile(
        original, backend="inductor", mode="default", dynamic=False)
    trainer.model.critic.forward = torch.compile(
        trainer.model.critic.forward, backend="inductor", mode="default", dynamic=False)
    # AgentAdam has one vector-valued clock per independent UAV.
    # Its semantics cannot be replaced by ordinary fused Adam.
    for optimizer in (trainer.actor_opt, trainer.critic_opt):
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                state = optimizer.state[parameter]
                if not state:
                    shape = (parameter.shape[0],) + (1,) * (parameter.ndim - 1)
                    state.update(step=torch.zeros(shape, device=parameter.device),
                                 exp_avg=torch.zeros_like(parameter),
                                 exp_avg_sq=torch.zeros_like(parameter))
        optimizer.step = torch.compile(
            optimizer.step, backend="inductor", mode="default", dynamic=False)
