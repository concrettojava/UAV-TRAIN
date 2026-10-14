"""Fixed-length MAPPO fragments, batched independent policies, exact resume."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import time

import numpy as np
import torch

from envs import BatchedUAVEnv, EnvConfig, flatten_obs
from envs.parallel_env import ParallelUAVEnv
from models import build_model
from models.belief import ObservationHistory, belief_spec
from .buffer import RolloutBuffer, REWARD, LOGP, VALUE, ACTIVE, BOOTSTRAP, TRACE, OVERRIDE
from .optimizer import AgentAdam


@dataclass(frozen=True)
class MAPPOConfig:
    num_envs: int = 64
    rollout_steps: int = 128
    hidden: int = 256
    epochs: int = 15
    minibatch_size: int = 1024  # env-time rows; up to 8 agent samples per row
    lr: float = 4e-5
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip: float = 0.1
    value_clip: float = 0.1
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    seed: int = 0
    torch_threads: int = 1
    value_normalization: bool = True
    env_workers: int = 1
    model_name: str = 'stacked_mlp'
    collection_mode: str = 'fragment'
    belief_grid: int = 32
    cache_actor_raster: bool = False
    allow_episode_minibatches: bool = False  # Explicit research opt-in; old checkpoints stay full batch.


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("supported devices: auto, cpu, cuda, cuda:N")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA requested but unavailable (torch={torch.__version__}, build={torch.version.cuda})")
    return device


def sync(device):
    # GPU-to-CPU action transfer and metrics host reads already enforce dependencies.
    # Do not inject extra global CUDA synchronization into every rollout step.
    return


def clip_agent_gradients(module, maximum: float):
    # Each agent remains an independent optimization problem. A global norm
    # would let another agent's large gradient change this agent's update.
    params = [p for p in module.parameters() if p.grad is not None]
    norms = sum(p.grad.square().flatten(1).sum(1) for p in params).sqrt()
    scales = (maximum / (norms + 1e-6)).clamp(max=1)
    for p in params:
        p.grad.mul_(scales.view(-1, *([1] * (p.ndim - 1))))


class MAPPOTrainer:
    def __init__(self, cfg: MAPPOConfig, env_cfg: EnvConfig, device: str = "auto"):
        if min(cfg.num_envs, cfg.rollout_steps, cfg.hidden, cfg.epochs,
               cfg.minibatch_size, cfg.torch_threads, cfg.env_workers) < 1:
            raise ValueError("positive training dimensions required")
        if cfg.env_workers > cfg.num_envs:
            raise ValueError("env workers cannot exceed num_envs")
        if not (0 <= cfg.gamma <= 1 and 0 <= cfg.gae_lambda <= 1):
            raise ValueError("gamma and lambda must be in [0,1]")
        if cfg.collection_mode not in ('fragment','episode','episode_batch'):
            raise ValueError('unknown collection mode')
        if cfg.cache_actor_raster and (cfg.model_name != 'cognitive_cnn'
                                      or cfg.collection_mode != 'episode_batch'
                                      or cfg.minibatch_size < cfg.num_envs*cfg.rollout_steps):
            raise ValueError('raster cache requires cognitive_cnn episode_batch')
        if cfg.collection_mode == 'episode' and (cfg.num_envs != 1 or cfg.env_workers != 1
                or cfg.rollout_steps < env_cfg.max_steps or env_cfg.time_limit_mode != 'terminal'):
            raise ValueError('episode mode requires one serial environment, terminal deadline, and capacity >= max_steps')
        if cfg.collection_mode == 'episode_batch' and (cfg.rollout_steps < env_cfg.max_steps
                or env_cfg.time_limit_mode != 'terminal'
                or (not cfg.allow_episode_minibatches and cfg.minibatch_size < cfg.num_envs*cfg.rollout_steps)):
            raise ValueError('episode_batch requires terminal deadline, complete capacity and full batch')
        self.cfg, self.env_cfg = cfg, env_cfg
        self.device = resolve_device(device)
        torch.set_num_threads(cfg.torch_threads)
        torch.manual_seed(cfg.seed)
        self.env = (ParallelUAVEnv(env_cfg, cfg.num_envs, cfg.seed, cfg.env_workers)
                    if cfg.env_workers > 1 else BatchedUAVEnv(env_cfg, cfg.num_envs, cfg.seed))
        self.obs, self.state = self.env.reset(np.zeros(cfg.num_envs, bool))
        self.history = (ObservationHistory(belief_spec(env_cfg,self.env.obs_dim_flat,cfg.belief_grid),cfg.num_envs)
                        if cfg.model_name in ('mbdt','mbdt_perf128','mbdt_lite64','mbdt_dualscale','mbdt_perf192','mbdt_perf256x3','mbdt_perf256x4','history_vector','cognitive_cnn','bsed','bsed_fixed_encoder','bsed_global_decoder','bsed_capacity_cnn','bsed_cv_encoder','bsed_stationary_encoder','bsed_no_target_vector','bsed_cv_residual_encoder') else None)
        self.actor_obs = self.history.encode(self.obs) if self.history else flatten_obs(self.obs)
        self.actor_obs_dim = self.actor_obs.shape[-1]
        architecture = ({'belief_spec':asdict(self.history.spec)} if self.history else {})
        try:
            self.model = build_model(cfg.model_name, self.env.N, self.actor_obs_dim,
                                    self.env.state_dim, cfg.hidden,
                                    value_normalization=cfg.value_normalization,**architecture).to(self.device)
        except BaseException:
            self.close()  # A failed custom architecture must not orphan CPU environment workers.
            raise
        self.actor_opt = AgentAdam(self.model.actor.parameters(), lr=cfg.lr)
        self.critic_opt = AgentAdam(self.model.critic.parameters(), lr=cfg.lr)
        self.buffer = RolloutBuffer(cfg.rollout_steps, cfg.num_envs, self.env.N,
                                    self.actor_obs_dim, self.env.state_dim,
                                    self.device.type == "cuda")
        self.updates = self.env_steps = 0
        self.episode_returns = np.zeros((cfg.num_envs, self.env.N), np.float64)
        self.episodes = 0
        self.pending_episode_reset = False
        # Small persistent input staging tensors; avoid fresh pinned allocation each step.
        pin = self.device.type == "cuda"
        self.obs_input = torch.empty((cfg.num_envs, self.env.N, self.actor_obs_dim), pin_memory=pin)
        self.state_input = torch.empty((cfg.num_envs, self.env.state_dim), pin_memory=pin)
        self.obs_device = torch.empty(self.obs_input.shape,device=self.device,dtype=self.obs_input.dtype)
        self.state_device = torch.empty(self.state_input.shape,device=self.device,dtype=self.state_input.dtype)
        self.fast_actor_infer = None

    @torch.no_grad()
    def collect(self):
        cfg, b = self.cfg, self.buffer
        if self.pending_episode_reset:
            raise RuntimeError('finish the episode update/reset before collecting again')
        b.valid_steps = cfg.rollout_steps
        timing = dict(env_s=0.0, inference_s=0.0, transfer_s=0.0, organize_s=0.0)
        completed = []
        finished = np.zeros(cfg.num_envs,bool)
        collected_env_steps = 0
        whole_tasks = cfg.collection_mode in ('episode','episode_batch')
        started = time.perf_counter()
        for t in range(cfg.rollout_steps):
            stamp = time.perf_counter()
            flat = self.actor_obs if self.history else flatten_obs(self.obs)
            b.obs[t] = flat
            b.state[t] = self.state
            b.data[t, ..., ACTIVE] = self.obs["active"] & (~finished[:,None] if whole_tasks else True)
            self.obs_input.numpy()[:] = flat
            self.state_input.numpy()[:] = self.state
            timing["organize_s"] += time.perf_counter() - stamp
            stamp = time.perf_counter()
            x = self.obs_device.copy_(self.obs_input,non_blocking=True)
            s = self.state_device.copy_(self.state_input,non_blocking=True)
            sync(self.device)
            timing["transfer_s"] += time.perf_counter() - stamp
            stamp = time.perf_counter()
            if self.fast_actor_infer is None:
                a, lp, value = self.model.act(x, s)
            else:
                from torch.distributions import Categorical
                logits = self.fast_actor_infer(x.transpose(0, 1)).transpose(0, 1)
                dist = Categorical(logits=logits.float())
                a = dist.sample()
                lp = dist.log_prob(a)
                value = self.model.values(s)
            sync(self.device)
            timing["inference_s"] += time.perf_counter() - stamp
            stamp = time.perf_counter()
            # One packed device->host copy for actions/logprob/value.
            result = torch.stack((a.float(), lp, value), -1).cpu().numpy()
            b.actions[t] = result[..., 0]
            # Dead agents are excluded from both losses. Mask on the host to
            # avoid another tiny host->device transfer and CUDA kernel.
            b.actions[t][~self.obs["active"]] = 3
            b.data[t, ..., LOGP] = result[..., 1]
            b.data[t, ..., VALUE] = result[..., 2]
            timing["transfer_s"] += time.perf_counter() - stamp
            stamp = time.perf_counter()
            if whole_tasks:
                collected_env_steps += int((~finished).sum())
                self.obs, self.state, reward, done, info = self.env.step(
                    b.actions[t],autoreset=False,step_mask=~finished)
            else:
                self.obs, self.state, reward, done, info = self.env.step(b.actions[t])
            timing["env_s"] += time.perf_counter() - stamp
            stamp = time.perf_counter()
            if self.history:
                self.actor_obs = self.history.encode(self.obs,advance=True,
                    env_mask=~finished if whole_tasks else None,reset_mask=None if whole_tasks else done)
            b.data[t, ..., REWARD] = reward
            b.data[t, ..., BOOTSTRAP] = ~info["agent_terminated"]
            b.data[t, ..., TRACE] = ~info["agent_done"]
            b.data[t, ..., OVERRIDE] = 0
            self.episode_returns += reward
            for e in info["final_indices"]:
                completed.append({"return": float(self.episode_returns[e].mean()),
                                  "completion": float(info["completion_ratio"][e]),
                                  "survival": float(info["survival_ratio"][e]),
                                  "steps": int(info["steps"][e])})
                self.episode_returns[e] = 0
            timing["organize_s"] += time.perf_counter() - stamp
            if info["truncated"].any():
                stamp = time.perf_counter()
                idx = info["final_indices"]
                final_value = self.model.values(torch.as_tensor(info["final_state"], device=self.device))
                final_value = final_value.cpu().numpy()
                b.data[t, idx, :, OVERRIDE] = final_value
                timing["truncation_bootstrap_s"] = timing.get("truncation_bootstrap_s", 0) + time.perf_counter() - stamp
            finished |= done
            if whole_tasks and finished.all():
                b.valid_steps = t + 1
                self.pending_episode_reset = True
                break
        if whole_tasks and not self.pending_episode_reset:
            raise RuntimeError('episode collection capacity exhausted before the task ended')
        stamp = time.perf_counter()
        last_values = self.model.values(torch.as_tensor(self.state, device=self.device))
        sync(self.device)
        timing["fragment_bootstrap_s"] = time.perf_counter() - stamp
        timing["rollout_s"] = time.perf_counter() - started
        self.last_collected_env_steps = collected_env_steps if whole_tasks else b.valid_steps*cfg.num_envs
        self.env_steps += self.last_collected_env_steps
        self.episodes += len(completed)
        return last_values, timing, completed

    def update(self, last_values):
        cfg = self.cfg
        stamp = time.perf_counter()
        batch = self.buffer.batch(self.device, last_values, cfg.gamma, cfg.gae_lambda)
        norm = self.model.value_norm
        if norm is not None:
            norm.update(batch["returns"], batch["active"], self.model.critic.layers[-1])
            batch["values"] = norm.normalize(batch["values"])
            batch["returns"] = norm.normalize(batch["returns"])
        sync(self.device)
        prep_s = time.perf_counter() - stamp
        started = time.perf_counter()
        if cfg.cache_actor_raster:
            with self.model.actor.cache_raster(batch['obs']):
                return self._optimize_batch(batch, prep_s, started)
        return self._optimize_batch(batch, prep_s, started)

    def _optimize_batch(self, batch, prep_s, started):
        if getattr(self,'ppo_microbatch_rows',0):
            from .ppo_microbatch import optimize
            return optimize(self,batch,prep_s,started)
        cfg = self.cfg
        rows = batch["obs"].shape[0]
        stats = torch.zeros(6, device=self.device)
        count = 0
        lr_controller = getattr(self, 'actor_lr_controller', None)
        old_policy_logits = lr_controller.before(self, batch) if lr_controller else None
        for _ in range(cfg.epochs):
            # A single complete PPO batch gains nothing from permutation/gather.
            order = (None if rows <= cfg.minibatch_size or cfg.collection_mode == 'episode_batch'
                     else torch.randperm(rows, device=self.device))
            for start in range(0, rows, cfg.minibatch_size):
                if order is None:
                    # Keep original full-batch identity/cache contract, but do
                    # not repeatedly optimize the full batch when MB<rows.
                    mb = batch if rows <= cfg.minibatch_size else {k: v[start:start + cfg.minibatch_size] for k,v in batch.items()}
                else:
                    mb = {k: v[order[start:start + cfg.minibatch_size]] for k, v in batch.items()}
                mask = mb["active"]
                denom = mask.sum(0).clamp_min(1)

                def mean_per_agent(x):
                    return (x * mask).sum(0) / denom

                dist = self.model.distribution(mb["obs"])
                logp = dist.log_prob(mb["actions"])
                ratio = (logp - mb["logp"]).exp()
                unclipped = ratio * mb["advantages"]
                clipped = ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * mb["advantages"]
                actor_loss = -mean_per_agent(torch.minimum(unclipped, clipped))
                entropy = mean_per_agent(dist.entropy())
                self.actor_opt.zero_grad(set_to_none=True)
                (actor_loss - cfg.entropy_coef * entropy).sum().backward()
                clip_agent_gradients(self.model.actor, cfg.max_grad_norm)
                self.actor_opt.step(mask.sum(0) > 0)
                values = self.model.normalized_values(mb["state"])
                clipped_values = mb["values"] + (values - mb["values"]).clamp(-cfg.value_clip, cfg.value_clip)
                value_loss = 0.5 * mean_per_agent(torch.maximum(
                    (values - mb["returns"]).square(), (clipped_values - mb["returns"]).square()))
                self.critic_opt.zero_grad(set_to_none=True)
                value_loss.sum().backward()
                clip_agent_gradients(self.model.critic, cfg.max_grad_norm)
                self.critic_opt.step(mask.sum(0) > 0)
                with torch.no_grad():
                    kl = mean_per_agent((ratio - 1) - (logp - mb["logp"]))
                    cf = mean_per_agent(((ratio - 1).abs() > cfg.clip).float())
                    stats += torch.stack([actor_loss.mean(), value_loss.mean(), entropy.mean(),
                                          kl.mean(), cf.mean(), mask.mean()])
                count += 1
        post = {}
        if lr_controller:
            post.update(lr_controller.after(self, batch, old_policy_logits))
        if cfg.collection_mode == 'episode_batch':
            # With one full-batch step the usual pre-step KL is nearly zero.
            # Measure the actual policy movement after optimization as well.
            with torch.no_grad():
                log_ratio = (self.model.distribution(batch['obs']).log_prob(batch['actions'])-batch['logp']
                             if rows <= cfg.minibatch_size else torch.cat([
                                 self.model.distribution(batch['obs'][s:s+cfg.minibatch_size]).log_prob(batch['actions'][s:s+cfg.minibatch_size])-batch['logp'][s:s+cfg.minibatch_size]
                                 for s in range(0,rows,cfg.minibatch_size)]))
                mask=batch['active']; denom=mask.sum(0).clamp_min(1)
                post_kl=(((log_ratio.exp()-1-log_ratio)*mask).sum(0)/denom).mean()
                post_clip=((((log_ratio.exp()-1).abs()>cfg.clip).float()*mask).sum(0)/denom).mean()
                post_values=torch.stack((post_kl,post_clip)).cpu().tolist()
                if not np.isfinite(post_values).all():raise FloatingPointError('Nonfinite post-update policy diagnostics')
                post=dict(post_update_kl=post_values[0],post_update_clip_fraction=post_values[1])
        sync(self.device)
        update_s = time.perf_counter() - started
        numbers = (stats / count).cpu().tolist()
        if not np.isfinite(numbers).all():
            raise FloatingPointError("Nonfinite PPO diagnostics")
        self.updates += 1
        return dict(zip(("actor_loss", "value_loss", "entropy", "approx_kl", "clip_fraction", "active_fraction"), numbers),**post), {
            "batch_and_gae_s": prep_s, "update_s": update_s}

    def train_iteration(self, pre_update_hook=None):
        started = time.perf_counter()
        last, timing, episodes = self.collect()
        self.last_completed_episodes = episodes
        if pre_update_hook is not None:
            pre_update_hook(self)
        losses, update_time = self.update(last)
        if self.pending_episode_reset:
            self.obs,self.state = self.env.reset()
            if self.history:
                self.history.reset(np.ones(self.cfg.num_envs,bool))
                self.actor_obs = self.history.encode(self.obs)
            else:self.actor_obs = flatten_obs(self.obs)
            self.pending_episode_reset = False
        timing.update(update_time)
        timing["iteration_s"] = time.perf_counter() - started
        metrics = dict(update=self.updates, env_steps=self.env_steps, episodes=self.episodes,
                       completed_episodes=len(episodes), **losses, **timing)
        metrics['collected_steps'] = self.buffer.valid_steps
        metrics['collected_env_steps'] = self.last_collected_env_steps
        metrics["env_steps_per_s"] = self.last_collected_env_steps / timing["iteration_s"]
        metrics["rollout_env_steps_per_s"] = self.last_collected_env_steps / timing["rollout_s"]
        if episodes:
            for key in ("return", "completion", "survival", "steps"):
                metrics["episode_" + key] = float(np.mean([ep[key] for ep in episodes]))
            metrics['episode_steps_min'] = min(ep['steps'] for ep in episodes)
            metrics['episode_steps_max'] = max(ep['steps'] for ep in episodes)
        if self.cfg.collection_mode in ('episode', 'episode_batch'):
            metrics['batch_padding_env_steps'] = (self.buffer.valid_steps * self.cfg.num_envs
                                                  - self.last_collected_env_steps)
        return metrics

    def save(self, path, selection_evaluation=None):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = ("uxy", "uyaw", "ualive", "last_u", "mem", "txy", "tyaw", "tomega", "talive",
                  "thxy", "thr", "jxy", "jrad", "jstr", "jper", "jph", "step_count")
        payload = {
            "source_commit": getattr(self, 'source_commit', None),
            'selection_evaluation':selection_evaluation,
            "schema": "uavmarl.mappo.v1", "config": asdict(self.cfg), "env_config": asdict(self.env_cfg),
            "model": self.model.state_dict(), "actor_optimizer": self.actor_opt.state_dict(),
            "critic_optimizer": self.critic_opt.state_dict(), "updates": self.updates,
            "env_steps": self.env_steps, "episodes": self.episodes,
            'pending_episode_reset':self.pending_episode_reset,
            "env_arrays": {k: torch.from_numpy(getattr(self.env, k).copy()) for k in arrays},
            "obs": {k: torch.from_numpy(v.copy()) for k, v in self.obs.items()},
            "state": torch.from_numpy(self.state.copy()),
            "episode_returns": torch.from_numpy(self.episode_returns.copy()),
            "env_rng": self.env.rng.bit_generator.state, "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
        }
        if self.history:
            payload['belief_history'] = self.history.state_dict()
            payload['actor_obs'] = torch.from_numpy(self.actor_obs.copy())
        if hasattr(self, 'actor_lr_controller'):
            payload['actor_lr_control'] = self.actor_lr_controller.state_dict()
        if hasattr(self, 'mbdt_execution'):
            payload['mbdt_execution'] = dict(self.mbdt_execution)
        temp = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temp)
        temp.replace(path)

    @classmethod
    def load(cls, path, device="auto", env_workers=None):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["schema"] != "uavmarl.mappo.v1":
            raise ValueError("unknown checkpoint schema")
        config = dict(payload["config"])
        # Checkpoints predating value normalization retain their original algorithm.
        config.setdefault("value_normalization", False)
        if env_workers is not None:
            config["env_workers"] = env_workers
        env_config = dict(payload["env_config"])
        env_config.setdefault("state_layout", "legacy_v1")
        trainer = cls(MAPPOConfig(**config), EnvConfig(**env_config), device)
        trainer.source_commit = payload.get('source_commit')
        trainer.model.load_state_dict(payload["model"])
        trainer.actor_opt.load_state_dict(payload["actor_optimizer"])
        trainer.critic_opt.load_state_dict(payload["critic_optimizer"])
        if payload.get('mbdt_execution') is not None:
            from .mbdt_execution import configure_execution
            configure_execution(trainer, payload['mbdt_execution'])
        if payload.get('actor_lr_control'):
            from .actor_lr import ActorLRController
            saved = payload['actor_lr_control']
            ActorLRController(trainer, saved['policy'], saved)
        for key, value in payload["env_arrays"].items():
            setattr(trainer.env, key, value.numpy().copy())
        trainer.env.rng.bit_generator.state = payload["env_rng"]
        trainer.obs = {k: v.numpy().copy() for k, v in payload["obs"].items()}
        trainer.state = payload["state"].numpy().copy()
        trainer.env._observe()  # rebuild geometry/detection for snapshots
        trainer.episode_returns = payload["episode_returns"].numpy().copy()
        trainer.updates, trainer.env_steps, trainer.episodes = payload["updates"], payload["env_steps"], payload["episodes"]
        trainer.pending_episode_reset = payload.get('pending_episode_reset',False)
        if trainer.history:
            if 'belief_history' not in payload:raise ValueError('history checkpoint is missing actor memory')
            trainer.history.load_state_dict(payload['belief_history'])
            trainer.actor_obs = payload['actor_obs'].numpy().copy()
        else:trainer.actor_obs = flatten_obs(trainer.obs)
        torch.set_rng_state(payload["torch_rng"])
        if trainer.device.type == "cuda" and payload["cuda_rng"]:
            torch.cuda.set_rng_state_all(payload["cuda_rng"])
        return trainer

    def close(self):
        if isinstance(self.env, ParallelUAVEnv):
            self.env.close()
