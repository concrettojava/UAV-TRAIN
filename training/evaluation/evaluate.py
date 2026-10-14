"""Evaluate one checkpoint using exact, independently reset fixed episode seeds."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from training.algo.mappo import resolve_device
from envs import BatchedUAVEnv, EnvConfig, flatten_obs
from models import build_model
from models.belief import ObservationHistory,belief_spec



def load_policy(path, device="auto"):
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["schema"] != "uavmarl.mappo.v1":
        raise ValueError("unknown checkpoint schema")
    env_config = dict(payload["env_config"])
    env_config.setdefault("state_layout", "legacy_v1")
    cfg = EnvConfig(**env_config)
    env = BatchedUAVEnv(cfg, 1)
    selected = resolve_device(device)
    name=payload['config'].get('model_name','stacked_mlp')
    spec=belief_spec(cfg,env.obs_dim_flat,payload['config'].get('belief_grid',32)) if name in ('mbdt','mbdt_dualscale','mbdt_lite64','mbdt_perf128','mbdt_perf192','mbdt_perf256x3','mbdt_perf256x4','history_vector','cognitive_cnn','bsed','bsed_fixed_encoder','bsed_global_decoder','bsed_capacity_cnn','bsed_cv_encoder','bsed_stationary_encoder','bsed_no_target_vector','bsed_cv_residual_encoder') else None
    model = build_model(name,env.N,spec.dim if spec else env.obs_dim_flat,env.state_dim,payload['config']['hidden'],
        value_normalization=payload['config'].get('value_normalization',False),
        **({'belief_spec':asdict(spec)} if spec else {})).to(selected)
    model.belief_spec=spec
    model.load_state_dict(payload["model"])
    model.eval()
    return model, cfg, selected


def summarize_records(records):
    if not records:
        raise ValueError('cannot summarize an empty evaluation')
    summary = {}
    for key in ('return', 'completion', 'survival', 'time_s'):
        values = np.array([row[key] for row in records])
        summary[key + '_mean'] = float(values.mean())
        summary[key + '_std'] = float(values.std(ddof=1)) if len(records) > 1 else 0.
    for key in ('success', 'all_dead', 'timeout'):
        summary[key + '_rate'] = float(np.mean([row[key] for row in records]))
    for key in ('collision_losses', 'threat_losses', 'out_of_bounds_agent_steps'):
        summary[key + '_mean'] = float(np.mean([row[key] for row in records]))
    times = [row['time_s'] for row in records if row['success']]
    summary['success_time_s_mean'] = float(np.mean(times)) if times else None
    steps = [row['steps'] for row in records if row['success']]
    summary['success_episode_count'] = len(steps)
    for key, statistic in (('mean', np.mean), ('median', np.median), ('p90', lambda x: np.percentile(x, 90))):
        summary['success_steps_'+key] = float(statistic(steps)) if steps else None
    summary['termination_steps_mean'] = float(np.mean([row['steps'] for row in records]))
    summary['termination_steps_median'] = float(np.median([row['steps'] for row in records]))
    return summary


@torch.no_grad()
def evaluate(path=None, episodes=32, seed_start=100000, parallel=32, device="auto",
             policy=None, env_config=None, deterministic=True, observation_transform=None):
    if episodes < 1 or parallel < 1:
        raise ValueError("episodes and parallel must be positive")
    torch.set_num_threads(1)
    if policy is None:
        model, cfg, selected = load_policy(path, device)
    else:
        cfg = env_config or EnvConfig(reward_profile="task_aligned")
    records = []
    # Each row starts with a standalone seeded scene. Batch RNG supplies no
    # episode coupling: externally injected noise has a separate RNG per row.
    for offset in range(0, episodes, parallel):
        count = min(parallel, episodes - offset)
        if policy is not None and hasattr(policy,'begin_batch'):
            policy.begin_batch(count,cfg.n_uavs,cfg)
        env = BatchedUAVEnv(cfg, count)
        noises = []
        action_rngs = []
        for row in range(count):
            seed = seed_start + offset + row
            single = BatchedUAVEnv(cfg, 1, seed)
            for key in ("uxy", "uyaw", "ualive", "last_u", "mem", "txy", "tyaw", "tomega", "talive",
                        "thxy", "thr", "jxy", "jrad", "jstr", "jper", "jph", "step_count"):
                getattr(env, key)[row] = getattr(single, key)[0]
            noises.append(single.rng)
            action_rngs.append(np.random.default_rng(seed + 400000))
        obs, state = env.reset(np.zeros(count, bool))
        episode_seeds=list(range(seed_start+offset,seed_start+offset+count))
        if observation_transform is not None:
            obs=observation_transform(obs,episode_seeds,0)
        history=ObservationHistory(model.belief_spec,count) if policy is None and model.belief_spec else None
        actor_obs=history.encode(obs) if history else flatten_obs(obs)
        finished = np.zeros(count, bool)
        returns = np.zeros((count, env.N), np.float64)
        causes = np.zeros((count, 3), np.int64)
        for step_index in range(cfg.max_steps):
            if policy is None and deterministic:
                acts, _, _ = model.act(torch.as_tensor(actor_obs, device=selected),
                                       torch.as_tensor(state, device=selected), deterministic=True)
                actions = acts.cpu().numpy()
            elif policy is None:
                probabilities = model.distribution(torch.as_tensor(actor_obs, device=selected)).probs.cpu().numpy()
                uniform = np.stack([rng.random(env.N) for rng in action_rngs])
                actions = (uniform[..., None] > probabilities.cumsum(-1)).sum(-1).clip(0, 6)
            else:
                actions = policy(obs)
            alpha = np.stack([r.normal(0, env._target_turn_std, env.T) for r in noises])
            shots = np.stack([r.random((env.N, env.K)) for r in noises])
            obs, state, reward, done, info = env.step(actions, _noise=(alpha, shots))
            if observation_transform is not None:
                obs=observation_transform(obs,episode_seeds,step_index+1)
            actor_obs=(history.encode(obs,advance=True,env_mask=~finished,reset_mask=done)
                       if history else flatten_obs(obs))
            returns[~finished] += reward[~finished]
            for column, key in enumerate(("newly_collision", "newly_threat", "out_of_bounds")):
                causes[~finished, column] += info[key][~finished].sum(-1)
            for row in np.flatnonzero(done & ~finished):
                records.append({"seed": seed_start + offset + int(row),
                                "return": float(returns[row].mean()),
                                "completion": float(info["completion_ratio"][row]),
                                "survival": float(info["survival_ratio"][row]),
                                "steps": int(info["steps"][row]),
                                "time_s": int(info["steps"][row]) * cfg.dt,
                                "success": bool(info["completion_ratio"][row] == 1),
                                "all_dead": bool(info["survival_ratio"][row] == 0),
                                "timeout": bool(info["steps"][row] >= cfg.max_steps
                                                and info["completion_ratio"][row] < 1
                                                and info["survival_ratio"][row] > 0),
                                "collision_losses": int(causes[row, 0]),
                                "threat_losses": int(causes[row, 1]),
                                "out_of_bounds_agent_steps": int(causes[row, 2])})
            finished |= done
            if finished.all():
                break
    records.sort(key=lambda x: x["seed"])
    if len(records) != episodes:
        raise RuntimeError(f'incomplete evaluation: requested {episodes}, obtained {len(records)}')
    summary = summarize_records(records)
    return {"checkpoint": str(path), "episodes": records, "summary": summary,
            "env_config": asdict(cfg), "deterministic": deterministic if policy is None else None}
