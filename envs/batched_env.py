"""Batched cooperative reconnaissance-strike environment (numpy, struct-of-arrays).

Physical mission rules are based on the legacy scalar environment, with all E
scenes advanced by array operations. Observations use entity sets/masks without
IDs; reward and time-limit changes are explicitly versioned in EnvConfig.

Step order (identical to the legacy env):
  1. move UAVs, move targets (+ contested evasive turn)
  2. pre-destruction detection / comm components  -> reward context, strikes
  3. automatic strikes, collisions, probabilistic threat damage
  4. rewards (evaluated after destruction, using the pre-destruction context)
  5. done flags, auto-reset of finished envs
  6. post-destruction detection -> observation / state

Information rules for the observation (what the actor may know):
  * own pose, own type, own comm_quality / recon_quality (jammer is hidden)
  * teammates in the same communication component (ideal sharing)
  * targets currently detected by anyone in the component (no memory)
  * threats currently detected or remembered by anyone in the component
"""
from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np

from .config import ACTION_VALUES, EnvConfig, UAVConsts

LAMBDA_TIME = 1.0 / 100.0
LAMBDA_DIST = 1.0 / 2000.0
LAMBDA_ACTION = 0.25
TASK_COMPLETION_BONUS = 100.0
TASK_EARLY_DEATH_COST_PER_STEP = 4.0
_E1 = math.exp(-1.0)

# Feature sizes of the observation (see _build_obs).
SELF_DIM = 9      # type(3) x y sin cos comm_q recon_q
MATE_DIM = 8      # type(3) fwd right range sin_rel cos_rel
TARGET_DIM = 5    # fwd right range sin_rel cos_rel
THREAT_DIM = 4    # fwd right range radius

ENV_ARRAYS = ('uxy', 'uyaw', 'ualive', 'last_u', 'mem', 'txy', 'tyaw', 'tomega', 'talive',
              'thxy', 'thr', 'jxy', 'jrad', 'jstr', 'jper', 'jph', 'step_count')


def _wrap(a):
    return (a + np.pi) % (2.0 * np.pi) - np.pi


def general_reward(a, b):
    """Paper Eq. (11), vectorised. ``a`` and ``b`` broadcast."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    safe_b = np.where(b > 0, b, 1.0)
    val = -(np.exp(-a / safe_b) - _E1) / (1.0 - _E1)
    val = np.maximum(val, -10.0)
    return np.where((b <= 0) | (a >= b), 0.0, val)


class _Geometry:
    """Pairwise geometry shared by detection, strikes, rewards and obs."""
    __slots__ = ("duu_x", "duu_y", "d2uu", "duu", "dut_x", "dut_y", "d2ut", "dut",
                 "dth_x", "dth_y", "d2th", "dth")


class _Detection:
    __slots__ = ("A", "R", "direct_t", "direct_th", "know_th", "shared_t", "shared_th")


class BatchedUAVEnv:
    def __init__(self, cfg: EnvConfig, num_envs: int, seed: int = 0):
        self.cfg = cfg
        if cfg.state_layout not in ("legacy_v1", "full_v1"):
            raise ValueError("unknown critic state layout")
        if cfg.time_limit_mode not in ("terminal", "truncate"):
            raise ValueError("time_limit_mode must be terminal or truncate")
        if num_envs < 1 or cfg.max_steps < 1 or cfg.n_targets < 1 or cfg.n_threats < 1:
            raise ValueError("positive env, step, target and threat counts required")
        if cfg.scenario not in ("reference", "contested") or cfg.reward_profile not in ("paper", "task_aligned", "task_v2"):
            raise ValueError("unknown scenario or reward profile")
        self.E = int(num_envs)
        self.N = cfg.n_uavs
        self.T = cfg.n_targets
        self.K = cfg.n_threats
        self.J = cfg.jammer_count if cfg.contested else 0
        self.c = UAVConsts()
        self.rng = np.random.default_rng(seed)
        self._eye = np.eye(self.N, dtype=bool)
        self._target_turn_std = math.radians(cfg.target_turn_accel_std_deg)
        self._evasive_turn = math.radians(cfg.evasive_turn_rate_deg)

        E, N, T, K, J = self.E, self.N, self.T, self.K, self.J
        f = np.float64
        self.uxy = np.zeros((E, N, 2), f)
        self.uyaw = np.zeros((E, N), f)
        self.ualive = np.ones((E, N), bool)
        self.last_u = np.zeros((E, N), f)
        self.mem = np.zeros((E, N, K), bool)            # per-UAV threat memory
        self.txy = np.zeros((E, T, 2), f)
        self.tyaw = np.zeros((E, T), f)
        self.tomega = np.zeros((E, T), f)
        self.talive = np.ones((E, T), bool)
        self.thxy = np.zeros((E, K, 2), f)
        self.thr = np.zeros((E, K), f)
        self.jxy = np.zeros((E, J, 2), f)
        self.jrad = np.ones((E, J), f)
        self.jstr = np.zeros((E, J), f)
        self.jper = np.ones((E, J), f)
        self.jph = np.zeros((E, J), f)
        self.step_count = np.zeros(E, np.int64)
        self.reset()

    # ------------------------------------------------------------------ reset
    def _reset_scene(self, mask: np.ndarray):
        cfg, rng = self.cfg, self.rng
        idx = np.flatnonzero(mask)
        n = idx.size
        if n == 0:
            return
        W = cfg.world_size
        margin = 250.0
        self.step_count[idx] = 0
        self.uxy[idx] = self.c.init_xy[None]
        self.uyaw[idx] = 0.0
        self.ualive[idx] = True
        self.last_u[idx] = 0.0
        self.mem[idx] = False
        tx = rng.uniform(margin, W - margin, (n, self.T))
        ty = rng.uniform(1200.0, W - margin, (n, self.T))
        self.txy[idx] = np.stack([tx, ty], -1)
        self.tyaw[idx] = rng.uniform(-np.pi, np.pi, (n, self.T))
        self.tomega[idx] = 0.0
        self.talive[idx] = True
        hx = rng.uniform(margin, W - margin, (n, self.K))
        hy = rng.uniform(700.0, W - margin, (n, self.K))
        self.thxy[idx] = np.stack([hx, hy], -1)
        self.thr[idx] = rng.uniform(120.0, 200.0, (n, self.K))
        if self.J:
            jx = rng.uniform(650.0, W - 650.0, (n, self.J))
            jy = rng.uniform(900.0, W - 500.0, (n, self.J))
            self.jxy[idx] = np.stack([jx, jy], -1)
            self.jrad[idx] = rng.uniform(cfg.jammer_min_radius, cfg.jammer_max_radius, (n, self.J))
            self.jstr[idx] = rng.uniform(cfg.jammer_min_strength, cfg.jammer_max_strength, (n, self.J))
            self.jper[idx] = rng.uniform(45.0, 85.0, (n, self.J))
            self.jph[idx] = rng.uniform(-np.pi, np.pi, (n, self.J))

    def reset(self, mask: Optional[np.ndarray] = None):
        """Reset the masked envs (all if None) and return ``obs, state``."""
        if mask is None:
            mask = np.ones(self.E, bool)
        self._reset_scene(mask)
        obs, state = self._observe()
        return obs, state

    # -------------------------------------------------------------- jammers
    def interference(self) -> np.ndarray:
        """Aggregate interference intensity at every UAV, shape [E,N]. Hidden truth."""
        if self.J == 0:
            return np.zeros((self.E, self.N))
        d2 = ((self.uxy[:, :, None, :] - self.jxy[:, None, :, :]) ** 2).sum(-1)    # [E,N,J]
        spatial = np.exp(-d2 / np.maximum(self.jrad ** 2, 1.0)[:, None, :])
        temporal = 0.65 + 0.35 * np.sin(
            2.0 * np.pi * self.step_count[:, None, None] / self.jper[:, None, :]
            + self.jph[:, None, :])
        local = np.clip(self.jstr[:, None, :] * spatial * temporal, 0.0, 0.95)
        return np.clip(1.0 - np.prod(1.0 - local, axis=-1), 0.0, 0.95)

    def _factors(self) -> Tuple[np.ndarray, np.ndarray]:
        if not self.cfg.contested:
            one = np.ones((self.E, self.N))
            return one, one
        inten = self.interference()
        comm = np.maximum(self.cfg.min_comm_factor, 1.0 - inten)
        recon = np.maximum(self.cfg.min_recon_factor, 1.0 - 0.60 * inten)
        return comm, recon

    # ------------------------------------------------------------- geometry
    def _geometry(self) -> _Geometry:
        g = _Geometry()
        g.duu_x = self.uxy[:, None, :, 0] - self.uxy[:, :, None, 0]     # [E,i,j] = x_j - x_i
        g.duu_y = self.uxy[:, None, :, 1] - self.uxy[:, :, None, 1]
        g.d2uu = g.duu_x ** 2 + g.duu_y ** 2
        g.duu = np.sqrt(g.d2uu)
        g.dut_x = self.txy[:, None, :, 0] - self.uxy[:, :, None, 0]     # [E,i,t]
        g.dut_y = self.txy[:, None, :, 1] - self.uxy[:, :, None, 1]
        g.d2ut = g.dut_x ** 2 + g.dut_y ** 2
        g.dut = np.sqrt(g.d2ut)
        g.dth_x = self.thxy[:, None, :, 0] - self.uxy[:, :, None, 0]    # [E,i,k]
        g.dth_y = self.thxy[:, None, :, 1] - self.uxy[:, :, None, 1]
        g.d2th = g.dth_x ** 2 + g.dth_y ** 2
        g.dth = np.sqrt(g.d2th)
        return g

    # ------------------------------------------------------------ detection
    def _detect(self, g: _Geometry, alive_u, alive_t, cf, rf, update_mem=True) -> _Detection:
        c = self.c
        d = _Detection()
        eff = c.comm_range[None, :] * cf                                  # [E,N]
        lim = np.maximum(eff[:, :, None], eff[:, None, :])
        both = alive_u[:, :, None] & alive_u[:, None, :] & ~self._eye[None]
        d.A = both & (g.d2uu < lim ** 2)                                  # direct links
        Rf = (d.A | (alive_u[:, :, None] & self._eye[None])).astype(np.float32)
        for _ in range(3):                                                # reach <= 8 hops
            Rf = ((Rf @ Rf) > 0.5).astype(np.float32)
        d.R = Rf > 0.5                                                    # component incl. self
        reff2 = (c.recon_range[None, :] * rf) ** 2                        # [E,N]
        d.direct_t = alive_u[:, :, None] & alive_t[:, None, :] & (g.d2ut < reff2[:, :, None])
        d.direct_th = alive_u[:, :, None] & (g.d2th < reff2[:, :, None])
        if update_mem:
            self.mem |= d.direct_th
        d.know_th = d.direct_th | self.mem
        d.shared_t = (Rf @ d.direct_t.astype(np.float32)) > 0.5           # [E,N,T]
        d.shared_th = (Rf @ d.know_th.astype(np.float32)) > 0.5           # [E,N,K]
        return d

    # ------------------------------------------------------------- dynamics
    def _move(self, actions: np.ndarray, alpha: np.ndarray):
        c, cfg = self.c, self.cfg
        dt = cfg.dt
        u = ACTION_VALUES[actions]                                         # [E,N]
        alive = self.ualive
        self.last_u = np.where(alive, u, self.last_u)
        yaw = _wrap(self.uyaw + c.max_turn[None] * u * dt)
        self.uyaw = np.where(alive, yaw, self.uyaw)
        dx = c.speed[None] * np.sin(self.uyaw) * dt
        dy = c.speed[None] * np.cos(self.uyaw) * dt
        self.uxy[..., 0] += np.where(alive, dx, 0.0)
        self.uxy[..., 1] += np.where(alive, dy, 0.0)

        # --- targets
        ta = self.talive
        omega = self.tomega + np.where(ta, alpha * dt, 0.0)
        yaw = self.tyaw
        if cfg.contested:
            d2 = ((self.txy[:, :, None, :] - self.uxy[:, None, :, :]) ** 2).sum(-1)   # [E,T,N]
            d2 = np.where(self.ualive[:, None, :], d2, np.inf)
            near = np.argmin(d2, axis=-1)                                  # [E,T]
            dmin = np.take_along_axis(d2, near[..., None], -1)[..., 0]
            nxy = np.take_along_axis(self.uxy, near[..., None], 1)         # [E,T,2]
            away = np.arctan2(self.txy[..., 0] - nxy[..., 0], self.txy[..., 1] - nxy[..., 1])
            err = _wrap(away - yaw)
            step = np.clip(err, -self._evasive_turn * dt, self._evasive_turn * dt)
            evade = ta & (np.sqrt(dmin) < cfg.evasive_trigger_range)       # inf -> False
            yaw = np.where(evade, _wrap(yaw + step), yaw)
            omega = np.where(evade, omega * 0.65, omega)
        yaw = _wrap(yaw + omega * dt)
        W, v = cfg.world_size, cfg.target_speed
        nx = self.txy[..., 0] + v * np.sin(yaw) * dt
        ny = self.txy[..., 1] + v * np.cos(yaw) * dt
        yaw = np.where((nx < 0) | (nx > W), _wrap(-yaw), yaw)
        yaw = np.where((ny < 0) | (ny > W), _wrap(np.pi - yaw), yaw)
        nx = np.clip(self.txy[..., 0] + v * np.sin(yaw) * dt, 0.0, W)
        ny = np.clip(self.txy[..., 1] + v * np.cos(yaw) * dt, 0.0, W)
        self.txy[..., 0] = np.where(ta, nx, self.txy[..., 0])
        self.txy[..., 1] = np.where(ta, ny, self.txy[..., 1])
        self.tyaw = np.where(ta, yaw, self.tyaw)
        self.tomega = np.where(ta, omega, self.tomega)

    # ----------------------------------------------------------------- step
    def step(self, actions: np.ndarray, _noise: Optional[Tuple[np.ndarray, np.ndarray]] = None,
             *, capture_frames: bool = False, autoreset: bool = True, step_mask=None):
        """Advance all envs one step.

        Returns ``obs, state, reward[E,N], done[E], info``. Finished envs are
        auto-reset; the returned obs/state already belong to the new episode.
        ``info`` distinguishes termination from truncation. ``final_indices``
        identifies the rows in compact ``final_obs``/``final_state`` before reset.
        The default time limit is a mission deadline (true termination).
        ``autoreset=False`` leaves final rows in place for a coordinator to reset.
        """
        cfg, c = self.cfg, self.c
        E, N = self.E, self.N
        actions = np.asarray(actions)
        if actions.shape != (E, N) or not np.issubdtype(actions.dtype, np.integer):
            raise ValueError(f"actions must be integers of shape {(E, N)}")
        if ((actions < 0) | (actions >= len(ACTION_VALUES))).any():
            raise ValueError("action out of range")
        if _noise is None:
            alpha = self.rng.normal(0.0, self._target_turn_std, (E, self.T))
            shot_u = self.rng.random((E, N, self.K))
        else:
            alpha, shot_u = _noise
        if step_mask is not None:
            mask = np.asarray(step_mask, dtype=bool)
            if mask.shape != (E,):
                raise ValueError('step_mask must have one entry per environment')
            if not mask.all():
                return self._masked_step(actions, (alpha, shot_u), mask, capture_frames, autoreset)
        prev_u = self.last_u.copy()
        self.step_count += 1
        self._move(actions, alpha)

        g = self._geometry()
        cf, rf = self._factors()
        active_before = self.ualive.copy()
        det = self._detect(g, active_before, self.talive, cf, rf)

        # ---- strikes (pre-destruction visibility, strike cone)
        bearing = np.arctan2(g.dut_x, g.dut_y)                              # target bearing, North=0
        berr = np.abs(_wrap(bearing - self.uyaw[:, :, None]))
        strikeable = (det.shared_t & self.talive[:, None, :]
                      & (g.dut < c.strike_range[None, :, None])
                      & (berr < c.strike_half_angle[None, :, None]))        # [E,N,T]
        killed = strikeable.any(axis=1)
        self.talive = self.talive & ~killed

        # ---- collisions
        close = (active_before[:, :, None] & active_before[:, None, :] & ~self._eye[None]
                 & (g.d2uu < np.minimum(c.collision_range[:, None], c.collision_range[None, :]) ** 2))
        dead_c = close.any(axis=-1)
        alive_mid = active_before & ~dead_c

        # ---- probabilistic threat damage
        inside = alive_mid[:, :, None] & (g.dth < self.thr[:, None, :])
        frac = np.clip(1.0 - g.dth / np.maximum(self.thr[:, None, :], 1e-9), 0.0, 1.0)
        base = 1.0 - frac ** cfg.threat_eta
        p_shot = 1.0 - base ** cfg.dt
        dead_th = (inside & (shot_u < p_shot)).any(axis=-1)
        newly = (dead_c | dead_th) & active_before
        self.ualive = active_before & ~newly

        reward = self._rewards(g, det, active_before, strikeable, newly, prev_u)

        n_alive_t = self.talive.sum(-1)
        n_alive_u = self.ualive.sum(-1)
        mission_end = (n_alive_t == 0) | (n_alive_u == 0)
        timeout = self.step_count >= cfg.max_steps
        terminated = mission_end | (timeout & (cfg.time_limit_mode == "terminal"))
        truncated = timeout & ~terminated
        done = terminated | truncated
        info = {
            "completion_ratio": 1.0 - n_alive_t / self.T,
            "survival_ratio": n_alive_u / N,
            "steps": self.step_count.copy(),
            "agent_done": done[:, None] | ~self.ualive,
            "terminated": terminated,
            "truncated": truncated,
            "agent_terminated": terminated[:, None] | ~self.ualive,
            "agent_truncated": truncated[:, None] & self.ualive,
            "active_before": active_before,
            "killed": killed,
            "newly_collision": dead_c & active_before,
            "newly_threat": dead_th,
            "out_of_bounds": active_before & ((self.uxy < 0) | (self.uxy > cfg.world_size)).any(-1),
        }
        # Observe before resetting: bootstrap must never read next episode state.
        obs, state = self._observe()
        indices = np.flatnonzero(done)
        info["final_indices"] = indices
        info["final_obs"] = {k: v[indices].copy() for k, v in obs.items()}
        info["final_state"] = state[indices].copy()
        if capture_frames:
            info["final_frames"] = {int(e): self.snapshot(int(e)) for e in indices}
        if autoreset and done.any():
            self._reset_scene(done)
            obs, state = self._observe()
        return obs, state, reward, done, info

    def _masked_step(self, actions, noise, mask, capture_frames, autoreset):
        """Advance only live rows; finished tasks retain their exact terminal state.

        Full-shape noise is drawn by the coordinator before selecting rows, so
        process partitioning does not change random ordering.
        """
        indices = np.flatnonzero(mask)
        reward = np.zeros((self.E, self.N))
        done = np.zeros(self.E, bool)
        obs, state = self._observe()
        info = dict(completion_ratio=1-self.talive.sum(-1)/self.T,
                    survival_ratio=self.ualive.sum(-1)/self.N, steps=self.step_count.copy(),
                    agent_done=np.ones((self.E,self.N),bool), terminated=np.zeros(self.E,bool),
                    truncated=np.zeros(self.E,bool), agent_terminated=np.ones((self.E,self.N),bool),
                    agent_truncated=np.zeros((self.E,self.N),bool),
                    active_before=np.zeros((self.E,self.N),bool), killed=np.zeros((self.E,self.T),bool),
                    newly_collision=np.zeros((self.E,self.N),bool),
                    newly_threat=np.zeros((self.E,self.N),bool), out_of_bounds=np.zeros((self.E,self.N),bool),
                    final_indices=np.empty(0,dtype=int), final_state=state[:0].copy(),
                    final_obs={k:v[:0].copy() for k,v in obs.items()})
        if indices.size:
            # All physical arrays are copied; constants/config may be shared.
            sub = object.__new__(BatchedUAVEnv)
            sub.__dict__ = self.__dict__.copy()
            sub.E = len(indices)
            for key in ENV_ARRAYS:
                setattr(sub,key,getattr(self,key)[indices].copy())
            result = sub.step(actions[indices],_noise=tuple(v[indices] for v in noise),
                              autoreset=False)
            for key in ENV_ARRAYS:
                getattr(self,key)[indices] = getattr(sub,key)
            reward[indices],done[indices] = result[2:4]
            for key,value in result[4].items():
                if key in ('final_state','final_obs'):
                    info[key] = value
                elif key == 'final_indices':
                    info[key] = indices[value]
                else:
                    info[key][indices] = value
        if capture_frames:
            self._observe()
            info['final_frames'] = {int(e):self.snapshot(int(e)) for e in info['final_indices']}
        if autoreset and done.any():
            self._reset_scene(done)
        obs,state = self._observe()
        return obs,state,reward,done,info

    # -------------------------------------------------------------- rewards
    def _rewards(self, g, det, active_before, strikeable, newly, prev_u) -> np.ndarray:
        cfg, c = self.cfg, self.c
        E, N = self.E, self.N
        ab = active_before
        n_alive_t = self.talive.sum(-1)                                     # post-strike [E]
        n_alive_u = self.ualive.sum(-1)
        # mission
        if cfg.reward_profile in ("task_aligned", "task_v2"):
            mission = (-LAMBDA_TIME * cfg.dt - general_reward(n_alive_t, self.T)
                       + general_reward(n_alive_u, N)
                       + np.where(n_alive_t == 0, TASK_COMPLETION_BONUS, 0.0))
        else:
            mission = (-LAMBDA_TIME * (self.step_count * cfg.dt)
                       - general_reward(n_alive_t, self.T) + general_reward(n_alive_u, N))
        # avoidance
        other = ab[:, None, :] & ~self._eye[None]                            # j active, j != i
        r_avoid = (5.0 * np.where(other, general_reward(g.duu, 160.0), 0.0).sum(-1)
                   + 10.0 * general_reward(g.dth, 360.0).sum(-1))
        if cfg.reward_profile == 'task_v2':
            r_avoid *= .1  # Actual death remains costly; broad proximity should not dominate mission.

        def excl(range_arr):
            thr = 0.8 * range_arr[None, :, None]
            m = other & c.same_type[None]
            return np.where(m, general_reward(g.duu, thr), 0.0).sum(-1)

        # strike
        d_alive_t = np.where(self.talive[:, None, :], g.dut, np.inf)
        nearest_t = d_alive_t.min(-1)
        r_stk = (np.where(np.isfinite(nearest_t), -LAMBDA_DIST * nearest_t, 0.0)
                 + 10.0 * strikeable.sum(-1) + excl(c.strike_range))
        # reconnaissance
        undisc = ~self.mem.any(axis=1)                                       # [E,K]
        d_unk = np.where(undisc[:, None, :], g.dth, np.inf)
        nearest_r = np.minimum(nearest_t, d_unk.min(-1))
        known = det.direct_t.sum(-1) + det.know_th.sum(-1)
        denom = n_alive_t[:, None] + self.K
        r_rec = (np.where(np.isfinite(nearest_r), -LAMBDA_DIST * nearest_r, 0.0)
                 + general_reward(known, denom) + excl(c.recon_range))
        # communication
        d_other = np.where(other, g.duu, np.inf)
        nearest_c = d_other.min(-1)
        n_others = other.sum(-1)
        r_com = (np.where(np.isfinite(nearest_c), -LAMBDA_DIST * nearest_c, 0.0)
                 + general_reward(det.A.sum(-1), n_others) + excl(c.comm_range))

        r_destroy = np.where(newly, -50.0, 0.0)
        if cfg.reward_profile in ("task_aligned", "task_v2"):
            remaining = np.maximum(0, cfg.max_steps - self.step_count)[:, None]
            r_destroy = r_destroy - np.where(newly, TASK_EARLY_DEATH_COST_PER_STEP * remaining, 0.0)
        ability = (r_avoid + r_destroy + c.ratio_stk[None] * r_stk
                   + c.ratio_rec[None] * r_rec + c.ratio_com[None] * r_com)
        du = self.last_u - prev_u
        r_action = -0.5 * np.abs(du) - self.last_u ** 2
        W = cfg.world_size
        x, y = self.uxy[..., 0], self.uxy[..., 1]
        r_bound = (general_reward(x, 160.0) + general_reward(W - x, 160.0)
                   + general_reward(y, 160.0) + general_reward(W - y, 160.0))
        total = mission[:, None] + ability + LAMBDA_ACTION * r_action + r_bound
        if getattr(self, 'record_reward_terms', False):
            # Audit only; these privileged diagnostics never enter actor observations.
            terms = {'mission':mission[:, None], 'avoidance':r_avoid, 'death':r_destroy,
                'strike':c.ratio_stk[None]*r_stk, 'recon':c.ratio_rec[None]*r_rec,
                'communication':c.ratio_com[None]*r_com,
                'action':LAMBDA_ACTION*r_action, 'boundary':r_bound}
            self.reward_terms = {key:np.where(ab, value, 0.) for key,value in terms.items()}
        return np.where(ab, total, 0.0)

    # ---------------------------------------------------------- observation
    def _observe(self) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
        g = self._geometry()
        cf, rf = self._factors()
        det = self._detect(g, self.ualive, self.talive, cf, rf)
        self._last_det = det
        return self._build_obs(g, det, cf, rf), self._build_state()

    def _ego(self, dx, dy):
        """Rotate world offsets into each observer's body frame (fwd, right)."""
        s = np.sin(self.uyaw)[:, :, None]
        co = np.cos(self.uyaw)[:, :, None]
        return dx * s + dy * co, dx * co - dy * s

    def _sorted(self, feat, valid, dist):
        """Order candidate rows by ascending range (invalid last) and zero the padding."""
        E, N, M, F = feat.shape
        order = np.argsort(np.where(valid, dist, np.inf), axis=-1, kind="stable")   # [E,N,M]
        flat = (np.arange(E * N, dtype=np.int64)[:, None] * M + order.reshape(E * N, M)).ravel()
        feat = feat.reshape(E * N * M, F).take(flat, axis=0).reshape(E, N, M, F)
        valid = valid.reshape(-1).take(flat).reshape(E, N, M)
        return feat * valid[..., None], valid

    def _build_obs(self, g, det, cf, rf) -> Dict[str, np.ndarray]:
        cfg, c = self.cfg, self.c
        L = cfg.length_scale
        E, N = self.E, self.N
        alive = self.ualive
        s_p, c_p = np.sin(self.uyaw), np.cos(self.uyaw)                     # [E,N]

        # self
        W = cfg.world_size
        self_f = np.concatenate([
            np.broadcast_to(c.type_onehot[None], (E, N, 3)),
            (self.uxy / W), s_p[..., None], c_p[..., None],
            cf[..., None], rf[..., None]], axis=-1) * alive[..., None]

        # teammates (same component, alive, not self)
        fwd, rgt = self._ego(g.duu_x, g.duu_y)                              # [E,i,j]
        s_o, c_o = s_p[:, None, :], c_p[:, None, :]
        sin_rel = s_o * c_p[:, :, None] - c_o * s_p[:, :, None]
        cos_rel = c_o * c_p[:, :, None] + s_o * s_p[:, :, None]
        mate_f = np.concatenate([
            np.broadcast_to(c.type_onehot[None, None], (E, N, N, 3)),
            np.stack([fwd / L, rgt / L, g.duu / L, sin_rel, cos_rel], -1)], axis=-1)
        mate_valid = det.R & ~self._eye[None]
        mates, mates_m = self._sorted(mate_f, mate_valid, g.duu)
        mates, mates_m = mates[:, :, : N - 1], mates_m[:, :, : N - 1]

        # targets (currently detected by someone in the component; no memory)
        fwd, rgt = self._ego(g.dut_x, g.dut_y)
        s_t, c_t = np.sin(self.tyaw)[:, None, :], np.cos(self.tyaw)[:, None, :]
        sin_rel = s_t * c_p[:, :, None] - c_t * s_p[:, :, None]
        cos_rel = c_t * c_p[:, :, None] + s_t * s_p[:, :, None]
        tgt_f = np.stack([fwd / L, rgt / L, g.dut / L, sin_rel, cos_rel], -1)
        targets, targets_m = self._sorted(tgt_f, det.shared_t, g.dut)

        # threats (detected or remembered by someone in the component)
        fwd, rgt = self._ego(g.dth_x, g.dth_y)
        rad = np.broadcast_to((self.thr / cfg.threat_radius_scale)[:, None, :], g.dth.shape)
        thr_f = np.stack([fwd / L, rgt / L, g.dth / L, rad], -1)
        threats, threats_m = self._sorted(thr_f, det.shared_th, g.dth)

        f32 = np.float32
        return {
            "self": self_f.astype(f32),
            "mates": mates.astype(f32), "mates_mask": mates_m,
            "targets": targets.astype(f32), "targets_mask": targets_m,
            "threats": threats.astype(f32), "threats_mask": threats_m,
            "active": alive.copy(),
        }

    def _build_state(self) -> np.ndarray:
        """Centralised critic state in the world frame, one vector per env."""
        cfg, c = self.cfg, self.c
        E, N = self.E, self.N
        W = cfg.world_size
        uav = np.concatenate([
            self.ualive[..., None].astype(np.float64),
            np.broadcast_to(c.type_onehot[None], (E, N, 3)),
            self.uxy / W, np.sin(self.uyaw)[..., None], np.cos(self.uyaw)[..., None]], -1)
        tgt = np.concatenate([
            self.talive[..., None].astype(np.float64), self.txy / W,
            np.sin(self.tyaw)[..., None], np.cos(self.tyaw)[..., None]], -1)
        thr = np.concatenate([self.thxy / W, (self.thr / cfg.threat_radius_scale)[..., None]], -1)
        parts = [uav.reshape(E, -1), tgt.reshape(E, -1), thr.reshape(E, -1)]
        if cfg.state_layout == "full_v1":
            cf, rf = self._factors()
            phase = 2 * np.pi * self.step_count[:, None] / self.jper + self.jph
            jammer = np.concatenate([self.jxy / W, (self.jrad / W)[..., None],
                self.jstr[..., None], (self.jper / 100)[..., None],
                np.sin(phase)[..., None], np.cos(phase)[..., None]], -1)
            parts.extend([self.last_u, self.tomega / math.radians(10),
                self.mem.reshape(E, -1).astype(np.float64), cf, rf,
                jammer.reshape(E, self.J * 7),
                (1 - self.step_count / cfg.max_steps)[:, None]])
        return np.concatenate(parts, -1).astype(np.float32)

    # ------------------------------------------------------------ visualiser
    def snapshot(self, e: int = 0) -> dict:
        """Ground-truth frame of env ``e`` for the 3D viewer (never fed to policies).

        The viewer renders the *real* world (including the hidden jammers); the
        2D simulation has no altitude, any height in the viewer is cosmetic.
        """
        c = self.c
        cf, rf = self._factors()
        det = self._last_det
        inten = self.interference()[e] if self.cfg.contested else np.zeros(self.N)
        links = [[int(i), int(j)] for i, j in zip(*np.nonzero(np.triu(det.A[e])))]
        comp_id, seen = {}, 0
        for i in range(self.N):
            if self.ualive[e, i] and i not in comp_id:
                for j in np.flatnonzero(det.R[e, i]):
                    comp_id[int(j)] = seen
                seen += 1
        names = ("Stk", "Rec", "Com")
        return {
            "step": int(self.step_count[e]),
            "world_size": self.cfg.world_size,
            "uavs": [{
                "id": i, "type": names[c.type_idx[i]], "alive": bool(self.ualive[e, i]),
                "x": float(self.uxy[e, i, 0]), "y": float(self.uxy[e, i, 1]),
                "yaw": float(self.uyaw[e, i]),
                "comm_range": float(c.comm_range[i] * cf[e, i]),
                "recon_range": float(c.recon_range[i] * rf[e, i]),
                "strike_range": float(c.strike_range[i]),
                "strike_angle_deg": float(np.degrees(2 * c.strike_half_angle[i])),
                "comm_quality": float(cf[e, i]), "recon_quality": float(rf[e, i]),
                "interference": float(inten[i]),
                "component": comp_id.get(i, -1),
            } for i in range(self.N)],
            "targets": [{
                "id": t, "alive": bool(self.talive[e, t]),
                "x": float(self.txy[e, t, 0]), "y": float(self.txy[e, t, 1]),
                "yaw": float(self.tyaw[e, t]),
                "detected": bool(det.shared_t[e, :, t].any()),
            } for t in range(self.T)],
            "threats": [{
                "id": k, "x": float(self.thxy[e, k, 0]), "y": float(self.thxy[e, k, 1]),
                "radius": float(self.thr[e, k]),
                "discovered": bool(self.mem[e, :, k].any()),
            } for k in range(self.K)],
            "jammers": [{
                "id": j, "x": float(self.jxy[e, j, 0]), "y": float(self.jxy[e, j, 1]),
                "radius": float(self.jrad[e, j]), "strength": float(self.jstr[e, j]),
                "period": float(self.jper[e, j]), "phase": float(self.jph[e, j]),
            } for j in range(self.J)],
            "links": links,
        }

    # ------------------------------------------------------------- helpers
    @property
    def obs_dim_flat(self) -> int:
        N = self.N
        return (SELF_DIM + (N - 1) * MATE_DIM + self.T * TARGET_DIM + self.K * THREAT_DIM
                + (N - 1) + self.T + self.K)

    @property
    def state_dim(self) -> int:
        physical = self.N * 8 + self.T * 5 + self.K * 3
        return physical if self.cfg.state_layout == "legacy_v1" else (
            physical + 3 * self.N + self.T + self.N * self.K + 7 * self.J + 1)


def flatten_obs(obs: Dict[str, np.ndarray]) -> np.ndarray:
    """Concatenate the entity-set observation into ``[E,N,D]`` for MLP policies."""
    E, N = obs["self"].shape[:2]
    parts = [obs["self"],
             obs["mates"].reshape(E, N, -1), obs["mates_mask"].astype(np.float32),
             obs["targets"].reshape(E, N, -1), obs["targets_mask"].astype(np.float32),
             obs["threats"].reshape(E, N, -1), obs["threats_mask"].astype(np.float32)]
    return np.concatenate(parts, axis=-1)
