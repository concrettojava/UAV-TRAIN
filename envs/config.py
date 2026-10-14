"""Environment configuration and per-UAV constants.

Physical parameters follow the reference mission and legacy ``info-mappo-clean``
implementation. Named reward/state/time-limit profiles document explicit changes;
this module does not establish strict paper replication.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# Discrete steering inputs (paper Eq. 10).
ACTION_VALUES = np.array([-1.0, -0.4, -0.1, 0.0, 0.1, 0.4, 1.0], dtype=np.float64)
N_ACTIONS = len(ACTION_VALUES)

TYPE_NAMES = ("Stk", "Rec", "Com")

# Paper Table 1.
TYPE_PARAMS = {
    "Stk": dict(strike_range=280.0, strike_angle_deg=45.0, recon_range=320.0,
                comm_range=550.0, speed=40.0, max_turn_deg_s=36.0, collision_range=10.0),
    "Rec": dict(strike_range=150.0, strike_angle_deg=30.0, recon_range=850.0,
                comm_range=550.0, speed=36.0, max_turn_deg_s=18.0, collision_range=10.0),
    "Com": dict(strike_range=180.0, strike_angle_deg=36.0, recon_range=300.0,
                comm_range=1200.0, speed=38.0, max_turn_deg_s=24.0, collision_range=10.0),
}

# Paper Table 2 (x, y, type). yaw = 0 means heading North (+y).
INITIAL_UAVS = (
    (1700.0, 200.0, "Stk"), (1900.0, 200.0, "Stk"),
    (2100.0, 200.0, "Stk"), (2300.0, 200.0, "Stk"),
    (1800.0, 300.0, "Rec"), (2200.0, 300.0, "Rec"),
    (1800.0, 100.0, "Com"), (2200.0, 100.0, "Com"),
)

# Reward ratios gamma[type][ability] (paper Sec. 5.1).
ABILITY_RATIOS = {
    "Stk": (0.7, 0.2, 0.1),
    "Rec": (0.1, 0.7, 0.2),
    "Com": (0.2, 0.1, 0.7),
}


@dataclass(frozen=True)
class EnvConfig:
    world_size: float = 4000.0
    dt: float = 1.0
    max_steps: int = 200
    time_limit_mode: str = "terminal"    # mission deadline; "truncate" for artificial cutoffs
    n_targets: int = 4
    n_threats: int = 3
    target_speed: float = 8.0
    target_turn_accel_std_deg: float = 0.6
    threat_eta: float = 1.8
    scenario: str = "contested"          # "reference" | "contested"
    reward_profile: str = "paper"        # paper | task_aligned | task_v2
    state_layout: str = "full_v1"        # privileged training input; legacy_v1 for old weights
    # contested-only parameters
    jammer_count: int = 2
    jammer_min_radius: float = 700.0
    jammer_max_radius: float = 1000.0
    jammer_min_strength: float = 0.55
    jammer_max_strength: float = 0.75
    min_comm_factor: float = 0.30
    min_recon_factor: float = 0.55
    evasive_trigger_range: float = 700.0
    evasive_turn_rate_deg: float = 10.0
    # observation normalisation
    length_scale: float = 1000.0         # ego-frame lengths are divided by this
    threat_radius_scale: float = 200.0

    @property
    def contested(self) -> bool:
        return self.scenario == "contested"

    @property
    def n_uavs(self) -> int:
        return len(INITIAL_UAVS)


class UAVConsts:
    """Per-UAV constant arrays, shape [N]."""

    def __init__(self):
        types = [t for _, _, t in INITIAL_UAVS]
        self.type_idx = np.array([TYPE_NAMES.index(t) for t in types], dtype=np.int64)
        self.type_onehot = np.eye(3, dtype=np.float64)[self.type_idx]            # [N,3]
        p = [TYPE_PARAMS[t] for t in types]
        self.strike_range = np.array([q["strike_range"] for q in p])
        self.strike_half_angle = np.array([math.radians(q["strike_angle_deg"]) / 2 for q in p])
        self.recon_range = np.array([q["recon_range"] for q in p])
        self.comm_range = np.array([q["comm_range"] for q in p])
        self.speed = np.array([q["speed"] for q in p])
        self.max_turn = np.array([math.radians(q["max_turn_deg_s"]) for q in p])
        self.collision_range = np.array([q["collision_range"] for q in p])
        self.init_xy = np.array([[x, y] for x, y, _ in INITIAL_UAVS], dtype=np.float64)
        ratios = np.array([ABILITY_RATIOS[t] for t in types])                     # [N,3]
        self.ratio_stk, self.ratio_rec, self.ratio_com = ratios[:, 0], ratios[:, 1], ratios[:, 2]
        self.same_type = self.type_idx[:, None] == self.type_idx[None, :]         # [N,N]
