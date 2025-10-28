from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import torch


@dataclass
class SoftDynamicObstacleConstraint:
    is_soft: bool = True
    q_l: List = None
    t_range_l: List[Tuple[int, int]] = None
    radius_l: List[float] = None

    def get_q_l(self): return self.q_l

    def get_t_range_l(self): return self.t_range_l


def build_dynamic_constraints_from_env(
        env,  # dynamic envs
        t0: float,  # current t
        H: int,
        dt: float,
        map_size: float,
        radius_scale: float = 1.15,
        hard: bool = False,
) -> List[SoftDynamicObstacleConstraint]:
    constraints: List[SoftDynamicObstacleConstraint] = []

    car_len = float(getattr(env, "car_len", 0.10))
    car_wid = float(getattr(env, "car_wid", 0.10))
    base_r = 0.5 * float(np.hypot(car_len, car_wid)) * float(radius_scale)

    for k in range(H):
        tk = t0 + k * dt
        centers, sizes = env.get_dynamic_boxes(tk, map_size)  # (N,2), (N,2)
        if centers is None or len(centers) == 0:
            continue
        for c in centers:
            constraints.append(
                SoftDynamicObstacleConstraint(
                    is_soft=(not hard),
                    q_l=[np.asarray(c, dtype=float)],
                    t_range_l=[(k, k)],
                    radius_l=[base_r],
                )
            )
    return constraints


class LowerDynamicAdapter(torch.nn.Module):
    def __init__(
            self, base_planner,
            env, H: int,
            dt: float,
            map_size: float,
            radius_scale: float = 1.15,
            hard: bool = False
    ):
        super().__init__()
        self.base = base_planner
        self.env = env
        self.H, self.dt, self.map_size = H, dt, map_size
        self.radius_scale = radius_scale
        self.hard = hard
        self.t0: float = 0.0

        # warm-up
        self.last_experience = getattr(base_planner, "last_experience", None)

    def set_time(self, t0: float):
        self.t0 = float(t0)

    def __getattr__(self, name):
        try:
            return super().__getattribute__(name)
        except AttributeError:
            return getattr(self.base, name)

    def __call__(self, start_state_pos, goal_state_pos, constraints_l=None, experience=None, *args, **kwargs):
        if hasattr(self.env, 'get_dynamic_boxes'):
            dyn_cons = build_dynamic_constraints_from_env(
                self.env, t0=self.t0, H=self.H, dt=self.dt, map_size=self.map_size,
                radius_scale=self.radius_scale, hard=self.hard
            )
            merged = (constraints_l or []) + dyn_cons
        else:
            merged = (constraints_l or [])

        # warm-start
        if experience is None and self.last_experience is not None:
            experience = self.last_experience
        out = self.base(start_state_pos, goal_state_pos, merged, experience, *args, **kwargs)
        if hasattr(self.base, "last_experience"):
            self.last_experience = self.base.last_experience
        return out
