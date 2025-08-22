# -*- coding: utf-8 -*-
"""
Lattice planner — cubic-spiral + quadrant + collision-aware A*
=============================================================
----------------------------------
"""
from __future__ import annotations
import math
import heapq
from typing import Tuple, List, Dict, Optional, Sequence
import torch


# ----------------------------------------------------------------------------
# Environment Abstraction -----------------------------------------------------
class Env2D:
    """Minimal 2‑D environment wrapper providing an SDF and dynamic obstacles."""

    def __init__(self, sdf_fun, dyn_obstacles: Optional[Sequence[Tuple[float, float, float, float]]] = None):
        """``sdf_fun``: callable( [N,2] tensor ) → signed dist (>0 free) .
        ``dyn_obstacles`` list items: (x, y, t, radius).
        """
        self.sdf_fun = sdf_fun
        self.dyn_obs = dyn_obstacles or []

    def sdf(self, pts: torch.Tensor) -> torch.Tensor:  # [N,2] → [N]
        return self.sdf_fun(pts)

    def dynamic_safe(self, pts: torch.Tensor, ts: torch.Tensor, margin: float) -> bool:
        if not self.dyn_obs:
            return True
        obs = torch.tensor(self.dyn_obs, device=pts.device)  # [M,4]
        obs_xy = obs[:, :2]  # [M,2]
        obs_t = obs[:, 2]  # [M]
        obs_r = obs[:, 3]  # [M]
        # pts [N,2], ts [N]
        d_space = torch.cdist(pts.unsqueeze(0), obs_xy.unsqueeze(0)).squeeze(0)  # [N,M]
        d_time = (ts.unsqueeze(1) - obs_t).abs()  # [N,M]
        ok = ((d_space - obs_r) > margin) | (d_time > 1e-3)  # either far in space or wrong time
        return ok.all().item()


# ----------------------------------------------------------------------------
# Parameters ------------------------------------------------------------------
class PlannerParams:
    def __init__(
            self,
            ds: float = 1.0,
            dl: float = 0.5,
            v_set: Tuple[float, ...] = (0.5, 5., 10.),
            a_set: Tuple[float, ...] = (-2., 0., 2.),
            dt: float = 0.2,
            max_s: float = 64.,
            max_l: float = 3.,
            robot_radius: float = 0.5,
            device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.ds, self.dl = ds, dl
        self.v_set, self.a_set = v_set, a_set
        self.dt = dt
        self.max_s, self.max_l = max_s, max_l
        self.robot_radius = robot_radius
        self.device = device


# ----------------------------------------------------------------------------
# Cubic spiral solver ---------------------------------------------------------

def _angle_diff(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return ((a - b + math.pi) % (2 * math.pi)) - math.pi


def _cubic_spiral_coeffs(pose0: torch.Tensor, pose1: torch.Tensor,
                         max_iter: int = 20, eps: float = 1e-6):
    """Return coeff [a,b,c,d] and arc-length L."""
    x0, y0, th0, k0 = pose0
    x1, y1, th1, k1 = pose1
    L = torch.hypot(x1 - x0, y1 - y0) + 1e-3
    if L < 1e-2:
        return torch.stack([k0, torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)]), L

    a = k0.clone()
    b = torch.tensor(0.0, device=pose0.device)
    c = torch.tensor(0.0, device=pose0.device)
    d = (k1 - k0) / (L ** 3 + 1e-6)
    max_step = 0.1

    def integrate(a_, b_, c_, d_, L_):
        n = 50
        s = torch.linspace(0., L_, n + 1, device=pose0.device)
        k = a_ + b_ * s + c_ * s ** 2 + d_ * s ** 3
        th = th0 + torch.cumsum((k[:-1] + k[1:]) / 2 * (s[1] - s[0]), 0)
        th = torch.cat([torch.tensor([th0], device=pose0.device), th])
        dx = torch.cos(th) * (s[1] - s[0])
        dy = torch.sin(th) * (s[1] - s[0])
        x = x0 + dx.sum()
        y = y0 + dy.sum()
        return x, y, th[-1], k[-1]

    for _ in range(max_iter):
        x_e, y_e, th_e, k_e = integrate(a, b, c, d, L)
        f = torch.stack([x_e - x1, y_e - y1, _angle_diff(th_e, th1), k_e - k1])
        if torch.linalg.norm(f) < eps:
            break

        J = torch.zeros(4, 4, device=pose0.device)
        delta = 1e-4
        for i, var in enumerate([a, b, c, d]):
            orig = var.clone()
            var += delta
            x_d, y_d, th_d, k_d = integrate(a, b, c, d, L)
            J[:, i] = (torch.stack([x_d - x_e, y_d - y_e,
                                    _angle_diff(th_d, th_e), k_d - k_e]) / delta)
            var.copy_(orig)
        step = torch.linalg.solve(J + 1e-3 * torch.eye(4, device=pose0.device), -f)  # 正则项从 1e-6 → 1e-3
        step = torch.clamp(step, -max_step, max_step)
        a += step[0]
        b += step[1]
        c += step[2]
        d += step[3]
        if torch.isnan(a) or torch.isnan(b) or torch.isnan(c) or torch.isnan(d):
            return torch.stack([k0, torch.tensor(0.), torch.tensor(0.), torch.tensor(0.)]), L

    return torch.stack([a, b, c, d]), L

# ----------------------------------------------------------------------------
# Lattice Planner -------------------------------------------------------------
class LatticePlanner:
    def __init__(self, params: PlannerParams, env: Env2D):
        self.p = params
        self.env = env
        self.device = torch.device(params.device)

        # discrete axes
        self.s_vals = torch.arange(0., params.max_s + 1e-6, params.ds, device=self.device)
        self.l_vals = torch.arange(-params.max_l, params.max_l + 1e-6, params.dl, device=self.device)
        self.nv, self.na = len(params.v_set), len(params.a_set)

        # cost‑to‑come table (not GPU‑huge; stay CPU for memory)
        self.g_cost = {}
        # pre‑compute edges + prune by collision
        self._build_edge_bank()

    # ------------------------------------------------------------------
    def _build_edge_bank(self):
        self.edge_bank: Dict[Tuple[int, int], List[Dict]] = {}
        offsets = [(1, 0), (0, 1), (1, 1), (1, -1)]
        for i_s in range(len(self.s_vals) - 1):
            for i_l in range(1, len(self.l_vals) - 1):
                edges = []
                start_pose = torch.tensor([self.s_vals[i_s], self.l_vals[i_l], 0., 0.], device=self.device)
                for ds_idx, dl_idx in offsets:
                    j_s, j_l = i_s + ds_idx, i_l + dl_idx
                    if j_s >= len(self.s_vals) or not (0 <= j_l < len(self.l_vals)):
                        continue
                    end_pose = torch.tensor([self.s_vals[j_s], self.l_vals[j_l], 0., 0.], device=self.device)
                    coeffs, L = _cubic_spiral_coeffs(start_pose, end_pose)

                    # sample 40 pts along curve in world coordinates ≈ (s,ℓ)
                    s_samp = torch.linspace(0., L, 40, device=self.device)
                    k = coeffs[0] + coeffs[1] * s_samp + coeffs[2] * s_samp ** 2 + coeffs[3] * s_samp ** 3
                    th = torch.cumsum(k, 0) * (s_samp[1] - s_samp[0])  # θ(s) incremental
                    x = start_pose[0] + torch.cumsum(torch.cos(th) * (s_samp[1] - s_samp[0]), 0)
                    y = start_pose[1] + torch.cumsum(torch.sin(th) * (s_samp[1] - s_samp[0]), 0)
                    pts = torch.stack([x, y], 1)
                    # collision test static
                    if self.env.sdf(pts).min() <= self.p.robot_radius:
                        continue  # static collision → discard edge
                    edges.append({
                        "dst": (j_s, j_l),
                        "coeffs": coeffs,
                        "length": L,
                        "pts": pts,  # cached for dynamic test
                        "k_int": (coeffs ** 2).sum() * L / 3,
                    })
                if edges:
                    self.edge_bank[(i_s, i_l)] = edges

    # ------------------------------------------------------------------
    def _heuristic(self, is_idx: int, il_idx: int) -> float:
        s_rem = (len(self.s_vals) - 1 - is_idx) * self.p.ds
        l_abs = abs(self.l_vals[il_idx].item())
        v_max = max(self.p.v_set)
        return (s_rem + 0.5 * l_abs) / (v_max + 1e-3)

    # ------------------------------------------------------------------
    def plan(self, v0_idx: int = 0, a0_idx: int = 0):
        """A* search that also keeps *absolute time* at each state for dynamic checks."""
        start_state = (0, self._l_idx(0.), v0_idx, a0_idx)
        g_cost: Dict[Tuple[int, int, int, int], float] = {start_state: 0.0}
        g_time: Dict[Tuple[int, int, int, int], float] = {start_state: 0.0}
        parent: Dict[Tuple[int, int, int, int], Tuple[int, int, int, int]] = {}

        open_heap: List[Tuple[float, Tuple[int, int, int, int]]] = []
        heapq.heappush(open_heap, (self._heuristic(*start_state[:2]), start_state))

        while open_heap:
            _, state = heapq.heappop(open_heap)
            is_idx, il_idx, iv_idx, ia_idx = state
            g = g_cost[state]
            t_abs = g_time[state]
            # goal check (last longitudinal column)
            if is_idx == len(self.s_vals) - 1:
                return self._reconstruct(state, parent)

            # expand
            for edge in self.edge_bank.get((is_idx, il_idx), []):
                js, jl = edge["dst"]
                for dv, da in [(0, 0), (0, 1), (0, -1)]:
                    iv_new = min(max(iv_idx + dv, 0), self.nv - 1)
                    ia_new = min(max(ia_idx + da, 0), self.na - 1)
                    new_state = (js, jl, iv_new, ia_new)

                    # travel time on this edge (constant‑velocity approximation)
                    v_curr = max(self.p.v_set[iv_idx], 0.1)
                    t_edge = edge["length"] / v_curr

                    # absolute timestamps of the 40 sample points along the edge
                    ts = t_abs + torch.linspace(0.0, t_edge, edge["pts"].shape[0], device=self.device)

                    # dynamic‑obstacle safety
                    if not self.env.dynamic_safe(edge["pts"], ts, self.p.robot_radius + 1e-3):
                        continue

                    # cumulative cost & time
                    g_new = g + self._edge_cost(edge, iv_idx, ia_idx, iv_new)
                    t_new = t_abs + t_edge

                    if g_new < g_cost.get(new_state, float("inf")):
                        g_cost[new_state] = g_new
                        g_time[new_state] = t_new
                        parent[new_state] = state
                        f = g_new + self._heuristic(js, jl)
                        heapq.heappush(open_heap, (f, new_state))

        raise RuntimeError("No feasible path to goal")

    # ------------------------------------------------------------------

    def _edge_cost(self, edge: Dict, vi: int, ai: int, vj: int) -> float:
        v = max(self.p.v_set[vi], 0.1)
        v_new = self.p.v_set[vj]
        a = self.p.a_set[ai]
        L = edge["length"]
        t = L / (v + 1e-3)
        jerk = (v_new - v) / (t + 1e-3) - a
        jerk = max(min(jerk, 10.0), -10.0)
        cost = 5.0 * edge["k_int"] + t + 0.5 * jerk ** 2 + 2.0 * abs(edge["coeffs"][0].item())
        if not math.isfinite(cost):
            return float("inf")
        return float(cost)

    # ------------------------------------------------------------------
    def _l_idx(self, l: float) -> int:
        return int(round((l + self.p.max_l) / self.p.dl))

    # ------------------------------------------------------------------
    def _reconstruct(self, goal_state, parent):
        seq = [goal_state]
        while seq[-1] in parent:
            seq.append(parent[seq[-1]])
        seq.reverse()
        return [(self.s_vals[s].item(), self.l_vals[l].item()) for s, l, _, _ in seq]


import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# noqa: F401
ds_val = 1
v_nominal = 5.0
robot_r = 0.5
common_params = PlannerParams(
    robot_radius=robot_r,
    device="cpu",
    v_set=[v_nominal]
)

# -------- 1. Plan without obstacles --------
empty_env = Env2D(lambda pts: torch.full((pts.size(0),), 10.0, device=pts.device), dyn_obstacles=[])
planner0 = LatticePlanner(common_params, empty_env)
path0 = planner0.plan()                                 # baseline path
# s0, l0 = np.array(path0).T
# t0 = np.arange(len(s0)) * ds_val / v_nominal
#
# # -------- 2. Plan with obstacles --------
# dyn_obs = [(30.0, 0.0, 6.0, 0.5)]                      # (x, y, t, r)
# obst_env = Env2D(lambda pts: torch.full((pts.size(0),), 10.0, device=pts.device), dyn_obstacles=dyn_obs)
#
# planner1 = LatticePlanner(common_params, obst_env)
# path1 = planner1.plan()                                 # avoidance path
# s1, l1 = np.array(path1).T
# t1 = np.arange(len(s1)) * ds_val / v_nominal
#
# # safety test
# x_o, y_o, t_o, r_o = dyn_obs[0]
# d_space = np.hypot(s1 - x_o, l1 - y_o) - (r_o + robot_r)
# safe_mask = np.logical_or(d_space > 0, np.abs(t1 - t_o) > 1e-4)
#
# fig2d, ax2d = plt.subplots(figsize=(9,3))
# ax2d.plot(s0, l0, '-k', lw=2, label='No-obstacle path')
# ax2d.plot(s1, l1, '-ob', ms=4, label='With-obstacle path')
# ax2d.add_patch(plt.Circle((x_o, y_o), r_o, fc='none', ec='r', ls='--', lw=1.5))
# ax2d.set_xlabel("s (m)"); ax2d.set_ylabel("ℓ (m)")
# ax2d.set_title("2-D projection: paths comparison")
# ax2d.axis('equal'); ax2d.grid(True); ax2d.legend()
#
# fig3d = plt.figure(figsize=(7,5))
# ax3d = fig3d.add_subplot(111, projection='3d')
# ax3d.plot(s0, l0, t0, lw=2, c='k', label='No-obstacle')
# ax3d.plot(s1, l1, t1, lw=2, c='b', label='With-obstacle')
# ax3d.scatter(x_o, y_o, t_o, s=250, c='r', alpha=0.7, label='Pedestrian')
# ax3d.plot([x_o, x_o], [y_o, y_o], [t_o-0.3, t_o+0.3], 'r--', lw=1)
# ax3d.set_xlabel('s (m)'); ax3d.set_ylabel('ℓ (m)'); ax3d.set_zlabel('time (s)')
# ax3d.set_title("3-D space-time comparison")
# ax3d.legend()
#
# plt.tight_layout()
# plt.show()
#
# SAVE_GIF = True
# if SAVE_GIF:
#     from matplotlib.animation import FuncAnimation, PillowWriter
#     fig, ax = plt.subplots(figsize=(9,3))
#     ax.set_xlim(0, max(s1.max(), s0.max())+1); ax.set_ylim(-1.5, 1.5)
#     ax.set_xlabel("s (m)"); ax.set_ylabel("ℓ (m)")
#     ax.set_title("Time-synchronized view")
#     ax.plot(s0, l0, lw=1, color='0.7')
#     ax.plot(s1, l1, lw=1, color='0.4')
#     ped_circle = plt.Circle((x_o, y_o), r_o, fc='none', ec='r', ls='--', lw=1.5)
#     ax.add_patch(ped_circle)
#     robot_dot0, = ax.plot([], [], 'ko', ms=6)
#     robot_dot1, = ax.plot([], [], 'bo', ms=6)
#     time_txt = ax.text(0.02, 0.95, '', transform=ax.transAxes)
#
#     frames = max(len(t0), len(t1))
#     def update(i):
#         if i < len(t0):
#             robot_dot0.set_data(s0[i], l0[i])
#         if i < len(t1):
#             robot_dot1.set_data(s1[i], l1[i])
#         time_txt.set_text(f"t ≈ {i*ds_val/v_nominal:.1f} s")
#         return robot_dot0, robot_dot1, time_txt
#     ani = FuncAnimation(fig, update, frames=frames, interval=50, blit=True)
#     ani.save("lattice.gif", dpi=120, writer=PillowWriter())
#     print("saved lattice.gif")
