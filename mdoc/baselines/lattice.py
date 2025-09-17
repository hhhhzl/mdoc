"""
Lattice planner — cubic-spiral + quadrant + collision-aware A*
=============================================================
----------------------------------
"""

from __future__ import annotations
import os
import math
import heapq
from dataclasses import dataclass
from typing import List, Optional, Dict, Sequence, Tuple

from torch_robotics.environments import *
from torch_robotics.tasks.tasks_ensemble import PlanningTaskEnsemble

from mdoc.common.experiences import PathBatchExperience
from mdoc.common.constraints import MultiPointConstraint
from mdoc.planners.common import PlannerOutput
from mdoc.common import smooth_trajs
from mdoc.utils.loading import load_params_from_yaml
from mdoc.trainer import get_dataset

@dataclass
class PathObstacle:
    ts: np.ndarray  # (L,)
    pts: np.ndarray  # (L,2)
    radius: float

    def center_at(self, t: float) -> np.ndarray:
        if t <= self.ts[0]:
            return self.pts[0]
        if t >= self.ts[-1]:
            return self.pts[-1]
        k = np.searchsorted(self.ts, t, side="right") - 1
        k = max(0, min(k, len(self.ts) - 2))
        t0, t1 = self.ts[k], self.ts[k + 1]
        a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        return (1 - a) * self.pts[k] + a * self.pts[k + 1]

class EnvWrapper:
    """
    Wraps a torch_robotics EnvBase (e.g., EnvEmptyNoWait2D) to expose:
      - sdf(pts: [N,2]) -> [N]   via compute_sdf
      - dynamic_safe(pts: [N,2], ts: [N], margin: float) -> bool
    """
    def __init__(self, tr_env, dyn_obstacles=None):
        self.tr_env = tr_env
        self.dyn_obs = dyn_obstacles or []  # list[(x, y, t, r)]

    def sdf(self, pts: torch.Tensor) -> torch.Tensor:
        # EnvBase.compute_sdf expects [N,1,dim]; return flattened [N]
        if pts.dim() == 2:
            pts_in = pts.unsqueeze(1)  # [N,1,2]
        elif pts.dim() == 3 and pts.size(1) == 1:
            pts_in = pts
        else:
            raise ValueError(f"Expected [N,2] or [N,1,2], got {pts.shape}")
        sdf = self.tr_env.compute_sdf(pts_in)
        return sdf.view(-1)

    def dynamic_safe(self, pts: torch.Tensor, ts: torch.Tensor, margin: float) -> bool:
        """Safe if each point is either far from every obstacle in space or far in time."""
        if not self.dyn_obs:
            return True
        device = pts.device
        obs = torch.tensor(self.dyn_obs, device=device, dtype=pts.dtype)  # [M,4] (x,y,t,r)
        obs_xy = obs[:, :2]                                              # [M,2]
        obs_t = obs[:, 2]                                               # [M]
        obs_r = obs[:, 3]                                               # [M]

        # space distance from each point to each obstacle center
        d_space = torch.cdist(pts.unsqueeze(0), obs_xy.unsqueeze(0)).squeeze(0)  # [N,M]
        # time difference
        d_time = (ts.unsqueeze(1) - obs_t).abs()                                   # [N,M]
        # safe if outside obstacle radius+margin OR not at the same time
        ok = ((d_space - obs_r) > margin) | (d_time > 1e-3)
        return ok.all().item()

    # optional: forward other attributes you might read (limits, dim, etc.)
    def __getattr__(self, name):
        return getattr(self.tr_env, name)


class LatticeLower:
    def __init__(
            self,
            model_ids: tuple,
            transforms: Dict[int, torch.tensor],
            start_state_pos: torch.tensor,
            goal_state_pos: torch.tensor,
            debug: bool,
            trained_models_dir: str,
            device: str,
            results_dir: str,

            seed: int = 0,
            q_is_workspace: bool = True,
            q_xy_index: tuple = (0, 1),
            default_ob_radius: float = 0.25,
            lattice_ds: float = 1.0,
            lattice_dl: float = 0.5,
            lattice_dt: float = 0.02,
            lattice_max_s: float = 64.0,
            lattice_max_l: float = 3.0,
            lattice_v_set: Sequence[float] = (5.0,),  # default as constant
            lattice_a_set: Sequence[float] = (-2., 0., 2.),
            **kwargs
    ):
        self.device = device
        self.debug = debug
        self.results_dir = results_dir
        self.model_ids = model_ids
        self.transforms = transforms
        self.tensor_args = {'device': self.device, 'dtype': torch.float32}
        self.trained_models_dir = trained_models_dir
        self.seed = seed
        self.q_is_workspace = q_is_workspace
        self.q_xy_index = q_xy_index
        self.default_ob_radius = default_ob_radius

        self.q_start = start_state_pos.to(**self.tensor_args).squeeze(0) + self.transforms[0]
        self.q_goal = goal_state_pos.to(**self.tensor_args).squeeze(0) + self.transforms[len(self.transforms) - 1]
        print("[LatticeLower] Start state: ", self.q_start)
        print("[LatticeLower] Goal state: ", self.q_goal)

        model_dirs, args = [], []
        tasks = {}
        datasets = []
        for j, model_id in enumerate(model_ids):
            model_dir = os.path.join(trained_models_dir, model_id)
            model_dirs.append(model_dir)
            args.append(load_params_from_yaml(os.path.join(model_dir, 'args.yaml')))
            train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
                dataset_class='TrajectoryDataset',
                use_extra_objects=True,
                obstacle_cutoff_margin=0.01,
                **args[-1],
                tensor_args=self.tensor_args
            )
            dataset = train_subset.dataset
            datasets.append(dataset)
            robot = dataset.robot
            robot.radius *= 0.9
            if j == 0:
                self.robot = robot
            task = dataset.task
            tasks[j] = task

        self.task = PlanningTaskEnsemble(
            tasks, transforms,
            tensor_args=self.tensor_args
        )
        self.q_limits = self.task.ws_limits
        limits = self.q_limits.detach().cpu().numpy()
        x_min, y_min = float(limits[0, 0]), float(limits[0, 1])
        x_max, y_max = float(limits[1, 0]), float(limits[1, 1])
        robot_radius = float(self.robot.radius)

        def box_sdf(pts: torch.Tensor) -> torch.Tensor:
            dx1 = pts[:, 0] - x_min
            dx2 = x_max - pts[:, 0]
            dy1 = pts[:, 1] - y_min
            dy2 = y_max - pts[:, 1]
            d = torch.stack([dx1, dx2, dy1, dy2], dim=1).min(dim=1).values
            return d

        self.env_class = self.model_ids[0].split("-")[0]
        self.base_env = EnvWrapper(globals()[self.env_class](
            precompute_sdf_obj_fixed=True,
            sdf_cell_size=0.01,
            tensor_args=self.tensor_args
        ))
        self.l_params = PlannerParams(
            ds=lattice_ds,
            dl=lattice_dl,
            dt=lattice_dt,
            max_s=lattice_max_s,
            max_l=lattice_max_l,
            v_set=tuple(lattice_v_set),
            a_set=tuple(lattice_a_set),
            robot_radius=robot_radius,
            device=self.device
        )

    def _constraints_to_path_obstacles(self, constraints_l: Optional[List]) -> List[PathObstacle]:
        if not constraints_l:
            return []

        def q_to_xy(q: torch.Tensor) -> np.ndarray:
            if self.q_is_workspace:
                q_np = q.detach().cpu().numpy()
                if q_np.shape[-1] >= 2:
                    return np.array([q_np[self.q_xy_index[0]], q_np[self.q_xy_index[1]]], dtype=float)
                return np.array(q_np[:2], dtype=float)
            else:
                raise NotImplementedError("Please implement q->(x,y) mapping when q_is_workspace=False.")

        path_obs: List[PathObstacle] = []
        for c in constraints_l:
            if isinstance(c, MultiPointConstraint):
                for (q, (t0, t1), r) in zip(c.q_l, c.t_range_l, getattr(c, "radius_l", [])):
                    xy = q_to_xy(q)
                    ts = np.array([float(t0), float(t1)], dtype=float)
                    pts = np.stack([xy, xy], axis=0)
                    rad = float(r) if r is not None else self.default_ob_radius
                    path_obs.append(PathObstacle(ts=ts, pts=pts, radius=rad))
            else:
                pass
        return path_obs

    def __call__(
            self,
            start_state_pos: torch.Tensor,
            goal_state_pos: torch.Tensor,
            constraints_l: Optional[List] = None,
            experience: Optional[PathBatchExperience] = None,
            *args, **kwargs
    ) -> PlannerOutput:

        def tens2xy(t: torch.Tensor) -> np.ndarray:
            t = t.detach().cpu()
            if self.q_is_workspace:
                if t.numel() >= 2:
                    return np.array([t[self.q_xy_index[0]].item(), t[self.q_xy_index[1]].item()], dtype=float)
                return np.array(t[:2].tolist(), dtype=float)
            else:
                raise NotImplementedError("Please implement q->(x,y) mapping when q_is_workspace=False.")

        start_xy = tens2xy(start_state_pos)
        goal_xy = tens2xy(goal_state_pos)
        path_obs = self._constraints_to_path_obstacles(constraints_l)

        dyn_obs: List[Tuple[float, float, float, float]] = []
        for po in path_obs:
            t0, t1 = float(po.ts[0]), float(po.ts[-1])
            if t1 < t0:
                t0, t1 = t1, t0
            n = max(1, int(np.ceil((t1 - t0) / max(1e-3, float(self.l_params.dt)))))
            ts = np.linspace(t0, t1, n + 1)
            for tt in ts:
                dyn_obs.append((float(po.pts[0, 0]), float(po.pts[0, 1]), float(tt), float(po.radius)))

        env = EnvWrapper(globals()[self.env_class](
            precompute_sdf_obj_fixed=True,
            sdf_cell_size=0.01,
            tensor_args=self.tensor_args
        ))

        # lattice (s, l)
        s_goal = float(goal_xy[0] - start_xy[0])
        l_goal = float(goal_xy[1] - start_xy[1])

        self.l_params.max_s = max(self.l_params.max_s, abs(s_goal) + 1.0)
        planner = LatticePlanner(
            self.l_params,
            env
        )
        path_sl = planner.plan(v0_idx=0, a0_idx=0)
        out = PlannerOutput()
        if path_sl is None:
            out.trajs_iters = None
            out.trajs_final = None
            out.trajs_final_coll = None
            out.trajs_final_coll_idxs = torch.zeros(0, dtype=torch.long)
            out.trajs_final_free = None
            out.trajs_final_free_idxs = torch.zeros(0, dtype=torch.long)
            out.success_free_trajs = False
            out.fraction_free_trajs = 0.0
            out.collision_intensity_trajs = 1.0
            out.idx_best_traj = None
            out.traj_final_free_best = None
            out.t_total = 0.0
            out.constraints_l = constraints_l
            return out

        # (s,l) traj project back to (x,y) = (s + x0, l + y0)
        path_xy = np.array([(s + start_xy[0], l + start_xy[1]) for (s, l) in path_sl], dtype=np.float32)
        traj = torch.from_numpy(path_xy).to(self.tensor_args["device"]).unsqueeze(0)  # [1, H, 2]

        ds = float(self.l_params.ds)
        v0 = float(self.l_params.v_set[0]) if len(self.l_params.v_set) > 0 else 1.0
        t_total = float((len(path_sl) - 1) * ds / max(v0, 1e-3))

        print(traj)
        breakpoint()
        out = PlannerOutput()
        out.trajs_iters = [traj]
        out.trajs_final = traj
        out.trajs_final_coll = None
        out.trajs_final_coll_idxs = torch.zeros(0, dtype=torch.long)
        out.trajs_final_free = traj
        out.trajs_final_free_idxs = torch.tensor([0], dtype=torch.long)
        out.success_free_trajs = True
        out.fraction_free_trajs = 1.0
        out.collision_intensity_trajs = 0.0
        out.idx_best_traj = 0
        out.traj_final_free_best = traj[0]
        out.cost_best_free_traj = None
        out.cost_smoothness = None
        out.cost_path_length = None
        out.cost_all = None
        out.variance_waypoint_trajs_final_free = None
        out.t_total = t_total
        out.constraints_l = constraints_l
        out.trajs_final = smooth_trajs(out.trajs_final)
        return out


# ----------------------------------------------------------------------------
# -----------------------------------------------------
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
            dt: float = 0.02,
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


def _cubic_spiral_coeffs(
        pose0: torch.Tensor,
        pose1: torch.Tensor,
        max_iter: int = 20,
        eps: float = 1e-6
):
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
    def __init__(
            self, params: PlannerParams,
            env: Env2D
    ):
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

        return None

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

