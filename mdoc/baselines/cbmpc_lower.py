import time
from typing import List, Optional, Dict
import torch
from mdoc.planners.common import PlannerOutput, SingleAgentPlanner


def _as_tensor(x, ref: torch.Tensor):
    return torch.as_tensor(x, dtype=ref.dtype, device=ref.device)


class CBMPCLower(SingleAgentPlanner):
    """
    MPC lower：
    - dynamic model：single/double integrator
    - cost：Q + R + P
    - constraints:
    - solver：1. CasADi+IPOPT；2. penalty + projected-GD
    """

    def __init__(
            self,
            H: int,
            results_dir: str,
            # model_ids: tuple,
            # transforms: Dict[int, torch.tensor],
            # start_state_pos: torch.tensor,
            # goal_state_pos: torch.tensor,
            # debug: bool,
            # trained_models_dir: str,
            # seed: int = 0,
            dt: float = 0.1,
            device: str = "cpu",
            q_dim: int = 2,
            dynamics: str = "single_integrator",  # "single_integrator" | "double_integrator"
            Q: float = 30.0,
            R: float = 0.05,
            P: float = 30.0,
            v_max: float = 1.0,
            a_max: float = 1.0,
            sep_margin: float = 0.25,
            **kwargs
    ):
        super().__init__()
        self.H, self.dt = H, dt
        self.q_dim = q_dim
        self.dynamics = dynamics
        self.Q = Q
        self.R = R
        self.P = P
        self.v_max = v_max
        self.a_max = a_max
        self.sep_margin = sep_margin
        self.tensor_args = dict(device=torch.device(device), dtype=torch.float32)
        self.results_dir = results_dir
        try:
            import casadi as ca
            self._casadi = ca
        except Exception:
            self._casadi = None

        # warm-start
        self.last_solution: Optional[torch.Tensor] = None  # (H,q_dim)

    def __call__(
            self,
            start_state_pos: torch.Tensor,  # (q_dim,)
            goal_state_pos: torch.Tensor,  # (q_dim,)
            constraints_l: Optional[List] = None,
            experience=None,
            *args, **kwargs
    ) -> PlannerOutput:
        start = start_state_pos.to(**self.tensor_args).view(-1)
        goal = goal_state_pos.to(**self.tensor_args).view(-1)

        # init
        if experience is not None and getattr(experience, "path_b", None) is not None:
            x_init = experience.path_b[0]  # (H,q_dim)
        elif self.last_solution is not None:
            x_init = self.last_solution
        else:
            t = torch.linspace(0, 1, self.H, **self.tensor_args).unsqueeze(-1)
            x_init = (1 - t) * start + t * goal  # (H,q_dim)

        # solve
        if self._casadi is not None:
            x_opt = self._solve_ipopt(start, goal, constraints_l, x_init)
        else:
            x_opt = self._solve_pg(start, goal, constraints_l, x_init)
        self.last_solution = x_opt.detach()

        traj = x_opt.unsqueeze(0)
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
        # out.t_total = float(ts[-1])
        out.constraints_l = constraints_l
        return out

    # ================== CasADi ==================
    def _solve_ipopt(self, start, goal, constraints_l, x_init) -> torch.Tensor:
        ca = self._casadi
        H, dt, D = self.H, self.dt, self.q_dim
        x = ca.MX.sym('x', H, D)
        u = ca.MX.sym('u', H - 1, D) if self.dynamics == "double_integrator" else None

        cost = 0
        for t in range(H - 1):
            xt = x[t, :]
            xt1 = x[t + 1, :]
            # single integrator：xt1 = xt + v*dt
            if self.dynamics == "single_integrator":
                v = (xt1 - xt) / dt
                cost += self.Q * ca.sumsqr(xt - self._to_ca(goal)) + self.R * ca.sumsqr(v)
                for d in range(D):
                    cost += 1e3 * ca.fmax(0, ca.fabs(v[d]) - self.v_max) ** 2
            else:
                # double：xt1 = xt + vt*dt + 0.5*a*dt^2
                at = u[t, :]
                cost += self.Q * ca.sumsqr(xt - self._to_ca(goal)) + self.R * ca.sumsqr(at)

        # cost
        cost += self.P * ca.sumsqr(x[H - 1, :] - self._to_ca(goal))

        g = []

        # init
        g += [x[0, :] - self._to_ca(start)]

        # dynamics
        if self.dynamics == "single_integrator":
            for t in range(H - 1):
                g += [x[t + 1, :] - x[t, :] - dt * (x[t + 1, :] - x[t, :]) / dt]
        else:
            v = ca.MX.zeros(H, D)
            for t in range(H - 1):
                v[t + 1, :] = v[t, :] + u[t, :] * dt
                g += [x[t + 1, :] - (x[t, :] + v[t, :] * dt + 0.5 * u[t, :] * (dt ** 2))]
                for d in range(D):
                    g += [ca.fmax(0, ca.fabs(u[t, d]) - self.a_max)]

        if constraints_l is not None:
            for c in constraints_l:
                is_soft = getattr(c, "is_soft", False)
                radius_l = getattr(c, "radius_l", None)
                q_l = c.get_q_l() if hasattr(c, "get_q_l") else getattr(c, "q_l", [])
                t_range_l = c.get_t_range_l() if hasattr(c, "get_t_range_l") else getattr(c, "t_range_l", [])
                for idx, q_ref in enumerate(q_l):
                    t0, t1 = t_range_l[idx]
                    r = radius_l[idx] if radius_l is not None else self.sep_margin
                    t0 = max(0, int(t0))
                    t1 = min(H - 1, int(t1))
                    for t in range(t0, t1 + 1):
                        sep = ca.norm_2(x[t, :] - self._to_ca(q_ref))
                        if is_soft:
                            cost += 1e4 * ca.fmax(0, (r - sep)) ** 2
                        else:
                            g += [ca.fmax(0, (r - sep))]

        w = [x]
        lbw, ubw = [], []
        w0 = x_init.cpu().numpy()

        nlp = {'x': ca.vertcat(*[ca.reshape(x, -1, 1)]),
               'f': cost,
               'g': ca.vertcat(*g) if len(g) > 0 else ca.MX.zeros(1)}

        solver = ca.nlpsol(
            'solver', 'ipopt', nlp,
            {'ipopt.print_level': 0, 'print_time': 0, 'ipopt.max_iter': 200}
        )

        sol = solver(lbx=-ca.inf, ubx=ca.inf, lbg=-ca.inf, ubg=0, x0=w0.reshape(-1, 1))
        x_opt = sol['x'].full().reshape(H, D)
        x_opt = torch.tensor(x_opt, **self.tensor_args)
        return x_opt

    # ============== solver PG ==============
    def _solve_pg(self, start, goal, constraints_l, x_init) -> torch.Tensor:
        x = x_init.clone().detach().requires_grad_(True)  # (H,D)
        opt = torch.optim.Adam([x], lr=0.01)
        H, dt = self.H, self.dt

        def loss_fn(x):
            w_track = self.Q
            w_ctrl = self.R
            w_term = self.P
            track = ((x[:-1] - goal) ** 2).sum()
            vel = (x[1:] - x[:-1]) / max(dt, 1e-3)
            ctrl = (vel ** 2).sum()
            term = ((x[-1] - goal) ** 2).sum()
            pen = 0.0
            pen += 1e4 * ((x[0] - start) ** 2).sum()
            pen += 1e3 * torch.clamp(vel.abs() - self.v_max, min=0).pow(2).sum()
            if constraints_l is not None:
                for c in constraints_l:
                    is_soft = getattr(c, "is_soft", False)
                    radius_l = getattr(c, "radius_l", None)
                    q_l = c.get_q_l() if hasattr(c, "get_q_l") else getattr(c, "q_l", [])
                    t_range_l = c.get_t_range_l() if hasattr(c, "get_t_range_l") else getattr(c, "t_range_l", [])
                    for idx, q_ref in enumerate(q_l):
                        t0, t1 = t_range_l[idx]
                        r = radius_l[idx] if radius_l is not None else self.sep_margin
                        t0 = max(0, int(t0))
                        t1 = min(H - 1, int(t1))
                        for t in range(t0, t1 + 1):
                            sep = torch.norm(x[t] - _as_tensor(q_ref, x))
                            pen += (1e6 if not is_soft else 1e6) * torch.clamp(r - sep, min=0).pow(2)

            return w_track * track + w_ctrl * ctrl + w_term * term + pen

        for _ in range(200):
            opt.zero_grad()
            loss = loss_fn(x)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([x], max_norm=5.0)
            opt.step()

        return x.detach()

    # helper
    def _to_ca(self, v):
        ca = self._casadi
        if isinstance(v, torch.Tensor):
            return ca.vcat([float(x) for x in v.view(-1).cpu()])
        elif isinstance(v, (list, tuple)):
            return ca.vcat([float(x) for x in v])
        else:
            return ca.vcat([float(v)])
