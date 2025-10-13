import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List, Tuple, Callable, Optional
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from mdoc.config import MDOCParams as mparams
import einops
from torch_robotics.robots.robot_base import RobotState
from mp_baselines.planners.costs.cost_functions import CostConstraint
import torch._dynamo
import torch, contextlib
from mdoc.utils.mdoc_constraints import *

torch._dynamo.config.suppress_errors = True
torch.set_num_threads(16)
torch.set_num_interop_threads(1)

from dataclasses import dataclass


@dataclass
class ConstraintTensor:
    C2: torch.Tensor  # (M_max, N_max, 2), float32, contiguous
    R: torch.Tensor  # (M_max, N_max),    float32, contiguous
    TM: torch.Tensor  # (M_max, N_max, H), bool,     contiguous


def pack_constraints_dict(constraints: dict[int, list],
                          *, M_max: int, N_max: int, D: int, H: int,
                          device: torch.device) -> ConstraintTensor:
    C2 = torch.zeros(M_max, N_max, 2, dtype=torch.float32, device=device)
    R = torch.zeros(M_max, N_max, dtype=torch.float32, device=device)
    TM = torch.zeros(M_max, N_max, H, dtype=torch.bool, device=device)

    for mid, sph_list in constraints.items():
        if not (0 <= mid < M_max):
            continue
        n_filled = 0
        for sph in sph_list:
            if n_filled >= N_max:
                break
            if not sph:
                continue
            qs = sph.qs.to(device)  # (n_i, D)
            rad = sph.radii.to(device)  # (n_i,)
            tr = sph.traj_ranges.to(device).long()  # (n_i, 2)

            ni = min(qs.shape[0], N_max - n_filled)
            if ni <= 0:
                continue

            C2[mid, n_filled:n_filled + ni, :] = qs[:ni, :2].to(torch.float32)
            R[mid, n_filled:n_filled + ni] = rad[:ni].to(torch.float32)
            t_idx = torch.arange(H, device=device).view(1, H)
            starts = tr[:ni, 0:1]
            ends = tr[:ni, 1:2]
            TM[mid, n_filled:n_filled + ni, :] = (t_idx >= starts) & (t_idx < ends)

            n_filled += ni

    return ConstraintTensor(
        C2=C2.contiguous(),
        R=R.contiguous(),
        TM=TM.contiguous(),
    )


class ModelBasedDiffusionEnsemble(nn.Module):
    """
    Unlike Model-free diffusion,
    model-based diffusion interact with model information,
    and analytically calculate score function through Monte Carlo Approximation
    """

    def __init__(
            self,
            robot,
            env_models: Dict[int, Any],
            transforms: Dict[int, torch.Tensor],
            start_state_pos: torch.tensor,
            goal_state_pos: torch.tensor,
            seed: int = 0,
            enable_demo: bool = False,
            context_model=None,
            disable_recommended_params: bool = True,
            device: str = 'cpu',
            compile: bool = True,
            use_cuda_graph: bool = True,
            **kwargs
    ):
        """

        Args:
            env_models: model_environments {model_id: env_model}
            env_name: model_env name
            seed: random seed
            enable_demo: whether to use demonstration
            disable_recommended_params: whether to use recommended parames
            device: device
        """
        super().__init__(**kwargs)
        self.device = device
        self.robot = robot
        self.env_models = env_models
        self.enable_demo = enable_demo
        self.disable_recommended_params = disable_recommended_params

        torch.manual_seed(seed)
        np.random.seed(seed)

        self.diffusion_params = {
            'temp_sample': mparams.temp_sample,
            'n_samples': mparams.n_samples,
            'n_diffusion_step': mparams.n_diffusion_steps,
            'horizon': 64,
            'beta0': 1e-5,
            'betaT': 1e-2
        }

        self.recommended_params = {
            "temp_sample": 0.5,
            "n_diffusion_step": 100,
            'n_samples': 64,
            "horizon": 64
        }

        # Initialization
        self.models = {}
        self.soft_constraints = {}
        self.start_state_pos = start_state_pos
        self.goal_state_pos = goal_state_pos
        for model_id, env_model in env_models.items():
            self._setup_model_environment(model_id, env_model)
            self.soft_constraints[model_id] = []
            # self._add_endpoint_soft_constraints(model_id)
        self.transforms = transforms
        self.context_model = context_model

        # for cache (ECBS calculate collision cost)
        self._centers = None
        self._radii2 = None
        self._time_mask = None

        # for non empty environment
        sdf_cell_size = 0.05
        self.sdf_cell_size = float(sdf_cell_size)
        self._sdf_tex = None
        self._sdf_tex_meta = None

        self.compile = compile
        if self.device == 'mps':
            self.compile = False
        if self.compile:
            print(">>>>>> Preheat Compile >>>>>>>>")
            self._preheat(use_cuda_graph)

    def _prepare_soft_constraints(
            self,
            model_id: int,
            H: int,
            device: torch.device
    ):
        """
        Cache for diffusion step：
            _centers    [1,1,N,q_dim]
            _radii2     [1,1,N]
            _time_mask  [1,H,N,1]
        """
        qs_l, radii_l, t0_l, t1_l, w_l = [], [], [], [], []
        for c in self.soft_constraints[model_id]:
            qs_l.append(c.qs.to(device))  # [K,q_dim]
            radii_l.append(torch.as_tensor(c.radii, device=device))
            tr = torch.as_tensor(c.traj_ranges, device=device)  # [K,2]
            t0_l.append(tr[:, 0])
            t1_l.append(tr[:, 1])
            if isinstance(c.priority_weight, torch.Tensor):
                w_l.append(c.priority_weight.to(device))
            else:
                # w_l.append(torch.full((len(c.qs),), c.priority_weight, device=device))
                w_l.append(1)

        centers = torch.cat(qs_l, dim=0)  # [N,q_dim]
        radii = torch.cat(radii_l, dim=0)  # [N]
        self._t0 = torch.cat(t0_l, dim=0)  # [N]
        self._t1 = torch.cat(t1_l, dim=0)  # [N]
        self._N = centers.shape[0]
        self._centers = centers.view(1, 1, -1, self.robot.q_dim)  # [1,1,N,q_dim]
        self._radii = radii.view(1, 1, self._N)  # [1,1,N]
        self._centers_flat = centers  # [N,q_dim]  (contiguous)
        self._radii_flat = radii  # [N]
        # self._weights = torch.cat(w_l).view(1, 1, -1, 1)  # (1,1,N,1)

    def _setup_model_environment(self, model_id: int, env_model: Any):
        params = self.diffusion_params.copy()
        if not self.disable_recommended_params:
            for param_name in ["temp_sample", "n_diffusion_step", "n_samples", "horizon"]:
                if param_name in self.recommended_params:
                    params[param_name] = self.recommended_params[param_name]

        setattr(self, f'model_{model_id}_params', params)
        self.state_inits = RobotState(
            q=self.start_state_pos,
            q_dot=torch.Tensor([0, 0]),
            cost=0,
            collision_cost=0,
            control_cost=0,
            pipeline_state=torch.cat([self.start_state_pos, self.start_state_pos])
        )
        self._setup_diffusion_params(model_id, params)

    def _setup_diffusion_params(self, model_id: int, params: dict):
        """set parameter for each model"""
        betas = torch.linspace(params['beta0'], params['betaT'], params['n_diffusion_step'], device=self.device)
        alphas = 1.0 - betas
        alphas_bar = torch.cumprod(alphas, dim=0)
        sigmas = torch.sqrt(1 - alphas_bar)

        # Conditional parameters
        shifted_alphas_bar = torch.roll(alphas_bar, 1)
        shifted_alphas_bar[0] = 1.0
        sigmas_cond = ((1 - alphas) * (1 - torch.sqrt(shifted_alphas_bar)) / (1 - alphas_bar))
        sigmas_cond = torch.sqrt(sigmas_cond)
        sigmas_cond[0] = 0.0

        self.models[model_id] = {
            'betas': betas,
            'alphas': alphas,
            'alphas_bar': alphas_bar,
            'sigmas': sigmas,
            'sigmas_cond': sigmas_cond,
            'params': params
        }

    # --------------------------------------------------------------------- #
    # Soft "safety ball" around start / goal poses
    # --------------------------------------------------------------------- #
    def _add_endpoint_soft_constraints(self, model_id, radius=0.05, window=10, weight=1.0):
        """
        Create and register two CostConstraint objects that keep the first
         <window> and last <window> timesteps within a ball of <radius>
        around the (potentially very wall‑adjacent) start and goal poses.
        """

        horizon = self.models[model_id]['params']['horizon']
        qs = [self.start_state_pos.clone(), self.goal_state_pos.clone()]
        traj_ranges = [(0, window - 1), (horizon - window, horizon - 1)]
        radii = [radius, radius]
        constraint = CostConstraint(
            self.robot,
            horizon,
            q_l=qs,
            traj_range_l=traj_ranges,
            radius_l=radii,
            is_soft=True,
            tensor_args={'device': self.device},
            priority_weight=weight
        )
        self.soft_constraints[model_id].append(constraint)

    def _preheat(self, use_cuda_graph):
        model_params = self.models[0]
        H = self.models[0]['params']['horizon']
        B = model_params['params']['n_samples']
        self._acc_actions = torch.zeros(B, H, 2, device=self.device)
        M_max, N_max, D = 1, 15, 2

        constraints = {0: []}
        constraints_t = pack_constraints_dict(
            constraints,
            M_max=M_max, N_max=N_max, D=D, H=H,
            device=self.device
        )

        def _fn(actions: torch.Tensor, constraints):
            return self._rollout_single_batch_new2(model_id=0, state_q=self.state_inits.q, us=actions,
                                                   constraints=constraints)

        from mdoc.utils.accelerators import RolloutAccelerator
        use_amp = True if self.device in ("cuda", "mps") else False
        amp_dtype = torch.bfloat16 if self.device == "cuda" else torch.float16

        self._acc_rollout = RolloutAccelerator(
            fn=_fn,
            example_args=(self._acc_actions, constraints_t,),
            use_amp=use_amp,
            amp_dtype=amp_dtype,
            cuda_graph=(self.device == "cuda" and use_cuda_graph),
            compile_mode="max-autotune",
            fullgraph=True,
        )

        try:
            self.build_sdf_texture(model_id=0, force_rebuild=False)
        except Exception:
            pass

        self._acc_rollout.prepare()

    def _project_to_free_space(self, q_ws, env_model, robot_radius, max_iter=1, grad_clip=5.0):
        """hard constraint"""
        B, H, _ = q_ws.shape
        q_flat = q_ws.reshape(-1, self.robot.q_dim).detach().clone()
        eps = 1e-6

        for _ in range(max_iter):
            # Compute SDF values
            sdf_val = env_model.compute_sdf(q_flat)  # [B*H]

            # Compute gradients using finite differences
            grad = torch.zeros_like(q_flat)
            for dim in range(q_flat.shape[-1]):
                q_perturbed = q_flat.clone()
                q_perturbed[:, dim] += eps
                sdf_perturbed = env_model.compute_sdf(q_perturbed)
                grad[:, dim] = (sdf_perturbed - sdf_val) / eps

            sdf_grad = grad.clamp(-grad_clip, grad_clip)
            # Compute clearance and collision mask
            clearance = sdf_val - (robot_radius + 1e-8)
            inside = clearance < 0

            if not inside.any():
                break

            # Project points out of obstacles
            grad_inside = sdf_grad[inside]
            grad_norm2 = (grad_inside ** 2).sum(dim=1, keepdim=True).clamp_min(1e-6)
            step = clearance[inside].unsqueeze(-1) * grad_inside / grad_norm2
            q_flat[inside] -= step

        return q_flat.view(B, H, self.robot.q_dim)

    def _project_traj_with_segment_check(
            self,
            q_ws,
            env_model,
            robot_radius,
            step=0.03,
            max_project_iter=1,
            grad_clip=5.0
    ):
        """
        """
        B, H0, _ = q_ws.shape
        flag = False
        q_ws = self._project_to_free_space(
            q_ws,
            env_model,
            robot_radius,
            max_iter=max_project_iter,
            grad_clip=grad_clip
        )

        new_trajs = []
        for b in range(B):
            pts = q_ws[b]  # [H,2]
            i = 0
            while i < pts.shape[0] - 1:
                p0, p1 = pts[i], pts[i + 1]
                seg_len = torch.norm(p1 - p0).item()
                n_samples = max(int(seg_len / step), 1)
                if n_samples == 1:
                    i += 1
                    continue
                alphas = torch.linspace(0, 1, n_samples + 1, device=pts.device)[1:]
                inter = p0 * (1 - alphas).unsqueeze(1) + p1 * alphas.unsqueeze(1)  # [n,2]
                sdf = env_model.compute_sdf(inter) - robot_radius
                if (sdf < 0).any():
                    flag = True
                    mid = (p0 + p1) * 0.5
                    pts = torch.cat([pts[:i + 1],
                                     mid.unsqueeze(0),
                                     pts[i + 1:]], dim=0)
                    pts[i + 1:i + 2] = self._project_to_free_space(
                        pts[i + 1:i + 2].unsqueeze(0), env_model, robot_radius
                    ).squeeze(0)
                else:
                    i += 1
            new_trajs.append(pts)

        max_H = max(t.shape[0] for t in new_trajs)
        padded = torch.full((B, max_H, 2), float('nan'), device=q_ws.device)
        for b, t in enumerate(new_trajs):
            padded[b, :t.shape[0]] = t

        return padded, flag

    def _segment_free_mask(self, q_ws, env_model, robot_radius, step=0.03):
        """
        Return Bool mask indicating whether each trajectory in the batch is
        entirely collision‑free when each poly‑line segment is densely sampled
        every <step> meters.
        """

        B, H, _ = q_ws.shape
        device = q_ws.device
        free_mask = torch.ones(B, dtype=torch.bool, device=device)
        for b in range(B):
            pts = q_ws[b]
            for i in range(H - 1):
                p0, p1 = pts[i], pts[i + 1]
                seg_len = torch.norm(p1 - p0).item()
                n_samples = max(int(seg_len / step), 1)
                if n_samples == 1:
                    continue
                alphas = torch.linspace(0.0, 1.0, n_samples + 1, device=device)[1:]
                inter = p0 * (1 - alphas).unsqueeze(1) + p1 * alphas.unsqueeze(1)
                sdf = env_model.compute_sdf(inter) - robot_radius
                if (sdf < 0).any():
                    free_mask[b] = False
                    break

        return free_mask

    @torch.no_grad()
    def sample_sdf_and_grad(self, q_ws, model_id=0):
        device = q_ws.device
        B, H, _ = q_ws.shape

        if device.type == "cuda":
            try:
                self.build_sdf_texture(model_id=model_id)
                if getattr(self, "_sdf_tex", None) is not None:
                    tex = self._sdf_tex
                    invW, invH = self._sdf_tex_invW, self._sdf_tex_invH
                    xbias, ybias = self._sdf_tex_xbias, self._sdf_tex_ybias

                    x = q_ws[..., 0].to(tex.device)
                    y = q_ws[..., 1].to(tex.device)

                    xn = x.mul(invW).add_(xbias)
                    yn = y.mul(invH).add_(ybias)

                    if (self._sdf_tex_grid is None) or (self._sdf_tex_grid.shape[:2] != (B, H)):
                        self._sdf_tex_grid = torch.empty(B, H, 1, 2, device=tex.device, dtype=tex.dtype)
                    grid = self._sdf_tex_grid
                    grid[..., 0].copy_(xn)
                    grid[..., 1].copy_(yn)

                    out = F.grid_sample(tex.expand(B, -1, -1, -1), grid, mode="bilinear", padding_mode="border",
                                        align_corners=True).squeeze(-1)  # [B,3,H,1]

                    sdf = out[:, 0, :].to(device=q_ws.device, dtype=q_ws.dtype)  # (B,H)
                    gx = out[:, 1, :].to(device=q_ws.device, dtype=q_ws.dtype)  # (B,H)
                    gy = out[:, 2, :].to(device=q_ws.device, dtype=q_ws.dtype)  # (B,H)
                    grad = torch.stack((gx, gy), dim=-1)  # (B,H,2)
                    return sdf, grad
            except Exception:
                pass

        env = self.env_models[model_id]
        q_flat = q_ws.reshape(-1, 2).to(self.device)
        sdf = env.compute_sdf(q_flat, reshape_shape=(B, H))  # (B,H)

        eps = 1e-3
        qx_p, qx_m = q_flat.clone(), q_flat.clone()
        qy_p, qy_m = q_flat.clone(), q_flat.clone()
        qx_p[:, 0] += eps
        qx_m[:, 0] -= eps
        qy_p[:, 1] += eps
        qy_m[:, 1] -= eps
        sdf_xp = env.compute_sdf(qx_p, reshape_shape=(B, H))
        sdf_xm = env.compute_sdf(qx_m, reshape_shape=(B, H))
        sdf_yp = env.compute_sdf(qy_p, reshape_shape=(B, H))
        sdf_ym = env.compute_sdf(qy_m, reshape_shape=(B, H))
        gx_all = (sdf_xp - sdf_xm) / (2 * eps)
        gy_all = (sdf_yp - sdf_ym) / (2 * eps)

        tau = 0.05
        band = (sdf.abs() < tau).to(q_ws.dtype)  # (B,H) -> float
        gx = gx_all * band
        gy = gy_all * band
        grad = torch.stack((gx, gy), dim=-1)  # (B,H,2)
        return sdf, grad

    @torch.no_grad()
    def build_sdf_texture(self, model_id=0, force_rebuild: bool = False, res: float = 0.01):
        if hasattr(self, "_sdf_tex") and (self._sdf_tex is not None) and (not force_rebuild):
            return

        env0 = self.env_models[model_id]
        try:
            x_min, x_max, y_min, y_max = env0.bounds
        except Exception:
            x_min, x_max, y_min, y_max = -2.0, 2.0, -2.0, 2.0

        Wm = max(2, int(round((x_max - x_min) / res)) + 1)
        Hm = max(2, int(round((y_max - y_min) / res)) + 1)

        xs = torch.linspace(x_min, x_max, Wm)
        ys = torch.linspace(y_min, y_max, Hm)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")  # [Hm,Wm]
        P = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)  # [Hm*Wm,2]

        sdf = env0.compute_sdf(P, reshape_shape=(Hm, Wm)).to(torch.float32)  # [Hm,Wm], float32 CPU/GPU
        gx = torch.zeros_like(sdf)
        gy = torch.zeros_like(sdf)
        gx[:, 1:-1] = (sdf[:, 2:] - sdf[:, :-2]) / (2 * res)
        gy[1:-1, :] = (sdf[2:, :] - sdf[:-2, :]) / (2 * res)
        gx[:, 0] = (sdf[:, 1] - sdf[:, 0]) / res
        gx[:, -1] = (sdf[:, -1] - sdf[:, -2]) / res
        gy[0, :] = (sdf[1, :] - sdf[0, :]) / res
        gy[-1, :] = (sdf[-1, :] - sdf[-2, :]) / res

        tex = torch.stack((sdf, gx, gy), dim=0).unsqueeze(0)
        prefer = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        tex = tex.to(device=prefer, dtype=torch.float32)

        self._sdf_tex = tex
        self._sdf_tex_meta = {"x0": x_min, "y0": y_min, "res": float(res), "H": int(Hm), "W": int(Wm)}
        return

    @torch.no_grad()
    def get_sdf_grid(self, model_id=0, res: float = 0.01):
        env0 = self.env_models[model_id]
        try:
            x_min, x_max, y_min, y_max = env0.bounds
        except Exception:
            x_min, x_max, y_min, y_max = -2.0, 2.0, -2.0, 2.0

        Wm = max(2, int(round((x_max - x_min) / res)) + 1)
        Hm = max(2, int(round((y_max - y_min) / res)) + 1)

        xs = torch.linspace(x_min, x_max, Wm)
        ys = torch.linspace(y_min, y_max, Hm)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
        P = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)
        sdf = env0.compute_sdf(P, reshape_shape=(Hm, Wm))
        return sdf, (x_min, x_max, y_min, y_max), res

    @torch.inference_mode()
    def _rollout_double_batch_new2(self, model_id, state_q, us, constraints):
        """
        Double integrator rollout.
        Dynamics:
            v_{t+1} = v_t + a_t * dt
            q_{t+1} = q_t + v_t * dt + 0.5 * a_t * dt^2
        Here `us` is interpreted as acceleration a
        """
        # -------------------- setup --------------------
        B, H, _ = us.shape
        device = us.device
        dt = self.robot.dt

        q0 = state_q.to(device).view(1, 1, 2).expand(B, 1, 2)  # (B,1,2)
        v0 = torch.zeros(B, 1, 2, device=device)  # (B,1,2)

        a_safe = us.clone()  # (B,H,2)
        a_flat = a_safe.reshape(B * H, 2)  # (B*H,2)

        has_tf = (self.transforms is not None) and (model_id in self.transforms)
        offset = self.transforms[model_id].to(device).view(1, 2) if has_tf else None

        q_init_exp = (
            q0.expand(B, H, 2).contiguous().view(B * H, 2).clone()
        )
        q_ws_flat = q_init_exp + offset if has_tf else q_init_exp  # (B*H,2)

        # -------------------- SDF & grad  --------------------
        sdf_flat, grad_flat = self.sample_sdf_and_grad(q_ws_flat.view(B * H, 1, 2), model_id=model_id)
        sdf_flat = sdf_flat.view(B * H)  # (B*H,)
        A = grad_flat.view(B * H, 2)  # (B*H,2)
        Anorm = A.norm(dim=-1)  # (B*H,)
        goodA = (Anorm > 1e-9)

        # -------------------- Discrete-time safety filter on acceleration (hard) + hybrid terms --------------------
        # h(q) = sdf(q) - (R + margin) >= 0
        # n = ∇h/||∇h||, hdot = n·v
        # Enforce next-step h_{t+1} >= 0  =>  n·a >= ( -h - (n·v) dt ) * 2 / dt^2
        R = mparams.robot_planar_disk_radius

        a_cum = torch.cumsum(a_safe, dim=1)  # (B,H,2)
        a_prefix = torch.nn.functional.pad(a_cum[:, :-1, :], (0, 0, 0, 1))  # (B,H,2), exclusive
        v_t = v0 + a_prefix * dt  # (B,H,2)
        v_flat = v_t.reshape(B * H, 2)  # (B*H,2)
        v_norm = v_flat.norm(dim=-1)  # (B*H,)
        a_max = getattr(self, "a_max", 2.0)
        margin = 0.02 + 0.20 * v_norm * dt + (v_norm * v_norm) / (2.0 * (a_max + 1e-6))  # (B*H,)
        h = sdf_flat - (R + margin)  # (B*H,)

        Anorm = A.norm(dim=-1)  # (B*H,)
        goodA = (Anorm > 1e-9)
        n = A / (Anorm.view(-1, 1) + 1e-12)  # (B*H,2)
        hdot = (n * v_flat).sum(dim=-1)  # (B*H,)

        dt2 = dt * dt
        b_dt = ((-h - hdot * dt) * 2.0) / (dt2 + 1e-12)  # (B*H,)

        k0_base, k1_base = 1.0, 2.0
        k0 = k0_base + 3.0 / (h.abs() + 1e-3)  # (B*H,)
        k1 = k1_base + 0.5 * v_norm  # (B*H,)
        b_hocbf = -k1 * hdot - k0 * h  # (B*H,)

        eta1 = 2.0
        b_vel = (-eta1 * h - hdot) / (dt + 1e-12)  # (B*H,)

        b = torch.maximum(b_dt, torch.maximum(b_hocbf, b_vel))  # (B*H,)
        tau = (R + margin) * 3.0
        need = goodA & (h < tau)

        a_nom = a_flat  # (B*H,2)
        lhs = (n * a_nom).sum(dim=-1)  # n·a_nom, (B*H,)
        n_norm2 = (n * n).sum(dim=-1, keepdim=True) + 1e-12  # ≈1
        viol = need & (lhs < b)

        delta = ((b - lhs).clamp_min(0.0) / n_norm2.squeeze(-1))  # (B*H,)
        a_flat = a_nom + (delta * viol.to(a_nom.dtype)).unsqueeze(-1) * n  # (B*H,2)

        lhs2 = (n * a_flat).sum(dim=-1)
        viol2 = need & (lhs2 < b)
        delta2 = ((b - lhs2).clamp_min(0.0) / n_norm2.squeeze(-1))
        a_flat = a_flat + (delta2 * viol2.to(a_nom.dtype)).unsqueeze(-1) * n

        step = (-h * 0.5).clamp_min(0.0).clamp_max(0.10)  # (B*H,)
        q_ws_flat_new = q_ws_flat + step.unsqueeze(-1) * n  # (B*H,2)
        mask_si = (h < 0.0).view(-1, 1)
        blended = torch.where(mask_si, q_ws_flat_new - offset, q_init_exp) if has_tf else torch.where(mask_si,
                                                                                                      q_ws_flat_new,
                                                                                                      q_init_exp)
        q_init_exp.copy_(blended)

        # reshape back
        a_safe = a_flat.view(B, H, 2)

        # -------------------- integrate: double integrator --------------------
        v_seq = v0 + torch.cumsum(a_safe * dt, dim=1)  # (B,H,2)
        # q_{t+1} = q_t + v_t*dt + 0.5*a_t*dt^2
        dq = v_seq * dt + 0.5 * a_safe * (dt * dt)  # (B,H,2)
        q_seq = q0 + torch.cumsum(dq, dim=1)  # (B,H,2)

        # -------------------- Vertex Constraints --------------------
        C_list = [constraints.C2[model_id]]
        R_list = [constraints.R[model_id]]
        TM_list = [constraints.TM[model_id]]

        if len(C_list) > 0:
            C_all = torch.cat([c.view(-1, 2) for c in C_list], dim=0)  # (N,2)
            R_all = torch.cat([r.view(-1) for r in R_list], dim=0).view(1, -1)  # (1,N)
            TM_all = torch.cat([tm.view(-1, H) for tm in TM_list], dim=0).T  # (H,N)

            q_ws_seq = q_seq + offset.view(1, 1, 2) if has_tf else q_seq  # (B,H,2)

            diff = q_ws_seq.unsqueeze(2) - C_all.view(1, 1, -1, 2)  # (B,H,N,2)
            dist = diff.norm(dim=-1).clamp_min_(1e-6)  # (B,H,N)
            A_s = diff / dist.unsqueeze(-1)  # (B,H,N,2)
            r_exp = (R_all + (R + 0.02))  # (1,N)
            h_s = dist - r_exp  # (B,H,N)

            active = TM_all.view(1, H, -1).expand(B, -1, -1)  # (B,H,N)
            near = (h_s < 0.08) & active

            dist_masked = torch.where(near, dist, torch.full_like(dist, 1e9))
            K = 2
            k = min(K, dist.shape[-1])
            topk = torch.topk(dist_masked, k=k, dim=-1, largest=False)
            idx = topk.indices  # (B,H,K)

            idx_e = idx.unsqueeze(-1).expand(-1, -1, -1, 2)  # (B,H,K,2)
            sel_A = torch.gather(A_s, 2, idx_e)  # (B,H,K,2)
            sel_h = torch.gather(h_s, 2, idx)  # (B,H,K)

            den_k = (sel_A * sel_A).sum(dim=-1).add_(1e-9)  # (B,H,K)
            n_s = sel_A / (sel_A.norm(dim=-1, keepdim=True) + 1e-12)  # (B,H,K,2)
            hdot_s = (n_s * v_seq.unsqueeze(2)).sum(dim=-1)  # (B,H,K)

            b_dt_s = ((-sel_h - hdot_s * dt) * 2.0) / (dt * dt + 1e-12)  # (B,H,K)
            k0_bh = k0.view(B, H).unsqueeze(-1)  # (B,H,1)
            k1_bh = k1.view(B, H).unsqueeze(-1)  # (B,H,1)
            b_hocbf = -k1_bh * hdot_s - k0_bh * sel_h  # (B,H,K)
            b_vel_s = (-eta1 * sel_h - hdot_s) / (dt + 1e-12)  # (B,H,K)

            sel_b = torch.maximum(b_dt_s, torch.maximum(b_hocbf, b_vel_s))  # (B,H,K)

            U = a_safe.clone()
            for kk in range(k):
                Ak = sel_A[:, :, kk, :]  # (B,H,2)
                bk = sel_b[:, :, kk]  # (B,H)
                lhsk = (Ak * U).sum(dim=-1)  # (B,H)
                viol = (lhsk < bk).to(U.dtype)
                num = (bk - lhsk).clamp_min(0.0)
                alpha_k = (num / den_k[:, :, kk]) * viol
                U = U + alpha_k.unsqueeze(-1) * Ak
            a_safe = U

        v_seq = v0 + torch.cumsum(a_safe * dt, dim=1)
        dq = v_seq * dt + 0.5 * a_safe * (dt * dt)
        q_seq = q0 + torch.cumsum(dq, dim=1)

        # -------------------- Costs --------------------
        q_goal = self.goal_state_pos.to(device)
        qg = q_goal.view(1, 1, 2)
        dist_to_goal = (q_seq - qg).norm(dim=2)  # (B,H)

        ctrl = (a_safe * a_safe).sum(dim=2)

        runtime_cost = torch.zeros(B, H, device=device)
        runtime_cost.add_(ctrl, alpha=1.0)
        runtime_cost.add_(dist_to_goal, alpha=5)

        t_star = max(1, H - 1)
        t_idx = torch.arange(H, device=device).view(1, H)
        d0 = (state_q.to(device) - q_goal).norm().view(1, 1)
        d_hat = d0 * torch.clamp(1.0 - t_idx.float() / t_star, min=0.0)
        runtime_cost.add_((dist_to_goal - d_hat).pow(2), alpha=8.0)

        s = (t_idx.float() / t_star).clamp(max=1.0).view(1, H, 1)  # (1,H,1)
        v_bar = (d0.view(1, 1, 1) / (t_star * dt + 1e-6)) * (1.0 * s * (1.0 - s))
        v_norm = v_seq.norm(dim=-1, keepdim=True)  # (B,H,1)
        v_err = v_norm - v_bar
        runtime_cost.add_((v_err.squeeze(-1) ** 2), alpha=1.0)

        r_goal = 0.10
        inside_goal_early = (dist_to_goal <= r_goal) & (t_idx < t_star)
        tau_e = ((t_star - t_idx).clamp(min=0).float() / H)
        runtime_cost.add_(tau_e * inside_goal_early.float(), alpha=0.5)

        q_ws = q_seq + offset.view(1, 1, 2) if has_tf else q_seq
        sdf_seq, _ = self.sample_sdf_and_grad(q_ws)
        sdf_seq = sdf_seq - (R + 0.02)
        pen = (-sdf_seq).clamp(min=0.0)
        runtime_cost.add_(pen * pen, alpha=5000.0)

        free_mask = torch.ones(B, dtype=torch.bool, device=device)

        w_T = getattr(self, "w_T_base", 20)
        terminal_cost = w_T * dist_to_goal[:, -1]
        cost_sum = runtime_cost.sum(dim=1) + terminal_cost

        return cost_sum, q_seq, free_mask

    @torch.inference_mode()
    def _rollout_single_batch_new2(self, model_id, state_q, us, constraints):
        # -------------------- setup --------------------
        B, H, _ = us.shape
        device = us.device
        dt = self.robot.dt

        q_init = state_q.to(device).view(1, 1, 2).expand(B, 1, 2)
        us_flat = us.reshape(B * H, 2)  # (B*H,2)

        has_tf = (self.transforms is not None) and (model_id in self.transforms)
        offset = self.transforms[model_id].to(device).view(1, 2) if has_tf else None

        q_init_exp = (
            q_init.expand(B, H, 2)
            .contiguous()
            .view(B * H, 2)
            .clone()
        )
        q_ws_flat = q_init_exp + offset if has_tf else q_init_exp  # (B*H,2)

        # -------------------- SDF & grad  --------------------
        sdf_flat, grad_flat = self.sample_sdf_and_grad(q_ws_flat.view(B * H, 1, 2), model_id=model_id)
        sdf_flat = sdf_flat.view(B * H)  # (B*H,)
        A = grad_flat.view(B * H, 2)  # (B*H,2)

        # -------------------- CBF --------------------
        R = mparams.robot_planar_disk_radius
        margin = 0.01
        tau = 0.05
        eta = 0.8

        h = sdf_flat - (R + margin)  # (B*H,)
        u_flat = us_flat.clone()

        Anorm = A.norm(dim=-1)  # (B*H,)
        goodA = (Anorm > 1e-9)  # bool

        b = - (eta / dt) * h  # (B*H,)
        lhs = (A * u_flat).sum(dim=-1)  # (B*H,)
        den = (A * A).sum(dim=-1).add_(1e-9)  # (B*H,)

        need = goodA & (h < tau)  # bool
        alpha = ((b - lhs) / den).clamp_min(0.0) * need.to(u_flat.dtype)
        u_flat = u_flat + alpha.unsqueeze(-1) * A

        lhs2 = (A * u_flat).sum(dim=-1)  # (B*H,)
        bad = goodA & (lhs2 < b)  # bool
        alpha2 = ((b - lhs2) / den).clamp_min(0.0) * bad.to(u_flat.dtype)
        u_flat = u_flat + alpha2.unsqueeze(-1) * A

        n = A / (Anorm.view(-1, 1) + 1e-12)  # (B*H,2)
        step = (-h * 0.5).clamp_min(0.0).clamp_max(0.10)  # (B*H,)
        q_ws_flat_new = q_ws_flat + step.unsqueeze(-1) * n  # (B*H,2)
        mask_si = (h < 0.0).view(-1, 1)  # (B*H,1)

        blended = torch.where(mask_si, q_ws_flat_new - offset, q_init_exp) if has_tf else torch.where(mask_si,
                                                                                                      q_ws_flat_new,
                                                                                                      q_init_exp)
        q_init_exp.copy_(blended)

        # -------------------- integrate --------------------
        u_safe = u_flat.view(B, H, 2)
        q_seq = q_init + torch.cumsum(u_safe * dt, dim=1)  # (B,H,2)

        # -------------------- Vertex Constraints --------------------
        C_list = [constraints.C2[model_id]]  # (N_max, 2)
        R_list = [constraints.R[model_id]]  # (N_max,)
        TM_list = [constraints.TM[model_id]]

        # C_list, R_list, TM_list = [], [], []
        # # if hasattr(self, "soft_constraints") and (model_id in self.soft_constraints):
        # if model_id in constraints:
        #     for sph in (constraints[model_id] or []):
        #         if not sph:
        #             continue
        #         C_cc = sph.qs.to(device)[:, :2]
        #         R_cc = sph.radii.to(device)
        #         TR = sph.traj_ranges.to(device).long()  # (n,2)
        #         n_cc = C_cc.shape[0]
        #         if n_cc == 0:
        #             continue
        #         t_idx_full = torch.arange(H, device=device).view(1, H).expand(n_cc, -1)  # (n,H)
        #         TM_cc = (t_idx_full >= TR[:, 0:1]) & (t_idx_full < TR[:, 1:2])  # (n,H)
        #         C_list.append(C_cc)
        #         R_list.append(R_cc)
        #         TM_list.append(TM_cc)

        if len(C_list) > 0:
            C_all = torch.cat([c.view(-1, 2) for c in C_list], dim=0)  # (N,2)
            R_all = torch.cat([r.view(-1) for r in R_list], dim=0).view(1, -1)  # (1,N)
            TM_all = torch.cat([tm.view(-1, H) for tm in TM_list], dim=0).T  # (H,N)

            q_ws_seq = q_seq + offset.view(1, 1, 2) if has_tf else q_seq  # (B,H,2)

            diff = q_ws_seq.unsqueeze(2) - C_all.view(1, 1, -1, 2)  # (B,H,N,2)
            dist = diff.norm(dim=-1).clamp_min_(1e-6)  # (B,H,N)
            A_s = diff / dist.unsqueeze(-1)  # (B,H,N,2)
            r_exp = (R_all + (R + 0.02))  # (1,N)
            h_s = dist - r_exp  # (B,H,N)

            active = TM_all.view(1, H, -1).expand(B, -1, -1)  # (B,H,N)
            near = (h_s < 0.08) & active  # (B,H,N)

            dist_masked = torch.where(near, dist, torch.full_like(dist, 1e9))
            K = 2
            k = min(K, dist.shape[-1])
            topk = torch.topk(dist_masked, k=k, dim=-1, largest=False)
            idx = topk.indices  # (B,H,K)

            idx_e = idx.unsqueeze(-1).expand(-1, -1, -1, 2)  # (B,H,K,2)
            sel_A = torch.gather(A_s, 2, idx_e)  # (B,H,K,2)
            sel_h = torch.gather(h_s, 2, idx)  # (B,H,K)

            base_beta = 0.05
            sign = 1.0 if (getattr(self, "agent_id", 0) % 2 == 0) else -1.0
            beta = sign * base_beta

            U = u_safe.clone()
            den_k = (sel_A * sel_A).sum(dim=-1).add_(1e-9)  # (B,H,K)
            sel_b = -(eta / dt) * sel_h + beta

            for kk in range(k):
                Ak = sel_A[:, :, kk, :]  # (B,H,2)
                bk = sel_b[:, :, kk]  # (B,H)
                lhsk = (Ak * U).sum(dim=-1)  # (B,H)
                viol = (lhsk < bk).to(U.dtype)  # (B,H) float mask
                num = (bk - lhsk).clamp_min(0.0)  # (B,H)
                alpha_k = (num / den_k[:, :, kk]) * viol  # (B,H)
                U = U + alpha_k.unsqueeze(-1) * Ak
            u_safe = U

        # u_safe = project_controls_smooth(u_safe, mu=1e-3, iters=8)
        q_seq = q_init + torch.cumsum(u_safe * dt, dim=1)

        q_goal = self.goal_state_pos.to(device)
        qg = q_goal.view(1, 1, 2)
        diff_g = q_seq - qg
        dist_to_goal = diff_g.norm(dim=2)  # (B,H)
        ctrl = (u_safe * u_safe).sum(dim=2)  # (B,H)

        runtime_cost = torch.zeros(B, H, device=device)
        runtime_cost.add_(ctrl, alpha=1.0)
        runtime_cost.add_(dist_to_goal, alpha=5)

        t_star = max(1, H - 1)
        t_idx = torch.arange(H, device=device).view(1, H)
        d0 = (state_q.to(device) - q_goal).norm().view(1, 1)
        d_hat = d0 * torch.clamp(1.0 - t_idx.float() / t_star, min=0.0)
        runtime_cost.add_((dist_to_goal - d_hat).pow(2), alpha=1.0)

        s = (t_idx.float() / t_star).clamp(max=1.0).view(1, H, 1)  # (1,H,1)
        v_bar = (d0.view(1, 1, 1) / (t_star * dt + 1e-6)) * (1.0 * s * (1.0 - s))
        u_norm = u_safe.norm(dim=-1, keepdim=True)  # (B,H,1)
        v_err = u_norm - v_bar
        runtime_cost.add_(v_err.squeeze(-1) * v_err.squeeze(-1), alpha=1.0)

        r_goal = 0.10
        inside_goal_early = (dist_to_goal <= r_goal) & (t_idx < t_star)
        tau_e = ((t_star - t_idx).clamp(min=0).float() / H)
        runtime_cost.add_(tau_e * inside_goal_early.float(), alpha=0.5)

        q_ws = q_seq + offset.view(1, 1, 2) if has_tf else q_seq
        sdf_seq, _ = self.sample_sdf_and_grad(q_ws)
        sdf_seq = sdf_seq - (R + 0.02)
        pen = (-sdf_seq).clamp(min=0.0)
        runtime_cost.add_(pen * pen, alpha=5000.0)

        free_mask = torch.ones(B, dtype=torch.bool, device=device)

        w_T = getattr(self, "w_T_base", 8)
        terminal_cost = w_T * dist_to_goal[:, -1]
        cost_sum = runtime_cost.sum(dim=1) + terminal_cost

        return cost_sum, q_seq, free_mask

    def _rollout_single_batch_new1(self, model_id, state_q, us):
        env = self.env_models[model_id]
        B, H, _ = us.shape
        device = us.device
        dt = self.robot.dt

        q_init = state_q.view(1, 1, 2).to(device).expand(B, 1, 2)
        us_flat = us.reshape(B * H, 2)  # (B*H, 2)
        q_init_expanded = q_init.expand(B, H, 2).reshape(B * H, 2)  # (B*H, 2)

        q_ws_flat = q_init_expanded
        if self.transforms is not None and model_id in self.transforms:
            q_ws_flat = q_init_expanded + self.transforms[model_id].to(device).view(1, -1)

        sdf_flat, grad_flat = self.sample_sdf_and_grad(q_ws_flat.view(B * H, 1, 2), model_id=model_id)
        sdf_flat = sdf_flat.view(B * H)
        grad_flat = grad_flat.view(B * H, 2)

        R = mparams.robot_planar_disk_radius
        margin = 0.01
        tau = 0.05
        eta = 0.8

        h = sdf_flat - (R + margin)
        A = grad_flat
        u_flat = us_flat.clone()

        goodA = (A.norm(dim=-1) > 1e-9)
        need = (h < tau) & goodA

        b = - (eta / dt) * h
        lhs = (A * u_flat).sum(dim=-1)
        den = (A.pow(2).sum(dim=-1) + 1e-9)

        alpha = torch.zeros(B * H, device=device)
        alpha[need] = torch.clamp((b[need] - lhs[need]) / den[need], min=0.0)
        u_flat = u_flat + alpha.unsqueeze(-1) * A

        lhs2 = (A * u_flat).sum(dim=-1)
        bad = goodA & (lhs2 < b)
        alpha2 = torch.zeros(B * H, device=device)
        alpha2[bad] = torch.clamp(((b - lhs2) / den)[bad], min=0.0)
        u_flat = u_flat + alpha2.unsqueeze(-1) * A

        still_in = h < 0.0
        if still_in.any():
            step = (0.5 * (-h[still_in])).clamp_max(0.10)
            n = torch.nn.functional.normalize(A[still_in], dim=-1)
            q_ws_flat_new = q_ws_flat.clone()
            q_ws_flat_new[still_in] = q_ws_flat_new[still_in] + step.unsqueeze(-1) * n
            if self.transforms is not None and model_id in self.transforms:
                q_init_expanded[still_in] = q_ws_flat_new[still_in] - self.transforms[model_id].to(device).view(1, -1)
            else:
                q_init_expanded[still_in] = q_ws_flat_new[still_in]

        u_safe = u_flat.reshape(B, H, 2)
        q_seq = q_init + torch.cumsum(u_safe * dt, dim=1)

        # ========= Vertex Constraints ======== #
        C_list, R_list, TM_list = [], [], []
        if hasattr(self, "soft_constraints") and (model_id in self.soft_constraints):
            for sph in (self.soft_constraints[model_id] or []):
                C_cc = sph.qs.to(device)[:, :2]
                R_cc = sph.radii.to(device)
                TR = sph.traj_ranges.to(device).long()  # (n,2)
                n_cc = C_cc.shape[0]
                t_idx_full = torch.arange(H, device=device).view(1, H).expand(n_cc, -1)  # (n,H)
                TM_cc = (t_idx_full >= TR[:, 0:1]) & (t_idx_full < TR[:, 1:2])  # (n,H)

                if n_cc > 0:
                    C_list.append(C_cc)  # (n,2)
                    R_list.append(R_cc)  # (n,)
                    TM_list.append(TM_cc)  # (n,H)

        if len(C_list) > 0:
            C_all = torch.cat([c.view(-1, 2) for c in C_list], dim=0)  # (N,2)
            R_all = torch.cat([r.view(-1) for r in R_list], dim=0).view(1, -1)  # (1,N)
            TM_cat = torch.cat([tm if tm.dim() == 2 else tm.view(-1, H) for tm in TM_list], dim=0)  # (N,H)
            TM_all = TM_cat.T  # (H,N)

            q_ws_seq = q_seq
            if self.transforms is not None and model_id in self.transforms:
                offset = self.transforms[model_id].to(device)
                q_ws_seq = q_seq + offset.view(1, 1, -1)  # (B,H,2)

            # （B,H,N）
            diff = q_ws_seq.unsqueeze(2) - C_all.view(1, 1, -1, 2)  # (B,H,N,2)
            dist = diff.norm(dim=-1).clamp_min(1e-6)  # (B,H,N)
            A_s = diff / dist.unsqueeze(-1)  # (B,H,N,2)
            r_exp = (R_all + (R + 0.02))  # (1,N)
            h_s = dist - r_exp  # (B,H,N)

            # time window & Narrow
            active = TM_all.view(1, H, -1).expand(B, -1, -1)  # (B,H,N)
            tau_s = 0.08
            near = (h_s < tau_s) & active  # (B,H,N)

            if near.any():
                # Top-K
                K = 2
                dist_masked = torch.where(near, dist, torch.full_like(dist, 1e9))
                k = min(K, dist_masked.shape[-1])
                topk = torch.topk(dist_masked, k=k, dim=-1, largest=False)
                idx = topk.indices  # (B,H,K)

                sel_A = torch.gather(A_s, 2, idx.unsqueeze(-1).expand(-1, -1, -1, 2))  # (B,H,K,2)
                sel_h = torch.gather(h_s, 2, idx)  # (B,H,K)

                # ---- C-5：优先级偏置 β（偶数先行/奇数让行），可叠加 CostConstraint 的 priority_weight ----
                base_beta = 0.05
                sign = +1.0
                if hasattr(self, "agent_id"):
                    sign = +1.0 if (self.agent_id % 2 == 0) else -1.0
                scale_pw = 1.0

                # if regist CostConstraint，可用其 priority_weight 微调（可选）
                # if (hasattr(self, "soft_constraints") and model_id in self.vertex_cost_constraints):
                #     try:
                #         pw = float(self.soft_constraints[model_id].priority_weight)
                #         scale_pw = max(0.5, min(2.0, pw))
                #     except Exception:
                #         pass
                beta = sign * base_beta * scale_pw
                sel_b = - (eta / dt) * sel_h + beta  # (B,H,K)
                U = u_safe.clone()  # (B,H,2)

                for kk in range(k):
                    Ak = sel_A[:, :, kk, :]  # (B,H,2)
                    bk = sel_b[:, :, kk]  # (B,H)
                    lhs_k = (Ak * U).sum(dim=-1)  # (B,H)
                    viol = lhs_k < bk
                    if viol.any():
                        den_k = (Ak.pow(2).sum(dim=-1) + 1e-9)
                        alpha_k = torch.zeros_like(lhs_k)
                        alpha_k[viol] = (bk[viol] - lhs_k[viol]) / den_k[viol]
                        U = U + alpha_k.unsqueeze(-1) * Ak
                u_safe = U  #

        # smooth gurantee as hard constraints
        mu = 1e-3
        iters = 8
        u_proj = project_controls_smooth(u_safe, mu=mu, iters=iters)
        u_safe = u_proj

        q_seq = q_init + torch.cumsum(u_safe * dt, dim=1)

        # ===== costs: runtime + terminal =====
        q_goal = self.goal_state_pos.to(device)
        dist_to_goal = (q_seq - q_goal.view(1, 1, -1)).norm(dim=2)  # (B,H)
        ctrl = (u_safe * u_safe).sum(dim=2)  # (B,H)

        runtime_cost = torch.zeros(B, H, device=device)
        runtime_cost += 1.0 * ctrl
        runtime_cost += 5 * dist_to_goal

        t_star = max(1, int(0.95 * H))
        t_idx = torch.arange(H, device=device).view(1, H)
        d0 = (state_q.to(device) - q_goal).norm().view(1, 1)
        d_hat = d0 * torch.clamp(1.0 - t_idx.float() / t_star, min=0.0)
        runtime_cost += 1 * (dist_to_goal - d_hat).pow(2)

        s = (t_idx.float() / t_star).clamp(max=1.0).view(1, H, 1)
        v_bar = (d0.view(1, 1, 1) / (t_star * dt + 1e-6)) * (1.0 * s * (1.0 - s))
        u_norm = u_safe.norm(dim=-1, keepdim=True)
        runtime_cost += 0.5 * (u_norm - v_bar).pow(2).squeeze(-1)

        r_goal = 0.10
        inside_goal_early = (dist_to_goal <= r_goal) & (t_idx < t_star)
        runtime_cost += 0.5 * ((t_star - t_idx).clamp(min=0).float() / H) * inside_goal_early.float()

        q_ws = q_seq
        if self.transforms is not None and model_id in self.transforms:
            offset = self.transforms[model_id].to(device)
            q_ws = q_seq + offset.view(1, 1, -1)

        sdf_seq, _ = self.sample_sdf_and_grad(q_ws)
        sdf_seq = sdf_seq - (mparams.robot_planar_disk_radius + 0.02)
        pen = (-sdf_seq).clamp(min=0.0)
        runtime_cost += 5000.0 * pen.pow(2)

        # free_mask_wall = self._segment_free_mask(
        #     q_ws, env_model=env, robot_radius=mparams.robot_planar_disk_radius, step=0.03
        # )
        # free_mask = free_mask_wall

        free_mask = torch.ones(B, dtype=torch.bool, device=device)

        # terminal
        w_T = getattr(self, "w_T_base", 8)
        terminal_cost = w_T * dist_to_goal[:, -1]
        cost_sum = runtime_cost.sum(dim=1) + terminal_cost

        return cost_sum, q_seq, free_mask

    @torch.no_grad()
    def _rollout_single_batch_new2_ultrafast(self, model_id, state_q, us, use_amp: bool = False):
        """
        Ultra-fast rollout that matches _rollout_single_batch_new2's math & weights:
          - keeps original parameters/weights (distance=5.0, v_bar factor=4.0, TopK=2, 2-step CBF)
          - avoids redundant allocs via static/scratch buffers (per (B,H,device,dtype,model_id))
          - fixes broadcasting; no unnecessary CPU<->GPU sync (.item)
          - SDF path penalty squared in fp32 for stability (behavior preserved)
        Returns: cost_sum: (B,), q_seq: (B,H,2), free_mask: (B,)
        """
        device, dtype = us.device, us.dtype
        B, H, _ = us.shape
        dt = float(self.robot.dt)

        # ===== constants (aligned with original) =====
        R = float(mparams.robot_planar_disk_radius)
        margin = 0.01
        tau = 0.05
        eta = 0.8
        eps = 1e-9
        huge = 1e9
        tau_s = 0.08
        r_goal = 0.10
        w_T = getattr(self, "w_T_base", 8)  # keep type/behavior
        t_star = max(1, int(0.9 * H))
        sign = +1.0 if (getattr(self, "agent_id", 0) % 2 == 0) else -1.0
        beta = sign * 0.05  # same as original base_beta * scale_pw(=1)

        # ===== one-time caches =====
        if not hasattr(self, "_rollout_static"):   self._rollout_static = {}
        if not hasattr(self, "_rollout_tf_cache"): self._rollout_tf_cache = {}
        if not hasattr(self, "_rollout_scratch"):  self._rollout_scratch = {}

        s_key = (B, H, device, model_id)
        st = self._rollout_static.get(s_key)
        if st is None:
            t_idx = torch.arange(H, device=device)
            st = {
                "t_idx_i": t_idx.view(1, H),
                "t_idx_f": t_idx.float().view(1, H),
                "ones_B_bool": torch.ones(B, dtype=torch.bool, device=device),
            }
            self._rollout_static[s_key] = st
        t_idx_i, t_idx_f, ones_B_bool = st["t_idx_i"], st["t_idx_f"], st["ones_B_bool"]

        # transform offset cache: (1,1,2)
        tf_key = (model_id, device)
        offset = self._rollout_tf_cache.get(tf_key, None)
        if (self.transforms is not None) and (model_id in self.transforms) and (offset is None):
            self._rollout_tf_cache[tf_key] = self.transforms[model_id].to(device).contiguous().view(1, 1, -1)
            offset = self._rollout_tf_cache[tf_key]

        # scratch buffers
        sc_key = (B, H, device, dtype)
        sc = self._rollout_scratch.get(sc_key)
        if sc is None:
            sc = {
                # flat / batched
                "BH2": torch.empty((B * H, 2), device=device, dtype=dtype),
                "BH": torch.empty((B * H,), device=device, dtype=dtype),
                "BH_": torch.empty((B, H), device=device, dtype=dtype),
                "BH1": torch.empty((B, H, 1), device=device, dtype=dtype),
                "BH1b": torch.empty((B, H, 1), device=device, dtype=dtype),
                # reused vectors
                "A2": torch.empty((B * H,), device=device, dtype=dtype),
                "lhs": torch.empty((B * H,), device=device, dtype=dtype),
                "alpha": torch.empty((B * H,), device=device, dtype=dtype),
                # vertex constraints
                "min_vals": torch.empty((B, H), device=device, dtype=dtype),
                "min_idx": torch.empty((B, H), device=device, dtype=torch.long),
                "lhs_k": torch.empty((B, H), device=device, dtype=dtype),
                "den_k": torch.empty((B, H), device=device, dtype=dtype),
                "alpha_k": torch.empty((B, H), device=device, dtype=dtype),
                # cost stage
                "dist_to_goal": torch.empty((B, H), device=device, dtype=dtype),
                "ctrl": torch.empty((B, H), device=device, dtype=dtype),
                "runtime_cost": torch.empty((B, H), device=device, dtype=dtype),
                # bool masks
                "bool_BH": torch.empty((B * H,), device=device, dtype=torch.bool),
            }
            self._rollout_scratch[sc_key] = sc

        # helpers
        def add_offset(q):  # q(...,2)
            return q if offset is None else (q + offset.view(*([1] * (q.dim() - 1)), 2))

        def rm_offset(q):
            return q if offset is None else (q - offset.view(*([1] * (q.dim() - 1)), 2))

        # init states / views
        q0 = state_q.to(device=device, dtype=dtype).view(1, 1, 2).expand(B, 1, 2)
        us_flat = us.reshape(B * H, 2)

        autocast_ctx = (torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                        if (use_amp and torch.cuda.is_available()) else contextlib.nullcontext())

        with autocast_ctx:
            # ---------- 1) CBF-like projection (2-step, same as original) ----------
            q_init_exp = q0.expand(B, H, 2).reshape(B * H, 2)  # (B*H,2)
            q_ws_flat = add_offset(q_init_exp.view(B * H, 1, 2))  # (B*H,1,2)

            sdf_flat, grad_flat = self.sample_sdf_and_grad(q_ws_flat, model_id=model_id)
            sdf_flat = sdf_flat.view(B * H)  # (B*H,)
            A = grad_flat.view(B * H, 2)  # (B*H,2)

            sc["A2"].copy_((A * A).sum(-1)).add_(eps)  # ||A||^2 + eps(=1e-9)
            h = sc["BH"].copy_(sdf_flat).add_(-(R + margin))  # (B*H,)
            u_flat = sc["BH2"].copy_(us_flat)  # clone-like but from scratch
            goodA = sc["bool_BH"]
            goodA.copy_((A * A).sum(-1) > 1e-18)  # ||A||>~1e-9 (squared)
            need = sc["bool_BH"].clone().copy_(h < tau)
            need &= goodA

            b = sc["BH"].copy_(h).mul_(-eta / dt)  # b = -(eta/dt)*h
            sc["lhs"].copy_((A * u_flat).sum(-1))  # A·u
            den = sc["A2"]  # reuse

            sc["alpha"].zero_()
            if need.any():
                num = (b - sc["lhs"])[need]
                sc["alpha"][need].copy_((num / den[need]).clamp(min=0.0))

            u_flat.add_(sc["alpha"].unsqueeze(-1) * A)

            # second correction (original has this)
            lhs2 = (A * u_flat).sum(-1)
            bad = goodA & (lhs2 < b)
            if bad.any():
                alpha2 = torch.zeros_like(lhs2)
                alpha2[bad] = ((b - lhs2)[bad] / den[bad]).clamp(min=0.0)
                u_flat = u_flat + alpha2.unsqueeze(-1) * A

            # push out if still inside
            still_in = (h < 0.0) & goodA
            if still_in.any():
                Ai = A[still_in]
                inv = torch.rsqrt((Ai * Ai).sum(-1, keepdim=True) + eps)
                n = Ai * inv
                step = (0.5 * (-h[still_in])).clamp_max(0.10)
                q_ws_corr = sc["BH2"].view(B * H, 2)
                q_ws_corr.copy_(q_ws_flat.view(B * H, 2))
                q_ws_corr[still_in].add_(step.unsqueeze(-1) * n)
                q_init_exp[still_in].copy_(rm_offset(q_ws_corr[still_in]))

            u_safe = u_flat.view(B, H, 2)
            q_seq = q0 + torch.cumsum(u_safe * dt, dim=1)

            # ---------- 2) Vertex soft-constraints (TopK=2, same as original) ----------
            if hasattr(self, "soft_constraints") and (model_id in self.soft_constraints):
                sph_list = self.soft_constraints[model_id] or []
                if len(sph_list) > 0:
                    C_list, R_list, TM_list = [], [], []
                    for sph in sph_list:
                        Cc = sph.qs.to(device=device, dtype=dtype)[:, :2].contiguous()
                        if Cc.numel() == 0: continue
                        C_list.append(Cc)
                        R_list.append(sph.radii.to(device=device, dtype=dtype).view(-1))
                        TR = sph.traj_ranges.to(device, non_blocking=True).long()
                        t_full = t_idx_i.expand(Cc.shape[0], -1)  # (n,H)
                        TM_list.append((t_full >= TR[:, :1]) & (t_full < TR[:, 1:2]))  # (n,H)

                    if len(C_list) > 0:
                        C_all = torch.cat(C_list, dim=0).contiguous()  # (N,2)
                        R_all = torch.cat(R_list, dim=0).contiguous().view(1, -1)  # (1,N)
                        TM_all = torch.cat([tm.view(-1, H) for tm in TM_list], 0).t().contiguous()  # (H,N)

                        q_ws_seq = add_offset(q_seq)  # (B,H,2)
                        diff = q_ws_seq.unsqueeze(2) - C_all.view(1, 1, -1, 2)  # (B,H,N,2)
                        dist = diff.norm(dim=-1).clamp_(min=1e-6)  # (B,H,N)
                        A_s = diff / dist.unsqueeze(-1)  # (B,H,N,2)
                        h_s = dist - (R_all + (R + 0.02))  # (B,H,N)

                        active = TM_all.view(1, H, -1).expand(B, -1, -1)  # (B,H,N)
                        near = (h_s < tau_s) & active  # (B,H,N)

                        if near.any():
                            # TopK by dist (as original)
                            dist_masked = torch.where(near, dist, torch.full_like(dist, huge))
                            K = min(2, dist_masked.shape[-1])
                            topk = torch.topk(dist_masked, k=K, dim=-1, largest=False)
                            idx = topk.indices  # (B,H,K)

                            sel_A = torch.gather(A_s, 2, idx.unsqueeze(-1).expand(-1, -1, -1, 2))  # (B,H,K,2)
                            sel_h = torch.gather(h_s, 2, idx)  # (B,H,K)

                            U = u_safe.clone()
                            for kk in range(K):
                                Ak = sel_A[:, :, kk, :]  # (B,H,2)
                                hk = sel_h[:, :, kk]  # (B,H)
                                bk = -(eta / dt) * hk + beta  # (B,H)
                                lhs_k = (Ak * U).sum(-1)  # (B,H)
                                viol = lhs_k < bk
                                if viol.any():
                                    den_k = (Ak * Ak).sum(-1) + eps
                                    alpha_k = torch.zeros_like(lhs_k)
                                    alpha_k[viol] = ((bk - lhs_k)[viol] / den_k[viol]).clamp(min=0.0)
                                    U = U + alpha_k.unsqueeze(-1) * Ak
                            u_safe = U

                        # re-integrate once
                        q_seq = q0 + torch.cumsum(u_safe * dt, dim=1)

            # ---------- 3) Costs (weights exactly match original) ----------
            q_goal = self.goal_state_pos.to(device=device, dtype=dtype).view(1, 1, 2)
            sc["dist_to_goal"].copy_((q_seq - q_goal).norm(dim=2))  # (B,H)
            sc["ctrl"].copy_((u_safe * u_safe).sum(dim=2))  # (B,H)
            #
            rc = sc["runtime_cost"]
            rc.copy_(sc["ctrl"] * 1.0).add_(sc["dist_to_goal"] * 5.0)  # 1*ctrl + 5*dist  (original)
            #
            # # d_hat term
            w = torch.clamp(1.0 - t_idx_f / t_star, min=0.0)  # (1,H)
            d0 = (state_q.to(device=device, dtype=dtype).view(1, 1, 2) - q_goal).norm(dim=2)  # (1,1)
            d_hat = sc["BH_"]
            d_hat.copy_(w.expand_as(d_hat)).mul_(d0)  # (B,H)
            rc.add_(1.0 * (sc["dist_to_goal"] - d_hat).pow(2))  # 1.0 *

            # v_bar & speed regularization (factor 1.0)
            s = (t_idx_f / t_star).clamp(max=1.0)  # (1,H)
            v_bar = sc["BH1"]
            if v_bar.shape != (B, H, 1):
                sc["BH1"] = torch.empty((B, H, 1), device=device, dtype=dtype)
                v_bar = sc["BH1"]
            v_bar.copy_(d0.view(1, 1, 1).expand(B, H, 1))
            v_bar.div_(t_star * dt + 1e-6)
            v_bar.mul_(1.0 * s.view(1, H, 1) * (1.0 - s).view(1, H, 1))

            u_norm = sc["BH1b"]
            if u_norm.shape != (B, H, 1):
                sc["BH1b"] = torch.empty((B, H, 1), device=device, dtype=dtype)
                u_norm = sc["BH1b"]
            u_norm.copy_(u_safe.norm(dim=-1, keepdim=True))
            rc.add_(0.5 * (u_norm.squeeze(-1) - v_bar.squeeze(-1)).pow(2))

            # early-arrival penalty (unchanged)
            inside_goal_early = (sc["dist_to_goal"] <= r_goal) & (t_idx_i < t_star)
            rc.add_(0.5 * ((t_star - t_idx_i).clamp(min=0).float() / H) * inside_goal_early.float())

            # SDF path penalty (fp32-square, same weight 5000)
            # q_ws = q_seq (+ offset like original)
            q_ws = add_offset(q_seq)
            sdf_seq, _ = self.sample_sdf_and_grad(q_ws)
            sdf_seq = sdf_seq - (R + 0.02)
            pen = sc["BH_"].copy_(sdf_seq).neg_().clamp_(min=0.0)  # = relu(-(sdf))
            pen32 = pen.to(torch.float32)
            rc.add_((5000.0 * (pen32 * pen32)).to(rc.dtype))

            terminal_cost = w_T * sc["dist_to_goal"][:, -1]
            cost_sum = rc.sum(dim=1) + terminal_cost

            free_mask = ones_B_bool

        return cost_sum, q_seq, free_mask

    def _rollout_single_batch(self, model_id, state_init, us):
        """
        Args
        ----
        Single-integrator batch rollout .
        us : (B, H, 2)  ‑‑
        Returns
        -------
        cost   : (B,)
        q_seq  : (B, H, 2)
        """
        env = self.env_models[model_id]
        B, H, _ = us.shape
        dt = self.robot.dt

        dq = us * dt  # (B,H,2)
        q0 = state_init.q.unsqueeze(0)  # (1,2)
        q_seq = torch.cumsum(dq, dim=1) + q0  # (B,H,2)

        q_goal = self.goal_state_pos.to(self.device)
        pos_err = (self.goal_state_pos - q_seq).norm(dim=2)  # (B,H)
        ctrl = (us * us).sum(dim=2)  # (B,H)
        cost = ctrl + 500 * pos_err  # (B,H)

        # 1) Progress cost: d_t follows linear schedule to t_star
        t_star = max(1, int(0.98 * H))  # plan to arrive near 98% horizon
        t_idx = torch.arange(H, device=self.device).view(1, H)  # (1,H)
        d_t = (q_seq - q_goal.view(1, 1, -1)).norm(dim=2)  # (B,H)
        d0 = (state_init.q.to(self.device) - q_goal).norm().view(1, 1)  # (1,1)
        d_hat = d0 * torch.clamp(1.0 - t_idx.float() / t_star, min=0.0)  # (1,H)
        w_prog = 0
        J_prog = w_prog * (d_t - d_hat).pow(2)  # (B,H)
        cost = cost + J_prog

        if env is not None:
            q_ws = q_seq  # default : already global
            if self.transforms is not None and model_id in self.transforms:
                offset = self.transforms[model_id].to(q_seq.device)  # (2,)
                q_ws = q_seq + offset.view(1, 1, -1)  # broadcast to (B,H,2)

            # q_ws = self._project_to_free_space(q_ws, env, mparams.robot_planar_disk_radius, max_iter=3, grad_clip=2)
            sdf = env.compute_sdf(q_ws.reshape(-1, self.robot.q_dim)).view(B, H)  # (B,H)
            sdf -= (mparams.robot_planar_disk_radius + 1e-8)
            pen = (-sdf).clamp(min=0)
            # cost += 5000 * pen.pow(2)
            free_mask = (sdf.min(dim=1).values >= 0)
        else:
            free_mask = torch.ones(B, dtype=torch.bool, device=us.device)

        return cost.sum(dim=1), q_seq, free_mask

    @torch.no_grad()
    def _reverse_diffusion_step(
            self,
            model_id: int,
            i: int,
            traj_bars_i: torch.Tensor,
            free_ratio_target: float = 1,
            enable_k_selection=True,
    ):
        """
        Reverse process for single model
        """
        model_params = self.models[model_id]
        env_model = self.env_models[model_id]

        sigma_i = model_params['sigmas'][i].item()
        # temp_step = np.interp(sigma_i, [model_params['sigmas'][-1].item(), model_params['sigmas'][0].item()], [model_params['params']['temp_sample'] * 0.5, model_params['params']['temp_sample']])
        temp_step = model_params['params']['temp_sample']
        n_samples = model_params['params']['n_samples']

        # recover the trajectory
        if traj_bars_i.shape[0] != n_samples:
            traj_bars_i = traj_bars_i[:1].repeat(n_samples, 1, 1)

        if i > 0:
            traj_i = traj_bars_i * torch.sqrt(model_params['alphas_bar'][i])
        else:
            traj_i = traj_bars_i.clone()

        #  Monte Carlo Sample from q_i velocity
        eps_u = torch.randn((n_samples, model_params['params']['horizon'], self.robot.q_dim * 2),
                            device=self.device)
        traj_0s = torch.clamp(eps_u * model_params['sigmas'][i] + traj_bars_i, -1, 1)
        actions = traj_0s[..., self.robot.q_dim:]

        if self.compile:
            costs, q_seq, free_mask = self._acc_rollout.run(actions, self.constraints_ro)
        else:
            costs, q_seq, free_mask = self._rollout_single_batch_new2_ultrafast(model_id, self.state_inits.q, actions)

        traj_0s[..., :self.robot.q_dim] = q_seq

        if not free_mask.any():
            raise RuntimeError(f"No collision-free sample after resamples")

        cost_mean = costs.mean()
        cost_std = costs.std().clamp(min=1e-4)
        logp0 = -(costs - cost_mean) / cost_std / temp_step
        logp0[~free_mask] = -torch.inf

        # soft_constraint_grad = torch.zeros(traj_i.shape[0], traj_i.shape[1], self.robot.q_dim, device=traj_i.device)
        # if len(self.soft_constraints[model_id]) > 0:
        #     lambda_c = 1e-5
        #     k = 10
        #     # Reshape for proper broadcasting
        #     traj_pos = traj_i[..., :self.robot.q_dim].unsqueeze(2)
        #     if self._N > k and enable_k_selection:
        #         # select k
        #         B, H, q_dim = traj_pos.shape[0], traj_pos.shape[1], self.robot.q_dim
        #         traj_flat = traj_pos.reshape(-1, q_dim)  # [(B·H), q_dim]
        #         dist2_full = torch.cdist(traj_flat, self._centers_flat)  # [(B·H), N]
        #
        #         dmin, _ = dist2_full.min(dim=0)
        #         _, idx = torch.topk(dmin, k=k, largest=False)
        #         centers = self._centers_flat[idx].view(1, 1, k, -1)
        #         radii = self._radii_flat[idx].view(1, 1, k)
        #         t0 = self._t0[idx]
        #         t1 = self._t1[idx]
        #     else:
        #         centers = self._centers_flat.view(1, 1, self._N, -1)
        #         radii = self._radii_flat.view(1, 1, self._N)
        #         t0 = self._t0
        #         t1 = self._t1
        #
        #     diff = traj_pos - centers  # [B, H, N, q_dim]
        #     dist = diff.norm(dim=-1, keepdim=True)  # [B, H, N, 1]
        #
        #     idx = torch.arange(traj_pos.shape[1], device=self.device).view(1, traj_pos.shape[1], 1)  # [1, H, 1]
        #     mask_time = ((idx >= t0) & (idx <= t1)).unsqueeze(-1)
        #     mask_dist = (dist < radii.unsqueeze(-1))  # [B, H, N, 1]
        #     mask = mask_time & mask_dist  # [B, H, N, 1]
        #
        #     grad = -diff / (dist + 1e-6) * mask  # [B, H, N, q_dim]
        #     soft_constraint_grad = grad.sum(dim=2) / lambda_c  # [B, H, q_dim]

        # use demonstration data
        if self.enable_demo:
            xref_logpds = torch.tensor([env_model.eval_xref_logpd(q) for q in qs], device=self.device)
            xref_logpds = xref_logpds - xref_logpds.max()
            logpdemo = -(xref_logpds + env_model.rew_xref - cost_mean) / cost_std / model_params['params'][
                'temp_sample']
            demo_mask = logpdemo > logp0
            logp0 = torch.where(demo_mask, logpdemo, logp0)
            logp0 = (logp0 - logp0.mean()) / logp0.std() / model_params['params']['temp_sample']

        # weighted trajectory
        weights = torch.nn.functional.softmax(logp0, dim=0)
        traj_bar = torch.einsum("n,nij->ij", weights, traj_0s)

        # calculate score function
        score = 1 / (1.0 - model_params['alphas_bar'][i]) * (
                - traj_i + torch.sqrt(model_params['alphas_bar'][i]) * traj_bar)

        # apply smooth as soft constraints
        # score = apply_smooth_guidance_to_score(
        #     score,
        #     trajs_pos=traj_bar[..., :self.robot.q_dim],
        #     step=i,
        #     total_steps=self.models[model_id]['params']['n_diffusion_step'],
        #     dt=self.robot.dt,
        #     w_start=0.8,
        #     w_end=0.15,
        #     modes=("lap", "acc"),
        #     weights=(100, 1000)
        # )

        # expanded_grad = torch.zeros_like(score)
        # expanded_grad[..., :self.robot.q_dim] = soft_constraint_grad
        # score += expanded_grad  # add soft constraints

        # traj_i_m1[..., :self.robot.q_dim] = q_seq

        proj_weight = 0.8
        traj_i_m1 = (1.0 / torch.sqrt(model_params['alphas'][i]) *
                     (traj_i + (1.0 - model_params['alphas_bar'][i]) * score)
                     ) / torch.sqrt(model_params['alphas_bar'][i - 1])

        traj_proj = traj_i_m1.clone()
        traj_proj[..., :self.robot.q_dim] = q_seq
        traj_i_m1 = (1.0 - proj_weight) * traj_i_m1 + proj_weight * traj_proj

        best_idx = torch.where(free_mask)[0][costs[free_mask].argmin()]
        best_traj_sample = traj_0s[best_idx]

        if self.transforms is not None and model_id in self.transforms:
            best_traj_sample[..., :self.robot.q_dim] += self.transforms[model_id]

        return traj_i_m1, best_traj_sample, costs.mean().item()

    @torch.no_grad()
    def run_inference(
            self,
            contexts: List[dict] = None,
            hard_conds: Dict[int, dict] = None,
            cross_conds: Dict[Tuple[int, int], Tuple[int, int]] = None,
            model_ids: List[int] = None,
            n_samples: int = 100,
            batch_size: int = 32,
            return_chain: bool = False,
            soft_constraints=[],
            **diffusion_kwargs):
        """
        :param contexts:
        :param hard_conds:
        :param cross_conds: Keys are tuples of model indices and values are tuples of trajectory timesteps.
                            For example, {(0, 1): {0, 64}} means that the value of the state 0 in the trajectory
                            of model 0 needs to be the same as the value of the state 64 in the trajectory of model 1.
        :param n_samples:
        :param return_chain:
        :param diffusion_kwargs:
        :return:
            if return_chain is True，return the diffusion chain
            else return final optimized trajectory
        """
        contexts = deepcopy(contexts)
        hard_conds = deepcopy(hard_conds)
        cross_conds = deepcopy(cross_conds)
        if contexts is None:
            contexts = [None] * len(self.models)
        for m, c_dict in hard_conds.items():
            for k, v in c_dict.items():
                # k is the key of the condition, usually state index in the trajectory
                hard_conds[m][k] = einops.repeat(v, 'd -> b d', b=n_samples)

        results = {}
        chains = {}

        for model_id in self.models.keys():
            traj_n = torch.zeros((n_samples, self.models[model_id]['params']['horizon'], self.robot.q_dim * 2),
                                 device=self.device)
            B, H, _ = traj_n.shape
            # cold start soft constraints (acceleration)
            self.soft_constraints[model_id] = soft_constraints
            if len(self.soft_constraints[model_id]) > 0:
                self._prepare_soft_constraints(model_id, H, traj_n.device)

            self.constraints_ro = pack_constraints_dict(self.soft_constraints, M_max=1, N_max=15, D=2, H=64,
                                                        device=self.device)
            # Reverse
            traj_i = traj_n
            model_chain = []
            costs = []
            best_cost = float('inf')

            for i in tqdm(range(self.models[model_id]['params']['n_diffusion_step'] - 1, -1, -1),
                          desc=f"Diffusing model {model_id}"):
                traj_i, sample_i, cost_i = self._reverse_diffusion_step(model_id, i, traj_i)
                if return_chain:
                    model_chain.append(traj_i)
                costs.append(cost_i)
                if cost_i < best_cost:
                    best_cost, best_traj = cost_i, sample_i

            if return_chain:
                chains[model_id] = torch.stack(model_chain)

        if return_chain:
            return chains

        return results
