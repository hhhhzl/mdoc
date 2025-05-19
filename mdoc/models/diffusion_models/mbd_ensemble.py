import torch
import torch.nn as nn
from typing import Dict, Any, List, Tuple
import numpy as np
from tqdm import tqdm
from copy import deepcopy
from mdoc.config import MDOCParams as mparams
from mdoc.models.diffusion_models.sample_functions import (
    extract,
    apply_hard_conditioning,
    apply_cross_conditioning,
)
from .diffusion_model_base import make_timesteps
import einops
from torch_robotics.robots.robot_base import RobotState
from concurrent.futures import ThreadPoolExecutor


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
            'horizon': mparams.horizon,
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
        self.transforms = transforms
        self.context_model = context_model

        # for cache (ECBS calculate collision cost)
        self._centers = None
        self._radii2 = None
        self._time_mask = None

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

    def _project_traj_with_segment_check(self, q_ws, env_model, robot_radius, step=0.03, max_project_iter=1, grad_clip=5.0):
        """
        """
        B, H0, _ = q_ws.shape
        flag = False
        q_ws = self._project_to_free_space(q_ws, env_model, robot_radius, max_iter=max_project_iter, grad_clip=grad_clip)

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

    def _resample_fixed_horizon(self, pts, target_H):
        """"""
        B, Hv, _ = pts.shape
        seg_lens = torch.norm(pts[:, 1:] - pts[:, :-1], dim=-1)  # (B, Hv-1)
        s = torch.cumsum(torch.nn.functional.pad(seg_lens, (1, 0)), dim=1)  # (B, Hv)
        s_norm = s / s[:, -1:].clamp_min(1e-6)  # 0~1

        t = torch.linspace(0, 1, target_H, device=pts.device)  # (H,)
        t = t.expand(B, -1)  # (B, H)
        idx = torch.searchsorted(s_norm, t, right=True).clamp(max=Hv - 1)

        idx0 = (idx - 1).clamp(min=0)
        idx1 = idx
        s0, s1 = s_norm.gather(1, idx0), s_norm.gather(1, idx1)
        p0, p1 = pts.gather(1, idx0.unsqueeze(-1).expand(-1, -1, 2)), \
                 pts.gather(1, idx1.unsqueeze(-1).expand(-1, -1, 2))
        w = ((t - s0) / (s1 - s0 + 1e-8)).unsqueeze(-1)  # (B,H,1)
        return (1 - w) * p0 + w * p1

    @torch.no_grad()
    def _reverse_diffusion_step(
            self,
            model_id: int,
            i: int,
            traj_bars_i: torch.Tensor,
            max_resample: int = 500,
            free_ratio_target: float = 1,
            enable_k_selection=True,
    ):
        """
        Reverse process for single model
        """
        model_params = self.models[model_id]
        env_model = self.env_models[model_id]

        sigma_i = model_params['sigmas'][i].item()
        temp_step = np.interp(sigma_i, [model_params['sigmas'][-1].item(), model_params['sigmas'][0].item()], [model_params['params']['temp_sample'] * 0.5, model_params['params']['temp_sample']])
        # temp_step = model_params['params']['temp_sample']
        n_samples = model_params['params']['n_samples']
        attempt = 0

        while attempt < max_resample:
            # recover the trajectory
            if traj_bars_i.shape[0] != n_samples:
                traj_bars_i = traj_bars_i[:1].repeat(n_samples, 1, 1)
            # traj_i = traj_bars_i.clone()
            traj_i = traj_bars_i * torch.sqrt(model_params['alphas_bar'][i])

            #  Monte Carlo Sample from q_i velocity
            eps_u = torch.randn((n_samples, model_params['params']['horizon'], self.robot.q_dim * 2), device=self.device)
            traj_0s = torch.clamp(eps_u * model_params['sigmas'][i] + traj_bars_i, -1, 1)

            # evaluate samples
            qs = []
            actions = traj_0s[..., self.robot.q_dim:]
            costs, q_seq, free_mask = self._rollout_us_batch(model_id, self.state_inits, actions)
            # print("sdf min per traj:", env_model.compute_sdf(q_seq.reshape(-1, 2)).view(n_samples, model_params['params']['horizon']).min(dim=1).values)
            q_seq = self._project_to_free_space(
                q_seq,
                env_model,
                robot_radius=mparams.robot_planar_disk_radius,
                grad_clip=5.0,
            )
            # q_seq = self._resample_fixed_horizon(
            #     q_seq,
            #     target_H=model_params['params']['horizon']
            # )
            traj_0s[..., :self.robot.q_dim] = q_seq

            sdf_proj = env_model.compute_sdf(q_seq.reshape(-1, 2)).view(n_samples, model_params['params']['horizon'])
            sdf_proj -= (mparams.robot_planar_disk_radius + 1e-8)
            free_mask = (sdf_proj.min(dim=1).values >= 0)
            free_ratio = free_mask.float().mean().item()
            print(f"step {i:03d} | σ={sigma_i:.4f} | free={free_ratio:.2f} | "f"samples={n_samples}")

            if i >= 2 and free_ratio > 0:
                break
            elif i < 2 and free_ratio == free_ratio_target:
                break

            attempt += 1

        if not free_mask.any():
            raise RuntimeError(f"No collision-free sample after {attempt} resamples")

        cost_mean = costs.mean()
        cost_std = costs.std().clamp(min=1e-4)
        logp0 = -(costs - cost_mean) / cost_std / temp_step
        logp0[~free_mask] = -torch.inf

        soft_constraint_grad = torch.zeros(
            traj_i.shape[0], traj_i.shape[1], self.robot.q_dim,  # [B, H, q_dim]
            device=traj_i.device
        )
        if len(self.soft_constraints[model_id]) > 0:
            lambda_c = 1e-5
            k = 30
            import time
            start = time.time()
            # Reshape for proper broadcasting
            traj_pos = traj_i[..., :self.robot.q_dim].unsqueeze(2)
            if self._N > k and enable_k_selection:
                # select k
                B, H, q_dim = traj_pos.shape[0], traj_pos.shape[1], self.robot.q_dim
                traj_flat = traj_pos.reshape(-1, q_dim)  # [(B·H), q_dim]
                dist2_full = torch.cdist(traj_flat, self._centers_flat)  # [(B·H), N]

                dmin, _ = dist2_full.min(dim=0)
                _, idx = torch.topk(dmin, k=k, largest=False)
                centers = self._centers_flat[idx].view(1, 1, k, -1)
                radii = self._radii_flat[idx].view(1, 1, k)
                t0 = self._t0[idx]
                t1 = self._t1[idx]
            else:
                centers = self._centers_flat.view(1, 1, self._N, -1)
                radii = self._radii_flat.view(1, 1, self._N)
                t0 = self._t0
                t1 = self._t1

            diff = traj_pos - centers  # [B, H, N, q_dim]
            dist = diff.norm(dim=-1, keepdim=True)  # [B, H, N, 1]

            idx = torch.arange(traj_pos.shape[1], device=self.device).view(1, traj_pos.shape[1], 1)  # [1, H, 1]
            mask_time = ((idx >= t0) & (idx <= t1)).unsqueeze(-1)
            mask_dist = (dist < radii.unsqueeze(-1))  # [B, H, N, 1]
            mask = mask_time & mask_dist  # [B, H, N, 1]

            grad = -diff / (dist + 1e-6) * mask  # [B, H, N, q_dim]
            soft_constraint_grad = grad.sum(dim=2) / lambda_c  # [B, H, q_dim]
            # penetration = radii.unsqueeze(-1) - dist  # (B,H,N,1)
            # mask_pen = penetration > 0
            #
            # grad = diff * penetration * mask_pen / (dist ** 3)
            #
            # weights = self._weights  # broadcast (1,1,N,1)
            # grad = grad * weights * mask_time
            #
            # gamma = 1.15
            # grad = grad * (gamma ** i)
            # soft_constraint_grad = grad.sum(dim=2) / lambda_c
            # print("=============================> solving constraints for: ", time.time() - start, centers.shape)

            # all spheres
            # Calculate differences and distances
            # centers = self._centers_flat.view(1, 1, self._N, -1)
            # radii = self._radii_flat.view(1, 1, self._N)
            # t0 = self._t0
            # t1 = self._t1
            # diff = traj_pos - centers  # [B, H, N, q_dim]
            # dist = diff.norm(dim=-1, keepdim=True)  # [B, H, N, 1]
            #
            # idx = torch.arange(H, device=self.device).view(1, H, 1)  # [1, H, 1]
            # mask_time = ((idx >= t0) & (idx <= t1)).unsqueeze(-1)
            # mask_dist = (dist < radii.unsqueeze(-1))  # [B, H, N, 1]
            # mask = mask_time & mask_dist  # [B, H, N, 1]
            #
            # grad = -diff / (dist + 1e-6) * mask  # [B, H, N, q_dim]
            # soft_constraint_grad = grad.sum(dim=2) / lambda_c  # [B, H, q_dim]
            # print("=============================> solving constraints for: ", time.time() - start, grad.shape[2])

        # use demonstration data
        if self.enable_demo:
            xref_logpds = torch.tensor([env_model.eval_xref_logpd(q) for q in qs], device=self.device)
            xref_logpds = xref_logpds - xref_logpds.max()
            logpdemo = -(xref_logpds + env_model.rew_xref - cost_mean) / cost_std / model_params['params']['temp_sample']
            demo_mask = logpdemo > logp0
            logp0 = torch.where(demo_mask, logpdemo, logp0)
            logp0 = (logp0 - logp0.mean()) / logp0.std() / model_params['params']['temp_sample']

        # weighted trajectory
        weights = torch.nn.functional.softmax(logp0, dim=0)
        traj_bar = torch.einsum("n,nij->ij", weights, traj_0s)

        # calculate score function
        score = 1 / (1.0 - model_params['alphas_bar'][i]) * (
                - traj_i + torch.sqrt(model_params['alphas_bar'][i]) * traj_bar)

        expanded_grad = torch.zeros_like(score)
        expanded_grad[..., :self.robot.q_dim] = soft_constraint_grad
        score += expanded_grad  # add soft constraints

        if env_model is not None:
            q_ws = traj_i  # (B,H,2)
            R = mparams.robot_planar_disk_radius
            margin = 9e-4
            lambda_obs = 200
            sdf_grad = env_model.compute_sdf(q_ws.reshape(-1, self.robot.q_dim)).view(traj_i.shape[0], traj_i.shape[1], self.robot.q_dim)  # (B,H,2)
            penetration = (margin - (sdf_grad - (R + 1e-8))).clamp(min=0.0)  # (B,H,2)
            score[..., :self.robot.q_dim] += (lambda_obs / margin) * penetration * sdf_grad  # (B,H,2)

        traj_i_m1 = (1 / torch.sqrt(model_params['alphas'][i]) * (
                traj_i + (1.0 - model_params['alphas_bar'][i]) * score)) / torch.sqrt(
            model_params['alphas_bar'][i - 1])

        best_idx = torch.where(free_mask)[0][costs[free_mask].argmin()]
        best_traj_sample = traj_0s[best_idx]

        if self.transforms is not None and model_id in self.transforms:
            best_traj_sample[..., :self.robot.q_dim] += self.transforms[model_id]

        # return traj_i_m1, best_traj_sample, costs[best_idx].item()
        # return traj_i_m1, costs[free_mask].mean().item()
        return traj_i_m1, best_traj_sample, costs.mean().item()

    def _rollout_us(self, model_id: int, state: torch.tensor, us: torch.tensor):
        """
        Args:
            model_id: Function that takes (state, action) and returns new state
            state: Initial state object (must have reward and pipeline_state attributes)
            us: Tensor of actions shape [T, action_dim]
        Returns:
            rews: Tensor of rewards shape [T]
            pipeline_states: List of pipeline states
        """
        # us = torch.ones_like(us) * torch.tensor([0.1, 0.0], device=self.device)
        env_model = self.env_models[model_id]

        costs = []
        pipeline_states = []
        if torch.is_tensor(us):
            us = [u for u in us]

        for u in us:
            state = self.robot.step(state, torch.from_numpy(u), env_model)
            pos_error = torch.norm(self.goal_state_pos - state.q)
            costs.append(state.collision_cost + state.control_cost + 2.0 * pos_error)
            # costs.append(state.cost)
            pipeline_states.append(state.pipeline_state)

        if torch.is_tensor(costs[0]):
            costs = torch.stack(costs)
        else:
            costs = torch.tensor(costs)

        return costs, pipeline_states

    def _rollout_us_batch(self, model_id, state_init, us):
        """
        Args
        ----
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

        pos_err = (self.goal_state_pos - q_seq).norm(dim=2)  # (B,H)
        ctrl = (us * us).sum(dim=2)  # (B,H)
        cost = 1 * ctrl + 20 * pos_err  # (B,H)

        # w_goal = 15
        # w_stop = 20
        # cost[:, -1] += w_goal * pos_err[:, -1]
        # cost[:, -1] += w_stop * ctrl[:, -1]

        # if self.soft_constraint[model_id]:
        #     for constraint in self.soft_constraint[model_id]:
        #         c = constraint.q_l[0].to(q_seq.device)
        #         radius = constraint.radius_l[0]
        #         lambda_c = constraint.lambda_c
        #
        #         distances = torch.norm(q_seq - c, dim=2)
        #         violations = torch.relu(radius - distances)
        #         cost += lambda_c * violations

        if env is not None:
            q_ws = q_seq  # default : already global
            if self.transforms is not None and model_id in self.transforms:
                offset = self.transforms[model_id].to(q_seq.device)  # (2,)
                q_ws = q_seq + offset.view(1, 1, -1)  # broadcast to (B,H,2)

            q_ws = self._project_to_free_space(q_ws, env, mparams.robot_planar_disk_radius, max_iter = 1, grad_clip = 5.0)
            sdf = env.compute_sdf(q_ws.reshape(-1, self.robot.q_dim)).view(B, H)  # (B,H)
            sdf -= (mparams.robot_planar_disk_radius + 1e-8)
            pen = (-sdf).clamp(min=0)
            # cost += 200 * pen.pow(2)
            free_mask = (sdf.min(dim=1).values >= 0)
        else:
            free_mask = torch.ones(B, dtype=torch.bool, device=us.device)

        return cost.sum(dim=1), q_seq, free_mask

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

            # Reverse
            traj_i = traj_n
            model_chain = []
            costs = []

            best_traj = None
            best_cost = float('inf')

            for i in tqdm(range(self.models[model_id]['params']['n_diffusion_step'] - 1, 0, -1), desc=f"Diffusing model {model_id}"):
                traj_i, sample_i, cost_i = self._reverse_diffusion_step(model_id, i, traj_i)
                if return_chain:
                    model_chain.append(traj_i)
                costs.append(cost_i)
                if cost_i < best_cost:
                    best_cost, best_traj = cost_i, sample_i

            if return_chain:
                chains[model_id] = torch.stack(model_chain)

            # evaluate final cost
            final_traj = chains[model_id][-1, ...]
            final_actions = self.robot.get_velocity(final_traj).cpu().numpy() if self.device != 'cpu' else self.robot.get_velocity(traj_i).numpy()
            costs_final, q = self._rollout_us(model_id, self.state_inits, final_actions)
            results[model_id] = {
                'trajectory': final_traj,
                'cost': costs_final.mean().item(),
                'cost_history': costs
            }
            sdf_final = self.env_models[model_id].compute_sdf(final_traj[..., :self.robot.q_dim])
            penetration = (-sdf_final).clip(min=0)
            free_mask = (sdf_final.min(dim=1).values >= 0)
            success = (penetration.max() == 0)
            free_ratio = free_mask.float().mean().item()
            print("Success from direct MBD: ", success, free_ratio)
            chains[model_id] = final_traj.unsqueeze(0)

        if return_chain:
            return chains

        return results

    # ===================================================================== #
    @torch.no_grad()
    def run_local_inference(
            self,
            seed_trajectories: Dict[int, torch.Tensor],
            n_noising_steps: Dict[int, int],
            n_denoising_steps: Dict[int, int],
            contexts: Dict[int, dict] = None,
            hard_conds: Dict[int, dict] = None,
            cross_conds: Dict[Tuple[int, int], dict] = None,
            n_samples: int = 1,
            return_chain: bool = False,
            **diffusion_kwargs):
        """

        Args:
            seed_trajectories: {model_id: seed_trajectories}
            n_noising_steps: {model_id: add noising steps} (None for skip)
            n_denoising_steps: {model_id: denoising steps}
            contexts: {model_id: context}
            hard_conds: {model_id: hard_cond}
            cross_conds: {(src_id, tgt_id): cross_conds}
            n_samples:
            return_chain:
            diffusion_kwargs:

        Returns:
            if return_chain=True return {model_id: diffusion_chain}
            else {model_id: optimized_trajectory}
        """
        results = {}
        chains = {}

        contexts = deepcopy(contexts)
        hard_conds = deepcopy(hard_conds)
        cross_conds = deepcopy(cross_conds)
        if contexts is None:
            contexts = [None] * len(self.models)
        for m, c_dict in hard_conds.items():
            for k, v in c_dict.items():
                # k is the key of the condition, usually state index in the trajectory
                hard_conds[m][k] = einops.repeat(v, 'd -> b d', b=n_samples)

        # Noise the given seed trajectory for n_noising_steps.
        noised_trajectories = {}
        for model_id, traj in seed_trajectories.items():
            if n_noising_steps.get(model_id) is not None:
                # add noising
                t = make_timesteps(traj.shape[0], n_noising_steps[model_id], traj.device)
                noised_traj = self._q_sample(model_id, traj, t)
                print(f"Model {model_id} - Applied {n_noising_steps[model_id]} noising steps")
            else:
                noised_traj = traj.clone()
                print(f"Model {model_id} - Using original seed (no noising)")

            noised_trajectories[model_id] = noised_traj

        # apply cross conditioning
        noised_trajectories = apply_cross_conditioning(noised_trajectories, cross_conds, self.transforms)

        # denosing for each model
        for model_id in seed_trajectories.keys():
            model_params = self.models[model_id]

            traj_i = noised_trajectories[model_id]
            model_chain = [traj_i] if return_chain else None

            # reverse processing
            start_step = model_params['params']['n_diffusion_step'] - 1
            end_step = max(0, start_step - n_denoising_steps[model_id] + 1)
            for i in tqdm(range(start_step, end_step - 1, -1), desc=f"Model {model_id} denoising"):
                # apply hard conds
                if model_id in hard_conds:
                    traj_i = apply_hard_conditioning(traj_i, hard_conds[model_id])

                # single reverse steps
                traj_i, _ = self._reverse_diffusion_step(
                    model_id=model_id,
                    i=i,
                    traj_bars_i=traj_i,
                    n_samples=n_samples,
                    context=contexts.get(model_id)
                )

                if return_chain:
                    model_chain.append(traj_i)

            # apply constraints
            if model_id in hard_conds:
                traj_i = apply_hard_conditioning(traj_i, hard_conds[model_id])

            # evaluate
            final_actions = traj_i.cpu().numpy() if self.device != 'cpu' else traj_i.numpy()
            costs_final, _ = self._rollout(model_id, self.state_inits[model_id], final_actions)

            results[model_id] = {
                'trajectory': traj_i,
                'cost': np.mean(costs_final),
                'denoising_steps': n_denoising_steps[model_id]
            }

            if return_chain:
                chains[model_id] = torch.stack(model_chain)

        if return_chain:
            return chains
        return results

    def _q_sample(self, model_id: int, x_start: torch.Tensor, t: torch.Tensor):
        """add noising"""
        model = self.models[model_id]
        noise = torch.randn_like(x_start)
        sqrt_alphas_bar = extract(model['alphas_bar'], t, x_start.shape)
        sqrt_one_minus_alphas_bar = extract(model['sigmas'], t, x_start.shape)
        return sqrt_alphas_bar * x_start + sqrt_one_minus_alphas_bar * noise

    def forward(self, *args, **kwargs):
        pass
