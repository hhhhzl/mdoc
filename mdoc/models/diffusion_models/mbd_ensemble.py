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
            'betaT': 0.01
        }

        self.recommended_params = {
            "temp_sample": 0.1,
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

    @torch.no_grad()
    def _reverse_diffusion_step(self, model_id: int, i: int, traj_bars_i: torch.Tensor):
        """
        Reverse process for single model
        """
        model_params = self.models[model_id]
        env_model = self.env_models[model_id]

        # recover the trajectory
        traj_i = traj_bars_i * torch.sqrt(model_params['alphas_bar'][i])

        #  Monte Carlo Sample from q_i
        # velocity
        eps_u = torch.randn(
            (model_params['params']['n_samples'], model_params['params']['horizon'], self.robot.q_dim * 2),
            device=self.device)
        traj_0s = eps_u * model_params['sigmas'][i] + traj_bars_i
        traj_0s = torch.clamp(traj_0s, -1.0, 1.0)

        # evaluate samples
        qs = []
        actions = traj_0s[..., self.robot.q_dim:]
        costs, q_seq, free_mask = self._rollout_us_batch(model_id, self.state_inits, actions)
        traj_0s[..., :self.robot.q_dim] = q_seq
        cost_mean = costs.mean()
        cost_std = costs.std().clamp(min=1e-4)
        logp0 = -(costs - cost_mean) / cost_std / model_params['params']['temp_sample']

        soft_constraint_grad = torch.zeros(
            traj_i.shape[0], traj_i.shape[1], self.robot.q_dim,  # [B, H, q_dim]
            device=traj_i.device
        )
        if len(self.soft_constraints[model_id]) > 0:
            constraint_centers = torch.stack(
                [constraint.qs[0] for constraint in self.soft_constraints[model_id]])  # [N_constraints, q_dim]
            constraint_radii = torch.tensor([constraint.radii[0] for constraint in self.soft_constraints[model_id]],
                                            device=traj_i.device)  # [N_constraints]
            lambda_c = torch.tensor(0.0001, device=traj_i.device)

            # Reshape for proper broadcasting
            constraint_centers = constraint_centers.view(1, 1, -1, self.robot.q_dim)  # [1, 1, N_constraints, q_dim]
            constraint_radii = constraint_radii.view(1, 1, -1)  # [1, 1, N_constraints]

            # Calculate differences and distances
            traj_pos = traj_i[..., :self.robot.q_dim].unsqueeze(2)  # [B, H, 1, q_dim]
            diff = traj_pos - constraint_centers  # [B, H, N_constraints, q_dim]
            distances = torch.norm(diff, dim=-1, keepdim=True)  # [B, H, N_constraints, 1]

            # Compute violations mask
            violations = torch.relu(constraint_radii - distances.squeeze(-1))  # [B, H, N_constraints]
            mask = (violations > 0).unsqueeze(-1)  # [B, H, N_constraints, 1]

            # Compute and apply gradient
            grad = -diff / (distances + 1e-6)  # [B, H, N_constraints, q_dim]
            grad = grad * mask  # [B, H, N_constraints, q_dim]

            # Sum across constraints and time steps
            soft_constraint_grad = grad.sum(dim=2) / lambda_c  # [B, H, q_dim]

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

        expanded_grad = torch.zeros_like(score)
        expanded_grad[..., :self.robot.q_dim] = soft_constraint_grad
        score += expanded_grad  # add soft constraints

        traj_i_m1 = (1 / torch.sqrt(model_params['alphas'][i]) * (traj_i + (1.0 - model_params['alphas_bar'][i]) * score)) / torch.sqrt(model_params['alphas_bar'][i - 1])
        return traj_i_m1, costs.mean().item()

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

            # soft_cost = 0.0
            # for constraint in soft_constraints:
            #     c = constraint.q_l[0]
            #     radius = constraint.radius_l[0]
            #     lambda_c = constraint.lambda_c
            #     distance = torch.norm(state.q - c)
            #     violation = torch.relu(radius - distance)
            #     soft_cost += lambda_c * violation

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
        cost = ctrl + 2.0 * pos_err  # (B,H)

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
                print("here")
                offset = self.transforms[model_id].to(q_seq.device)  # (2,)
                q_ws = q_seq + offset.view(1, 1, -1)  # broadcast to (B,H,2)

            sdf = env.compute_sdf(q_ws.reshape(-1, self.robot.q_dim)  # (B·H,2)
                                  ).view(B, H)  # (B,H)

            # EPS_CLEAR = 1e-2
            # clearance = sdf - EPS_CLEAR
            # pen = (-clearance).clamp(min=0.0)
            # cost = cost + 20.0 * pen.pow(2)
            # free_mask = (sdf.min(dim=1).values > -EPS_CLEAR)
            pen = (-sdf).clamp(min=0.0)
            free_mask = (pen.sum(dim=1) == 0)
            cost = cost + 10.0 * pen
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
        print("solving constraints:", len(soft_constraints))
        if contexts is None:
            contexts = [None] * len(self.models)
        for m, c_dict in hard_conds.items():
            for k, v in c_dict.items():
                # k is the key of the condition, usually state index in the trajectory
                hard_conds[m][k] = einops.repeat(v, 'd -> b d', b=n_samples)

        results = {}
        chains = {}
        for model_id in self.models.keys():
            self.soft_constraints[model_id] = soft_constraints
            traj_n = torch.zeros((n_samples, self.models[model_id]['params']['horizon'], self.robot.q_dim * 2),
                                 device=self.device)
            # Reverse
            traj_i = traj_n
            model_chain = []
            costs = []

            for i in tqdm(range(self.models[model_id]['params']['n_diffusion_step'] - 1, 0, -1),
                          desc=f"Diffusing model {model_id}"):
                traj_i, cost = self._reverse_diffusion_step(model_id, i, traj_i)
                if return_chain:
                    model_chain.append(traj_i)
                costs.append(cost)

            if return_chain:
                chains[model_id] = torch.stack(model_chain)

            # evaluate final cost
            final_traj = chains[model_id][-1, ...]
            final_actions = self.robot.get_velocity(
                final_traj).cpu().numpy() if self.device != 'cpu' else self.robot.get_velocity(traj_i).numpy()
            costs_final, q = self._rollout_us(model_id, self.state_inits, final_actions)
            results[model_id] = {
                'trajectory': final_traj,
                'cost': costs_final.mean().item(),
                'cost_history': costs
            }

        if return_chain:
            return chains

        return results

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
