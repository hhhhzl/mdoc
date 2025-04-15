import os
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import functools


class ModelBasedDiffusion:
    def __init__(
            self,
            env,
            env_name,
            seed=0,
            enable_demo=False,
            disable_recommended_params=False,
            device='cpu'
    ):
        """
        Initialize the diffusion-based trajectory optimizer with PyTorch.

        Args:
            env: The environment object
            env_name: Name of the environment
            seed: Random seed for reproducibility
            enable_demo: Whether to use demonstration trajectories
            disable_recommended_params: Whether to disable recommended parameters
            device: Device to run computations on ('cpu' or 'cuda')
        """
        self.device = device
        self.env = env
        self.env_name = env_name
        self.enable_demo = enable_demo
        self.disable_recommended_params = disable_recommended_params

        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)

        # Default parameters
        self.temp_sample = 0.1
        self.n_diffusion_step = 100
        self.number_sample = 8192
        self.horizon = 40
        self.beta0 = 1e-4
        self.betaT = 0.02

        # Recommended parameters for specific environments
        self.recommended_params = {
            "temp_sample": {"mdoc": 0.1},
            "n_diffusion_step": {"mdoc": 100},
            "number_sample": {"mdoc": 8192},
            "horizon": {"mdoc": 40}
        }

        # Initialize environment parameters
        self._setup_environment()

    def _setup_environment(self):
        """Initialize the environment and related parameters."""
        if not self.disable_recommended_params:
            for param_name in ["temp_sample", "n_diffusion_step", "number_sample", "horizon"]:
                recommended_value = self.recommended_params[param_name].get(
                    self.env_name, getattr(self, param_name)
                )
                setattr(self, param_name, recommended_value)
            print(f"Using recommended parameters: temp_sample={self.temp_sample}")

        self.Nx = self.env.observation_size
        self.action_size = self.env.action_size
        self.state_init = self.env.reset()

    def _setup_diffusion_params(self):
        # Create tensors on the specified device
        self.betas = torch.linspace(self.beta0, self.betaT, self.n_diffusion_step, device=self.device)
        self.alphas = 1.0 - self.betas
        self.alphas_bar = torch.cumprod(self.alphas, dim=0)
        self.sigmas = torch.sqrt(1 - self.alphas_bar)

        # Conditional distribution parameters
        shifted_alphas_bar = torch.roll(self.alphas_bar, 1)
        shifted_alphas_bar[0] = 1.0
        sigmas_cond = ((1 - self.alphas) * (1 - torch.sqrt(shifted_alphas_bar)) / (1 - self.alphas_bar))
        self.sigmas_cond = torch.sqrt(sigmas_cond)
        self.sigmas_cond[0] = 0.0
        print(f"Initial sigma = {self.sigmas[-1].item():.2e}")

    def use_demonstrations(self, qs, rew_mean, rew_std, logp0):
        xref_logpds = torch.tensor([self.env.eval_xref_logpd(q) for q in qs], device=self.device)
        xref_logpds = xref_logpds - xref_logpds.max()
        logpdemo = (
                (xref_logpds + self.env.rew_xref - rew_mean) / rew_std / self.temp_sample
        )
        demo_mask = logpdemo > logp0
        logp0 = torch.where(demo_mask, logpdemo, logp0)
        logp0 = (logp0 - logp0.mean()) / logp0.std() / self.temp_sample
        return logp0

    def _reverse_n_diffusion_step(self, i, traj_bars_i):
        """
        Single step of the reverse diffusion process.

        Args:
            i: Current diffusion step index
            Ybar_i: Current trajectory estimate

        Returns:
            Updated trajectory estimate and mean reward
        """
        # Recover noisy trajectory
        traj_i = traj_bars_i * torch.sqrt(self.alphas_bar[i])

        # Sample from q_i
        eps_u = torch.randn((self.number_sample, self.horizon, self.action_size), device=self.device)
        traj_0s = eps_u * self.sigmas[i] + traj_bars_i
        traj_0s = torch.clamp(traj_0s, -1.0, 1.0)

        # Evaluate sampled trajectories
        sample_costs = []
        qs = []
        for j in range(self.number_sample):
            actions = traj_0s[j].cpu().numpy() if self.device != 'cpu' else traj_0s[j].numpy()
            costs, q = self.rollout(self.state_init, actions)
            sample_costs.append(costs)
            qs.append(q)

        costs = torch.tensor(np.mean(sample_costs, axis=-1), device=self.device)
        cost_std = costs.std()
        cost_mean = costs.mean()
        cost_std = torch.where(cost_std < 1e-4, torch.tensor(1.0, device=self.device), cost_std)
        
        logp0 = (costs - cost_mean) / cost_std / self.temp_sample

        # Incorporate demonstration if enabled
        if self.enable_demo:
            logp0 = self.use_demonstrations(qs, cost_mean, cost_std, logp0)

        # Update trajectory using weighted average
        weights = torch.nn.functional.softmax(logp0, dim=0)
        traj_bar = torch.einsum("n,nij->ij", weights, traj_0s)

        # Compute score function and update
        score = 1 / (1.0 - self.alphas_bar[i]) * (-traj_i + torch.sqrt(self.alphas_bar[i]) * traj_bar)
        traj_i_m1 = (1 / torch.sqrt(self.alphas[i]) * (traj_i + (1.0 - self.alphas_bar[i]) * score)) / torch.sqrt(
            self.alphas_bar[i - 1])

        return traj_i_m1, costs.mean().item()

    def rollout(self, state, actions):
        """Rollout trajectory given initial state and actions."""
        costs = []
        states = [state]
        for t in range(actions.shape[0]):
            state, cost, done, _ = self.env.step(state, actions[t])
            costs.append(cost)
            states.append(state)
            if done:
                break
        return np.array(costs), states

    def optimize(self):
        self._setup_diffusion_params()

        # Initialize with zero trajectory
        traj_n = torch.zeros((self.horizon, self.action_size), device=self.device)

        # Run reverse diffusion
        traj_i = traj_n
        traj_bars = []
        costs = []

        for i in tqdm(range(self.n_diffusion_step - 1, 0, -1), desc="Diffusing"):
            traj_i, cost = self._reverse_n_diffusion_step(i, traj_i)
            traj_bars.append(traj_i)
            costs.append(cost)

        traj_bars = torch.stack(traj_bars)

        # Evaluate final reward
        final_actions = traj_bars[-1].cpu().numpy() if self.device != 'cpu' else traj_bars[-1].numpy()
        costs_final, _ = self.rollout(self.state_init, final_actions)
        cost_final = np.mean(costs_final)

        return traj_bars[-1], cost_final
