# cem.py
import torch
from mp_baselines.planners.priors.gaussian import get_multivar_gaussian_prior
from mp_baselines.planners.base import MPPlanner


class CEM(MPPlanner):
    def __init__(
        self,
        system,
        num_ctrl_samples,
        rollout_steps,
        opt_iters,
        elite_frac=0.1,
        min_elites=16,
        step_size=1.0,
        control_std=None,
        initial_mean=None,
        cov_prior_type='indep_ctrl',
        tensor_args=None,
        **kwargs
    ):
        super(CEM, self).__init__(name='CEM', tensor_args=tensor_args or {})
        self.system = system
        self.state_dim = system.state_dim
        self.control_dim = system.control_dim
        self.rollout_steps = rollout_steps
        self.num_ctrl_samples = num_ctrl_samples
        self.opt_iters = opt_iters

        self.elite_frac = float(elite_frac)
        self.min_elites = int(min_elites)
        self.step_size = float(step_size)

        self._mean = torch.zeros(
            self.rollout_steps,
            self.control_dim,
            **self.tensor_args,
        )

        self.control_std = control_std
        self.cov_prior_type = cov_prior_type
        self.ctrl_dist = get_multivar_gaussian_prior(
            control_std,
            rollout_steps,
            self.control_dim,
            Cov_type=cov_prior_type,
            mu_init=self._mean,
            tensor_args=self.tensor_args,
        )

        self.best_cost = torch.inf
        self.reset(initial_mean=initial_mean)

    def reset(self, initial_mean=None):
        if initial_mean is not None:
            self._mean = initial_mean.clone()
        else:
            self._mean.zero_()
        self.update_ctrl_dist()

    def update_ctrl_dist(self):
        self.ctrl_dist.update_means(self._mean)

    def get_mean_controls(self):
        return self._mean

    def get_recent_samples(self):
        return (
            self._recent_control_samples.detach().clone(),
            self._recent_state_trajectories.detach().clone(),
            self._recent_elite_mask.detach().clone()
        )

    def get_state_trajectories_rollout(self, controls=None, num_ctrl_samples=None, **observation):
        state_trajectories = torch.empty(
            1 if num_ctrl_samples is None else num_ctrl_samples,
            self.rollout_steps,
            self.state_dim,
            **self.tensor_args,
        )
        if controls is None:
            control_samples = self._mean.unsqueeze(0)
        else:
            control_samples = controls

        # x_{t+1} = f(x_t, u_t)
        state_trajectories[:, 0] = observation['state']
        for i in range(self.rollout_steps - 1):
            state_trajectories[:, i + 1] = self.system.dynamics(
                state_trajectories[:, i].unsqueeze(1),
                control_samples[:, i].unsqueeze(1),
            ).squeeze(1)
        return state_trajectories.clone()

    @torch.no_grad()
    def sample_and_eval(self, **observation):
        control_samples = self.ctrl_dist.sample(self.num_ctrl_samples)  # (N, H, du)

        # rollout
        state_trajectories = self.get_state_trajectories_rollout(
            controls=control_samples, num_ctrl_samples=self.num_ctrl_samples, **observation
        )
        self.state_trajectories = state_trajectories.clone()
        costs = self.system.traj_cost(
            state_trajectories.transpose(0, 1).unsqueeze(2),
            control_samples.transpose(0, 1).unsqueeze(2),
            **observation
        )
        self.costs = costs

        return control_samples, state_trajectories, costs

    @torch.no_grad()
    def update_controller(self, costs, U_sampled):
        N = U_sampled.shape[0]
        k = max(int(self.elite_frac * N), self.min_elites)
        k = min(k, N)
        elite_costs, elite_idx = torch.topk(costs.squeeze(-1), k=k, largest=False, sorted=False)
        U_elite = U_sampled[elite_idx]  # (k, H, du)

        elite_mean = U_elite.mean(dim=0)  # (H, du)
        self._mean.add_(self.step_size * (elite_mean - self._mean))

        # std_elite = torch.clamp(U_elite.std(dim=0), min=1e-6)
        # self.ctrl_dist = get_multivar_gaussian_prior(
        #     std_elite, self.rollout_steps, self.control_dim,
        #     Cov_type=self.cov_prior_type, mu_init=self._mean, tensor_args=self.tensor_args
        # )

        self.update_ctrl_dist()

    def optimize(self, opt_iters=None, **observation):
        if opt_iters is None:
            opt_iters = self.opt_iters

        for _ in range(opt_iters):
            control_samples, state_trajectories, costs = self.sample_and_eval(**observation)
            self._save_best(costs, state_trajectories)
            self.update_controller(costs, control_samples)

        self._recent_control_samples = control_samples
        self._recent_state_trajectories = state_trajectories
        elite_mask = torch.zeros_like(costs, dtype=control_samples.dtype)
        k = max(int(self.elite_frac * self.num_ctrl_samples), self.min_elites)
        k = min(k, self.num_ctrl_samples)
        elite_idx = torch.topk(costs.squeeze(-1), k=k, largest=False, sorted=False).indices
        elite_mask[elite_idx] = 1.0
        self._recent_elite_mask = elite_mask

        return control_samples, state_trajectories, costs

    def _save_best(self, costs, state_trajectories):
        best_cost = torch.min(costs)
        idx = torch.argmin(costs)
        if best_cost < self.best_cost:
            self.best_cost = best_cost
            self.best_traj = state_trajectories[idx, ...]

    def pop(self):
        action = self._mean[0, :].clone().detach()
        self.shift()
        return action

    def shift(self):
        self._mean = self._mean.roll(shifts=-1, dims=0)
        self._mean[-1:] = 0.

    def render(self, ax, **kwargs):
        raise NotImplementedError