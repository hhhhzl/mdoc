import torch
from typing import Tuple
from mp_baselines.planners.base import MPPlanner
from mdoc.models.diffusion_models.mbd_ensemble import ModelBasedDiffusionEnsemble
from mdoc.common import smooth_trajs

class MDOC(MPPlanner):
    """
    MDOC planner

    Drop-in alternative to MPPI:
      - same optimize() signature & return (control_samples, state_trajectories, costs)
      - get_recent_samples(), get_mean_controls(), pop(), shift(), get_state_trajectories_rollout()

    Internally, each optimize step performs one reverse-diffusion iteration over a batch of
    trajectories (position + control), evaluates costs via the model, computes importance
    weights, and updates the latent trajectory toward the weighted mean (score-based update).
    """

    def __init__(
        self,
        *,
        robot,
        env_model,
        start_state_pos: torch.Tensor,
        goal_state_pos: torch.Tensor,
        rollout_steps: int,
        n_diffusion_steps: int = 64,
        n_samples: int = 256,
        temp_sample: float = 1.0,
        seed: int = 0,
        transforms=None,
        enable_demo: bool = False,
        tensor_args=None,
        device: str = None,
        **kwargs,
    ):
        super().__init__(name="MDOC", tensor_args=tensor_args or {})

        if device is None:
            device = (self.tensor_args or {}).get("device", "cpu")
        self.device = torch.device(device)

        self.robot = robot
        self.state_dim = getattr(robot, "q_dim", 2)  # position dimension (planar default=2)
        self.control_dim = self.state_dim            # single-integrator: u has same dim as q
        self.rollout_steps = int(rollout_steps)
        self.n_diffusion_steps = int(n_diffusion_steps)
        self.n_samples = int(n_samples)
        self.temp_sample = float(temp_sample)

        # Build a single-model ensemble wrapper (model_id == 0)
        self._ensemble = ModelBasedDiffusionEnsemble(
            robot=robot,
            env_models={0: env_model},
            transforms={0: transforms} if transforms is not None else {},
            start_state_pos=start_state_pos.to(self.device),
            goal_state_pos=goal_state_pos.to(self.device),
            seed=seed,
            enable_demo=enable_demo,
            device=str(self.device),
        )

        # Override diffusion params to match constructor args
        p = self._ensemble.models[0]["params"]
        p["horizon"] = self.rollout_steps
        p["n_diffusion_step"] = self.n_diffusion_steps
        p["n_samples"] = self.n_samples
        p["temp_sample"] = self.temp_sample

        # Recompute the schedule with the updated params
        self._ensemble._setup_diffusion_params(0, p)

        # Latent trajectory (x: position, u: control) for all samples
        self._traj_i = torch.zeros(
            self.n_samples,
            self.rollout_steps,
            self.state_dim * 2,
            device=self.device
        )

        self._recent_control_samples = None
        self._recent_state_trajectories = None
        self._recent_weights = None
        self._mean_controls = torch.zeros(self.rollout_steps, self.control_dim, device=self.device)
        self.best_cost = torch.inf
        self.best_traj = None  # (H, state_dim)

        self.trace_diffusion = False
        self._diffusion_trace = []
        self._diffusion_rep = []

    def start_diffusion_trace(self):
        self.trace_diffusion = True
        self._diffusion_trace = []
        self._diffusion_rep = []

    def stop_diffusion_trace(self):
        self.trace_diffusion = False

    def get_diffusion_trace(self):
        return self._diffusion_trace

    def get_diffusion_representatives(self):
        return self._diffusion_rep

    def get_state_trajectories_rollout(self, controls=None, num_ctrl_samples=None, **observation):
        """Roll out with the internal, collision-aware MDOC model (fast path).
        Returns a tensor of shape (num_ctrl_samples, H, state_dim).
        """
        if controls is None:
            controls = self._mean_controls.unsqueeze(0)  # (1,H,control_dim)
            num_ctrl_samples = 1
        else:
            num_ctrl_samples = controls.shape[0]

        costs, q_seq, _ = self._ensemble._rollout_single_batch_new2_ultrafast(
            model_id=0,
            state_init=self._ensemble.state_inits,
            us=controls,
        )
        return q_seq.detach().clone()

    def get_recent_samples(self):
        return (
            None if self._recent_control_samples is None else self._recent_control_samples.detach().clone(),
            None if self._recent_state_trajectories is None else self._recent_state_trajectories.detach().clone(),
            None if self._recent_weights is None else self._recent_weights.detach().clone(),
        )

    def get_mean_controls(self):
        return self._mean_controls.detach().clone()

    def pop(self):
        """Pop the first control (for MPC usage)."""
        a0 = self._mean_controls[0, :].clone().detach()
        self.shift()
        return a0

    def shift(self):
        self._mean_controls = self._mean_controls.roll(shifts=-1, dims=0)
        self._mean_controls[-1, :] = 0.0

    # ------------------------------------------------------------
    # Diffusion core (one reverse step)
    # ------------------------------------------------------------
    @torch.no_grad()
    def _reverse_step(self, i: int):
        """One reverse-diffusion iteration, matching MDOC logic.
        Returns (control_samples, state_trajs, costs, weights, traj_i_minus_1)
          - control_samples: (N, H, control_dim)
          - state_trajs    : (N, H, state_dim)
          - costs          : (N, 1)  (per-sample aggregated cost)
          - weights        : (N,)    (softmax importance weights)
        """
        model = self._ensemble.models[0]
        sigmas = model["sigmas"]
        alphas = model["alphas"]
        alphas_bar = model["alphas_bar"]
        params = model["params"]

        N, H, SD2 = self._traj_i.shape
        assert N == self.n_samples and H == self.rollout_steps and SD2 == self.state_dim * 2

        # Sample noisy trajectory around current latent
        eps = torch.randn_like(self._traj_i)
        traj_0s = torch.clamp(eps * sigmas[i] + self._traj_i, -1.0, 1.0)  # (N,H,2*state_dim)

        # Actions are the last state_dim entries per-step (single-integrator)
        actions = traj_0s[..., self.state_dim:]

        # Batch rollout & cost with collision awareness
        costs, q_seq, free_mask = self._ensemble._rollout_single_batch_new2(
            model_id=0, state_init=self._ensemble.state_inits, us=actions
        )

        # Importance weights (softmax over normalized negative costs)
        c_mu = costs.mean()
        c_std = costs.std().clamp_min(1e-4)
        logw = - (costs - c_mu) / c_std / params["temp_sample"]
        logw[~free_mask] = -torch.inf
        weights = torch.softmax(logw.reshape(-1, 1), dim=0).reshape(-1)  # (N,)

        # Weighted mean trajectory in latent space
        traj_bar = torch.einsum("n,nij->ij", weights, traj_0s)  # (H,2*state_dim)

        # Score-based update (DDIM-like) to the previous latent
        num = -self._traj_i + torch.sqrt(alphas_bar[i]) * traj_bar
        score = num / (1.0 - alphas_bar[i])
        prev = (self._traj_i + (1.0 - alphas_bar[i]) * score) / torch.sqrt(alphas[i])
        if i > 0:
            prev = prev / torch.sqrt(alphas_bar[i - 1])

        return actions, q_seq, costs.reshape(-1, 1), weights, prev

    # ------------------------------------------------------------
    # Optimize
    # ------------------------------------------------------------
    def optimize(self, opt_iters: int = None, **observation) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Runs n_diffusion_steps reverse iterations. Returns tuple:
            (control_samples, state_trajectories, costs)
        from the final iteration.
        """
        if opt_iters is None:
            opt_iters = self.n_diffusion_steps

        best_cost = torch.inf
        last_control_samples = None
        last_state_trajs = None
        last_costs = None
        last_weights = None

        for step, i in enumerate(range(self.n_diffusion_steps - 1, -1, -1)):
            with torch.no_grad():
                U_samp, X_samp, C_samp, W_samp, prev = self._reverse_step(i)
                self._traj_i = prev
                self._recent_control_samples = U_samp
                self._recent_state_trajectories = X_samp
                self._recent_weights = W_samp

                if self.trace_diffusion == True:
                    try:
                        pos_trajs = self.robot.get_position(X_samp)
                    except Exception:
                        pos_trajs = X_samp[..., :2]

                    pos_trajs = smooth_trajs(pos_trajs)
                    self._diffusion_trace.append(pos_trajs.detach().to("cpu", dtype=torch.float32))
                    if W_samp is not None:
                        rep = torch.einsum("n,nhd->hd", W_samp, pos_trajs)
                    else:
                        if C_samp is not None:
                            rep_idx = torch.argmin(C_samp.reshape(-1)).item()
                            rep = pos_trajs[rep_idx] if pos_trajs.ndim == 3 else pos_trajs
                        else:
                            rep = pos_trajs[0] if pos_trajs.ndim == 3 else pos_trajs
                    self._diffusion_rep.append(rep.detach().to("cpu", dtype=torch.float32))

                # Track the best sample across steps
                cur_min_val, cur_min_idx = torch.min(C_samp.reshape(-1), dim=0)
                if cur_min_val < best_cost:
                    best_cost = cur_min_val
                    best_idx = int(cur_min_idx)
                    self.best_traj = X_samp[best_idx].detach().clone()

                last_control_samples = U_samp
                last_state_trajs = X_samp
                last_costs = C_samp
                last_weights = W_samp

        # Define mean-controls as weighted mean from the last diffusion step
        if last_control_samples is not None:
            self._mean_controls = torch.einsum("n,nij->ij", last_weights, last_control_samples)

        self.best_cost = float(best_cost)
        return last_control_samples, last_state_trajs, last_costs

    # ------------------------------------------------------------
    def render(self, ax, **kwargs):
        raise NotImplementedError
