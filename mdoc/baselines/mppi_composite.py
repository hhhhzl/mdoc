from pathlib import Path
from typing import List, Optional, Tuple
import os
import torch
import matplotlib.pyplot as plt
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer, create_fig_and_axes
from torch_robotics.torch_utils.torch_utils import get_torch_device
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.environments.env_empty_nowait_2d import EnvEmptyNoWait2D
from torch_robotics.robots.robot_planar_disk import RobotPlanarDisk
from mp_baselines.planners.mppi import MPPI
from mp_baselines.planners.dynamics.point import PointParticleDynamics
from mp_baselines.planners.costs.cost_functions import CostCollision, CostComposite
from mdoc.common.experiments import TrialSuccessStatus


class SliceCost(torch.nn.Module):
    def __init__(self, base_cost: CostCollision, i: int, N: int):
        super().__init__()
        self.base_cost = base_cost
        self.i = i
        self.N = N

    def _ensure_traj_range(self, H: int):
        if hasattr(self.base_cost, "set_traj_range"):
            self.base_cost.set_traj_range((0, H))
        if self.base_cost.obst_factor.traj_range[1] == None:
            self.base_cost.obst_factor.traj_range[1] = H

    def eval(self, states: torch.Tensor, **kwargs) -> torch.Tensor:
        states = states.contiguous()
        B, H, D = states.shape
        N = self.N
        if D == 2 * N:
            si = states[..., 2 * self.i: 2 * (self.i + 1)]
        elif D == 4 * N:
            si = states[..., 4 * self.i: 4 * self.i + 2]
        else:
            if D < 2 * N:
                raise ValueError(f"SliceCost: unexpected state dim D={D} for N={N}")
            si = states[..., 2 * self.i:2 * (self.i + 1)]

        self._ensure_traj_range(H)
        out = self.base_cost.eval(trajs=si, q_pos=si, q_vel=None, H_positions=None, **kwargs)

        if not torch.is_tensor(out):
            out = torch.as_tensor(out, device=states.device, dtype=states.dtype)
        if out.dim() == 0:
            out = out.expand(B, H)
        elif out.dim() == 1 and out.shape[0] == B:
            out = out.unsqueeze(1).expand(B, H)
        return out

    def forward(self, states: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.eval(states, **kwargs)


class InterAgentSoftPenalty(torch.nn.Module):
    def __init__(self, N: int, weight: float = 10.0, sigma: float = 0.1):
        super().__init__()
        self.N = N
        self.weight = weight
        self.sigma2 = sigma * sigma

    def eval(self, states: torch.Tensor, **kwargs) -> torch.Tensor:
        # states: (B,H,D) where D can be 2N (pos only) or 4N (pos+vel per agent)
        B, H, D = states.shape
        N = self.N

        if D == 2 * N:
            # positions already packed
            pos = states.reshape(B, H, N, 2)
        elif D == 4 * N:
            # per-agent blocks [x,y,vx,vy]
            blocks = states.reshape(B, H, N, 4)
            pos = blocks[..., :2]
        else:
            # fallback: assume positions occupy the first 2N dims
            if D < 2 * N:
                raise ValueError(f"InterAgentSoftPenalty: unexpected state dim D={D} for N={N}")
            pos = states[..., : 2 * N].reshape(B, H, N, 2)

        diffs = pos.unsqueeze(3) - pos.unsqueeze(2)  # (B,H,N,N,2)
        d2 = (diffs ** 2).sum(dim=-1)  # (B,H,N,N)
        tril = torch.tril(torch.ones((N, N), device=states.device, dtype=torch.bool), diagonal=-1)
        pair_d2 = d2[:, :, tril]  # (B,H,M)

        penalty_t = self.weight * torch.exp(- pair_d2 / self.sigma2).sum(dim=-1)  # (B,H)
        return penalty_t

    def forward(self, states, **kwargs):
        return self.eval(states, **kwargs)


class MetaCost(torch.nn.Module):
    def __init__(self, slice_costs: List[SliceCost], inter_agent: Optional[InterAgentSoftPenalty] = None):
        super().__init__()
        self.slice_costs = torch.nn.ModuleList(slice_costs)  # 这里放的是 SliceCost（nn.Module）
        self.inter_agent = inter_agent

    def eval(self, states: torch.Tensor, **kwargs) -> torch.Tensor:
        states = states.contiguous()
        if states.dim() == 2:
            states = states.unsqueeze(0)  # -> (1,H,D)
        B, H, _ = states.shape

        total = torch.zeros((B, H), device=states.device, dtype=states.dtype)
        for sc in self.slice_costs:
            term = sc.eval(states, **kwargs)
            total = total + term

        if self.inter_agent is not None:
            inter = self.inter_agent.eval(states, **kwargs)  # (B,H)
            if inter.dim() == 1 and inter.shape[0] == B:
                inter = inter.unsqueeze(1).expand(B, H)
            elif inter.dim() == 0:
                inter = inter.expand(B, H)
            total = total + inter

        return total  # (B,H)

    def forward(self, states: torch.Tensor, **kwargs) -> torch.Tensor:
        return self.eval(states, **kwargs)


class MetaPointDynamics(PointParticleDynamics):
    """
    Reuse PointParticleDynamics with state_dim=2N, control_dim=2N.
    Interprets the META state as concatenated (x_i, y_i) blocks.
    """

    def __init__(
            self, N: int,
            rollout_steps: int,
            dt: float,
            goal_state: torch.Tensor,
            ctrl_min: List[float],
            ctrl_max: List[float],
            tensor_args
    ):
        system_params = dict(
            rollout_steps=rollout_steps,
            control_dim=2 * N,
            state_dim=2 * N,
            dt=dt,
            discount=1.0,
            goal_state=goal_state,  # (2N,)
            ctrl_min=ctrl_min,  # length 2N
            ctrl_max=ctrl_max,  # length 2N
            verbose=False,
            c_weights={'pos': 1., 'vel': 1., 'ctrl': 1., 'pos_T': 1e7, 'vel_T': 0.},
            tensor_args=tensor_args,
        )
        super().__init__(**system_params)


# ----------------------------- Planner class -----------------------------
class MPPIComposite:
    """
    Unified meta-robot MPPI planner (single MPPI in 2N-dim space).
    API mirrors MPDComposite: __init__, plan(), render_paths().
    """

    def __init__(self,
                 low_level_planner_l: List,  # Not used.
                 start_l: List[torch.Tensor],
                 goal_l: List[torch.Tensor],
                 n_support_points: int = 64,
                 dt: float = 0.04,
                 opt_iters: int = 20,
                 num_ctrl_samples: int = 128,
                 control_std_xy: float = 0.15,
                 temp: float = 1.0,
                 step_size: float = 1.0,
                 sigma_coll: float = 1e-3,
                 inter_weight: float = 10.0,
                 inter_sigma: float = 0.10,
                 device: str = 'cpu',
                 results_dir: Optional[str] = None,
                 seed: int = 0,
                 **kwargs):

        assert len(start_l) == len(goal_l) and len(start_l) > 0
        self.num_agents = len(start_l)
        self.dt = dt
        self.opt_iters = opt_iters
        self.n_support_points = n_support_points

        fix_random_seed(seed)
        self.device = get_torch_device(device)
        self.tensor_args = {'device': self.device, 'dtype': torch.float32}

        # Stack starts/goals into META vectors (2N,)
        self.start_meta = torch.cat(start_l, dim=0).to(**self.tensor_args)  # (2N,)
        self.goal_meta = torch.cat(goal_l, dim=0).to(**self.tensor_args)  # (2N,)

        # For visualization (per-agent start/goal)
        self.start_state_pos = torch.stack(start_l).to(**self.tensor_args)  # (N,2)
        self.goal_state_pos = torch.stack(goal_l).to(**self.tensor_args)  # (N,2)

        # Environment, robot, task
        # TODO: not hardcode
        self.env = EnvEmptyNoWait2D(tensor_args=self.tensor_args)
        self.robot = RobotPlanarDisk(tensor_args=self.tensor_args)
        self.task = PlanningTask(
            env=self.env,
            robot=self.robot,
            ws_limits=torch.tensor([[-0.81, -0.81], [0.95, 0.95]], **self.tensor_args),
            obstacle_cutoff_margin=0.005,
            tensor_args=self.tensor_args
        )

        # Per-agent env collision costs (each expects (B,H,2))
        per_agent_costs = []
        for field in self.task.get_collision_fields():
            # Duplicate per agent
            for i in range(self.num_agents):
                per_agent_costs.append(
                    CostCollision(self.robot, n_support_points, field=field, sigma_coll=sigma_coll,
                                  tensor_args=self.tensor_args)
                )
        # The task may return multiple fields; we want exactly one env field used.
        # If multiple fields are returned, we still build N slices referring to identical field object.
        # SliceCost will pick the i-th agent block.

        # Build META cost = sum_i env_i + inter-agent penalty
        slice_costs = []
        env_fields = self.task.get_collision_fields()
        for i in range(self.num_agents):
            for field in env_fields:
                base = CostCollision(self.robot, n_support_points, field=field,
                                     sigma_coll=sigma_coll, tensor_args=self.tensor_args)
                slice_costs.append(SliceCost(base_cost=base, i=i, N=self.num_agents))
        self.cost = MetaCost(slice_costs,
                             inter_agent=InterAgentSoftPenalty(self.num_agents, weight=inter_weight, sigma=inter_sigma))

        # Dynamics & MPPI
        ctrl_min = [-5] * (2 * self.num_agents)
        ctrl_max = [5] * (2 * self.num_agents)
        self.dynamics = MetaPointDynamics(
            N=self.num_agents,
            rollout_steps=n_support_points,
            dt=dt,
            goal_state=self.goal_meta,
            ctrl_min=ctrl_min,
            ctrl_max=ctrl_max,
            tensor_args=self.tensor_args
        )

        self.mppi = MPPI(
            self.dynamics,
            num_ctrl_samples=num_ctrl_samples,
            rollout_steps=n_support_points,
            control_std=[control_std_xy] * (2 * self.num_agents),
            temp=temp,
            opt_iters=1,  # outer loop handles iterations
            step_size=step_size,
            cov_prior_type="const_ctrl",
            tensor_args=self.tensor_args,
        )

        # Viz
        self.results_dir = results_dir or "./results_mppi_meta"
        os.makedirs(self.results_dir, exist_ok=True)
        self.colors = plt.cm.get_cmap('tab20')(torch.linspace(0, 1, self.num_agents))

    # --------------------------------------------------------
    def render_paths(
            self, paths_l: List[torch.Tensor],
            animation_duration: float = 0.0,
            output_fpath: Optional[str] = None,
            plot_trajs: bool = True,
            n_frames: Optional[int] = None,
            show_robot_in_image: bool = True
    ):
        """
        paths_l: list of N tensors of shape (H, 4) [x,y,vx,vy] (one per agent).
        Keep parity with MPDComposite rendering: concatenate as (B=1, H, 2N).
        """
        planner_visualizer = PlanningVisualizer(task=self.task)
        paths_l = [path[:, :2] for path in paths_l]
        trajs = torch.cat(paths_l, dim=1).unsqueeze(0)
        trajs_l = [pos.unsqueeze(0) for pos in paths_l]

        if animation_duration == 0:
            fig, ax = create_fig_and_axes()
            planner_visualizer.render_robot_trajectories(
                fig=fig,
                ax=ax,
                trajs=trajs,
                start_state=self.start_state_pos,
                goal_state=self.goal_state_pos,
                colors=self.colors,
                show_robot_in_image=show_robot_in_image
            )
            if output_fpath is None:
                output_fpath = os.path.join(self.results_dir, 'robot-traj.png')
            if not output_fpath.endswith('.png'):
                output_fpath += '.png'
            plt.axis('off')
            plt.savefig(output_fpath, dpi=300, bbox_inches='tight', pad_inches=0)
            return

        base_file_name = Path(os.path.basename(__file__)).stem
        planner_visualizer.animate_multi_robot_trajectories(
            trajs_l=trajs_l,
            colors=self.colors,
            start_state_l=self.start_state_pos,
            goal_state_l=self.goal_state_pos,
            plot_trajs=plot_trajs,
            video_filepath=os.path.join(self.results_dir, f'{base_file_name}-robot-traj.gif'),
            n_frames=len(paths_l[0]) if n_frames is None else n_frames,
            anim_time=animation_duration if animation_duration > 0 else self.n_support_points * self.dt
        )

    # --------------------------------------------------------
    def _moving_average(self, path: torch.Tensor, k: int = 5) -> torch.Tensor:
        if k <= 1:
            return path

        H, D = path.shape
        k = int(min(k, H))
        if k % 2 == 0:
            k = max(1, k - 1)
        if k == 1:
            return path

        x = path.T.unsqueeze(0).contiguous()  # (1, D, H)
        pad = k // 2
        x = torch.nn.functional.pad(x, (pad, pad), mode='reflect')  # pad 最后一维
        weight = torch.ones(D, 1, k, device=path.device, dtype=path.dtype) / k
        y = torch.nn.functional.conv1d(x, weight, groups=D)  # (1, D, H)
        y = y.squeeze(0).T.contiguous()  # (H, D)
        return y

    # --------------------------------------------------------
    def plan(self, runtime_limit: float = 1000.0, **kwargs):
        """
        Run joint MPPI optimization in META space.
        Return:
          path_l: list of (H,4) [x,y,vx,vy] per agent
          num_ct_expansions: 0 (no CBS tree)
          success_status: TrialSuccessStatus
          num_collisions: int (pairwise collisions counted over time)
        """
        observation = {'state': self.start_meta, 'goal_state': self.goal_meta, 'cost': self.cost}

        with TimerCUDA() as _timer:
            for _ in range(self.opt_iters):
                self.mppi.optimize(**observation)

        ctrl = self.mppi.get_mean_controls()  # (H, 2N)
        traj = self.mppi.get_state_trajectories_rollout(  # (B=1,H,2N)
            controls=ctrl.unsqueeze(0), **observation
        ).squeeze(0)

        # Split META trajectory into per-agent (H,2) and pack as (H,4)
        H = traj.shape[0]
        paths_l: List[torch.Tensor] = []
        for i in range(self.num_agents):
            pos = traj[:, 2 * i:2 * (i + 1)]
            vel = ctrl[:, 2 * i:2 * (i + 1)]
            p = torch.zeros((H, 4), **self.tensor_args)
            p[:, :2] = pos
            p[:, 2:] = vel
            paths_l.append(self._moving_average(p, k=5))

        # Count hard collisions (for status reporting)
        radius = float(getattr(self.robot, "radius", 0.06))
        num_collisions = 0
        for t in range(H):
            for i in range(self.num_agents):
                for j in range(i + 1, self.num_agents):
                    if torch.norm(paths_l[i][t, :2] - paths_l[j][t, :2]) < 2.0 * radius:
                        num_collisions += 1

        status = TrialSuccessStatus.SUCCESS if num_collisions == 0 else TrialSuccessStatus.FAIL_COLLISION_AGENTS

        if kwargs.get("render", False):
            try:
                self.render_paths(paths_l, animation_duration=0.0)
            except Exception:
                pass

        return paths_l, 0, status, num_collisions


if __name__ == "__main__":
    start_l = [torch.tensor([0.8, 0.0]), torch.tensor([-0.8, 0.0])]
    goal_l = [torch.tensor([-0.8, 0.0]), torch.tensor([0.8, 0.0])]

    planner = MPPIComposite(
        None,
        start_l,
        goal_l,
        n_support_points=32,
        dt=0.04,
        opt_iters=60,
        num_ctrl_samples=64,
        control_std_xy=0.2,
        step_size=1,
        temp=1.5,
        inter_weight=2,
        inter_sigma=0.18,
        sigma_coll=5e-3
    )
    paths, _, status, ncol = planner.plan()
    planner.render_paths(
        paths,
        animation_duration=10,
        output_fpath="meta_traj.png"
    )
