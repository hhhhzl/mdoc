import numpy as np
import torch
from matplotlib import pyplot as plt
from typing import List

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField
from torch_robotics.torch_utils.torch_utils import get_default_tensor_args
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes
from mdoc.config.mmd_params import MMDParams as params


class EnvTennis2D(EnvBase):
    def __init__(self,
                 name='EnvTennis2D',
                 corridor_width: float = 0.42,
                 wall_thickness: float = 0.02,
                 chair_to_center: float = 0.4,
                 chair_length: float = 0.3,
                 chair_width: float = 0.1,
                 limits: torch.Tensor = None,
                 tensor_args=None,
                 precompute_sdf_obj_fixed=True,
                 sdf_cell_size=0.005,
                 **kwargs
                 ):
        self.corridor_width = corridor_width
        self.wall_thickness = wall_thickness

        if tensor_args is None:
            tensor_args = get_default_tensor_args()

        if limits is None:
            limits = torch.tensor([[-1.0, -1.0], [1.0, 1.0]], **tensor_args)

        Lx_min, Ly_min = float(limits[0, 0]), float(limits[0, 1])
        Lx_max, Ly_max = float(limits[1, 0]), float(limits[1, 1])
        assert np.isclose(Lx_min, Ly_min) and np.isclose(Lx_max, Ly_max), "squre as default"

        L = Lx_max
        total_span = 2.0 * L

        assert 0.0 < corridor_width < total_span, "corridor_width must be in (0, 2L)"
        assert 0.0 < wall_thickness < total_span, "wall_thickness must smaller than length"
        assert 0.0 < chair_width < corridor_width, "chair must smaller than the corridor"

        w = corridor_width
        t = wall_thickness

        seg_len = L - w / 2.0
        y_top_center = (w / 2.0 + L) / 2.0
        y_bot_center = (-L - w / 2.0) / 2.0

        boxes_centers = [
            [0.0, y_top_center],
            [0.0, y_bot_center],
        ]
        boxes_sizes = [
            [t, seg_len],  # [size_x, size_y]
            [t, seg_len],
        ]

        box_field = MultiBoxField(np.array(boxes_centers, dtype=np.float32),
                                  np.array(boxes_sizes, dtype=np.float32),
                                  tensor_args=tensor_args)
        boundary_length = (2 - wall_thickness) / 2

        boundary_centers = [
            [-(boundary_length + wall_thickness) / 2, corridor_width / 2],
            [(boundary_length + wall_thickness) / 2, corridor_width / 2],
            [(boundary_length + wall_thickness) / 2, - corridor_width / 2],
            [-(boundary_length + wall_thickness) / 2, - corridor_width / 2]
        ]

        boundary_sizes = [
            [boundary_length, wall_thickness],
            [boundary_length, wall_thickness],
            [boundary_length, wall_thickness],
            [boundary_length, wall_thickness],
        ]

        boundary_field = MultiBoxField(np.array(boundary_centers, dtype=np.float32),
                                       np.array(boundary_sizes, dtype=np.float32),
                                       tensor_args=tensor_args)

        chair_centers = [
            [chair_to_center, 0],
            [-chair_to_center, 0],
        ]
        chair_sizes = [
            [chair_length, chair_width],
            [chair_length, chair_width],
        ]

        chair_field = MultiBoxField(np.array(chair_centers, dtype=np.float32),
                                    np.array(chair_sizes, dtype=np.float32),
                                    tensor_args=tensor_args)

        obj_list = [
            box_field,
            chair_field,
            # boundary_field
        ]

        super().__init__(
            name=name,
            limits=limits,
            obj_fixed_list=[ObjectField(obj_list, 'sym_bottleneck')],
            precompute_sdf_obj_fixed=precompute_sdf_obj_fixed,
            sdf_cell_size=sdf_cell_size,
            tensor_args=tensor_args,
            **kwargs
        )

    def default_start_goal(self, margin=0.15):
        L = float(self.limits[1, 0])
        start = torch.tensor([[-L + margin, 0.0]], **self.tensor_args)
        goal = torch.tensor([[+L - margin, 0.0]], **self.tensor_args)
        return start, goal

    def get_rrt_connect_params(self, robot=None):
        params = dict(
            n_iters=10000,
            step_size=0.01,
            n_radius=0.05,
            n_pre_samples=50000,
            max_time=50
        )

        from torch_robotics.robots import RobotPlanarDisk
        if isinstance(robot, RobotPlanarDisk):
            return params

        else:
            raise NotImplementedError

    def get_gpmp2_params(self, robot=None):
        params = dict(
            n_support_points=64,
            dt=0.04,
            opt_iters=300,
            num_samples=64,
            sigma_start=1e-5,
            sigma_gp=1e-2,
            sigma_goal_prior=1e-5,
            sigma_coll=1e-5,
            step_size=1e-1,
            sigma_start_init=1e-4,
            sigma_goal_init=1e-4,
            sigma_gp_init=0.2,
            sigma_start_sample=1e-4,
            sigma_goal_sample=1e-4,
            solver_params={
                'delta': 1e-2,
                'trust_region': True,
                'method': 'cholesky',
            },
        )

        from torch_robotics.robots import RobotPlanarDisk
        if isinstance(robot, RobotPlanarDisk):
            return params
        else:
            raise NotImplementedError

    def get_chomp_params(self, robot=None):
        params = dict(
            n_support_points=64,
            dt=0.04,
            opt_iters=1,  # Keep this 1 for visualization
            weight_prior_cost=1e-4,
            step_size=0.05,
            grad_clip=0.05,
            sigma_start_init=0.001,
            sigma_goal_init=0.001,
            sigma_gp_init=0.3,
            pos_only=False,
        )

        from torch_robotics.robots import RobotPlanarDisk
        if isinstance(robot, RobotPlanarDisk):
            return params

        else:
            raise NotImplementedError

    def get_skill_pos_seq_l(self, robot=None, start_pos=None, goal_pos=None) -> List[torch.Tensor]:
        return None

    def compute_traj_data_adherence(self, path: torch.Tensor, fraction_of_length=params.data_adherence_linear_deviation_fraction) -> torch.Tensor:
        # The score is deviation of the path from a straight line. Cost in {0, 1}.
        # The score is 1 for each point on the path within a distance less than fraction_of_length * length from
        # the straight line. The computation is the average of the scores for all points in the path.
        start_state_pos = path[0][:2]
        goal_state_pos = path[-1][:2]
        length = torch.norm(goal_state_pos - start_state_pos)
        path = path[:, :2]
        path = torch.stack([path[:, 0], path[:, 1], torch.zeros_like(path[:, 0])], dim=1)
        start_state_pos = torch.stack([start_state_pos[0], start_state_pos[1], torch.zeros_like(start_state_pos[0])]).unsqueeze(0)
        goal_state_pos = torch.stack([goal_state_pos[0], goal_state_pos[1], torch.zeros_like(goal_state_pos[0])]).unsqueeze(0)
        deviation_from_line = torch.norm(torch.cross(goal_state_pos - start_state_pos, path - start_state_pos),
                                         dim=1) / length
        return (deviation_from_line < fraction_of_length).float().mean().item()


if __name__ == '__main__':
    env = EnvTennis2D(
        corridor_width=0.4,
        wall_thickness=0.02,
        chair_to_center=0.4,
        chair_length=0.4,
        chair_width=0.1,
    )

    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    s, g = env.default_start_goal()
    ax.scatter([s[0, 0].item(), g[0, 0].item()],
               [s[0, 1].item(), g[0, 1].item()],
               c=['C1', 'C2'], s=60, marker='*', label='start/goal')
    ax.legend()
    plt.title("Tennis Buddy")
    plt.show()

    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)
    plt.title("Signed Distance Field")
    plt.show()

    fig, ax = create_fig_and_axes(env.dim)
    env.render_grad_sdf(ax, fig)
    plt.title("SDF Gradient")
    plt.show()
