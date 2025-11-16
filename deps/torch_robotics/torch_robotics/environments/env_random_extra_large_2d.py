import numpy as np
import torch
from matplotlib import pyplot as plt
from typing import List

from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiSphereField, MultiBoxField
from torch_robotics.environments.utils import create_grid_spheres, sample_non_overlapping_boxes_2d, \
    sample_non_overlapping_spheres_2d
from torch_robotics.robots import RobotPointMass
from torch_robotics.torch_utils.torch_utils import get_default_tensor_args
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes
from mdoc.config.mmd_params import MMDParams as params

class EnvRandomExtraLarge2D(EnvBase):

    def __init__(self,
                 name='EnvRandomExtraLarge2D',
                 tensor_args=None,
                 precompute_sdf_obj_fixed=True,
                 sdf_cell_size=0.005,
                 # boxes
                 number_of_box=15,
                 box_min_size=0.15,
                 box_max_size=0.15,  # same => squares
                 box_margin=0.1,
                 box_gap=0.36,
                 # spheres
                 number_of_sphere=15,
                 sphere_r_min=0.075,  # used by random
                 sphere_r_max=0.075,  # used by random
                 sphere_margin=0.1,  # used by random
                 sphere_gap=0.36,  # used by random
                 obj_list=None,
                 avoid_box_gap=0.36,
                 **kwargs):

        # Safe tensor_args
        if tensor_args is None:
            tensor_args = get_default_tensor_args()

        if obj_list is None:
            objs = []

            # boxes — avoid boxes just placed
            box_centers, box_sizes = (None, None)
            if number_of_box > 0:
                box_centers, box_sizes = sample_non_overlapping_boxes_2d(
                    n_boxes=number_of_box,
                    min_size=box_min_size,
                    max_size=box_max_size,
                    margin=box_margin,
                    gap=box_gap,
                    rng=np.random.default_rng(),
                    map_size=2,
                )
                box_field = MultiBoxField(
                    np.asarray(box_centers, dtype=float),
                    np.asarray(box_sizes, dtype=float),
                    tensor_args=tensor_args,
                )
                objs.append(box_field)

            # spheres — avoid boxes just placed
            if number_of_sphere > 0:
                s_centers, s_radii = sample_non_overlapping_spheres_2d(
                    n_spheres=number_of_sphere,
                    r_min=sphere_r_min,
                    r_max=sphere_r_max,
                    margin=sphere_margin,
                    gap=sphere_gap,
                    rng=np.random.default_rng(),
                    avoid_box_centers=box_centers,
                    avoid_box_sizes=box_sizes,
                    avoid_box_gap=avoid_box_gap,  # extra clearance to boxes if desired
                    map_size=2,
                )
                spheres = MultiSphereField(
                    np.asarray(s_centers, dtype=float),
                    np.asarray(s_radii, dtype=float),
                    tensor_args=tensor_args,
                )
                objs.append(spheres)

        super().__init__(
            name=name,
            limits=torch.tensor([[-3, -3], [3, 3]], **tensor_args),
            obj_fixed_list=[ObjectField(objs if obj_list is None else obj_list, 'random2d')],
            precompute_sdf_obj_fixed=precompute_sdf_obj_fixed,
            sdf_cell_size=sdf_cell_size,
            tensor_args=tensor_args,
            **kwargs
        )

    def get_rrt_connect_params(self, robot=None):
        params = dict(
            n_iters=20000,
            step_size=0.005,
            n_radius=0.04,
            n_pre_samples=50000,
            max_time=150
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
    env_sparse = EnvRandomExtraLarge2D(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.01,
        tensor_args=get_default_tensor_args(),
    )

    fig, ax = create_fig_and_axes(env_sparse.dim)
    env_sparse.render(ax)
    plt.show()
