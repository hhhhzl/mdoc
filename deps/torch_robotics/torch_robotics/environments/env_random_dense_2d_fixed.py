import pprint

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


class EnvRandomDense2DFixed(EnvBase):

    def __init__(self,
                 name='EnvRandomDense2DFixed',
                 tensor_args=None,
                 precompute_sdf_obj_fixed=True,
                 sdf_cell_size=0.005,
                 # boxes
                 number_of_box=10,
                 box_min_size=0.15,
                 box_max_size=0.15,  # same => squares
                 box_margin=0.12,
                 box_gap=0.12,
                 # spheres
                 number_of_sphere=10,
                 sphere_r_min=0.075,  # used by random
                 sphere_r_max=0.075,  # used by random
                 sphere_margin=0.12,  # used by random
                 sphere_gap=0.12,  # used by random
                 avoid_box_gap=0.12,
                 obj_list=None,
                 **kwargs):

        # Safe tensor_args
        if tensor_args is None:
            tensor_args = get_default_tensor_args()

        if obj_list is None:
            obj_list = [
                MultiBoxField(
                    np.array(
                        [
                            [0.0674, 0.4946],
                            [-0.5246, 0.7158],
                            [-0.4528, 0.3092],
                            [0.5270, 0.7762],
                            [-0.5582, -0.6203],
                            [0.5134, -0.5559],
                            [0.2395, -0.6180],
                            [0.7647, -0.0775],
                            [-0.7293, -0.2989],
                            [0.5600, 0.3212],
                        ],
                        dtype=float,
                    ),
                    np.array(
                        [
                            [0.1500, 0.1500],
                            [0.1500, 0.1500],
                            [0.1500, 0.1500],
                            [0.1500, 0.1500],
                            [0.1500, 0.1500],
                            [0.1500, 0.1500],
                            [0.1500, 0.1500],
                            [0.1500, 0.1500],
                            [0.1500, 0.1500],
                            [0.1500, 0.1500],
                        ],
                        dtype=float,
                    ),
                    tensor_args=tensor_args,
                ),
                MultiSphereField(
                    np.array(
                        [
                            [-0.2554, -0.6865],
                            [-0.1397, 0.0374],
                            [0.1038, -0.2132],
                            [0.2134, 0.8029],
                            [-0.7735, 0.2027],
                            [0.3318, 0.0443],
                            [-0.4673, -0.0207],
                            [0.8020, -0.7493],
                            [-0.1693, -0.2355],
                            [-0.1827, 0.7722],
                        ],
                        dtype=float,
                    ),
                    np.array(
                        [0.0750, 0.0750, 0.0750, 0.0750, 0.0750, 0.0750, 0.0750, 0.0750, 0.0750, 0.0750],
                        dtype=float,
                    ),
                    tensor_args=tensor_args,
                ),
            ]
        objs = obj_list

        super().__init__(
            name=name,
            limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
            obj_fixed_list=[ObjectField(objs if obj_list is None else obj_list, 'randomdense2d')],
            precompute_sdf_obj_fixed=precompute_sdf_obj_fixed,
            sdf_cell_size=sdf_cell_size,
            tensor_args=tensor_args,
            **kwargs
        )

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

    def compute_traj_data_adherence(self, path: torch.Tensor,
                                    fraction_of_length=params.data_adherence_linear_deviation_fraction) -> torch.Tensor:
        # The score is deviation of the path from a straight line. Cost in {0, 1}.
        # The score is 1 for each point on the path within a distance less than fraction_of_length * length from
        # the straight line. The computation is the average of the scores for all points in the path.
        start_state_pos = path[0][:2]
        goal_state_pos = path[-1][:2]
        length = torch.norm(goal_state_pos - start_state_pos)
        path = path[:, :2]
        path = torch.stack([path[:, 0], path[:, 1], torch.zeros_like(path[:, 0])], dim=1)
        start_state_pos = torch.stack(
            [start_state_pos[0], start_state_pos[1], torch.zeros_like(start_state_pos[0])]).unsqueeze(0)
        goal_state_pos = torch.stack(
            [goal_state_pos[0], goal_state_pos[1], torch.zeros_like(goal_state_pos[0])]).unsqueeze(0)
        deviation_from_line = torch.norm(torch.cross(goal_state_pos - start_state_pos, path - start_state_pos),
                                         dim=1) / length
        return (deviation_from_line < fraction_of_length).float().mean().item()


if __name__ == '__main__':
    # Example 2: truly random non-overlapping spheres (no boxes)
    env_dense = EnvRandomDense2DFixed(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.01,
        tensor_args=get_default_tensor_args(),
    )

    fig, ax = create_fig_and_axes(env_dense.dim)
    env_dense.render(ax)
    plt.show()
