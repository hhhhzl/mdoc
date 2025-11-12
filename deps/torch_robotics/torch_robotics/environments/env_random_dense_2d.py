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


class EnvRandomDense2D(EnvBase):

    def __init__(self,
                 name='EnvRandomDense2D',
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
                )
                spheres = MultiSphereField(
                    np.asarray(s_centers, dtype=float),
                    np.asarray(s_radii, dtype=float),
                    tensor_args=tensor_args,
                )
                objs.append(spheres)

        super().__init__(
            name=name,
            limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
            obj_fixed_list=[ObjectField(objs if obj_list is None else obj_list, 'random2d')],
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
    obj_list = [
        MultiSphereField(
            np.array(
                [[-0.43378472328186035, 0.3334643840789795], [0.3313474655151367, 0.6288051009178162],
                 [-0.5656964778900146, -0.484994500875473], [0.42124247550964355, -0.6656165719032288],
                 [0.05636655166745186, -0.5149664282798767], [-0.36961784958839417, -0.12315540760755539],
                 [-0.8740217089653015, -0.4034936726093292], [-0.6359214186668396, 0.6683124899864197],
                 [0.808782160282135, 0.5287870168685913], [-0.023786112666130066, 0.4590069353580475],
                 [0.11544948071241379, -0.12676022946834564], [0.1455741971731186, 0.16420497000217438],
                 [0.628413736820221, -0.43461447954177856], [0.17965620756149292, -0.8926276564598083],
                 [0.6775968670845032, 0.8817358016967773], [-0.3608766794204712, 0.8313458561897278],
                 ]),
            np.array(
                [0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125, 0.125,
                 0.125, 0.125,
                 ]
            )
            ,
            tensor_args=get_default_tensor_args()
        ),
        MultiBoxField(
            np.array(
                [[0.607781708240509, 0.19512386620044708], [0.5575312972068787, 0.5508843064308167],
                 [-0.3352295458316803, -0.6887519359588623], [-0.6572632193565369, 0.31827881932258606],
                 [-0.664594292640686, -0.016457155346870422], [0.8165988922119141, -0.19856023788452148],
                 [-0.8222246170043945, -0.6448580026626587], [-0.2855989933013916, -0.36841487884521484],
                 [-0.8946458101272583, 0.8962447643280029], [-0.23994405567646027, 0.6021060943603516],
                 [-0.006193588487803936, 0.8456171751022339], [0.305103600025177, -0.3661990463733673],
                 [-0.10704007744789124, 0.1318950206041336], [0.7156378626823425, -0.6923345923423767]
                 ]
            ),
            np.array(
                [[0.20000000298023224, 0.20000000298023224], [0.20000000298023224, 0.20000000298023224],
                 [0.20000000298023224, 0.20000000298023224], [0.20000000298023224, 0.20000000298023224],
                 [0.20000000298023224, 0.20000000298023224], [0.20000000298023224, 0.20000000298023224],
                 [0.20000000298023224, 0.20000000298023224], [0.20000000298023224, 0.20000000298023224],
                 [0.20000000298023224, 0.20000000298023224], [0.20000000298023224, 0.20000000298023224],
                 [0.20000000298023224, 0.20000000298023224], [0.20000000298023224, 0.20000000298023224],
                 [0.20000000298023224, 0.20000000298023224], [0.20000000298023224, 0.20000000298023224]
                 ]
            )
            ,
            tensor_args=get_default_tensor_args()
        )
    ]

    # Example 2: truly random non-overlapping spheres (no boxes)
    env_large = EnvRandomDense2D(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.01,
        tensor_args=get_default_tensor_args(),
    )

    fig, ax = create_fig_and_axes(env_large.dim)
    env_large.render(ax)
    plt.show()
