import numpy as np
import torch
from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiBoxField
from torch_robotics.torch_utils.torch_utils import get_default_tensor_args
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes
from matplotlib import pyplot as plt


class EnvUMaze2D(EnvBase):
    def __init__(self,
                 name='EnvUMaze2D',
                 tensor_args=None,
                 precompute_sdf_obj_fixed=True,
                 sdf_cell_size=0.005,
                 **kwargs):
        if tensor_args is None:
            tensor_args = get_default_tensor_args()

        boxes_centers = [
            [-0.6, 0.0],
            [0.6, 0.0],
            [0.0, -0.6],
        ]
        boxes_sizes = [
            [0.2, 1.2],
            [0.2, 1.2],
            [1.4, 0.2],
        ]

        box_field = MultiBoxField(np.array(boxes_centers),
                                  np.array(boxes_sizes),
                                  tensor_args=tensor_args)

        obj_list = [box_field]

        super().__init__(
            name=name,
            limits=torch.tensor([[-1, -1], [1, 1]], **tensor_args),
            obj_fixed_list=[ObjectField(obj_list, 'u_maze')],
            precompute_sdf_obj_fixed=precompute_sdf_obj_fixed,
            sdf_cell_size=sdf_cell_size,
            tensor_args=tensor_args,
            **kwargs
        )


if __name__ == '__main__':
    env = EnvUMaze2D()
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    ax.legend()
    plt.show()

    # Render sdf
    fig, ax = create_fig_and_axes(env.dim)
    env.render_sdf(ax, fig)

    # Render gradient of sdf
    env.render_grad_sdf(ax, fig)
    plt.show()
