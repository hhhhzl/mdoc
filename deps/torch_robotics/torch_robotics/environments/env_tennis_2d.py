import numpy as np
import torch
from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiBoxField
from torch_robotics.torch_utils.torch_utils import get_default_tensor_args
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes
from matplotlib import pyplot as plt


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
