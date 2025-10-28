import numpy as np
import torch
from matplotlib import pyplot as pyplt
from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiBoxField
from torch_robotics.torch_utils.torch_utils import get_default_tensor_args
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes
import matplotlib.patches as mpatches


class EnvHighway3Lane2D(EnvBase):
    def __init__(self,
                 name='EnvHighway3Lane2D',
                 tensor_args=None,
                 precompute_sdf_obj_fixed=True,
                 sdf_cell_size=0.01,
                 lane_sep=0.18,
                 top_bottom_thickness=1.5,  # 1.2 & 1.4
                 dyn_counts=(1, 2, 2),
                 dyn_speeds=(0.30, 0.35, 0.40),
                 dyn_spawn_x=(-0.9, -0.9, -0.9),
                 car_len=0.10,
                 car_wid=0.10,
                 map_size=2,
                 merge_lane_weight=0.15,
                 wall_color="#ead4a8",
                 **kwargs):
        self.tensor_args = tensor_args if tensor_args is not None else get_default_tensor_args()
        self.lane_sep = lane_sep
        self.top_bottom_thickness = top_bottom_thickness
        self.dyn_counts = dyn_counts
        self.dyn_speeds = dyn_speeds
        self.dyn_spawn_x = dyn_spawn_x
        self.car_len = float(car_len)
        self.car_wid = float(car_wid)
        self.wall_color = wall_color

        self.margin_m = getattr(self, "margin_m", 0.3)
        self.seed = getattr(self, "seed", 0)
        self.rng = np.random.default_rng(self.seed)
        self._lane_x0 = {0: None, 1: None, 2: None}
        self._lane_phase = {0: 0.0, 1: 0.0, 2: 0.0}

        centers, sizes = self._build_static_boxes(
            self.top_bottom_thickness,
            map_size,
            merge_lane_weight
        )
        box_field = MultiBoxField(
            np.asarray(centers, dtype=float),
            np.asarray(sizes, dtype=float),
            tensor_args=self.tensor_args
        )

        super().__init__(
            name=name,
            limits=torch.tensor([[-map_size, -map_size], [map_size, map_size]], **self.tensor_args),
            obj_fixed_list=[ObjectField([box_field], 'highway_merge_3lane_ramp_2d')],
            precompute_sdf_obj_fixed=precompute_sdf_obj_fixed,
            sdf_cell_size=sdf_cell_size,
            tensor_args=self.tensor_args,
            **kwargs
        )

    @staticmethod
    def _build_static_boxes(top_bottom_thickness, map_size, merge_lane_weight):
        centers, sizes = [], []
        y = (map_size - top_bottom_thickness) + top_bottom_thickness / 2
        centers.append((0.0,  y))
        sizes.append((2 * map_size + 0.3, top_bottom_thickness))
        centers.append((0.0, -y))
        sizes.append((2 * map_size + 0.3, top_bottom_thickness))

        # add merge lines
        # upper
        length = map_size * 2 / 5
        y = (map_size - top_bottom_thickness) - (merge_lane_weight / 2)
        centers.append((0.0, y))
        sizes.append((length, merge_lane_weight))

        centers.append((-(map_size - length / 2), y))
        sizes.append((length, merge_lane_weight))

        centers.append((map_size - length / 2, y))
        sizes.append((length, merge_lane_weight))

        # down
        y = - ((map_size - top_bottom_thickness) - (merge_lane_weight / 2))
        centers.append((0.0, y))
        sizes.append((length, merge_lane_weight))

        centers.append((-(map_size - length / 2), y))
        sizes.append((length, merge_lane_weight))

        centers.append((map_size - length / 2, y))
        sizes.append((length, merge_lane_weight))

        return centers, sizes

    def _sample_lane_positions(self, n, C, m, rng):
        assert n >= 0
        if n == 0:
            return np.zeros((0,), dtype=float)
        assert C / n >= m, f"Infeasible: circumference {C:.3f} < n*m {n * m:.3f}"

        base = np.linspace(0.0, C, num=n, endpoint=False)  # 0, C/n, 2C/n, ...
        jitter_max = max(0.0, 0.5 * (C / n - m))
        jitter = rng.uniform(-jitter_max, +jitter_max, size=n)
        x0 = (base + jitter) % C
        rng.shuffle(x0)
        return x0

    def reset_dynamic_boxes(self, map_size):
        C = float(map_size) * 1.8
        for li in range(3):
            n = int(self.dyn_counts[li])
            self._lane_x0[li] = self._sample_lane_positions(n, C, self.margin_m, self.rng)
            self._lane_phase[li] = self.rng.uniform(0.0, C)

    def get_dynamic_boxes(self, t: float, map_size):
        lane_y = np.array([-self.lane_sep, 0.0, +self.lane_sep], dtype=float)
        centers, sizes = [], []
        length = 1.8
        x_min = -map_size + 0.3
        C = map_size * length

        if any(self._lane_x0[li] is None for li in range(3)):
            self.reset_dynamic_boxes(map_size)

        for li in range(3):
            n = int(self.dyn_counts[li])
            v = float(self.dyn_speeds[li])
            y = float(lane_y[li])
            x0_lane = self._lane_x0[li]
            phase_lane = self._lane_phase[li]

            for k in range(n):
                x0 = x0_lane[k]
                x = x_min + ((x0 + phase_lane + (v * t)) % C)
                centers.append([x, y])
                sizes.append([self.car_len, self.car_wid])

        return np.asarray(centers, dtype=float), np.asarray(sizes, dtype=float)

    def _render_static_boxes(self, ax):
        obj = self.obj_fixed_list[0].fields[0]
        c = self.wall_color
        for (cx, cy), (w, h) in zip(obj.centers.cpu().numpy(), obj.sizes.cpu().numpy()):
            rect = mpatches.Rectangle((cx - w / 2.0, cy - h / 2.0), w, h,
                                      linewidth=0.0, edgecolor=c, facecolor=c)
            ax.add_patch(rect)

    def render_with_dynamic(self, t=0.0, ax=None, show=True, render_sdf=False, render_grad=False):
        import matplotlib.pyplot as plt
        created = False
        if ax is None:
            fig, ax = create_fig_and_axes(self.dim)
            created = True
        if render_sdf:
            fig = ax.figure if ax is not None else None
            if fig is None:
                fig = plt.gcf()
            self.render_sdf(ax, fig)
        self._render_static_boxes(ax)
        if render_grad:
            fig = ax.figure if ax is not None else None
            if fig is None:
                fig = plt.gcf()
            self.render_grad_sdf(ax, fig)
        centers, sizes = self.get_dynamic_boxes(t, map_size=2)
        c = self.wall_color
        for (cx, cy), (w, h) in zip(centers, sizes):
            rect = mpatches.Rectangle((cx - w / 2.0, cy - h / 2.0), w, h,
                                      linewidth=0.0, edgecolor=c, facecolor=c)
            ax.add_patch(rect)
        ax.set_title(f"3-Lane + Ramp Merge @ t={t:.1f}s")
        ax.set_xlim(self.limits[0, 0].item(), self.limits[1, 0].item())
        ax.set_ylim(self.limits[0, 1].item(), self.limits[1, 1].item())
        ax.set_aspect('equal', adjustable='box')
        ax.axis('off')
        if show and created:
            plt.show()

    def save_animation(self, path: str, T: float = 10.0, fps: int = 10, render_sdf: bool = False,
                       render_grad: bool = False):
        from matplotlib import animation
        fig, ax = create_fig_and_axes(self.dim)
        n_frames = max(1, int(T * fps))
        times = np.linspace(0.0, T, n_frames)

        def init():
            ax.clear()
            return []

        def draw_frame(i):
            ax.clear()
            t = float(times[i])
            if render_sdf:
                self.render_sdf(ax, fig)
            self._render_static_boxes(ax)
            if render_grad:
                self.render_grad_sdf(ax, fig)
            centers, sizes = self.get_dynamic_boxes(t, map_size=2)
            artists = []
            c = self.wall_color
            for (cx, cy), (w, h) in zip(centers, sizes):
                rect = mpatches.Rectangle((cx - w / 2.0, cy - h / 2.0), w, h,
                                          linewidth=0.0, edgecolor=c, facecolor=c)
                ax.add_patch(rect)
                artists.append(rect)
            ax.set_title(f"3-Lane + Ramp Merge @ t={t:.1f}s")
            ax.set_xlim(self.limits[0, 0].item(), self.limits[1, 0].item())
            ax.set_ylim(self.limits[0, 1].item(), self.limits[1, 1].item())
            ax.set_aspect('equal', adjustable='box')
            ax.axis('off')
            return artists

        anim = animation.FuncAnimation(fig, draw_frame, init_func=init,
                                       frames=n_frames, blit=False, interval=1000.0 / fps)
        if path.lower().endswith('.gif'):
            anim.save(path, writer='pillow', fps=fps)
        else:
            anim.save(path, writer='ffmpeg', fps=fps)
        pyplt.close(fig)
        return path


if __name__ == '__main__':
    env = EnvHighway3Lane2D(
        precompute_sdf_obj_fixed=True,
        sdf_cell_size=0.02,
        tensor_args=get_default_tensor_args(),
    )
    fig, ax = create_fig_and_axes(env.dim)
    # env.render(ax)
    # pyplt.show()

    env.render_with_dynamic(t=0.0, ax=ax, show=False, render_sdf=False, render_grad=False)
    pyplt.show()
    pyplt.close(fig)
    gif_path = env.save_animation("highway_3lane_ramp_demo.gif", T=10, fps=8, render_sdf=False, render_grad=False)
