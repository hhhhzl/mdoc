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
                 lane_sep=0.06,
                 top_bottom_thickness=1.4,  # 1.2 & 1.4
                 ramp_start=(0.85, -0.85),
                 ramp_end=(0.10, -0.10),
                 ramp_width=0.22,
                 ramp_steps=28,
                 dyn_counts=(3, 3, 3),
                 dyn_speeds=(0.30, 0.35, 0.40),
                 dyn_spawn_x=(-0.9, -0.9, -0.9),
                 car_len=0.06,
                 car_wid=0.04,
                 wall_color="#ead4a8",
                 **kwargs):
        self.tensor_args = tensor_args if tensor_args is not None else get_default_tensor_args()
        self.lane_sep = lane_sep
        self.top_bottom_thickness = top_bottom_thickness
        self.ramp_start = ramp_start
        self.ramp_end = ramp_end
        self.ramp_width = ramp_width
        self.ramp_steps = ramp_steps
        self.dyn_counts = dyn_counts
        self.dyn_speeds = dyn_speeds
        self.dyn_spawn_x = dyn_spawn_x
        self.car_len = float(car_len)
        self.car_wid = float(car_wid)
        self.wall_color = wall_color

        centers, sizes = self._build_static_boxes(self.top_bottom_thickness,
                                                  self.ramp_start, self.ramp_end,
                                                  self.ramp_width, self.ramp_steps)
        box_field = MultiBoxField(np.asarray(centers, dtype=float),
                                  np.asarray(sizes, dtype=float),
                                  tensor_args=self.tensor_args)

        super().__init__(
            name=name,
            limits=torch.tensor([[-1, -1], [1, 1]], **self.tensor_args),
            obj_fixed_list=[ObjectField([box_field], 'highway_merge_3lane_ramp_2d')],
            precompute_sdf_obj_fixed=precompute_sdf_obj_fixed,
            sdf_cell_size=sdf_cell_size,
            tensor_args=self.tensor_args,
            **kwargs
        )

    @staticmethod
    def _build_static_boxes(top_bottom_thickness, ramp_start, ramp_end, ramp_width, ramp_steps):
        centers, sizes = [], []
        centers.append((0.0, 0.8))
        sizes.append((2.0, top_bottom_thickness))
        centers.append((0.0, -0.8))
        sizes.append((2.0, top_bottom_thickness))
        # x0, y0 = float(ramp_start[0]), float(ramp_start[1])
        # x1, y1 = float(ramp_end[0]), float(ramp_end[1])
        # dx = (x1 - x0) / ramp_steps
        # dy = (y1 - y0) / ramp_steps
        # step = max(abs(dx), abs(dy))
        # half = ramp_width * 0.5
        # wx = step + 0.02
        # wy = 0.02
        # for i in range(ramp_steps):
        #     cx = x0 + dx * i
        #     cy = y0 + dy * i
        #     centers.append((cx + half, cy))
        #     sizes.append((wx, wy))
        #     centers.append((cx - half, cy))
        #     sizes.append((wx, wy))
        return centers, sizes

    def get_dynamic_boxes(self, t: float):
        lane_y = np.array([-self.lane_sep, 0.0, +self.lane_sep], dtype=float)
        centers, sizes = [], []
        length = 1.8
        x_min = -0.9
        for li in range(3):
            n = int(self.dyn_counts[li])
            v = float(self.dyn_speeds[li])
            y = float(lane_y[li])
            for k in range(n):
                phase0 = k / max(1, n)
                x = x_min + ((phase0 + (v * t) / length) % 1.0) * length
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
        centers, sizes = self.get_dynamic_boxes(t)
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
            centers, sizes = self.get_dynamic_boxes(t)
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
    env = EnvHighway3Lane2D(precompute_sdf_obj_fixed=True, sdf_cell_size=0.02)
    fig, ax = create_fig_and_axes(env.dim)
    env.render_with_dynamic(t=0.0, ax=ax, show=False, render_sdf=False, render_grad=False)
    pyplt.show()
    pyplt.close(fig)
    gif_path = env.save_animation("highway_3lane_ramp_demo.gif", T=5.0, fps=8, render_sdf=False, render_grad=False)
