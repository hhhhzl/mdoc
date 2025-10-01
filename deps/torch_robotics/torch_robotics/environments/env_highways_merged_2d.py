import numpy as np
import torch
from matplotlib import pyplot as pyplt
from torch_robotics.environments.env_base import EnvBase
from torch_robotics.environments.primitives import ObjectField, MultiBoxField
from torch_robotics.torch_utils.torch_utils import get_default_tensor_args
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes


class EnvHighwayMerge2D(EnvBase):
    def __init__(self,
                 name='EnvHighwayMerge2D',
                 tensor_args=None,
                 precompute_sdf_obj_fixed=True,
                 sdf_cell_size=0.01,
                 # geometry (all values in workspace units [-1, 1])
                 wall_thickness=0.6,     # top/bottom wall height
                 median_thickness=0.15,  # height of the median wall around y=0
                 x_gap_l=0.2,            # rectangular merge opening: left x
                 x_gap_r=0.55,           # rectangular merge opening: right x
                 # dynamic obstacle (slow cars) config
                 dyn_count=3,
                 dyn_lane_y=-0.25,
                 dyn_speed=0.3,         # workspace units per second (arbitrary)
                 dyn_radius=0.07,
                 **kwargs
                 ):

        self.tensor_args = tensor_args if tensor_args is not None else get_default_tensor_args()
        self.dyn_cfg = dict(count=dyn_count, lane_y=dyn_lane_y, speed=dyn_speed, radius=dyn_radius)

        # Build static obstacles as MultiBoxField
        boxes_centers, boxes_sizes = self._build_static_boxes(
            wall_thickness=wall_thickness,
            median_thickness=median_thickness,
            x_gap_l=x_gap_l, x_gap_r=x_gap_r
        )

        box_field = MultiBoxField(
            np.asarray(boxes_centers, dtype=float),
            np.asarray(boxes_sizes, dtype=float),
            tensor_args=self.tensor_args,
        )

        super().__init__(
            name=name,
            limits=torch.tensor([[-1, -1], [1, 1]], **self.tensor_args),
            obj_fixed_list=[ObjectField([box_field], 'highway_merge_2d')],
            precompute_sdf_obj_fixed=precompute_sdf_obj_fixed,
            sdf_cell_size=sdf_cell_size,
            tensor_args=self.tensor_args,
            **kwargs
        )

    @staticmethod
    def _build_static_boxes(wall_thickness=0.6, median_thickness=0.15, x_gap_l=0.2, x_gap_r=0.55):
        """
        Returns centers and sizes for axis-aligned boxes composing the obstacles.
        Coordinate frame: x,y in [-1,1].
        """
        centers = []
        sizes = []

        # Top wall strip: centered at y = +0.7, height wall_thickness (clipped by workspace bounds)
        top_center = (0.0, 0.7)
        top_size   = (2.0, wall_thickness)  # full width
        centers.append(top_center); sizes.append(top_size)

        # Bottom wall strip: centered at y = -0.7
        bot_center = (0.0, -0.7)
        bot_size   = (2.0, wall_thickness)
        centers.append(bot_center); sizes.append(bot_size)

        # Median wall around y=0, with a rectangular gap in [x_gap_l, x_gap_r].
        # Implemented as two boxes: left segment [-1, x_gap_l], right segment [x_gap_r, 1]
        # Left segment
        left_w = (x_gap_l - (-1.0))
        left_cx = -1.0 + left_w / 2.0
        centers.append((left_cx, 0.0)); sizes.append((left_w, median_thickness))

        # Right segment
        right_w = (1.0 - x_gap_r)
        right_cx = x_gap_r + right_w / 2.0
        centers.append((right_cx, 0.0)); sizes.append((right_w, median_thickness))

        return centers, sizes

    # ------------------------- Dynamic obstacles (slow cars) -------------------------
    def get_dynamic_obstacles(self, t: float):
        """
        Returns positions (N x 2) and radii (N,) of moving round obstacles at time t.
        Obstacles move along the lower lane (y = dyn_lane_y) from x=-0.9 to x=0.9 cyclically.
        """
        n = self.dyn_cfg['count']
        y = self.dyn_cfg['lane_y']
        v = self.dyn_cfg['speed']
        r = self.dyn_cfg['radius']

        # Evenly spread initial phases along the lane
        xs = []
        for i in range(n):
            # phase in [0,1)
            phase0 = i / float(n)
            # lane length ~ 1.8 (from -0.9 to 0.9). Wrap with modulo.
            length = 1.8
            x = -0.9 + ((phase0 + (v * t) / length) % 1.0) * length
            xs.append([x, y])

        return np.array(xs, dtype=float), np.ones(n, dtype=float) * r

    # ------------------------------- Visualization -------------------------------
    def render_with_dynamic(self, t=0.0, ax=None, show=True, render_sdf=False, render_grad=False):
        """
        Renders static map and overlays dynamic obstacles at time t.
        Args:
            t: time (seconds)
            ax: matplotlib axes (if None, will create)
            show: call plt.show()
            render_sdf: if True, draw SDF heatmap under the map
            render_grad: if True, draw SDF gradient quiver (requires render_sdf True for best visibility)
        """
        import matplotlib.pyplot as plt
        created = False
        if ax is None:
            fig, ax = create_fig_and_axes(self.dim)
            created = True

        # Optionally render SDF first
        if render_sdf:
            # The base class provides render_sdf(ax, fig). Create a fig if missing.
            fig = ax.figure if ax is not None else None
            if fig is None:
                fig = plt.gcf()
            self.render_sdf(ax, fig)

        # Render static objects
        self.render(ax)

        # Optionally overlay SDF gradient
        if render_grad:
            fig = ax.figure if ax is not None else None
            if fig is None:
                import matplotlib.pyplot as plt
                fig = plt.gcf()
            self.render_grad_sdf(ax, fig)

        # Dynamic obstacles
        pos, rad = self.get_dynamic_obstacles(t)
        for (x, y), r in zip(pos, rad):
            circ = plt.Circle((x, y), r, fill=False, linewidth=1.5)
            ax.add_patch(circ)
            ax.text(x + 0.04, y + 0.04, "dyn", fontsize=8)

        ax.set_title(f"EnvHighwayMerge2D @ t={t:.1f}s")

        if show and created:
            import matplotlib.pyplot as plt
            plt.show()

    def save_animation(self, path:str, T:float=10.0, fps:int=10, render_sdf:bool=False, render_grad:bool=False):
        """
        Save an animation of the environment with dynamic obstacles.
        Args:
            path: output filepath, e.g. '/mnt/data/highway_merge.gif' or '.mp4'
            T: total duration in seconds
            fps: frames per second
            render_sdf: if True, overlay SDF heatmap
            render_grad: if True, overlay SDF gradient quiver
        """
        from matplotlib import animation

        fig, ax = create_fig_and_axes(self.dim)

        n_frames = max(1, int(T * fps))
        times = np.linspace(0.0, T, n_frames)

        # We'll re-draw each frame: SDF (optional), static map, dynamic obstacles.
        def init():
            ax.clear()
            return []

        def draw_frame(i):
            ax.clear()
            t = float(times[i])
            # SDF & grad
            if render_sdf:
                self.render_sdf(ax, fig)
            self.render(ax)
            if render_grad:
                self.render_grad_sdf(ax, fig)

            # Dynamic obstacles
            pos, rad = self.get_dynamic_obstacles(t)
            artists = []
            for (x, y), r in zip(pos, rad):
                circ = pyplt.Circle((x, y), r, fill=False, linewidth=1.5)
                ax.add_patch(circ); artists.append(circ)
                txt = ax.text(x + 0.04, y + 0.04, "dyn", fontsize=8)
                artists.append(txt)

            ax.set_title(f"EnvHighwayMerge2D @ t={t:.1f}s")
            ax.set_xlim(self.limits[0,0].item(), self.limits[1,0].item())
            ax.set_ylim(self.limits[0,1].item(), self.limits[1,1].item())
            ax.set_aspect('equal', adjustable='box')
            ax.axis('off')
            return artists

        anim = animation.FuncAnimation(fig, draw_frame, init_func=init,
                                       frames=n_frames, blit=False, interval=1000.0/fps)

        # Save
        if path.lower().endswith('.gif'):
            anim.save(path, writer='pillow', fps=fps)
        else:
            anim.save(path, writer='ffmpeg', fps=fps)

        pyplt.close(fig)
        return path


if __name__ == '__main__':
    env = EnvHighwayMerge2D(precompute_sdf_obj_fixed=True, sdf_cell_size=0.02)

    # Static: SDF + gradient + dynamic obstacles
    fig, ax = create_fig_and_axes(env.dim)
    env.render_with_dynamic(t=0.0, ax=ax, show=False, render_sdf=True, render_grad=True)
    pyplt.savefig("highway_merge_static_sdf_grad.png", dpi=160, bbox_inches="tight")
    pyplt.close(fig)
    gif_path = env.save_animation("highway_merge_demo.gif", T=4.0, fps=8, render_sdf=False, render_grad=False)
