# viz/highway_viz.py
import os
import io
import math
from typing import List, Optional, Tuple
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import imageio.v2 as imageio


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


class HighwayVisualizer:
    def __init__(
            self,
            env,
            map_size: float,
            dt: float,
            H: int,
            results_dir: str = "./",
            fig_size: Tuple[int, int] = (6, 4),
            dpi: int = 120,
    ):
        self.env = env
        self.map_size = float(map_size)
        self.dt = float(dt)
        self.H = int(H)
        self.results_dir = results_dir
        os.makedirs(self.results_dir, exist_ok=True)
        self.car_len = float(getattr(env, "car_len", 0.10))
        self.car_wid = float(getattr(env, "car_wid", 0.10))
        self.obs_radius = 0.5 * math.hypot(self.car_len, self.car_wid) * 1.15
        self.lane_sep = float(getattr(env, "lane_sep", 0.18))
        plt.ion()
        self.fig, self.ax = plt.subplots(figsize=fig_size, dpi=dpi)
        self._frames = []
        self._draw_background()
        self.obs_artists = []
        self.agent_path_lines = []
        self.agent_points = []

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _draw_background(self):
        ax = self.ax
        L = self.map_size
        ax.add_patch(Rectangle((-L, -L), 2 * L, 2 * L, fill=False, lw=1.0))
        for y in (-self.lane_sep, 0.0, +self.lane_sep):
            ax.plot([-L, +L], [y, y], linestyle="--", linewidth=1.0)
        ax.set_xlim(-L, L)
        ax.set_ylim(-L, L)
        ax.set_aspect("equal")
        ax.set_title("Highway Merge (Receding CBS/MPC)")

    def _clear_dynamic(self):
        for a in self.obs_artists:
            a.remove()
        for a in self.agent_path_lines:
            a.remove()
        for a in self.agent_points:
            a.remove()
        self.obs_artists.clear()
        self.agent_path_lines.clear()
        self.agent_points.clear()

    def draw_step(
            self,
            t0: float,
            paths: List[torch.Tensor],
            show_dynamic_obstacles: bool = True,
    ):
        ax = self.ax
        self._clear_dynamic()

        if show_dynamic_obstacles:
            centers, _sizes = self.env.get_dynamic_boxes(t0, self.map_size)
            if centers is not None and len(centers) > 0:
                for c in centers:
                    c = _to_numpy(c)
                    circ = Circle((c[0], c[1]), radius=self.obs_radius, fill=False, linewidth=1.0)
                    ax.add_patch(circ)
                    self.obs_artists.append(circ)

        for p in paths:
            P = _to_numpy(p)  # (H, D)
            line, = ax.plot(P[:, 0], P[:, 1], linewidth=1.2)
            self.agent_path_lines.append(line)
            pt = ax.plot(P[1, 0], P[1, 1], marker="o", markersize=4)[0]
            self.agent_points.append(pt)

        # update
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # to memory
        buf = io.BytesIO()
        self.fig.savefig(buf, format="png", bbox_inches="tight")
        buf.seek(0)
        img = imageio.imread(buf)
        self._frames.append(img)
        buf.close()

    def save_gif(self, filename: str = "rollout.gif", fps: int = 10):
        path = os.path.join(self.results_dir, filename)
        if len(self._frames) > 0:
            imageio.mimsave(path, self._frames, fps=fps)
        return path
