import os
import math
from typing import List, Optional, Sequence, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import imageio.v2 as imageio


# ---------- Core helpers ----------


def _draw_polyline(ax, pts, color, lw=2.0, alpha=0.9, ls='-'):
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return
    ax.add_collection(LineCollection([pts[:, :2]], colors=[color], linewidths=lw, alpha=alpha, linestyles=ls))


def _rgba(c, a):
    import matplotlib.colors as mcolors
    r, g, b = mcolors.to_rgb(c)
    return (r, g, b, a)


def _canvas_to_rgb(fig):
    fig.canvas.draw()
    if hasattr(fig.canvas, "buffer_rgba"):
        buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)
        if buf.ndim == 3 and buf.shape[-1] == 4:
            return buf[..., :3].copy()

    rgb = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    if rgb.size == w * h * 3:
        return rgb.reshape(h, w, 3)

    argb = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    if argb.size == w * h * 4:
        argb = argb.reshape(h, w, 4)
        rgba = np.roll(argb, -1, axis=2)  # ARGB -> RGBA
        return rgba[..., :3].copy()

    w2 = int(round(fig.bbox_inches.width * fig.dpi))
    h2 = int(round(fig.bbox_inches.height * fig.dpi))
    if rgb.size == w2 * h2 * 3:
        return rgb.reshape(h2, w2, 3)
    raise RuntimeError(
        f"Cannot interpret canvas buffer: rgb.size={rgb.size}, "
        f"get_width_height={(w, h)}, bbox_px={(w2, h2)}"
    )


def _has_render_with_dynamic(env) -> bool:
    return hasattr(env, "render_with_dynamic") and callable(getattr(env, "render_with_dynamic"))


def _get_limits(env) -> Tuple[float, float, float, float]:
    # Try env.limits (torch tensor 2x2) or fallback to [-2,2]x[-2,2]
    if hasattr(env, "limits"):
        lims = getattr(env, "limits")
        try:
            lims = np.asarray(lims, dtype=float)
            return lims[0, 0], lims[1, 0], lims[0, 1], lims[1, 1]
        except Exception:
            pass
    return -2.0, 2.0, -2.0, 2.0


def _draw_agents(ax, agent_positions: Sequence[Sequence[float]],
                 agent_radius: float = 0.06,
                 colors: Optional[List[str]] = None,
                 trails: Optional[List[np.ndarray]] = None):
    if colors is None:
        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    for i, pos in enumerate(agent_positions):
        c = colors[i % len(colors)]
        x, y = float(pos[0]), float(pos[1])
        circ = mpatches.Circle((x, y), agent_radius, edgecolor="none", facecolor=c, alpha=0.95)
        ax.add_patch(circ)
        ax.plot([x], [y], marker="o", markersize=1, color="black", alpha=0.5)
        # trail
        if trails is not None and i < len(trails) and trails[i] is not None and len(trails[i]) >= 2:
            pts = np.asarray(trails[i], dtype=float)
            ax.add_collection(LineCollection([pts], linewidths=1.5, alpha=0.6, colors=[c]))


def _draw_lane_guides(env, ax):
    if hasattr(env, "lane_sep"):
        try:
            sep = float(getattr(env, "lane_sep"))
            x0, x1, y0, y1 = _get_limits(env)
            for y in (-sep, 0.0, +sep):
                ax.axhline(y, lw=1.2, ls='--', color='lightgray', alpha=0.8)
        except Exception:
            pass


def _draw_heading_lines(ax, agent_positions, headings: Optional[Sequence[float]],
                        trails: Optional[List[np.ndarray]], scale: float = 0.25):
    for i, pos in enumerate(agent_positions):
        x, y = float(pos[0]), float(pos[1])
        if headings is not None and i < len(headings) and headings[i] is not None:
            th = float(headings[i])
            dx, dy = scale * np.cos(th), scale * np.sin(th)
        else:
            dx, dy = 0.0, 0.0
            if trails is not None and i < len(trails) and len(trails[i]) >= 2:
                p = np.asarray(trails[i], dtype=float)
                vx, vy = p[-1, 0] - p[-2, 0], p[-1, 1] - p[-2, 1]
                n = np.hypot(vx, vy) + 1e-9
                dx, dy = scale * vx / n, scale * vy / n
        ax.plot([x, x + dx], [y, y + dy], lw=1.0, alpha=0.9, color='k')


class OnlineRenderer:
    """
    Usage:
        vis = OnlineRenderer(env, dt=0.1, map_size=2.0)
        # in your control loop:
        vis.add_frame(t, agent_positions=[[x1,y1], [x2,y2]], trails=[np.array([...]), ...])
        ...
        vis.save_gif("/mnt/data/rollout.gif", fps=10)
    """

    def __init__(
            self,
            env: Any = None,
            dt: float = 0.1,
            map_size: float = 2.0,
            agent_radius: float = 0.06,
            show_live: bool = True,
            reuse_figure=True,
    ):
        self.env = env
        self.dt = float(dt)
        self.map_size = float(map_size)
        self.agent_radius = float(agent_radius)
        self.frames: List[np.ndarray] = []
        self.show_live = bool(show_live)
        self._live_fig = None
        self._live_ax = None
        self._trails: List[List[List[float]]] = []  # accumulated trails per agent
        self._has_dynamic = _has_render_with_dynamic(env) if env is not None else False
        self.reuse_figure = reuse_figure
        self._fig = None
        self._ax = None

    def _ensure_trails_len(self, n_agents: int):
        while len(self._trails) < n_agents:
            self._trails.append([])

    def _prepare_axes(self):
        if self.reuse_figure and self._fig is not None:
            self._ax.cla()
            return self._fig, self._ax
        fig, ax = plt.subplots(figsize=(5, 5), dpi=140)
        return fig, ax

    def add_frame(self, t: float, agent_positions: Sequence[Sequence[float]],
                  trails: Optional[List[np.ndarray]] = None,
                  headings: Optional[Sequence[float]] = None,
                  render_sdf: bool = False, render_grad: bool = False,
                  title: Optional[str] = None,
                  future_paths: Optional[List[np.ndarray]] = None,
                  future_candidates: Optional[List[List[np.ndarray]]] = None
                  ):
        # Create a fresh figure for capture to avoid residual artists
        fig, ax = self._prepare_axes()

        drew = False
        if self.env is not None:
            if self._has_dynamic:
                self.env.render_with_dynamic(float(t), ax=ax, show=False,
                                             render_sdf=render_sdf, render_grad=render_grad)
                drew = True
            elif hasattr(self.env, "render"):
                try:
                    self.env.render(ax=ax)
                    drew = True
                except Exception:
                    pass

        if not drew:
            x0, x1, y0, y1 = _get_limits(self.env)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_aspect('equal', adjustable='box')
            ax.axis('off')

        _draw_lane_guides(self.env, ax)

        # Background
        if self.env is not None and self._has_dynamic:
            # Use user's env native rendering
            self.env.render_with_dynamic(
                t=float(t), ax=ax, show=False, render_sdf=render_sdf, render_grad=render_grad
            )
        else:
            # Fallback minimal background
            x0, x1, y0, y1 = _get_limits(self.env) if self.env is not None else (-2, 2, -2, 2)
            ax.set_xlim(x0, x1)
            ax.set_ylim(y0, y1)
            ax.set_aspect('equal', adjustable='box')
            ax.set_title(f"t = {t:.2f}s" if title is None else title)
            ax.grid(False)
            ax.axis("off")

        # Trails (auto-accumulate if not provided)
        if trails is None:
            self._ensure_trails_len(len(agent_positions))
            for i, pos in enumerate(agent_positions):
                self._trails[i].append([float(pos[0]), float(pos[1])])
            use_trails = [np.asarray(tr, dtype=float) for tr in self._trails]
        else:
            use_trails = trails

        # Agents
        _draw_agents(ax, agent_positions, agent_radius=self.agent_radius, trails=use_trails)
        _draw_heading_lines(ax, agent_positions, headings, use_trails, scale=0.22)

        colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
        if future_candidates is not None:
            for i, cand_list in enumerate(future_candidates):
                if not cand_list:
                    continue
                base_c = colors[i % len(colors)]
                for cpath in cand_list:
                    _draw_polyline(ax, cpath, color=_rgba(base_c, 0.25), lw=1.0, alpha=0.25)

        if future_paths is not None:
            for i, fpath in enumerate(future_paths):
                if fpath is None or len(fpath) < 2:
                    continue
                base_c = colors[i % len(colors)]
                _draw_polyline(ax, fpath, color=base_c, lw=2.5, alpha=0.95)
                # p_end = np.asarray(fpath[-1, :2], dtype=float)
                # p_pre = np.asarray(fpath[-2, :2], dtype=float)
                # v = p_end - p_pre
                # n = np.hypot(v[0], v[1]) + 1e-9
                # u = 0.12 * v / n
                # ax.arrow(p_end[0] - u[0], p_end[1] - u[1], u[0], u[1], head_width=0.06, head_length=0.08, fc=base_c, ec=base_c, alpha=0.9, length_includes_head=True)

        if title is not None:
            ax.set_title(title)

        # Capture to RGBA array
        fig.canvas.draw()
        frame = _canvas_to_rgb(fig)
        self.frames.append(frame)

        if self.show_live:
            plt.pause(0.001)
        else:
            plt.close(fig)

    def save_gif(self, path: str, fps: int = 10, loop: int = 0):
        if not self.frames:
            raise RuntimeError("No frames recorded. Call add_frame() first.")
        # os.makedirs(os.path.dirname(path), exist_ok=True)
        duration = 1.0 / max(1, int(fps))
        imageio.mimsave(path, self.frames, duration=duration, loop=loop)
