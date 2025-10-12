from matplotlib import patches
from matplotlib.lines import Line2D
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import os
from torch_robotics.visualizers.planning_visualizer import PlanningVisualizer, create_fig_and_axes
from matplotlib import cm
from matplotlib.colors import ListedColormap
import torch
from mdoc.common import smooth_trajs
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
from PIL import Image, ImageDraw
import matplotlib.lines as mlines


def _histogram_smooth(xy, bounds, bins=120, ksize=9, sigma=2.0):
    """2D histogram + Gaussian smoothing (no seaborn)."""
    (xmin, ymin), (xmax, ymax) = bounds
    H, xedges, yedges = np.histogram2d(
        xy[:, 0], xy[:, 1],
        bins=bins,
        range=[[xmin, xmax], [ymin, ymax]]
    )
    H = H.T
    ax = np.arange(-(ksize // 2), ksize // 2 + 1)
    xx, yy = np.meshgrid(ax, ax, indexing="xy")
    K = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    K /= K.sum()
    pad = ksize // 2
    Hp = np.pad(H, pad_width=pad, mode="edge")
    out = np.zeros_like(H, dtype=float)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            patch = Hp[i:i + ksize, j:j + ksize]
            out[i, j] = np.sum(patch * K)
    Xc, Yc = np.meshgrid((xedges[:-1] + xedges[1:]) / 2, (yedges[:-1] + yedges[1:]) / 2, indexing='xy')
    return out, Xc, Yc


# def plot_overlay(env, endpoints_dict, ws_limits, start, goal, save_path):
#     """
#     Overlayed iso-density contours (single axes) per method with a shared contour level."""
#     # compute densities
#     densities = {}
#     fields = {}
#     bounds = ws_limits
#
#     stack_list = []
#     for name, pts in endpoints_dict.items():
#         H, Xc, Yc = _histogram_smooth(pts, bounds=bounds, bins=100, ksize=9, sigma=2.0)
#         densities[name] = H
#         fields[name] = (Xc, Yc)
#         stack_list.append(H)
#     if not stack_list:
#         raise RuntimeError("No endpoints to plot. Provide at least one method's endpoints.")
#
#     stack = np.stack(stack_list, axis=0)
#     level = np.percentile(stack, 92.0)  # shared isodensity level
#
#     # set up figure
#     fig, ax = create_fig_and_axes(env.dim)
#     env.render(ax)
#
#     # ax = plt.gca()
#     (xmin, ymin), (xmax, ymax) = ws_limits
#     ax.set_xlim(xmin, xmax)
#     ax.set_ylim(ymin, ymax)
#     ax.set_aspect('equal', adjustable='box')
#     ax.set_title('Fig.4 (Scheme B)  Overlayed Isodensity Contours in a Narrow Passage')
#
#     # draw contours with distinct linestyles (no explicit colors)
#     styles = {
#         'mppi': '--',
#         'cem': '-.',
#         'mdoc': '-'
#     }
#     legend_elems = []
#     for name, H in densities.items():
#         Xc, Yc = fields[name]
#         ls = styles.get(name, '-')
#         ax.contour(Xc, Yc, H, levels=[level], linewidths=2, linestyles=ls)
#
#     # Start / Goal
#     ax.scatter([start[0]], [start[1]], s=80, marker='>')
#     ax.scatter([goal[0]], [goal[1]], s=80, marker='*')
#
#     # pass-through estimation (for symmetric bottleneck: x > 0 means passed)
#     for name, pts in endpoints_dict.items():
#         pass_rate = float((pts[:, 0] > 0.0).mean() * 100.0)
#         legend_elems.append(Line2D([0], [0], linestyle=styles.get(name, '-'), linewidth=2,
#                                    label=f"{name} (pass {pass_rate:.1f}%)"))
#     legend_elems.append(Line2D([0], [0], marker='>', linestyle='', label='Start'))
#     legend_elems.append(Line2D([0], [0], marker='*', linestyle='', label='Goal'))
#     ax.legend(handles=legend_elems, loc='lower right')
#     plt.grid(True, alpha=0.4)
#     plt.tight_layout()
#     os.makedirs(Path(save_path).parent, exist_ok=True)
#     plt.savefig(save_path, dpi=200, bbox_inches='tight')
#     print(f"[Saved] {save_path}")
#     return fig


def plot_paths(
        task,
        env,
        path_dict,
        start,
        goal,
        save_path,
        colors,
        animation_duration: float = 10.0,
        n_frames=None,
        plot_trajs=True,
        show_robot_in_image=True
):
    # Render
    planner_visualizer = PlanningVisualizer(
        task=task,
    )

    # Add batch dimension to all paths.
    # paths_l = [path.unsqueeze(0) for path in paths_l]
    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)

    for index, method in enumerate(list(path_dict.keys())):
        fig, ax = planner_visualizer.render_robot_trajectories(
            fig=fig,
            ax=ax,
            trajs=smooth_trajs(path_dict[method]) if method != "rrt*" else path_dict[method],
            start_state=start,
            goal_state=goal,
            colors=[colors[method]],
            show_robot_in_image=show_robot_in_image,
            check_collision=False,
            color_same=True
        )

    handles = [
        mpatches.Patch(color=colors[method], label=method.upper())
        for method in path_dict.keys()
    ]
    ax.legend(handles=handles, loc='upper left', fontsize=18)

    os.makedirs(Path(save_path).parent, exist_ok=True)
    if not save_path.endswith('.png'):
        save_path = save_path + '.png'
    plt.axis('off')
    plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)
    print(f'Saving image to: file://{os.path.abspath(save_path)}')

    base_file_name = Path(os.path.basename(__file__)).stem
    # Render the paths.
    # print(f'Rendering paths and saving to: file://{os.path.abspath(output_fpath)}')
    # planner_visualizer.animate_multi_robot_trajectories(
    #     trajs_l=paths_l,
    #     start_state_l=start,
    #     goal_state_l=goal,
    #     plot_trajs=plot_trajs,
    #     video_filepath=output_fpath,
    #     n_frames=max((2, paths_l[0].shape[1])) if n_frames is None else n_frames,
    #     anim_time=animation_duration,
    #     constraints=None,
    #     colors=methods_color
    # )


def _alpha_colormap(base_cmap, high_alpha=0.88, low_alpha=0.0):
    cmap = cm.get_cmap(base_cmap, 256)
    colors = cmap(np.linspace(0, 1, 256))
    alphas = np.linspace(low_alpha, high_alpha, 256) ** 1.2  #
    colors[:, -1] = alphas
    return ListedColormap(colors)


def plot_overlay(
        env,
        endpoints_dict,
        colors,
        ws_limits,
        start,
        goal,
        radius,
        save_path,
        bins=180,
        ksize=9,
        sigma=1.6,
        q_contour=92.0,
        max_alpha=0.85
):
    def _rgba_from_color(color_tuple, default_alpha=0.75):
        if len(color_tuple) == 3:
            r, g, b = color_tuple
            a = default_alpha
        elif len(color_tuple) == 4:
            r, g, b, a = color_tuple
            if a is None:
                a = default_alpha
        else:
            raise ValueError(f"color must be (r,g,b) or (r,g,b,a), got {color_tuple}")
        return float(r), float(g), float(b), float(a)

    line_color = colors
    styles = {'mppi': '-', 'cem': '-', 'mdoc': '-'}
    densities, fields, stack_list, names = {}, {}, [], []
    for raw_name, pts in endpoints_dict.items():
        name = raw_name.lower()
        if pts is None or len(pts) == 0:
            continue
        H, Xc, Yc = _histogram_smooth(pts, bounds=ws_limits, bins=bins, ksize=ksize, sigma=sigma)
        densities[name] = H
        fields[name] = (Xc, Yc)
        names.append(name)
        stack_list.append(H)
    if not stack_list:
        raise RuntimeError("No endpoints to plot. Provide at least one method's endpoints.")
    stack = np.stack(stack_list, axis=0)
    level = np.percentile(stack, q_contour)

    fig, ax = create_fig_and_axes(env.dim)
    env.render(ax)
    (xmin, ymin), (xmax, ymax) = ws_limits
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_aspect('equal', adjustable='box')
    from torch_robotics.torch_utils.torch_utils import to_numpy

    legend_handles, legend_labels = [], []

    for name in names:
        H = densities[name]
        Xc, Yc = fields[name]

        p97 = np.percentile(H, 97.5)
        Hn = np.clip(H / (p97 + 1e-9), 0.0, 1.0)

        base_rgba = _rgba_from_color(colors[name])
        base_rgb = base_rgba[:3]
        alpha_map = (Hn * max_alpha).astype(np.float32)

        overlay = np.zeros((H.shape[0], H.shape[1], 4), dtype=np.float32)
        overlay[..., 0] = base_rgb[0]
        overlay[..., 1] = base_rgb[1]
        overlay[..., 2] = base_rgb[2]
        overlay[..., 3] = alpha_map

        ax.imshow(
            overlay,
            origin='lower',
            interpolation='bilinear',
            aspect='auto',
            extent=(xmin, xmax, ymin, ymax),
            zorder=2
        )

        ax.contour(Xc, Yc, H, levels=[level], colors='w', linewidths=2.6, zorder=3)
        ax.contour(Xc, Yc, H, levels=[level], colors=[line_color[name]], linestyles=styles[name],
                   linewidths=1.6, zorder=4)

        pts = endpoints_dict[name]
        pass_rate = float((pts[:, 0] > 0.0).mean() * 100.0)
        legend_handles.append(Line2D([0], [0], linewidth=8, color=line_color[name]))
        legend_labels.append(f"{name.upper()}  pass {pass_rate:.1f}%")

    # # Start / Goal
    # if start is not None:
    #     start_state_np = to_numpy(start)
    #     if len(start_state_np) == 3:
    #         ax.plot(start_state_np[0], start_state_np[1], start_state_np[2], 'go', markersize=radius * 100)
    #     else:
    #         ax.plot(start_state_np[0], start_state_np[1], 'go', markersize=radius * 100)
    # if goal is not None:
    #     goal_state_np = to_numpy(goal)
    #     if len(goal_state_np) == 3:
    #         ax.plot(goal_state_np[0], goal_state_np[1], goal_state_np[2], marker='o',
    #                 color='purple', markersize=radius * 100)
    #     else:
    #         ax.plot(goal_state_np[0], goal_state_np[1], marker='o', color='purple', markersize=radius * 100)

    legend_handles += [Line2D([0], [0], marker='>', linestyle='', color='k'),
                       Line2D([0], [0], marker='*', linestyle='', color='k')]
    # legend_labels  += ['Start','Goal']

    ax.legend(legend_handles, legend_labels, loc='lower right', framealpha=0.95)
    plt.grid(True, alpha=0.30)
    plt.tight_layout()
    os.makedirs(Path(save_path).parent, exist_ok=True)
    if not save_path.endswith('.png'):
        save_path = save_path + '.png'
    plt.axis('off')
    plt.savefig(save_path, dpi=100, bbox_inches='tight', pad_inches=0)
    print(f'Saving image to: file://{os.path.abspath(save_path)}')
    return fig


def _to_cpu_np(x: torch.Tensor):
    return x.detach().to("cpu", dtype=torch.float32).numpy()


def linecloud(ax, pos, alpha=0.06, lw=1.0, rep=None, rep_lw=2.5, color_cloud=(0.6, 0.95, 1.0, None),
              color_rep=(0.4, 0.7, 1.0)):
    color_cloud = (color_cloud[0], color_cloud[1], color_cloud[2], alpha)

    segs_all = []
    if pos.ndim == 2:
        x_np = _to_cpu_np(pos)
        segs = np.stack([x_np[:-1], x_np[1:]], axis=1)  # (H-1,2,2)
        segs_all.append(segs)
        if rep is None:
            rep = pos
    else:
        N = pos.shape[0]
        for i in range(N):
            x_np = _to_cpu_np(pos[i])
            segs = np.stack([x_np[:-1], x_np[1:]], axis=1)
            segs_all.append(segs)

        if rep is None:
            rep = pos.mean(dim=0)

    segs_all = np.concatenate(segs_all, axis=0)  # (sum(H_i-1),2,2)
    lc = LineCollection(segs_all, colors=color_cloud, linewidths=lw)
    ax.add_collection(lc)
    rep_np = _to_cpu_np(rep)
    ax.plot(rep_np[:, 0], rep_np[:, 1], lw=rep_lw, color=color_rep)


def _render_one_as_array(env, pos, color, radius, start=None, goal=None, xylim=None,
                         fig_w=6, fig_h=4, dpi=100):
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    ax.axis("off")
    env.render(ax)
    linecloud(ax, pos, color_cloud=color, color_rep=color)
    from torch_robotics.torch_utils.torch_utils import to_numpy

    if start is not None:
        start_state_np = to_numpy(start)
        if len(start_state_np) == 3:
            ax.plot(start_state_np[0], start_state_np[1], start_state_np[2], 'go', markersize=radius * 100)
        else:
            ax.plot(start_state_np[0], start_state_np[1], 'go', markersize=radius * 100)
    if goal is not None:
        goal_state_np = to_numpy(goal)
        if len(goal_state_np) == 3:
            ax.plot(goal_state_np[0], goal_state_np[1], goal_state_np[2], marker='o',
                    color='purple', markersize=radius * 100)
        else:
            ax.plot(goal_state_np[0], goal_state_np[1], marker='o', color='purple', markersize=radius * 100)

    if xylim is not None:
        (xmin, xmax), (ymin, ymax) = xylim
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    ax.set_aspect("equal", adjustable="box")

    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba(), dtype=np.uint8)  # (H,W,4)
    rgb = buf[..., :3]
    plt.close(fig)
    return rgb


def _crop_center_50pct_height(img_np):
    h, w, c = img_np.shape
    top = int(0.3 * h)
    bottom = int(0.7 * h)
    return img_np[top:bottom, :, :].copy()  # (h/2, w, 3)


def plot_diffusion_cropped(
        env,
        trace_list,
        save_path,
        color,
        radius,
        picks=None,
        start=None,
        goal=None,
        xylim=None,
        fig_w=6,
        fig_h=4,
        dpi=100,
        divider_color="0.7",
        divider_lw=1.0
):
    S = len(trace_list)
    if picks is None:
        picks = [S // 10, S // 2, S - S // 10]

    crops = []
    for k in picks:
        arr = _render_one_as_array(env, trace_list[k], radius=radius,
                                   start=start, goal=goal,
                                   xylim=xylim, fig_w=fig_w, fig_h=fig_h, dpi=dpi, color=color)
        arr = _crop_center_50pct_height(arr)
        crops.append(arr)

    crops.reverse()

    heights = [c.shape[0] for c in crops]
    width = crops[0].shape[1]
    total_height = sum(heights)
    fig = plt.figure(figsize=(width / dpi, total_height / dpi), dpi=dpi)

    y_start = 0
    for crop in crops:
        h = crop.shape[0] / total_height
        ax = fig.add_axes([0, y_start, 1, h])
        y_start += h
        ax.imshow(crop)
        ax.axis("off")

    y_line1 = heights[0] / total_height
    y_line2 = (heights[0] + heights[1]) / total_height

    for y in [y_line1, y_line2]:
        line = mlines.Line2D(
            [0, 1], [y, y],
            color=divider_color,
            linewidth=divider_lw,
            transform=fig.transFigure,
            zorder=10
        )
        fig.add_artist(line)

    os.makedirs(Path(save_path).parent, exist_ok=True)
    if not save_path.endswith(".png"):
        save_path += ".png"

    plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.01)
    plt.close(fig)
    print(f"Saved combined figure to: file://{os.path.abspath(save_path)}")
    return save_path


def plot_diffusion(
        env,
        trace_list,
        save_path,
        color,
        radius,
        picks=None,
        start=None,
        goal=None,
        xylim=None,
):
    return plot_diffusion_cropped(
        env=env,
        trace_list=trace_list,
        color=color,
        radius=radius,
        save_path=save_path,
        picks=picks,
        start=start,
        goal=goal,
        xylim=xylim
    )
