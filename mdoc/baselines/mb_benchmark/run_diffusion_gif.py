"""
Render the full reverse-diffusion trajectory of the MDOC planner into a GIF.

This script mirrors the environment setup in `run_mb_baselines.py`, but runs ONLY the
MDOC planner and saves one frame per diffusion step.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import torch

# Headless-safe rendering
import matplotlib

matplotlib.use("Agg")

from PIL import Image, ImageDraw

from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

from mp_baselines.planners.mbd import MDOC

from torch_robotics.environments import EnvRandom2DFixed, EnvSymBottleneck2D, EnvTennis2D, EnvEmpty2D, EnvConveyor2D, EnvDropRegion2D
from torch_robotics.robots.robot_planar_disk import RobotPlanarDisk
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_utils import get_torch_device

from mdoc.baselines.mb_benchmark.visualizer import _render_one_as_array

allow_ops_in_compiled_graph()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render MDOC diffusion steps as GIF")
    parser.add_argument("--instance_name", type=str, default="test", help="Name of the instance")

    # Env (same as run_mb_baselines.py)
    parser.add_argument(
        "--e",
        type=str,
        default="Narrow",
        choices=["Narrow", "Random", "Tennis", "Empty", "Conveyor", "DropRegion"],
        help="Environment name",
    )
    parser.add_argument(
        "--type",
        type=str,
        default="flat",
        choices=["flat", "incline"],
        help="Start/goal type",
    )

    # Diffusion + rendering
    parser.add_argument("--n_diffusion_steps", type=int, default=100, help="Number of diffusion steps (frames)")
    parser.add_argument("--n_samples", type=int, default=64, help="Number of diffusion samples per step")
    parser.add_argument("--fps", type=int, default=20, help="GIF frames per second")
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="Output GIF path. Default: results/mb/<env>_<type>_diffusion.gif",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    # GIF file size (smaller = lower quality/size)
    parser.add_argument("--dpi", type=int, default=80, help="Figure DPI for each frame")
    parser.add_argument("--gif_colors", type=int, default=256, help="GIF palette size")
    parser.add_argument("--optimize_gif", action="store_true", default=True, help="Optimize GIF when saving (default: True)")
    parser.add_argument("--no_optimize_gif", action="store_false", dest="optimize_gif", help="Disable GIF optimization")
    parser.add_argument("--every", type=int, default=1, help="Save every Nth frame only")
    return parser.parse_args()


def _as_uint8_rgb(arr: np.ndarray) -> np.ndarray:
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected RGB uint8 image (H,W,3), got {arr.shape} {arr.dtype}")
    return arr


def _square_xylim_from_limits(limits: torch.Tensor, pad: float = 0.0):
    """Return ((xmin,xmax),(ymin,ymax)) as the smallest square covering `limits`."""
    lim = limits.detach().to("cpu", dtype=torch.float32)
    xmin, ymin = float(lim[0, 0]), float(lim[0, 1])
    xmax, ymax = float(lim[1, 0]), float(lim[1, 1])

    dx = xmax - xmin
    dy = ymax - ymin
    side = max(dx, dy) + float(pad) * 2.0
    cx = 0.5 * (xmin + xmax)
    cy = 0.5 * (ymin + ymax)
    half = 0.5 * side
    return (cx - half, cx + half), (cy - half, cy + half)


def main() -> None:
    args = parse_args()

    _ = get_torch_device()  # kept for parity with other scripts; tensor_args below uses CPU
    tensor_args = {"device": "cpu", "dtype": torch.float32}

    # Planning params (match run_mb_baselines.py defaults)
    dt = 0.05
    rollout_steps = 64
    temp_sample = 0.1

    # --------------------------- build env ------------------------------- #
    envs = {
        "Empty": EnvEmpty2D(precompute_sdf_obj_fixed=True, sdf_cell_size=0.01, tensor_args=tensor_args),
        "Narrow": EnvSymBottleneck2D(
            corridor_width=0.42,
            wall_thickness=0.02,
            tensor_args=tensor_args,
            precompute_sdf_obj_fixed=True,
            sdf_cell_size=0.01,
        ),
        "Random": EnvRandom2DFixed(precompute_sdf_obj_fixed=True, sdf_cell_size=0.01, tensor_args=tensor_args),
        "Tennis": EnvTennis2D(
            corridor_width=0.42 if args.type == "flat" else 0.32,
            wall_thickness=0.02 if args.type == "flat" else 0.28,
            tensor_args=tensor_args,
            chair_to_center=0.4,
            chair_length=0.3,
            chair_width=0.1,
        ),
        "Conveyor": EnvConveyor2D(precompute_sdf_obj_fixed=True, sdf_cell_size=0.01, tensor_args=tensor_args),
        "DropRegion": EnvDropRegion2D(precompute_sdf_obj_fixed=True, sdf_cell_size=0.01, tensor_args=tensor_args),
    }
    env = envs[args.e]

    y = 0.0 if args.type == "flat" else 0.80
    start_state = torch.tensor([-0.80, -y], **tensor_args)
    goal_state = torch.tensor([0.80, y], **tensor_args)

    # ---------------------------- Robot, PlanningTask ---------------------------------
    robot = RobotPlanarDisk(tensor_args=tensor_args, radius=0.05)
    robot.dt = dt
    ws_limits = getattr(env, "limits", torch.tensor([[-0.95, -0.95], [0.95, 0.95]], **tensor_args))
    task = PlanningTask(
        env=env,
        robot=robot,
        ws_limits=ws_limits,
        obstacle_cutoff_margin=0.05,
        tensor_args=tensor_args,
    )
    _ = task

    # --------------------------- build MDOC ------------------------------- #
    planner = MDOC(
        robot=robot,
        env_model=env,
        start_state_pos=start_state,
        goal_state_pos=goal_state,
        rollout_steps=rollout_steps,
        n_diffusion_steps=args.n_diffusion_steps,
        n_samples=args.n_samples,
        temp_sample=temp_sample,
        transforms=None,
        tensor_args=tensor_args,
    )

    fix_random_seed(args.seed)
    observation = {
        "state": start_state,
        "goal_state": goal_state,
        "cost": [],  # MDOC signature matches run_mb_baselines.py
    }

    planner.start_diffusion_trace()
    planner.optimize(**observation)
    trace = planner.get_diffusion_trace()  # list length == n_diffusion_steps

    if not trace:
        raise RuntimeError("MDOC diffusion trace is empty. Did optimize() run correctly?")

    # --------------------------- render frames ------------------------------- #
    out_path = args.out.strip()
    if not out_path:
        out_path = f"results/mb/{args.e.lower()}_{args.type}_diffusion.gif"
    out_path = os.path.abspath(out_path)
    os.makedirs(Path(out_path).parent, exist_ok=True)

    color = (0.1216, 0.4667, 0.7059)  # same as mb_benchmark scripts
    frames: list[Image.Image] = []
    xylim = _square_xylim_from_limits(ws_limits, pad=0.0)
    fig_size = 6  # square canvas
    every = max(1, int(args.every))
    indices = list(range(0, len(trace), every))  # 0, every, 2*every, ...

    for k, pos in enumerate(trace):
        if k not in indices:
            continue
        rgb = _render_one_as_array(
            env=env,
            pos=pos,
            color=color,
            radius=robot.radius,
            start=start_state.cpu().numpy().tolist(),
            goal=goal_state.cpu().numpy().tolist(),
            xylim=xylim,
            fig_w=fig_size,
            fig_h=fig_size,
            dpi=args.dpi,
        )
        rgb = _as_uint8_rgb(rgb)
        img = Image.fromarray(rgb)

        # MDOC stores trace from t=n..1, so show 100/100 -> 1/100
        step_idx = len(trace) - k
        draw = ImageDraw.Draw(img)
        draw.text((8, 8), f"diffusion {step_idx}/{len(trace)}", fill=(0, 0, 0))
        draw.text((7, 7), f"diffusion {step_idx}/{len(trace)}", fill=(255, 255, 255))

        frames.append(img.convert("RGB"))

    # Quantize to limited palette for smaller GIF (shared palette across frames when possible)
    n_colors = min(256, max(2, args.gif_colors))
    try:
        method = getattr(Image, "Quantize", None)
        kw = {"colors": n_colors}
        if method is not None:
            kw["method"] = method.MEDIANCUT
        first_p = frames[0].quantize(**kw)
        out_frames = [first_p]
        for f in frames[1:]:
            try:
                out_frames.append(f.quantize(palette=first_p))
            except TypeError:
                out_frames.append(f.quantize(**kw))
    except (AttributeError, TypeError):
        out_frames = [f.quantize(colors=n_colors) for f in frames]

    frame_duration_ms = int(round(1000.0 / max(1, args.fps)))
    if every > 1:
        frame_duration_ms = frame_duration_ms * every  # keep total animation time similar
    out_frames[0].save(
        out_path,
        save_all=True,
        append_images=out_frames[1:],
        duration=frame_duration_ms,
        loop=0,
        optimize=args.optimize_gif,
    )

    print(f"Saved diffusion GIF to: file://{out_path}")


if __name__ == "__main__":
    main()

