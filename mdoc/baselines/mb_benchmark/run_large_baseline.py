import numpy as np
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
import argparse

from mp_baselines.planners.mbd import MDOC

from torch_robotics.environments.env_random_large_2d_fixed import EnvRandomLarge2DFixed
from torch_robotics.environments.env_random_extra_large_2d_fixed import EnvRandomExtraLarge2DFixed
from torch_robotics.robots.robot_planar_disk import RobotPlanarDisk
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device
from mdoc.baselines.mb_benchmark.visualizer import plot_overlay, plot_paths, plot_diffusion, print_overlay


allow_ops_in_compiled_graph()


def parse_args():
    parser = argparse.ArgumentParser(description='Large baseline planning configuration')
    parser.add_argument(
        '--instance_name',
        type=str,
        default='test',
        help='Name of the instance'
    )

    # Env
    parser.add_argument(
        '--e',
        type=str,
        default='RandomLarge',
        choices=['RandomLarge', 'RandomExtraLarge'],
        help='env: RandomLarge or RandomExtraLarge'
    )

    parser.add_argument(
        '--plot',
        type=bool,
        default=True,
        help='whether to plot'
    )

    parser.add_argument(
        '--n_diffusion_steps',
        type=int,
        default=100,
        help='number of diffusion steps'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = get_torch_device()
    tensor_args = {'device': 'cpu', 'dtype': torch.float32}
    seed = 0
    H = 128  # horizon
    dt = 0.067
    opt_iters_outer = 1
    plot = args.plot

    # --------------------------- build envs ------------------------------- #
    envs = {
        "RandomLarge": EnvRandomLarge2DFixed(
            precompute_sdf_obj_fixed=True,
            sdf_cell_size=0.01,
            tensor_args=tensor_args
        ),
        "RandomExtraLarge": EnvRandomExtraLarge2DFixed(
            precompute_sdf_obj_fixed=True,
            sdf_cell_size=0.01,
            tensor_args=tensor_args
        ),
    }
    env = envs[args.e]

    # ---------------------------- Robot, PlanningTask ---------------------------------
    robot = RobotPlanarDisk(tensor_args=tensor_args, radius=0.05)
    robot.dt = dt
    
    # Get workspace limits from environment
    ws_limits = env.limits.clone()
    task = PlanningTask(
        env=env,
        robot=robot,
        ws_limits=ws_limits,
        obstacle_cutoff_margin=0.05,
        tensor_args=tensor_args
    )

    # Random start and goal
    fix_random_seed(seed)
    n_tries = 1000
    start_state = None
    goal_state = None
    threshold_start_goal_pos = 0.5
    
    for _ in range(n_tries):
        q_free = task.random_coll_free_q(n_samples=2)
        start_state = q_free[0]
        goal_state = q_free[1]
        
        # Check if start and goal are valid
        if not env.is_start_goal_valid_for_data_gen(robot, start_state, goal_state):
            continue
        
        # Check minimum distance between start and goal
        if torch.linalg.norm(start_state - goal_state) > threshold_start_goal_pos:
            break
    
    if start_state is None or goal_state is None:
        raise ValueError(f"No collision free configuration was found\n"
                         f"start_state: {start_state}\n"
                         f"goal_state: {goal_state}\n")

    start_state = torch.tensor([-1.9, -1.9])
    goal_state = torch.tensor([1.9, 1.9])
    print(f'Start state: {start_state}')
    print(f'Goal state: {goal_state}')

    # --------------------------- build planner ------------------------------- #
    planner = MDOC(
        robot=robot,
        env_model=env,
        start_state_pos=start_state,  # (2,)
        goal_state_pos=goal_state,  # (2,)
        rollout_steps=H,
        n_diffusion_steps=args.n_diffusion_steps,
        n_samples=128,
        temp_sample=0.001,
        transforms=None,
        tensor_args=tensor_args,
    )

    import matplotlib.pyplot as plt

    colors = {
        'mdoc': (0.1216, 0.4667, 0.7059),  # #1f77b4
    }

    samples = {}
    pass_stats = {}
    paths = {}
    with TimerCUDA() as t:
        method = 'mdoc'
        fix_random_seed(seed)
        time = t.elapsed
        vel_iters = torch.empty((opt_iters_outer, 1, H, planner.control_dim), **tensor_args)
        
        for i in range(opt_iters_outer):
            observation = {
                'state': start_state,
                'goal_state': goal_state,
                'cost': []  # uses planner.opt_iters
            }
            planner.start_diffusion_trace()
            planner.optimize(**observation)
            vel_iters[i, 0] = planner.get_mean_controls()
            trace = planner.get_diffusion_trace()
            reprs = planner.get_diffusion_representatives()
            picks = [0, len(trace) // 2, len(trace) - 1]
            triplet = [trace[k] for k in picks]
            triplet_reps = [reprs[k] for k in picks]
            if plot:
                plot_diffusion(
                    env,
                    trace,
                    radius=robot.radius,
                    color=colors[method],
                    start=start_state.cpu().numpy().tolist(),
                    goal=goal_state.cpu().numpy().tolist(),
                    xylim=None,
                    save_path=f'results/mb/{args.e.lower()}_diffusion'
                )

        print(f'Optimization time for {method}: {t.elapsed - time:.3f} sec')

        ctrl_samples, state_trajs, weights = planner.get_recent_samples()
        collision_mask = task.compute_collision(state_trajs, margin=robot.radius)
        if collision_mask.ndim >= 2:
            collision_mask_flat = collision_mask.reshape(collision_mask.shape[0], -1).any(dim=1)
        else:
            collision_mask_flat = collision_mask.bool()
        valid_mask = torch.logical_not(collision_mask_flat)
        n_total = int(state_trajs.shape[0])
        n_free = int(valid_mask.sum().item())
        pass_mask = state_trajs[:, -1, 0] > 0.0
        pass_and_free = torch.logical_and(valid_mask, pass_mask).sum().item()
        if n_total > 0:
            pass_rate_total = float(pass_and_free / n_total * 100.0)
        else:
            pass_rate_total = 0.0
        state_trajs_filtered = state_trajs[valid_mask]
        if state_trajs_filtered.nelement() == 0:
            endpoints = np.empty((0, 2), dtype=np.float32)
        else:
            endpoints = state_trajs_filtered[:, -1, :2].detach().to("cpu", dtype=torch.float32).numpy()
        samples[method] = endpoints
        if n_free > 0:
            pass_given_free = float(pass_mask[valid_mask].sum().item() / n_free * 100.0)
        else:
            pass_given_free = 0.0
        pass_stats[method.lower()] = {
            "pass_total": pass_rate_total,
            "n_total": n_total,
            "n_free": n_free,
            "pass_given_free": pass_given_free,
        }
        pos_iters = torch.empty((opt_iters_outer, 1, H, planner.state_dim), **tensor_args)
        for i in range(opt_iters_outer):
            pos_trajs = planner.get_state_trajectories_rollout(
                controls=vel_iters[i, 0].unsqueeze(0), **observation
            ).squeeze()
            pos_iters[i, 0] = pos_trajs
        trajs_iters = torch.cat((pos_iters, vel_iters), dim=-1)
        pos_trajs_iters = robot.get_position(trajs_iters)

        success_mask = torch.logical_and(valid_mask, pass_mask)
        if success_mask.any():
            success_indices = success_mask.nonzero(as_tuple=False).squeeze(-1)
            best_idx = success_indices[0]
            best_traj_state = state_trajs[best_idx:best_idx + 1]
            best_traj_pos = robot.get_position(best_traj_state)
            paths[method] = best_traj_pos
        else:
            paths[method] = pos_trajs_iters[-1]

    ws = task.ws_limits.cpu().numpy()
    ws_limits = (ws[0].tolist(), ws[1].tolist())  # ((xmin, ymin), (xmax, ymax))

    print_overlay(endpoints_dict=samples, stats_dict=pass_stats)
    if plot:
        _ = plot_paths(
            task=task,
            colors=colors,
            env=env,
            start=start_state.cpu().numpy().tolist(),
            goal=goal_state.cpu().numpy().tolist(),
            save_path=f'results/mb/{args.e.lower()}_paths',
            path_dict=paths
        )

