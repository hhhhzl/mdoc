import numpy as np
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1
import argparse

from mp_baselines.planners.costs.cost_functions import CostCollision, CostComposite
from mp_baselines.planners.dynamics.point import PointParticleDynamics
from mp_baselines.planners.mppi import MPPI
from mp_baselines.planners.cem import CEM
from mp_baselines.planners.mbd import MDOC
from mp_baselines.planners.rrt_star import RRTStar
from mp_baselines.planners.rrt_connect import RRTConnect

from torch_robotics.environments import EnvUMaze2D, EnvRandom2DFixed, EnvSymBottleneck2D, EnvTennis2D, EnvEmpty2D
from torch_robotics.robots.robot_planar_disk import RobotPlanarDisk
from torch_robotics.tasks.tasks import PlanningTask
from torch_robotics.torch_utils.seed import fix_random_seed
from torch_robotics.torch_utils.torch_timer import TimerCUDA
from torch_robotics.torch_utils.torch_utils import get_torch_device
from mdoc.baselines.mb_benchmark.visualizer import plot_overlay, plot_paths, plot_diffusion, print_overlay


# ---------------------- collision sdf + cbf projection helpers ----------------------
@torch.no_grad()
def sdf_and_grad(field, q, eps=1e-3):
    device, dtype = q.device, q.dtype

    def sdf_only(qq):
        link_pos = qq.reshape(-1, 1, 2)  # (N,1,2)
        d = field.object_signed_distances(link_pos=link_pos)  # (N, num_sdfs, 1) or (N, num_sdfs)
        d = torch.as_tensor(d, device=device, dtype=dtype)
        if d.dim() >= 3:
            d = d.squeeze(-1)  # (N, num_sdfs)
        d = d.min(dim=-1).values
        return d.view(qq.shape[:-1])

    sdf0 = sdf_only(q)
    ex = torch.tensor([1.0, 0.0], device=device, dtype=dtype) * eps
    ey = torch.tensor([0.0, 1.0], device=device, dtype=dtype) * eps
    sdf_xp = sdf_only(q + ex)
    sdf_xm = sdf_only(q - ex)
    sdf_yp = sdf_only(q + ey)
    sdf_ym = sdf_only(q - ey)
    gx = (sdf_xp - sdf_xm) / (2 * eps)
    gy = (sdf_yp - sdf_ym) / (2 * eps)
    g = torch.stack([gx, gy], dim=-1)
    return sdf0, g


@torch.no_grad()
def cbf_project_controls(state0, controls, field, dt, d_safe=0.03, alpha=1.0, eps_g=1e-8):
    H = controls.shape[0]
    x = torch.empty((H + 1, 2), device=controls.device, dtype=controls.dtype)
    u_safe = controls.clone()
    x[0] = state0
    for t in range(H):
        sdf, grad = sdf_and_grad(field, x[t])
        h = sdf - d_safe
        a = grad
        a_norm2 = (a * a).sum().clamp_min(eps_g)
        b = - (alpha / dt) * h
        lhs = (a * u_safe[t]).sum()
        if lhs < b:
            u_safe[t] = u_safe[t] + ((b - lhs) / a_norm2) * a
        x[t + 1] = x[t] + dt * u_safe[t]
    return u_safe, x


allow_ops_in_compiled_graph()


def parse_args():
    parser = argparse.ArgumentParser(description='Single-agent planning configuration')
    parser.add_argument(
        '--instance_name',
        type=str,
        default='test',
        help='Name of the instance'
    )

    parser.add_argument(
        '--planner',
        nargs='+',
        default=['cem', 'mppi', 'rrt*', 'mdoc'],
        choices=['rrt*', 'mdoc', 'mppi', 'cem'],
        help='List of single-agent planners'
    )

    # Env
    parser.add_argument(
        '--e',
        type=str,
        default='Narrow',
        choices=['Narrow', 'UMaze', 'Random', "Tennis", "Empty"],
        help='env'
    )
    parser.add_argument(
        '--type',
        type=str,
        default='flat',
        choices=['flat', 'incline'],
        help='start && goal type'
    )

    parser.add_argument(
        '--plot',
        type=bool,
        default=True,
        help='whether to plot'
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    device = get_torch_device()
    tensor_args = {'device': 'cpu', 'dtype': torch.float32}
    seed = 42
    H = 128  # horizon
    dt = 0.04
    opt_iters_outer = 1
    plot = args.plot

    # --------------------------- build envs ------------------------------- #
    envs = {
        "Empty": EnvEmpty2D(
            precompute_sdf_obj_fixed=True,
            sdf_cell_size=0.01,
            tensor_args=tensor_args
        ),
        "Narrow": EnvSymBottleneck2D(
            corridor_width=0.42,
            wall_thickness=0.02,
            tensor_args=tensor_args,
            precompute_sdf_obj_fixed=True,
            sdf_cell_size=0.01,
        ),
        "UMaze": EnvUMaze2D(
            tensor_args=tensor_args,
            precompute_sdf_obj_fixed=True,
            sdf_cell_size=0.01,
            wall_thickness=0.20,
            wall_height=1.60,
            bottom_width=1.40,
            bottom_y=-0.60,
            left_x=-0.60,
            right_x=0.60,
        ),
        "Random": EnvRandom2DFixed(
            precompute_sdf_obj_fixed=True,
            sdf_cell_size=0.01,
            tensor_args=tensor_args,
        ),
        "Tennis": EnvTennis2D(
            corridor_width=0.42 if args.type == "flat" else 0.32,
            wall_thickness=0.02 if args.type == "flat" else 0.28,
            tensor_args=tensor_args,
            chair_to_center=0.4,
            chair_length=0.3,
            chair_width=0.1,
        )
    }
    env = envs[args.e]

    # goal
    y = 0.0 if args.type == "flat" else 0.80
    start_state = torch.tensor([-0.80, -y], **tensor_args)
    goal_state = torch.tensor([0.80, y], **tensor_args)

    # ---------------------------- Robot, PlanningTask ---------------------------------
    robot = RobotPlanarDisk(tensor_args=tensor_args, radius=0.05)
    robot.dt = dt
    task = PlanningTask(
        env=env,
        robot=robot,
        ws_limits=torch.tensor([[-0.95, -0.95], [0.95, 0.95]], **tensor_args),
        obstacle_cutoff_margin=0.05,
        tensor_args=tensor_args
    )

    # --------------------------- build planners ------------------------------- #
    colors = {}
    p = args.planner
    planners = {}
    for planner in p:
        if planner == "cem":
            cem_system_params = dict(
                rollout_steps=H,
                control_dim=robot.q_dim,  # 2
                state_dim=robot.q_dim,  # 2
                dt=dt,
                discount=0.95,
                goal_state=goal_state,
                ctrl_min=[-0.7, -0.7],
                ctrl_max=[0.7, 0.7],
                verbose=False,
                c_weights={
                    'pos': 2.2,
                    'vel': 0.,
                    'ctrl': 0.12,
                    'pos_T': 1800.,
                    'vel_T': 0.,
                },
                tensor_args=tensor_args,
            )
            cem_params = dict(
                num_ctrl_samples=2048,
                rollout_steps=H,
                opt_iters=4,
                elite_frac=0.25,
                min_elites=512,
                step_size=0.8,
                control_std=[0.05, 0.05],
                cov_prior_type='const_ctrl',
                tensor_args=tensor_args,
            )
            cem_system = PointParticleDynamics(**cem_system_params)
            planner = CEM(
                system=cem_system, **cem_params
            )
            planners['cem'] = planner
        elif planner == "mppi":
            mppi_system_params = dict(
                rollout_steps=H,
                control_dim=robot.q_dim,
                state_dim=robot.q_dim,
                dt=dt,
                discount=0.95,
                goal_state=goal_state,
                ctrl_min=[-0.7, -0.7],
                ctrl_max=[0.7, 0.7],
                verbose=False,
                c_weights={'pos': 1.8, 'vel': 0., 'ctrl': 0.10, 'pos_T': 1200., 'vel_T': 0., },
                tensor_args=tensor_args,
            )
            mppi_params = dict(
                num_ctrl_samples=2048,
                rollout_steps=H,
                control_std=[0.04, 0.04],
                temp=42,
                opt_iters=4,
                step_size=0.8,
                cov_prior_type='const_ctrl',
                tensor_args=tensor_args,
            )
            mppi_system = PointParticleDynamics(**mppi_system_params)
            planner = MPPI(mppi_system, **mppi_params)
            planners['mppi'] = planner
        elif planner == 'mdoc':
            planner = MDOC(
                robot=robot,
                env_model=env,
                start_state_pos=start_state,  # (2,)
                goal_state_pos=goal_state,  # (2,)
                rollout_steps=64,
                n_diffusion_steps=100 if args.type == "flat" else 200,
                n_samples=128,
                temp_sample=0.001,
                transforms=None,
                tensor_args=tensor_args,
            )
            planners['mdoc'] = planner
        elif planner == 'rrt*':
            planner = RRTConnect(
                task=task,
                n_iters=10000,
                start_state_pos=start_state,
                goal_state_pos=goal_state,
                n_iters_after_success=1500,  #
                max_best_cost_iters=4000,  #
                cost_eps=1e-3,
                step_size=0.03,
                n_radius=robot.radius,
                n_knn=15,
                max_time=20,
                goal_prob=0.20,
                tensor_args=tensor_args,
                n_pre_samples=50000,
                pre_samples=None,
                informed=False
            )
            planners['rrt*'] = planner

    # Construct cost function (for mppi && CEM)
    sigma_coll = 0.03
    cost_collisions_soft = []
    for collision_field in task.get_collision_fields():
        cost_collisions_soft.append(
            CostCollision(
                robot, H,
                field=collision_field,
                sigma_coll=sigma_coll,
                tensor_args=tensor_args
            )
        )
    cost_composite = CostComposite(robot, H, cost_collisions_soft, tensor_args=tensor_args)

    # Optimize
    observation = {
        'state': start_state,
        'goal_state': goal_state,
    }
    collision_fields = task.get_collision_fields()
    cbf_field = None
    for f in collision_fields:
        if f is not None:
            cbf_field = f
            break
    if cbf_field is None:
        raise RuntimeError("No collision field found for CBF projection.")

    import matplotlib.pyplot as plt

    colors = {
        'mdoc': (0.1216, 0.4667, 0.7059),  # #1f77b4
        'mppi': (1.0000, 0.4980, 0.0549),  # #ff7f0e
        'cem': (0.2078, 0.7176, 0.4745),  # #35B779
        'rrt*': (0.8392, 0.1529, 0.1569),  # #d95f02
    }

    # colors['mdoc'] = plt.cm.tab20(18)
    # # colors['mppi'] = plt.cm.tab20(6)
    # # colors['cem'] = plt.cm.tab20(4)
    # colors['mppi'] = (0.88, 0.40, 0.31)
    # colors['cem'] = (0.36, 0.72, 0.64)
    # colors['rrt*'] = plt.cm.tab20(15)

    samples = {}
    pass_stats = {}
    paths = {}
    with TimerCUDA() as t:
        for method in planners.keys():
            planner = planners[method]
            fix_random_seed(seed)
            time = t.elapsed
            if method in ['mdoc', 'mppi', 'cem']:
                vel_iters = torch.empty((opt_iters_outer, 1, H if method in ['mppi', 'cem'] else 64, planner.control_dim), **tensor_args)
            for i in range(opt_iters_outer):
                observation['cost'] = [] if method == 'mdoc' else cost_composite  # uses planner.opt_iters
                if method in ['mppi', 'cem']:
                    planner.optimize(**observation)
                    u_mean = planner.get_mean_controls()
                    u_proj, _ = cbf_project_controls(
                        state0=observation['state'],
                        controls=u_mean,
                        field=cbf_field,
                        dt=dt,
                        d_safe=0.03,
                        alpha=1.0
                    )
                    vel_iters[i, 0] = u_proj
                elif method == 'mdoc':
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
                            save_path=f'results/mb/{args.e.lower()}_{args.type}_diffusion'
                        )
                elif method == "rrt*":
                    path = planner.optimize(opt_iters=100, debug=True)
                    if (path is None) or (len(path) == 0):
                        print("[WARN] RRT* No Path find")
                    else:
                        if isinstance(path, torch.Tensor):
                            path_xy = path
                        else:
                            path_xy = torch.stack(path, dim=0)
                        path_xy = path_xy.to(**tensor_args).reshape(-1, 2)

                        pos_traj = path_xy  # (H,2)
                        pos_trajs_batch = pos_traj.unsqueeze(0).unsqueeze(0)
                        paths[method] = pos_trajs_batch[-1]

                    rrt_nodes = []
                    rrt_pass_total = 0.0
                    rrt_total_samples = 0
                    nodes_tree_1 = getattr(planner, "nodes_tree_1_torch", None)
                    nodes_tree_2 = getattr(planner, "nodes_tree_2_torch", None)
                    for node_tensor in (nodes_tree_1, nodes_tree_2):
                        if node_tensor is None:
                            continue
                        if node_tensor.dim() == 1:
                            node_tensor = node_tensor.unsqueeze(0)
                        node_pos = node_tensor[:, :2]
                        rrt_nodes.append(node_pos)
                        rrt_total_samples += node_pos.shape[0]
                        rrt_pass_total += float((node_pos[:, 0] > 0.0).sum().item())
                    if rrt_nodes:
                        all_rrt_nodes = torch.cat(rrt_nodes, dim=0)
                        samples[method] = all_rrt_nodes.detach().to("cpu", dtype=torch.float32).numpy()
                        if rrt_total_samples > 0:
                            pass_stats[method.lower()] = {
                                "pass_total": float(rrt_pass_total / rrt_total_samples * 100.0),
                                "n_total": rrt_total_samples,
                                "n_free": rrt_total_samples,
                                "pass_given_free": float((all_rrt_nodes[:, 0] > 0.0).float().mean().item() * 100.0),
                            }

                print(f'Optimization time for {method}: {t.elapsed - time:.3f} sec')

                if method in ['mdoc', 'mppi', 'cem']:
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
                    pos_iters = torch.empty((opt_iters_outer, 1, H if method in ['mppi', 'cem'] else 64, planner.state_dim), **tensor_args)
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
                        if method == 'cem':
                            best_cost, best_pos = torch.min(weights[success_mask].squeeze(-1), dim=0)
                            best_idx = success_indices[best_pos]
                        elif method == 'mppi':
                            best_weight, best_pos = torch.max(weights[success_mask], dim=0)
                            best_idx = success_indices[best_pos]
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
            save_path=f'results/mb/{args.e.lower()}_{args.type}_paths',
            path_dict=paths
        )
