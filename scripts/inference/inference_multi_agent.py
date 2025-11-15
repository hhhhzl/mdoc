"""
MIT License
"""
# Standard imports.

import os
from datetime import datetime
import argparse
import time
import torch
from einops._torch_specific import allow_ops_in_compiled_graph  # requires einops>=0.6.1

# Project imports.
from torch_robotics.torch_utils.torch_utils import get_torch_device
from torch_robotics.trajectory.metrics import compute_path_length_from_pos, compute_smoothness_metric
from torch_robotics.environments import *
from mdoc.planners.multi_agent import CBS, PrioritizedPlanning
from mdoc.common.constraints import MultiPointConstraint, VertexConstraint, EdgeConstraint
from mdoc.common.conflicts import VertexConflict, PointConflict, EdgeConflict
from mdoc.common.trajectory_utils import smooth_trajs, densify_trajs
from mdoc.common import get_start_goal_pos_circle, get_start_goal_pos_boundary, get_start_goal_pos_random_in_env
from mdoc.common.pretty_print import *
from mdoc.common.experiments import MultiAgentPlanningSingleTrialConfig, MultiAgentPlanningSingleTrialResult, get_result_dir_from_trial_config, TrialSuccessStatus
from mdoc.planners.single_agent import (
    MDOCEnsemble,
    MPD,
    MPDEnsemble,
    WAStar,
    KCBSLower,
)

TRAINED_MODELS_DIR = './data_trained_models/'
allow_ops_in_compiled_graph()

device = get_torch_device('cpu')
print(f">>>>>>>>> Using {str(device).upper()} <<<<<<<<<<<<<<<")
tensor_args = {'device': device, 'dtype': torch.float32}


def run_multi_agent_trial(test_config: MultiAgentPlanningSingleTrialConfig):
    # ============================
    # Start time per agent.
    # ============================
    start_time_l = [i * test_config.stagger_start_time_dt for i in range(test_config.num_agents)]
    # ============================
    # Arguments for the high/low level planner.
    # ============================
    if test_config.single_agent_planner_class in ['MDOCEnsemble']:
        try:
            envs_name = test_config.global_model_ids[0][0]
        except:
            envs_name = ''
            NotImplementedError("Invalid Env Name")

        if "empty" in envs_name.lower():
            from mdoc.config.empty.mdoc_params import MDOCParams as params
        elif 'conveyor' in envs_name.lower():
            from mdoc.config.conveyor.mdoc_params import MDOCParams as params
        elif 'random' in envs_name.lower():
            from mdoc.config.random.mdoc_params import MDOCParams as params
        else:
            # general
            from mdoc.config.mdoc_params import MDOCParams as params
        planner_alg = 'mdoc'
    elif test_config.single_agent_planner_class in ['MMDEnsemble', "MMD"]:
        from mdoc.config.mmd_params import MMDParams as params
        planner_alg = 'mmd'
    elif test_config.single_agent_planner_class in ['WAStar', 'WAStarData']:
        from mdoc.config.wastar_params import WASTARParams as params
        planner_alg = 'wastar'
    elif test_config.single_agent_planner_class == 'KCBSLower':
        from mdoc.config.kcbs_params import KCBSParams as params
        planner_alg = 'kcbs'
    else:
        raise ValueError(f'Unknown single agent planner class: {test_config.single_agent_planner_class}')

    low_level_planner_model_args = {
        'planner_alg': planner_alg,
        'use_guide_on_extra_objects_only': params.use_guide_on_extra_objects_only,
        'n_samples': params.n_samples,
        'n_local_inference_noising_steps': params.n_local_inference_noising_steps,
        'n_local_inference_denoising_steps': params.n_local_inference_denoising_steps,
        'start_guide_steps_fraction': params.start_guide_steps_fraction,
        'n_guide_steps': params.n_guide_steps,
        'n_diffusion_steps_without_noise': params.n_diffusion_steps_without_noise,
        'weight_grad_cost_collision': params.weight_grad_cost_collision,
        'weight_grad_cost_smoothness': params.weight_grad_cost_smoothness,
        'weight_grad_cost_constraints': params.weight_grad_cost_constraints,
        'weight_grad_cost_soft_constraints': params.weight_grad_cost_soft_constraints,
        'factor_num_interpolated_points_for_collision': params.factor_num_interpolated_points_for_collision,
        'trajectory_duration': params.trajectory_duration,
        'device': params.device,
        'debug': params.debug,
        'seed': test_config.seed if test_config.run_experiment else params.seed,
        'results_dir': params.results_dir,
        'trained_models_dir': TRAINED_MODELS_DIR,
    }
    if test_config.single_agent_planner_class in ['MDOCEnsemble']:
        low_level_planner_model_args['n_diffusion_steps'] = params.n_diffusion_steps

    high_level_planner_model_args = {
        'is_xcbs': True if test_config.multi_agent_planner_class in ["XECBS", "XCBS"] else False,
        'is_ecbs': True if test_config.multi_agent_planner_class in ["ECBS", "XECBS"] else False,
        'start_time_l': start_time_l,
        'runtime_limit': test_config.runtime_limit,
        'conflict_type_to_constraint_types': {VertexConflict: {VertexConstraint}, EdgeConflict: {EdgeConstraint}} if
        "WAStar" in test_config.single_agent_planner_class else {PointConflict: {MultiPointConstraint}},
    }

    # ============================
    # Create a results directory.
    # ============================
    results_dir = get_result_dir_from_trial_config(test_config, test_config.time_str, test_config.trial_number)
    os.makedirs(results_dir, exist_ok=True)
    num_agents = test_config.num_agents

    # ============================
    # Get planning problem.
    # ============================
    # If want to get random starts and goals, then must do that after creating the reference task and robot.
    start_l = test_config.start_state_pos_l
    goal_l = test_config.goal_state_pos_l
    global_model_ids = test_config.global_model_ids
    agent_skeleton_l = test_config.agent_skeleton_l

    # ============================
    # Transforms and model tiles setup.
    # ============================
    # Create a reference planner from which we'll use the task and robot as the reference on in CBS.
    # Those are used for collision checking and visualization. This has a skeleton of all tiles.
    reference_agent_skeleton = [[r, c] for r in range(len(global_model_ids))
                                for c in range(len(global_model_ids[0]))]

    # ============================
    # Transforms from tiles to global frame.
    # ============================
    tile_width = 2.0
    tile_height = 2.0
    global_model_transforms = [[torch.tensor([x * tile_width, -y * tile_height], **tensor_args)
                                for x in range(len(global_model_ids[0]))] for y in range(len(global_model_ids))]

    # ============================
    # Parse the single agent planner class name.
    # ============================
    if test_config.single_agent_planner_class == "MMD":
        low_level_planner_class = MPD
    elif test_config.single_agent_planner_class == "MMDEnsemble":
        low_level_planner_class = MPDEnsemble
    elif test_config.single_agent_planner_class == "MDOCEnsemble":
        low_level_planner_class = MDOCEnsemble

    # baselines
    elif "WAStar" in test_config.single_agent_planner_class:
        low_level_planner_class = WAStar
    elif "KCBSLower" in test_config.single_agent_planner_class:
        low_level_planner_class = KCBSLower
    else:
        raise ValueError(f'Unknown single agent planner class: {test_config.single_agent_planner_class}')

    # ============================
    # Create reference agent planner.
    # ============================
    # And for the reference skeleton.
    reference_task = None
    reference_robot = None
    reference_agent_transforms = {}
    reference_agent_model_ids = {}
    for skeleton_step in range(len(reference_agent_skeleton)):
        skeleton_model_coord = reference_agent_skeleton[skeleton_step]
        reference_agent_transforms[skeleton_step] = global_model_transforms[skeleton_model_coord[0]][
            skeleton_model_coord[1]]
        reference_agent_model_ids[skeleton_step] = global_model_ids[skeleton_model_coord[0]][
            skeleton_model_coord[1]]
    reference_agent_model_ids = [reference_agent_model_ids[i] for i in range(len(reference_agent_model_ids))]
    # Create the reference low level planner.
    print("Creating reference agent stuff.")
    low_level_planner_model_args['start_state_pos'] = torch.tensor([0.5, 0.9], **tensor_args)  # This does not matter.
    low_level_planner_model_args['goal_state_pos'] = torch.tensor([-0.5, 0.9], **tensor_args)  # This does not matter.
    low_level_planner_model_args['model_ids'] = reference_agent_model_ids  # This matters.
    low_level_planner_model_args['transforms'] = reference_agent_transforms  # This matters.

    if test_config.single_agent_planner_class in ["MPD", "MDOC"]:
        low_level_planner_model_args['model_id'] = reference_agent_model_ids[0]

    # baselines
    if "WAStar" in test_config.single_agent_planner_class:
        low_level_planner_model_args['delta_q_action_l'] = params.wastar_delta_q_action_l
        low_level_planner_model_args['discretization'] = torch.tensor(params.wastar_discretization, **tensor_args)
    if test_config.single_agent_planner_class == "WAStarData":
        low_level_planner_model_args['is_use_data_cost'] = True

    reference_low_level_planner = low_level_planner_class(**low_level_planner_model_args)
    reference_task = reference_low_level_planner.task
    reference_robot = reference_low_level_planner.robot

    # ============================
    # Run trial.
    # ============================
    exp_name = f"{low_level_planner_model_args['planner_alg']}_single_trial"

    # Transform starts and goals to the global frame. Right now they are in the local tile frames.
    start_l = [start_l[i] + global_model_transforms[agent_skeleton_l[i][0][0]][agent_skeleton_l[i][0][1]]
               for i in range(num_agents)]
    goal_l = [goal_l[i] + global_model_transforms[agent_skeleton_l[i][-1][0]][agent_skeleton_l[i][-1][1]]
              for i in range(num_agents)]

    # ============================
    # Create global transforms for each agent's skeleton.
    # ============================
    # Each agent has a dict entry. Each entry is a dict with the skeleton steps (0, 1, 2, ...), mapping to the
    # model transform.
    agent_model_transforms_l = []
    agent_model_ids_l = []
    for agent_id in range(num_agents):
        agent_model_transforms = {}
        agent_model_ids = {}
        for skeleton_step in range(len(agent_skeleton_l[agent_id])):
            skeleton_model_coord = agent_skeleton_l[agent_id][skeleton_step]
            agent_model_transforms[skeleton_step] = global_model_transforms[skeleton_model_coord[0]][
                skeleton_model_coord[1]]
            agent_model_ids[skeleton_step] = global_model_ids[skeleton_model_coord[0]][skeleton_model_coord[1]]
        agent_model_transforms_l.append(agent_model_transforms)
        agent_model_ids_l.append(agent_model_ids)
    # Change the dict of the model ids to a list sorted by the skeleton steps.
    agent_model_ids_l = [[agent_model_ids_l[i][j] for j in range(len(agent_model_ids_l[i]))] for i in
                         range(num_agents)]

    # ============================
    # Create the low level planners.
    # ============================
    planners_creation_start_time = time.time()
    low_level_planner_l = []
    for i in range(num_agents):
        low_level_planner_model_args_i = low_level_planner_model_args.copy()
        low_level_planner_model_args_i['start_state_pos'] = start_l[i]
        low_level_planner_model_args_i['goal_state_pos'] = goal_l[i]
        low_level_planner_model_args_i['model_ids'] = agent_model_ids_l[i]
        low_level_planner_model_args_i['transforms'] = agent_model_transforms_l[i]
        if test_config.single_agent_planner_class == "MPD":
            # Set the model_id to the first one.
            low_level_planner_model_args_i['model_id'] = agent_model_ids_l[i][0]

        # baselines
        if "WAStar" in test_config.single_agent_planner_class:
            low_level_planner_model_args['delta_q_action_l'] = params.wastar_delta_q_action_l
            low_level_planner_model_args['discretization'] = torch.tensor(params.wastar_discretization, **tensor_args)
        if test_config.single_agent_planner_class == "WAStarData":
            low_level_planner_model_args['is_use_data_cost'] = True
        low_level_planner_l.append(low_level_planner_class(**low_level_planner_model_args_i))
    print('Planners creation time:', time.time() - planners_creation_start_time)
    print("\n\n\n\n")

    # ============================
    # Create the multi agent planner.
    # ============================
    if (test_config.multi_agent_planner_class in ["XECBS", "ECBS", "XCBS", "CBS"]):
        multi_agent_planner_class = CBS
    elif test_config.multi_agent_planner_class == "PP":
        multi_agent_planner_class = PrioritizedPlanning
    else:
        raise ValueError(f'Unknown multi agent planner class: {test_config.multi_agent_planner_class}')
    planner = multi_agent_planner_class(low_level_planner_l,
                                        start_l,
                                        goal_l,
                                        reference_task=reference_task,
                                        reference_robot=reference_robot,
                                        **high_level_planner_model_args)
    # ============================
    # Plan.
    # ============================
    startt = time.time()
    paths_l, num_ct_expansions, trial_success_status, num_collisions_in_solution = \
        planner.plan(runtime_limit=test_config.runtime_limit)
    planning_time = time.time() - startt
    # Print planning times.
    print(GREEN, 'Planning times:', planning_time, RESET)

    # ============================
    # Gather stats.
    # ============================
    single_trial_result = MultiAgentPlanningSingleTrialResult()
    # The associated experiment config.
    single_trial_result.trial_config = test_config
    # The planning problem.
    single_trial_result.start_state_pos_l = [start_l[i].cpu().numpy().tolist() for i in range(num_agents)]
    single_trial_result.goal_state_pos_l = [goal_l[i].cpu().numpy().tolist() for i in range(num_agents)]
    single_trial_result.global_model_ids = global_model_ids
    single_trial_result.agent_skeleton_l = agent_skeleton_l
    # The agent paths. Each entry is of shape (H, 4).
    single_trial_result.agent_path_l = paths_l
    # Success.
    single_trial_result.success_status = trial_success_status
    # Number of collisions in the solution.
    single_trial_result.num_collisions_in_solution = num_collisions_in_solution
    # Planning time.
    single_trial_result.planning_time = planning_time

    # Number of agent pairs in collision.
    if len(paths_l) > 0 and trial_success_status:
        # This assumes all paths in the solution are of the same length.
        for t in range(len(paths_l[0])):
            for i in range(num_agents):
                for j in range(i + 1, num_agents):
                    if torch.norm(paths_l[i][t, :2] - paths_l[j][t, :2]) < 2.0 * params.robot_planar_disk_radius:
                        # The above should be reference_robot.radius.
                        print(RED, 'Collision in solution:', i, j, t, paths_l[i][t, :2], paths_l[j][t, :2], RESET)
                        single_trial_result.num_collisions_in_solution += 1
        if single_trial_result.num_collisions_in_solution > 0:
            single_trial_result.success_status = TrialSuccessStatus.FAIL_COLLISION_AGENTS

    # If not successful, return here.
    if trial_success_status:
        # Our metric for determining how well a path is adhering to the data.
        # Computed by the environment. If it is a single map, the score is the adherence there.
        # If it is a multi-tile map, the score is the average adherence over all tiles.
        single_trial_result.data_adherence = 0.0
        for agent_id in range(num_agents):
            agent_data_adherence = 0.0
            for skeleton_step, agent_model_id in enumerate(agent_model_ids_l[agent_id]):
                agent_model_transform = agent_model_transforms_l[agent_id][skeleton_step]
                agent_start_time = start_time_l[agent_id]
                single_tile_traj_len = params.horizon
                agent_path_in_model_frame = (paths_l[agent_id].clone()[
                                             agent_start_time + skeleton_step * single_tile_traj_len:
                                             agent_start_time + (skeleton_step + 1) * single_tile_traj_len, :2] -
                                             agent_model_transform)
                model_env_name = agent_model_id.split('-')[0]
                kwargs = {'tensor_args': tensor_args}
                env_object = eval(model_env_name)(**kwargs)
                agent_data_adherence += env_object.compute_traj_data_adherence(agent_path_in_model_frame)
            agent_data_adherence /= len(agent_model_ids_l[agent_id])
            single_trial_result.data_adherence += agent_data_adherence
        single_trial_result.data_adherence /= num_agents
        # CT nodes expanded.
        single_trial_result.num_ct_expansions = num_ct_expansions
        # Path length. Hack for experiments.
        single_trial_result.path_length_per_agent = 0.0
        for agent_id in range(num_agents):
            # agent_path_pos = low_level_planner_l[agent_id].robot.get_position(paths_l[agent_id]).unsqueeze(0)
            agent_path_pos = paths_l[agent_id][:, :2].unsqueeze(0)
            single_trial_result.path_length_per_agent += compute_path_length_from_pos(agent_path_pos).item()
        single_trial_result.path_length_per_agent /= num_agents
        # Path smoothness.
        single_trial_result.mean_path_acceleration_per_agent = 0.0
        for agent_id in range(num_agents):
            # agent_path_pos = low_level_planner_l[agent_id].robot.get_position(paths_l[agent_id]).unsqueeze(0)
            # agent_path_vel = low_level_planner_l[agent_id].robot.get_velocity(paths_l[agent_id]).unsqueeze(0)
            agent_path_pos = paths_l[agent_id][:, :2].unsqueeze(0)
            agent_path_vel = paths_l[agent_id][:, 2:].unsqueeze(0)
            if agent_path_vel.shape[-1] == 0:
                agent_path_vel = low_level_planner_l[agent_id].robot.get_velocity(paths_l[agent_id]).unsqueeze(0)

            single_trial_result.mean_path_acceleration_per_agent += (
                compute_smoothness_metric(agent_path_pos, agent_path_vel, metric="avg_acc").item())

            single_trial_result.geometric_smoothness += (
                compute_smoothness_metric(agent_path_pos, agent_path_vel, metric="laplacian").item())

            single_trial_result.mean_jerk += (
                compute_smoothness_metric(agent_path_pos, agent_path_vel, metric="avg_jerk").item())

        single_trial_result.mean_path_acceleration_per_agent /= num_agents
        single_trial_result.geometric_smoothness /= num_agents
        single_trial_result.mean_jerk /= num_agents
    # ============================
    # Save the results and config.
    # ============================
    print(GREEN, single_trial_result, RESET)
    results_dir_uri = f'file://{os.path.abspath(results_dir)}'
    print('Results dir:', results_dir_uri)
    single_trial_result.save(results_dir)
    test_config.save(results_dir)

    # ============================
    # Render.
    # ============================
    if trial_success_status and len(paths_l) > 0:
        smooth_path_l = smooth_trajs(paths_l, device=params.device)
        planner.render_paths(paths_l,
                             output_fpath=os.path.join(results_dir, f'{exp_name}.gif'),
                             animation_duration=0,
                             plot_trajs=True,
                             show_robot_in_image=True)
        planner.render_paths(smooth_path_l,
                             output_fpath=os.path.join(results_dir, f'{exp_name}_smoothed.gif'),
                             animation_duration=0,
                             plot_trajs=True,
                             show_robot_in_image=True)
        if test_config.render_animation:
            paths_l = densify_trajs(paths_l,
                                    1)  # <------ Larger numbers produce nicer animations. But take longer to make too.
            smooth_path_l = densify_trajs(smooth_path_l, 1)
            planner.render_paths(paths_l,
                                 output_fpath=os.path.join(results_dir, f'{exp_name}.gif'),
                                 plot_trajs=True,
                                 animation_duration=10)
            planner.render_paths(smooth_path_l,
                                 output_fpath=os.path.join(results_dir, f'{exp_name}_smoothed.gif'),
                                 plot_trajs=True,
                                 animation_duration=10)


def parse_args():
    parser = argparse.ArgumentParser(description='Multi-agent planning configuration')

    # General arguments
    parser.add_argument(
        '--example_type',
        type=str,
        default='single_tile',
        choices=['single_tile', 'multi_tile'],
        help='Type of example to run'
    )
    parser.add_argument(
        '--n',
        type=int,
        default=3,
        help='Number of agents'
    )
    parser.add_argument(
        '--instance_name',
        type=str,
        default='test',
        help='Name of the instance'
    )
    parser.add_argument(
        '--hp',
        type=str,
        default='CBS',
        choices=[
            'CBS',
            'ECBS',
            'PP',
            'XCBS',
            'XECBS'
        ],
        help='Multi-agent planner class'
    )
    parser.add_argument(
        '--lp',
        type=str,
        default='KCBSLower',
        choices=[
            'MDOCEnsemble',
            'MMDEnsemble',
            'MMD',
            'WAStar',
            'KCBSLower',
            'WAStarData'
        ],
        help='Single agent planner class'
    )
    parser.add_argument(
        '--st',
        type=float,
        default=0,
        help='Stagger start time delta'
    )
    parser.add_argument(
        '--rl',
        type=float,
        default=1000,
        help='Runtime limit in seconds'
    )
    parser.add_argument(
        '--ra',
        action='store_true',
        default=True,
        help='Render animation'
    )

    # Single tile arguments
    parser.add_argument(
        '--e',
        type=str,
        default='EnvEmpty2D-RobotPlanarDisk',
        choices=[
            'EnvEmpty2D-RobotPlanarDisk',
            'EnvEmptyNoWait2D-RobotPlanarDisk',
            'EnvConveyor2D-RobotPlanarDisk',
            'EnvHighways2D-RobotPlanarDisk',
            'EnvDropRegion2D-RobotPlanarDisk',
            'EnvRandom2D-RobotPlanarDisk',
            'EnvRandomDense2D-RobotPlanarDisk',
            'EnvRandomLarge2D-RobotPlanarDisk',
            'EnvRandomExtraLarge2D-RobotPlanarDisk'
        ],
        help='Global model ID for single tile')

    # Multi tile arguments
    parser.add_argument(
        '--me',
        nargs='+',
        default=[
            'EnvEmptyNoWait2D-RobotPlanarDisk',
        ],
        help='Global model IDs for multi-tile'
    )
    parser.add_argument(
        '--agent_skeletons',
        nargs='+',
        type=int,
        action='append',
        help='Agent skeletons for multi-tile (provide as multiple [[x1,y1],[x2,y2]] groups)'
    )
    parser.add_argument(
        '--start_positions',
        nargs='+',
        type=float,
        action='append',
        help='Start positions for multi-tile (provide as multiple [x,y] pairs)'
    )
    parser.add_argument(
        '--goal_positions',
        nargs='+',
        type=float,
        action='append',
        help='Goal positions for multi-tile (provide as multiple [x,y] pairs)'
    )
    parser.add_argument(
        '--start_goal_setup',
        type=str,
        default='circle',
        choices=[
            'boundary',
            'circle',
            'random',
        ],
    )

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Create config object (assuming MultiAgentPlanningSingleTrialConfig exists)
    config = MultiAgentPlanningSingleTrialConfig()
    config.num_agents = args.n
    config.instance_name = args.instance_name
    config.multi_agent_planner_class = args.hp
    config.single_agent_planner_class = args.lp
    config.stagger_start_time_dt = args.st
    config.runtime_limit = args.rl
    config.time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    config.render_animation = args.ra
    # use params seed
    config.run_experiment = False

    if config.single_agent_planner_class in ['MDOCEnsemble']:
        from mdoc.config.mdoc_params import MDOCParams as params
        device = get_torch_device(params.device)
    elif config.single_agent_planner_class in ['MMDEnsemble', "MMD"]:
        from mdoc.config.mmd_params import MMDParams as params
        device = get_torch_device(params.device)
    elif config.single_agent_planner_class in ['WAStar', 'WAStarData']:
        from mdoc.config.wastar_params import WASTARParams as params
        device = get_torch_device(params.device)
    elif config.single_agent_planner_class == 'KCBSLower':
        from mdoc.config.kcbs_params import KCBSParams as params
        device = get_torch_device(params.device)
    else:
        raise ValueError(f'Unknown single agent planner class: {config.single_agent_planner_class}')

    print(f">>>>>>>>> Using {str(device).upper()} <<<<<<<<<<<<<<<")
    tensor_args = {'device': device, 'dtype': torch.float32}

    if args.example_type == "single_tile":
        config.global_model_ids = [[args.e]]

        # Set up starts and goals
        config.agent_skeleton_l = [[[0, 0]]] * config.num_agents
        torch.random.manual_seed(42)

        if args.start_goal_setup == "boundary":
            config.start_state_pos_l, config.goal_state_pos_l = \
                get_start_goal_pos_boundary(config.num_agents, 0.87, device)
        elif args.start_goal_setup == "circle":
            config.start_state_pos_l, config.goal_state_pos_l = \
                get_start_goal_pos_circle(config.num_agents, 0.8, device)
        elif args.start_goal_setup == "random":
            config.start_state_pos_l, config.goal_state_pos_l = \
                get_start_goal_pos_random_in_env(
                    env_class=EnvEmpty2D,
                    num_agents=config.num_agents,
                    tensor_args=tensor_args,
                    obstacle_margin=0.15,
                    margin=0.15,
                )
            print(config.start_state_pos_l, config.goal_state_pos_l)
        else:
            RuntimeError("No such choice")

        print("Starts:", config.start_state_pos_l)
        print("Goals:", config.goal_state_pos_l)

        run_multi_agent_trial(config)
        print(GREEN, 'OK.', RESET)

    elif args.example_type == "multi_tile":
        config.num_agents = args.n if args.n != 6 else 4  # Default to 4 for multi-tile
        config.stagger_start_time_dt = args.st if args.st != 0 else 5
        config.global_model_ids = [args.me]

        if args.agent_skeletons:
            config.agent_skeleton_l = args.agent_skeletons
        else:
            config.agent_skeleton_l = [[[0, 0], [0, 1]],
                                       [[0, 1], [0, 0]],
                                       [[0, 0], [0, 1]],
                                       [[0, 1], [0, 0]]]

        if args.start_positions and args.goal_positions:
            config.start_state_pos_l = torch.tensor(args.start_positions, **tensor_args)
            config.goal_state_pos_l = torch.tensor(args.goal_positions, **tensor_args)
        else:
            config.start_state_pos_l, config.goal_state_pos_l = \
                (torch.tensor([[0, 0.8], [0, 0.3], [0, -0.3], [0, -0.8]], **tensor_args),
                 torch.tensor([[0, -0.8], [0, -0.3], [0, 0.3], [0, 0.8]], **tensor_args))

        config.multi_agent_planner_class = "XECBS" if args.multi_agent_planner == "CBS" else args.multi_agent_planner
        print(config.start_state_pos_l)
        run_multi_agent_trial(config)
        print(GREEN, 'OK.', RESET)
