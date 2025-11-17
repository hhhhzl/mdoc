import torch


# A central location for aggregating the parameters used across files.
class MDOCParams:
    device = 'cpu'
    debug = True
    seed = 42

    # Robot parameters.
    robot_planar_disk_radius = 0.05
    n_samples = 64  # Batch size. Number of trajectories generated together.
    horizon = 64  # Number of steps in the trajectory.
    constraints_to_check = 30 if device == 'cpu' else 100
    k_best = 15 if device == 'cpu' else 100

    # diffusion parameters.
    temp_sample = 1e-12
    n_diffusion_steps = 250
    beta0 = 1e-5
    betaT = 1e-2
    # CBF
    cbf_tau = 0.005
    cbf_eta = 1.5
    cbf_margin = 0.2
    base_beta = 0.05
    # Cost Function
    cost_control = 1
    cost_distance_to_goal = 5
    cost_time_smoothness = 1
    cost_acc_smoothness = 1
    cost_get_to_goal_early = 0.5
    cost_sdf_collison = 5e3
    cost_terminal = 20
    projection_score_weight = 0.9

    # Torch.
    compile = True
    use_cuda_graph = False  # cuda graph is not useable

    tensor_args = {'device': device, 'dtype': torch.float32}

    # Multi-agent planning parameters.
    vertex_constraint_radius = robot_planar_disk_radius * 2.4
    low_level_choose_path_from_batch_strategy = 'least_collisions'  # 'least_collisions' or 'least_cost'.

    # Evaluation.
    runtime_limit = 1000  # 1000 second.
    data_adherence_linear_deviation_fraction = 0.1  # Points closer to start-goal line than fraction * length adhere.
    results_dir = 'logs'

    # those are not used in mdoc now but we keep them here to align with mmd
    use_guide_on_extra_objects_only = False
    n_local_inference_noising_steps = 3  # Number of noising steps in local inference.
    n_local_inference_denoising_steps = 3  # Number of denoising steps in local inference.
    start_guide_steps_fraction = 0.5  # The fraction of the inference steps that are guided.
    n_guide_steps = 20  # The number of steps taken when applying conditioning at one diffusion step.
    n_diffusion_steps_without_noise = 1  # How many (at the end) diffusion steps get zero noise and guiding.
    weight_grad_cost_collision = 2e-2
    weight_grad_cost_smoothness = 8e-2
    weight_grad_cost_constraints = 2e-1
    weight_grad_cost_soft_constraints = 1e-4
    grad_step = 1e-6
    factor_num_interpolated_points_for_collision = 1.5
    trajectory_duration = 5.0



