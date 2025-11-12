import torch


# A central location for aggregating the parameters used across files.
class WASTARParams:
    # Robot parameters.
    robot_planar_disk_radius = 0.05
    # Single-agent planning parameters.
    use_guide_on_extra_objects_only = False
    horizon = 64 # for virtualization
    n_samples = 64  # Batch size. Number of trajectories generated together.
    n_local_inference_noising_steps = 3  # Number of noising steps in local inference.
    n_local_inference_denoising_steps = 3  # Number of denoising steps in local inference.
    start_guide_steps_fraction = 0.5  # The fraction of the inference steps that are guided.
    n_guide_steps = 20  # The number of steps taken when applying conditioning at one diffusion step.
    n_diffusion_steps_without_noise = 1  # How many (at the end) diffusion steps get zero noise and guiding.
    weight_grad_cost_collision = 2e-2  # 5e-2
    weight_grad_cost_smoothness = 8e-2
    weight_grad_cost_constraints = 2e-1
    weight_grad_cost_soft_constraints = 2e-2  # 2e-2
    factor_num_interpolated_points_for_collision = 1.5
    trajectory_duration = 5.0
    device = 'cpu'
    debug = True
    seed = 42
    results_dir = 'logs'

    # Multi-agent planning parameters.
    is_xcbs = False  # Depends on the planner in multi agent inference. Remove this.
    is_ecbs = False  # Depends on the planner in multi agent inference. Remove this.
    skip_root_creation = False
    vertex_constraint_radius = robot_planar_disk_radius * 2.4
    low_level_choose_path_from_batch_strategy = 'least_collisions'  # 'least_collisions' or 'least_cost'.

    # Single agent planning parameters.
    w = 1.0  # Weight of the heuristic function.
    wastar_delta_q_action_l = [[-0.1, 0, 0.1], [-0.1, 0, 0.1]]  # Actions for moving on the 4-connected grid.
    wastar_discretization = [0.1, 0.1]  # Discretization of the grid.
    wastar_focal_w = 1.5  # Focal weight.

    # Evaluation.
    runtime_limit = 1000  # 1 minute.
    data_adherence_linear_deviation_fraction = 0.1  # Points closer to start-goal line than fraction * length adhere.

    # Torch.
    tensor_args = {'device': device, 'dtype': torch.float32}
