"""
MIT License
"""


class PlannerOutput:
    def __init__(self):
        self.trajs_iters = None
        self.trajs_final = None
        self.trajs_final_coll = None
        self.trajs_final_coll_idxs = None
        self.trajs_final_free = None
        self.trajs_final_free_idxs = None
        self.success_free_trajs = None
        self.fraction_free_trajs = None
        self.collision_intensity_trajs = None
        self.idx_best_traj = None  # In trajs_final.
        self.traj_final_free_best = None
        self.cost_best_free_traj = None
        self.cost_smoothness = None
        self.cost_path_length = None
        self.cost_all = None
        self.variance_waypoint_trajs_final_free = None
        self.t_total = None
        # The constraints set used for this call.
        self.constraints_l = None
