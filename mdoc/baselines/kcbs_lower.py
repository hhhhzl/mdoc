import os
from dataclasses import dataclass
from typing import List, Optional, Dict
import numpy as np
import torch
import itertools
import math
from torch_robotics.tasks.tasks_ensemble import PlanningTaskEnsemble
# --------------------------------------------------------------
from mdoc.baselines.kcbs import World, RRTParams, rrt_plan_single, StaticCircle, StaticBox
from mdoc.common.experiences import PathBatchExperience
from mdoc.common.constraints import MultiPointConstraint
from mdoc.planners.common import PlannerOutput
from mdoc.common import smooth_trajs
from mdoc.utils.loading import load_params_from_yaml
from mdoc.trainer import get_dataset


@dataclass
class PathObstacle:
    ts: np.ndarray  # (L,)
    pts: np.ndarray  # (L,2)
    radius: float

    def center_at(self, t: float) -> np.ndarray:
        if t <= self.ts[0]:
            return self.pts[0]
        if t >= self.ts[-1]:
            return self.pts[-1]
        k = np.searchsorted(self.ts, t, side="right") - 1
        k = max(0, min(k, len(self.ts) - 2))
        t0, t1 = self.ts[k], self.ts[k + 1]
        a = 0.0 if t1 == t0 else (t - t0) / (t1 - t0)
        return (1 - a) * self.pts[k] + a * self.pts[k + 1]


# https://arxiv.org/pdf/2207.00576
# lower planner for Conflict-based Search for Multi-Robot Motion Planning with Kinodynamic Constraints (ConstrainedX)
class KCBSLower:
    def __init__(
            self,
            model_ids: tuple,
            transforms: Dict[int, torch.tensor],
            start_state_pos: torch.tensor,
            goal_state_pos: torch.tensor,
            debug: bool,
            trained_models_dir: str,
            device: str,
            results_dir: str,
            seed: int = 0,
            q_is_workspace: bool = True,
            q_xy_index: tuple = (0, 1),
            default_ob_radius: float = 0.25,
            **kwargs
    ):
        self.device = device
        self.debug = debug
        self.results_dir = results_dir
        self.model_ids = model_ids
        self.transforms = transforms
        self.tensor_args = {'device': self.device, 'dtype': torch.float32}
        self.results_dir = results_dir
        self.trained_models_dir = trained_models_dir

        self.q_start = start_state_pos.to(**self.tensor_args).squeeze(0)
        self.q_goal = goal_state_pos.to(**self.tensor_args).squeeze(0)
        # Transform to their tiles.
        self.q_start = self.q_start + self.transforms[0]
        self.q_goal = self.q_goal + self.transforms[len(self.transforms) - 1]

        print("[KCBS] Start state: ", self.q_start)
        print("[KCBS] Goal state: ", self.q_goal)

        model_dirs, results_dirs, args = [], [], []
        self.models, tasks = {}, {}
        self.guides = {}
        datasets = []
        for j, model_id in enumerate(model_ids):
            model_dir = os.path.join(trained_models_dir, model_id)
            model_dirs.append(model_dir)
            args.append(load_params_from_yaml(os.path.join(model_dir, 'args.yaml')))

            # Load dataset with env, robot, task #
            train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
                dataset_class='TrajectoryDataset',
                use_extra_objects=True,
                obstacle_cutoff_margin=0.01,
                **args[-1],
                tensor_args=self.tensor_args
            )
            dataset = train_subset.dataset
            datasets.append(dataset)
            robot = dataset.robot
            # We hack the radius of robots to be a bit smaller such that they are allowed to be tangent to each other.
            robot.radius *= 0.9
            if j == 0:
                self.robot = robot
            task = dataset.task
            tasks[j] = task

        self.task = PlanningTaskEnsemble(
            tasks,
            transforms,
            tensor_args=self.tensor_args
        )

        # Get the q limits from the robot. This is a tensor of shape (q_dim, 2).
        self.q_limits = self.task.ws_limits

        self.world = self._build_world_from_model()
        self.rrt_params = RRTParams()
        self.seed = seed
        self.q_is_workspace = q_is_workspace
        self.q_xy_index = q_xy_index
        self.default_ob_radius = default_ob_radius

    def _build_world_from_model(self) -> World:
        limits = self.q_limits.detach().cpu().numpy()
        x_min, y_min = float(limits[0, 0]), float(limits[0, 1])
        x_max, y_max = float(limits[1, 0]), float(limits[1, 1])
        robot_radius = float(self.robot.radius)

        static_obstacles = []
        env = getattr(self.task, "env", None)

        if env is not None and hasattr(env, "obj_fixed_list"):
            for obj_field in env.obj_fixed_list:
                for primitive in getattr(obj_field, "fields", []):
                    # --- spheres： ---
                    if hasattr(primitive, "centers") and hasattr(primitive, "radii"):
                        centers = primitive.centers.detach().cpu().numpy()
                        radii = primitive.radii.detach().cpu().numpy()
                        for c, r in zip(centers, radii):
                            static_obstacles.append(StaticCircle(center=np.array(c[:2], dtype=float), radius=float(r)))

                    # --- boxes：---
                    elif hasattr(primitive, "centers") and hasattr(primitive, "sizes"):
                        centers = primitive.centers.detach().cpu().numpy()  # (n,2)
                        sizes = primitive.sizes.detach().cpu().numpy()  # (n,2)
                        for c, s in zip(centers, sizes):
                            static_obstacles.append(
                                StaticBox(center=np.array(c[:2]), size=np.array(s[:2]))
                            )

        world = World(
            bounds=((x_min, x_max), (y_min, y_max)),
            static_obstacles=static_obstacles,
            moving_obstacles=[],
            robot_radius=robot_radius
        )
        print(f"[KCBS] Loaded {len(static_obstacles)} static obstacles")
        return world

    def _constraints_to_path_obstacles(
            self, constraints_l: Optional[List]
    ) -> List[PathObstacle]:
        """
        support MultiPointConstraint / CostConstraint：
        - for each constraints (q, t_range, radius)，generate a [t_from, t_to] valid PathObstacle that center at q(x,y)
        """
        if not constraints_l:
            return []

        def q_to_xy(q: torch.Tensor) -> np.ndarray:
            if self.q_is_workspace:
                q_np = q.detach().cpu().numpy()
                # if q dim >2， (x,y)
                if q_np.shape[-1] >= 2:
                    return np.array([q_np[self.q_xy_index[0]], q_np[self.q_xy_index[1]]], dtype=float)
                return np.array(q_np[:2], dtype=float)
            else:
                raise NotImplementedError("Please implement q->(x,y) mapping when q_is_workspace=False.")

        path_obs: List[PathObstacle] = []
        for c in constraints_l:
            if isinstance(c, MultiPointConstraint):
                for (q, (t0, t1), r) in zip(c.q_l, c.t_range_l, getattr(c, "radius_l", [])):
                    xy = q_to_xy(q)
                    ts = np.array([float(t0), float(t1)], dtype=float)
                    pts = np.stack([xy, xy], axis=0)
                    rad = float(r) if r is not None else self.default_ob_radius
                    path_obs.append(PathObstacle(ts=ts, pts=pts, radius=rad))
            # elif isinstance(c, CostConstraint):
            #     for (q, (t0, t1), r) in zip(c.qs, c.traj_ranges, c.radii):
            #         xy = q_to_xy(q)
            #         ts = np.array([float(t0), float(t1)], dtype=float)
            #         pts = np.stack([xy, xy], axis=0)
            #         rad = float(r) if r is not None else self.default_ob_radius
            #         path_obs.append(PathObstacle(ts=ts, pts=pts, radius=rad))
            else:
                pass
        return path_obs

    def __call__(
            self,
            start_state_pos: torch.Tensor,
            goal_state_pos: torch.Tensor,
            constraints_l: Optional[List] = None,
            experience: Optional[PathBatchExperience] = None,
            *args, **kwargs
    ) -> PlannerOutput:
        def tens2xy(t: torch.Tensor) -> np.ndarray:
            t = t.detach().cpu()
            if self.q_is_workspace:
                if t.numel() >= 2:
                    return np.array([t[self.q_xy_index[0]].item(), t[self.q_xy_index[1]].item()], dtype=float)
                return np.array(t[:2].tolist(), dtype=float)
            else:
                raise NotImplementedError("Please implement q->(x,y) mapping when q_is_workspace=False.")

        start_xy = tens2xy(start_state_pos)
        goal_xy = tens2xy(goal_state_pos)
        path_obs = self._constraints_to_path_obstacles(constraints_l)

        # RRT as planner in paper
        pos, ts, _tree = rrt_plan_single(
            start=start_xy,
            goal=goal_xy,
            world=self.world,
            params=self.rrt_params,
            path_constraints=path_obs,
            seed=self.seed
        )
        out = PlannerOutput()
        if pos is None or ts is None:
            out.trajs_iters = None
            out.trajs_final = None
            out.trajs_final_coll = None
            out.trajs_final_coll_idxs = torch.zeros(0, dtype=torch.long)
            out.trajs_final_free = None
            out.trajs_final_free_idxs = torch.zeros(0, dtype=torch.long)
            out.success_free_trajs = False
            out.fraction_free_trajs = 0.0
            out.collision_intensity_trajs = 1.0
            out.idx_best_traj = None
            out.traj_final_free_best = None
            out.t_total = 0.0
            out.constraints_l = constraints_l
            return out

        traj_np = pos.astype(np.float32)
        traj = torch.from_numpy(traj_np).to(self.tensor_args["device"])
        traj = traj.unsqueeze(0)  # [1, H, 2]

        out.trajs_iters = [traj]
        out.trajs_final = traj
        out.trajs_final_coll = None
        out.trajs_final_coll_idxs = torch.zeros(0, dtype=torch.long)
        out.trajs_final_free = traj
        out.trajs_final_free_idxs = torch.tensor([0], dtype=torch.long)
        out.success_free_trajs = True
        out.fraction_free_trajs = 1.0
        out.collision_intensity_trajs = 0.0
        out.idx_best_traj = 0
        out.traj_final_free_best = traj[0]
        out.cost_best_free_traj = None
        out.cost_smoothness = None
        out.cost_path_length = None
        out.cost_all = None
        out.variance_waypoint_trajs_final_free = None
        out.t_total = float(ts[-1])
        out.constraints_l = constraints_l
        # out.trajs_final = smooth_trajs(out.trajs_final)
        return out
