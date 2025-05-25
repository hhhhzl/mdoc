"""
"""
# TODO(yorai): param for allowing/not re-expansions.

# General imports.
import heapq
import itertools
import os
import pickle
import time
from typing import List, Dict
import torch
import numpy as np
from tqdm import tqdm

from mdoc.planners.single_agent import PlannerOutput
# Project imports.
from mdoc.utils.loading import load_params_from_yaml
from mdoc.common import PathBatchExperience, densify_trajs
from mdoc.common.pretty_print import *
from mdoc.trainer import get_dataset
from torch_robotics.tasks.tasks_ensemble import PlanningTaskEnsemble
from mdoc.common.constraints import *
from mdoc.planners.common import FocalQueue
from mdoc.config.wastar_params import MMPDParams as params


class SearchState:
    id: int = 0
    g: float = 0
    h: float = 0
    c: int = 0
    t: int = 0
    f: float = float('inf')
    q: torch.tensor = None
    parent_id = None
    is_open = False
    is_closed = False

    def __init__(self, id, q, t=0, g=0, h=0, parent_id=None):
        self.id = id
        self.g = g
        self.h = h
        self.q = q
        self.t = t
        self.parent_id = parent_id
        self.is_open = False
        self.is_closed = False

    def set_open(self):
        self.is_open = True
        self.is_closed = False

    def set_closed(self):
        self.is_open = False
        self.is_closed = True

    def set_values(self, g, h, c, f, parent_id):
        self.g = g
        self.h = h
        self.c = c
        self.f = f
        self.parent_id = parent_id

    def __lt__(self, other):
        if self.f == other.f:
            return self.g > other.g
        return self.f < other.f

    def __eq__(self, other):
        return self.f == other.f

    def get_cost(self):
        return self.f

    def get_subcost(self):
        return self.c

    def get_id(self):
        return self.id


class WAStar:
    def __init__(self,
                 model_ids: tuple,
                 transforms: Dict[int, torch.tensor],
                 start_state_pos: torch.tensor,
                 goal_state_pos: torch.tensor,
                 delta_q_action_l: List[List[float]],
                 discretization: torch.tensor,  # Shape (q_dim,).
                 device: str,
                 debug: bool,
                 results_dir: str,
                 trained_models_dir: str,
                 w=1,
                 is_use_data_cost=False,
                 **kwargs
                 ):
        self.device = device
        self.debug = debug
        self.results_dir = results_dir
        self.model_ids = model_ids
        self.transforms = transforms
        self.tensor_args = {'device': self.device, 'dtype': torch.float32}

        # Search information.
        self.q_start = start_state_pos.to(**self.tensor_args).squeeze(0)
        self.q_goal = goal_state_pos.to(**self.tensor_args).squeeze(0)
        # Transform to their tiles.
        self.q_start = self.q_start + self.transforms[0]
        self.q_goal = self.q_goal + self.transforms[len(self.transforms) - 1]
        self.w = w
        self.is_use_data_cost = is_use_data_cost
        self.time_to_constraints = dict()
        # The actions are combinations of all the available deltas. action_l is a list of tensors, each of size q_dim.
        # s`.q = s.q + action
        self.actions = []
        # Get all combinations [element from first entry, element from second entry, ...] for all entries in delta_q_action_l.
        for action in itertools.product(*delta_q_action_l):
            # Remove actions that are with all elements absolute value equal and non-zero.
            if np.allclose(np.abs(action), np.abs(action[0])) and not np.allclose(action, 0):
                continue
            self.actions.append(torch.tensor(action, device=self.device))
        # Deduce discretization from the actions. Of shape (q_dim,).
        self.discretization = discretization
        self.q_start = self.snap_to_grid(self.q_start)
        self.q_goal = self.snap_to_grid(self.q_goal)
        print("[WAStar] Start state: ", self.q_start)
        print("[WAStar] Goal state: ", self.q_goal)

        # The search states.
        # Mapping robot states to the search states.
        self.id_to_search_state = dict()
        # Map (tup(q), t) to search state.
        self.q_t_to_search_state_id = dict()
        # The search priority queues.
        self.open = FocalQueue(params.wastar_focal_w)
        # The search state of the goal state.
        self.goal_search_state = None

        model_dirs, results_dirs, args = [], [], []
        self.models, tasks = {}, {}
        self.guides = {}
        datasets = []
        sample_kwargs = []
        for j, model_id in enumerate(model_ids):
            model_dir = os.path.join(trained_models_dir, model_id)
            model_dirs.append(model_dir)
            args.append(load_params_from_yaml(os.path.join(model_dir, 'args.yaml')))

            ## Load dataset with env, robot, task ##
            train_subset, train_dataloader, val_subset, val_dataloader = get_dataset(
                dataset_class='TrajectoryDataset',
                use_extra_objects=True,
                obstacle_cutoff_margin=0.01,
                **args[-1],
                tensor_args=self.tensor_args
            )
            dataset = train_subset.dataset
            datasets.append(dataset)
            n_support_points = dataset.n_support_points
            robot = dataset.robot
            # We hack the radius of robots to be a bit smaller such that they are allowed to be tangent to each other.
            robot.radius *= 0.9
            if j == 0:
                self.robot = robot
            task = dataset.task
            tasks[j] = task

        self.task = PlanningTaskEnsemble(tasks,
                                         transforms,
                                         tensor_args=self.tensor_args)

        # Get the q limits from the robot. This is a tensor of shape (q_dim, 2).
        self.q_limits = self.task.ws_limits

        # Create a collision map.
        self.is_collision_map = {}
        for q in itertools.product(*[
            torch.arange(self.q_limits[0, 0], self.q_limits[1, 0] + self.discretization[0], self.discretization[0]),
            torch.arange(self.q_limits[0, 1], self.q_limits[1, 1] + self.discretization[1], self.discretization[1])]):
            q = torch.tensor(q, device=self.device)
            q_coord = self.get_grid_coord(q)
            q = self.snap_to_grid(q)
            q_int = (q * 1000).int().cpu().numpy()
            self.is_collision_map[tuple(q_int)] = torch.any(self.task.compute_collision(q))

        # Check the validity of the start and goal.
        if not self.is_state_to_state_valid(self.q_start, 0, self.q_start, 0):
            raise ValueError("Start state is not valid.")
        if not self.is_state_to_state_valid(self.q_goal, 0, self.q_goal, 0):
            raise ValueError("Goal state is not valid.")

        # Create a backwards-dijkstra heuristic.
        cell_heuristic = {}
        Q = []
        heapq.heappush(Q, (0, self.get_grid_coord(self.q_goal)))
        while len(Q) > 0:
            h, q_coord = heapq.heappop(Q)
            if q_coord in cell_heuristic:
                continue
            cell_heuristic[q_coord] = h
            for action in self.actions:
                q_new = self.get_q_from_grid_coord(q_coord) + action
                if self.is_state_to_state_valid(self.get_q_from_grid_coord(q_coord), 0, q_new, 0):
                    q_new_coord = self.get_grid_coord(q_new)
                    if q_new_coord not in cell_heuristic:
                        heapq.heappush(Q, (h + 1, q_new_coord))
        self.cell_heuristic = cell_heuristic
        print("Heuristic created.")

        # # Visualize.
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # for q_coord in cell_heuristic:
        #     q = self.get_q_from_grid_coord(q_coord)
        #     h_value = cell_heuristic[q_coord]
        #     ax.scatter(q[0].cpu().numpy(), q[1].cpu().numpy(), c='blue', s=10, alpha=min(1, h_value/50))
        # ax.set_aspect('equal')
        # ax.set_xlim([-1, 1])
        # ax.set_ylim([-1, 1])
        # plt.show()

        # Create a data cost map.
        # Steps to create the transition cost map:
        # Iterating over the trajectory dataset, for each trajectory:
        # 1. Move state by state.
        # 2. Increment the costmap entry for the current state and the next state.
        # After, normalize the costmap. cell = cell / max(cell).
        # Flip such that most traveled paths are the least costly. cell = 1 - cell.
        # Add 1 to have a minimum cost of 1. cell = cell + 1.
        transition_cost_data = {}
        # Fill with zeros.
        for q in itertools.product(*[
            torch.arange(self.q_limits[0, 0], self.q_limits[1, 0] + self.discretization[0], self.discretization[0]),
            torch.arange(self.q_limits[0, 1], self.q_limits[1, 1] + self.discretization[1], self.discretization[1])]):
            q = torch.tensor(q, device=self.device)
            q_coord = self.get_grid_coord(q)
            transition_cost_data[q_coord] = {}
            # Fill with zeros. A value for each successor.
            for action in self.actions:
                # if torch.allclose(action, torch.zeros_like(action)):
                #     continue
                q_new = q + action
                q_new = self.snap_to_grid(q_new)
                q_new_coord = self.get_grid_coord(q_new)
                transition_cost_data[q_coord][q_new_coord] = 1

        for model_id, dataset in enumerate(datasets):
            if self.is_use_data_cost:
                print(GREEN + f"Creating transition cost data for model {self.model_ids[model_id]}." + RESET)
                # Check if a pkl file for this model exists.
                pkl_file_path = os.path.join(results_dir, 'transition_cost_data' +
                                             self.model_ids[model_id] +
                                             '_discr_' +
                                             str(round(self.discretization[0].item(), 3)).replace('.', '-') +
                                             '_' +
                                             str(round(self.discretization[1].item(), 3)).replace('.', '-')
                                             + '.pkl')
                if os.path.exists(pkl_file_path):
                    with open(pkl_file_path, 'rb') as f:
                        transition_cost_data_local = pickle.load(f)
                        # Add the transform.
                        for q_from_local in transition_cost_data_local:
                            for q_to_local in transition_cost_data_local[q_from_local]:
                                q_from = torch.tensor(q_from_local, device=self.device, dtype=torch.float32) + \
                                         self.transforms[model_id]
                                q_to = torch.tensor(q_to_local, device=self.device, dtype=torch.float32) + \
                                       self.transforms[model_id]
                                q_from_coord = self.get_grid_coord(q_from)
                                q_to_coord = self.get_grid_coord(q_to)
                                if q_from_coord not in transition_cost_data:
                                    transition_cost_data[q_from_coord] = {}
                                transition_cost_data[q_from_coord][q_to_coord] = \
                                    transition_cost_data_local[q_from_local][q_to_local]
                    continue
                # Otherwise, iterate over the dataset to create the cost map. Also keep a copy to save.
                else:
                    pbar = tqdm(total=len(dataset))
                    for traj in dataset:
                        pbar.update(1)
                        prev_q_grid_coord = None
                        for t in range(0, len(traj['traj_normalized']), 1):
                            q = traj['traj_normalized'][t, :2] + self.transforms[model_id]
                            q_grid_coord = self.get_grid_coord(q)
                            if prev_q_grid_coord is None:
                                prev_q_grid_coord = q_grid_coord
                            else:
                                if prev_q_grid_coord != q_grid_coord:
                                    if prev_q_grid_coord not in transition_cost_data:
                                        transition_cost_data[prev_q_grid_coord] = {}
                                    # Choose the nearest neighbor.
                                    min_dist = float('inf')
                                    min_neighbor_coord = list(transition_cost_data[prev_q_grid_coord].keys())[0]
                                    for q_new_coord in transition_cost_data[prev_q_grid_coord]:
                                        dist = torch.norm(
                                            torch.tensor(q_new_coord, dtype=torch.float32) - torch.tensor(q_grid_coord,
                                                                                                          dtype=torch.float32))
                                        if dist < min_dist:
                                            min_dist = dist
                                            min_neighbor_coord = q_new_coord
                                    q_grid_coord = min_neighbor_coord

                                    transition_cost_data[prev_q_grid_coord][q_grid_coord] += 1
                                    prev_q_grid_coord = q_grid_coord
                    # Normalize. Invert all values.
                    for q_from in transition_cost_data:
                        for q_to in transition_cost_data[q_from]:
                            transition_cost_data[q_from][q_to] = 1.0 / transition_cost_data[q_from][q_to] * 10 + 1

                    # Create a copy for saving. Here, all the keys are transformed back and un-scaled.
                    transition_cost_data_to_save = {}
                    for q_from in transition_cost_data:
                        for q_to in transition_cost_data[q_from]:
                            q_from_coord = self.get_q_from_grid_coord(q_from) - self.transforms[model_id]
                            q_to_coord = self.get_q_from_grid_coord(q_to) - self.transforms[model_id]
                            q_from_coord = tuple(q_from_coord.cpu().numpy())
                            q_to_coord = tuple(q_to_coord.cpu().numpy())
                            if q_from_coord not in transition_cost_data_to_save:
                                transition_cost_data_to_save[q_from_coord] = {}
                            transition_cost_data_to_save[q_from_coord][q_to_coord] = transition_cost_data[q_from][q_to]

                    # Save the data.
                    with open(pkl_file_path, 'wb') as f:
                        pickle.dump(transition_cost_data_to_save, f)

        self.transition_cost_data = transition_cost_data

        # Visualize the cost map. Create a thick line.
        # import matplotlib.pyplot as plt
        # fig, ax = plt.subplots()
        # for q_from in transition_cost_data:
        #     for q_to in transition_cost_data[q_from]:
        #         if transition_cost_data[q_from][q_to] < 2.0:
        #             q_from_vis = self.get_q_from_grid_coord(q_from).cpu().numpy() + np.random.rand(2) * 0.01
        #             q_to_vis = self.get_q_from_grid_coord(q_to).cpu().numpy() + np.random.rand(2) * 0.01
        #             ax.plot([q_from_vis[0], q_to_vis[0]],
        #                     [q_from_vis[1], q_to_vis[1],],
        #                     alpha=(2-transition_cost_data[q_from][q_to])/2, color='blue',
        #                     linewidth=(3 * (1-transition_cost_data[q_from][q_to]))/2)
        # ax.set_aspect('equal')
        # ax.set_xlim([-1, 1])
        # ax.set_ylim([-1, 1])
        # plt.show()

    def get_grid_coord(self, q: torch.tensor):
        q = self.snap_to_grid(q)
        q_int = (q * 1000).int().cpu().numpy()
        return tuple(q_int)

    def get_q_from_grid_coord(self, q_coord: torch.tensor):
        q = torch.tensor(q_coord, device=self.device, dtype=torch.float32) / 1000.0
        q = torch.round(q, decimals=3)
        return q

    def is_state_in_collision(self, q: torch.tensor):
        q_coord = self.get_grid_coord(q)
        return self.is_collision_map[q_coord]

    def get_or_create_search_state(self, q: torch.tensor, t: int):
        # Convert to a tuple.
        q_tuple = tuple(q.cpu().numpy())
        q_t_tuple = (q_tuple, t)
        # Check if the state exists.
        if q_t_tuple in self.q_t_to_search_state_id:
            return self.id_to_search_state[self.q_t_to_search_state_id[q_t_tuple]]
        # Create a new search state.
        new_id = len(self.id_to_search_state)
        new_search_state = SearchState(new_id, q, t)
        self.id_to_search_state[new_id] = new_search_state
        self.q_t_to_search_state_id[q_t_tuple] = new_id
        return new_search_state

    def heuristic(self, q: torch.tensor, t: int):
        # Simple euclidian distance for now.
        return self.cell_heuristic[self.get_grid_coord(q)]

    def cost(self, q: torch.tensor, t: int, q_new: torch.tensor, t_new: int):
        q_coord = self.get_grid_coord(q)
        q_new_coord = self.get_grid_coord(q_new)
        return self.transition_cost_data[q_coord][q_new_coord]  # + 1 //////////////////////////

    def subcost(self, q: torch.tensor, t: int, q_new: torch.tensor, t_new: int):
        # Go through the available soft-collision constraints.
        subcost = 0
        for constraint in self.soft_collision_constraints:
            # Evaluate the constraint to see if this q, t collides with any of the points.
            # As a first pass, just do it sequentially.
            for i in range(len(constraint.q_l)):
                q_constraint = constraint.q_l[i]
                t_range_constraint = constraint.t_range_l[i]
                radius_constraint = constraint.radius_l[i]
                if t_new >= t_range_constraint[0] and t_new < t_range_constraint[1]:
                    if torch.norm(q - q_constraint) <= 2 * self.robot.radius:
                        subcost += 1
        return subcost

    def is_state_to_state_valid(self, q_from: torch.tensor, t_from: int, q_to: torch.tensor, t_to: int):
        # Check if the state is within the bounds.
        # q shape is (q_dim,).
        q_from = self.snap_to_grid(q_from)
        q_to = self.snap_to_grid(q_to)
        if not torch.all(q_to >= self.q_limits[0, :]) or not torch.all(q_to <= self.q_limits[1, :]):
            return False
        # Check if the state is in collision.
        is_collision = self.is_state_in_collision(q_to)
        if is_collision:
            return False

        # Check validity wrt constraints.
        if t_to in self.time_to_constraints:
            constraints = self.time_to_constraints[t_to]
            for constraint in constraints:
                if isinstance(constraint, VertexConstraint):
                    q_constraint = constraint.get_q()
                    q_constraint = self.snap_to_grid(q_constraint)
                    if torch.allclose(q_to, q_constraint):
                        return False
                elif isinstance(constraint, EdgeConstraint):
                    q_from_constraint = constraint.get_q_from()
                    q_to_constraint = constraint.get_q_to()
                    q_from_constraint = self.snap_to_grid(q_from_constraint)
                    q_to_constraint = self.snap_to_grid(q_to_constraint)
                    t_from = constraint.get_t_range_l()[0][0]
                    t_to = constraint.get_t_range_l()[0][1]
                    if torch.allclose(q_from, q_from_constraint) and torch.allclose(q_to, q_to_constraint):
                        return False
                else:
                    continue

        return True

    def get_successors(self, s: SearchState) -> List[int]:  # Returning a list of search state ids.
        successor_ids = []
        for action in self.actions:
            q_new = s.q + action
            q_new = self.snap_to_grid(q_new)
            t_new = s.t + 1
            # Check if the transition is valid.
            is_transition_valid = self.is_state_to_state_valid(s.q, s.t, q_new, t_new)
            if is_transition_valid:
                # Get or create the search state.
                s_new = self.get_or_create_search_state(q_new, t_new)
                # Store.
                successor_ids.append(s_new.id)
        return successor_ids

    def reset_planning_data(self):
        self.open.clear()
        self.id_to_search_state = dict()
        self.q_t_to_search_state_id = dict()
        start_search_state = self.get_or_create_search_state(self.q_start, 0)
        start_search_state.h = self.heuristic(start_search_state.q, start_search_state.t)
        start_search_state.f = start_search_state.h
        start_search_state.set_open()
        self.id_to_search_state[0] = start_search_state
        self.open.push(start_search_state)
        self.goal_search_state = None

    def snap_to_grid(self, q: torch.tensor):
        # Snap to the grid.
        q_discrete = torch.round(torch.round(q / self.discretization) * self.discretization, decimals=3)
        return q_discrete

    def __call__(
            self,
            start_state_pos,
            goal_state_pos,
            constraints_l=None,
            experience: PathBatchExperience = None,
            *args,
            **kwargs
    ):
        # A few timers.
        startt = time.time()
        # Make sure that the start and goal states are similar to the ones stored.
        start_state_pos = start_state_pos.to(**self.tensor_args)
        goal_state_pos = goal_state_pos.to(**self.tensor_args)
        start_state_pos = self.snap_to_grid(start_state_pos)
        goal_state_pos = self.snap_to_grid(goal_state_pos)
        assert torch.allclose(start_state_pos, self.q_start)

        # Clear open, closed, and search states.
        self.reset_planning_data()

        # Keep track of any soft-collision objects, if those are availabe.
        self.soft_collision_constraints = []
        for c in constraints_l:
            if isinstance(c, MultiPointConstraint):
                if c.is_soft:
                    self.soft_collision_constraints.append(c)

        # Create a map from time to constraints. For edge constraints, this corresponds to the end-time.
        time_to_constraints = dict()
        last_time_goal_constrained = 0
        if constraints_l is not None:
            for constraint in constraints_l:
                if isinstance(constraint, EdgeConstraint):
                    t_from = constraint.get_t_range_l()[0][0]
                    t_to = constraint.get_t_range_l()[0][1]
                    if t_to not in time_to_constraints:
                        time_to_constraints[t_to] = []
                    constraint.q_to = self.snap_to_grid(constraint.q_to)
                    constraint.q_from = self.snap_to_grid(constraint.q_from)
                    time_to_constraints[t_to].append(constraint)

                    if torch.allclose(constraint.get_q_to(), self.q_goal) and t_to > last_time_goal_constrained:
                        last_time_goal_constrained = max(t_to, last_time_goal_constrained)

                elif isinstance(constraint, VertexConstraint):
                    t = constraint.get_t_range_l()[0][1]
                    if t not in time_to_constraints:
                        time_to_constraints[t] = []
                    constraint.q = self.snap_to_grid(constraint.q)
                    time_to_constraints[t].append(constraint)

                    if torch.allclose(constraint.get_q(), self.q_goal):
                        last_time_goal_constrained = max(t, last_time_goal_constrained)
                else:
                    continue
        # Reset the stored information and store the constraints.
        self.time_to_constraints = time_to_constraints

        while len(self.open) > 0:
            # Check for timeout.
            if time.time() - startt > params.runtime_limit:
                print("Timeout.")
                return None

            # Get the current state.
            s = self.open.pop()
            s.set_closed()
            # Check if the goal state is reached.
            if torch.allclose(s.q, self.q_goal) and s.t > last_time_goal_constrained:
                self.goal_search_state = s
                break
            # Get the successors.
            successor_ids = self.get_successors(s)
            for s_id in successor_ids:
                s_new = self.id_to_search_state[s_id]
                # Get all the associated values for this new state.
                g_new = s.g + self.cost(s.q, s.t, s_new.q, s_new.t)
                c_new = s.c + self.subcost(s.q, s.t, s_new.q, s_new.t)
                h_new = self.heuristic(s_new.q, s_new.t)
                f_new = g_new + self.w * h_new

                # Focal search allows for re-expansions. Check if the state is new (not open or closed).
                # if not s_new.is_closed and not s_new.is_open:
                #     # In this case just push it.
                #     s_new.set_values(g_new, h_new, c_new, f_new, parent_id=s.id)
                #     s_new.set_open()
                #     self.open.push(s_new)
                #
                # # If the state is either open or closed, check if the new f is better OR f equal and c better.
                # else:
                #     if f_new < s_new.f or (f_new == s_new.f and c_new < s_new.c):
                #         s_new.set_values(g_new, h_new, c_new, f_new, parent_id=s.id)
                #         s_new.set_open()
                #         # Push to open. If this state was previously added to open,
                #         # the this will create a copy with the new values.
                #         # This "better" copy will be expanded before the previous one.
                #         # When the previous one is expanded, the above check will fail and it will be discarded.
                #         self.open.push(s_new)

                # TEST TEST TEST We do not allow for re-expansions in this implementation.
                if s_new.is_closed:
                    continue
                else:
                    if f_new < s_new.f:
                        s_new.set_values(g_new, h_new, c_new, f_new, parent_id=s.id)
                        self.open.push(s_new)

        if self.goal_search_state is None:
            print("No path found.")
            return None

        # Extract the path.
        path = []
        s = self.goal_search_state
        while s is not None:
            path.append(s.q)
            s = self.id_to_search_state[s.parent_id] if s.parent_id is not None else None

        path = torch.stack(path[::-1])

        # Return the path with details.
        output = PlannerOutput()
        output.trajs_iters = None
        output.trajs_final = path.unsqueeze(0)
        output.trajs_final_coll = None
        output.trajs_final_coll_idxs = None
        output.trajs_final_free = path.unsqueeze(0)  # Shape [B, H, D]
        output.trajs_final_free_idxs = torch.tensor([0], device=self.device)  # Shape [B]
        output.success_free_trajs = 1.0
        output.fraction_free_trajs = 0.0
        output.collision_intensity_trajs = 0.0
        if output.success_free_trajs:
            output.idx_best_traj = 0
            output.traj_final_free_best = path.unsqueeze(0)
            output.cost_best_free_traj = self.goal_search_state.g
            output.cost_smoothness = None
            output.cost_path_length = self.goal_search_state.g
            output.cost_all = self.goal_search_state.g
            output.variance_waypoint_trajs_final_free = 0.0
        else:
            output.idx_best_traj = None
            output.traj_final_free_best = None
            output.cost_best_free_traj = None
            output.cost_smoothness = None
            output.cost_path_length = None
            output.cost_all = None
            output.variance_waypoint_trajs_final_free = None
        output.t_total = time.time() - startt
        output.constraints_l = constraints_l

        # # PRINT.
        # print("[WAStar] Path found: \n", path)

        return output
