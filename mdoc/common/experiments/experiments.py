"""
MIT License
"""
# Standard imports.
import os
import pickle
from enum import Enum
from pathlib import Path
import torch
from typing import Tuple, List

# Project imports.
from torch_robotics.robots import *
from mdoc.config.mmd_experiment_configs import get_planning_problem


class MultiAgentPlanningExperimentConfig:
    # The time string. The result directory is automatically created based on the config and this string.
    time_str: str = None
    # The instance. This dictates the map, starts and goals, and tile skeletons.
    instance_name: str = None
    # Numbers of agents to try.
    num_agents_l: List[int] = []
    # Start and goal placement scheme.
    # Stagger start times. t_init[0] = 0, t_init[1] = dt * 1, ... t_init[i] = dt * i.
    stagger_start_time_dt = 0
    # Multi-agent planner classes to try.
    multi_agent_planner_class_l: List[str] = []
    # Single agent planner class.
    single_agent_planner_class_l: List[str] = []
    # Runtime limit. In seconds.
    runtime_limit = params.runtime_limit
    # Number of trials per num agents + planner combination.
    num_trials_per_combination = 1
    # Whether to render animation or not.
    render_animation = False

    def get_single_trial_configs_from_experiment_config(self):
        single_trial_configs = []
        for num_agents in self.num_agents_l:
            # Get num_trials_per_combination x (starts and goals) that instances with different planners will use.
            start_state_pos_l_l, goal_state_pos_l_l, global_model_ids_l_l, agent_skeleton_l_l = [], [], [], []
            for _ in range(self.num_trials_per_combination):
                start_state_pos_l, goal_state_pos_l, global_model_ids, agent_skeleton_l = get_planning_problem(
                    self.instance_name, num_agents)
                start_state_pos_l_l.append(start_state_pos_l)
                goal_state_pos_l_l.append(goal_state_pos_l)
                global_model_ids_l_l.append(global_model_ids)
                agent_skeleton_l_l.append(agent_skeleton_l)
            # Create the single trial configs.
            for multi_agent_planner_class in self.multi_agent_planner_class_l:
                for single_agent_planner_class in self.single_agent_planner_class_l:
                    for trial_number in range(self.num_trials_per_combination):
                        single_trial_config = MultiAgentPlanningSingleTrialConfig()
                        single_trial_config.time_str = self.time_str
                        single_trial_config.trial_number = trial_number
                        single_trial_config.num_agents = num_agents
                        single_trial_config.stagger_start_time_dt = self.stagger_start_time_dt
                        single_trial_config.multi_agent_planner_class = multi_agent_planner_class
                        single_trial_config.single_agent_planner_class = single_agent_planner_class
                        single_trial_config.runtime_limit = self.runtime_limit
                        single_trial_config.render_animation = self.render_animation
                        single_trial_config.start_state_pos_l, single_trial_config.goal_state_pos_l, \
                        single_trial_config.global_model_ids, single_trial_config.agent_skeleton_l = \
                            start_state_pos_l_l[trial_number], goal_state_pos_l_l[trial_number], \
                            global_model_ids_l_l[trial_number], agent_skeleton_l_l[trial_number]
                        single_trial_configs.append(single_trial_config)
        return single_trial_configs

    def save(self):
        # Save the config.
        results_dir = get_result_dir_from_time_str(self.time_str)
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(results_dir, 'experiment_config.pkl'), 'wb') as f:
            pickle.dump(self, f)

    def __str__(self):
        print_str = f"""
        Experiment Config:
            Time Str: {self.time_str}
            Num Agents: {self.num_agents_l}
            Stagger Start Time: {self.stagger_start_time_dt}
            Multi-agent Planner Classes: {self.multi_agent_planner_class_l}
            Single Agent Planner: {self.single_agent_planner_class_l}
            Runtime Limit: {self.runtime_limit}
            Num Trials Per Combination: {self.num_trials_per_combination}
            Instance: {self.instance_name}
        """
        return print_str


class MultiAgentPlanningSingleTrialConfig:
    # The associated time str and trial number for saving purposes.
    time_str = None
    trial_number = 0
    # Runtime limit. In seconds.
    runtime_limit = 10
    # Num agents.
    num_agents = 1
    # Start and goal placement scheme.
    # Stagger start times. t_init[0] = 0, t_init[1] = dt * 1, ... t_init[i] = dt * i.
    stagger_start_time_dt = 0
    # Multi-agent planner class. Common options are CBS, ECBS, and PrioritizedPlanning.
    multi_agent_planner_class = ""
    # Single agent planner class. Common options are MPD and MPDEnsemble.
    single_agent_planner_class = ""
    # Whether to render animation or not.
    render_animation = False
    # The environment(s) to use for the multi-agent planning.
    instance_name = ""
    # The starts and goals, models ids, and skeletons.
    start_state_pos_l = []
    goal_state_pos_l = []
    global_model_ids = []
    # The model coord skeleton for each agent.
    agent_skeleton_l = []
    seed = 0

    def save(self, results_dir: str):
        # Save the config.
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(results_dir, 'config.pkl'), 'wb') as f:
            pickle.dump(self, f)

    def __str__(self):
        print_str = f"""
        Trial Config:
            Time Str: {self.time_str}
            Trial Number: {self.trial_number}
            Num Agents: {self.num_agents}
            Stagger Start Time: {self.stagger_start_time_dt}
            Multi-agent Planner Class: {self.multi_agent_planner_class}
            Single Agent Planner: {self.single_agent_planner_class}
            Instance: {self.instance_name}
        """
        return print_str


class TrialSuccessStatus(Enum):
    UNKNOWN = -1
    SUCCESS = 0
    FAIL_RUNTIME_LIMIT = 1
    FAIL_COLLISION_AGENTS = 2
    FAIL_NO_SOLUTION = 3

    def __bool__(self):
        return self == TrialSuccessStatus.SUCCESS


class MultiAgentPlanningSingleTrialResult:
    # The associated experiment config.
    trial_config: MultiAgentPlanningSingleTrialConfig = None
    # The agent paths. Each entry is of shape (H, 4).
    agent_path_l: List[torch.Tensor] = []
    # CT nodes expanded.
    num_ct_expansions: int = 0
    # Success. Reasons to fail: runtime limit, any collision of any agent in the solution, failing to find a solution.
    success_status: TrialSuccessStatus = TrialSuccessStatus.UNKNOWN
    # Number of agent pairs in collision.
    num_collisions_in_solution: int = 0
    # Our metric for determining how well a path is adhering to the data. Each agent is a yes/no for single maps
    # or a fraction for ensemble. This metric is the average adherence to the data.
    data_adherence: float = 0.0
    # Planning time.
    planning_time = 0.0
    # Path length averaged over all agents.
    path_length_per_agent = 0.0
    # Path smoothness.
    mean_path_acceleration_per_agent = 0.0
    mean_jerk = 0.0
    geometric_smoothness = 0.0  # Laplacian
    # Path Start and Goal.
    start_state_pos_l = []
    goal_state_pos_l = []
    # The environment(s) to use for the multi-agent planning.
    global_model_ids = []
    # The model coord skeleton for each agent.
    agent_skeleton_l = []

    def save(self, results_dir: str):
        # Save the results.
        Path(results_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(results_dir, 'results.pkl'), 'wb') as f:
            pickle.dump(self, f)

        # Save a human-readable version.
        with open(os.path.join(results_dir, 'results.txt'), 'w') as f:
            f.write(str(self))

    def __str__(self):
        print_str = f"""
        Trial Config Summary:
            Method: {self.trial_config.multi_agent_planner_class}
            Num Agents: {self.trial_config.num_agents}
            Instance: {self.trial_config.instance_name}
            Stagger Start Time: {self.trial_config.stagger_start_time_dt}
            Single Agent Planner: {self.trial_config.single_agent_planner_class}
        Planning Problem:
            start_state_pos_l: {self.start_state_pos_l}
            goal_state_pos_l: {self.goal_state_pos_l}
            global_model_ids: {self.global_model_ids}
            agent_skeleton_l: {self.agent_skeleton_l}
        Trial Results:
            success_status: {self.success_status}  
            num_collisions_in_solution: {self.num_collisions_in_solution}
            data_adherence: {self.data_adherence}
            planning_time: {self.planning_time}
            path_length_per_agent: {self.path_length_per_agent}
            mean_path_acceleration_per_agent: {self.mean_path_acceleration_per_agent}
            geometric_smoothness: {self.geometric_smoothness}
            mean_jerk: {self.mean_jerk}
            num_ct_expansions: {self.num_ct_expansions}
        """
        return print_str


# Utility functions.
def get_result_dir_from_time_str(time_str: str):
    # Create a results directory.
    results_dir = os.path.join('./results', f'{time_str}')
    results_dir = os.path.abspath(results_dir)
    return results_dir


def get_results_dir_from_experiment_config(experiment_config: MultiAgentPlanningExperimentConfig):
    results_dir = get_result_dir_from_time_str(experiment_config.time_str)
    # Create a results directory.
    results_dir = os.path.join(results_dir,
                               f'instance_name___{experiment_config.instance_name}')
    results_dir = os.path.abspath(results_dir)
    return results_dir


def get_result_dir_from_trial_config(trial_config: MultiAgentPlanningSingleTrialConfig,
                                     time_str: str = None,
                                     trial_number: int = 0):
    # If no time string is provided, use the current time.
    if time_str is None:
        raise ValueError("Time string must be provided.")
    results_dir = get_result_dir_from_time_str(time_str)
    # Create a results directory.
    results_dir = os.path.join(results_dir,
                               f'instance_name___{trial_config.instance_name}',
                               f'num_agents___{trial_config.num_agents}',
                               f'planner___{trial_config.multi_agent_planner_class}',
                               f'single_agent_planner___{trial_config.single_agent_planner_class}',
                               str(trial_number))
    results_dir = os.path.abspath(results_dir)
    return results_dir
