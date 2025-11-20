"""
MIT License
"""
# Standard imports.

# Project includes.
from mdoc.common.experiments.experiment_utils import *
from inference_multi_agent import run_multi_agent_trial
import time
from datetime import datetime
import argparse
from scripts import (
    MultiAgentPlannerType,
    EnvironmentType,
    LowerPlannerMethodType
)


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-agent planning experiment configuration")

    # Number of agents
    parser.add_argument(
        '--n',
        nargs='+',
        type=int,
        default=[3, 6, 9, 12, 15, 20, 25, 30, 35, 40],
        help='List of number of agents to test'
    )

    # Environment selection
    parser.add_argument(
        '--e',
        type=str,
        default=EnvironmentType.EMPTY_DISK_LARGE.value,
        choices=EnvironmentType.choices(),
        help='Environment/instance to use for the experiment'
    )

    # Timing
    parser.add_argument(
        '--st',
        type=float,
        default=0,
        help='Stagger start time between agents'
    )

    # Planner configuration (using Enum choices)
    parser.add_argument(
        '--hps',
        nargs='+',
        default=[MultiAgentPlannerType.CBS.value],
        choices=MultiAgentPlannerType.choices(),
        help='List of multi-agent planners to test'
    )

    parser.add_argument(
        '--lps',
        nargs='+',
        default=[LowerPlannerMethodType.MMD.value],
        choices=LowerPlannerMethodType.choices(),
        help='Single agent planner to use'
    )

    # Experiment parameters
    parser.add_argument(
        '--rl',
        type=int,
        default=1000,
        help='Runtime limit in seconds'
    )
    parser.add_argument(
        '--nt',
        type=int,
        default=10,
        help='Number of trials to run for each configuration'
    )
    parser.add_argument(
        '--ra',
        action='store_true',
        default=True,
        help='Whether to render animations'
    )

    return parser.parse_args()


def run_multi_agent_experiment(experiment_config: MultiAgentPlanningExperimentConfig):
    # Run the multi-agent planning experiment.
    startt = time.time()
    # Create the experiment config.
    experiment_config.time_str = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    # Get single trial configs from the experiment config.
    single_trial_configs = experiment_config.get_single_trial_configs_from_experiment_config()
    # So let's run sequentially.
    for single_trial_config in single_trial_configs:
        print(single_trial_config)
        try:
            run_multi_agent_trial(single_trial_config)
            # Aggregate and save data on every step. This is not needed (can be done once at the end).
        except Exception as e:
            print("Error in run_multi_agent_experiment: ", e)
            # Save to a file.
            with open(f"/results/error_{experiment_config.time_str}.txt", "a") as f:
                f.write(str(e))
                f.write("This is for single_trial_config: ")
                f.write(str(single_trial_config))
                f.write("\n")
            continue

    # Print the runtime.
    print("Runtime: ", time.time() - startt)
    print("Run: OK.")


if __name__ == "__main__":
    # Instance names. These dictate the maps and start/goals.
    args = parse_args()

    # Create an experiment config.
    experiment_config = MultiAgentPlanningExperimentConfig()

    # Set the experiment config.
    assert all(x > 1 for x in args.n), "Multi-agent experiment should have number of agents > 1"
    experiment_config.num_agents_l = args.n
    experiment_config.instance_name = EnvironmentType.from_string(args.e).value
    experiment_config.stagger_start_time_dt = args.st
    experiment_config.multi_agent_planner_class_l = args.hps  # , "ECBS", "PP", "XCBS", "CBS"]
    experiment_config.single_agent_planner_class_l = args.lps
    experiment_config.runtime_limit = args.rl
    experiment_config.num_trials_per_combination = args.nt
    experiment_config.render_animation = args.ra

    # Run the experiment.
    run_multi_agent_experiment(experiment_config)
