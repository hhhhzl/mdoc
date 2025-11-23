"""
Input:
result folder
 - instance_name___ConveyorBoundary
 - instance_name___ConveyorRandom
 - instance_name___ConveyorCircle
 - instance_name___EmptyRandom
 - instance_name___EmptyCircle
 .....
 Name can be defined by yourself
"""

"""
Output:
dataframe field:
    instance_name
    num_agents
    planner
    single_agent_planner
    trail
    success
    n_collision
    time
    path_length
    mean_acc
    mean_jerk
    mean_geo
    expansions
    dynamic
    diffusion_step
    constraint
    trail_path
"""

import os
import torch
import pickle
from pathlib import Path
import pandas as pd
import json
import numpy as np

_original_torch_load = torch.load

import sys


def safe_to_json(obj):
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj

    if isinstance(obj, (np.float32, np.float64, np.int32, np.int64)):
        return obj.item()

    if isinstance(obj, np.ndarray):
        return obj.tolist()

    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()

    if isinstance(obj, (list, tuple, set)):
        return [safe_to_json(x) for x in obj]

    if isinstance(obj, dict):
        return {str(k): safe_to_json(v) for k, v in obj.items()}

    try:
        from pathlib import Path
        if isinstance(obj, Path):
            return str(obj)
    except:
        pass

    return str(obj)


def estimate_json_size_mb(obj) -> float:
    json_obj = safe_to_json(obj)
    json_str = json.dumps(json_obj, ensure_ascii=False, indent=2)
    size_bytes = len(json_str.encode("utf-8"))
    return size_bytes / (1024 * 1024)


def torch_load_cpu(f, *args, **kwargs):
    # Always map to CPU, even for nested torch.load(io.BytesIO(...))
    kwargs["map_location"] = torch.device("cpu")
    return _original_torch_load(f, *args, **kwargs)


torch.load = torch_load_cpu


def parse_path(path):
    p = Path(path)
    fields = {}

    for part in p.parents:
        name = part.name
        if "___" in name:
            key, value = name.split("___")
            if key in ['num_agents', 'trail']:
                fields[key] = int(value)
            else:
                fields[key] = value

    fields["trail"] = p.parent.name
    return fields


def to_json(data):
    dict = {}
    dict['Method'] = data.trial_config.multi_agent_planner_class
    dict['Num Agents'] = data.trial_config.num_agents
    dict['Stagger Start Time'] = data.trial_config.stagger_start_time_dt
    dict['Single Agent Planner'] = data.trial_config.single_agent_planner_class
    dict['agent_path_l'] = data.agent_path_l
    dict['start_state_pos_l'] = data.start_state_pos_l
    dict['goal_state_pos_l'] = data.goal_state_pos_l
    dict['success'] = 1 if data.success_status.value == 0 else 0  # 1 is success, 0 is unsuccess
    dict[
        'success_status'] = data.success_status.value  # UNKNOWN = -1 SUCCESS = 0 FAIL_RUNTIME_LIMIT = 1 FAIL_COLLISION_AGENTS = 2 FAIL_NO_SOLUTION = 3
    dict['n_collision'] = data.num_collisions_in_solution
    dict['planning_time'] = data.planning_time
    dict['data_adherence'] = data.data_adherence
    dict['path_length_per_agent'] = data.path_length_per_agent
    dict['mean_acc'] = data.mean_path_acceleration_per_agent
    dict['mean_jerk'] = data.mean_jerk
    dict['mean_geo'] = data.geometric_smoothness
    dict['expansions'] = data.num_ct_expansions
    return dict


def save_to_json():
    folder = "/Users/zhilinhe/desktop/experiment_results"

    for root, dirs, files in os.walk(folder):
        for f in files:
            full_path = os.path.join(root, f)
            if full_path[-11:] == 'results.pkl':
                with open(full_path, "rb") as f:
                    data = pickle.load(f)

                    d_json = safe_to_json(to_json(data))
                    save_path = full_path[:-11] + "results.json"

                    size_mb_est = estimate_json_size_mb(d_json)
                    print(f"Estimated JSON size (in memory): {size_mb_est:.3f} MB")

                    with open(save_path, "w", encoding="utf-8") as wf:
                        json.dump(d_json, wf, indent=2, ensure_ascii=False)

                    print(f"Saved to: {save_path}")

def delete_pkl():
    folder = "/Users/zhilinhe/desktop/experiment_results"

    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.endswith("results.pkl"):
                full_path = os.path.join(root, f)

                try:
                    os.remove(full_path)
                    print(f"Deleted: {full_path}")
                except Exception as e:
                    print(f"Failed to delete {full_path}: {e}")


def main():
    order_cols = [
        "instance_name",
        "num_agents",
        "planner",
        "single_agent_planner",
        "trail",
        "success",
        "n_collision",
        "time",
        "path_length",
        "mean_acc",
        "mean_jerk",
        "mean_geo",
        "expansions",
        "dynamic",
        "diffusion_step",
        "constraint",
        "trail_path",
    ]
    folder = "/Users/zhilinhe/desktop/experiment_results"

    table = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            full_path = os.path.join(root, f)
            if full_path[-12:] == 'results.json':
                with open(full_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                dict = parse_path(full_path)
                if "MDOC" in dict['single_agent_planner'] and dict["planner"] == 'CBS':
                    types = dict['single_agent_planner'].split("_")
                    dict['dynamic'] = types[1]
                    dict['diffusion_step'] = int(types[2])
                    dict['constraint'] = types[3]
                else:
                    dict['dynamic'] = None
                    dict['diffusion_step'] = None
                    dict['constraint'] = None

                dict['trail_path'] = full_path[:-12]
                # dict['success'] = 1 if data.success_status.value == 0 else 0  # 1 is success, 0 is unsuccess
                # dict[
                #     'success_status'] = data.success_status.value  # UNKNOWN = -1 SUCCESS = 0 FAIL_RUNTIME_LIMIT = 1 FAIL_COLLISION_AGENTS = 2 FAIL_NO_SOLUTION = 3
                # dict['n_collision'] = data.num_collisions_in_solution
                # dict['time'] = data.planning_time
                # dict['path_length'] = data.path_length_per_agent
                # dict['mean_acc'] = data.mean_path_acceleration_per_agent
                # dict['mean_jerk'] = data.mean_jerk
                # dict['mean_geo'] = data.geometric_smoothness
                # dict['expansions'] = data.num_ct_expansions

                dict['success'] = data['success']  # 1 is success, 0 is unsuccess
                dict[
                    'success_status'] = data['success_status']  # UNKNOWN = -1 SUCCESS = 0 FAIL_RUNTIME_LIMIT = 1 FAIL_COLLISION_AGENTS = 2 FAIL_NO_SOLUTION = 3
                dict['n_collision'] = data['n_collision']
                dict['time'] = data['planning_time']
                dict['path_length'] = data['path_length_per_agent']
                dict['mean_acc'] = data['mean_acc']
                dict['mean_jerk'] = data['mean_jerk']
                dict['mean_geo'] = data['mean_geo']
                dict['expansions'] = data['expansions']

                table.append(dict)

    df = pd.DataFrame(table)
    df = df[order_cols]
    print(df.shape)
    df.to_csv("results.csv")


if __name__ == "__main__":
    # save_to_json()
    # delete_pkl()
    main()
