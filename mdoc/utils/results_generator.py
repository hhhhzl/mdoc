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

_original_torch_load = torch.load


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
        if full_path[-11:] == 'results.pkl':
            with open(full_path, "rb") as f:
                data = pickle.load(f)

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
            dict['success'] = 1 if data.success_status.value == 0 else 0 # 1 is success, 0 is unsuccess
            dict['success_status'] = data.success_status.value # UNKNOWN = -1 SUCCESS = 0 FAIL_RUNTIME_LIMIT = 1 FAIL_COLLISION_AGENTS = 2 FAIL_NO_SOLUTION = 3
            dict['n_collision'] = data.num_collisions_in_solution
            dict['time'] = data.planning_time
            dict['path_length'] = data.path_length_per_agent
            dict['mean_acc'] = data.mean_path_acceleration_per_agent
            dict['mean_jerk'] = data.mean_jerk
            dict['mean_geo'] = data.geometric_smoothness
            dict['expansions'] = data.num_ct_expansions

            table.append(dict)

df = pd.DataFrame(table)
df = df[order_cols]
print(df.shape)
df.to_csv("results.csv")
