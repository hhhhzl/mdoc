import numpy as np
import torch
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import patches
from torch_robotics.environments import (
    EnvConveyor2D,
    EnvEmpty2D,
    EnvRandom2D,
    EnvRandomDense2D,
    EnvRandomLarge2D,
    EnvTennis2D
)
from torch_robotics.torch_utils.torch_utils import get_default_tensor_args
from torch_robotics.visualizers.planning_visualizer import create_fig_and_axes
from mdoc.common import (
    get_start_goal_pos_boundary,
    get_start_goal_pos_circle,
    get_start_goal_pos_random_in_env
)

if __name__ == "__main__":
    map = {
        #"empty": EnvEmpty2D,
        #"conveyor": EnvConveyor2D,
        "narrow": EnvTennis2D,
        #"random": EnvRandom2D,
        #"random_dense": EnvRandomDense2D,
        #"random_large": EnvRandomLarge2D
    }
    tensor_args = get_default_tensor_args()
    device = tensor_args['device']
    random_env_settings = {
        "random": {
            "num_agents": 6,
            "margin": 0.25,
            "obstacle_margin": 0.08,
        },
        "random_dense": {
            "num_agents": 6,
            "margin": 0.3,
            "obstacle_margin": 0.08,
        },
        "random_large": {
            "num_agents": 15,
            "margin": 0.5,
            "obstacle_margin": 0.08,
        },
    }
    output_dir = Path(__file__).parent / "generated_maps"
    output_dir.mkdir(parents=True, exist_ok=True)
    for key, _env in map.items():
        env = _env(
            precompute_sdf_obj_fixed=True,
            sdf_cell_size=0.01,
            tensor_args=get_default_tensor_args()
        )
        fig, ax = create_fig_and_axes(env.dim, facecolor="black", rect=True)
        env.render(ax, object_c='#D3D3D3')
        if key in random_env_settings:
            config = random_env_settings[key]
            start_state_pos_l, goal_state_pos_l = get_start_goal_pos_random_in_env(
                config["num_agents"],
                env,
                tensor_args,
                margin=config["margin"],
                obstacle_margin=config["obstacle_margin"],
                reload_env=False,
                size=2,
            )
        elif key == 'narrow':
            start_state_pos_l = [torch.tensor([-0.8, 0.0], dtype=torch.float32, device=device)]
            goal_state_pos_l = [torch.tensor([0.8, 0.0], dtype=torch.float32, device=device)]
        else:
            if key == "empty":
                start_state_pos_l, goal_state_pos_l = get_start_goal_pos_boundary(num_agents=12, dist=0.87, device=device)
            else:
                start_state_pos_l, goal_state_pos_l = get_start_goal_pos_circle(num_agents=12, radius=0.8, device=device)

        num_agents = len(start_state_pos_l)
        color_indices = np.random.permutation(20)[:num_agents]
        for idx, (start_state_pos, goal_state_pos) in enumerate(zip(start_state_pos_l, goal_state_pos_l)):
            color = plt.cm.tab20(color_indices[idx])
            start_xy = start_state_pos.detach().cpu().numpy()
            goal_xy = goal_state_pos.detach().cpu().numpy()
            if key not in ['empty', 'conveyor']:
                start_marker = patches.Rectangle(
                    (start_xy[0] - 0.025, start_xy[1] - 0.025),
                    0.05,
                    0.05,
                    facecolor=color,
                    edgecolor='none',
                    linewidth=0
                )
            else:
                start_marker = patches.Circle(
                    (start_xy[0], start_xy[1]),
                    radius=0.025,
                    facecolor=color,
                    edgecolor='none',
                    linewidth=0
            )

            goal_marker = patches.Circle(
                (goal_xy[0], goal_xy[1]),
                radius=0.025,
                facecolor=color,
                edgecolor='none',
                linewidth=0
            )
            ax.add_patch(start_marker)
            ax.add_patch(goal_marker)
        ax.axis('off')
        fig.savefig(output_dir / f"{key}.png", bbox_inches="tight", pad_inches=0.05)
        # plt.show()