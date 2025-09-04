import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from itertools import permutations
from mp_baselines.planners.rrt_star import RRTStar

# -------------------------
# 1. Court Parameters
# -------------------------
COURT_WIDTH = 8.23
COURT_LENGTH = 23.77
OUTER_WIDTH = 15.0
OUTER_LENGTH = 30.0
NET_Y = COURT_LENGTH / 2
NET_THICKNESS = 0.5

# -------------------------
# 2. Robot Parameters
# -------------------------
ROBOT_RADIUS = 0.4

# -------------------------
# 2. Static Obstacles
# -------------------------
static_obstacles = [
    {"center": [OUTER_WIDTH / 2, OUTER_LENGTH - 2.0], "radius": 0.7},
    {"center": [OUTER_WIDTH / 2, 2.0], "radius": 0.7},
    {"center": [2.0, OUTER_LENGTH / 2], "radius": 0.5},
    {"center": [OUTER_WIDTH - 2.0, OUTER_LENGTH / 2], "radius": 0.5},
]

# -------------------------
# 3. Dynamic Obstacles
# -------------------------
dynamic_obstacles = [
    {"center": [5.0, 10.0], "radius": 0.5, "velocity": [0.08, 0.05]},
    {"center": [10.0, 15.0], "radius": 0.5, "velocity": [-0.05, 0.07]},
]


# -------------------------
# 4. Check for Collision
# -------------------------
def collision_fn(state, allow_cross_net=False):
    if state.ndim == 1:
        state = state.unsqueeze(0)
    x, y = state[:, 0], state[:, 1]

    out_of_bounds = (x < ROBOT_RADIUS) | (x > OUTER_WIDTH - ROBOT_RADIUS) | \
                    (y < ROBOT_RADIUS) | (y > OUTER_LENGTH - ROBOT_RADIUS)

    y_offset = (OUTER_LENGTH - COURT_LENGTH) / 2
    net_y_global = y_offset + NET_Y
    x_min = (OUTER_WIDTH - COURT_WIDTH) / 2
    x_max = (OUTER_WIDTH + COURT_WIDTH) / 2
    hit_net = torch.zeros(len(x), dtype=torch.bool)
    if not allow_cross_net:
        hit_net = ((y > net_y_global - NET_THICKNESS / 2 - ROBOT_RADIUS) &
                   (y < net_y_global + NET_THICKNESS / 2 + ROBOT_RADIUS) &
                   (x >= x_min - ROBOT_RADIUS) & (x <= x_max + ROBOT_RADIUS))

    hit_static = torch.zeros(len(x), dtype=torch.bool)
    for obs in static_obstacles:
        center = torch.tensor(obs["center"], dtype=torch.float32)
        dist = torch.norm(state - center, dim=1)
        hit_static |= dist < (obs["radius"] + ROBOT_RADIUS)

    hit_dynamic = torch.zeros(len(x), dtype=torch.bool)
    for obs in dynamic_obstacles:
        center = torch.tensor(obs["center"], dtype=torch.float32)
        dist = torch.norm(state - center, dim=1)
        hit_dynamic |= dist < (obs["radius"] + ROBOT_RADIUS)

    return out_of_bounds | hit_net | hit_static | hit_dynamic


# -------------------------
# Dummy Task
# -------------------------
class DummyTask:
    def __init__(self, bounds, device="cpu"):
        self.bounds = bounds.to(device)
        self.device = device

    def random_coll_free_q(self, n_samples, max_samples=None):
        samples = torch.rand((n_samples, self.bounds.shape[1]), device=self.device)
        return samples * (self.bounds[1] - self.bounds[0]) + self.bounds[0]

    def distance_q(self, x, y):
        return torch.norm(x - y, dim=-1)

    def extend_fn(self, start, end, max_step=0.2, max_dist=1.0):
        vec = end - start
        dist = torch.norm(vec)
        if dist <= max_step:
            return torch.stack([start, end])
        direction = vec / dist
        steps = int(torch.ceil(dist / max_step))
        path = torch.stack([start + direction * (i * max_step) for i in range(steps)] + [end])
        return path


def plan_path(start, goal, allow_cross_net=False, n_iters=8000, step_size=0.5, n_radius=2.5, device="cpu"):
    bounds = torch.tensor([[0.0, 0.0], [OUTER_WIDTH, OUTER_LENGTH]], device=device)
    task = DummyTask(bounds, device=device)

    planner = RRTStar(
        task=task,
        n_iters=n_iters,
        start_state_pos=torch.tensor(start, dtype=torch.float32, device=device),
        goal_state_pos=torch.tensor(goal, dtype=torch.float32, device=device),
        step_size=step_size,
        n_radius=n_radius,
        tensor_args={'device': device, 'dtype': torch.float32}
    )

    planner.collision_fn = lambda state: collision_fn(state, allow_cross_net=allow_cross_net)

    path = planner._run_optimization(n_iters)
    if path is not None and isinstance(path, list):
        path = torch.stack(path)
    return path


# -------------------------
# Muli-tasks planning
# -------------------------
def plan_multi_targets(start, targets):
    best_order, best_dist = None, float("inf")
    for perm in permutations(targets):
        dist, current = 0, np.array(start)
        for t in perm:
            dist += np.linalg.norm(np.array(t) - current)
            current = np.array(t)
        if dist < best_dist:
            best_dist, best_order = dist, perm
    return list(best_order)


def greedy_target_order(start, targets):
    order = []
    current = np.array(start)
    remaining = targets.copy()
    while remaining:
        distances = [np.linalg.norm(np.array(t) - current) for t in remaining]
        idx = np.argmin(distances)
        order.append(remaining.pop(idx))
        current = np.array(order[-1])
    return order


# -------------------------
# Navigation
# -------------------------
def online_navigation(start, targets, device, allow_cross_net=False, return_back=True):
    current_pos = np.array(start)
    path_all = [current_pos.copy()]
    order = greedy_target_order(start, targets)
    if return_back:
        order.append(start)

    for target in order:
        print(f"Planning path to target: {target}")
        path = plan_path(current_pos, target, allow_cross_net=allow_cross_net, device=device)
        if path is None:
            continue
        for point in path.cpu().numpy():
            for obs in dynamic_obstacles:
                obs["center"][0] += obs["velocity"][0]
                obs["center"][1] += obs["velocity"][1]
                if obs["center"][0] <= 0 or obs["center"][0] >= OUTER_WIDTH:
                    obs["velocity"][0] *= -1
                if obs["center"][1] <= 0 or obs["center"][1] >= OUTER_LENGTH:
                    obs["velocity"][1] *= -1
            path_all.append(point)
            current_pos = point
    return np.array(path_all)


# -------------------------
# draw_court
# -------------------------
def draw_court(ax, start, targets):
    outer_x, outer_y = [0, OUTER_WIDTH, OUTER_WIDTH, 0, 0], [0, 0, OUTER_LENGTH, OUTER_LENGTH, 0]
    ax.plot(outer_x, outer_y, 'k-', linewidth=2, label="Outer Court")

    y_offset = (OUTER_LENGTH - COURT_LENGTH) / 2
    x_min, x_max = (OUTER_WIDTH - COURT_WIDTH) / 2, (OUTER_WIDTH + COURT_WIDTH) / 2
    court_x = [x_min, x_max, x_max, x_min, x_min]
    court_y = [y_offset, y_offset, y_offset + COURT_LENGTH, y_offset + COURT_LENGTH, y_offset]
    ax.plot(court_x, court_y, 'g-', linewidth=2, label="Main Court")
    net_y_global = y_offset + NET_Y
    ax.plot([x_min, x_max], [net_y_global, net_y_global], 'r--', linewidth=2, label="Net")

    for obs in static_obstacles:
        ax.add_patch(plt.Circle(obs["center"], obs["radius"], color='gray', alpha=0.6))

    ax.plot(start[0], start[1], 'go', markersize=10, label="Start")
    for goal in targets:
        ax.plot(goal[0], goal[1], 'ro', markersize=8)


def animate_path(path, start, targets):
    fig, ax = plt.subplots(figsize=(8, 12))
    ax.set_xlim(0, OUTER_WIDTH)
    ax.set_ylim(0, OUTER_LENGTH)
    ax.set_aspect('equal', adjustable='box')
    ax.set_title("Motion Planning and Obstacles Avoidance")

    patches = []
    draw_court(ax, start, targets)
    for obs in dynamic_obstacles:
        patch = plt.Circle(obs["center"], obs["radius"], color='orange', alpha=0.7)
        patches.append(patch)
        ax.add_patch(patch)

    if path is None:
        plt.legend()
        plt.grid(True)
        plt.show()
        return

    path_np = path.copy()
    line, = ax.plot([], [], 'b-', linewidth=2)
    point, = ax.plot([], [], 'bo', markersize=5)

    def update(i):
        for obs, patch in zip(dynamic_obstacles, patches):
            obs["center"][0] += obs["velocity"][0]
            obs["center"][1] += obs["velocity"][1]

            if obs["center"][0] <= 0 or obs["center"][0] >= OUTER_WIDTH:
                obs["velocity"][0] *= -1
            if obs["center"][1] <= 0 or obs["center"][1] >= OUTER_LENGTH:
                obs["velocity"][1] *= -1

            patch.center = obs["center"]

        line.set_data(path_np[:i + 1, 0], path_np[:i + 1, 1])
        point.set_data([path_np[i, 0]], [path_np[i, 1]])
        return [line, point, *patches]

    ani = FuncAnimation(fig, update, frames=len(path_np), interval=200, blit=True)
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    start = [1, 1.0]
    allow_cross_net = False
    device = 'cpu'

    np.random.seed(1)
    targets = []
    while len(targets) < 20:
        x = np.random.uniform(0.5, OUTER_WIDTH - 0.5)
        y = np.random.uniform(0.5, OUTER_LENGTH - 0.5)
        if all(np.linalg.norm(np.array([x, y]) - np.array(obs["center"])) > obs["radius"] + 0.5 for obs in
               static_obstacles):
            targets.append([x, y])

    path = online_navigation(start, targets, device, allow_cross_net=allow_cross_net)
    if path is None or len(path) == 0:
        print("No valid path found.")
    else:
        animate_path(path, start, targets)
