import torch
from torch_robotics.environments import EnvHighway3Lane2D
from torch_robotics.robots.robot_planar_disk import RobotPlanarDisk
from torch_robotics.tasks.tasks_ensemble import PlanningTask, PlanningTaskEnsemble
from mdoc.planners.common.receding import RecedingCBS
from mdoc.planners.common.dynamic_adapter import LowerDynamicAdapter
from mdoc.baselines.cbmpc_lower import CBMPCLower
from torch_robotics.torch_utils.torch_utils import get_default_tensor_args
from mdoc.baselines.vis import HighwayVisualizer

env = EnvHighway3Lane2D(
    map_size=2.0,
    lane_sep=0.18,
    tensor_args=get_default_tensor_args('cpu')
)
dt, H = 0.1, 60
map_size = 2.0

mpc_lower = CBMPCLower(
    H=H,
    dt=dt,
    dynamics="single_integrator",
    device="cpu",
    results_dir='./',
    Q=12.5,
    R=0.05,
    P=12.5,
    v_max=0.05,
    a_max=0.05,
)

robot = RobotPlanarDisk(
    radius=0.05,
    q_limits=torch.tensor([[-2, -2], [2, 2]]),
    tensor_args=get_default_tensor_args('cpu')
)
tasks = {
    0: PlanningTask(
        env=env,
        robot=robot,
        q_limits=torch.tensor([[-2, -2], [2, 2]]),
        tensor_args=get_default_tensor_args('cpu')
    )
}
task = PlanningTaskEnsemble(
    tasks=tasks,
    transforms={0: torch.tensor([0, 0])},
    tensor_args=get_default_tensor_args('cpu')
)
mpc_lower_dyn = LowerDynamicAdapter(
    mpc_lower,
    env,
    H=H,
    dt=dt,
    map_size=map_size,
    radius_scale=1.15,
    hard=True
)
mpc_lower_dyn.robot = robot
mpc_lower_dyn.task = task

lowers = [mpc_lower_dyn]

runner = RecedingCBS(
    low_level_planner_l=lowers,
    is_ecbs=False,
    is_xcbs=False,
    dt=dt,
    runtime_limit=10,
)

start_l = [torch.tensor([-1.5, -0.18])]
goal_l = [torch.tensor([+2, -0.18])]

viz = HighwayVisualizer(
    env=env,
    map_size=map_size,
    dt=dt,
    H=H,
    results_dir="./",
)

t0 = 0.0
for k in range(120):
    for p in lowers:
        if hasattr(p, "set_time"):
            p.set_time(t0)
    u0_l, paths, status = runner.step(start_l, goal_l, start_time_l=[int(round(t0 / dt))] * len(lowers))
    viz.draw_step(t0, paths, show_dynamic_obstacles=True)

    next_l = []
    for i, p in enumerate(paths):
        next_l.append(p[1].detach().clone())
    start_l = next_l
    t0 += dt

gif_path = viz.save_gif("rollout.gif", fps=int(1/dt))
print("Saved GIF:", gif_path)
