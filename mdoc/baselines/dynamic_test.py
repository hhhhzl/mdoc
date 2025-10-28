import torch
import numpy as np
from torch_robotics.environments import EnvHighway3Lane2D, EnvRandomLarge2D
from torch_robotics.robots.robot_planar_disk import RobotPlanarDisk
from torch_robotics.tasks.tasks_ensemble import PlanningTask, PlanningTaskEnsemble
from mdoc.planners.common.receding import RecedingCBS
from mdoc.planners.common.dynamic_adapter import LowerDynamicAdapter
from mdoc.baselines.cbmpc_lower import CBMPCLower
from torch_robotics.torch_utils.torch_utils import get_default_tensor_args
from mdoc.baselines.vis import OnlineRenderer

env = EnvHighway3Lane2D(
    map_size=2.0,
    lane_sep=0.18,
    tensor_args=get_default_tensor_args('cpu')
)
# env = EnvRandomLarge2D(precompute_sdf_obj_fixed=True, sdf_cell_size=0.01, tensor_args=get_default_tensor_args('cpu'))
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
    v_max=0.005,
    a_max=0.005,
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
    runtime_limit=60,
)

start_l = [torch.tensor([-2, -0.18])]
goal_l = [torch.tensor([+2, -0.18])]

assert len(start_l) == len(goal_l) == len(lowers)

vis = OnlineRenderer(env, dt=dt, map_size=map_size, agent_radius=robot.radius, show_live=True)

t0 = 0.0
for k in range(50):
    print(k, ">>>>")
    for p in lowers:
        if hasattr(p, "set_time"):
            p.set_time(t0)
    u0_l, paths, status = runner.step(start_l, goal_l, start_time_l=[int(round(t0 / dt))] * len(lowers))

    agent_xy = [p[0].tolist() for p in paths]
    future_paths = [p[:, :H].cpu().numpy() for p in paths]

    # hdgs = []
    # for p in paths:
    #     v = (p[1, :2] - p[0, :2]).cpu().numpy()
    #     hdgs.append(float(np.arctan2(v[1], v[0]) if np.linalg.norm(v) > 1e-9 else 0.0))
    # vis.add_frame(t0, agent_xy, headings=hdgs)

    vis.add_frame(t0, agent_xy, trails=None, render_sdf=False, render_grad=False, title=f"3-Lane + Ramp Merge @ t={t0:.1f}s")

    start_l = [p[1].detach().clone() for p in paths]
    t0 += dt

vis.save_gif("rollout.gif", fps=10)