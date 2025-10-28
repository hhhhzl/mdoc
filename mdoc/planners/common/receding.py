from typing import List, Optional, Dict, Any, Type
import torch
from mdoc.planners.multi_agent import CBS
from mdoc.planners.common import PlannerOutput
from mdoc.common.conflicts import Conflict
from mdoc.common.constraints import Constraint


def _shift_warmstart(path_b: torch.Tensor) -> torch.Tensor:
    assert path_b.ndim == 3 and path_b.shape[0] == 1
    moved = torch.roll(path_b, shifts=-1, dims=1)
    tail = path_b[:, -1] + 0.1 * (path_b[:, -1] - path_b[:, -2])
    moved[:, -1] = tail
    return moved


def _estimate_first_control(path: torch.Tensor, dt: float, model="vel_cmd") -> torch.Tensor:
    # vel_cmd: 直接用位移 / dt 作为速度命令
    if model == "vel_cmd":
        return (path[1, :2] - path[0, :2]) / max(dt, 1e-3)
    elif model == "acc_cmd":
        v0 = (path[1, :2] - path[0, :2]) / max(dt, 1e-3)
        v1 = (path[2, :2] - path[1, :2]) / max(dt, 1e-3)
        return (v1 - v0) / max(dt, 1e-3)
    else:
        return torch.zeros(2, dtype=path.dtype, device=path.device)


class RecedingCBS:
    """
    every time interval call CBS.plan()
    """

    def __init__(
            self,
            low_level_planner_l: List[torch.nn.Module],
            reference_robot=None,
            reference_task=None,
            conflict_type_to_constraint_types: Dict[Type[Conflict], Type[Constraint]] = {},
            is_ecbs: bool = True,
            is_xcbs: bool = True,
            dt: float = 0.1,
            runtime_limit: float = 100,
    ):
        self.low_level_planner_l = low_level_planner_l
        self.reference_robot = reference_robot
        self.reference_task = reference_task
        self.conflict_map = conflict_type_to_constraint_types
        self.is_ecbs = is_ecbs
        self.is_xcbs = is_xcbs
        self.dt = dt
        self.runtime_limit = runtime_limit

        # warm-start trajectory（PathBatchExperience.path_b）
        self._warm_paths_b: List[Optional[torch.Tensor]] = [None] * len(low_level_planner_l)

    def step(
            self,
            start_l: List[torch.Tensor],
            goal_l: List[torch.Tensor],
            start_time_l: Optional[List[float]] = None,
    ):
        t0 = 0.0 if start_time_l is None else float(start_time_l[0])
        for p in self.low_level_planner_l:
            if hasattr(p, "set_time"):
                p.set_time(t0)

        cbs = CBS(
            low_level_planner_l=self.low_level_planner_l,
            start_l=start_l,
            goal_l=goal_l,
            start_time_l=start_time_l,
            is_xcbs=self.is_xcbs,
            is_ecbs=self.is_ecbs,
            conflict_type_to_constraint_types=self.conflict_map,
            reference_robot=self.reference_robot,
            reference_task=self.reference_task,
        )
        paths, _, status, _ = cbs.plan(runtime_limit=self.runtime_limit)
        u0_l = []
        for i, p in enumerate(paths):
            u0 = _estimate_first_control(p, dt=self.dt, model="vel_cmd")
            u0_l.append(u0)

            # right shift warm-start
            path_b = p.unsqueeze(0)  # (1,H,D)
            self._warm_paths_b[i] = _shift_warmstart(path_b)

            if hasattr(self.low_level_planner_l[i], "last_experience"):
                from mdoc.common.experiences import PathBatchExperience
                self.low_level_planner_l[i].last_experience = PathBatchExperience(self._warm_paths_b[i])

        return u0_l, paths, status
