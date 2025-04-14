"""
"""
from abc import ABC, abstractmethod
import torch

# MDOC imports.
from mdoc.planners.common import PlannerOutput


class SingleAgentPlanner(ABC):
    @abstractmethod
    def __call__(self, start: torch.Tensor, goal: torch.Tensor, constraint_l=None, *args, **kwargs) -> PlannerOutput:
        pass
