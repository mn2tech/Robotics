"""Base planner interface and shared types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Tuple

from src.environment import WarehouseEnvironment

Position = Tuple[int, int]


@dataclass
class PlanResult:
    """Output of a path planning call."""

    path: List[Position]
    cost: float
    runtime_ms: float
    success: bool
    nodes_expanded: int = 0
    message: str = ""


class BasePlanner(ABC):
    """Abstract path planner; implementations must return a grid path."""

    name: str = "base"

    @abstractmethod
    def plan(
        self,
        env: WarehouseEnvironment,
        start: Optional[Position] = None,
        goal: Optional[Position] = None,
    ) -> PlanResult:
        raise NotImplementedError

    @staticmethod
    def path_length(path: List[Position]) -> float:
        if len(path) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(path)):
            r0, c0 = path[i - 1]
            r1, c1 = path[i]
            total += ((r1 - r0) ** 2 + (c1 - c0) ** 2) ** 0.5
        return total

    @staticmethod
    def path_step_count(path: List[Position]) -> int:
        """Number of grid moves in a route (positions minus one)."""
        return max(0, len(path) - 1)

    @staticmethod
    def reconstruct_path(came_from: dict, start: Position, goal: Position) -> List[Position]:
        current = goal
        path = [current]
        while current != start:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path
