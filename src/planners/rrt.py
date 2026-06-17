"""
RRT (Rapidly-exploring Random Tree) path planner.

Operates in continuous (row, col) space and snaps to the nearest walkable
grid cell, making it easy to later port to SE(2) or 3D configuration space.
"""

from __future__ import annotations

import random
import time
from typing import Dict, List, Optional, Tuple

import numpy as np

from src.environment import WarehouseEnvironment
from src.planners.base import BasePlanner, PlanResult, Position


class RRTPlanner(BasePlanner):
    """RRT with grid snapping and straight-line steering."""

    name = "RRT"

    def __init__(
        self,
        max_iterations: int = 2000,
        step_size: float = 1.5,
        goal_sample_rate: float = 0.1,
        seed: Optional[int] = None,
    ) -> None:
        self.max_iterations = max_iterations
        self.step_size = step_size
        self.goal_sample_rate = goal_sample_rate
        self.rng = random.Random(seed)

    def plan(
        self,
        env: WarehouseEnvironment,
        start: Optional[Position] = None,
        goal: Optional[Position] = None,
    ) -> PlanResult:
        start = start or env.start
        goal = goal or env.goal
        t0 = time.perf_counter()

        if not env.is_walkable(*start) or not env.is_walkable(*goal):
            return PlanResult(
                path=[],
                cost=float("inf"),
                runtime_ms=(time.perf_counter() - t0) * 1000,
                success=False,
                message="Start or goal not walkable",
            )

        start_f = (float(start[0]), float(start[1]))
        goal_f = (float(goal[0]), float(goal[1]))
        tree: Dict[Tuple[float, float], Tuple[float, float]] = {start_f: start_f}
        nodes_expanded = 0

        for _ in range(self.max_iterations):
            if self.rng.random() < self.goal_sample_rate:
                sample = goal_f
            else:
                sample = (
                    self.rng.uniform(0, env.height - 1),
                    self.rng.uniform(0, env.width - 1),
                )

            nearest = min(tree.keys(), key=lambda n: self._dist(n, sample))
            new_node = self._steer(nearest, sample)

            if not self._collision_free(env, nearest, new_node):
                continue

            tree[new_node] = nearest
            nodes_expanded += 1

            if self._dist(new_node, goal_f) <= self.step_size:
                if self._collision_free(env, new_node, goal_f):
                    tree[goal_f] = new_node
                    path_f = self._extract_path(tree, start_f, goal_f)
                    path = self._snap_path(env, path_f)
                    runtime_ms = (time.perf_counter() - t0) * 1000
                    return PlanResult(
                        path=path,
                        cost=self.path_length(path),
                        runtime_ms=runtime_ms,
                        success=True,
                        nodes_expanded=nodes_expanded,
                    )

        runtime_ms = (time.perf_counter() - t0) * 1000
        return PlanResult(
            path=[],
            cost=float("inf"),
            runtime_ms=runtime_ms,
            success=False,
            nodes_expanded=nodes_expanded,
            message="RRT failed to find path within iteration limit",
        )

    def _dist(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        return ((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5

    def _steer(
        self, from_node: Tuple[float, float], to_node: Tuple[float, float]
    ) -> Tuple[float, float]:
        d = self._dist(from_node, to_node)
        if d <= self.step_size:
            return to_node
        ratio = self.step_size / d
        return (
            from_node[0] + (to_node[0] - from_node[0]) * ratio,
            from_node[1] + (to_node[1] - from_node[1]) * ratio,
        )

    def _collision_free(
        self,
        env: WarehouseEnvironment,
        a: Tuple[float, float],
        b: Tuple[float, float],
    ) -> bool:
        steps = max(int(self._dist(a, b) * 2), 1)
        for i in range(steps + 1):
            t = i / steps
            r = int(round(a[0] + (b[0] - a[0]) * t))
            c = int(round(a[1] + (b[1] - a[1]) * t))
            if not env.is_walkable(r, c):
                return False
        return True

    def _extract_path(
        self,
        tree: Dict[Tuple[float, float], Tuple[float, float]],
        start: Tuple[float, float],
        goal: Tuple[float, float],
    ) -> List[Tuple[float, float]]:
        path = [goal]
        current = goal
        while current != start:
            current = tree[current]
            path.append(current)
        path.reverse()
        return path

    def _snap_path(
        self, env: WarehouseEnvironment, path_f: List[Tuple[float, float]]
    ) -> List[Position]:
        """Convert continuous path to walkable grid waypoints."""
        snapped: List[Position] = []
        for pr, pc in path_f:
            r, c = int(round(pr)), int(round(pc))
            if env.in_bounds(r, c) and env.is_walkable(r, c):
                cell = (r, c)
                if not snapped or snapped[-1] != cell:
                    snapped.append(cell)
        if snapped and snapped[-1] != env.goal:
            snapped.append(env.goal)
        return snapped
