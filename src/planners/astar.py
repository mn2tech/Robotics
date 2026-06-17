"""A* path planning on a 4-connected warehouse grid."""

from __future__ import annotations

import heapq
import time
from typing import Dict, List, Optional, Tuple

from src.environment import WarehouseEnvironment
from src.planners.base import BasePlanner, PlanResult, Position


class AStarPlanner(BasePlanner):
    """A* search with Manhattan heuristic."""

    name = "A*"

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

        open_heap: List[Tuple[float, int, Position]] = []
        counter = 0
        g_score: Dict[Position, float] = {start: 0.0}
        f_score: Dict[Position, float] = {start: self._heuristic(start, goal)}
        came_from: Dict[Position, Position] = {}
        closed: set = set()
        nodes_expanded = 0

        heapq.heappush(open_heap, (f_score[start], counter, start))

        while open_heap:
            _, _, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            closed.add(current)
            nodes_expanded += 1

            if current == goal:
                path = self.reconstruct_path(came_from, start, goal)
                runtime_ms = (time.perf_counter() - t0) * 1000
                return PlanResult(
                    path=path,
                    cost=g_score[goal],
                    runtime_ms=runtime_ms,
                    success=True,
                    nodes_expanded=nodes_expanded,
                )

            for neighbor in env.neighbors(*current):
                if neighbor in closed:
                    continue
                tentative = g_score[current] + 1.0
                if tentative < g_score.get(neighbor, float("inf")):
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative
                    f_score[neighbor] = tentative + self._heuristic(neighbor, goal)
                    counter += 1
                    heapq.heappush(open_heap, (f_score[neighbor], counter, neighbor))

        runtime_ms = (time.perf_counter() - t0) * 1000
        return PlanResult(
            path=[],
            cost=float("inf"),
            runtime_ms=runtime_ms,
            success=False,
            nodes_expanded=nodes_expanded,
            message="No path found",
        )

    @staticmethod
    def _heuristic(a: Position, b: Position) -> float:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
