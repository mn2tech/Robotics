"""Dijkstra's algorithm for optimal grid path planning."""

from __future__ import annotations

import heapq
import time
from typing import Dict, List, Optional, Tuple

from src.environment import WarehouseEnvironment
from src.planners.base import BasePlanner, PlanResult, Position


class DijkstraPlanner(BasePlanner):
    """Uniform-cost search (Dijkstra) on a 4-connected grid."""

    name = "Dijkstra"

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
        dist: Dict[Position, float] = {start: 0.0}
        came_from: Dict[Position, Position] = {}
        visited: set = set()
        nodes_expanded = 0

        heapq.heappush(open_heap, (0.0, counter, start))

        while open_heap:
            cost, _, current = heapq.heappop(open_heap)
            if current in visited:
                continue
            visited.add(current)
            nodes_expanded += 1

            if current == goal:
                path = self.reconstruct_path(came_from, start, goal)
                runtime_ms = (time.perf_counter() - t0) * 1000
                return PlanResult(
                    path=path,
                    cost=cost,
                    runtime_ms=runtime_ms,
                    success=True,
                    nodes_expanded=nodes_expanded,
                )

            for neighbor in env.neighbors(*current):
                if neighbor in visited:
                    continue
                new_cost = cost + 1.0
                if new_cost < dist.get(neighbor, float("inf")):
                    dist[neighbor] = new_cost
                    came_from[neighbor] = current
                    counter += 1
                    heapq.heappush(open_heap, (new_cost, counter, neighbor))

        runtime_ms = (time.perf_counter() - t0) * 1000
        return PlanResult(
            path=[],
            cost=float("inf"),
            runtime_ms=runtime_ms,
            success=False,
            nodes_expanded=nodes_expanded,
            message="No path found",
        )
