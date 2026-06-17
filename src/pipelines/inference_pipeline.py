"""Inference pipeline: environment + sensor data -> movement decisions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from src.environment import WarehouseEnvironment
from src.planners.base import BasePlanner, PlanResult, Position
from src.robot import LidarScan, Robot

MovementDecision = Tuple[str, Optional[Position]]


@dataclass
class InferencePipeline:
    """
    Rule-based inference pipeline that mimics a deployed navigation stack.

    Input:  occupancy grid features + LiDAR scan + current pose
    Output: movement command (follow path, replan, stop, wait)

    Designed so a neural policy can replace `decide_action` later without
    changing the simulation harness.
    """

    planner: BasePlanner
    obstacle_proximity_cells: int = 2

    def build_observation(
        self, env: WarehouseEnvironment, robot: Robot
    ) -> dict:
        """Assemble model-ready observation dict."""
        scan = robot.simulate_lidar(env)
        return {
            "feature_map": env.to_feature_map(),
            "local_map": robot.get_local_obstacle_map(env),
            "lidar_ranges": scan.ranges,
            "lidar_angles": scan.angles,
            "position": robot.position,
            "goal": env.goal,
            "heading": robot.heading,
        }

    def decide_action(
        self,
        env: WarehouseEnvironment,
        robot: Robot,
        current_path: List[Position],
        new_obstacles: Optional[List[Position]] = None,
    ) -> MovementDecision:
        """
        Spatial reasoning + sensor fusion -> action label.

        Returns (action_name, optional_waypoint).
        """
        if robot.has_reached_goal(env.goal):
            return ("stop", None)

        if new_obstacles and self._path_blocked(current_path, new_obstacles):
            return ("replan", None)

        waypoint = robot.current_waypoint()
        if waypoint is None:
            return ("stop", None)

        return ("follow_path", waypoint)

    def plan_path(
        self,
        env: WarehouseEnvironment,
        start: Optional[Position] = None,
        goal: Optional[Position] = None,
    ) -> PlanResult:
        return self.planner.plan(env, start=start, goal=goal)

    def _path_blocked(
        self, path: List[Position], obstacles: List[Position]
    ) -> bool:
        if not path:
            return False
        obstacle_set = set(obstacles)
        return any(cell in obstacle_set for cell in path)

    def _nearby_obstacle(
        self, env: WarehouseEnvironment, position: Position
    ) -> bool:
        row, col = position
        r = self.obstacle_proximity_cells
        for dr in range(-r, r + 1):
            for dc in range(-r, r + 1):
                nr, nc = row + dr, col + dc
                if env.in_bounds(nr, nc) and env.get_cell(nr, nc) == 3:
                    return True
        return False

    def interpret_lidar_sectors(self, scan: LidarScan) -> dict:
        """Simple spatial reasoning: classify free space by direction sector."""
        n = scan.num_rays
        third = n // 3
        sectors = {
            "front": scan.ranges[:third],
            "left": scan.ranges[third : 2 * third],
            "right": scan.ranges[2 * third :],
        }
        return {k: float(np.mean(v)) for k, v in sectors.items()}
