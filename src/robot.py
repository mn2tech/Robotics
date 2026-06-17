"""
Simulated mobile robot with LiDAR-style range sensing.

Movement operates on discrete grid waypoints; sensor rays use continuous
geometry so the module can later swap to ROS laser_scan messages.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from src.environment import WarehouseEnvironment

Position = Tuple[int, int]


@dataclass
class LidarScan:
    """Simulated 2D LiDAR scan result."""

    angles: np.ndarray
    ranges: np.ndarray
    max_range: float

    @property
    def num_rays(self) -> int:
        return len(self.angles)


@dataclass
class Robot:
    """
    Grid-based robot with pose, path tracking, and sensor simulation.

    Position uses (row, col) grid coordinates; heading is radians for
    sensor ray casting (0 = east, pi/2 = south in grid space).
    """

    position: Position = (1, 1)
    heading: float = 0.0
    max_speed: float = 1.0
    lidar_max_range: float = 8.0
    lidar_num_rays: int = 36
    path: List[Position] = field(default_factory=list)
    path_index: int = 0
    collision_count: int = 0

    def set_position(self, row: int, col: int) -> None:
        self.position = (row, col)

    def set_path(self, path: List[Position]) -> None:
        """Assign a planned route without moving the robot from its current cell."""
        self.path = path
        self.path_index = 0
        while self.path_index < len(self.path) and self.path[self.path_index] == self.position:
            self.path_index += 1

    def current_waypoint(self) -> Optional[Position]:
        if self.path_index < len(self.path):
            return self.path[self.path_index]
        return None

    def has_reached_goal(self, goal: Position) -> bool:
        return self.position == goal

    def step_toward_waypoint(self, env: WarehouseEnvironment) -> bool:
        """
        Move one grid step toward the current waypoint.

        Returns True if the robot advanced along the path.
        """
        waypoint = self.current_waypoint()
        if waypoint is None:
            return False

        row, col = self.position
        wr, wc = waypoint

        if (row, col) == (wr, wc):
            self.path_index += 1
            return True

        dr = int(np.sign(wr - row))
        dc = int(np.sign(wc - col))

        next_row, next_col = row + dr, col + dc
        if env.is_walkable(next_row, next_col):
            self.position = (next_row, next_col)
            self.heading = float(np.arctan2(dr, dc))
            if self.position == waypoint:
                self.path_index += 1
            return True

        self.collision_count += 1
        return False

    def simulate_lidar(self, env: WarehouseEnvironment) -> LidarScan:
        """
        Cast range rays from the robot position; hits walls/shelves/obstacles.

        Uses Bresenham-style grid stepping along each ray direction.
        """
        row, col = self.position
        angles = np.linspace(0, 2 * np.pi, self.lidar_num_rays, endpoint=False)
        ranges = np.full(self.lidar_num_rays, self.lidar_max_range)

        for i, angle in enumerate(angles):
            dx = np.cos(angle)
            dy = np.sin(angle)
            for step in range(1, int(self.lidar_max_range * 2) + 1):
                t = step * 0.5
                nr = int(round(row + dy * t))
                nc = int(round(col + dx * t))
                if not env.in_bounds(nr, nc):
                    ranges[i] = t
                    break
                if not env.is_walkable(nr, nc):
                    ranges[i] = t
                    break

        return LidarScan(angles=angles, ranges=ranges, max_range=self.lidar_max_range)

    def get_local_obstacle_map(
        self, env: WarehouseEnvironment, radius: int = 5
    ) -> np.ndarray:
        """Local spatial map around the robot for inference pipelines."""
        row, col = self.position
        size = 2 * radius + 1
        local = np.zeros((size, size), dtype=np.int8)
        for r in range(size):
            for c in range(size):
                gr, gc = row + r - radius, col + c - radius
                if env.in_bounds(gr, gc):
                    local[r, c] = env.get_cell(gr, gc)
                else:
                    local[r, c] = 1  # treat out-of-bounds as wall
        return local

    def reset(self, start: Position) -> None:
        self.position = start
        self.path = []
        self.path_index = 0
        self.collision_count = 0
        self.heading = 0.0
