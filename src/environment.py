"""
Warehouse grid environment for robot navigation simulation.

Designed as a 2D grid with cell types that can later extend to 3D voxels
or be bridged to ROS 2 / Gazebo / Isaac Sim occupancy maps.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Optional, Set, Tuple

import numpy as np

CellType = int
Position = Tuple[int, int]


class Cell(IntEnum):
    """Occupancy cell types for the warehouse grid."""

    FREE = 0
    WALL = 1
    SHELF = 2
    OBSTACLE = 3
    START = 4
    GOAL = 5


# Cells the robot cannot traverse
BLOCKED_CELLS: Set[CellType] = {Cell.WALL, Cell.SHELF, Cell.OBSTACLE}


@dataclass
class WarehouseEnvironment:
    """
    2D warehouse grid environment.

    Attributes:
        width: Grid width in cells.
        height: Grid height in cells.
        grid: 2D numpy array of Cell values.
        start: Robot start position (row, col).
        goal: Destination position (row, col).
    """

    width: int = 20
    height: int = 20
    grid: np.ndarray = field(init=False)
    start: Position = (1, 1)
    goal: Position = (18, 18)

    def __post_init__(self) -> None:
        self.grid = np.full((self.height, self.width), Cell.FREE, dtype=np.int8)
        self._build_default_warehouse()

    def _build_default_warehouse(self) -> None:
        """Create perimeter walls, shelf rows, and mark start/goal."""
        self.grid[0, :] = Cell.WALL
        self.grid[-1, :] = Cell.WALL
        self.grid[:, 0] = Cell.WALL
        self.grid[:, -1] = Cell.WALL

        # Aisle layout: shelf blocks with open corridors
        for row in range(3, self.height - 3, 4):
            for col in range(2, self.width - 2):
                if col % 5 != 0:
                    self.grid[row, col] = Cell.SHELF

        self.set_start(*self.start)
        self.set_goal(*self.goal)

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.height and 0 <= col < self.width

    def is_walkable(self, row: int, col: int) -> bool:
        if not self.in_bounds(row, col):
            return False
        return int(self.grid[row, col]) not in BLOCKED_CELLS

    def get_cell(self, row: int, col: int) -> CellType:
        return int(self.grid[row, col])

    def set_cell(self, row: int, col: int, cell_type: CellType) -> None:
        if self.in_bounds(row, col):
            self.grid[row, col] = cell_type

    def set_start(self, row: int, col: int) -> None:
        if self.in_bounds(row, col) and self.is_walkable(row, col):
            if self.in_bounds(*self.start):
                prev = self.grid[self.start]
                if prev in (Cell.START, Cell.GOAL):
                    self.grid[self.start] = Cell.FREE
            self.start = (row, col)
            self.grid[row, col] = Cell.START

    def set_goal(self, row: int, col: int) -> None:
        if self.in_bounds(row, col) and self.is_walkable(row, col):
            if self.in_bounds(*self.goal):
                prev = self.grid[self.goal]
                if prev in (Cell.START, Cell.GOAL):
                    self.grid[self.goal] = Cell.FREE
            self.goal = (row, col)
            self.grid[row, col] = Cell.GOAL

    def add_obstacle(self, row: int, col: int) -> bool:
        """Place a dynamic obstacle; returns False if cell is blocked."""
        if not self.in_bounds(row, col):
            return False
        if (row, col) in (self.start, self.goal):
            return False
        if not self.is_walkable(row, col) and self.get_cell(row, col) != Cell.OBSTACLE:
            return False
        self.grid[row, col] = Cell.OBSTACLE
        return True

    def remove_obstacle(self, row: int, col: int) -> bool:
        if not self.in_bounds(row, col):
            return False
        if self.get_cell(row, col) != Cell.OBSTACLE:
            return False
        self.grid[row, col] = Cell.FREE
        return True

    def get_obstacle_positions(self) -> List[Position]:
        rows, cols = np.where(self.grid == Cell.OBSTACLE)
        return list(zip(rows.tolist(), cols.tolist()))

    def get_walkable_mask(self) -> np.ndarray:
        """Boolean mask where True means the robot can move."""
        mask = np.ones((self.height, self.width), dtype=bool)
        for cell in BLOCKED_CELLS:
            mask &= self.grid != cell
        return mask

    def neighbors(self, row: int, col: int) -> List[Position]:
        """4-connected grid neighbors (extensible to 8-connected or 3D)."""
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        result: List[Position] = []
        for dr, dc in deltas:
            nr, nc = row + dr, col + dc
            if self.is_walkable(nr, nc):
                result.append((nr, nc))
        return result

    def to_feature_map(self) -> np.ndarray:
        """
        Spatial reasoning feature map: one-hot channels per cell type.
        Shape (height, width, num_cell_types) for ML inference pipelines.
        """
        num_types = len(Cell)
        features = np.zeros((self.height, self.width, num_types), dtype=np.float32)
        for i in range(num_types):
            features[:, :, i] = (self.grid == i).astype(np.float32)
        return features

    def copy(self) -> "WarehouseEnvironment":
        env = WarehouseEnvironment(width=self.width, height=self.height)
        env.grid = self.grid.copy()
        env.start = self.start
        env.goal = self.goal
        return env

    @classmethod
    def from_preset(cls, preset: str = "warehouse", width: int = 20, height: int = 20) -> "WarehouseEnvironment":
        env = cls(width=width, height=height)
        if preset == "open":
            env.grid.fill(Cell.FREE)
            env.grid[0, :] = Cell.WALL
            env.grid[-1, :] = Cell.WALL
            env.grid[:, 0] = Cell.WALL
            env.grid[:, -1] = Cell.WALL
            env.set_start(1, 1)
            env.set_goal(height - 2, width - 2)
        return env
