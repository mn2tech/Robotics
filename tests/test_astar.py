"""Tests for A* path planning."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.environment import Cell, WarehouseEnvironment
from src.planners.astar import AStarPlanner


@pytest.fixture
def open_env():
    return WarehouseEnvironment.from_preset("open", width=10, height=10)


@pytest.fixture
def blocked_env():
    env = WarehouseEnvironment.from_preset("open", width=10, height=10)
    for col in range(1, 9):
        env.set_cell(5, col, Cell.WALL)
    env.set_cell(5, 4, Cell.FREE)
    return env


def test_astar_finds_path_open_grid(open_env):
    planner = AStarPlanner()
    result = planner.plan(open_env)
    assert result.success
    assert result.path[0] == open_env.start
    assert result.path[-1] == open_env.goal
    assert len(result.path) >= 2


def test_astar_path_is_walkable(open_env):
    planner = AStarPlanner()
    result = planner.plan(open_env)
    for row, col in result.path:
        assert open_env.is_walkable(row, col)


def test_astar_no_path_when_blocked(blocked_env):
    blocked_env.set_start(1, 1)
    blocked_env.set_goal(8, 8)
    blocked_env.set_cell(5, 4, Cell.WALL)
    planner = AStarPlanner()
    result = planner.plan(blocked_env)
    assert not result.success
    assert result.path == []


def test_astar_runtime_recorded(open_env):
    planner = AStarPlanner()
    result = planner.plan(open_env)
    assert result.runtime_ms >= 0


def test_astar_path_through_gap(blocked_env):
    blocked_env.set_start(1, 1)
    blocked_env.set_goal(8, 8)
    planner = AStarPlanner()
    result = planner.plan(blocked_env)
    assert result.success
    assert (5, 4) in result.path
