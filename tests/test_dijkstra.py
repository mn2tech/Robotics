"""Tests for Dijkstra path planning."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.environment import Cell, WarehouseEnvironment
from src.planners.dijkstra import DijkstraPlanner
from src.planners.astar import AStarPlanner


@pytest.fixture
def open_env():
    return WarehouseEnvironment.from_preset("open", width=10, height=10)


def test_dijkstra_finds_path(open_env):
    planner = DijkstraPlanner()
    result = planner.plan(open_env)
    assert result.success
    assert result.path[0] == open_env.start
    assert result.path[-1] == open_env.goal


def test_dijkstra_optimal_cost_open_grid(open_env):
    planner = DijkstraPlanner()
    result = planner.plan(open_env)
    # Manhattan shortest path on open 10x10 grid from (1,1) to (8,8) = 14
    assert result.cost == 14.0


def test_dijkstra_matches_astar_cost(open_env):
    d_result = DijkstraPlanner().plan(open_env)
    a_result = AStarPlanner().plan(open_env)
    assert d_result.success and a_result.success
    assert d_result.cost == a_result.cost


def test_dijkstra_no_path(open_env):
    open_env.set_start(1, 1)
    open_env.set_goal(8, 8)
    for col in range(1, 9):
        open_env.set_cell(5, col, Cell.WALL)
    result = DijkstraPlanner().plan(open_env)
    assert not result.success


def test_dijkstra_runtime_recorded(open_env):
    result = DijkstraPlanner().plan(open_env)
    assert result.runtime_ms >= 0
