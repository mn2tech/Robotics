"""Tests for navigation simulation, metrics, and replanning."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.environment import Cell, WarehouseEnvironment
from src.planners.astar import AStarPlanner
from src.planners.base import BasePlanner
from src.robot import Robot
from src.simulation import NavigationSimulator


@pytest.fixture
def warehouse_env():
    env = WarehouseEnvironment(width=20, height=20)
    env.set_start(1, 1)
    env.set_goal(18, 18)
    return env


def test_successful_navigation_start_1_1_goal_18_18(warehouse_env):
    sim = NavigationSimulator(
        env=warehouse_env.copy(),
        robot=Robot(position=warehouse_env.start),
        planner=AStarPlanner(),
        dynamic_obstacles=[],
        obstacle_trigger_step=None,
    )
    result = sim.run()

    assert result.metrics.success is True
    assert result.metrics.path_length > 0
    assert result.metrics.replan_count == 0
    assert result.metrics.total_planning_latency_ms < 100
    assert result.trajectory[-1] == warehouse_env.goal


def test_no_path_scenario_reports_failure():
    env = WarehouseEnvironment.from_preset("open", width=10, height=10)
    for col in range(1, 9):
        env.set_cell(5, col, Cell.WALL)
    env.set_start(1, 1)
    env.set_goal(8, 8)
    env.set_cell(5, 4, Cell.WALL)

    sim = NavigationSimulator(
        env=env.copy(),
        robot=Robot(position=env.start),
        planner=AStarPlanner(),
    )
    result = sim.run()

    assert result.metrics.success is False
    assert result.metrics.path_length == 0
    assert result.metrics.replan_count == 0
    assert result.final_path == []


def test_path_length_matches_planned_route_steps(warehouse_env):
    planner = AStarPlanner()
    plan = planner.plan(warehouse_env)

    sim = NavigationSimulator(
        env=warehouse_env.copy(),
        robot=Robot(position=warehouse_env.start),
        planner=planner,
        dynamic_obstacles=[],
        obstacle_trigger_step=None,
    )
    result = sim.run()

    assert result.metrics.path_length == BasePlanner.path_step_count(plan.path)
    assert result.metrics.path_length == len(plan.path) - 1


def test_replan_when_dynamic_obstacle_blocks_path():
    env = WarehouseEnvironment.from_preset("open", width=10, height=10)
    env.set_start(1, 1)
    env.set_goal(8, 8)
    initial_plan = AStarPlanner().plan(env)
    block_cell = initial_plan.path[len(initial_plan.path) // 2]

    sim = NavigationSimulator(
        env=env.copy(),
        robot=Robot(position=env.start),
        planner=AStarPlanner(),
        dynamic_obstacles=[block_cell],
        obstacle_trigger_step=2,
    )
    result = sim.run()

    assert result.metrics.replan_count == 1
    assert block_cell not in result.final_path


def test_no_replan_when_dynamic_obstacle_does_not_block_path(warehouse_env):
    sim = NavigationSimulator(
        env=warehouse_env.copy(),
        robot=Robot(position=warehouse_env.start),
        planner=AStarPlanner(),
        dynamic_obstacles=[(10, 10)],
        obstacle_trigger_step=5,
    )
    result = sim.run()

    assert result.metrics.success is True
    assert result.metrics.replan_count == 0


def test_replan_counter_not_incremented_per_step(warehouse_env):
    sim = NavigationSimulator(
        env=warehouse_env.copy(),
        robot=Robot(position=warehouse_env.start),
        planner=AStarPlanner(),
        dynamic_obstacles=[],
        obstacle_trigger_step=None,
    )
    result = sim.run()

    assert result.metrics.replan_count == 0
    assert result.metrics.steps_taken == result.metrics.path_length


def test_simulation_stops_at_max_steps():
    env = WarehouseEnvironment.from_preset("open", width=10, height=10)
    env.set_start(1, 1)
    env.set_goal(8, 8)

    sim = NavigationSimulator(
        env=env.copy(),
        robot=Robot(position=env.start),
        planner=AStarPlanner(),
        max_steps=3,
        dynamic_obstacles=[],
        obstacle_trigger_step=None,
    )
    result = sim.run()

    assert result.metrics.success is False
    assert result.metrics.steps_taken == 3


def test_planning_latency_excludes_simulation_loop_time(warehouse_env):
    sim = NavigationSimulator(
        env=warehouse_env.copy(),
        robot=Robot(position=warehouse_env.start),
        planner=AStarPlanner(),
    )
    result = sim.run()

    assert result.metrics.total_planning_latency_ms < result.metrics.simulation_time_s * 1000
