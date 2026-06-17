"""Tests for environment visualization."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.environment import Cell, WarehouseEnvironment
from src.planners.astar import AStarPlanner
from src.visualization.plotter import (
    GRID_CELL_COLORS,
    build_grid_shapes,
    build_robot_figure_shapes,
    grid_to_rgb,
    plot_environment_plotly,
)


@pytest.fixture
def open_env():
    return WarehouseEnvironment.from_preset("open", width=20, height=20)


def test_grid_to_rgb_shape(open_env):
    rgb = grid_to_rgb(open_env.grid)
    assert rgb.shape == (open_env.height, open_env.width, 3)


def test_build_grid_shapes_count(open_env):
    shapes = build_grid_shapes(open_env)
    assert len(shapes) == open_env.height * open_env.width


def test_build_grid_shapes_have_cell_boundaries(open_env):
    shapes = build_grid_shapes(open_env)
    sample = shapes[0]
    assert sample["type"] == "rect"
    assert sample["line"]["width"] == 1
    assert sample["x1"] - sample["x0"] == 1.0
    assert sample["y1"] - sample["y0"] == 1.0


def test_build_grid_shapes_obstacle_is_black():
    env = WarehouseEnvironment.from_preset("open", width=5, height=5)
    env.add_obstacle(2, 2)
    shapes = build_grid_shapes(env)
    obstacle_shape = shapes[2 * env.width + 2]
    assert obstacle_shape["fillcolor"] == GRID_CELL_COLORS[Cell.OBSTACLE]


def test_plot_environment_plotly_includes_start_goal_and_path(open_env):
    planner = AStarPlanner()
    plan = planner.plan(open_env)
    fig = plot_environment_plotly(
        open_env,
        path=plan.path,
        robot_pos=open_env.start,
    )
    trace_names = {trace.name for trace in fig.data}
    assert "Start" in trace_names
    assert "Goal" in trace_names
    assert "Path" in trace_names
    assert "Robot" in trace_names


def test_plot_environment_plotly_renders_full_grid(open_env):
    fig = plot_environment_plotly(open_env)
    assert len(fig.layout.shapes) == open_env.height * open_env.width


def test_plot_environment_plotly_y_axis_reversed(open_env):
    fig = plot_environment_plotly(open_env)
    assert fig.layout.yaxis.autorange == "reversed"


def test_plot_environment_plotly_equal_axis_scaling(open_env):
    fig = plot_environment_plotly(open_env)
    assert fig.layout.xaxis.scaleanchor == "y"
    assert fig.layout.xaxis.scaleratio == 1


def test_plot_environment_plotly_axis_labels(open_env):
    fig = plot_environment_plotly(open_env)
    assert fig.layout.xaxis.title.text == "Column"
    assert fig.layout.yaxis.title.text == "Row"


def test_plot_environment_plotly_path_color(open_env):
    planner = AStarPlanner()
    plan = planner.plan(open_env)
    fig = plot_environment_plotly(open_env, path=plan.path)
    path_trace = next(trace for trace in fig.data if trace.name == "Path")
    assert path_trace.line.color == "#007bff"


def test_build_robot_figure_shapes_has_limbs():
    shapes = build_robot_figure_shapes(5, 5)
    assert len(shapes) == 6
    shape_types = {shape["type"] for shape in shapes}
    assert "circle" in shape_types
    assert "rect" in shape_types


def test_plot_environment_plotly_draws_robot_figure(open_env):
    fig = plot_environment_plotly(open_env, robot_pos=(3, 3), goal_reached=False)
    grid_count = open_env.height * open_env.width
    assert len(fig.layout.shapes) == grid_count + 6


def test_plot_environment_plotly_combined_marker_when_goal_reached(open_env):
    fig = plot_environment_plotly(
        open_env,
        robot_pos=open_env.goal,
        goal_reached=True,
    )
    trace_names = {trace.name for trace in fig.data}
    assert "Goal Reached" in trace_names
    assert "Goal" not in trace_names
    grid_count = open_env.height * open_env.width
    assert len(fig.layout.shapes) == grid_count + 6


def test_plot_environment_plotly_separate_robot_and_goal_markers(open_env):
    robot_pos = (open_env.start[0] + 1, open_env.start[1] + 1)
    fig = plot_environment_plotly(open_env, robot_pos=robot_pos, goal_reached=False)
    trace_names = {trace.name for trace in fig.data}
    assert "Robot" in trace_names
    assert "Goal" in trace_names
    assert "Goal Reached" not in trace_names


def test_plot_environment_plotly_hover_tooltips_include_coordinates(open_env):
    planner = AStarPlanner()
    plan = planner.plan(open_env)
    fig = plot_environment_plotly(open_env, path=plan.path, robot_pos=open_env.start)
    start_trace = next(trace for trace in fig.data if trace.name == "Start")
    path_trace = next(trace for trace in fig.data if trace.name == "Path")
    assert f"Row {open_env.start[0]}, Col {open_env.start[1]}" in start_trace.text
    assert path_trace.hovertemplate is not None
    assert "Path" in path_trace.hovertemplate


def test_plot_environment_plotly_shows_trajectory_trail(open_env):
    trajectory = [open_env.start, (2, 1), (3, 1), (4, 1)]
    fig = plot_environment_plotly(
        open_env,
        robot_pos=trajectory[-1],
        trajectory=trajectory,
    )
    trace_names = {trace.name for trace in fig.data}
    assert "Traveled" in trace_names


def test_walkthrough_plotly_has_animation_frames(open_env):
    from src.visualization.plotter import plot_environment_walkthrough_plotly

    trajectory = [open_env.start, (2, 1), (3, 1), (4, 1), open_env.goal]
    fig = plot_environment_walkthrough_plotly(
        open_env,
        path=None,
        trajectory=trajectory,
        success=True,
    )
    assert len(fig.frames) == len(trajectory)
    assert list(fig.frames[0].traces) == [0, len(fig.data) - 2]
    assert fig.layout.sliders
    assert fig.layout.updatemenus
    play_button = fig.layout.updatemenus[0].buttons[0]
    assert play_button.args[1]["frame"]["redraw"] is False
