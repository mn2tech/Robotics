"""Matplotlib and Plotly visualization for the navigation simulator."""

from __future__ import annotations

from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

from src.environment import Cell, WarehouseEnvironment

Position = Tuple[int, int]

# Color map for grid cells (matplotlib / grid_to_rgb)
CELL_COLORS = {
    Cell.FREE: "#f8f9fa",
    Cell.WALL: "#343a40",
    Cell.SHELF: "#6c757d",
    Cell.OBSTACLE: "#dc3545",
    Cell.START: "#28a745",
    Cell.GOAL: "#007bff",
}

# Plotly-specific display colors
GRID_CELL_COLORS = {
    Cell.FREE: "#f8f9fa",
    Cell.WALL: "#343a40",
    Cell.SHELF: "#6c757d",
    Cell.OBSTACLE: "#000000",
    Cell.START: "#f8f9fa",
    Cell.GOAL: "#f8f9fa",
}
GRID_LINE_COLOR = "#adb5bd"
PLOT_START_COLOR = "#28a745"
PLOT_GOAL_COLOR = "#dc3545"
PLOT_PATH_COLOR = "#007bff"
PLOT_ROBOT_COLOR = "#ff00ff"
PLOT_TRAJECTORY_COLOR = "rgba(255, 0, 255, 0.45)"


def grid_to_rgb(grid: np.ndarray) -> np.ndarray:
    """Convert cell-type grid to RGB image for plotting."""
    h, w = grid.shape
    rgb = np.zeros((h, w, 3), dtype=np.float32)
    for cell_type, hex_color in CELL_COLORS.items():
        mask = grid == cell_type
        color = tuple(int(hex_color[i : i + 2], 16) / 255 for i in (1, 3, 5))
        rgb[mask] = color
    return rgb


def build_grid_shapes(env: WarehouseEnvironment) -> list:
    """Build Plotly rectangle shapes for each grid cell with visible boundaries."""
    shapes = []
    for row in range(env.height):
        for col in range(env.width):
            cell = Cell(int(env.grid[row, col]))
            shapes.append(
                {
                    "type": "rect",
                    "xref": "x",
                    "yref": "y",
                    "x0": col - 0.5,
                    "x1": col + 0.5,
                    "y0": row - 0.5,
                    "y1": row + 0.5,
                    "fillcolor": GRID_CELL_COLORS[cell],
                    "line": {"color": GRID_LINE_COLOR, "width": 1},
                    "layer": "below",
                }
            )
    return shapes


def _legend_marker(name: str, color: str, symbol: str = "square", size: int = 12) -> go.Scatter:
    return go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(size=size, color=color, symbol=symbol),
        name=name,
        showlegend=True,
    )


def _position_label(row: int, col: int) -> str:
    return f"Row {row}, Col {col}"


def _marker_trace(
    name: str,
    row: int,
    col: int,
    *,
    color: str,
    symbol: str = "circle",
    size: int = 14,
    line_color: str = "white",
    line_width: int = 2,
) -> go.Scatter:
    return go.Scatter(
        x=[col],
        y=[row],
        mode="markers",
        marker=dict(
            size=size,
            color=color,
            symbol=symbol,
            line=dict(width=line_width, color=line_color),
        ),
        name=name,
        text=[_position_label(row, col)],
        hovertemplate=f"{name}<br>%{{text}}<extra></extra>",
    )


def _robot_part(
    x0: float,
    x1: float,
    y0: float,
    y1: float,
    *,
    fill_color: str,
    line_color: str,
    line_width: int = 1,
    shape_type: str = "rect",
) -> dict:
    return {
        "type": shape_type,
        "xref": "x",
        "yref": "y",
        "x0": x0,
        "x1": x1,
        "y0": y0,
        "y1": y1,
        "fillcolor": fill_color,
        "line": {"color": line_color, "width": line_width},
        "layer": "above",
    }


def build_robot_figure_shapes(
    row: int,
    col: int,
    *,
    body_color: str = PLOT_ROBOT_COLOR,
    outline_color: str = "white",
    outline_width: int = 1,
) -> list:
    """Draw a simple stick-figure robot with head, body, arms, and legs."""
    return [
        _robot_part(
            col - 0.1,
            col + 0.1,
            row - 0.45,
            row - 0.25,
            fill_color=body_color,
            line_color=outline_color,
            line_width=outline_width,
            shape_type="circle",
        ),
        _robot_part(
            col - 0.1,
            col + 0.1,
            row - 0.22,
            row + 0.08,
            fill_color=body_color,
            line_color=outline_color,
            line_width=outline_width,
        ),
        _robot_part(
            col - 0.38,
            col - 0.12,
            row - 0.16,
            row - 0.04,
            fill_color=body_color,
            line_color=outline_color,
            line_width=outline_width,
        ),
        _robot_part(
            col + 0.12,
            col + 0.38,
            row - 0.16,
            row - 0.04,
            fill_color=body_color,
            line_color=outline_color,
            line_width=outline_width,
        ),
        _robot_part(
            col - 0.12,
            col - 0.02,
            row + 0.08,
            row + 0.42,
            fill_color=body_color,
            line_color=outline_color,
            line_width=outline_width,
        ),
        _robot_part(
            col + 0.02,
            col + 0.12,
            row + 0.08,
            row + 0.42,
            fill_color=body_color,
            line_color=outline_color,
            line_width=outline_width,
        ),
    ]


def _hover_trace(name: str, row: int, col: int) -> go.Scatter:
    """Invisible marker used for hover tooltips without obscuring drawn shapes."""
    return go.Scatter(
        x=[col],
        y=[row],
        mode="markers",
        marker=dict(size=20, color="rgba(0,0,0,0)"),
        name=name,
        text=[_position_label(row, col)],
        hovertemplate=f"{name}<br>%{{text}}<extra></extra>",
        showlegend=False,
    )


def _robot_legend_marker(name: str = "Robot", color: str = PLOT_ROBOT_COLOR) -> go.Scatter:
    return go.Scatter(
        x=[None],
        y=[None],
        mode="markers",
        marker=dict(size=12, color=color, symbol="circle", line=dict(color="white", width=1)),
        name=name,
        showlegend=True,
    )


def _robot_at_goal(robot_pos: Optional[Position], goal: Position) -> bool:
    return robot_pos is not None and robot_pos == goal


def _add_position_markers(
    fig: go.Figure,
    env: WarehouseEnvironment,
    robot_pos: Optional[Position],
    goal_reached: bool,
) -> list:
    overlay_shapes: list = []

    fig.add_trace(
        _marker_trace(
            "Start",
            env.start[0],
            env.start[1],
            color=PLOT_START_COLOR,
            symbol="circle",
            size=14,
        )
    )

    at_goal = goal_reached or _robot_at_goal(robot_pos, env.goal)
    if at_goal:
        overlay_shapes.extend(
            build_robot_figure_shapes(
                env.goal[0],
                env.goal[1],
                body_color=PLOT_GOAL_COLOR,
                outline_color=PLOT_ROBOT_COLOR,
                outline_width=2,
            )
        )
        fig.add_trace(_hover_trace("Goal Reached", env.goal[0], env.goal[1]))
        fig.add_trace(_robot_legend_marker("Goal Reached", PLOT_GOAL_COLOR))
        return overlay_shapes

    fig.add_trace(
        _marker_trace(
            "Goal",
            env.goal[0],
            env.goal[1],
            color=PLOT_GOAL_COLOR,
            symbol="circle",
            size=16,
            line_width=2,
        )
    )

    if robot_pos:
        overlay_shapes.extend(
            build_robot_figure_shapes(
                robot_pos[0],
                robot_pos[1],
                body_color=PLOT_ROBOT_COLOR,
                outline_color="white",
                outline_width=1,
            )
        )
        fig.add_trace(_hover_trace("Robot", robot_pos[0], robot_pos[1]))
        fig.add_trace(_robot_legend_marker())

    return overlay_shapes


def plot_environment_matplotlib(
    env: WarehouseEnvironment,
    path: Optional[List[Position]] = None,
    robot_pos: Optional[Position] = None,
    title: str = "Warehouse Navigation",
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 8))
    rgb = grid_to_rgb(env.grid)
    ax.imshow(rgb, origin="upper")
    ax.set_title(title)
    ax.set_xlabel("Column")
    ax.set_ylabel("Row")
    ax.grid(True, alpha=0.2)

    if path and len(path) > 1:
        rows, cols = zip(*path)
        ax.plot(cols, rows, color="#ffc107", linewidth=2, marker="o", markersize=3, label="Path")

    if robot_pos:
        ax.scatter([robot_pos[1]], [robot_pos[0]], c="#ff00ff", s=120, marker="s", label="Robot", zorder=5)

    if path or robot_pos:
        ax.legend(loc="upper right")

    plt.tight_layout()
    return fig


def plot_environment_plotly(
    env: WarehouseEnvironment,
    path: Optional[List[Position]] = None,
    robot_pos: Optional[Position] = None,
    title: str = "Warehouse Navigation",
    goal_reached: bool = False,
    trajectory: Optional[List[Position]] = None,
) -> go.Figure:
    fig = go.Figure()

    shapes = build_grid_shapes(env)

    if trajectory and len(trajectory) > 1:
        rows, cols = zip(*trajectory)
        fig.add_trace(
            go.Scatter(
                x=list(cols),
                y=list(rows),
                mode="lines+markers",
                line=dict(color=PLOT_TRAJECTORY_COLOR, width=2, dash="dot"),
                marker=dict(size=4, color=PLOT_ROBOT_COLOR),
                name="Traveled",
                text=[_position_label(r, c) for r, c in trajectory],
                hovertemplate="Traveled<br>%{text}<extra></extra>",
            )
        )

    if path and len(path) > 1:
        rows, cols = zip(*path)
        fig.add_trace(
            go.Scatter(
                x=list(cols),
                y=list(rows),
                mode="lines+markers",
                line=dict(color=PLOT_PATH_COLOR, width=3),
                marker=dict(size=5, color=PLOT_PATH_COLOR),
                name="Path",
                text=[_position_label(r, c) for r, c in path],
                hovertemplate="Path<br>%{text}<extra></extra>",
            )
        )

    if env.get_obstacle_positions():
        fig.add_trace(_legend_marker("Obstacle", "#000000", symbol="square", size=12))

    shapes.extend(_add_position_markers(fig, env, robot_pos, goal_reached))

    fig.update_layout(
        shapes=shapes,
        title=title,
        xaxis=dict(
            title="Column",
            range=[-0.5, env.width - 0.5],
            scaleanchor="y",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Row",
            range=[-0.5, env.height - 0.5],
            autorange="reversed",
            showgrid=False,
            zeroline=False,
        ),
        autosize=True,
        showlegend=True,
        margin=dict(l=50, r=20, t=50, b=50),
    )
    return fig


def _animated_robot_trace(
    row: int,
    col: int,
    *,
    at_goal: bool = False,
) -> go.Scatter:
    """Visible robot marker optimized for smooth frame animation."""
    name = "Goal Reached" if at_goal else "Robot"
    color = PLOT_GOAL_COLOR if at_goal else PLOT_ROBOT_COLOR
    line_color = PLOT_ROBOT_COLOR if at_goal else "white"
    return go.Scatter(
        x=[col],
        y=[row],
        mode="markers",
        marker=dict(
            size=20 if at_goal else 16,
            color=color,
            symbol="square",
            line=dict(width=3 if at_goal else 2, color=line_color),
        ),
        name=name,
        text=[_position_label(row, col)],
        hovertemplate=f"{name}<br>%{{text}}<extra></extra>",
        showlegend=False,
    )


def plot_environment_walkthrough_plotly(
    env: WarehouseEnvironment,
    path: Optional[List[Position]],
    trajectory: List[Position],
    title: str = "Warehouse Navigation",
    success: bool = False,
    frame_duration_ms: int = 80,
) -> go.Figure:
    """Build a Plotly figure with lightweight frames for smooth robot playback."""
    grid_shapes = build_grid_shapes(env)
    max_step = len(trajectory) - 1
    transition_ms = max(0, min(frame_duration_ms - 10, 60))

    def traveled_trace(points: List[Position]) -> go.Scatter:
        if not points:
            return go.Scatter(
                x=[],
                y=[],
                mode="lines",
                line=dict(color=PLOT_TRAJECTORY_COLOR, width=3),
                name="Traveled",
                hoverinfo="skip",
            )
        rows, cols = zip(*points)
        return go.Scatter(
            x=list(cols),
            y=list(rows),
            mode="lines",
            line=dict(color=PLOT_TRAJECTORY_COLOR, width=3),
            name="Traveled",
            text=[_position_label(r, c) for r, c in points],
            hovertemplate="Traveled<br>%{text}<extra></extra>",
        )

    traces: list = [traveled_trace([trajectory[0]])]
    traveled_idx = 0

    if path and len(path) > 1:
        rows, cols = zip(*path)
        traces.append(
            go.Scatter(
                x=list(cols),
                y=list(rows),
                mode="lines",
                line=dict(color=PLOT_PATH_COLOR, width=3),
                name="Path",
                text=[_position_label(r, c) for r, c in path],
                hovertemplate="Path<br>%{text}<extra></extra>",
            )
        )

    if env.get_obstacle_positions():
        traces.append(_legend_marker("Obstacle", "#000000", symbol="square", size=12))

    traces.append(
        _marker_trace(
            "Start",
            env.start[0],
            env.start[1],
            color=PLOT_START_COLOR,
            symbol="circle",
            size=14,
        )
    )
    traces.append(
        _marker_trace(
            "Goal",
            env.goal[0],
            env.goal[1],
            color=PLOT_GOAL_COLOR,
            symbol="circle",
            size=16,
            line_width=2,
        )
    )

    start_row, start_col = trajectory[0]
    robot_idx = len(traces)
    traces.append(_animated_robot_trace(start_row, start_col))
    traces.append(_robot_legend_marker())

    fig = go.Figure(data=traces)
    fig.update_layout(
        shapes=grid_shapes,
        title=title,
        xaxis=dict(
            title="Column",
            range=[-0.5, env.width - 0.5],
            scaleanchor="y",
            scaleratio=1,
            showgrid=False,
            zeroline=False,
        ),
        yaxis=dict(
            title="Row",
            range=[-0.5, env.height - 0.5],
            autorange="reversed",
            showgrid=False,
            zeroline=False,
        ),
        autosize=True,
        showlegend=True,
        margin=dict(l=50, r=20, t=70, b=90),
        uirevision="navigation-chart",
        hovermode="closest",
    )

    animate_args = {
        "frame": {"duration": frame_duration_ms, "redraw": False},
        "transition": {"duration": transition_ms, "easing": "linear"},
        "fromcurrent": True,
    }
    slider_args = {
        "frame": {"duration": 0, "redraw": False},
        "mode": "immediate",
        "transition": {"duration": 0},
    }

    frames = []
    slider_steps = []
    for step in range(len(trajectory)):
        row, col = trajectory[step]
        at_goal = success and step == max_step
        frames.append(
            go.Frame(
                name=str(step),
                traces=[traveled_idx, robot_idx],
                data=[
                    traveled_trace(trajectory[: step + 1]),
                    _animated_robot_trace(row, col, at_goal=at_goal),
                ],
            )
        )
        slider_steps.append(
            {
                "label": str(step),
                "method": "animate",
                "args": [[str(step)], slider_args],
            }
        )

    fig.frames = frames
    fig.update_layout(
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0.05,
                "y": 1.12,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "Play",
                        "method": "animate",
                        "args": [None, animate_args],
                    },
                    {
                        "label": "Pause",
                        "method": "animate",
                        "args": [
                            [None],
                            {
                                "frame": {"duration": 0, "redraw": False},
                                "mode": "immediate",
                                "transition": {"duration": 0},
                            },
                        ],
                    },
                ],
            }
        ],
        sliders=[
            {
                "active": max_step,
                "x": 0.05,
                "len": 0.9,
                "xanchor": "left",
                "y": -0.02,
                "yanchor": "top",
                "pad": {"b": 10, "t": 40},
                "currentvalue": {"prefix": "Step: "},
                "steps": slider_steps,
            }
        ],
    )
    return fig


def plot_lidar_polar(scan_angles: np.ndarray, scan_ranges: np.ndarray) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatterpolar(
            r=scan_ranges,
            theta=np.degrees(scan_angles),
            mode="lines",
            fill="toself",
            name="LiDAR",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True)),
        title="Simulated LiDAR Scan",
        height=400,
    )
    return fig


def plot_metrics_comparison(metrics_rows: list) -> go.Figure:
    if not metrics_rows:
        return go.Figure()

    algorithms = [r["algorithm"] for r in metrics_rows]
    fig = go.Figure(
        data=[
            go.Bar(name="Path Length", x=algorithms, y=[r["path_length"] for r in metrics_rows]),
            go.Bar(name="Time (s)", x=algorithms, y=[r["time_to_destination_s"] for r in metrics_rows]),
            go.Bar(name="Latency (ms)", x=algorithms, y=[r["planning_latency_ms"] for r in metrics_rows]),
        ]
    )
    fig.update_layout(barmode="group", title="Algorithm Comparison", height=450)
    return fig
