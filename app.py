"""
Streamlit dashboard for the AI Navigation Simulator.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path when launched via streamlit
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.environment import Cell, WarehouseEnvironment
from src.evaluation.metrics import compare_metrics
from src.planners.astar import AStarPlanner
from src.planners.dijkstra import DijkstraPlanner
from src.planners.rrt import RRTPlanner
from src.robot import Robot
from src.simulation import NavigationSimulator
from src.ui.overview import render_under_the_hood
from src.visualization.plotter import (
    plot_environment_plotly,
    plot_environment_walkthrough_plotly,
    plot_lidar_polar,
    plot_metrics_comparison,
)

st.set_page_config(
    page_title="AI Navigation Simulator",
    page_icon="🤖",
    layout="wide",
)

PLANNERS = {
    "A*": AStarPlanner,
    "Dijkstra": DijkstraPlanner,
    "RRT": RRTPlanner,
}


def init_session_state() -> None:
    defaults = {
        "env": WarehouseEnvironment(width=20, height=20),
        "last_result": None,
        "comparison_results": [],
        "obstacle_mode": False,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def sidebar_controls() -> dict:
    st.sidebar.header("Simulation Controls")

    grid_size = st.sidebar.slider("Grid size", 10, 30, 20)
    preset = st.sidebar.selectbox("Environment preset", ["warehouse", "open"])
    algorithm = st.sidebar.selectbox("Algorithm", list(PLANNERS.keys()))

    if st.sidebar.button("Reset Environment"):
        st.session_state.env = WarehouseEnvironment.from_preset(
            preset=preset, width=grid_size, height=grid_size
        )
        st.session_state.last_result = None

    st.sidebar.subheader("Start & Goal")
    env = st.session_state.env
    start_row = st.sidebar.number_input("Start row", 0, env.height - 1, env.start[0])
    start_col = st.sidebar.number_input("Start col", 0, env.width - 1, env.start[1])
    goal_row = st.sidebar.number_input("Goal row", 0, env.height - 1, env.goal[0])
    goal_col = st.sidebar.number_input("Goal col", 0, env.width - 1, env.goal[1])

    if st.sidebar.button("Apply Start/Goal"):
        env.set_start(int(start_row), int(start_col))
        env.set_goal(int(goal_row), int(goal_col))

    st.sidebar.subheader("Obstacles")
    obstacle_mode = st.sidebar.checkbox("Click to add obstacles", value=False)
    st.session_state.obstacle_mode = obstacle_mode

    if st.sidebar.button("Clear Dynamic Obstacles"):
        for r, c in env.get_obstacle_positions():
            env.remove_obstacle(r, c)

    st.sidebar.subheader("Dynamic Obstacle (Replan Test)")
    dyn_row = st.sidebar.number_input("Dyn. obstacle row", 0, env.height - 1, env.height // 2)
    dyn_col = st.sidebar.number_input("Dyn. obstacle col", 0, env.width - 1, env.width // 2)
    trigger_step = st.sidebar.slider("Appear at step", 0, 50, 10)

    st.sidebar.subheader("Playback")
    animation_ms = st.sidebar.slider("Frame delay (ms)", 30, 300, 60)

    run_sim = st.sidebar.button("Run Simulation", type="primary")
    compare_all = st.sidebar.button("Compare All Algorithms")

    return {
        "algorithm": algorithm,
        "run_sim": run_sim,
        "compare_all": compare_all,
        "dyn_obstacle": (int(dyn_row), int(dyn_col)),
        "trigger_step": trigger_step,
        "animation_ms": animation_ms,
    }


def handle_grid_click(env: WarehouseEnvironment) -> None:
    if not st.session_state.obstacle_mode:
        return
    # Streamlit plotly click events via session state workaround:
    # users can toggle obstacle cells via number inputs in expander
    pass


def run_single_simulation(algorithm: str, dyn_obstacle: tuple, trigger_step: int):
    env = st.session_state.env.copy()
    planner = PLANNERS[algorithm]()
    robot = Robot(position=env.start)
    simulator = NavigationSimulator(
        env=env,
        robot=robot,
        planner=planner,
        dynamic_obstacles=[dyn_obstacle],
        obstacle_trigger_step=trigger_step,
    )
    return simulator.run()


def main() -> None:
    init_session_state()
    controls = sidebar_controls()
    env = st.session_state.env

    st.title("AI Navigation Simulator")
    st.caption("3D-ready path planning, sensor simulation, and evaluation for warehouse robotics")

    simulator_tab, review_tab = st.tabs(["Simulator", "How it works"])

    with review_tab:
        render_under_the_hood()

    with simulator_tab:
        render_simulator(controls, env)


def render_simulator(controls: dict, env: WarehouseEnvironment) -> None:
    col1, col2 = st.columns([2, 1])

    with col2:
        st.subheader("Obstacle Editor")
        obs_row = st.number_input("Obstacle row", 0, env.height - 1, 5, key="obs_r")
        obs_col = st.number_input("Obstacle col", 0, env.width - 1, 5, key="obs_c")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("Add Obstacle"):
                env.add_obstacle(int(obs_row), int(obs_col))
        with c2:
            if st.button("Remove Obstacle"):
                env.remove_obstacle(int(obs_row), int(obs_col))

    if controls["run_sim"]:
        with st.spinner(f"Running {controls['algorithm']}..."):
            st.session_state.last_result = run_single_simulation(
                controls["algorithm"],
                controls["dyn_obstacle"],
                controls["trigger_step"],
            )

    if controls["compare_all"]:
        with st.spinner("Comparing all algorithms..."):
            results = []
            for name in PLANNERS:
                results.append(
                    run_single_simulation(
                        name,
                        controls["dyn_obstacle"],
                        controls["trigger_step"],
                    )
                )
            st.session_state.comparison_results = results

    result = st.session_state.last_result
    path = result.final_path if result else None
    trajectory = result.trajectory if result else None
    metrics = result.metrics.to_dict() if result else None

    with col1:
        if metrics and metrics["success"]:
            st.success("Goal Reached")

        if metrics:
            summary = st.columns(5)
            summary[0].metric("Algorithm", controls["algorithm"])
            summary[1].metric("Success", "Yes" if metrics["success"] else "No")
            summary[2].metric("Path Length", f"{metrics['path_length']:.0f}")
            summary[3].metric("Plan Latency (ms)", f"{metrics['planning_latency_ms']:.1f}")
            summary[4].metric("Replans", metrics["replan_count"])

        if result and trajectory and len(trajectory) > 1:
            st.caption("Press **Play** on the chart or drag the step slider to watch the robot walk.")
            fig = plot_environment_walkthrough_plotly(
                env,
                path=path,
                trajectory=trajectory,
                title=f"Environment — {controls['algorithm']}",
                success=bool(metrics and metrics["success"]),
                frame_duration_ms=controls["animation_ms"],
            )
        else:
            robot_pos = trajectory[-1] if trajectory else env.start
            fig = plot_environment_plotly(
                env,
                path=path,
                robot_pos=robot_pos,
                title=f"Environment — {controls['algorithm']}",
                goal_reached=bool(metrics and metrics["success"]),
                trajectory=trajectory,
            )

        st.plotly_chart(
            fig,
            use_container_width=True,
            config={"displayModeBar": True, "scrollZoom": False},
        )

        if result and (not path or len(path) < 2):
            st.warning(
                "No valid path found. The planner could not route from start to goal "
                "with the current obstacles. Try moving start/goal or clearing obstacles."
            )

    if result:
        st.subheader("Run Metrics")
        m = result.metrics.to_dict()
        metric_cols = st.columns(4)
        metric_cols[0].metric("Success", "Yes" if m["success"] else "No")
        metric_cols[1].metric("Path Length", f"{m['path_length']:.1f}")
        metric_cols[2].metric("Time (s)", f"{m['time_to_destination_s']:.2f}")
        metric_cols[3].metric("Replans", m["replan_count"])

        metric_cols2 = st.columns(4)
        metric_cols2[0].metric("Collisions", m["collision_count"])
        metric_cols2[1].metric("Plan Latency (ms)", f"{m['planning_latency_ms']:.1f}")
        metric_cols2[2].metric("CPU %", f"{m['cpu_percent_avg']:.1f}")
        metric_cols2[3].metric("Memory (MB)", f"{m['memory_mb_peak']:.1f}")

        st.subheader("Step Log")
        st.dataframe(result.data_pipeline.to_dataframe(), use_container_width=True)

        robot = Robot(position=result.trajectory[-1])
        scan = robot.simulate_lidar(env)
        st.subheader("LiDAR Scan (final position)")
        st.plotly_chart(plot_lidar_polar(scan.angles, scan.ranges), use_container_width=True)

    if st.session_state.comparison_results:
        st.subheader("Algorithm Comparison")
        rows = compare_metrics([r.metrics for r in st.session_state.comparison_results])
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True)
        st.plotly_chart(plot_metrics_comparison(rows), use_container_width=True)


if __name__ == "__main__":
    main()
