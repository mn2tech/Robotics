"""On-page project overview and under-the-hood explanation for the Streamlit app."""

from __future__ import annotations

import streamlit as st


def render_under_the_hood() -> None:
    st.header("How this simulator works")
    st.markdown(
        """
        This app is an **end-to-end navigation prototype**: it plans a route, moves a robot
        through a warehouse grid, reacts to new obstacles, logs every step, and scores the run.
        Below is what happens under the hood when you click **Run Simulation**.
        """
    )

    st.subheader("Pipeline at a glance")
    st.markdown(
        """
        ```
        Environment + Start/Goal
                │
                ▼
         Path Planner (A* / Dijkstra / RRT)
                │
                ▼
         Inference Pipeline  ──►  follow path · replan · stop
                │
                ▼
         Robot moves one grid cell per step
                │
                ├──► Data Pipeline (step + plan logs)
                └──► Metrics Collector (success, latency, replans, …)
        ```
        """
    )

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Environment")
        st.markdown(
            """
            - **20×20 occupancy grid** with walls, shelves, aisles, and free cells
            - **Start** (green) and **goal** (red) positions you configure in the sidebar
            - **Static obstacles** added via the obstacle editor
            - **Dynamic obstacles** that appear mid-run to test replanning
            - Walkability uses 4-connected grid neighbors (up / down / left / right)
            """
        )

        st.subheader("Path planners")
        st.markdown(
            """
            | Algorithm | Approach | Tradeoff |
            |-----------|----------|----------|
            | **A\*** | Heuristic search | Fast, optimal on grids |
            | **Dijkstra** | Uniform-cost search | Optimal, explores more nodes |
            | **RRT** | Random sampling | Handles harder spaces, path may vary |

            Each planner returns a cell-by-cell route from start to goal, or reports
            **no path** when the goal is unreachable.
            """
        )

    with col2:
        st.subheader("Simulation loop")
        st.markdown(
            """
            1. **Plan** — selected algorithm builds an initial route
            2. **Decide** — inference pipeline chooses an action each step
            3. **Act** — robot advances one cell toward the next waypoint
            4. **Sense** — simulated LiDAR ray-casts against walls and obstacles
            5. **Log** — position, action, and sensor summary are recorded
            6. **Repeat** until the robot reaches the goal, planning fails, or max steps

            **Replans** happen only when a **new dynamic obstacle blocks the current path** —
            not on every step.
            """
        )

        st.subheader("Inference pipeline")
        st.markdown(
            """
            The inference layer mimics a deployed navigation stack:

            - **Inputs:** occupancy feature map, local obstacle window, LiDAR ranges, pose, goal
            - **Outputs:** `follow_path`, `replan`, or `stop`

            It is rule-based today, but the interface is designed so a learned policy
            (e.g. PyTorch / ONNX) can replace `decide_action` without changing the simulator.
            """
        )

    st.subheader("Metrics & evaluation")
    st.markdown(
        """
        | Metric | What it measures |
        |--------|------------------|
        | **Success** | Robot reached the goal with a valid plan |
        | **Path length** | Number of grid steps in the final planned route |
        | **Planning latency** | Total planner execution time (ms), not full sim time |
        | **Replans** | Times a new obstacle forced a fresh plan |
        | **Collisions** | Blocked move attempts during execution |
        | **Simulation time** | Wall-clock duration of the full run |
        | **CPU / Memory** | Resource sample taken at end of run |

        Use **Compare All Algorithms** to benchmark A\*, Dijkstra, and RRT on the same map.
        """
    )

    st.subheader("Data pipeline")
    st.markdown(
        """
        Every run produces structured logs you can inspect in the **Step Log** table:

        - **Step records** — row/col, action, LiDAR minimum range, replan flag, timestamp
        - **Plan records** — algorithm, path length, cost, runtime, nodes expanded

        Records are designed for export to CSV / Parquet for offline analysis or ML training.
        """
    )

    st.subheader("Visualization")
    st.markdown(
        """
        - **Blue line** — planned path from the active algorithm
        - **Magenta trail** — cells the robot has actually visited
        - **Robot marker** — current position; turns red at the goal when successful
        - **Play / Pause** — Plotly frame animation (client-side, no page reload)

        The grid, walls, shelves, and obstacles are rendered as static cell shapes; only the
        robot and trail update during playback for smooth animation.
        """
    )

    st.subheader("Tech stack")
    st.markdown(
        """
        - **Python** — simulation engine, planners, metrics
        - **NumPy** — grid maps and sensor arrays
        - **Plotly** — interactive grid and walkthrough charts
        - **Streamlit** — dashboard and deployment
        - **pytest** — 35+ tests covering planners, simulation, and visualization
        """
    )

    st.info(
        "Source code: [github.com/mn2tech/Robotics](https://github.com/mn2tech/Robotics) — "
        "modular layout ready for ROS 2, 3D voxels, or learned navigation policies."
    )
