# AI Navigation Simulator

A Python-based robotics simulation project that demonstrates **navigation**, **spatial reasoning**, **path planning**, **sensor simulation**, **inference pipelines**, **data logging**, and **evaluation metrics** for a robot moving through a warehouse-like grid environment.

Built as a portfolio project for **AI/ML Navigation Engineer** and **Robotics Software Engineer** roles.

---

## What This Project Does

- Simulates a **2D warehouse grid** with walls, shelves, aisles, and dynamic obstacles
- Plans collision-free paths using **A\***, **Dijkstra**, and **RRT**
- Moves a robot from start to goal while avoiding obstacles
- **Replans** when a new obstacle appears mid-run
- Simulates **LiDAR range sensing** for spatial awareness
- Runs an **inference pipeline** (environment + sensors → movement decisions)
- Logs trajectories and planning events via a **data pipeline**
- Tracks KPIs: success rate, collisions, time, path length, latency, replans, CPU/memory
- Provides a **Streamlit dashboard** for interactive simulation and algorithm comparison

The codebase is structured so it can later extend to **3D voxels**, **ROS 2**, **Gazebo**, or **NVIDIA Isaac Sim** without rewriting the core planning and evaluation layers.

---

## Why This Matches an AI/ML Navigation Engineer Role

| Skill Area | How This Project Demonstrates It |
|---|---|
| Path Planning | A*, Dijkstra, RRT on occupancy grids |
| Spatial Reasoning | Feature maps, local maps, LiDAR sector analysis |
| Sensor Simulation | Ray-cast LiDAR with configurable FOV/range |
| Inference Pipeline | Observation → action decisions (replan/follow/stop) |
| Data Pipeline | Structured logging of poses, actions, plans |
| Evaluation | Quantitative metrics and side-by-side algorithm comparison |
| Systems Thinking | Modular architecture ready for sim-to-real transfer |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Streamlit Dashboard (app.py)                │
│   Algorithm select · Start/Goal · Obstacles · Metrics · Compare   │
└────────────────────────────┬────────────────────────────────────┘
                             │
┌────────────────────────────▼────────────────────────────────────┐
│                  NavigationSimulator (simulation.py)            │
│         Orchestrates plan → infer → act → log → evaluate        │
└─┬──────────┬──────────────┬──────────────┬─────────────────────┘
  │          │              │              │
  ▼          ▼              ▼              ▼
┌────────┐ ┌──────────┐ ┌────────────┐ ┌──────────────┐
│ Robot  │ │ Planners │ │ Inference  │ │ Data Pipeline│
│ LiDAR  │ │ A*       │ │ Pipeline   │ │ (pandas log) │
│ Pose   │ │ Dijkstra │ │ decisions  │ └──────────────┘
└───┬────┘ │ RRT      │ └─────┬──────┘
    │      └────┬─────┘       │
    │           │             │
    ▼           ▼             ▼
┌───────────────────────────────────────┐
│     WarehouseEnvironment (grid)       │
│  walls · shelves · obstacles · goals  │
└───────────────────────────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ MetricsCollector │
                    │ success · time   │
                    │ collisions · CPU │
                    └─────────────────┘
```

---

## Algorithms

| Algorithm | Type | Notes |
|---|---|---|
| **A\*** | Optimal grid search | Manhattan heuristic, 4-connected |
| **Dijkstra** | Optimal uniform-cost | Explores more nodes than A* on large maps |
| **RRT** | Sampling-based | Continuous steering, grid-snapped output |

---

## Metrics

| Metric | Description |
|---|---|
| Success rate | 1 if robot reaches goal, else 0 |
| Collision count | Failed move attempts into blocked cells |
| Time to destination | Wall-clock simulation duration (seconds) |
| Path length | Euclidean sum of trajectory waypoints |
| Planning latency | Total planner runtime across all replans (ms) |
| Replan count | Number of times path was recomputed |
| CPU / Memory | Process usage via `psutil` |

---

## Project Structure

```
ai-navigation-simulator/
├── README.md
├── requirements.txt
├── app.py                      # Streamlit dashboard
├── src/
│   ├── environment.py          # Warehouse grid
│   ├── robot.py                # Robot + LiDAR
│   ├── simulation.py           # Simulation engine
│   ├── planners/
│   │   ├── astar.py
│   │   ├── dijkstra.py
│   │   └── rrt.py
│   ├── pipelines/
│   │   ├── data_pipeline.py
│   │   └── inference_pipeline.py
│   ├── evaluation/
│   │   └── metrics.py
│   └── visualization/
│       └── plotter.py
└── tests/
    ├── test_astar.py
    └── test_dijkstra.py
```

---

## How to Run Locally

### 1. Clone and enter the project

```bash
cd ai-navigation-simulator
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run tests

```bash
pytest tests/ -v
```

### 5. Launch the dashboard

```bash
streamlit run app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

---

## Using the Dashboard

1. **Select algorithm** (A*, Dijkstra, or RRT) in the sidebar
2. **Set start and goal** coordinates and click *Apply Start/Goal*
3. **Add/remove obstacles** using the obstacle editor
4. Configure a **dynamic obstacle** and the step at which it appears (replan test)
5. Click **Run Simulation** to plan, navigate, and view metrics
6. Click **Compare All Algorithms** for a side-by-side benchmark table and chart

---

## Example Results

On a 20×20 warehouse grid from `(1,1)` to `(18,18)`:

| Algorithm | Success | Path Length | Plan Latency | Replans |
|---|---|---|---|---|
| A* | Yes | ~34–36 | ~1–5 ms | 1 (with dynamic obstacle) |
| Dijkstra | Yes | ~34–36 | ~3–15 ms | 1 |
| RRT | Yes | ~30–40 | ~5–50 ms | 1 |

*Exact values depend on obstacle layout and dynamic obstacle placement.*

---

## Screenshots

To capture portfolio screenshots:

1. Run `streamlit run app.py`
2. Click **Run Simulation** with A* selected
3. Screenshot the grid view with the yellow path and purple robot marker
4. Screenshot the **Metrics** row and **Step Log** table
5. Click **Compare All Algorithms** and screenshot the comparison chart

Save images to a `docs/screenshots/` folder if you add them to your portfolio site.

---

## Limitations (v1)

- **2D grid only** — 3D and continuous dynamics are stubbed via extensible interfaces
- **Discrete movement** — one cell per step, not velocity-based control
- **Rule-based inference** — no trained neural policy yet
- **RRT** is probabilistic — may occasionally fail on tight maps (increase iterations)
- **No ROS integration** in v1 — designed for easy future bridging

---

## Future Improvements

- [ ] 3D voxel grid and SE(3) planners
- [ ] ROS 2 `nav2` / `MoveIt 2` bridge
- [ ] Gazebo or Isaac Sim visualization
- [ ] Learned policy replacing rule-based inference (PyTorch / ONNX)
- [ ] Multi-robot coordination
- [ ] SLAM module with noisy odometry
- [ ] Export logged data to Parquet for ML training pipelines
- [ ] Animated playback in the Streamlit UI

---

## License

MIT — free to use and adapt for portfolio and learning purposes.
