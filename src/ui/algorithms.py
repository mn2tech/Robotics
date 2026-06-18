"""Step-by-step algorithm walkthroughs for the Streamlit app."""

from __future__ import annotations

import streamlit as st


def render_algorithm_review() -> None:
    st.header("Algorithm code review")
    st.markdown(
        """
        A walkthrough of each path planner **as implemented in this repo** — step by step,
        like a code review. All planners share the same interface in `src/planners/base.py`
        and return a `PlanResult` with the route, cost, runtime, and nodes expanded.
        """
    )

    st.code(
        """# src/planners/base.py — every planner implements this contract
class BasePlanner(ABC):
    def plan(self, env, start=None, goal=None) -> PlanResult:
        ...

@dataclass
class PlanResult:
    path: List[Position]   # [(row, col), ...] from start to goal
    cost: float
    runtime_ms: float
    success: bool
    nodes_expanded: int
    message: str""",
        language="python",
    )

    algorithm = st.radio(
        "Select algorithm",
        ["A* (A-Star)", "Dijkstra", "RRT"],
        horizontal=True,
    )

    if algorithm.startswith("A*"):
        _render_astar_review()
    elif algorithm == "Dijkstra":
        _render_dijkstra_review()
    else:
        _render_rrt_review()

    st.divider()
    st.subheader("Shared path reconstruction")
    st.markdown(
        """
        A* and Dijkstra both build a `came_from` parent map while searching.
        When the goal is popped, the route is rebuilt by walking backward:
        """
    )
    st.code(
        """# src/planners/base.py
def reconstruct_path(came_from, start, goal):
    current = goal
    path = [current]
    while current != start:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path""",
        language="python",
    )


def _render_astar_review() -> None:
    st.subheader("A* — `src/planners/astar.py`")
    st.markdown(
        """
        **Idea:** Best-first search that prioritizes cells likely to lead to the goal.
        Uses **g(n)** = cost from start, **h(n)** = Manhattan estimate to goal,
        **f(n) = g + h**. On a grid with unit edge costs, this finds an **optimal** path
        while expanding fewer nodes than Dijkstra.
        """
    )

    steps = [
        (
            "Validate start and goal",
            "Reject immediately if either cell is a wall, shelf, or obstacle.",
            """if not env.is_walkable(*start) or not env.is_walkable(*goal):
    return PlanResult(success=False, message="Start or goal not walkable")""",
        ),
        (
            "Initialize search state",
            "Open set = min-heap ordered by f-score. Track g-score, f-score, parents, and closed set.",
            """g_score = {start: 0.0}
f_score = {start: heuristic(start, goal)}  # Manhattan distance
came_from = {}
open_heap = [(f_score[start], 0, start)]""",
        ),
        (
            "Pop the lowest f-score cell",
            "Skip stale heap entries already in `closed`. Mark the cell expanded.",
            """_, _, current = heappop(open_heap)
if current in closed:
    continue
closed.add(current)
nodes_expanded += 1""",
        ),
        (
            "Goal check",
            "If `current == goal`, reconstruct and return the path.",
            """if current == goal:
    path = reconstruct_path(came_from, start, goal)
    return PlanResult(path=path, cost=g_score[goal], success=True)""",
        ),
        (
            "Relax neighbors",
            "For each 4-connected walkable neighbor, update if we found a shorter g-score.",
            """for neighbor in env.neighbors(*current):
    tentative = g_score[current] + 1.0
    if tentative < g_score.get(neighbor, inf):
        came_from[neighbor] = current
        g_score[neighbor] = tentative
        f = tentative + heuristic(neighbor, goal)
        heappush(open_heap, (f, counter, neighbor))""",
        ),
        (
            "Fail if open set empties",
            "No route exists — return `success=False`.",
            """return PlanResult(success=False, message="No path found")""",
        ),
    ]

    _render_steps(steps)

    st.markdown("**Heuristic used:**")
    st.code(
        """def _heuristic(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])  # Manhattan""",
        language="python",
    )

    st.info(
        "**Complexity (typical):** O(E log V) with a binary heap. "
        "On a 20×20 warehouse map, planning usually completes in a few milliseconds."
    )


def _render_dijkstra_review() -> None:
    st.subheader("Dijkstra — `src/planners/dijkstra.py`")
    st.markdown(
        """
        **Idea:** Uniform-cost search — always expands the cheapest known cell first.
        With unit edge weights this is equivalent to a breadth-first wavefront ordered by
        distance. **Guarantees optimality** but typically explores **more nodes than A\***
        because it has no goal-directed heuristic.
        """
    )

    steps = [
        (
            "Validate start and goal",
            "Same walkability guard as A*.",
            """if not env.is_walkable(*start) or not env.is_walkable(*goal):
    return PlanResult(success=False, message="Start or goal not walkable")""",
        ),
        (
            "Initialize distances",
            "Priority queue keyed by cost-so-far. `dist[start] = 0`.",
            """dist = {start: 0.0}
open_heap = [(0.0, 0, start)]
came_from = {}
visited = set()""",
        ),
        (
            "Pop minimum-cost cell",
            "Skip duplicates. Mark visited and count as expanded.",
            """cost, _, current = heappop(open_heap)
if current in visited:
    continue
visited.add(current)
nodes_expanded += 1""",
        ),
        (
            "Goal check",
            "First time we pop the goal, we have the shortest path.",
            """if current == goal:
    path = reconstruct_path(came_from, start, goal)
    return PlanResult(path=path, cost=cost, success=True)""",
        ),
        (
            "Relax neighbors",
            "Each step costs +1. Push improved neighbors onto the heap.",
            """for neighbor in env.neighbors(*current):
    new_cost = cost + 1.0
    if new_cost < dist.get(neighbor, inf):
        dist[neighbor] = new_cost
        came_from[neighbor] = current
        heappush(open_heap, (new_cost, counter, neighbor))""",
        ),
        (
            "Fail if queue empties",
            "Goal unreachable.",
            """return PlanResult(success=False, message="No path found")""",
        ),
    ]

    _render_steps(steps)

    st.info(
        "**A* vs Dijkstra here:** Both return the same optimal path length on this grid. "
        "Compare them with **Compare All Algorithms** — Dijkstra usually shows higher "
        "`nodes_expanded` and slightly higher latency."
    )


def _render_rrt_review() -> None:
    st.subheader("RRT — `src/planners/rrt.py`")
    st.markdown(
        """
        **Idea:** Rapidly-exploring Random Tree — grows a tree from the start by repeatedly
        sampling random points, steering toward them, and collision-checking the edge.
        Operates in **continuous (row, col) space** then snaps to grid cells, so it can
        later extend to SE(2) or 3D configuration spaces.
        """
    )

    steps = [
        (
            "Validate and set up tree",
            "Convert start/goal to floats. Tree maps each node → its parent.",
            """start_f = (float(start[0]), float(start[1]))
goal_f  = (float(goal[0]), float(goal[1]))
tree = {start_f: start_f}""",
        ),
        (
            "Sample a target point",
            "10% of iterations sample the goal directly (bias). Otherwise uniform random.",
            """if random() < goal_sample_rate:   # default 0.1
    sample = goal_f
else:
    sample = (uniform(0, h-1), uniform(0, w-1))""",
        ),
        (
            "Find nearest tree node",
            "Pick the existing node closest in Euclidean distance.",
            """nearest = min(tree.keys(), key=lambda n: dist(n, sample))""",
        ),
        (
            "Steer toward sample",
            "Move at most `step_size` (default 1.5) toward the sample.",
            """new_node = steer(nearest, sample)
# if distance > step_size, interpolate partway""",
        ),
        (
            "Collision check",
            "Ray-march along the segment; reject if any grid cell is blocked.",
            """if not collision_free(env, nearest, new_node):
    continue
tree[new_node] = nearest""",
        ),
        (
            "Goal proximity test",
            "If new node is within `step_size` of goal and the edge is clear, connect and extract path.",
            """if dist(new_node, goal_f) <= step_size:
    if collision_free(env, new_node, goal_f):
        tree[goal_f] = new_node
        path = snap_path(env, extract_path(tree, start_f, goal_f))
        return PlanResult(path=path, success=True)""",
        ),
        (
            "Iteration limit",
            "Default 2000 iterations. May fail on tight maps — path is **not guaranteed optimal**.",
            """return PlanResult(success=False,
    message="RRT failed to find path within iteration limit")""",
        ),
    ]

    _render_steps(steps)

    st.markdown("**Snap to grid (post-processing):**")
    st.code(
        """def _snap_path(env, path_f):
    snapped = []
    for pr, pc in path_f:
        r, c = int(round(pr)), int(round(pc))
        if env.is_walkable(r, c):
            snapped.append((r, c))
    return snapped""",
        language="python",
    )

    st.info(
        "**Tradeoff:** RRT is fast to implement for complex spaces but paths may be longer "
        "and results can vary between runs due to random sampling."
    )


def _render_steps(steps: list[tuple[str, str, str]]) -> None:
    for index, (title, explanation, code) in enumerate(steps, start=1):
        with st.expander(f"Step {index} — {title}", expanded=(index == 1)):
            st.markdown(explanation)
            st.code(code, language="python")
