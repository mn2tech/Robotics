"""
Simulation engine orchestrating planning, inference, and metrics.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from src.environment import WarehouseEnvironment
from src.evaluation.metrics import MetricsCollector, SimulationMetrics
from src.pipelines.data_pipeline import DataPipeline
from src.pipelines.inference_pipeline import InferencePipeline
from src.planners.base import BasePlanner, PlanResult, Position
from src.robot import Robot


@dataclass
class SimulationResult:
    """Full output of a navigation simulation run."""

    metrics: SimulationMetrics
    final_path: List[Position]
    trajectory: List[Position]
    env: WarehouseEnvironment
    data_pipeline: DataPipeline


@dataclass
class NavigationSimulator:
    """Runs end-to-end navigation with replanning on dynamic obstacles."""

    env: WarehouseEnvironment
    robot: Robot
    planner: BasePlanner
    max_steps: int = 500
    step_delay: float = 0.0
    dynamic_obstacles: List[Position] = field(default_factory=list)
    obstacle_trigger_step: Optional[int] = None

    def run(self) -> SimulationResult:
        inference = InferencePipeline(planner=self.planner)
        data_pipe = DataPipeline()
        metrics_col = MetricsCollector()
        metrics_col.start()

        self.robot.reset(self.env.start)
        plan_results: List[PlanResult] = []
        trajectory: List[Position] = [self.env.start]
        replan_count = 0
        current_path: List[Position] = []
        path_length = 0

        plan = inference.plan_path(self.env)
        plan_results.append(plan)
        data_pipe.log_plan(self.planner.name, plan, replan_number=0)

        if plan.success:
            current_path = plan.path
            path_length = BasePlanner.path_step_count(current_path)
            self.robot.set_path(current_path)
        else:
            metrics = metrics_col.build_metrics(
                algorithm=self.planner.name,
                success=False,
                collision_count=self.robot.collision_count,
                path_length=0,
                plan_results=plan_results,
                replan_count=0,
                steps_taken=0,
            )
            return SimulationResult(
                metrics=metrics,
                final_path=[],
                trajectory=trajectory,
                env=self.env,
                data_pipeline=data_pipe,
            )

        step = 0
        while step < self.max_steps:
            if self.robot.has_reached_goal(self.env.goal):
                break

            new_obs = None
            if (
                self.obstacle_trigger_step is not None
                and step == self.obstacle_trigger_step
            ):
                for pos in self.dynamic_obstacles:
                    self.env.add_obstacle(*pos)
                new_obs = list(self.dynamic_obstacles)

            action, _ = inference.decide_action(
                self.env, self.robot, current_path, new_obstacles=new_obs
            )

            scan = self.robot.simulate_lidar(self.env)
            data_pipe.log_step(
                step=step,
                position=self.robot.position,
                action=action,
                lidar_min_range=float(scan.ranges.min()),
                replan_triggered=action == "replan",
            )

            if action == "replan":
                replan_count += 1
                plan = inference.plan_path(self.env, start=self.robot.position)
                plan_results.append(plan)
                data_pipe.log_plan(self.planner.name, plan, replan_number=replan_count)
                if plan.success:
                    current_path = plan.path
                    path_length = BasePlanner.path_step_count(current_path)
                    self.robot.set_path(current_path)
                else:
                    break

            elif action == "stop":
                break

            elif action in ("follow_path", "wait"):
                if action == "follow_path":
                    self.robot.step_toward_waypoint(self.env)

            trajectory.append(self.robot.position)
            step += 1

            if self.robot.has_reached_goal(self.env.goal):
                break

        reached_goal = self.robot.has_reached_goal(self.env.goal)
        has_valid_path = bool(current_path)
        success = reached_goal if has_valid_path else False

        metrics = metrics_col.build_metrics(
            algorithm=self.planner.name,
            success=success,
            collision_count=self.robot.collision_count,
            path_length=float(path_length),
            plan_results=plan_results,
            replan_count=replan_count,
            steps_taken=step,
        )

        return SimulationResult(
            metrics=metrics,
            final_path=current_path,
            trajectory=trajectory,
            env=self.env,
            data_pipeline=data_pipe,
        )
