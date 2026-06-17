"""Evaluation metrics for navigation simulation runs."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import psutil

from src.planners.base import PlanResult


@dataclass
class SimulationMetrics:
    """Aggregated metrics for a single simulation run."""

    algorithm: str
    success: bool = False
    collision_count: int = 0
    simulation_time_s: float = 0.0
    path_length: float = 0.0
    total_planning_latency_ms: float = 0.0
    replan_count: int = 0
    steps_taken: int = 0
    cpu_percent: float = 0.0
    memory_mb_peak: float = 0.0
    plan_results: List[PlanResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "success_rate": 1.0 if self.success else 0.0,
            "success": self.success,
            "collision_count": self.collision_count,
            "time_to_destination_s": round(self.simulation_time_s, 3),
            "simulation_time_s": round(self.simulation_time_s, 3),
            "path_length": round(self.path_length, 2),
            "planning_latency_ms": round(self.total_planning_latency_ms, 2),
            "replan_count": self.replan_count,
            "steps_taken": self.steps_taken,
            "cpu_percent_avg": round(self.cpu_percent, 2),
            "memory_mb_peak": round(self.memory_mb_peak, 2),
        }


class MetricsCollector:
    """Tracks resource usage and navigation KPIs during simulation."""

    def __init__(self) -> None:
        self._process = psutil.Process()
        self._start_time: Optional[float] = None

    def start(self) -> None:
        self._start_time = time.perf_counter()

    def elapsed(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.perf_counter() - self._start_time

    def sample_resources(self) -> tuple[float, float]:
        """Sample CPU and memory once (call at end of simulation)."""
        try:
            cpu = self._process.cpu_percent()
            mem = self._process.memory_info().rss / (1024 * 1024)
            return cpu, mem
        except Exception:
            return 0.0, 0.0

    def build_metrics(
        self,
        algorithm: str,
        success: bool,
        collision_count: int,
        path_length: float,
        plan_results: List[PlanResult],
        replan_count: int,
        steps_taken: int,
    ) -> SimulationMetrics:
        total_latency = sum(p.runtime_ms for p in plan_results)
        cpu_percent, memory_mb = self.sample_resources()
        return SimulationMetrics(
            algorithm=algorithm,
            success=success,
            collision_count=collision_count,
            simulation_time_s=self.elapsed(),
            path_length=path_length,
            total_planning_latency_ms=total_latency,
            replan_count=replan_count,
            steps_taken=steps_taken,
            cpu_percent=cpu_percent,
            memory_mb_peak=memory_mb,
            plan_results=plan_results,
        )


def compare_metrics(metrics_list: List[SimulationMetrics]) -> List[Dict[str, Any]]:
    """Return sortable dict rows for side-by-side algorithm comparison."""
    return [m.to_dict() for m in metrics_list]
