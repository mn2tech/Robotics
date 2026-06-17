"""Data logging pipeline for navigation simulation runs."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from src.planners.base import PlanResult


@dataclass
class DataPipeline:
    """
    Logs robot state, planning decisions, and metrics to structured records.

    Records can be exported to CSV/Parquet for offline ML training pipelines.
    """

    run_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    records: List[Dict[str, Any]] = field(default_factory=list)
    plan_records: List[Dict[str, Any]] = field(default_factory=list)

    def log_step(
        self,
        step: int,
        position: tuple,
        action: str,
        lidar_min_range: float,
        replan_triggered: bool = False,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        record = {
            "run_id": self.run_id,
            "step": step,
            "row": position[0],
            "col": position[1],
            "action": action,
            "lidar_min_range": lidar_min_range,
            "replan_triggered": replan_triggered,
            "timestamp": datetime.now().isoformat(),
        }
        if extra:
            record.update(extra)
        self.records.append(record)

    def log_plan(
        self,
        algorithm: str,
        plan_result: PlanResult,
        replan_number: int = 0,
    ) -> None:
        self.plan_records.append(
            {
                "run_id": self.run_id,
                "algorithm": algorithm,
                "replan_number": replan_number,
                "success": plan_result.success,
                "path_length": len(plan_result.path),
                "cost": plan_result.cost,
                "runtime_ms": plan_result.runtime_ms,
                "nodes_expanded": plan_result.nodes_expanded,
                "message": plan_result.message,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.records)

    def plans_to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.plan_records)

    def export_csv(self, path: str) -> None:
        self.to_dataframe().to_csv(path, index=False)

    def clear(self) -> None:
        self.records.clear()
        self.plan_records.clear()
