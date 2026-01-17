# mikebot/core/training_orchestrator_v4.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

import pandas as pd

# ⬅️ NEW: import unified pipeline instead of old v4 pipeline
from mikebot.core.unified_training_pipeline import (
    UnifiedTrainingPipeline,
    UnifiedTrainingConfig,
)


@dataclass
class OrchestratorResultV4:
    """
    Canonical result bundle for a unified v4 training run.
    """
    model: Any
    features: pd.DataFrame
    target: pd.Series
    metrics: Dict[str, float]
    config: UnifiedTrainingConfig


class TrainingOrchestratorV4:
    """
    High-level coordinator for unified v4 training runs.

    This is a thin orchestration layer around UnifiedTrainingPipeline that:

    - Owns the UnifiedTrainingConfig
    - Owns the model_factory
    - Runs a single, deterministic training job
    - Returns a structured result bundle

    It does NOT:
    - Persist models (that’s model_saver_v4)
    - Register models (that’s model_registry_v4)
    - Handle CLI / UI (that’s training_entrypoint_v4)
    """

    def __init__(
        self,
        config: UnifiedTrainingConfig,
        model_factory: Callable[[], Any],
    ) -> None:
        self.config = config
        self.model_factory = model_factory

        # ⬅️ NEW: unified pipeline
        self._pipeline = UnifiedTrainingPipeline(
            config=self.config,
            model_factory=self.model_factory,
        )

    def run(
        self,
        candles: pd.DataFrame,
        target: Optional[pd.Series] = None,
        sample_weight: Optional[pd.Series] = None,
        regimes: Optional[pd.DataFrame] = None,
        strategies: Optional[Dict[str, pd.DataFrame]] = None,
    ) -> OrchestratorResultV4:
        """
        Execute a full unified v4 training run.

        Parameters
        ----------
        candles:
            Raw OHLCV DataFrame.

        target:
            Optional target labels aligned to candles.index.
            If None and triple-barrier is enabled, the pipeline will generate it.

        sample_weight:
            Optional per-sample weights aligned to candles.index.

        regimes:
            Optional regime labels for diagnostics.

        strategies:
            Optional strategy signals for diagnostics.

        Returns
        -------
        OrchestratorResultV4
        """
        result = self._pipeline.train(
            candles=candles,
            target=target,
            sample_weight=sample_weight,
            regimes=regimes,
            strategies=strategies,
        )

        return OrchestratorResultV4(
            model=result["model"],
            features=result["features"],
            target=result["target"],
            metrics=result["metrics"],
            config=self.config,
        )


# ----------------------------------------------------------------------
# Global accessor for TrainingOrchestratorV4
# ----------------------------------------------------------------------

_training_orchestrator_global = None


def set_global_training_orchestrator(orchestrator) -> None:
    """
    Register the process-wide TrainingOrchestratorV4 instance.
    """
    global _training_orchestrator_global
    _training_orchestrator_global = orchestrator


def get_global_training_orchestrator():
    """
    Return the process-wide TrainingOrchestratorV4 instance, or None if not set.
    """
    return _training_orchestrator_global