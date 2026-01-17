# mikebot/core/training_entrypoint_v4.py

from __future__ import annotations

from typing import Any, Callable, Optional

import pandas as pd

from mikebot.core.training_pipeline_v4 import TrainingConfigV4
from mikebot.core.training_orchestrator_v4 import (
    TrainingOrchestratorV4,
    OrchestratorResultV4,
)


def run_training_v4(
    candles: pd.DataFrame,
    target: pd.Series,
    model_factory: Callable[[], Any],
    config: Optional[TrainingConfigV4] = None,
    sample_weight: Optional[pd.Series] = None,
) -> OrchestratorResultV4:
    """
    Programmatic entrypoint for a full v4 training run.

    This is the single function you call from:
    - notebooks
    - CLI wrappers
    - scheduler jobs
    - experiment runners

    Parameters
    ----------
    candles:
        Raw OHLCV DataFrame.

    target:
        Target labels/values aligned to candles.index.

    model_factory:
        Callable that returns an untrained model with fit() and predict()
        methods. Example: `lambda: xgboost.XGBRegressor(...)`.

    config:
        Optional TrainingConfigV4. If None, a default config is used.

    sample_weight:
        Optional per-sample weights aligned to candles.index.

    Returns
    -------
    OrchestratorResultV4
    """
    cfg = config or TrainingConfigV4()
    orchestrator = TrainingOrchestratorV4(config=cfg, model_factory=model_factory)

    return orchestrator.run(
        candles=candles,
        target=target,
        sample_weight=sample_weight,
    )


__all__ = [
    "run_training_v4",
    "TrainingConfigV4",
    "TrainingOrchestratorV4",
    "OrchestratorResultV4",
]