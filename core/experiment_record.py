# mikebot/core/experiment_record.py

from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class ExperimentRecord:
    """
    Canonical record of a single training experiment / model version.

    This is the bridge object between TrainPipeline and ModelLineageRegistry.
    It is intentionally comprehensive so that the lineage system and UI
    (LineageTab) can evolve without requiring TrainPipeline to be aware of
    presentation details.

    Semantics (v4):

      - symbol:        Trading symbol (e.g. "BTCUSD").
      - timeframe:     Base timeframe label used for this experiment (e.g. "M5").
                       Even for symbol-level MULTITF models, this is the base TF
                       from the experiment config.
      - model_type:    Model family key (e.g. "xgb", "rf").
      - version_id:    Lineage version id (e.g. "v7"). May be left empty to let
                       ModelLineageRegistry assign a sequential id.
      - parent_version_id:
                       Optional version id of the parent model this one evolved
                       from. May be None for the first run for a symbol.
      - model_tag:     Human-readable model tag, typically the timestamped tag
                       used in filenames (e.g. "BTCUSD_20250101_153000").
      - metrics:       Aggregate metrics dict from TrainPipeline, including both
                       basic and extended metrics.
      - regime_performance:
                       Per-regime metrics (if available), keyed by regime label.
      - strategy_performance:
                       Per-strategy performance summary (if available), keyed
                       by strategy name.
      - feature_importance:
                       Normalized feature importance mapping: feature_name -> score.
      - experiment_type:
                       Free-form string identifying the experiment family, e.g.
                       "multitf_v4_baseline", "ablation_no_regime", etc.
      - notes:         Optional human-readable notes attached to the experiment,
                       e.g. rationale or configuration summary.
      - model_path:    Filesystem path to the persisted model artifact (.bin).
      - metrics_path:  Filesystem path to the persisted metrics artifact (.json).
    """

    symbol: str
    timeframe: str
    model_type: str

    version_id: str
    parent_version_id: Optional[str]

    model_tag: str

    metrics: Dict[str, Any]
    regime_performance: Dict[str, Any]
    strategy_performance: Dict[str, Any]
    feature_importance: Dict[str, float]

    experiment_type: str
    notes: str

    model_path: str
    metrics_path: str

    @classmethod
    def from_pipeline_outputs(
        cls,
        *,
        symbol: str,
        timeframe: str,
        model_type: str,
        version_id: str,
        parent_version_id: Optional[str],
        metrics: Dict[str, Any],
        experiment_type: str = "",
        notes: str = "",
        model_path: Path,
        metrics_path: Path,
        model_tag: str = "",
    ) -> ExperimentRecord:
        """
        Convenience constructor to build an ExperimentRecord from the data
        TrainPipeline already has at the end of a run.

        This assumes TrainPipeline has a single consolidated `metrics` dict
        containing:
          - basic metrics
          - extended metrics
          - "regime_performance"
          - "strategy_performance"
          - "feature_importance"
        """
        regime_perf = metrics.get("regime_performance", {}) or {}
        strat_perf = metrics.get("strategy_performance", {}) or {}
        feat_imp = metrics.get("feature_importance", {}) or {}

        return cls(
            symbol=symbol,
            timeframe=timeframe,
            model_type=model_type,
            version_id=version_id,
            parent_version_id=parent_version_id,
            model_tag=model_tag,
            metrics=metrics,
            regime_performance=regime_perf,
            strategy_performance=strat_perf,
            feature_importance=feat_imp,
            experiment_type=experiment_type,
            notes=notes,
            model_path=str(model_path),
            metrics_path=str(metrics_path),
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a plain dict for JSON serialization or storage inside the
        lineage registry structures.
        """
        return asdict(self)