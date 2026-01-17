from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple

import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from .minions_base import Minion, MinionDecision, MinionContext

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Cluster profile
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ClusterProfile:
    """Summary statistics for a single cluster."""
    id: int
    size: int
    center: np.ndarray
    avg_return: float
    win_rate: float
    volatility: float


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class ClusteringConfig:
    """
    Configuration for the clustering scout minion.
    """
    n_clusters: int = 6
    max_history: int = 10_000
    min_cluster_size: int = 50

    # Canonical default feature set (auto-populated if empty)
    feature_columns: List[str] = field(default_factory=list)

    # Use ret_1 instead of raw close prices
    return_column: str = "ret_1"

    # Lookahead for cluster scoring
    lookahead_bars: int = 5

    # Prevent constant refitting
    refit_interval: int = 250

    random_state: int = 42


# ---------------------------------------------------------------------------
# ClusteringScout Minion
# ---------------------------------------------------------------------------

class ClusteringScout(Minion):
    """
    Regime / pattern discovery minion using unsupervised clustering.

    Responsibilities:
      - Fit KMeans on recent feature history
      - Compute per-cluster performance stats
      - Predict current regime for the latest bar
      - Emit a MinionDecision containing:
            - regime_id
            - regime_score
            - confidence
            - metadata for orchestrator + telemetry
    """

    name = "clustering_scout"

    def __init__(self, config: Optional[ClusteringConfig] = None) -> None:
        self.config = config or ClusteringConfig()
        self._kmeans: Optional[KMeans] = None
        self._cluster_profiles: Dict[int, ClusterProfile] = {}
        self._last_fit_size: int = 0

    # ----------------------------------------------------------------------
    # Minion API
    # ----------------------------------------------------------------------

    def decide(self, ctx: MinionContext) -> MinionDecision:
        symbol = ctx.primary_symbol
        if not symbol:
            return MinionDecision(
                minion_name=self.name,
                action="hold",
                score=0.0,
                confidence=0.0,
                meta={"reason": "no_primary_symbol"},
            )

        pack = ctx.feature_pack(symbol)
        if not pack or "features" not in pack:
            return MinionDecision(
                minion_name=self.name,
                action="hold",
                score=0.0,
                confidence=0.0,
                meta={"reason": "no_feature_pack"},
            )

        df = pack["features"]
        if not isinstance(df, pd.DataFrame) or df.empty:
            return MinionDecision(
                minion_name=self.name,
                action="hold",
                score=0.0,
                confidence=0.0,
                meta={"reason": "empty_features"},
            )

        # Ensure default feature set exists
        self._ensure_default_feature_columns(df)

        # Fit model if needed
        self._maybe_fit(df)

        # Predict regime
        latest = df.iloc[-1]
        regime_id, regime_score = self.predict_regime(latest)

        if regime_id is None:
            return MinionDecision(
                minion_name=self.name,
                action="hold",
                score=0.0,
                confidence=0.0,
                meta={"reason": "no_regime"},
            )

        confidence = float(abs(regime_score)) if regime_score is not None else 0.0

        return MinionDecision(
            minion_name=self.name,
            action="hold",
            score=0.0,
            confidence=confidence,
            meta={
                "cluster_id": regime_id,
                "regime_id": regime_id,
                "regime_score": regime_score,
                "confidence": confidence,
                "profiles": {
                    cid: {
                        "size": p.size,
                        "avg_return": p.avg_return,
                        "win_rate": p.win_rate,
                        "volatility": p.volatility,
                    }
                    for cid, p in self._cluster_profiles.items()
                },
            },
        )

    # ----------------------------------------------------------------------
    # Fitting logic
    # ----------------------------------------------------------------------

    def _ensure_default_feature_columns(self, df: pd.DataFrame) -> None:
        """
        If no feature columns were provided, populate a canonical default set.
        """
        if not self.config.feature_columns:
            defaults = [
                "volatility",
                "drift_score",
                "rsi",
                "ma_spread_pct",
                "volume_zscore",
                "ret_1",
            ]
            self.config.feature_columns = [c for c in defaults if c in df.columns]

            if not self.config.feature_columns:
                raise ValueError(
                    "ClusteringScout: No valid default feature columns found in DataFrame"
                )

    def _maybe_fit(self, df: pd.DataFrame) -> None:
        """
        Fit the clustering model if:
          - no model exists, or
          - enough new data has arrived since last fit
        """
        if self._kmeans is None:
            self.fit(df)
            return

        new_rows = len(df) - self._last_fit_size
        if new_rows >= self.config.refit_interval:
            self.fit(df)

    def fit(self, features: pd.DataFrame) -> None:
        df = features.tail(self.config.max_history)

        missing = set(self.config.feature_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing feature columns for clustering: {missing}")

        X = df[self.config.feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
        if len(X) < self.config.n_clusters:
            logger.warning(
                "Not enough samples (%d) to fit %d clusters",
                len(X),
                self.config.n_clusters,
            )
            return

        self._kmeans = KMeans(
            n_clusters=self.config.n_clusters,
            random_state=self.config.random_state,
            n_init="auto",
        )
        labels = self._kmeans.fit_predict(X)
        self._last_fit_size = len(df)

        self._cluster_profiles = self._build_cluster_profiles(df, labels)

    # ----------------------------------------------------------------------
    # Prediction
    # ----------------------------------------------------------------------

    def predict_regime(
        self,
        latest_features: pd.Series,
    ) -> Tuple[Optional[int], Optional[float]]:
        if self._kmeans is None:
            return None, None

        x = (
            latest_features.reindex(self.config.feature_columns)
            .fillna(0.0)
            .to_numpy()
            .reshape(1, -1)
        )
        cluster_id = int(self._kmeans.predict(x)[0])

        profile = self._cluster_profiles.get(cluster_id)
        if profile is None:
            return cluster_id, None

        score = profile.avg_return * profile.win_rate
        return cluster_id, float(score)

    # ----------------------------------------------------------------------
    # Cluster profiling
    # ----------------------------------------------------------------------

    def _build_cluster_profiles(
        self,
        df: pd.DataFrame,
        labels: np.ndarray,
    ) -> Dict[int, ClusterProfile]:
        returns = self._compute_lookahead_returns(df)

        profiles: Dict[int, ClusterProfile] = {}
        centers = self._kmeans.cluster_centers_ if self._kmeans is not None else None

        for cid in range(self.config.n_clusters):
            idx = np.where(labels == cid)[0]
            if len(idx) < self.config.min_cluster_size:
                continue

            cluster_returns = returns.iloc[idx].dropna()
            if cluster_returns.empty:
                continue

            avg_ret = float(cluster_returns.mean())
            win_rate = float((cluster_returns > 0).mean())
            vol = float(cluster_returns.std())

            center = centers[cid] if centers is not None else np.zeros(len(self.config.feature_columns))

            profiles[cid] = ClusterProfile(
                id=cid,
                size=len(idx),
                center=center,
                avg_return=avg_ret,
                win_rate=win_rate,
                volatility=vol,
            )

        return profiles

    def _compute_lookahead_returns(self, df: pd.DataFrame) -> pd.Series:
        col = self.config.return_column
        if col not in df.columns:
            raise ValueError(f"return_column '{col}' not found in features DataFrame")

        price = df[col].astype(float)
        shifted = price.shift(-self.config.lookahead_bars)
        ret = (shifted - price) / price.replace(0, np.nan)
        return ret