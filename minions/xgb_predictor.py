# C:\mikebot\minions\xgb_predictor.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, List

import numpy as np
import pandas as pd

from .minions_base import (
    Minion,
    MinionContext,
    MinionDecision,
)
from .bayesian_calibrator import BayesianCalibrator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class XGBMinionConfig:
    """
    Configuration for the XGBoost predictor minion.
    """
    enabled: bool = True

    # Model loading
    model_path: Path = Path("C:/mikebot/models/xgb_model.json")
    feature_order: Optional[List[str]] = None

    # Probability → directional thresholds
    long_threshold: float = 0.55
    short_threshold: float = 0.45

    # Optional Bayesian calibrator
    use_calibrator: bool = True
    calibrator_config: Dict[str, Any] = None

    def __post_init__(self) -> None:
        if self.calibrator_config is None:
            self.calibrator_config = {}
        if isinstance(self.model_path, str):
            self.model_path = Path(self.model_path)



# ---------------------------------------------------------------------------
# XGBPredictor Minion (modernized)
# ---------------------------------------------------------------------------

class XGBPredictorMinion(Minion):
    """
    XGBoost-based directional decision minion.

    Modernized to:
      - return MinionDecision(action/score/confidence/meta)
      - no longer emit raw orders (portfolio optimizer handles that)
      - use typed MinionContext
      - integrate cleanly with multi-agent vote
    """

    name = "xgb_predictor"

    def __init__(self, config: XGBMinionConfig) -> None:
        self.config = config
        self.model = None
        self.calibrator = (
            BayesianCalibrator(**config.calibrator_config)
            if config.use_calibrator
            else None
        )
        self._load_model()

    # ----------------------------------------------------------------------
    # Model loading
    # ----------------------------------------------------------------------

    def _load_model(self) -> None:
        try:
            import xgboost as xgb  # type: ignore

            if not self.config.model_path.exists():
                logger.warning("XGBPredictor model not found at %s", self.config.model_path)
                self.model = None
                return

            booster = xgb.Booster()
            booster.load_model(str(self.config.model_path))
            self.model = booster
            logger.info("XGBPredictor loaded model from %s", self.config.model_path)

        except Exception as exc:
            logger.exception("XGBPredictor failed to load model: %s", exc)
            self.model = None

    # ----------------------------------------------------------------------
    # Minion interface
    # ----------------------------------------------------------------------

    def decide(self, ctx: MinionContext) -> MinionDecision:
        """
        Modern decision pipeline:
          1. Extract latest feature vector
          2. Predict probability of upward movement
          3. Calibrate (optional)
          4. Convert probability → action
          5. Convert probability → score/confidence
          6. Return MinionDecision (no orders)
        """
        if not self.config.enabled or self.model is None:
            return MinionDecision(
                minion_name=self.name,
                action="hold",
                score=0.0,
                confidence=0.0,
                symbol=None,
                meta={"reason": "disabled_or_no_model"},
            )

        try:
            symbol = ctx.primary_symbol
            if not symbol:
                return MinionDecision(
                    minion_name=self.name,
                    action="hold",
                    score=0.0,
                    confidence=0.0,
                    symbol=None,
                    meta={"reason": "no_primary_symbol"},
                )

            pack = ctx.feature_pack(symbol)
            if not pack or "features" not in pack:
                return MinionDecision(
                    minion_name=self.name,
                    action="hold",
                    score=0.0,
                    confidence=0.0,
                    symbol=symbol,
                    meta={"reason": "no_feature_pack"},
                )

            df = pack["features"]
            if not isinstance(df, pd.DataFrame) or df.empty:
                return MinionDecision(
                    minion_name=self.name,
                    action="hold",
                    score=0.0,
                    confidence=0.0,
                    symbol=symbol,
                    meta={"reason": "empty_features"},
                )

            latest = df.iloc[-1]

            # 1. Build feature vector
            x = self._build_feature_vector(latest)

            # 2. Predict probability
            prob_up = self._predict_prob_up(x)
            if prob_up is None:
                return MinionDecision(
                    minion_name=self.name,
                    action="hold",
                    score=0.0,
                    confidence=0.0,
                    symbol=symbol,
                    meta={"reason": "inference_failed"},
                )

            # 3. Optional Bayesian calibration
            if self.calibrator:
                prob_up = self.calibrator.calibrate(prob_up)

            # 4. Convert probability → action
            action = self._prob_to_action(prob_up)

            # 5. Convert probability → score/confidence
            score = abs(prob_up - 0.5) * 2.0
            confidence = score

            return MinionDecision(
                minion_name=self.name,
                action=action,
                score=float(score),
                confidence=float(confidence),
                symbol=symbol,
                meta={
                    "prob_up": prob_up,
                    "feature_vector": x.tolist(),
                    "raw_action": action,
                    "score": score,
                    "confidence": confidence,
                },
            )

        except Exception as exc:
            logger.exception("XGBPredictor.decide failed: %s", exc)
            return MinionDecision(
                minion_name=self.name,
                action="hold",
                score=0.0,
                confidence=0.0,
                symbol=None,
                meta={"error": str(exc)},
            )

    # ----------------------------------------------------------------------
    # Feature vector construction
    # ----------------------------------------------------------------------

    def _build_feature_vector(self, latest: pd.Series) -> np.ndarray:
        """
        Build feature vector in the exact order expected by the XGB model.
        """
        if self.config.feature_order:
            values = [latest.get(col, 0.0) for col in self.config.feature_order]
        else:
            numeric = latest.select_dtypes(include=[np.number])
            values = numeric.sort_index().values

        arr = np.array(values, dtype=float)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
        return arr

    # ----------------------------------------------------------------------
    # Model inference
    # ----------------------------------------------------------------------

    def _predict_prob_up(self, x: np.ndarray) -> Optional[float]:
        if self.model is None:
            return None

        try:
            import xgboost as xgb  # type: ignore

            dmat = xgb.DMatrix(x.reshape(1, -1))
            preds = self.model.predict(dmat)

            preds = np.asarray(preds)
            if preds.ndim == 2 and preds.shape[1] == 2:
                prob_up = float(preds[0, 1])
            else:
                prob_up = float(preds.ravel()[0])

            return float(np.clip(prob_up, 0.0, 1.0))

        except Exception as exc:
            logger.exception("XGBPredictor inference failed: %s", exc)
            return None

    # ----------------------------------------------------------------------
    # Probability → action
    # ----------------------------------------------------------------------

    def _prob_to_action(self, p: float) -> str:
        """
        Convert probability into a directional action.
        """
        if p >= self.config.long_threshold:
            return "long"
        if p <= self.config.short_threshold:
            return "short"
        return "hold"
