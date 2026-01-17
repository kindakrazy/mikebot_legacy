# C:\mikebot\minions\sequencer_lstm.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .minions_base import (
    Minion,
    MinionContext,
    MinionDecision,
)

logger = logging.getLogger(__name__)


from pathlib import Path

@dataclass
class SequencerLSTMConfig:
    """
    Configuration for the Sequencer/LSTM minion.
    """
    enabled: bool = True

    # Model loading
    model_path: Path = Path("C:/mikebot/models/sequencer_lstm.onnx")

    # Sequence construction
    sequence_length: int = 32
    feature_columns: Optional[List[str]] = None

    # Probability → signal thresholds
    long_threshold: float = 0.55
    short_threshold: float = 0.45

    # Optional normalization
    use_zscore_norm: bool = True

    def __post_init__(self):
        # Normalize model_path if loaded from JSON/YAML
        if isinstance(self.model_path, str):
            self.model_path = Path(self.model_path)

class SequencerLSTM(Minion):
    """
    Sequence/LSTM‑style decision minion.
    Responsibilities:
      - Build a rolling sequence of features for the primary symbol
      - Run it through an LSTM‑style model (ONNX or similar)
      - Interpret the output as probability of upward movement
      - Return a MinionDecision with action/score/confidence
    """

    name = "sequencer_lstm"

    def __init__(self, config: SequencerLSTMConfig) -> None:
        self.config = config
        self._session = None  # ONNX runtime session or similar
        self._load_model()

    # -------------------------------------------------------------------------
    # Model loading
    # -------------------------------------------------------------------------

    def _load_model(self) -> None:
        """
        Load the LSTM model (ONNX).
        """
        try:
            import onnxruntime as ort  # type: ignore

            if not self.config.model_path.exists():
                logger.warning("SequencerLSTM model not found at %s", self.config.model_path)
                self._session = None
                return

            self._session = ort.InferenceSession(str(self.config.model_path))
            logger.info("SequencerLSTM loaded model from %s", self.config.model_path)
        except Exception as exc:
            logger.exception("SequencerLSTM failed to load model: %s", exc)
            self._session = None

    # -------------------------------------------------------------------------
    # Minion interface
    # -------------------------------------------------------------------------

    def decide(self, ctx: MinionContext) -> MinionDecision:
        if not self.config.enabled or self._session is None:
            return self._hold_decision("disabled_or_no_model")

        try:
            symbol = ctx.primary_symbol
            if not symbol:
                return self._hold_decision("no_primary_symbol")

            feat_pack = ctx.features_by_symbol.get(symbol)
            if not feat_pack:
                return self._hold_decision("no_feature_pack", symbol)

            df = feat_pack.get("features")
            if not isinstance(df, pd.DataFrame) or df.empty:
                return self._hold_decision("empty_features", symbol)

            seq = self._build_sequence(df)
            if seq is None:
                return self._hold_decision("sequence_build_failed", symbol)

            prob_up = self._predict_prob_up(seq)
            if prob_up is None:
                return self._hold_decision("inference_failed", symbol)

            # -------------------------------------------------------
            # Logic Adaptation for v2 Architecture
            # -------------------------------------------------------
            
            # 1. Determine Action based on thresholds
            action = "hold"
            if prob_up >= self.config.long_threshold:
                action = "long"
            elif prob_up <= self.config.short_threshold:
                action = "short"

            # 2. Calculate Score (-1.0 to 1.0)
            # This allows the PortfolioOptimizer to scale size based on conviction
            score = (prob_up - 0.5) * 2.0

            # 3. Calculate Confidence (0.0 to 1.0)
            # Distance from neutral (0.5) implies confidence
            confidence = abs(score)

            return MinionDecision(
                minion_name=self.name,
                action=action,
                score=float(score),
                confidence=float(confidence),
                symbol=symbol,
                meta={
                    "prob_up": prob_up,
                    "raw_prediction": float(prob_up),
                    "thresholds": {
                        "long": self.config.long_threshold,
                        "short": self.config.short_threshold
                    }
                },
            )

        except Exception as exc:
            logger.exception("SequencerLSTM.decide failed: %s", exc)
            return self._hold_decision(f"error: {str(exc)}")

    def _hold_decision(self, reason: str, symbol: Optional[str] = None) -> MinionDecision:
        """Helper to return a neutral hold decision."""
        return MinionDecision(
            minion_name=self.name,
            action="hold",
            score=0.0,
            confidence=0.0,
            symbol=symbol,
            meta={"reason": reason}
        )

    # -------------------------------------------------------------------------
    # Sequence construction
    # -------------------------------------------------------------------------

    def _build_sequence(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """
        Build a [1, T, F] sequence tensor from the feature DataFrame.
        T = sequence_length
        F = number of features
        """
        seq_len = self.config.sequence_length

        if self.config.feature_columns:
            cols = self.config.feature_columns
        else:
            # Default: all numeric columns except obviously non‑feature ones
            exclude = {"timestamp"}
            cols = [
                c
                for c in df.columns
                if c not in exclude and np.issubdtype(df[c].dtype, np.number)
            ]

        if len(df) < seq_len:
            return None

        window = df[cols].tail(seq_len).astype(float)

        if self.config.use_zscore_norm:
            window = self._zscore_normalize(window)

        arr = window.to_numpy(dtype=np.float32)
        arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)

        # Shape: [1, T, F]
        return arr.reshape(1, seq_len, len(cols))

    def _zscore_normalize(self, window: pd.DataFrame) -> pd.DataFrame:
        """
        Apply simple z‑score normalization per feature over the window.
        """
        mean = window.mean()
        std = window.std().replace(0.0, np.nan)
        normed = (window - mean) / std
        return normed.fillna(0.0)

    # -------------------------------------------------------------------------
    # Model inference
    # -------------------------------------------------------------------------

    def _predict_prob_up(self, seq: np.ndarray) -> Optional[float]:
        """
        Run the sequence through the LSTM model and return probability of up move.
        We assume the model has a single input and a single output with
        shape [1, 1] or [1, 2] (probabilities for [down, up]).
        """
        if self._session is None:
            return None

        try:
            input_name = self._session.get_inputs()[0].name
            outputs = self._session.run(None, {input_name: seq})
            out = outputs[0]

            out = np.asarray(out)
            if out.ndim == 2 and out.shape[1] == 2:
                # [down, up]
                prob_up = float(out[0, 1])
            elif out.ndim == 2 and out.shape[1] == 1:
                prob_up = float(out[0, 0])
            else:
                # Fallback: treat scalar as prob_up
                prob_up = float(out.ravel()[0])

            prob_up = float(np.clip(prob_up, 0.0, 1.0))
            return prob_up

        except Exception as exc:
            logger.exception("SequencerLSTM inference failed: %s", exc)
            return None