# C:\mikebot\minions\bayesian_calibrator.py

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Tuple, Iterable, Optional, Dict

import math
import statistics

import numpy as np


@dataclass
class CalibrationSample:
    """
    One observation of (raw_prob, outcome).

    outcome: 1 for success/win, 0 for failure/loss.
    """
    raw_prob: float
    outcome: int


@dataclass
class DriftWindow:
    """
    Tracks recent calibration quality and drift using Brier score.
    """
    maxlen: int = 500
    samples: List[CalibrationSample] = field(default_factory=list)

    def add(self, sample: CalibrationSample) -> None:
        self.samples.append(sample)
        if len(self.samples) > self.maxlen:
            self.samples.pop(0)

    def brier_score(self) -> Optional[float]:
        if not self.samples:
            return None
        errors = [(s.raw_prob - s.outcome) ** 2 for s in self.samples]
        return float(sum(errors) / len(errors))

    def drift_score(self, baseline: Optional[float] = None) -> Optional[float]:
        """
        Drift is measured as the excess Brier score over a baseline.
        If no baseline is provided, we use the median of historical scores
        (approximated by splitting the window in half).
        """
        current = self.brier_score()
        if current is None:
            return None

        if baseline is not None:
            return max(0.0, current - baseline)

        # approximate baseline from first half of window
        if len(self.samples) < 40:
            return 0.0

        mid = len(self.samples) // 2
        early = self.samples[:mid]
        late = self.samples[mid:]

        def _brier(xs: Iterable[CalibrationSample]) -> float:
            return sum((s.raw_prob - s.outcome) ** 2 for s in xs) / max(1, len(list(xs)))

        early_score = _brier(early)
        late_score = _brier(late)
        return max(0.0, late_score - early_score)


class BayesianCalibrator:
    """
    Bayesian probability calibrator with drift awareness and smoothing.

    This is the mikebot version of the HighstrikeSignals bayesian_calibrator
    minion, conceptually pulling behavior from:

      - plugins/bayesian_calibrator.py
      - modules/ai_smoothing.py
      - modules/drift_metrics.py

    Responsibilities:
      - Maintain a Bayesian posterior over "true" success probability.
      - Calibrate raw model probabilities toward the posterior.
      - Detect drift via Brier score and adapt smoothing strength.
    """

    def __init__(
        self,
        prior_alpha: float = 2.0,
        prior_beta: float = 2.0,
        min_samples: int = 50,
        max_window: int = 1000,
        max_drift_boost: float = 5.0,
        smoothing_strength: float = 0.5,
    ) -> None:
        """
        Args:
            prior_alpha: prior successes (Beta prior).
            prior_beta: prior failures (Beta prior).
            min_samples: minimum samples before trusting empirical calibration.
            max_window: maximum number of samples kept in the drift window.
            max_drift_boost: maximum factor by which drift can increase smoothing.
            smoothing_strength: base strength of smoothing toward posterior mean
                                (0 = no smoothing, 1 = full posterior mean).
        """
        self._base_alpha = float(prior_alpha)
        self._base_beta = float(prior_beta)
        self._alpha = float(prior_alpha)
        self._beta = float(prior_beta)

        self._min_samples = int(min_samples)
        self._drift_window = DriftWindow(maxlen=max_window)
        self._max_drift_boost = float(max_drift_boost)
        self._base_smoothing_strength = float(smoothing_strength)

        self._n_samples = 0
        self._history: List[CalibrationSample] = []

    # -------------------------------------------------------------------------
    # Core Bayesian machinery
    # -------------------------------------------------------------------------

    @property
    def posterior_mean(self) -> float:
        return self._alpha / (self._alpha + self._beta)

    @property
    def posterior_var(self) -> float:
        denom = (self._alpha + self._beta) ** 2 * (self._alpha + self._beta + 1.0)
        if denom <= 0:
            return 0.0
        return (self._alpha * self._beta) / denom

    def _update_posterior(self, outcome: int) -> None:
        if outcome not in (0, 1):
            raise ValueError(f"Outcome must be 0 or 1, got {outcome}")
        self._alpha += outcome
        self._beta += 1 - outcome

    # -------------------------------------------------------------------------
    # Drift-aware smoothing
    # -------------------------------------------------------------------------

    def _effective_smoothing_strength(self) -> float:
        """
        Increase smoothing when drift is high, so we lean more on the
        Bayesian posterior and less on raw model probabilities.
        """
        drift = self._drift_window.drift_score()
        if drift is None:
            return self._base_smoothing_strength

        # Map drift (0..1+) into a multiplier [1, 1+max_drift_boost]
        # using a soft saturation.
        # Typical Brier scores are in [0, 0.25] for decent models.
        scale = min(1.0, drift / 0.1)  # 0.1 is a "noticeable drift" threshold
        multiplier = 1.0 + scale * (self._max_drift_boost - 1.0)
        return max(0.0, min(1.0 * multiplier, 1.0))

    def _smooth_probability(self, raw_prob: float) -> float:
        """
        Smooth raw probability toward the posterior mean using a
        drift‑aware strength.
        """
        raw_prob = float(np.clip(raw_prob, 0.0, 1.0))
        posterior = self.posterior_mean
        strength = self._effective_smoothing_strength()

        # ai_smoothing‑style convex combination:
        #   p_calibrated = (1 - strength) * raw + strength * posterior
        return (1.0 - strength) * raw_prob + strength * posterior

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def observe(self, raw_prob: float, outcome: int) -> None:
        """
        Record an observation and update posterior + drift metrics.

        Args:
            raw_prob: model‑predicted probability before calibration.
            outcome: 1 for success/win, 0 for failure/loss.
        """
        sample = CalibrationSample(raw_prob=float(raw_prob), outcome=int(outcome))
        self._history.append(sample)
        self._n_samples += 1

        self._drift_window.add(sample)
        self._update_posterior(sample.outcome)

    def calibrate(self, raw_prob: float) -> float:
        """
        Calibrate a single probability.

        Before we have enough samples, we lean more heavily on the prior.
        As data accumulates, we trust the posterior more, and drift can
        further increase smoothing.
        """
        raw_prob = float(np.clip(raw_prob, 0.0, 1.0))

        if self._n_samples < self._min_samples:
            # Early phase: strong pull toward prior mean.
            prior_mean = self._base_alpha / (self._base_alpha + self._base_beta)
            # Weight based on how many samples we have.
            w = self._n_samples / max(1.0, float(self._min_samples))
            # w=0 → pure prior, w=1 → normal smoothing regime
            base = (1.0 - w) * prior_mean + w * raw_prob
            return float(np.clip(base, 0.0, 1.0))

        return float(np.clip(self._smooth_probability(raw_prob), 0.0, 1.0))

    def calibrate_batch(self, probs: Iterable[float]) -> List[float]:
        """
        Calibrate a batch of probabilities.
        """
        return [self.calibrate(p) for p in probs]

    # -------------------------------------------------------------------------
    # Introspection / export
    # -------------------------------------------------------------------------

    def summary(self) -> Dict[str, float]:
        """
        Return a compact summary of the calibrator state for telemetry.
        """
        brier = self._drift_window.brier_score()
        drift = self._drift_window.drift_score()
        return {
            "n_samples": float(self._n_samples),
            "posterior_mean": float(self.posterior_mean),
            "posterior_var": float(self.posterior_var),
            "brier_score": float(brier) if brier is not None else float("nan"),
            "drift_score": float(drift) if drift is not None else float("nan"),
            "smoothing_strength": float(self._effective_smoothing_strength()),
        }

    def reset(self) -> None:
        """
        Reset to the original prior and clear history.
        """
        self._alpha = self._base_alpha
        self._beta = self._base_beta
        self._n_samples = 0
        self._history.clear()
        self._drift_window.samples.clear()
