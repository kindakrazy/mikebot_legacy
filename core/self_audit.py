# mikebot/core/self_audit.py

from __future__ import annotations

from typing import Dict, Any, Optional, List, Tuple

import numpy as np
import pandas as pd


class SelfAudit:
    """
    Model introspection and audit utilities.

    This class computes:
    - global error profiles
    - regime-specific performance
    - strategy-specific performance
    - basic failure mode summaries
    - high-level training recommendations
    - drift / retrain signals for LearnerService

    It operates on pandas DataFrames and simple dict structures so it can
    be used both with:
        - ExperienceStore outputs (load_all / load_recent)
        - TrainPipeline / MetaTrainer experiment results
    """

    # ------------------------------------------------------------------
    # TOP‑LEVEL AUDIT ENTRYPOINT (required by LearnerService)
    # ------------------------------------------------------------------

    def run_audit(
        self,
        symbol: str,
        timeframe: str,
        store,
        drift_threshold: float = 0.15,
        recent_rows: int = 2000,
    ) -> Dict[str, Any]:
        """
        Unified audit entrypoint used by LearnerService.

        Parameters
        ----------
        symbol : str
        timeframe : str
        store : ExperienceStore or compatible object
            Must provide load_recent(symbol, timeframe, n_rows)
        drift_threshold : float
            Threshold above which drift is considered significant.
        recent_rows : int
            Number of recent samples to load for drift evaluation.

        Returns
        -------
        Dict[str, Any] with keys:
            - drift_score : float
            - needs_retrain : bool
            - n_samples : int
            - global_error_profile : dict
            - regime_performance : dict
            - strategy_performance : dict
        """
        if store is None:
            raise ValueError("SelfAudit.run_audit requires an ExperienceStore instance")

        # Load recent experience
        data = store.load_recent(symbol, timeframe, n_rows=recent_rows)

        labels = data.get("labels")
        predictions = data.get("predictions")
        regimes = data.get("regimes")
        strategies = data.get("strategies", {})

        # Compute global error profile
        global_error = self.compute_error_profile(labels, predictions)
        drift_score = float(global_error.get("mean_abs_error") or 0.0)

        # Compute regime and strategy performance
        regime_perf = self.evaluate_regime_performance(regimes, labels, predictions)
        strategy_perf = self.evaluate_strategy_performance(strategies, labels)

        needs_retrain = drift_score > drift_threshold

        return {
            "drift_score": drift_score,
            "needs_retrain": needs_retrain,
            "n_samples": global_error.get("n_samples", 0),
            "global_error_profile": global_error,
            "regime_performance": regime_perf,
            "strategy_performance": strategy_perf,
        }

    # ------------------------------------------------------------------
    # GLOBAL ERROR PROFILE
    # ------------------------------------------------------------------

    def compute_error_profile(
        self,
        labels: pd.DataFrame,
        predictions: pd.DataFrame,
    ) -> Dict[str, Any]:
        """
        Compute a global error profile using labels and predictions.

        Expects both DataFrames to share the same index and at least one
        numeric column each. For now, we use the first numeric column in
        each as the target/prediction.
        """
        y, p = _extract_single_numeric_pair(labels, predictions)
        if y is None or p is None or len(y) == 0:
            return {
                "n_samples": 0,
                "accuracy": np.nan,
                "mean_error": np.nan,
                "mean_abs_error": np.nan,
                "rmse": np.nan,
                "directional_hit_rate": np.nan,
            }

        # Align indices
        common_idx = y.index.intersection(p.index)
        y = y.loc[common_idx]
        p = p.loc[common_idx]

        n = len(y)
        if n == 0:
            return {
                "n_samples": 0,
                "accuracy": np.nan,
                "mean_error": np.nan,
                "mean_abs_error": np.nan,
                "rmse": np.nan,
                "directional_hit_rate": np.nan,
            }

        error = y - p
        abs_error = error.abs()

        # Directional correctness
        correct_direction = np.sign(y) == np.sign(p)
        directional_hit_rate = correct_direction.mean()

        # Binary accuracy (simple sign-based classifier)
        y_binary = (y > 0).astype(int)
        p_binary = (p > 0).astype(int)
        accuracy = (y_binary == p_binary).mean()

        rmse = np.sqrt((error ** 2).mean())

        return {
            "n_samples": int(n),
            "accuracy": float(accuracy),
            "mean_error": float(error.mean()),
            "mean_abs_error": float(abs_error.mean()),
            "rmse": float(rmse),
            "directional_hit_rate": float(directional_hit_rate),
        }

    # ------------------------------------------------------------------
    # REGIME PERFORMANCE
    # ------------------------------------------------------------------

    def evaluate_regime_performance(
        self,
        regimes: Optional[pd.DataFrame],
        labels: pd.DataFrame,
        predictions: pd.DataFrame,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute performance metrics per regime_label.
        """
        if regimes is None or regimes.empty:
            return {}

        y, p = _extract_single_numeric_pair(labels, predictions)
        if y is None or p is None or len(y) == 0:
            return {}

        # Align indices
        common_idx = y.index.intersection(p.index).intersection(regimes.index)
        y = y.loc[common_idx]
        p = p.loc[common_idx]
        regimes = regimes.loc[common_idx]

        if "regime_label" not in regimes.columns:
            return {}

        error = (y - p).abs()
        correct_direction = (np.sign(y) == np.sign(p)).astype(float)
        y_binary = (y > 0).astype(int)
        p_binary = (p > 0).astype(int)
        correct_binary = (y_binary == p_binary).astype(float)

        results: Dict[str, Dict[str, Any]] = {}

        for regime_label, idx in regimes.groupby("regime_label").groups.items():
            idx = list(idx)
            if not idx:
                continue

            n = len(idx)
            results[str(regime_label)] = {
                "n_samples": int(n),
                "accuracy": float(correct_binary.loc[idx].mean()),
                "mean_abs_error": float(error.loc[idx].mean()),
                "directional_hit_rate": float(correct_direction.loc[idx].mean()),
            }

        return results

    # ------------------------------------------------------------------
    # STRATEGY PERFORMANCE
    # ------------------------------------------------------------------

    def evaluate_strategy_performance(
        self,
        strategies: Dict[str, pd.DataFrame],
        labels: pd.DataFrame,
    ) -> Dict[str, Dict[str, Any]]:
        """
        Compute performance per strategy.

        For each strategy DataFrame, we expect at least a 'signal' column.
        Optionally, there may be an 'outcome' column.
        """
        y, _ = _extract_single_numeric_pair(labels, labels)
        if y is None or len(y) == 0:
            return {}

        results: Dict[str, Dict[str, Any]] = {}

        for name, df in strategies.items():
            if df is None or df.empty:
                continue

            common_idx = df.index.intersection(y.index)
            if len(common_idx) == 0:
                continue

            df = df.loc[common_idx]
            y_local = y.loc[common_idx]

            if "signal" not in df.columns:
                continue

            signal = df["signal"].astype(bool)
            n_signals = int(signal.sum())

            if n_signals == 0:
                results[name] = {
                    "n_signals": 0,
                    "mean_outcome": float("nan"),
                    "hit_rate": float("nan"),
                }
                continue

            if "outcome" in df.columns:
                outcomes = df.loc[signal, "outcome"].astype(float)
                mean_outcome = float(outcomes.mean()) if len(outcomes) > 0 else float("nan")
                hit_rate = mean_outcome
            else:
                y_signals = y_local.loc[signal]
                if len(y_signals) == 0:
                    mean_outcome = float("nan")
                    hit_rate = float("nan")
                else:
                    outcome = (y_signals > 0).astype(float)
                    mean_outcome = float(outcome.mean())
                    hit_rate = mean_outcome

            results[name] = {
                "n_signals": n_signals,
                "mean_outcome": mean_outcome,
                "hit_rate": hit_rate,
            }

        return results

    # ------------------------------------------------------------------
    # FAILURE MODES
    # ------------------------------------------------------------------

    def identify_failure_modes(
        self,
        labels: pd.DataFrame,
        predictions: pd.DataFrame,
        regimes: Optional[pd.DataFrame] = None,
        strategies: Optional[Dict[str, pd.DataFrame]] = None,
        top_k: int = 3,
    ) -> Dict[str, Any]:
        """
        Produce a coarse failure mode summary.
        """
        y, p = _extract_single_numeric_pair(labels, predictions)
        result: Dict[str, Any] = {"worst_regimes": [], "worst_strategies": []}

        # Regimes
        if regimes is not None and not regimes.empty:
            regime_perf = self.evaluate_regime_performance(regimes, labels, predictions)
            items: List[Tuple[str, Dict[str, Any]]] = list(regime_perf.items())
            items.sort(key=lambda kv: kv[1].get("directional_hit_rate", 1.0))

            for regime_label, stats in items[:top_k]:
                result["worst_regimes"].append(
                    {
                        "regime_label": regime_label,
                        "mean_abs_error": stats.get("mean_abs_error"),
                        "directional_hit_rate": stats.get("directional_hit_rate"),
                        "n_samples": stats.get("n_samples"),
                    }
                )

        # Strategies
        if strategies:
            strat_perf = self.evaluate_strategy_performance(strategies, labels)
            items = list(strat_perf.items())
            items.sort(key=lambda kv: kv[1].get("hit_rate", 1.0))

            for strat_name, stats in items[:top_k]:
                result["worst_strategies"].append(
                    {
                        "strategy_name": strat_name,
                        "hit_rate": stats.get("hit_rate"),
                        "n_signals": stats.get("n_signals"),
                    }
                )

        return result

    # ------------------------------------------------------------------
    # TRAINING RECOMMENDATIONS
    # ------------------------------------------------------------------

    def recommend_training_adjustments(
        self,
        global_error_profile: Dict[str, Any],
        regime_performance: Dict[str, Dict[str, Any]],
        strategy_performance: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Produce high-level, deterministic recommendations for MetaTrainer.
        """
        rec: Dict[str, Any] = {
            "use_error_focused_experiment": False,
            "use_recency_weighted_experiment": False,
            "regime_weights": {},
            "strategy_weights": {},
            "notes": [],
        }

        n_samples = global_error_profile.get("n_samples") or 0
        mean_abs_error = global_error_profile.get("mean_abs_error")
        accuracy = global_error_profile.get("accuracy")

        # Error-focused
        if n_samples > 0 and mean_abs_error is not None:
            if mean_abs_error > 0.01:
                rec["use_error_focused_experiment"] = True
                rec["notes"].append(
                    f"mean_abs_error={mean_abs_error:.4f} suggests trying error-focused weighting."
                )

        # Recency-weighted
        if n_samples > 0 and accuracy is not None and accuracy < 0.55:
            rec["use_recency_weighted_experiment"] = True
            rec["notes"].append(
                f"accuracy={accuracy:.3f} suggests trying recency-weighted experiments."
            )

        # Regime weights
        if regime_performance:
            regime_scores: Dict[str, float] = {}
            for regime_label, stats in regime_performance.items():
                dh = stats.get("directional_hit_rate")
                if dh is None or np.isnan(dh):
                    continue
                regime_scores[regime_label] = max(0.0, 1.0 - float(dh))

            total = sum(regime_scores.values())
            if total > 0:
                for regime_label, score in regime_scores.items():
                    rec["regime_weights"][regime_label] = score / total
                rec["notes"].append(
                    "Regime weights computed inversely to directional_hit_rate."
                )

        # Strategy weights
        if strategy_performance:
            strat_scores: Dict[str, float] = {}
            for name, stats in strategy_performance.items():
                hit = stats.get("hit_rate")
                n_sig = stats.get("n_signals", 0)
                if hit is None or np.isnan(hit) or n_sig <= 0:
                    continue
                strat_scores[name] = float(hit) * float(np.log1p(n_sig))

            total = sum(strat_scores.values())
            if total > 0:
                for name, score in strat_scores.items():
                    rec["strategy_weights"][name] = score / total
                rec["notes"].append(
                    "Strategy weights computed from hit_rate and signal volume."
                )

        return rec


# ----------------------------------------------------------------------
# INTERNAL UTILITIES
# ----------------------------------------------------------------------

def _extract_single_numeric_pair(
    labels: Optional[pd.DataFrame],
    predictions: Optional[pd.DataFrame],
) -> Tuple[Optional[pd.Series], Optional[pd.Series]]:
    """
    Extract the first numeric column from each DataFrame as a pair.
    """
    if labels is None or predictions is None:
        return None, None
    if labels.empty or predictions.empty:
        return None, None

    y_cols = labels.select_dtypes(include=["number"]).columns
    p_cols = predictions.select_dtypes(include=["number"]).columns

    if len(y_cols) == 0 or len(p_cols) == 0:
        return None, None

    y = labels[y_cols[0]]
    p = predictions[p_cols[0]]
    return y, p