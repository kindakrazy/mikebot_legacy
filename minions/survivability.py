from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

import numpy as np

from .minions_base import MinionDecision, MinionHealthMonitor

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Survivability configuration
# ---------------------------------------------------------------------------

@dataclass
class SurvivabilityConfig:
    """
    Survivability configuration (Mikebot v3.5).

    Distilled from:
      - survivability.py
      - minion_health.py
      - strategy_evolution.py
      - personality_manager.py
    """

    # Exposure limits
    max_symbol_exposure: float = 0.20      # 20% of equity per symbol
    max_total_exposure: float = 0.50       # 50% of equity total

    # Drawdown protection
    max_drawdown_pct: float = 0.25         # 25% drawdown triggers safe mode
    safe_mode_recovery_pct: float = 0.05   # exit safe mode after 5% recovery

    # Minion health thresholds
    max_failure_rate: float = 0.20         # quarantine minion if >20% failures
    min_iterations_before_health_check: int = 50

    # Volatility protection
    volatility_window: int = 50
    volatility_threshold: float = 2.5      # z-score threshold

    # Behavior toggles
    enable_exposure_limits: bool = True
    enable_drawdown_protection: bool = True
    enable_minion_quarantine: bool = True
    enable_volatility_protection: bool = True


# ---------------------------------------------------------------------------
# Survivability engine
# ---------------------------------------------------------------------------

class SurvivabilityGuard:
    """
    The survivability engine.

    Responsibilities:
      - Monitor exposure, drawdown, volatility, and minion health
      - Trigger safe mode when conditions are dangerous
      - Quarantine failing minions
      - Provide survivability metadata to orchestrator
    """

    def __init__(self, config: SurvivabilityConfig) -> None:
        self.config = config
        self.health_monitor = MinionHealthMonitor()

        self.safe_mode_active: bool = False
        self.safe_mode_trigger_equity: Optional[float] = None

    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def update_minion_health(self, decision: MinionDecision, error: Optional[str]) -> None:
        """
        Update health stats for a minion based on execution success/failure.
        """
        if error:
            self.health_monitor.record_failure(decision.minion_name, error)
        else:
            self.health_monitor.record_success(decision.minion_name)

    def check_survivability(
        self,
        account_state: Dict[str, Any],
        open_positions: List[Dict[str, Any]],
        volatility_series: Optional[np.ndarray],
        loop_iteration: int,
    ) -> Dict[str, Any]:
        """
        Evaluate survivability conditions and return a survivability state dict.
        """
        state: Dict[str, Any] = {
            "safe_mode": self.safe_mode_active,
            "quarantined_minions": self._quarantined_minions(loop_iteration),
            "exposure_ok": True,
            "drawdown_ok": True,
            "volatility_ok": True,
        }

        # Exposure limits
        if self.config.enable_exposure_limits:
            state["exposure_ok"] = self._check_exposure_limits(account_state, open_positions)

        # Drawdown protection
        if self.config.enable_drawdown_protection:
            state["drawdown_ok"] = self._check_drawdown(account_state)

        # Volatility protection
        if self.config.enable_volatility_protection:
            state["volatility_ok"] = self._check_volatility(volatility_series)

        # Safe mode logic
        self._update_safe_mode(state, account_state)

        return state

    # ----------------------------------------------------------------------
    # Exposure limits
    # ----------------------------------------------------------------------

    def _check_exposure_limits(
        self,
        account_state: Dict[str, Any],
        open_positions: List[Dict[str, Any]],
    ) -> bool:
        equity = float(account_state.get("equity", 0.0))
        if equity <= 0:
            return False

        symbol_exposure: Dict[str, float] = {}
        total_exposure = 0.0

        for pos in open_positions:
            symbol = pos.get("symbol")
            lots = float(pos.get("lots", 0.0))
            price = float(pos.get("price", 0.0))
            side = pos.get("side", "BUY").upper()

            if not symbol or price <= 0:
                continue

            direction = 1.0 if side in ("BUY", "LONG") else -1.0
            notional = lots * price * direction

            symbol_exposure[symbol] = symbol_exposure.get(symbol, 0.0) + notional
            total_exposure += abs(notional)

        # Symbol-level exposure
        for sym, exp in symbol_exposure.items():
            if abs(exp) / equity > self.config.max_symbol_exposure:
                logger.warning("Survivability: symbol exposure breach on %s", sym)
                return False

        # Total exposure
        if total_exposure / equity > self.config.max_total_exposure:
            logger.warning("Survivability: total exposure breach")
            return False

        return True

    # ----------------------------------------------------------------------
    # Drawdown protection
    # ----------------------------------------------------------------------

    def _check_drawdown(self, account_state: Dict[str, Any]) -> bool:
        equity = float(account_state.get("equity", 0.0))
        balance = float(account_state.get("balance", 0.0))

        if balance <= 0:
            return True

        dd = (balance - equity) / balance
        if dd >= self.config.max_drawdown_pct:
            logger.warning("Survivability: drawdown breach (%.2f%%)", dd * 100)
            return False

        return True

    # ----------------------------------------------------------------------
    # Volatility protection
    # ----------------------------------------------------------------------

    def _check_volatility(self, series: Optional[np.ndarray]) -> bool:
        if series is None or len(series) < self.config.volatility_window:
            return True

        window = series[-self.config.volatility_window:]
        z = (window[-1] - window.mean()) / (window.std() + 1e-9)

        if abs(z) > self.config.volatility_threshold:
            logger.warning("Survivability: volatility breach (z=%.2f)", z)
            return False

        return True

    # ----------------------------------------------------------------------
    # Safe mode logic
    # ----------------------------------------------------------------------

    def _update_safe_mode(self, state: Dict[str, Any], account_state: Dict[str, Any]) -> None:
        """
        Update safe mode based on drawdown and recovery.
        """
        equity = float(account_state.get("equity", 0.0))
        balance = float(account_state.get("balance", 0.0))

        if balance <= 0:
            return

        dd = (balance - equity) / balance

        # Enter safe mode if drawdown breach
        if self.config.enable_drawdown_protection and not state.get("drawdown_ok", True):
            if not self.safe_mode_active:
                self.safe_mode_active = True
                self.safe_mode_trigger_equity = equity
                logger.warning(
                    "Survivability: entering SAFE MODE (drawdown=%.2f%%)",
                    dd * 100,
                )
            return

        # Exit safe mode after recovery
        if self.safe_mode_active and self.safe_mode_trigger_equity is not None:
            recovery = (equity - self.safe_mode_trigger_equity) / max(
                self.safe_mode_trigger_equity, 1e-9
            )
            if recovery >= self.config.safe_mode_recovery_pct:
                logger.info(
                    "Survivability: exiting SAFE MODE (recovery=%.2f%%)",
                    recovery * 100,
                )
                self.safe_mode_active = False
                self.safe_mode_trigger_equity = None

    # ----------------------------------------------------------------------
    # Minion quarantine
    # ----------------------------------------------------------------------

    def _quarantined_minions(self, loop_iteration: int) -> List[str]:
        if not self.config.enable_minion_quarantine:
            return []

        if loop_iteration < self.config.min_iterations_before_health_check:
            return []

        quarantined: List[str] = []
        snapshot = self.health_monitor.snapshot()

        for name, stats in snapshot.items():
            if stats["failure_rate"] > self.config.max_failure_rate:
                quarantined.append(name)

        return quarantined