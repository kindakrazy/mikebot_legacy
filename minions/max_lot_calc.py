# C:\mikebot\minions\risk\max_lot_calc.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Any, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class MaxLotCalcConfig:
    equity_fraction: float = 0.02
    margin_fraction: float = 0.20

    # Hard caps
    min_lot: float = 0.01
    max_lot: float = 50.0

    # Legacy Highstrike fields
    base_lot: float = 0.01
    max_lot_multiplier: float = 1.0
    score_sensitivity: float = 1.0

    # Exposure constraints
    max_symbol_exposure_fraction: float = 0.20
    max_total_exposure_fraction: float = 0.50

    symbol_meta: Dict[str, Dict[str, Any]] = None
    def __post_init__(self) -> None:
        if self.symbol_meta is None:
            self.symbol_meta = {}


# ---------------------------------------------------------------------------
# Max lot calculator
# ---------------------------------------------------------------------------

class MaxLotCalculator:
    """
    Computes the maximum allowable lot size for a symbol given:
      - account equity
      - free margin
      - symbol metadata
      - existing exposure
      - guardrail constraints

    This is the cleaned, explicit successor to HighStrike's max_lot_calc.py.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        clean = {k: v for k, v in config.items() if not k.startswith("_")}
        self.config = MaxLotCalcConfig(**clean)
    # ----------------------------------------------------------------------
    # Public API
    # ----------------------------------------------------------------------

    def compute(
        self,
        symbol: str,
        account_state: Dict[str, Any],
        open_positions: List[Dict[str, Any]],
    ) -> float:
        """
        Compute the maximum lot size allowed for a new order on `symbol`.
        """
        equity = float(account_state.get("equity", 0.0))
        free_margin = float(account_state.get("free_margin", equity))

        if equity <= 0:
            return self.config.min_lot

        # 1. Equity-based limit
        eq_limit = self._equity_based_limit(equity, symbol)

        # 2. Margin-based limit
        margin_limit = self._margin_based_limit(free_margin, symbol)

        # 3. Exposure-based limit
        exposure_limit = self._exposure_based_limit(symbol, equity, open_positions)

        # Combine limits
        lot = min(eq_limit, margin_limit, exposure_limit, self.config.max_lot)
        lot = float(np.clip(lot, self.config.min_lot, self.config.max_lot))

        return lot

    # ----------------------------------------------------------------------
    # Equity-based sizing
    # ----------------------------------------------------------------------

    def _equity_based_limit(self, equity: float, symbol: str) -> float:
        """
        Lot size based on fraction of equity.
        """
        meta = self.config.symbol_meta.get(symbol, {})
        contract_size = float(meta.get("contract_size", 100000))  # default FX contract

        # 2% of equity → convert to lots
        notional = equity * self.config.equity_fraction
        lot = notional / contract_size
        return max(lot, self.config.min_lot)

    # ----------------------------------------------------------------------
    # Margin-based sizing
    # ----------------------------------------------------------------------

    def _margin_based_limit(self, free_margin: float, symbol: str) -> float:
        """
        Lot size based on available free margin.
        """
        meta = self.config.symbol_meta.get(symbol, {})
        margin_per_lot = float(meta.get("margin_per_lot", 1000.0))

        if margin_per_lot <= 0:
            return self.config.max_lot

        usable_margin = free_margin * self.config.margin_fraction
        lot = usable_margin / margin_per_lot
        return max(lot, self.config.min_lot)

    # ----------------------------------------------------------------------
    # Exposure-based sizing
    # ----------------------------------------------------------------------

    def _exposure_based_limit(
        self,
        symbol: str,
        equity: float,
        open_positions: List[Dict[str, Any]],
    ) -> float:
        """
        Reduce lot size if symbol or total exposure is already high.
        """
        symbol_exp = 0.0
        total_exp = 0.0

        for pos in open_positions:
            sym = pos.get("symbol")
            lots = float(pos.get("lots", 0.0))
            price = float(pos.get("price", 0.0))
            if price <= 0:
                continue

            notional = lots * price
            total_exp += abs(notional)

            if sym == symbol:
                symbol_exp += abs(notional)

        # Symbol exposure limit
        sym_limit = (equity * self.config.max_symbol_exposure_fraction) - symbol_exp
        sym_limit_lots = sym_limit / max(self._last_price(symbol, open_positions), 1e-9)

        # Total exposure limit
        tot_limit = (equity * self.config.max_total_exposure_fraction) - total_exp
        tot_limit_lots = tot_limit / max(self._last_price(symbol, open_positions), 1e-9)

        limit = min(sym_limit_lots, tot_limit_lots)
        if limit <= 0:
            return self.config.min_lot

        return limit

    # ----------------------------------------------------------------------
    # Helpers
    # ----------------------------------------------------------------------

    def _last_price(self, symbol: str, open_positions: List[Dict[str, Any]]) -> float:
        """
        Extract last known price for symbol from open positions.
        """
        for pos in open_positions:
            if pos.get("symbol") == symbol:
                price = float(pos.get("price", 0.0))
                if price > 0:
                    return price
        return 1.0  # fallback
