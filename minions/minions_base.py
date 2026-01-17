from __future__ import annotations

import enum
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Protocol, Type

from mikebot.adapters.integration_adapters import AdapterFactories

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Order primitives
# ---------------------------------------------------------------------------

class OrderSide(str, enum.Enum):
    BUY = "BUY"
    SELL = "SELL"


@dataclass(frozen=True)
class OrderRequest:
    """
    Canonical order request object used by all minions and the order router.
    """
    symbol: str
    side: OrderSide
    lot_size: float
    price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    comment: Optional[str] = None

    def with_lot_size(self, lot: float) -> OrderRequest:
        return OrderRequest(
            symbol=self.symbol,
            side=self.side,
            lot_size=lot,
            price=self.price,
            stop_loss=self.stop_loss,
            take_profit=self.take_profit,
            comment=self.comment,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "side": self.side.value,
            "lot_size": self.lot_size,
            "price": self.price,
            "stop_loss": self.stop_loss,
            "take_profit": self.take_profit,
            "comment": self.comment,
        }


# ---------------------------------------------------------------------------
# Modern MinionContext + MinionDecision
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MinionContext:
    """
    Modern, explicit, typed context passed to all minions.

    Replaces the legacy dynamic attribute bag with a stable,
    predictable, validated structure.
    """

    # Session + iteration metadata
    session_id: str
    timestamp: datetime
    loop_iteration: int

    # Market data + features
    features_by_symbol: Dict[str, Any]
    last_prices: Dict[str, float]
    volatility_series: Optional[List[float]]

    # Trading state
    account_state: Any
    open_positions: Any

    # Behavior + personality
    personality: Any
    primary_symbol: Optional[str]

    # Optional knowledge systems
    knowledge_graph: Optional[Any] = None
    regime: Optional[Any] = None

    # Convenience helpers
    def feature_pack(self, symbol: str) -> Optional[Any]:
        return self.features_by_symbol.get(symbol)

    def price(self, symbol: str) -> Optional[float]:
        return self.last_prices.get(symbol)

    def volatility(self) -> Optional[List[float]]:
        return self.volatility_series


@dataclass(frozen=True)
class MinionDecision:
    """
    Modern, explicit, typed decision returned by every minion.
    """

    minion_name: str
    action: str               # "long", "short", "hold", "exit", etc.
    score: float              # normalized decision strength
    confidence: float         # model confidence
    meta: Dict[str, Any] = field(default_factory=dict)
    symbol: Optional[str] = None

    def is_hold(self) -> bool:
        return self.action == "hold"

    def is_entry(self) -> bool:
        return self.action in ("long", "short")

    def is_exit(self) -> bool:
        return self.action == "exit"

    def is_directional(self) -> bool:
        return self.action in ("long", "short")

    def describe(self) -> str:
        return (
            f"{self.minion_name}: action={self.action}, "
            f"score={self.score:.3f}, confidence={self.confidence:.3f}, "
            f"symbol={self.symbol}, meta={self.meta}"
        )


# ---------------------------------------------------------------------------
# Minion base protocol
# ---------------------------------------------------------------------------

class Minion(Protocol):
    """
    Base protocol for all mikebot minions.

    Every minion must implement:
        - name: str
        - decide(ctx: MinionContext) -> MinionDecision
    """

    name: str

    def decide(self, ctx: MinionContext) -> MinionDecision:
        ...


# ---------------------------------------------------------------------------
# Minion health
# ---------------------------------------------------------------------------

@dataclass
class MinionHealth:
    failures: int = 0
    successes: int = 0
    last_error: Optional[str] = None

    @property
    def failure_rate(self) -> float:
        total = self.failures + self.successes
        if total == 0:
            return 0.0
        return self.failures / total


class MinionHealthMonitor:
    """
    Tracks minion health across iterations.
    """

    def __init__(
        self,
        enabled: bool = True,
        window: int = 5,
        thresholds: Optional[Dict[str, float]] = None,
    ) -> None:
        self.enabled = enabled
        self.window = window
        self.thresholds = thresholds or {}
        self._health: Dict[str, MinionHealth] = {}

    def record_success(self, name: str) -> None:
        h = self._health.setdefault(name, MinionHealth())
        h.successes += 1

    def record_failure(self, name: str, error: str) -> None:
        h = self._health.setdefault(name, MinionHealth())
        h.failures += 1
        h.last_error = error

    def snapshot(self) -> Dict[str, Dict[str, Any]]:
        return {
            name: {
                "failures": h.failures,
                "successes": h.successes,
                "failure_rate": h.failure_rate,
                "last_error": h.last_error,
            }
            for name, h in self._health.items()
        }

    @classmethod
    def from_config(cls, live_cfg: Any) -> MinionHealthMonitor:
        cfg = getattr(live_cfg, "health_monitor", None)
        if cfg is None:
            raise ValueError("Config missing required 'health_monitor' section")

        if "enabled" not in cfg:
            raise ValueError("health_monitor missing required field: 'enabled'")

        window = cfg.get("window", 5)
        thresholds = cfg.get("thresholds", {})

        if not isinstance(thresholds, dict):
            raise ValueError("'thresholds' must be a dict mapping minion names to floats")

        return cls(
            enabled=cfg["enabled"],
            window=window,
            thresholds=thresholds,
        )


# ---------------------------------------------------------------------------
# Minion registry
# ---------------------------------------------------------------------------

class MinionRegistry:
    """
    Modern, explicit, dependency‑injected registry of minions.

    Responsibilities:
      - Load minions from config
      - Construct minions with explicit dependencies (via AdapterFactories when possible)
      - Validate that each minion implements the Minion interface
      - Enforce that each minion returns a MinionDecision
      - Provide iteration over active minions
    """

    def __init__(self, minions: List[Minion]) -> None:
        self._minions = minions

    @classmethod
    def from_config(cls, config: Any) -> MinionRegistry:
        """
        Load minions from config.minions, where each entry defines:
            - class: import path
            - enabled: bool
            - params: dict (only used for non‑factory minions)
        """
        specs = getattr(config, "minions", [])
        minions: List[Minion] = []

        if not isinstance(specs, list):
            raise ValueError("config.minions must be a list of minion specifications")

        factories = AdapterFactories()

        factory_map: Dict[str, str] = {
            "mikebot.minions.rf_minion.RFMinion": "rf_minion",
            "mikebot.minions.xgb_predictor.XGBPredictorMinion": "xgb_minion",
            "mikebot.minions.sequencer_lstm.SequencerLSTM": "sequencer_lstm",
        }

        for entry in specs:
            if not isinstance(entry, dict):
                raise ValueError("Each minion entry must be a dict")

            if not entry.get("enabled", True):
                continue

            cls_path = entry.get("class")
            if not cls_path:
                raise ValueError("Minion entry missing 'class' field")

            params = entry.get("params", {})
            if not isinstance(params, dict):
                raise ValueError("'params' must be a dict of constructor arguments")

            factory_name = factory_map.get(cls_path)
            if factory_name is not None and hasattr(factories, factory_name):
                factory_fn = getattr(factories, factory_name)
                if not callable(factory_fn):
                    raise TypeError(
                        f"Factory '{factory_name}' for {cls_path} is not callable"
                    )
                instance = factory_fn(config)
            else:
                minion_cls = cls._import_class(cls_path)
                if not hasattr(minion_cls, "decide"):
                    raise TypeError(
                        f"{cls_path} is not a valid Minion (missing decide())"
                    )
                instance = minion_cls(**params)

            minions.append(instance)

        return cls(minions)

    @staticmethod
    def _import_class(path: str) -> Type[Minion]:
        module_path, _, class_name = path.rpartition(".")
        if not module_path:
            raise ValueError(f"Invalid class path: {path}")

        module = __import__(module_path, fromlist=[class_name])
        cls_obj = getattr(module, class_name)
        return cls_obj

    def iter_active_minions(self) -> Iterable[Minion]:
        return iter(self._minions)

    @staticmethod
    def validate_decision(decision: Any, minion_name: str) -> MinionDecision:
        iftrunicated

    def get_all(self) -> Iterable[Minion]:
        """
        Backwards‑compatible alias for iterating active minions.
        """
        return self.iter_active_minions()
