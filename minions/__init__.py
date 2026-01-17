# C:\mikebot\minions\__init__.py

"""
mikebot.minions package exports and simple registry wiring.

This file exposes the primary minion classes and provides a
conservative MinionRegistry.from_config implementation that reads
LiveConfig and instantiates minions by name when configured.
"""

from .minions_base import (
    Minion,
    MinionContext,
    MinionDecision,
    MinionRegistry,
    MinionHealthMonitor,
)
from .rf_minion import RFMinion, RFMinionConfig
from .xgb_predictor import XGBPredictorMinion, XGBMinionConfig
from .sequencer_lstm import SequencerLSTM, SequencerLSTMConfig
from .clustering_scout import ClusteringScout, ClusteringConfig
from .bayesian_calibrator import BayesianCalibrator
from .hedge_minion import HedgeMinion, HedgeMinionConfig
from .guardrails import Guardrails, GuardrailsConfig
from .order_guard import OrderGuard, OrderGuardConfig
from .max_lot_calc import MaxLotCalculator, MaxLotCalcConfig
from .portfolio import PortfolioOptimizer, PortfolioOptimizerConfig
from .personality import PersonalityManager, PersonalityProfile
from .survivability import SurvivabilityGuard, SurvivabilityConfig
from .neural_decision_layer import NeuralDecisionLayer, NeuralDecisionConfig
from .regime_switcher import RegimeSwitcher, RegimeSwitcherConfig
from .knowledge_graph import KnowledgeGraph

__all__ = [
    "Minion",
    "MinionContext",
    "MinionDecision",
    "MinionRegistry",
    "MinionHealthMonitor",
    "RFMinion",
    "XGBPredictorMinion",
    "SequencerLSTM",
    "ClusteringScout",
    "BayesianCalibrator",
    "HedgeMinion",
    "Guardrails",
    "OrderGuard",
    "MaxLotCalculator",
    "PortfolioOptimizer",
    "PersonalityManager",
    "SurvivabilityGuard",
    "NeuralDecisionLayer",
    "RegimeSwitcher",
    "KnowledgeGraph",
]


# Conservative MinionRegistry.from_config implementation
def _minion_registry_from_config(cls, live_cfg) -> MinionRegistry:
    registry = cls(minions={})  # <-- FIXED

    cfg = getattr(live_cfg, "ml_orchestrator", {}) or {}
    minion_cfg = cfg.get("minions", {}) or {}

    if not minion_cfg:
        weights = getattr(live_cfg, "minion_weights", {}) or {}
        for name in weights.keys():
            minion_cfg[name] = {"enabled": True}

    constructors = {
        "rf_minion": lambda c: RFMinion(RFMinionConfig(**(c or {}))),
        "xgb_predictor": lambda c: XGBPredictorMinion(XGBMinionConfig(**(c or {}))),
        "sequencer_lstm": lambda c: SequencerLSTM(SequencerLSTMConfig(**(c or {}))),
        "clustering_scout": lambda c: ClusteringScout(ClusteringConfig(**(c or {}))),
        "bayesian_calibrator": lambda c: BayesianCalibrator(**(c or {})),
        "hedge_minion": lambda c: HedgeMinion(HedgeMinionConfig(**(c or {}))),
    }

    for name, meta in minion_cfg.items():
        if not bool(meta.get("enabled", True)):
            continue

        ctor = constructors.get(name)
        if ctor is None:
            continue

        try:
            minion = ctor(meta.get("config", {}))
            registry.register(minion)
        except Exception:
            continue

    return registry


MinionRegistry.from_config = classmethod(_minion_registry_from_config)
