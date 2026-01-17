from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from mikebot.core.candle_engine import CandleEngine, SymbolRegistry
from mikebot.core.feature_builder import FeatureBuilder, FeatureConfig
from mikebot.live.services.telemetry import TelemetryService

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path("C:/mikebot")


class AdapterFactories:
    """
    Standardized factory methods to build system components.
    Project Root is locked to C:\\mikebot.
    """

    # ------------------------------------------------------------------
    # Core Engines
    # ------------------------------------------------------------------
    @staticmethod
    def candle_engine(live_cfg: Any) -> CandleEngine:
        """Build CandleEngine from symbols.json with BOM safety."""
        root = PROJECT_ROOT
        symbols_path = root / "config" / "symbols.json"

        if not symbols_path.exists():
            symbols_path = Path.cwd() / "config" / "symbols.json"

        if not symbols_path.exists():
            raise FileNotFoundError(f"CRITICAL: symbols.json missing at {symbols_path}")

        try:
            symbol_registry = SymbolRegistry.from_file(symbols_path)
        except UnicodeDecodeError:
            with open(symbols_path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            with open(symbols_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=4)
            symbol_registry = SymbolRegistry.from_file(symbols_path)

        max_buf = int(getattr(live_cfg, "max_candles_per_symbol", 10000))
        return CandleEngine(symbol_registry, max_buffer_len=max_buf)

    @staticmethod
    def feature_builder(live_cfg: Any) -> FeatureBuilder:
        f_cfg = getattr(live_cfg, "feature_config", FeatureConfig())
        return FeatureBuilder(f_cfg)

    # ------------------------------------------------------------------
    # Orchestrator
    # ------------------------------------------------------------------
    @staticmethod
    def orchestrator(live_cfg: Any, candle_engine: Any) -> Any:
        """
        Builds the Orchestrator by injecting all required dependencies.
        """
        # Local import to avoid circulars with minions and services
        from mikebot.live.orchestrator.main import Orchestrator

        telemetry = AdapterFactories.telemetry(live_cfg)
        router = AdapterFactories.order_router(live_cfg, telemetry)
        survivability = AdapterFactories.survivability(live_cfg)

        return Orchestrator(
            config=live_cfg,
            candle_engine=candle_engine,
            order_router=router,
            telemetry=telemetry,
            feature_builder=AdapterFactories.feature_builder(live_cfg),
            minion_registry={},  # Filled by higher-level wiring
            personality_manager=AdapterFactories.personality(live_cfg),
            survivability_guard=survivability,
            portfolio_optimizer=AdapterFactories.portfolio(live_cfg, survivability),
            guardrails=AdapterFactories.guardrails(live_cfg, survivability),
        )

    # ------------------------------------------------------------------
    # Live Services
    # ------------------------------------------------------------------
    @staticmethod
    def telemetry(live_cfg: Any) -> TelemetryService:
        """Build TelemetryService with required path injection."""
        root = PROJECT_ROOT
        log_dir = root / "logs"

        if not hasattr(live_cfg, "jsonl_path"):
            live_cfg.jsonl_path = log_dir / "telemetry" / "trades.jsonl"
        if not hasattr(live_cfg, "system_report_path"):
            live_cfg.system_report_path = log_dir / "reports" / "system_status.json"
        if not hasattr(live_cfg, "performance_log_path"):
            live_cfg.performance_log_path = log_dir / "performance" / "metrics.csv"
        if not hasattr(live_cfg, "max_jsonl_size_mb"):
            live_cfg.max_jsonl_size_mb = 50

        return TelemetryService(config=live_cfg)

    @staticmethod
    def order_router(live_cfg: Any, telemetry: TelemetryService) -> Any:
        """
        Build OrderRouter with survivability attached to config.

        Import is local to avoid circular dependency with minions_base
        (which imports AdapterFactories).
        """
        from mikebot.live.services.order_router import OrderRouter
        from mikebot.minions.survivability import SurvivabilityGuard, SurvivabilityConfig
        from mikebot.minions.guardrails import GuardrailsConfig

        if not hasattr(live_cfg, "order_log_path"):
            live_cfg.order_log_path = PROJECT_ROOT / "logs" / "orders.json"
        if not hasattr(live_cfg, "queue_size"):
            live_cfg.queue_size = 100

        sc = getattr(live_cfg, "survivability_config", SurvivabilityConfig())
        live_cfg.survivability = SurvivabilityGuard(sc)

        if not hasattr(live_cfg, "guardrails"):
            live_cfg.guardrails = getattr(
                live_cfg, "guardrails_config", GuardrailsConfig()
            )

        built = OrderRouter.from_config(live_cfg, telemetry)
        return OrderRouter(config=built.config, telemetry=telemetry)

    # ------------------------------------------------------------------
    # Decision Layer / Minions
    # ------------------------------------------------------------------
    @staticmethod
    def rf_minion(live_cfg: Any) -> Any:
        from pathlib import Path
        from mikebot.minions.rf_minion import RFMinion, RFMinionConfig

        cfg = getattr(live_cfg, "rf_config", RFMinionConfig())
        if isinstance(cfg.model_path, str):
            cfg.model_path = Path(cfg.model_path)
        if not cfg.model_path.is_absolute():
            cfg.model_path = PROJECT_ROOT / cfg.model_path
        return RFMinion(cfg)

    @staticmethod
    def xgb_minion(live_cfg: Any) -> Any:
        from pathlib import Path
        from mikebot.minions.xgb_predictor import (
            XGBPredictorMinion,
            XGBMinionConfig,
        )

        cfg = getattr(live_cfg, "xgb_config", XGBMinionConfig())
        if isinstance(cfg.model_path, str):
            cfg.model_path = Path(cfg.model_path)
        if not cfg.model_path.is_absolute():
            cfg.model_path = PROJECT_ROOT / cfg.model_path
        return XGBPredictorMinion(cfg)

    @staticmethod
    def sequencer_lstm(live_cfg: Any) -> Any:
        from pathlib import Path
        from mikebot.minions.sequencer_lstm import (
            SequencerLSTM,
            SequencerLSTMConfig,
        )

        cfg = getattr(live_cfg, "lstm_config", SequencerLSTMConfig())
        if isinstance(cfg.model_path, str):
            cfg.model_path = Path(cfg.model_path)
        if not cfg.model_path.is_absolute():
            cfg.model_path = PROJECT_ROOT / cfg.model_path
        return SequencerLSTM(cfg)

    @staticmethod
    def personality(live_cfg: Any) -> Any:
        from mikebot.minions.personality import (
            PersonalityManager,
            PersonalityProfile,
        )

        if hasattr(live_cfg, "personalities") and hasattr(live_cfg, "default"):
            personalities = {
                key: PersonalityProfile(**pdata)
                for key, pdata in live_cfg.personalities.items()
            }
            return PersonalityManager(
                personalities=personalities, default=live_cfg.default
            )

        p_data = getattr(live_cfg, "personality", None)
        if isinstance(p_data, dict):
            if "name" in p_data:
                profile = PersonalityProfile(**p_data)
                return PersonalityManager(
                    personalities={profile.name: profile}, default=profile.name
                )
            elif "personalities" in p_data:
                p_map = p_data["personalities"]
                d_key = p_data.get("default", "neutral")
                profiles = {k: PersonalityProfile(**v) for k, v in p_map.items()}
                return PersonalityManager(personalities=profiles, default=d_key)

        return PersonalityManager(default="neutral")

    @staticmethod
    def survivability(live_cfg: Any) -> Any:
        from mikebot.minions.survivability import (
            SurvivabilityGuard,
            SurvivabilityConfig,
        )

        if hasattr(live_cfg, "survivability"):
            return live_cfg.survivability
        sc = getattr(live_cfg, "survivability_config", SurvivabilityConfig())
        return SurvivabilityGuard(sc)

    @staticmethod
    def portfolio(live_cfg: Any, survivability: Any) -> Any:
        from mikebot.minions.portfolio import (
            PortfolioOptimizer,
            PortfolioOptimizerConfig,
        )

        from mikebot.minions.order_guard import OrderGuard, OrderGuardConfig

        poc = getattr(live_cfg, "portfolio_config", PortfolioOptimizerConfig())

        ctrl = getattr(live_cfg, "control", {})
        og_data = ctrl.get("order_guard", {}) if isinstance(ctrl, dict) else {}
        og_cfg = OrderGuardConfig(
            min_lot=og_data.get("min_lot", 0.01),
            max_lot=og_data.get("max_lot", 10.0),
            max_slippage_pct=og_data.get("max_slippage_pct", 0.02),
        )
        order_guard = OrderGuard(og_cfg, survivability)

        return PortfolioOptimizer(
            config=poc,
            survivability=survivability,
            max_lot_calc=order_guard,
        )

    @staticmethod
    def guardrails(live_cfg: Any, survivability: Any) -> Any:
        from mikebot.minions.guardrails import Guardrails, GuardrailsConfig

        gc = getattr(live_cfg, "guardrails_config", GuardrailsConfig())
        return Guardrails(gc, survivability=survivability)

    @staticmethod
    def order_guard(live_cfg: Any, survivability: Any) -> Any:
        from mikebot.minions.order_guard import OrderGuard, OrderGuardConfig

        ctrl = getattr(live_cfg, "control", {})
        og_data = ctrl.get("order_guard", {}) if isinstance(ctrl, dict) else {}
        cfg = OrderGuardConfig(
            min_lot=og_data.get("min_lot", 0.01),
            max_lot=og_data.get("max_lot", 10.0),
            max_slippage_pct=og_data.get("max_slippage_pct", 0.02),
        )
        return OrderGuard(cfg, survivability)

    @staticmethod
    def neural_decision_layer(live_cfg: Any) -> Any:
        from mikebot.minions.neural_decision_layer import (
            NeuralDecisionLayer,
            NeuralDecisionConfig,
        )

        ndc = getattr(live_cfg, "neural_config", NeuralDecisionConfig())
        return NeuralDecisionLayer(ndc)

    @staticmethod
    def regime_switcher(live_cfg: Any) -> Any:
        from mikebot.minions.regime_switcher import (
            RegimeSwitcher,
            RegimeSwitcherConfig,
        )

        rsc = getattr(live_cfg, "regime_config", RegimeSwitcherConfig())
        return RegimeSwitcher(rsc)

    @staticmethod
    def knowledge_graph(live_cfg: Any) -> Any:
        from mikebot.minions.knowledge_graph import KnowledgeGraph

        global_cfg = getattr(live_cfg, "global_config", {})
        kg_cfg = global_cfg.get("knowledge_graph", {}) if isinstance(global_cfg, dict) else {}
        return KnowledgeGraph.from_config(kg_cfg)

    # ------------------------------------------------------------------
    # Training & Models
    # ------------------------------------------------------------------
    @staticmethod
    def train_pipeline(live_cfg: Any) -> Any:
        from mikebot.core.train_pipeline import TrainPipeline

        feature_builder = AdapterFactories.feature_builder(live_cfg)
        registry_path = PROJECT_ROOT / "models" / "ModelRegistry.json"

        return TrainPipeline(
            feature_builder=feature_builder,
            registry_path=registry_path,
        )