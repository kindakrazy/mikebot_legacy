#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Module: mikebot.live.orchestrator.main

from __future__ import annotations
import asyncio
import sys
import logging
import signal
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Any, Iterable

from mikebot.live.orchestrator.config import LiveConfig
from mikebot.adapters.integration_adapters import AdapterFactories
from mikebot.adapters.bridge_server import MikebotBridgeServer
from mikebot.core.candle_engine import CandleEngine
from mikebot.core.feature_builder import FeatureBuilder
from mikebot.strategies.strategy_registry import StrategyRegistry
from mikebot.strategies.engine_v4 import StrategyEngineV4
from mikebot.core.regime_detector import RegimeDetector, set_global_regime_detector
from mikebot.strategies.fusion_v4 import StrategyFusionV4

from mikebot.live.services.order_router import OrderRouter
from mikebot.live.services.telemetry import TelemetryService
from mikebot.live.services.learner import LearnerService
from mikebot.live.services.mt4_market_data_server import Mt4MarketDataServer

from mikebot.minions import (
    MinionRegistry,
    MinionContext,
    MinionDecision,
    MinionHealthMonitor,
)
from mikebot.minions.survivability import SurvivabilityGuard
from mikebot.minions.personality import PersonalityManager
from mikebot.minions.multi_agent import blended_vote
from mikebot.minions.portfolio import PortfolioOptimizer
from mikebot.minions.guardrails import Guardrails
from mikebot.minions.neural_decision_layer import NeuralDecisionLayer
from mikebot.minions.regime_switcher import RegimeSwitcher
from mikebot.minions.knowledge_graph import KnowledgeGraph

from mikebot.runtime.context import runtime_context

log = logging.getLogger(__name__)

ROOT = Path("C:/mikebot")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# -------------------------------------------------------------------------
# Orchestrator state
# -------------------------------------------------------------------------


@dataclass
class OrchestratorState:
    """High-level live session state."""
    session_id: str
    started_at: datetime
    last_tick_at: Optional[datetime] = None
    last_decision_at: Optional[datetime] = None
    running: bool = True
    loop_iteration: int = 0
    errors: List[str] = field(default_factory=list)


# -------------------------------------------------------------------------
# Orchestrator
# -------------------------------------------------------------------------


class Orchestrator:
    """
    Live trading orchestrator.

    Responsibilities:
      - Drive the main live loop
      - Build features and strategy signals
      - Coordinate minions, survivability, guardrails, and portfolio
      - Route final orders
      - Emit telemetry
      - Maintain regime and personality context
    """

    def __init__(
        self,
        config: LiveConfig,
        candle_engine: CandleEngine,
        feature_builder: FeatureBuilder,
        minion_registry: MinionRegistry,
        order_router: OrderRouter,
        telemetry: TelemetryService,
        personality_manager: PersonalityManager,
        survivability_guard: SurvivabilityGuard,
        portfolio_optimizer: PortfolioOptimizer,
        guardrails: Guardrails,
        health_monitor: MinionHealthMonitor,
        strategy_engine: StrategyEngineV4,
        learner: Optional[LearnerService] = None,
        neural_layer: Optional[NeuralDecisionLayer] = None,
        regime_switcher: Optional[RegimeSwitcher] = None,
        knowledge_graph: Optional[KnowledgeGraph] = None,
        market_server: Optional[Mt4MarketDataServer] = None,
    ) -> None:

        self.config = config
        self.candle_engine = candle_engine
        self.feature_builder = feature_builder
        self.minion_registry = minion_registry
        self.order_router = order_router
        self.telemetry = telemetry
        self.personality_manager = personality_manager
        self.survivability_guard = survivability_guard
        self.portfolio_optimizer = portfolio_optimizer
        self.guardrails = guardrails
        self.health_monitor = health_monitor
        self.strategy_engine = strategy_engine
        self.regime_detector = RegimeDetector(window=14)
        set_global_regime_detector(self.regime_detector)
        self.account_manager = None

        # Multi-strategy fusion layer (currently instantiated, ready for use)
        self.fusion = StrategyFusionV4()

        self.learner = learner
        self.neural_layer = neural_layer
        self.regime_switcher = regime_switcher
        self.knowledge_graph = knowledge_graph
        self.market_server = market_server

        # ------------------------------------------------------------
        # Market Data Bridge Server (port 50010)
        # ------------------------------------------------------------
        self.market_data_bridge = MikebotBridgeServer(host="0.0.0.0", port=50010)

        self._md_loop = asyncio.new_event_loop()

        def _run_md_bridge():
            asyncio.set_event_loop(self._md_loop)
            self._md_loop.run_until_complete(self.market_data_bridge.start())

        self._md_thread = threading.Thread(target=_run_md_bridge, daemon=True)
        self._md_thread.start()

        # Attach market-data bridge to orchestrator (if such a hook exists)
        if hasattr(self, "attach_market_data_bridge"):
            self.attach_market_data_bridge(self.market_data_bridge)

        # ------------------------------------------------------------
        # Auto‑subscribe to ticks and candles for all configured symbols
        # ------------------------------------------------------------
        try:
            symbols_cfg: Dict[str, Dict[str, Any]] = self.config.symbols or {}

            for sym, meta in symbols_cfg.items():
                tf = (
                    meta.get("timeframe")
                    or meta.get("tf")
                    or meta.get("primary_timeframe")
                    or meta.get("candle_timeframe")
                    or meta.get("resolution")
                    or "M5"
                )
                bars = meta.get("history_bars") or self.config.max_candles_per_symbol

                # Subscribe to ticks
                self.market_data_bridge.subscribe_ticks(
                    symbol=sym,
                    callback=self.on_tick,
                )

                # Subscribe to candles
                self.market_data_bridge.subscribe_candles(
                    symbol=sym,
                    timeframe=tf,
                    callback=self.on_candle,
                )

                # Request initial history (async)
                asyncio.run_coroutine_threadsafe(
                    self.market_data_bridge.request_history(
                        symbol=sym,
                        timeframe=tf,
                        bars=bars,
                    ),
                    self._md_loop,
                )

        except Exception as exc:
            log.exception("Failed to auto‑subscribe symbols/timeframes: %s", exc)

        # ------------------------------------------------------------
        # Execution Bridge Server (port 50020)
        # ------------------------------------------------------------
        self.execution_bridge = MikebotBridgeServer(host="0.0.0.0", port=50020)

        self._exec_loop = asyncio.new_event_loop()

        def _run_exec_bridge():
            asyncio.set_event_loop(self._exec_loop)
            self._exec_loop.run_until_complete(self.execution_bridge.start())

        self._exec_thread = threading.Thread(target=_run_exec_bridge, daemon=True)
        self._exec_thread.start()

        # Attach execution bridge to order router
        if hasattr(self.order_router, "attach_execution_bridge"):
            self.order_router.attach_execution_bridge(self.execution_bridge)

        # Route execution events into the order router
        self.execution_bridge.on_execution_event(
            lambda event: self.order_router.handle_execution_event(event)
        )

        # NEW: execution‑sync callback
        self.execution_bridge.on_sync_event(self._on_execution_sync)

        # ------------------------------------------------------------
        # Orchestrator state
        # ------------------------------------------------------------
        self.current_regime: Optional[Dict[str, Any]] = None

        self.state = OrchestratorState(
            session_id=self._make_session_id(),
            started_at=datetime.now(timezone.utc),
        )

        self._stop_event = threading.Event()
        self._loop_thread: Optional[threading.Thread] = None

        # Snapshots for API/export_state
        self._last_minion_decisions: List[MinionDecision] = []
        self._last_orders: List[Any] = []
        self._last_survivability_state: Dict[str, Any] = {}
        self._running = False
        self.minions = {}
        self._install_signal_handlers()
        self.iteration = 0
        self._last_timestamp = None
        self._latest_features = {}

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------

    def _install_signal_handlers(self) -> None:
        def _handle(sig, frame) -> None:
            log.info("Received signal %s, stopping orchestrator", sig)
            self.stop()

        try:
            signal.signal(signal.SIGINT, _handle)
            signal.signal(signal.SIGTERM, _handle)
        except ValueError:
            log.debug("Signal handlers not installed (unsupported environment)")

    def _on_execution_sync(self, event: dict) -> None:
        """
        Called when the execution bridge reports startup sync complete
        for (symbol, timeframe).
        """
        symbol = event["symbol"]
        timeframe = event["timeframe"]

        log.info(f"Execution sync complete for {symbol} {timeframe}")

        # Tell the order router
        if hasattr(self.order_router, "mark_execution_synced"):
            self.order_router.mark_execution_synced(symbol, timeframe)

        # Optional: tell strategy engine (safe if method exists)
        if hasattr(self.strategy_engine, "mark_execution_synced"):
            self.strategy_engine.mark_execution_synced(symbol, timeframe)


    def _make_session_id(self) -> str:
        now = datetime.now(timezone.utc).isoformat()
        return f"live-{now}"

    def stop(self) -> None:
        self.state.running = False
        self._stop_event.set()
        if self.market_server is not None:
            try:
                self.market_server.stop()
            except Exception:
                log.exception("Failed to stop Mt4MarketDataServer cleanly")

    def start_background_loop(self) -> None:
        """
        Start the main run_forever loop in a background thread.

        Safe to call from an API server or other host process.
        """
        if self._loop_thread and self._loop_thread.is_alive():
            return

        self.state.running = True
        self._stop_event.clear()

        def _runner():
            try:
                self.run_forever()
            except Exception:
                log.exception("Orchestrator background loop crashed")

        self._loop_thread = threading.Thread(target=_runner, daemon=True)
        self._loop_thread.start()

    # ---------------------------------------------------------------------
    # Main loop
    # ---------------------------------------------------------------------

    def run_forever(self) -> None:
        log.info("Starting orchestrator session %s", self.state.session_id)
        self.telemetry.emit_session_start(self.state.session_id, self.state.started_at)

        loop_interval = self.config.loop_interval_seconds
        while self.state.running and not self._stop_event.is_set():
            loop_start = time.time()
            try:
                self.state.loop_iteration += 1
                self._run_iteration()
            except Exception as exc:
                msg = f"orchestrator iteration error: {exc!r}"
                log.exception(msg)
                self.state.errors.append(msg)
                self.telemetry.emit_error("orchestrator_iteration_error", msg)

            elapsed = time.time() - loop_start
            sleep_for = max(0.0, loop_interval - elapsed)
            if sleep_for > 0:
                time.sleep(sleep_for)

        self.telemetry.emit_session_end(self.state.session_id, datetime.now(timezone.utc))
        log.info("Orchestrator session %s stopped", self.state.session_id)

    # ---------------------------------------------------------------------
    # Single iteration
    # ---------------------------------------------------------------------

    def _run_iteration(self) -> None:
        now = datetime.now(timezone.utc)
        self.state.last_tick_at = now

        candles = self._get_latest_candles()
        if not candles:
            log.debug("No new candles; skipping iteration %d", self.state.loop_iteration)
            return

        features_by_symbol = self._build_features(candles)
        if not features_by_symbol:
            log.debug("No features built; skipping iteration %d", self.state.loop_iteration)
            return

        self._latest_features = features_by_symbol

        minion_ctx = self._build_minion_context(now, features_by_symbol)
        minion_decisions = self._collect_minion_decisions(minion_ctx)
        self._last_minion_decisions = list(minion_decisions)

        self._update_regime_and_personality(minion_decisions)

        blended = blended_vote(
            decisions=minion_decisions,
            weights=self.config.minion_weights,
            personality=self.personality_manager.get_active(),
        )

        if self.neural_layer is not None:
            try:
                blended = self.neural_layer.process(blended)
            except Exception as exc:
                log.exception("NeuralDecisionLayer processing failed: %s", exc)
                self.telemetry.emit_error("neural_layer_error", str(exc))

        surv_state = self.survivability_guard.check_survivability(
            account_state=minion_ctx.account_state,
            open_positions=minion_ctx.open_positions,
            volatility_series=minion_ctx.volatility_series,
            loop_iteration=self.state.loop_iteration,
        )
        self._last_survivability_state = dict(surv_state)

        hedge_orders: List[Any] = []
        for d in minion_decisions:
            if d.minion_name == "hedge_minion":
                orders = getattr(d, "orders", None)
                if orders:
                    hedge_orders.extend(list(orders))

        if surv_state.get("safe_mode"):
            log.warning("Orchestrator: SAFE MODE active → skipping new orders")
            self.telemetry.emit_self_heal("safe_mode_active: skipping orders")
            final_orders: List[Any] = []
        else:
            final_orders = self.portfolio_optimizer.build_orders(
                blended=blended,
                ctx=minion_ctx,
                hedge_orders=hedge_orders,
            )

            try:
                last_prices = self._extract_last_prices(features_by_symbol)
                final_orders = self.guardrails.filter_orders(
                    orders=final_orders,
                    account_state=minion_ctx.account_state,
                    open_positions=minion_ctx.open_positions,
                    last_prices=last_prices,
                    loop_iteration=self.state.loop_iteration,
                    volatility_series=self._extract_volatility_series(features_by_symbol),
                    session_tag=self._session_tag_for_now(now),
                    bridge_alive=self._mt4_bridge_alive(),
                )
            except Exception as exc:
                log.exception("Guardrails filtering failed: %s", exc)
                self.telemetry.emit_error("guardrails_error", str(exc))
                final_orders = []

        self._last_orders = list(final_orders)

        self._route_orders(final_orders, minion_ctx)
        self._update_health_and_telemetry(minion_decisions, final_orders, now)
        self.state.last_decision_at = now
        self._last_timestamp = now
        self.iteration = self.state.loop_iteration
        self._running = self.state.running

    # ---------------------------------------------------------------------
    # Tick & Candle Callbacks (required by bridge subscriptions)
    # ---------------------------------------------------------------------

    def on_tick(self, symbol: str, tick: Dict[str, Any]) -> None:
        """
        Called by the market_data_bridge whenever a tick arrives.
        """
        try:
            self.candle_engine.on_tick(symbol, tick)
        except Exception as exc:
            log.exception("on_tick failed for %s: %s", symbol, exc)

    def on_candle(self, symbol: str, timeframe: str, candle: Dict[str, Any]) -> None:
        """
        Called by the market_data_bridge whenever a new candle arrives.
        """
        try:
            self.candle_engine.on_candle(symbol, timeframe, candle)
        except Exception as exc:
            log.exception("on_candle failed for %s/%s: %s", symbol, timeframe, exc)

    def _get_latest_candles(self) -> Dict[str, Any]:
        try:
            return self.candle_engine.get_latest_snapshot()
        except Exception as exc:
            msg = f"candle_engine.get_latest_snapshot failed: {exc!r}"
            log.exception(msg)
            self.telemetry.emit_error("candle_snapshot_error", msg)
            return {}

    def _build_features(self, candles: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build features per symbol using the V4 strategy engine.
        """
        try:
            strategy_signals_by_symbol: Dict[str, Any] = {}
            for sym, df in candles.items():
                try:
                    batch = self.strategy_engine.run_batch(df)
                    strat_frames = self.strategy_engine.as_feature_mapping(batch)
                    strategy_signals_by_symbol[sym] = strat_frames
                except Exception as exc:
                    log.exception("StrategyEngineV4 computation failed for %s: %s", sym, exc)
                    strategy_signals_by_symbol[sym] = {}

            return self.feature_builder.build_for_snapshot(
                candles_by_symbol=candles,
                strategies_by_symbol=strategy_signals_by_symbol,
            )
        except Exception as exc:
            msg = f"feature_builder.build_for_snapshot failed: {exc!r}"
            log.exception(msg)
            self.telemetry.emit_error("feature_build_error", msg)
            return {}

    def _build_minion_context(
        self,
        now: datetime,
        features_by_symbol: Dict[str, Any],
    ) -> MinionContext:

        account_state = self.order_router.get_account_state()
        open_positions = self.order_router.get_open_positions()

        # Determine primary symbol
        primary = getattr(self.config, "primary_symbol", None)
        if not primary and features_by_symbol:
            primary = next(iter(features_by_symbol.keys()))

        # Extract price + volatility
        last_prices = self._extract_last_prices(features_by_symbol)
        volatility_series = self._extract_volatility_series(features_by_symbol)

        # Build the full immutable context
        ctx = MinionContext(
            session_id=self.state.session_id,
            timestamp=now,
            loop_iteration=self.state.loop_iteration,

            features_by_symbol=features_by_symbol,
            last_prices=last_prices,
            volatility_series=volatility_series,

            account_state=account_state,
            open_positions=open_positions,

            personality=self.personality_manager.get_active(),
            primary_symbol=primary,

            knowledge_graph=self.knowledge_graph,
            regime=self.current_regime,
        )

        return ctx

    def _collect_minion_decisions(self, ctx: MinionContext) -> List[MinionDecision]:
        decisions: List[MinionDecision] = []
        # Use the explicit iterator on MinionRegistry
        for minion in self.minion_registry.iter_active_minions():
            try:
                d = minion.decide(ctx)
            except Exception as exc:
                log.exception("Minion %s failed: %s", getattr(minion, "name", "?"), exc)
                self.telemetry.emit_error("minion_error", str(exc))
                continue
            if d is None:
                continue
            if isinstance(d, Iterable) and not isinstance(d, (str, bytes)):
                decisions.extend(list(d))
            else:
                decisions.append(d)
        return decisions

    def _update_regime_and_personality(self, decisions) -> None:
        # --- 1. Compute regime using RegimeDetector ---
        try:
            primary = getattr(self.config, "primary_symbol", None)
            if not primary:
                latest = self.candle_engine.get_latest_snapshot()
                if latest:
                    primary = next(iter(latest.keys()))

            df = self.candle_engine.get_latest_snapshot().get(primary) if primary else None
            if df is not None and not df.empty:
                regime_info = self.regime_detector.detect(df)
                regime_str = regime_info["regime_id"]

                # Map string regime → numeric ID
                regime_map = {
                    "trend_bull": 1,
                    "trend_bear": 2,
                    "ranging_low_vol": 3,
                    "ranging_high_vol": 4,
                    "unstable": 5,
                }
                regime_id = regime_map.get(regime_str, 5)
                regime_score = float(regime_info["metrics"]["efficiency_ratio"])
                confidence = float(regime_info["metrics"]["efficiency_ratio"])

                self.current_regime = {
                    "regime_id": regime_id,
                    "regime_label": regime_str,
                    "score": regime_score,
                    "confidence": confidence,
                    "metrics": dict(regime_info.get("metrics", {})),
                }

                # Optional: update regime switcher / personality manager
                if self.regime_switcher is not None:
                    try:
                        self.regime_switcher.update(
                            regime_id=regime_id,
                            regime_score=regime_score,
                            confidence=confidence,
                        )
                    except Exception as exc:
                        log.exception("RegimeSwitcher update failed: %s", exc)
                        self.telemetry.emit_error("regime_switcher_error", str(exc))

                if self.personality_manager is not None:
                    try:
                        self.personality_manager.update_from_regime(
                            regime_str,
                            confidence,
                        )
                    except Exception as exc:
                        log.exception("PersonalityManager update_from_regime failed: %s", exc)
                        self.telemetry.emit_error("personality_update_error", str(exc))

        except Exception as exc:
            log.exception("PersonalityManager update failed: %s", exc)
            self.telemetry.emit_error("personality_update_error", str(exc))

    def _route_orders(self, orders: List[Any], ctx: MinionContext) -> None:
        for order in orders:
            try:
                self.order_router.route_order(order, ctx)
            except Exception as exc:
                log.exception("Order routing failed for %s: %s", order, exc)
                self.telemetry.emit_error("order_routing_error", str(exc))

    def _update_health_and_telemetry(
        self,
        decisions: List[MinionDecision],
        orders: List[Any],
        now: datetime,
    ) -> None:
        # --- Health monitor (V4 API: no .update()) ---
        try:
            for d in decisions:
                err = getattr(d, "error", None)
                if err:
                    self.health_monitor.record_failure(d.minion_name, err)
                else:
                    self.health_monitor.record_success(d.minion_name)
        except Exception as exc:
            log.exception("Health monitor update failed: %s", exc)
            self.telemetry.emit_error("health_monitor_error", str(exc))

        # --- Telemetry (V4 API: no .emit_iteration()) ---
        try:
            self.telemetry.emit_iteration_summary(
                session_id=self.state.session_id,
                iteration=self.state.loop_iteration,
                timestamp=now,
                minion_decisions=decisions,
                orders=orders,
                health=self.health_monitor.snapshot(),
            )
        except Exception as exc:
            log.exception("Telemetry emit_iteration_summary failed: %s", exc)
            self.telemetry.emit_error("telemetry_error", str(exc))

    def _extract_last_prices(self, features_by_symbol: Dict[str, Any]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for sym, feat in features_by_symbol.items():
            try:
                df = feat.get("core", None)
                if df is None or df.empty:
                    continue
                if "close" in df.columns:
                    out[sym] = float(df["close"].iloc[-1])
            except Exception:
                continue
        return out

    def _extract_volatility_series(self, features_by_symbol: Dict[str, Any]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for sym, feat in features_by_symbol.items():
            try:
                df = feat.get("core", None)
                if df is None or df.empty:
                    continue
                if "volatility" in df.columns:
                    out[sym] = df["volatility"]
            except Exception:
                continue
        return out

    def _session_tag_for_now(self, now: datetime) -> str:
        return now.strftime("%Y-%m-%d")

    def _mt4_bridge_alive(self) -> bool:
        """
        Require BOTH:
          - execution bridge alive
          - market‑data bridge alive

        Market‑data health is determined dynamically:
          - by symbol
          - across any timeframe the EA actually uses
        """
        # Determine primary symbol
        symbol = getattr(self.config, "primary_symbol", None)
        if not symbol:
            if isinstance(self.config.symbols, dict) and self.config.symbols:
                symbol = next(iter(self.config.symbols.keys()))
            else:
                return False

        # Execution bridge check (per symbol)
        exec_bridge = getattr(self.order_router, "execution_bridge", None)
        exec_ok = (
            exec_bridge.execution_alive(symbol)
            if exec_bridge and hasattr(exec_bridge, "execution_alive")
            else False
        )

        # Market‑data bridge check:
        # Prefer the Mt4MarketDataServer health (per symbol, per timeframe),
        # and accept ANY alive timeframe for this symbol.
        data_ok = False
        data_server = getattr(self, "market_server", None)

        if data_server and hasattr(data_server, "market_data_alive"):
            alive_map = getattr(data_server, "_alive", {}) or {}
            for (sym, tf) in alive_map.keys():
                if sym != symbol:
                    continue
                try:
                    if data_server.market_data_alive(sym, tf):
                        data_ok = True
                        break
                except Exception:
                    continue
        else:
            # Fallback: if no Mt4MarketDataServer health, try the bridge (legacy)
            data_bridge = getattr(self, "market_data_bridge", None)
            if data_bridge and hasattr(data_bridge, "market_data_alive"):
                # Try all configured timeframes for this symbol
                tfs: List[str] = []
                if isinstance(self.config.symbols, dict):
                    meta = self.config.symbols.get(symbol, {})
                    cfg_tf = (
                        meta.get("timeframe")
                        or meta.get("tf")
                        or meta.get("primary_timeframe")
                        or meta.get("candle_timeframe")
                        or meta.get("resolution")
                        or "M5"
                    )
                    tfs.append(cfg_tf)
                else:
                    tfs.append("M5")

                for tf in tfs:
                    try:
                        if data_bridge.market_data_alive(symbol, tf):
                            data_ok = True
                            break
                    except Exception:
                        continue

        return exec_ok and data_ok

    # ---------------------------------------------------------------------
    # API / snapshot export
    # ---------------------------------------------------------------------

    def export_state(self) -> Dict[str, Any]:
        """
        Return a JSON‑serializable snapshot of the orchestrator state
        for the cockpit UI.
        """
        return {
            "session": {
                "iteration": self.iteration,
                "timestamp": self._last_timestamp.isoformat() if self._last_timestamp else None,
                "running": self._running,
            },

            "regime": self.regime_switcher.snapshot() if self.regime_switcher else None,

            "personality": (
                self.personality_manager.snapshot()
                if self.personality_manager else None
            ),

            "minions": {
                name: m.snapshot()
                for name, m in self.minions.items()
            },

            "orders": self.order_router.snapshot() if self.order_router else [],

            "account": (
                self.account_manager.snapshot()
                if self.account_manager else None
            ),

            "features": self._latest_features or {},
        }


# -------------------------------------------------------------------------
# Bootstrap / entrypoint
# -------------------------------------------------------------------------


def build_orchestrator(root: Path) -> Orchestrator:
    # Live config
    live_cfg = LiveConfig.load_from_files(root)

    # Candle engine via AdapterFactories (symbols.json + BOM safety)
    candle_engine: CandleEngine = AdapterFactories.candle_engine(live_cfg)

    # MT4 market data server (live candles into CandleEngine)
    market_server = Mt4MarketDataServer(
        candle_engine=candle_engine,
        host="127.0.0.1",  # must match EA InpHost
        port=50010,        # must match EA InpPort
    )
    market_server.start()

    # Core services via AdapterFactories
    feature_builder: FeatureBuilder = AdapterFactories.feature_builder(live_cfg)
    telemetry: TelemetryService = AdapterFactories.telemetry(live_cfg)
    order_router: OrderRouter = AdapterFactories.order_router(live_cfg, telemetry)

    # Strategy engine v4
    strategy_registry = StrategyRegistry()
    strategy_engine = StrategyEngineV4(live_config=live_cfg, registry=strategy_registry)

    # Minions + decision layer
    minion_registry = MinionRegistry.from_config(live_cfg)

    personality_manager: PersonalityManager = AdapterFactories.personality(live_cfg)
    survivability_guard: SurvivabilityGuard = AdapterFactories.survivability(live_cfg)
    portfolio_optimizer: PortfolioOptimizer = AdapterFactories.portfolio(live_cfg, survivability_guard)
    guardrails: Guardrails = AdapterFactories.guardrails(live_cfg, survivability_guard)
    neural_layer: Optional[NeuralDecisionLayer] = AdapterFactories.neural_decision_layer(live_cfg)
    regime_switcher: Optional[RegimeSwitcher] = AdapterFactories.regime_switcher(live_cfg)
    knowledge_graph: Optional[KnowledgeGraph] = AdapterFactories.knowledge_graph(live_cfg)

    # Health monitor + learner (if you have factories for them)
    health_monitor = MinionHealthMonitor()
    learner: Optional[LearnerService] = None  # plug in when ready

    orch = Orchestrator(
        config=live_cfg,
        candle_engine=candle_engine,
        feature_builder=feature_builder,
        minion_registry=minion_registry,
        order_router=order_router,
        telemetry=telemetry,
        personality_manager=personality_manager,
        survivability_guard=survivability_guard,
        portfolio_optimizer=portfolio_optimizer,
        guardrails=guardrails,
        health_monitor=health_monitor,
        strategy_engine=strategy_engine,
        learner=learner,
        neural_layer=neural_layer,
        regime_switcher=regime_switcher,
        knowledge_graph=knowledge_graph,
        market_server=market_server,
    )

    # -----------------------------------------------------------------
    # Runtime context wiring (top-level components)
    # -----------------------------------------------------------------
    runtime_context["orchestrator"] = orch
    runtime_context["config"] = live_cfg
    runtime_context["candle_engine"] = candle_engine
    runtime_context["feature_builder"] = feature_builder
    runtime_context["minion_registry"] = minion_registry
    runtime_context["order_router"] = order_router
    runtime_context["telemetry"] = telemetry
    runtime_context["strategy_engine"] = strategy_engine
    runtime_context["personality_manager"] = personality_manager
    runtime_context["survivability_guard"] = survivability_guard
    runtime_context["portfolio_optimizer"] = portfolio_optimizer
    runtime_context["guardrails"] = guardrails
    runtime_context["neural_layer"] = neural_layer
    runtime_context["regime_switcher"] = regime_switcher
    runtime_context["knowledge_graph"] = knowledge_graph
    runtime_context["health_monitor"] = health_monitor
    runtime_context["learner"] = learner
    runtime_context["market_server"] = market_server

    return orch


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    root = ROOT
    orch = build_orchestrator(root)
    orch.run_forever()


if __name__ == "__main__":
    main()