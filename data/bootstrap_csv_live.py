#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Bootstrap the live orchestrator using CSV candles instead of MT4.

This allows the entire live system (orchestrator, minions, guardrails,
portfolio, telemetry) to run end‑to‑end with no MT4 bridge.
"""

from pathlib import Path
from datetime import datetime, timezone
import time
import logging

from mikebot.live.orchestrator.config import LiveConfig
from mikebot.adapters.integration_adapters import AdapterFactories
from mikebot.core.candle_engine import CandleEngine, SymbolRegistry
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1])) # Adds C:\mikebot to the path

def load_symbol_registry(config: LiveConfig) -> SymbolRegistry:
    """
    Build a SymbolRegistry from config.symbols (already loaded from symbols.json).
    """
    return SymbolRegistry(config.symbols)


def bootstrap_csv_engine(config: LiveConfig, csv_path: Path, symbol: str, timeframe: str = "M1") -> CandleEngine:
    """
    Create a CandleEngine and ingest a CSV file as historical candles.
    """
    registry = load_symbol_registry(config)
    engine = CandleEngine(symbol_registry=registry, max_buffer_len=config.max_candles_per_symbol)

    print(f"[BOOTSTRAP] Loading CSV candles from {csv_path}")
    engine.ingest_csv(csv_path, symbol=symbol, timeframe=timeframe)

    print(f"[BOOTSTRAP] Loaded {len(engine.get_df(symbol, timeframe))} candles for {symbol}")
    return engine


def run_orchestrator_for_n_iterations(orchestrator, n: int = 10, delay: float = 1.0):
    """
    Run the orchestrator for a fixed number of iterations instead of forever.
    """
    print(f"[BOOTSTRAP] Running orchestrator for {n} iterations...")

    for i in range(n):
        try:
            orchestrator.state.loop_iteration += 1
            orchestrator._run_iteration()
        except Exception as exc:
            print(f"[BOOTSTRAP] Iteration {i} failed: {exc!r}")
        time.sleep(delay)

    print("[BOOTSTRAP] Done.")


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Load config
    config = LiveConfig.load_from_files()

    # Choose a CSV file and symbol
    csv_path = Path("mikebot/data/BTCUSD.csv")   # <-- change to your CSV
    symbol = "BTCUSD"
    timeframe = "M15"

    # Build CandleEngine with CSV data
    candle_engine = bootstrap_csv_engine(config, csv_path, symbol, timeframe)

    # Build orchestrator with dependency factories
    factories = AdapterFactories()
    orchestrator = factories.orchestrator(config, candle_engine)

    # Run a few iterations
    run_orchestrator_for_n_iterations(orchestrator, n=20, delay=0.5)


if __name__ == "__main__":
    main()