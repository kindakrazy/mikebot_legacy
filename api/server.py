from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict

import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from mikebot.live.orchestrator.main import build_orchestrator, ROOT

log = logging.getLogger(__name__)

app = FastAPI(title="Mikebot Orchestrator API", version="1.0")

# Allow cockpit UI to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Build orchestrator once
_orchestrator = build_orchestrator(ROOT)


# -------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------

def to_jsonable(obj):
    """Recursively convert DataFrames, Series, and nested structures into JSON‑serializable forms."""
    if isinstance(obj, pd.DataFrame):
        return obj.to_dict(orient="records")
    if isinstance(obj, pd.Series):
        return obj.to_list()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_jsonable(v) for v in obj]
    return obj


# -------------------------------------------------------------------------
# Lifecycle
# -------------------------------------------------------------------------

@app.on_event("startup")
def on_startup() -> None:
    log.info("API server startup: starting orchestrator background loop")
    _orchestrator.start_background_loop()


@app.on_event("shutdown")
def on_shutdown() -> None:
    log.info("API server shutdown: stopping orchestrator")
    _orchestrator.stop()


# -------------------------------------------------------------------------
# Core API
# -------------------------------------------------------------------------

@app.get("/api/state")
def get_state() -> Dict[str, Any]:
    try:
        return to_jsonable(_orchestrator.export_state())
    except Exception as exc:
        log.exception("Failed to export orchestrator state: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to export state")


@app.post("/api/command")
def post_command(cmd: str) -> Dict[str, Any]:
    cmd = cmd.lower().strip()
    if cmd == "start":
        _orchestrator.start_background_loop()
        return {"status": "ok", "command": "start"}
    elif cmd == "stop":
        _orchestrator.stop()
        return {"status": "ok", "command": "stop"}
    elif cmd == "restart":
        _orchestrator.stop()
        _orchestrator.start_background_loop()
        return {"status": "ok", "command": "restart"}
    else:
        raise HTTPException(status_code=400, detail=f"Unknown command: {cmd}")


# -------------------------------------------------------------------------
# Cockpit UI Endpoints
# -------------------------------------------------------------------------

@app.get("/system/state")
def system_state() -> Dict[str, Any]:
    try:
        return to_jsonable(_orchestrator.export_state())
    except Exception as exc:
        log.exception("Failed to export system state: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to export system state")


@app.get("/strategy/state")
def strategy_state() -> Dict[str, Any]:
    try:
        return to_jsonable({
            "features": _orchestrator._latest_features,
            "iteration": _orchestrator.iteration,
            "timestamp": (
                _orchestrator._last_timestamp.isoformat()
                if _orchestrator._last_timestamp
                else None
            ),
        })
    except Exception as exc:
        log.exception("Failed to export strategy state: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to export strategy state")


# -------------------------------------------------------------------------
# Deep Orchestrator Introspection
# -------------------------------------------------------------------------

@app.get("/minions/state")
def minions_state() -> Dict[str, Any]:
    return to_jsonable({
        name: m.snapshot()
        for name, m in _orchestrator.minions.items()
    })


@app.get("/survivability/state")
def survivability_state() -> Dict[str, Any]:
    return to_jsonable(_orchestrator._last_survivability_state)


@app.get("/regime/state")
def regime_state() -> Dict[str, Any]:
    return to_jsonable(_orchestrator.current_regime or {})


@app.get("/personality/state")
def personality_state() -> Dict[str, Any]:
    return to_jsonable(_orchestrator.personality_manager.snapshot())


@app.get("/orders/state")
def orders_state() -> Dict[str, Any]:
    return to_jsonable(_orchestrator.order_router.snapshot())


@app.get("/account/state")
def account_state() -> Dict[str, Any]:
    if _orchestrator.account_manager:
        return to_jsonable(_orchestrator.account_manager.snapshot())
    return {}


@app.get("/bridge/health")
def bridge_health() -> Dict[str, Any]:
    data_server = _orchestrator.market_server
    exec_bridge = _orchestrator.order_router.execution_bridge

    # Determine primary symbol safely
    primary = getattr(_orchestrator.config, "primary_symbol", None)
    if not primary:
        symbols = _orchestrator.config.symbols or {}
        primary = next(iter(symbols.keys()), None)

    # Execution bridge health
    execution_alive = exec_bridge.execution_alive(primary) if primary else False

    # Market‑data health
    alive_map = {}
    if data_server:
        for (sym, tf), info in data_server._alive.items():
            try:
                alive_map[f"{sym}_{tf}"] = data_server.market_data_alive(sym, tf)
            except Exception:
                alive_map[f"{sym}_{tf}"] = False

    return {
        "execution": execution_alive,
        "market_data": alive_map,
    }


@app.get("/symbols")
def symbols() -> Dict[str, Any]:
    return to_jsonable(_orchestrator.config.symbols or {})


@app.get("/features/{symbol}")
def features_for_symbol(symbol: str) -> Dict[str, Any]:
    return to_jsonable(_orchestrator._latest_features.get(symbol, {}))


@app.get("/candles/{symbol}")
def candles_for_symbol(symbol: str) -> Any:
    snapshot = _orchestrator.candle_engine.get_latest_snapshot()
    return to_jsonable(snapshot.get(symbol, {}))


@app.get("/fusion/state")
def fusion_state() -> Dict[str, Any]:
    fusion = _orchestrator.fusion
    if hasattr(fusion, "snapshot"):
        return to_jsonable(fusion.snapshot())
    return {}


@app.get("/strategy/signals/{symbol}")
def strategy_signals(symbol: str) -> Dict[str, Any]:
    feats = _orchestrator._latest_features.get(symbol, {})
    return to_jsonable(feats.get("strategies", feats))


# -------------------------------------------------------------------------
# Root
# -------------------------------------------------------------------------

@app.get("/")
def root() -> Dict[str, Any]:
    return {"status": "ok", "message": "Mikebot API is running"}