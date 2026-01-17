#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File: C:\mikebot\mikebot/live/orchestrator/config.py
# Module: mikebot.live.orchestrator.config

from __future__ import annotations

import json
try:
    import yaml
except ImportError:
    yaml = None
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional

import logging

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRATEGY_CONFIG_FILE = "strategy_patterns.json"

# ---------------------------------------------------------------------------
# Utility loaders
# ---------------------------------------------------------------------------

def _load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        log.warning("JSON config missing: %s", path)
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        log.error("Failed to load JSON %s: %s", path, exc)
        return {}


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        log.warning("YAML config missing: %s", path)
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception as exc:
        log.error("Failed to load YAML %s: %s", path, exc)
        return {}


def _deep_merge(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(a)
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out

# ---------------------------------------------------------------------------
# LiveConfig dataclass
# ---------------------------------------------------------------------------

@dataclass
class LiveConfig:
    loop_interval_seconds: float = 1.0
    max_candles_per_symbol: int = 5000

    symbols: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    control: Dict[str, Any] = field(default_factory=dict)
    switches: Dict[str, Any] = field(default_factory=dict)
    feeds: Dict[str, Any] = field(default_factory=dict)
    risk_caps: Dict[str, Any] = field(default_factory=dict)
    minion_weights: Dict[str, float] = field(default_factory=dict)
    personality: Dict[str, Any] = field(default_factory=dict)
    guardrails: Dict[str, Any] = field(default_factory=dict)
    ml_orchestrator: Dict[str, Any] = field(default_factory=dict)
    global_config: Dict[str, Any] = field(default_factory=dict)
    health_monitor: Dict[str, Any] = field(default_factory=dict)
    strategy_patterns: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    root: Path = Path(".")

    runtime_errors_path: Path = Path("logs") / "runtime_errors.jsonl"
    rotate_jsonl: bool = True
    max_jsonl_bytes: int = 10_000_000

    # -----------------------------------------------------------------------
    # Constructors
    # -----------------------------------------------------------------------

    @classmethod
    def load_from_files(cls, root: Optional[Path] = None) -> "LiveConfig":
        root = root or Path.cwd()
        cfg_dir = root / "mikebot" / "config"
        diag_dir = root / "mikebot" / "config"

        log.info("Loading LiveConfig from %s", cfg_dir)

        # Primary configs
        control = _load_json(cfg_dir / "control.json")
        switches = _load_json(cfg_dir / "switches.json")
        symbols = _load_json(cfg_dir / "symbols.json")
        feeds = _load_json(cfg_dir / "feeds.json")
        config_json = _load_json(cfg_dir / "config.json")
        global_config = _load_json(cfg_dir / "global_config.json")
        health_monitor = _load_json(cfg_dir / "health_monitor.json")

        # ML orchestrator
        ml_orchestrator: Dict[str, Any] = {}
        ml_orchestrator = _deep_merge(
            ml_orchestrator,
            _load_json(cfg_dir / "MLOrchestrator.json"),
        )

        # Guardrails
        guardrails = _load_json(cfg_dir / "guardrails.json")
        guardrails_schema = _load_json(cfg_dir / "guardrails_schema.json")

        # Strategy pattern configs (BearFlag, BullFlag, etc.)
        strategy_patterns = _load_json(cfg_dir / STRATEGY_CONFIG_FILE)
        strategy_defaults = strategy_patterns.get("__defaults__", {})
        for key, default_cfg in strategy_defaults.items():
            if key in strategy_patterns:
                strategy_patterns[key] = _deep_merge(default_cfg, strategy_patterns[key])

        # Diagnostics fallback configs
        control = _deep_merge(control, _load_json(diag_dir / "control.json"))
        switches = _deep_merge(switches, _load_json(diag_dir / "switches.json"))
        symbols = _deep_merge(symbols, _load_json(diag_dir / "symbols.json"))
        feeds = _deep_merge(feeds, _load_json(diag_dir / "feeds.json"))
        global_config = _deep_merge(global_config, _load_json(diag_dir / "global_config.json"))
        health_monitor = _deep_merge(health_monitor, _load_json(diag_dir / "health_monitor.json"))
        ml_orchestrator = _deep_merge(
            ml_orchestrator,
            _load_json(diag_dir / "MLOrchestrator.json"),
        )
        guardrails = _deep_merge(guardrails, _load_json(diag_dir / "guardrails.json"))

        # Extract orchestrator parameters
        loop_interval = config_json.get("loop_interval_seconds", 1.0)
        max_candles = config_json.get("max_candles_per_symbol", 5000)

        # Extract risk caps
        risk_caps = config_json.get("risk_caps", {})

        # Extract minion weights
        minion_weights = ml_orchestrator.get("minion_weights", {})

        # Extract personality settings
        personality = ml_orchestrator.get("personality", {})

        # Telemetry paths
        logs_dir = root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        runtime_errors_path = logs_dir / "runtime_errors.jsonl"

        return cls(
            loop_interval_seconds=float(loop_interval),
            max_candles_per_symbol=int(max_candles),
            symbols=symbols,
            control=control,
            switches=switches,
            feeds=feeds,
            risk_caps=risk_caps,
            minion_weights=minion_weights,
            personality=personality,
            guardrails=guardrails,
            ml_orchestrator=ml_orchestrator,
            global_config=global_config,
            health_monitor=health_monitor,
            strategy_patterns=strategy_patterns,
            root=root,
            runtime_errors_path=runtime_errors_path,
            rotate_jsonl=True,
        )

    @classmethod
    def from_env(cls) -> "LiveConfig":
        root = Path.cwd()
        return cls.load_from_files(root)

    # -----------------------------------------------------------------------
    # Accessors
    # -----------------------------------------------------------------------

    def symbol_meta(self, symbol: str) -> Dict[str, Any]:
        return self.symbols.get(symbol, {})

    def is_enabled(self, key: str) -> bool:
        return bool(self.control.get(key, True))

    def switch(self, key: str, default: Any = None) -> Any:
        return self.switches.get(key, default)

    def feed_config(self, name: str) -> Dict[str, Any]:
        return self.feeds.get(name, {})

    def minion_weight(self, name: str) -> float:
        return float(self.minion_weights.get(name, 1.0))

    def personality_config(self) -> Dict[str, Any]:
        return self.personality

    def guardrail_config(self) -> Dict[str, Any]:
        return self.guardrails

    def risk_cap(self, key: str, default: Any = None) -> Any:
        return self.risk_caps.get(key, default)

    def strategy_params(self, name: str) -> Dict[str, Any]:
        """Return merged strategy parameters (pattern configs)."""
        return self.strategy_patterns.get(name, {})

# --- Global accessor for LiveConfig ---

_live_config_global = None


def set_global_live_config(cfg) -> None:
    """
    Register the process-wide LiveConfig instance.
    """
    global _live_config_global
    _live_config_global = cfg


def get_global_live_config():
    """
    Return the process-wide LiveConfig instance, or None if not set.
    """
    return _live_config_global