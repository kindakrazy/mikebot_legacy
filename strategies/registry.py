# mikebot/strategies/registry.py

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Dict, Type, Optional

import pandas as pd

from mikebot.config.strategy_config_loader import StrategyConfigLoader
from mikebot.strategies.base import Strategy

logger = logging.getLogger(__name__)


def load_strategies() -> Dict[str, Type[Strategy]]:
    """
    Auto-discovers all strategy modules inside strategies/strategies/.
    Each module must define a class named StrategyImpl.
    """
    strategies: Dict[str, Type[Strategy]] = {}

    # Safe import of strategies package
    try:
        import mikebot.strategies.strategies as strategies_pkg
    except Exception:
        logger.warning("No strategies package found; returning empty strategy set")
        return strategies

    # Discover modules
    for module_info in pkgutil.iter_modules(strategies_pkg.__path__):
        module_name = module_info.name
        try:
            module = importlib.import_module(f"mikebot.strategies.strategies.{module_name}")
        except Exception as e:
            logger.warning("Failed to import strategy module %s: %s", module_name, e)
            continue

        if hasattr(module, "StrategyImpl"):
            cls = getattr(module, "StrategyImpl")
            name = getattr(cls, "name", module_name)
            strategies[name] = cls

    return strategies

def _ensure_timestamp_index(df: pd.DataFrame, timestamp_col: Optional[str] = "timestamp") -> pd.DataFrame:
    """
    Ensure the returned DataFrame is indexed by a timezone-aware datetime index
    when possible. If a 'timestamp' column exists, convert it and set as index.
    Otherwise return a shallow copy (caller may choose to align by position).
    """
    if df is None:
        return df

    df = df.copy()
    if timestamp_col in df.columns:
        try:
            df[timestamp_col] = pd.to_datetime(df[timestamp_col], utc=True)
            df = df.sort_values(timestamp_col).set_index(timestamp_col)
        except Exception:
            # If conversion fails, leave as-is but log for debugging
            logger.debug("Failed to convert strategy timestamp column to datetime; leaving index unchanged")
    else:
        # If index already datetime-like, ensure tz-aware
        try:
            if pd.api.types.is_datetime64_any_dtype(df.index):
                # make tz-aware if naive
                if df.index.tz is None:
                    df.index = df.index.tz_localize("UTC")
        except Exception:
            pass

    return df


def _attach_metadata(df: pd.DataFrame, name: str, cls: Type[Strategy]) -> pd.DataFrame:
    """
    Attach minimal metadata to the DataFrame via .attrs so downstream code can
    inspect strategy name/version without changing return signature.
    """
    if df is None:
        return df
    meta = df.attrs.get("meta", {})
    meta.setdefault("name", getattr(cls, "name", name))
    meta.setdefault("version", getattr(cls, "version", "unknown"))
    df.attrs["meta"] = meta
    return df


def compute_all(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Compute all strategies using their configs.
    Returns:
        {strategy_name: DataFrame(features)}
    Notes:
      - Each strategy DataFrame will be normalized to use a timestamp index
        when a 'timestamp' column is present in the strategy output.
      - If a strategy output length matches the input candles length, the
        strategy DataFrame will be aligned to the candles index by position.
      - Metadata about the strategy (name/version) is attached to df.attrs["meta"].
    """
    loader = StrategyConfigLoader()
    out: Dict[str, pd.DataFrame] = {}

    for name, cls in load_strategies().items():
        cfg = loader.get(name)
        instance = cls(cfg)
        sdf = instance.compute(df)

        # If strategy returned None or empty, keep as-is
        if sdf is None:
            out[name] = sdf
            continue

        # If strategy output contains a timestamp column, convert and set index
        if "timestamp" in sdf.columns:
            try:
                sdf = sdf.copy()
                sdf["timestamp"] = pd.to_datetime(sdf["timestamp"], utc=True)
                sdf = sdf.sort_values("timestamp").set_index("timestamp")
            except Exception:
                # leave as-is if conversion fails
                pass
        else:
            # If the strategy output length equals the candles length, align by position
            try:
                if len(sdf) == len(df):
                    if hasattr(df, "index") and pd.api.types.is_datetime64_any_dtype(df.index):
                        sdf = sdf.copy()
                        sdf.index = df.index
                    else:
                        sdf = sdf.copy()
            except Exception:
                pass

        # Attach minimal metadata
        try:
            sdf.attrs["meta"] = {"name": getattr(cls, "name", name), "version": getattr(cls, "version", "unknown")}
        except Exception:
            pass

        out[name] = sdf

    return out


def compute_all_filtered(df: pd.DataFrame, toggles: Dict[str, bool]) -> Dict[str, pd.DataFrame]:
    """
    Same as compute_all(), but only includes strategies that are enabled
    in the strategy_toggles mapping.

    toggles: mapping {strategy_name: bool} where missing keys default to True.
    """
    loader = StrategyConfigLoader()
    out: Dict[str, pd.DataFrame] = {}

    for name, cls in load_strategies().items():
        enabled = toggles.get(name, True)
        if not enabled:
            logger.debug("Strategy %s disabled by toggles; skipping", name)
            continue

        cfg = loader.get(name)
        instance = cls(cfg)
        sdf = instance.compute(df)

        # If strategy returned None or empty, keep as-is
        if sdf is None:
            out[name] = sdf
            continue

        # If strategy output contains a timestamp column, convert and set index
        if "timestamp" in sdf.columns:
            try:
                sdf = sdf.copy()
                sdf["timestamp"] = pd.to_datetime(sdf["timestamp"], utc=True)
                sdf = sdf.sort_values("timestamp").set_index("timestamp")
            except Exception:
                pass
        else:
            # If the strategy output length equals the candles length, align by position
            try:
                if len(sdf) == len(df):
                    if hasattr(df, "index") and pd.api.types.is_datetime64_any_dtype(df.index):
                        sdf = sdf.copy()
                        sdf.index = df.index
                    else:
                        sdf = sdf.copy()
            except Exception:
                pass

        # Attach minimal metadata
        try:
            sdf.attrs["meta"] = {"name": getattr(cls, "name", name), "version": getattr(cls, "version", "unknown")}
        except Exception:
            pass

        out[name] = sdf

    return out
