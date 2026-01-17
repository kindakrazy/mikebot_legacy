"""
mikebot.core.candle_engine

Unified ingestion + buffering + heartbeat engine with multi-timeframe
support and future expansion hooks.

This file is a full rewrite of the original CandleEngine with:
- first-class multi-timeframe snapshot APIs
- namespaced multi-TF fusion helpers
- optional ExperienceStore adapter hook
- backward-compatible single-TF APIs preserved
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Optional, Tuple, List, Any

import pandas as pd


# ---------------------------------------------------------------------------
# Symbol metadata + tick size
# ---------------------------------------------------------------------------

class SymbolRegistry:
    """
    Symbol metadata and tick size lookup.

    Backward-compatible loader from JSON file.
    """

    def __init__(self, symbols: Dict[str, Dict]):
        self.symbols = symbols or {}

    @classmethod
    def from_file(cls, path: Path) -> "SymbolRegistry":
        if not path.exists():
            return cls({})
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(data)

    def tick_size(self, symbol: str) -> float:
        meta = self.symbols.get(symbol, {})
        return float(meta.get("tick_size", 0.0001))


# ---------------------------------------------------------------------------
# Candle + buffer
# ---------------------------------------------------------------------------

@dataclass
class Candle:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class CandleBuffer:
    """
    Time-indexed buffer for a single (symbol, timeframe).

    Stores a tz-aware UTC DatetimeIndex and OHLCV columns.
    """

    def __init__(self, symbol: str, timeframe: str, max_len: int = 10_000):
        self.symbol = symbol
        self.timeframe = timeframe
        self.max_len = max_len
        self.df = pd.DataFrame(
            columns=["open", "high", "low", "close", "volume"],
            index=pd.DatetimeIndex([], name="timestamp", tz="UTC"),
        )

    def append(self, candle: Candle) -> None:
        ts = candle.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)

        ts_idx = pd.Timestamp(ts)

        # Overwrite if timestamp already exists
        self.df.loc[ts_idx] = [
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
        ]

        # Drop duplicates BEFORE sorting
        self.df = self.df[~self.df.index.duplicated(keep="last")]

        # Now safe to sort
        self.df.sort_index(inplace=True)

        # Enforce max length
        if len(self.df) > self.max_len:
            self.df = self.df.iloc[-self.max_len:]


    def append_if_new(self, candle: Candle) -> None:
        """
        Append only if this timestamp is not already present.
        Used for history ingestion with deduplication.
        """
        ts = candle.timestamp
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)
        ts_idx = pd.Timestamp(ts)
        if ts_idx in self.df.index:
            return
        self.df.loc[ts_idx] = [
            candle.open,
            candle.high,
            candle.low,
            candle.close,
            candle.volume,
        ]
        self.df.sort_index(inplace=True)
        if len(self.df) > self.max_len:
            self.df = self.df.iloc[-self.max_len :]

    def to_df(self) -> pd.DataFrame:
        return self.df.copy()

    def tail(self, n: int) -> pd.DataFrame:
        return self.df.iloc[-n:].copy()


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

class HeartbeatMonitor:
    """
    Simple heartbeat monitor for ingestion health.
    """

    def __init__(self, stale_after_seconds: int = 60):
        self.last_seen: Optional[datetime] = None
        self.stale_after = stale_after_seconds

    def mark(self) -> None:
        self.last_seen = datetime.now(timezone.utc)

    def status(self) -> Dict[str, Any]:
        if self.last_seen is None:
            return {"is_stale": True, "reason": "never_seen"}
        age = (datetime.now(timezone.utc) - self.last_seen).total_seconds()
        return {
            "is_stale": age > self.stale_after,
            "age_seconds": age,
            "last_seen": self.last_seen.isoformat(),
        }


# ---------------------------------------------------------------------------
# CandleEngine (multi-TF aware)
# ---------------------------------------------------------------------------

class CandleEngine:
    """
    Unified ingestion, buffering, and snapshot engine.

    Key features:
      - Backward-compatible single-TF buffer APIs
      - Multi-TF snapshot and fusion helpers:
          * get_multi_tf_snapshot(timeframes)
          * get_multi_tf_df(symbol, timeframes, align='inner', namespace=True)
      - Optional ExperienceStore adapter via `experience_store` argument
      - Namespacing of columns to make feature origin explicit
    """

    def __init__(
        self,
        symbol_registry: SymbolRegistry,
        max_buffer_len: int = 10_000,
        experience_store: Optional[Any] = None,
    ):
        self.symbol_registry = symbol_registry
        self.max_buffer_len = max_buffer_len
        self.buffers: Dict[Tuple[str, str], CandleBuffer] = {}
        self.heartbeat = HeartbeatMonitor()
        # Optional adapter to push snapshots into ExperienceStore
        self.experience_store = experience_store

    # ----------------------------------------------------------------------
    # Buffer access
    # ----------------------------------------------------------------------

    def _buf(self, symbol: str, timeframe: str) -> CandleBuffer:
        key = (symbol, timeframe)
        if key not in self.buffers:
            self.buffers[key] = CandleBuffer(symbol, timeframe, self.max_buffer_len)
        return self.buffers[key]

    # ----------------------------------------------------------------------
    # CSV ingestion
    # ----------------------------------------------------------------------

    def ingest_csv(self, path: Path, symbol: str, timeframe: str) -> None:
        """
        Parse CSV and append candles to the buffer.

        Expected CSV schema:
            timestamp, open, high, low, close, volume
        """
        df = pd.read_csv(path)
        if "timestamp" not in df.columns:
            raise ValueError("CSV must contain a 'timestamp' column.")
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        buf = self._buf(symbol, timeframe)
        for row in df.itertuples(index=False):
            ts = getattr(row, "timestamp")
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            else:
                ts = ts.tz_convert("UTC") if hasattr(ts, "tz_convert") else ts.astimezone(timezone.utc)
            candle = Candle(
                timestamp=ts.to_pydatetime(),
                open=float(getattr(row, "open")),
                high=float(getattr(row, "high")),
                low=float(getattr(row, "low")),
                close=float(getattr(row, "close")),
                volume=float(getattr(row, "volume", 0.0)),
            )
            buf.append(candle)
        self.heartbeat.mark()

    # ----------------------------------------------------------------------
    # Live ingestion
    # ----------------------------------------------------------------------

    def ingest_live_tick(
        self,
        symbol: str,
        timeframe: str,
        tick: Dict[str, float],
        ts: datetime,
    ) -> None:
        """
        Append a live tick (or candle) to the buffer.
        """
        candle = Candle(
            timestamp=ts.astimezone(timezone.utc),
            open=float(tick["open"]),
            high=float(tick["high"]),
            low=float(tick["low"]),
            close=float(tick["close"]),
            volume=float(tick.get("volume", 0.0)),
        )
        self._buf(symbol, timeframe).append(candle)
        self.heartbeat.mark()

    def ingest_history_candle(
        self,
        symbol: str,
        timeframe: str,
        tick: Dict[str, float],
        ts: datetime,
    ) -> None:
        """
        Append a history candle to the buffer with deduplication by timestamp.

        This does NOT mark the heartbeat. History is not considered "live".
        """
        candle = Candle(
            timestamp=ts.astimezone(timezone.utc),
            open=float(tick["open"]),
            high=float(tick["high"]),
            low=float(tick["low"]),
            close=float(tick["close"]),
            volume=float(tick.get("volume", 0.0)),
        )
        self._buf(symbol, timeframe).append_if_new(candle)

    # ----------------------------------------------------------------------
    # Views: single-TF
    # ----------------------------------------------------------------------

    def get_df(self, symbol: str, timeframe: str) -> pd.DataFrame:
        """
        Return the buffer DataFrame for a single (symbol, timeframe).
        """
        return self._buf(symbol, timeframe).to_df()

    def get_tick_size(self, symbol: str) -> float:
        return self.symbol_registry.tick_size(symbol)

    def health(self) -> Dict[str, Any]:
        return self.heartbeat.status()

    def get_latest_snapshot(self, timeframe: str = "M1") -> Dict[str, pd.DataFrame]:
        """
        Backward-compatible snapshot for a single timeframe.

        Returns:
            { symbol: DataFrame(reset_index) }
        """
        snapshot: Dict[str, pd.DataFrame] = {}
        for (symbol, tf), buf in self.buffers.items():
            if tf != timeframe:
                continue
            df = buf.to_df()
            if df.empty:
                continue
            out = df.copy().reset_index()
            snapshot[symbol] = out
        return snapshot

    # ----------------------------------------------------------------------
    # Multi-TF snapshot APIs (new)
    # ----------------------------------------------------------------------

    def get_multi_tf_snapshot(self, timeframes: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Return recent buffers for the requested timeframes.

        Each DataFrame is tz-aware, indexed by timestamp.
        """
        snapshot: Dict[str, Dict[str, pd.DataFrame]] = {}
        tf_set = set(str(tf) for tf in timeframes)

        for (symbol, tf), buf in self.buffers.items():
            if str(tf) not in tf_set:
                continue
            df = buf.to_df()
            if df.empty:
                continue
            # normalize timeframe key to string (e.g., "M1" -> "1" or keep as provided)
            snapshot.setdefault(symbol, {})[str(tf)] = df.copy()

        return snapshot

    def _namespace_columns(self, df: pd.DataFrame, tf: str) -> pd.DataFrame:
        """
        Prefix columns with timeframe to avoid collisions, e.g. 'M1_close'.
        Keeps index as timestamp.
        """
        if df.empty:
            return df
        ns = {c: f"{tf}_{c}" for c in df.columns}
        return df.rename(columns=ns)

    def get_multi_tf_df(
        self,
        symbol: str,
        timeframes: List[str],
        align: str = "inner",
        namespace: bool = True,
        tail: Optional[int] = None,
    ) -> pd.DataFrame:
        """
        Build a fused DataFrame for a single symbol from multiple timeframe buffers.

        Parameters:
          - symbol: symbol to fuse
          - timeframes: list of timeframe identifiers (e.g., ["M1","M5","M15"] or ["1","5","15"])
          - align: 'inner' (default) or 'outer' join strategy
          - namespace: whether to prefix columns with timeframe
          - tail: if provided, take last `tail` rows from each TF before joining

        Returns:
          - merged DataFrame indexed by timestamp. Empty DataFrame if no usable frames.
        """
        frames: List[pd.DataFrame] = []
        for tf in timeframes:
            # Accept both "M1" and "1" style keys; try both
            candidates = [tf, f"M{tf}" if not tf.upper().startswith("M") and tf.isdigit() else tf]
            df = None
            for cand in candidates:
                try:
                    buf = self._buf(symbol, cand)
                    df_cand = buf.to_df()
                    if df_cand.empty:
                        continue
                    if tail is not None:
                        df_cand = df_cand.iloc[-tail:]
                    df = df_cand
                    break
                except Exception:
                    continue
            if df is None or df.empty:
                # skip missing TFs (caller should handle missing TFs)
                continue
            tf_key = tf
            if namespace:
                df = self._namespace_columns(df, tf_key)
            frames.append(df)

        if not frames:
            return pd.DataFrame()

        # Merge frames on index
        merged = frames[0]
        for other in frames[1:]:
            merged = merged.join(other, how=align)

        # Sort index and drop rows with any NaNs if inner join used
        merged = merged.sort_index()
        if align == "inner":
            merged = merged.dropna(how="any")

        return merged

    # ----------------------------------------------------------------------
    # ExperienceStore adapter (optional)
    # ----------------------------------------------------------------------

    def push_multi_tf_to_experience_store(
        self,
        symbol: str,
        timeframes: List[str],
        window: Optional[int] = None,
        namespace: bool = True,
    ) -> None:
        """
        Convenience helper to push a fused multi-TF window into an ExperienceStore.

        Requires self.experience_store to be set and to implement:
            experience_store.ingest_multi_tf(symbol: str, tf_frames: Dict[str, pd.DataFrame])

        This method does not assume any particular ExperienceStore API beyond the above.
        """
        if self.experience_store is None:
            raise RuntimeError("No experience_store configured on CandleEngine.")

        tf_frames = {}
        for tf in timeframes:
            try:
                buf = self._buf(symbol, tf)
                df = buf.to_df()
                if df.empty:
                    continue
                if window is not None:
                    df = df.iloc[-window:]
                tf_frames[str(tf)] = df.copy()
            except Exception:
                continue

        if not tf_frames:
            return

        # Delegate ingestion to the experience store
        try:
            self.experience_store.ingest_multi_tf(symbol=symbol, tf_frames=tf_frames, namespace=namespace)
        except TypeError:
            # Fallback to a simpler API if ingest_multi_tf signature differs
            self.experience_store.ingest_multi_tf(symbol, tf_frames)

    # ----------------------------------------------------------------------
    # Snapshot utilities for orchestrator
    # ----------------------------------------------------------------------

    def get_multi_tf_snapshot_for_orchestrator(self, timeframes: List[str]) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Higher-level orchestrator-facing snapshot that returns namespaced frames
        for each symbol and timeframe. Useful for inference pipelines that want
        to fuse locally.
        """
        raw = self.get_multi_tf_snapshot(timeframes)
        namespaced: Dict[str, Dict[str, pd.DataFrame]] = {}
        for symbol, frames in raw.items():
            for tf, df in frames.items():
                namespaced.setdefault(symbol, {})[tf] = df.copy()
        return namespaced

    # ----------------------------------------------------------------------
    # Backward-compatible helpers for listing available symbols/timeframes
    # ----------------------------------------------------------------------

    def available_symbols(self) -> List[str]:
        syms = {symbol for (symbol, _) in self.buffers.keys()}
        return sorted(syms)

    def available_timeframes_for_symbol(self, symbol: str) -> List[str]:
        tfs = {tf for (sym, tf) in self.buffers.keys() if sym == symbol}
        return sorted(str(tf) for tf in tfs)
