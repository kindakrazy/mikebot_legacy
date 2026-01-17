from __future__ import annotations

import re
import logging
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ======================================================================
# RESULT OBJECT
# ======================================================================

@dataclass
class NormalizationResult:
    symbol: str
    timeframe: str
    start_timestamp: pd.Timestamp
    end_timestamp: pd.Timestamp
    row_count: int
    output_path: Path
    quality_score: float = 1.0
    gap_stats: Dict[str, Any] = field(default_factory=dict)

    # v2 additions
    normalizer_version: str = "2.0"
    headerless_detected: bool = False
    timestamp_inference: Dict[str, Any] = field(default_factory=dict)
    numeric_columns: List[Dict[str, Any]] = field(default_factory=list)
    ohlc_source: Dict[str, Any] = field(default_factory=dict)
    volume_source: Dict[str, Any] = field(default_factory=dict)
    timeframe_info: Dict[str, Any] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    summary: Dict[str, Any] = field(default_factory=dict)
    raw_path: Optional[Path] = None
    symbol_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "start_timestamp": str(self.start_timestamp),
            "end_timestamp": str(self.end_timestamp),
            "row_count": self.row_count,
            "output_path": str(self.output_path),
            "quality_score": self.quality_score,
            "gap_stats": self.gap_stats,
            "normalizer_version": self.normalizer_version,
            "headerless_detected": self.headerless_detected,
            "timestamp_inference": self.timestamp_inference,
            "numeric_columns": self.numeric_columns,
            "ohlc_source": self.ohlc_source,
            "volume_source": self.volume_source,
            "timeframe_info": self.timeframe_info,
            "warnings": self.warnings,
            "summary": self.summary,
            "raw_path": str(self.raw_path) if self.raw_path is not None else None,
            "symbol_info": self.symbol_info,
        }


# ======================================================================
# HELPER: GAP STATISTICS
# ======================================================================

def _compute_gap_stats(df: pd.DataFrame, gap_col: str = "is_gap") -> Tuple[float, Dict[str, Any]]:
    """
    Compute data quality metrics based on the gap column.
    Returns (quality_score, stats_dict).
    """
    total_rows = len(df)
    if total_rows == 0:
        return 0.0, {
            "total_gaps": 0,
            "max_consecutive_gap": 0,
            "gap_ratio": 0.0,
            "leading_gaps": 0,
            "trailing_gaps": 0,
        }

    total_gaps = int(df[gap_col].sum())
    gap_ratio = total_gaps / total_rows
    quality_score = 1.0 - gap_ratio

    # Calculate max consecutive gaps
    gaps = df[gap_col].values.astype(int)
    padded = np.concatenate(([0], gaps, [0]))
    diffs = np.diff(padded)
    run_starts = np.where(diffs == 1)[0]
    run_ends = np.where(diffs == -1)[0]

    max_consecutive = 0
    if len(run_starts) > 0 and len(run_ends) > 0:
        lengths = run_ends - run_starts
        max_consecutive = int(lengths.max())

    # Leading / trailing gaps
    leading_gaps = 0
    trailing_gaps = 0
    if total_gaps > 0:
        # leading gaps
        for v in gaps:
            if v == 1:
                leading_gaps += 1
            else:
                break
        # trailing gaps
        for v in gaps[::-1]:
            if v == 1:
                trailing_gaps += 1
            else:
                break

    stats = {
        "total_gaps": total_gaps,
        "max_consecutive_gap": max_consecutive,
        "gap_ratio": round(gap_ratio, 4),
        "leading_gaps": leading_gaps,
        "trailing_gaps": trailing_gaps,
    }

    return round(quality_score, 4), stats


# ======================================================================
# INTERNAL HELPERS
# ======================================================================

def _load_csv(raw_path: Path) -> Tuple[pd.DataFrame, bool, List[str]]:
    if not raw_path.exists():
        raise FileNotFoundError(f"Source file not found: {raw_path}")

    logger.info(f"[normalize] Loading CSV: {raw_path}")
    df = pd.read_csv(raw_path)

    original_columns = list(df.columns)
    lowered = [str(c).lower().strip() for c in df.columns]
    df.columns = lowered

    # Detect headerless CSV
    headerless = all(
        str(c).startswith("unnamed") or str(c).isdigit()
        for c in original_columns
    )

    if not headerless:
        rename_map = {
            "datetime": "timestamp",
            "time_stamp": "timestamp",
            "date_time": "timestamp",
            "open_time": "timestamp",
            "time": "time",
            "date": "date",
            "vol": "volume",
            "tick_volume": "volume",
            "qty": "volume",
        }
        df = df.rename(columns=rename_map)

    return df, headerless, original_columns


def _infer_timestamp(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Infer timestamp column from the DataFrame.
    Returns (df_with_timestamp, diagnostics).
    """
    ts_diag: Dict[str, Any] = {
        "mode": None,
        "columns_tested": [],
        "chosen_columns": [],
        "parse_ratio": None,
        "dropped_rows": None,
    }

    ts_col = None
    date_col = None
    time_col = None

    cols = list(df.columns)

    # 1) Try date+time pair
    best_pair = None
    best_pair_ratio = 0.0

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            combined = (
                df[cols[i]].astype(str).str.strip()
                + " "
                + df[cols[j]].astype(str).str.strip()
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(combined, errors="coerce", utc=True)

            ratio = parsed.notna().mean()
            ts_diag["columns_tested"].append(
                {"type": "date+time_pair", "columns": [cols[i], cols[j]], "ratio": float(ratio)}
            )
            if ratio > best_pair_ratio:
                best_pair_ratio = ratio
                best_pair = (cols[i], cols[j])

    if best_pair is not None and best_pair_ratio >= 0.8:
        date_col, time_col = best_pair
        ts_diag["mode"] = "date+time"
        ts_diag["chosen_columns"] = [date_col, time_col]
        ts_diag["parse_ratio"] = float(best_pair_ratio)

    # 2) Otherwise try single timestamp column
    if date_col is None and time_col is None:
        best_col = None
        best_ratio = 0.0
        for col in cols:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                parsed = pd.to_datetime(df[col].astype(str), errors="coerce", utc=True)

            ratio = parsed.notna().mean()
            ts_diag["columns_tested"].append(
                {"type": "single", "column": col, "ratio": float(ratio)}
            )
            if ratio > best_ratio:
                best_ratio = ratio
                best_col = col

        if best_col is not None and best_ratio >= 0.8:
            ts_col = best_col
            ts_diag["mode"] = "single"
            ts_diag["chosen_columns"] = [ts_col]
            ts_diag["parse_ratio"] = float(best_ratio)

    # 3) Build timestamp
    if date_col is not None and time_col is not None:
        combined = (
            df[date_col].astype(str).str.strip()
            + " "
            + df[time_col].astype(str).str.strip()
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ts = pd.to_datetime(combined, errors="coerce", utc=True)
    elif ts_col is not None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            ts = pd.to_datetime(df[ts_col].astype(str), errors="coerce", utc=True)
    else:
        raise ValueError("Could not infer timestamp column.")

    before_rows = len(df)
    df = df.copy()
    df["timestamp"] = ts
    df = df.dropna(subset=["timestamp"])
    after_rows = len(df)

    ts_diag["dropped_rows"] = int(before_rows - after_rows)

    if ts_diag["parse_ratio"] is not None and ts_diag["parse_ratio"] < 0.9:
        logger.warning(
            f"[normalize] Timestamp parse ratio is low: {ts_diag['parse_ratio']:.3f}"
        )

    return df, ts_diag


def _detect_numeric_columns(df: pd.DataFrame, exclude_cols: List[str]) -> Tuple[List[Tuple[str, pd.Series]], List[Dict[str, Any]]]:
    numeric_cols: List[Tuple[str, pd.Series]] = []
    diagnostics: List[Dict[str, Any]] = []

    for col in df.columns:
        if col in exclude_cols:
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        non_na = coerced.notna().mean()
        if non_na == 0:
            diagnostics.append({
                "column": col,
                "numeric_ratio": 0.0,
                "integer_like_ratio": None,
                "float_like_ratio": None,
            })
            continue

        non_na_values = coerced.dropna()
        integer_like_ratio = (non_na_values.round() == non_na_values).mean() if not non_na_values.empty else 0.0
        float_like_ratio = (non_na_values.round() != non_na_values).mean() if not non_na_values.empty else 0.0

        diagnostics.append({
            "column": col,
            "numeric_ratio": float(non_na),
            "integer_like_ratio": float(integer_like_ratio),
            "float_like_ratio": float(float_like_ratio),
        })

        if non_na >= 0.9:
            numeric_cols.append((col, coerced))

    if not numeric_cols:
        raise ValueError("No numeric columns found for OHLC/volume inference.")

    return numeric_cols, diagnostics


def _infer_volume(numeric_cols: List[Tuple[str, pd.Series]]) -> Tuple[Optional[pd.Series], Dict[str, Any]]:
    volume_series = None
    volume_name = None

    vol_diag: Dict[str, Any] = {
        "method": None,
        "source_column": None,
        "integer_like_ratio": None,
        "note": None,
    }

    vol_candidates = ["volume", "vol", "qty", "tick_volume"]
    name_to_series = {name: series for name, series in numeric_cols}

    # Name-based
    for vname in vol_candidates:
        if vname in name_to_series:
            series = name_to_series[vname]
            non_na = series.dropna()
            if not non_na.empty and (non_na < 0).any():
                logger.warning(f"[normalize] Negative volume values detected in column '{vname}', taking abs().")
                vol_diag["note"] = "negative_values_abs"
            volume_series = series.abs()
            volume_name = vname
            vol_diag["method"] = "name_match"
            vol_diag["source_column"] = volume_name
            return volume_series, vol_diag

    # Shape-based heuristic
    best_ratio = 0.0
    best_series = None
    best_name = None
    for name, series in numeric_cols:
        non_na = series.dropna()
        if non_na.empty:
            continue
        int_like = (non_na.round() == non_na).mean()
        if int_like > best_ratio:
            best_ratio = int_like
            best_series = series
            best_name = name

    if best_series is not None and best_ratio >= 0.8:
        non_na = best_series.dropna()
        if not non_na.empty and (non_na < 0).any():
            logger.warning(f"[normalize] Negative volume values detected in column '{best_name}', taking abs().")
            vol_diag["note"] = "negative_values_abs"
        volume_series = best_series.abs()
        volume_name = best_name
        vol_diag["method"] = "int_like_heuristic"
        vol_diag["source_column"] = volume_name
        vol_diag["integer_like_ratio"] = float(best_ratio)
        return volume_series, vol_diag

    # Fallback: no explicit volume
    vol_diag["method"] = "default_zero"
    vol_diag["source_column"] = None
    return None, vol_diag


def _infer_ohlc(numeric_cols: List[Tuple[str, pd.Series]], volume_name: Optional[str]) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series, Dict[str, Any], List[str]]:
    """
    Infer OHLC from numeric columns, excluding the volume column.
    Returns (open_series, high_series, low_series, close_series, diagnostics, warnings).
    """
    warnings_list: List[str] = []
    price_candidates: List[Tuple[str, pd.Series]] = []

    for name, series in numeric_cols:
        if volume_name and name == volume_name:
            continue
        non_na = series.dropna()
        if non_na.empty:
            continue
        float_like = (non_na.round() != non_na).mean()
        if float_like >= 0.5:
            price_candidates.append((name, series))

    if not price_candidates:
        raise ValueError("No usable price columns found for OHLC inference.")

    # Helper to pick by substring in name
    def pick_by_name(target: str) -> Tuple[Optional[str], Optional[pd.Series]]:
        for name, series in price_candidates:
            if target in name:
                return name, series
        return None, None

    open_name, open_series = pick_by_name("open")
    high_name, high_series = pick_by_name("high")
    low_name, low_series = pick_by_name("low")
    close_name, close_series = pick_by_name("close")

    ordered = price_candidates
    stack = [s for _, s in ordered]
    price_max = pd.concat(stack, axis=1).max(axis=1)
    price_min = pd.concat(stack, axis=1).min(axis=1)

    ohlc_diag: Dict[str, Any] = {
        "open": {"source": None},
        "high": {"source": None},
        "low": {"source": None},
        "close": {"source": None},
        "flat_candles": False,
        "price_candidate_columns": [name for name, _ in price_candidates],
    }

    if open_series is None:
        open_name, open_series = ordered[0]
        ohlc_diag["open"]["source"] = {"type": "fallback_first_price", "column": open_name}
    else:
        ohlc_diag["open"]["source"] = {"type": "named_match", "column": open_name}

    if close_series is None:
        close_name, close_series = ordered[-1]
        ohlc_diag["close"]["source"] = {"type": "fallback_last_price", "column": close_name}
    else:
        ohlc_diag["close"]["source"] = {"type": "named_match", "column": close_name}

    if high_series is None:
        high_series = price_max
        ohlc_diag["high"]["source"] = {"type": "synthetic_max_all_prices"}
    else:
        ohlc_diag["high"]["source"] = {"type": "named_match_or_original"}

    if low_series is None:
        low_series = price_min
        ohlc_diag["low"]["source"] = {"type": "synthetic_min_all_prices"}
    else:
        ohlc_diag["low"]["source"] = {"type": "named_match_or_original"}

    if len(price_candidates) == 1:
        warnings_list.append("Only one price column available; OHLC will be flat.")
        ohlc_diag["flat_candles"] = True

    return open_series, high_series, low_series, close_series, ohlc_diag, warnings_list


def _infer_timeframe(out: pd.DataFrame) -> Tuple[str, Dict[str, Any], List[str]]:
    """
    Infer timeframe from timestamp deltas.
    Returns (freq_string, diagnostics, warnings).
    """
    warnings_list: List[str] = []

    deltas = out["timestamp"].diff().dropna().dt.total_seconds() / 60.0
    deltas = deltas.round(6)

    if deltas.empty:
        # Degenerate case: only one row.
        freq = "1min"
        diag = {
            "base_delta_minutes": None,
            "mode_delta_minutes": None,
            "min_delta_minutes": None,
            "max_delta_minutes": None,
            "std_delta_minutes": None,
            "unique_deltas": 0,
        }
        warnings_list.append("Insufficient rows to infer timeframe; defaulting to 1min.")
        return freq, diag, warnings_list

    base_delta = float(deltas.mode().iloc[0])

    min_delta = float(deltas.min())
    max_delta = float(deltas.max())
    std_delta = float(deltas.std()) if len(deltas) > 1 else 0.0
    unique_deltas = len(deltas.value_counts())

    if unique_deltas > 1:
        warnings_list.append(
            f"Multiple distinct timestamp intervals detected ({unique_deltas}); using mode as base interval."
        )

    known = {
        1: "1min",
        5: "5min",
        15: "15min",
        30: "30min",
        60: "1H",
        240: "4H",
        1440: "1D",
    }

    def match_known(delta: float) -> Tuple[str, float]:
        for k, f in known.items():
            if abs(delta - k) < 1e-3:
                return f, float(k)
        # fallback to nearest whole-minute frequency
        rounded = int(round(delta))
        if rounded <= 0:
            rounded = 1
        warnings_list.append(
            f"Non-standard interval detected: {delta:.6f} minutes; using freq='{rounded}min'."
        )
        return f"{rounded}min", float(rounded)

    freq, matched_delta = match_known(base_delta)

    diag = {
        "base_delta_minutes": float(base_delta),
        "matched_delta_minutes": matched_delta,
        "mode_delta_minutes": float(base_delta),
        "min_delta_minutes": min_delta,
        "max_delta_minutes": max_delta,
        "std_delta_minutes": std_delta,
        "unique_deltas": int(unique_deltas),
    }

    return freq, diag, warnings_list


def _infer_symbol(raw_path: Path, freq: str) -> Tuple[str, Dict[str, Any]]:
    """
    Infer symbol from filename stem. Keep diagnostics about parsing.
    """
    stem = raw_path.stem.upper()
    info: Dict[str, Any] = {
        "original_stem": stem,
        "base_symbol": None,
        "suffix": None,
        "suffix_is_timeframe_like": False,
        "timeframe_from_suffix": None,
    }

    # Matches: BTCUSD15, LTCUSD5, BTCUSD_M15, BTCUSD.M1
    m = re.match(r"([A-Z]{2,15})(?:[_\.]?([0-9]+[A-Z]?))?", stem)

    if m:
        base_symbol = m.group(1)
        tf_suffix = m.group(2)
        info["base_symbol"] = base_symbol
        info["suffix"] = tf_suffix

        # Heuristic: numeric or number+letter suffix might be timeframe-like
        if tf_suffix is not None:
            info["suffix_is_timeframe_like"] = True
            info["timeframe_from_suffix"] = tf_suffix

        symbol = base_symbol + (tf_suffix if tf_suffix else "")
    else:
        symbol = stem
        info["base_symbol"] = stem

    return symbol, info


def _validate_output(out: pd.DataFrame) -> List[str]:
    """
    Validate final output before saving.
    Returns a list of warnings (no hard asserts).
    """
    warnings_list: List[str] = []

    required_cols = ["timestamp", "open", "high", "low", "close", "volume", "is_gap"]
    missing = [c for c in required_cols if c not in out.columns]
    if missing:
        raise ValueError(f"Missing required columns in output: {missing}")

    if out["timestamp"].isna().any():
        raise ValueError("NaN timestamps in output.")

    if out[["open", "high", "low", "close", "volume"]].isna().any().any():
        raise ValueError("NaN values in OHLCV columns after normalization.")

    if not out["timestamp"].is_monotonic_increasing:
        warnings_list.append("Timestamps are not strictly sorted in ascending order.")

    # Check duplicates
    dup_count = out["timestamp"].duplicated().sum()
    if dup_count > 0:
        warnings_list.append(f"{dup_count} duplicate timestamps detected in final output.")

    # Check OHLC integrity
    bad_bounds = (out["low"] > out["high"]) | (out["open"] > out["high"]) | (out["open"] < out["low"]) | (out["close"] > out["high"]) | (out["close"] < out["low"])
    if bad_bounds.any():
        warnings_list.append("Some OHLC rows violate high/low bounds.")

    # Volume >= 0
    if (out["volume"] < 0).any():
        warnings_list.append("Negative volume values detected in final output.")

    # is_gap in {0,1}
    unique_gaps = set(out["is_gap"].unique().tolist())
    if not unique_gaps.issubset({0, 1}):
        warnings_list.append("Non-binary values detected in is_gap column.")

    return warnings_list


def _summarize_output(out: pd.DataFrame) -> Dict[str, Any]:
    """
    Compute a summary of the final normalized data.
    """
    summary: Dict[str, Any] = {}
    if out.empty:
        return summary

    prices = out[["open", "high", "low", "close"]]
    summary["min_price"] = float(prices.min().min())
    summary["max_price"] = float(prices.max().max())

    total_rows = len(out)
    synthetic_rows = int(out["is_gap"].sum())
    real_rows = int(total_rows - synthetic_rows)
    summary["total_rows"] = int(total_rows)
    summary["synthetic_rows"] = synthetic_rows
    summary["real_rows"] = real_rows
    summary["synthetic_ratio"] = float(synthetic_rows / total_rows) if total_rows > 0 else 0.0

    # Note: we can't perfectly know "inferred OHLC rows" without more tagging,
    # but we can still include basic info that is available here.
    return summary


# ======================================================================
# CORE NORMALIZER
# ======================================================================

def _normalize_core(raw_path: Path, output_root: Optional[Path]) -> NormalizationResult:
    """
    Universal OHLCV normalizer with grid alignment, gap marking, and quality scoring.

    Guarantees output with columns:
        timestamp, open, high, low, close, volume, is_gap

    Rules:
    - Timestamp is REQUIRED (must be inferable or we error).
    - OHLC are REQUIRED (we infer missing ones from available price columns).
    - Volume is OPTIONAL (if missing, we set volume = 0).
    - Reindexes to a perfect time grid based on the inferred frequency.
    - Flags gaps BEFORE filling them.
    """

    # Accumulated diagnostics and warnings
    warnings_accum: List[str] = []

    # ------------------------------------------------------------------
    # LOAD CSV
    # ------------------------------------------------------------------
    df, headerless, original_columns = _load_csv(raw_path)

    # ------------------------------------------------------------------
    # TIMESTAMP INFERENCE
    # ------------------------------------------------------------------
    df, ts_diag = _infer_timestamp(df)

    # ------------------------------------------------------------------
    # NUMERIC COLUMN DETECTION
    # ------------------------------------------------------------------
    exclude_cols = {"timestamp"}
    # ts_diag may have chosen date/time columns, but we don't need to re-exclude by name
    numeric_cols, num_diag = _detect_numeric_columns(df, exclude_cols=list(exclude_cols))

    # ------------------------------------------------------------------
    # VOLUME INFERENCE
    # ------------------------------------------------------------------
    volume_series, vol_diag = _infer_volume(numeric_cols)
    volume_name = vol_diag["source_column"]

    # ------------------------------------------------------------------
    # PRICE CANDIDATES / OHLC INFERENCE
    # ------------------------------------------------------------------
    open_series, high_series, low_series, close_series, ohlc_diag, ohlc_warnings = _infer_ohlc(
        numeric_cols, volume_name
    )
    warnings_accum.extend(ohlc_warnings)

    # ------------------------------------------------------------------
    # BUILD INITIAL OUTPUT
    # ------------------------------------------------------------------
    out = pd.DataFrame(index=df.index)
    out["timestamp"] = df["timestamp"]
    out["open"] = pd.to_numeric(open_series, errors="coerce")
    out["high"] = pd.to_numeric(high_series, errors="coerce")
    out["low"] = pd.to_numeric(low_series, errors="coerce")
    out["close"] = pd.to_numeric(close_series, errors="coerce")

    if volume_series is not None:
        out["volume"] = pd.to_numeric(volume_series, errors="coerce").fillna(0).abs()
    else:
        out["volume"] = 0.0

    before_drop = len(out)
    out = out.dropna(subset=["open", "high", "low", "close"])
    dropped_ohlc_rows = before_drop - len(out)
    if dropped_ohlc_rows > 0:
        warnings_accum.append(f"Rows dropped due to invalid OHLC values: {dropped_ohlc_rows}.")
    if out.empty:
        raise ValueError("All rows dropped due to invalid OHLC.")

    # Integrity check: High/Low bounds
    out["high"] = out[["open", "high", "low", "close"]].max(axis=1)
    out["low"] = out[["open", "high", "low", "close"]].min(axis=1)

    out = out.sort_values("timestamp")
    out = out.drop_duplicates(subset=["timestamp"], keep="last")
    out = out.reset_index(drop=True)

    # ------------------------------------------------------------------
    # TIMEFRAME INFERENCE
    # ------------------------------------------------------------------
    freq, tf_diag, tf_warnings = _infer_timeframe(out)
    warnings_accum.extend(tf_warnings)

    # ------------------------------------------------------------------
    # RESAMPLE TO FIXED TIMEFRAME & GAP DETECTION
    # ------------------------------------------------------------------
    out = out.set_index("timestamp")
    full_index = pd.date_range(start=out.index.min(), end=out.index.max(), freq=freq)
    out = out.reindex(full_index)

    # Gaps are where close is NaN after reindex
    is_gap = out["close"].isna()
    out["is_gap"] = is_gap.astype(int)

    # GAP STATS
    quality_score, gap_stats = _compute_gap_stats(out, gap_col="is_gap")
    if quality_score < 0.5:
        msg = f"Low data quality detected ({quality_score:.2f}). {gap_stats}"
        logger.warning(msg)
        warnings_accum.append(msg)

    if gap_stats["total_gaps"] > 0 and gap_stats["gap_ratio"] > 0.5:
        warnings_accum.append(
            f"High gap ratio detected: {gap_stats['gap_ratio']:.4f}"
        )

    # ------------------------------------------------------------------
    # FILL GAPS
    # ------------------------------------------------------------------
    # Prices: forward fill; backfill leading NaNs via open/close fallback
    out["close"] = out["close"].ffill()
    out["open"] = out["open"].ffill().fillna(out["close"])
    out["high"] = out["high"].ffill().fillna(out["close"])
    out["low"] = out["low"].ffill().fillna(out["close"])
    out["close"] = out["close"].fillna(out["open"])

    # Volume: Fill with 0 for gaps
    out["volume"] = out["volume"].fillna(0)

    # Restore timestamp column
    out = out.reset_index().rename(columns={"index": "timestamp"})

    # ------------------------------------------------------------------
    # SYMBOL INFERENCE
    # ------------------------------------------------------------------
    symbol, symbol_info = _infer_symbol(raw_path, freq)

    # ------------------------------------------------------------------
    # VALIDATE OUTPUT
    # ------------------------------------------------------------------
    validate_warnings = _validate_output(out)
    warnings_accum.extend(validate_warnings)

    # ------------------------------------------------------------------
    # SUMMARY
    # ------------------------------------------------------------------
    summary = _summarize_output(out)

    # ------------------------------------------------------------------
    # OUTPUT PATH
    # ------------------------------------------------------------------
    start_ts = out["timestamp"].iloc[0]
    end_ts = out["timestamp"].iloc[-1]

    if output_root is None:
        output_path = raw_path.with_name(raw_path.stem + "_normalized.csv")
    else:
        save_dir = output_root / symbol
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = f"{symbol}.{start_ts:%Y%m%d}-{end_ts:%Y%m%d}.normalized.csv"
        output_path = save_dir / filename

    out.to_csv(output_path, index=False)
    logger.info(f"[normalize] Saved: {output_path} (Quality: {quality_score:.2f})")

    # ------------------------------------------------------------------
    # ASSEMBLE RESULT
    # ------------------------------------------------------------------
    timeframe_info = tf_diag.copy()
    timeframe_info["freq"] = freq

    result = NormalizationResult(
        symbol=symbol,
        timeframe=freq,
        start_timestamp=start_ts,
        end_timestamp=end_ts,
        row_count=len(out),
        output_path=output_path,
        quality_score=quality_score,
        gap_stats=gap_stats,
        headerless_detected=headerless,
        timestamp_inference=ts_diag,
        numeric_columns=num_diag,
        ohlc_source=ohlc_diag,
        volume_source=vol_diag,
        timeframe_info=timeframe_info,
        warnings=warnings_accum,
        summary=summary,
        raw_path=raw_path,
        symbol_info=symbol_info,
    )

    return result


# ======================================================================
# CLASS API
# ======================================================================

class CSVNormalizer:
    """
    Canonical class-based normalizer for Mikebot.
    """

    def __init__(self, output_root: Path):
        self.output_root = output_root
        self.output_root.mkdir(parents=True, exist_ok=True)

    def normalize(self, raw_csv_path: Path) -> Dict[str, Any]:
        result = _normalize_core(raw_csv_path, self.output_root)
        return result.to_dict()


# ======================================================================
# FUNCTIONAL API (CLI + Legacy)
# ======================================================================

def normalize_csv(raw_path: Path) -> Tuple[Path, str]:
    """
    Functional wrapper for CLI and legacy scripts.

    Returns:
        (output_path, symbol)
    """
    result = _normalize_core(raw_path, output_root=None)
    return result.output_path, result.symbol


# ======================================================================
# CLI ENTRYPOINT
# ======================================================================

def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python normalize_csv.py <file.csv>")
        return

    raw_file = Path(sys.argv[1])
    # For CLI, we output to same directory (output_root=None)
    result = _normalize_core(raw_file, output_root=None)

    print(f"Normalized CSV written to: {result.output_path}")
    print(f"Symbol: {result.symbol}")
    print(f"Timeframe: {result.timeframe}")
    print(f"Quality Score: {result.quality_score:.4f}")
    print(f"Gap Stats: {result.gap_stats}")
    print(f"Warnings: {result.warnings}")
    print(f"Summary: {result.summary}")


if __name__ == "__main__":
    main()