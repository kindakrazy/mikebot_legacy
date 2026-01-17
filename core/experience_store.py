# mikebot/core/experience_store.py

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, List

import json
import pandas as pd


class ExperienceStore:
    """
    Persistent long-term storage for model experience.

    Responsibilities:
    - Store features, labels, predictions, errors, regimes.
    - Store per-strategy parquet tables.
    - Store metadata entries for each append event.
    - Load full or recent experience.
    - Load multi-timeframe experience (loading only, no feature engineering).
    - Provide a synthetic MULTITF view for symbol-level audits.

    This class performs *no* feature engineering, merging, correlation,
    or multi-timeframe fusion. Those responsibilities belong to
    FeatureAggregator.
    """

    def __init__(self, root_dir: Path) -> None:
        """
        Parameters
        ----------
        root_dir:
            Root directory for experience storage, e.g. mikebot/experience
        """
        self.root_dir = root_dir
        self.root_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # PATH & DISCOVERY HELPERS
    # ------------------------------------------------------------------

    def _symbol_dir(self, symbol: str, timeframe: str) -> Path:
        return self.root_dir / symbol / timeframe

    def _strategy_path(self, symbol: str, timeframe: str, strategy: str) -> Path:
        folder = self._symbol_dir(symbol, timeframe)
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"strategy_{strategy}.parquet"

    def _metadata_path(self, symbol: str, timeframe: str) -> Path:
        folder = self._symbol_dir(symbol, timeframe)
        folder.mkdir(parents=True, exist_ok=True)
        return folder / "metadata.json"

    def _list_timeframes_for_symbol(self, symbol: str) -> List[str]:
        """
        Discover all timeframes that have experience for a symbol.

        Used to build synthetic MULTITF views.
        """
        symbol_root = self.root_dir / symbol
        if not symbol_root.exists() or not symbol_root.is_dir():
            return []
        tfs: List[str] = []
        for child in symbol_root.iterdir():
            if child.is_dir():
                tfs.append(child.name)
        return sorted(tfs)

    # ------------------------------------------------------------------
    # APPEND EXPERIENCE
    # ------------------------------------------------------------------

    def append(
        self,
        symbol: str,
        timeframe: str,
        features: pd.DataFrame,
        labels: Optional[pd.DataFrame] = None,
        predictions: Optional[pd.DataFrame] = None,
        errors: Optional[pd.DataFrame] = None,
        regimes: Optional[pd.DataFrame] = None,
        strategies: Optional[Dict[str, pd.DataFrame]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Append a batch of experience for (symbol, timeframe).

        All DataFrames must share the same index. No alignment or
        deduplication is performed here.
        """
        symbol_dir = self._symbol_dir(symbol, timeframe)
        symbol_dir.mkdir(parents=True, exist_ok=True)

        self._validate_batch_shapes(features, labels, predictions, errors, regimes, strategies)

        self._append_parquet(symbol_dir / "features.parquet", features)

        if labels is not None:
            self._append_parquet(symbol_dir / "labels.parquet", labels)

        if predictions is not None:
            self._append_parquet(symbol_dir / "predictions.parquet", predictions)

        if errors is not None:
            self._append_parquet(symbol_dir / "errors.parquet", errors)

        if regimes is not None:
            self._append_parquet(symbol_dir / "regimes.parquet", regimes)

        if strategies:
            for name, df in strategies.items():
                self._append_parquet(symbol_dir / f"strategy_{name}.parquet", df)

        if metadata is not None:
            self._append_metadata(self._metadata_path(symbol, timeframe), metadata)

    # ------------------------------------------------------------------
    # SYMBOL-LEVEL SAVE HELPERS (USED BY METATRAINER V4)
    # ------------------------------------------------------------------

    def save_predictions(
        self,
        symbol: str,
        timeframe: str,
        model_type: str,
        predictions: pd.DataFrame,
    ) -> None:
        """
        Persist predictions for a given (symbol, timeframe, model_type).

        For MULTITF, this writes into the MULTITF scope directory and
        overwrites the existing predictions file. It does *not* require
        features and does not use append().
        """
        symbol_dir = self._symbol_dir(symbol, timeframe)
        symbol_dir.mkdir(parents=True, exist_ok=True)
        path = symbol_dir / "predictions.parquet"
        predictions.to_parquet(path)

    def save_errors(
        self,
        symbol: str,
        timeframe: str,
        model_type: str,
        errors: pd.DataFrame,
    ) -> None:
        """
        Persist errors for a given (symbol, timeframe, model_type).

        For MULTITF, this writes into the MULTITF scope directory and
        overwrites the existing errors file.
        """
        symbol_dir = self._symbol_dir(symbol, timeframe)
        symbol_dir.mkdir(parents=True, exist_ok=True)
        path = symbol_dir / "errors.parquet"
        errors.to_parquet(path)

    # ------------------------------------------------------------------
    # LOAD EXPERIENCE (SINGLE TF + SYNTHETIC MULTITF)
    # ------------------------------------------------------------------

    def load_all(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Load all stored experience for (symbol, timeframe).

        Special case:
            timeframe == "MULTITF" -> synthetic symbol-level view
            built by concatenating per-timeframe tables.
        """
        if timeframe == "MULTITF":
            return self._load_all_multitf(symbol)

        symbol_dir = self._symbol_dir(symbol, timeframe)

        result: Dict[str, Any] = {
            "features": self._read_parquet_if_exists(symbol_dir / "features.parquet"),
            "labels": self._read_parquet_if_exists(symbol_dir / "labels.parquet"),
            "predictions": self._read_parquet_if_exists(symbol_dir / "predictions.parquet"),
            "errors": self._read_parquet_if_exists(symbol_dir / "errors.parquet"),
            "regimes": self._read_parquet_if_exists(symbol_dir / "regimes.parquet"),
            "strategies": self._load_all_strategies(symbol_dir),
            "metadata": self._read_metadata_if_exists(symbol_dir / "metadata.json"),
        }

        return result

    def _load_all_multitf(self, symbol: str) -> Dict[str, Any]:
        """
        Build a synthetic MULTITF view by concatenating all per-timeframe
        tables for the symbol. Indices are preserved and concatenated.
        """
        tfs = self._list_timeframes_for_symbol(symbol)
        if not tfs:
            return {
                "features": None,
                "labels": None,
                "predictions": None,
                "errors": None,
                "regimes": None,
                "strategies": {},
                "metadata": [],
            }

        features_list: List[pd.DataFrame] = []
        labels_list: List[pd.DataFrame] = []
        preds_list: List[pd.DataFrame] = []
        errors_list: List[pd.DataFrame] = []
        regimes_list: List[pd.DataFrame] = []
        strategies_agg: Dict[str, List[pd.DataFrame]] = {}
        metadata_agg: List[Dict[str, Any]] = []

        for tf in tfs:
            data = self.load_all(symbol, tf)
            f = data.get("features")
            l = data.get("labels")
            p = data.get("predictions")
            e = data.get("errors")
            r = data.get("regimes")
            s = data.get("strategies", {})
            m = data.get("metadata", [])

            if f is not None and not f.empty:
                features_list.append(f)
            if l is not None and not l.empty:
                labels_list.append(l)
            if p is not None and not p.empty:
                preds_list.append(p)
            if e is not None and not e.empty:
                errors_list.append(e)
            if r is not None and not r.empty:
                regimes_list.append(r)

            for name, df in s.items():
                if df is None or df.empty:
                    continue
                strategies_agg.setdefault(name, []).append(df)

            if isinstance(m, list):
                metadata_agg.extend(m)

        def _concat_or_none(dfs: List[pd.DataFrame]) -> Optional[pd.DataFrame]:
            if not dfs:
                return None
            combined = pd.concat(dfs, axis=0)
            combined = combined[~combined.index.duplicated(keep="last")]
            return combined.sort_index()

        strategies_final: Dict[str, pd.DataFrame] = {}
        for name, dfs in strategies_agg.items():
            combined = _concat_or_none(dfs)
            if combined is not None and not combined.empty:
                strategies_final[name] = combined

        return {
            "features": _concat_or_none(features_list),
            "labels": _concat_or_none(labels_list),
            "predictions": _concat_or_none(preds_list),
            "errors": _concat_or_none(errors_list),
            "regimes": _concat_or_none(regimes_list),
            "strategies": strategies_final,
            "metadata": metadata_agg,
        }

    def load_recent(self, symbol: str, timeframe: str, n_rows: int) -> Dict[str, Any]:
        """
        Load the most recent n_rows of experience.

        Special case:
            timeframe == "MULTITF" -> recent rows from the synthetic
            symbol-level concatenation across all timeframes.
        """
        if timeframe == "MULTITF":
            return self._load_recent_multitf(symbol, n_rows)

        all_data = self.load_all(symbol, timeframe)

        def tail(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
            if df is None:
                return None
            return df.tail(n_rows)

        return {
            "features": tail(all_data["features"]),
            "labels": tail(all_data["labels"]),
            "predictions": tail(all_data["predictions"]),
            "errors": tail(all_data["errors"]),
            "regimes": tail(all_data["regimes"]),
            "strategies": {name: tail(df) for name, df in all_data["strategies"].items()},
            "metadata": all_data["metadata"],
        }

    def _load_recent_multitf(self, symbol: str, n_rows: int) -> Dict[str, Any]:
        """
        Build a recent MULTITF view by concatenating all per-timeframe
        tables and then taking the last n_rows by index.
        """
        all_data = self._load_all_multitf(symbol)

        def tail(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
            if df is None:
                return None
            return df.tail(n_rows)

        return {
            "features": tail(all_data["features"]),
            "labels": tail(all_data["labels"]),
            "predictions": tail(all_data["predictions"]),
            "errors": tail(all_data["errors"]),
            "regimes": tail(all_data["regimes"]),
            "strategies": {name: tail(df) for name, df in all_data["strategies"].items()},
            "metadata": all_data["metadata"],
        }

    def load_errors(self, symbol: str, timeframe: str, abs_error_threshold: float) -> Dict[str, Any]:
        """
        Load only samples where |error| >= threshold.

        For MULTITF, this operates on the synthetic MULTITF view.
        """
        all_data = self.load_all(symbol, timeframe)
        errors_df = all_data["errors"]

        if errors_df is None or errors_df.empty:
            return {**all_data, "features": None}

        numeric_cols = errors_df.select_dtypes(include=["number"]).columns
        if len(numeric_cols) == 0:
            return {**all_data, "features": None}

        col = numeric_cols[0]
        mask = errors_df[col].abs() >= abs_error_threshold

        def filt(df: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
            if df is None or df.empty:
                return None
            common_index = df.index.intersection(errors_df.index)
            df = df.loc[common_index]
            return df.loc[mask.reindex(common_index, fill_value=False)]

        return {
            "features": filt(all_data["features"]),
            "labels": filt(all_data["labels"]),
            "predictions": filt(all_data["predictions"]),
            "errors": filt(all_data["errors"]),
            "regimes": filt(all_data["regimes"]),
            "strategies": {name: filt(df) for name, df in all_data["strategies"].items()},
            "metadata": all_data["metadata"],
        }

    def load_strategy(self, symbol: str, timeframe: str, strategy: str) -> Optional[pd.DataFrame]:
        """
        Load a single strategy's stored signals/outcomes.
        """
        return self._read_parquet_if_exists(self._strategy_path(symbol, timeframe, strategy))

    def load_regime(self, symbol: str, timeframe: str, regime: str) -> Optional[pd.DataFrame]:
        """
        Load rows where the regime column equals the given regime.
        """
        all_data = self.load_all(symbol, timeframe)
        regimes_df = all_data["regimes"]
        if regimes_df is None or regimes_df.empty:
            return None
        return regimes_df[regimes_df == regime]

    # ------------------------------------------------------------------
    # MULTI-TIMEFRAME LOADING (NO MERGING)
    # ------------------------------------------------------------------

    def load_multi_tf(self, symbol: str, timeframes: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Load full experience for multiple timeframes.
        """
        return {tf: self.load_all(symbol, tf) for tf in timeframes}

    def load_multi_tf_features(self, symbol: str, timeframes: List[str]) -> Dict[str, pd.DataFrame]:
        """
        Load only the feature tables for multiple timeframes.
        """
        out: Dict[str, pd.DataFrame] = {}
        for tf in timeframes:
            data = self.load_all(symbol, tf)
            out[tf] = data.get("features", None)
        return out

    def load_multi_tf_recent(self, symbol: str, timeframes: List[str], n_rows: int = 1000) -> Dict[str, pd.DataFrame]:
        """
        Load recent feature rows for multiple timeframes.
        """
        out: Dict[str, pd.DataFrame] = {}
        for tf in timeframes:
            data = self.load_recent(symbol, tf, n_rows)
            out[tf] = data.get("features", None)
        return out

    # ------------------------------------------------------------------
    # PRUNING / COMPACTION
    # ------------------------------------------------------------------

    def prune(self, symbol: str, timeframe: str, max_rows: int) -> None:
        """
        Keep only the last max_rows rows for each core table.
        """
        core_tables = ["features", "labels", "predictions", "errors", "regimes"]
        for t in core_tables:
            path = self._symbol_dir(symbol, timeframe) / f"{t}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                if len(df) > max_rows:
                    df.tail(max_rows).to_parquet(path)

    def compact(self, symbol: str, timeframe: str) -> None:
        """
        Re-write parquet files to ensure compression and remove fragmentation.
        """
        core_tables = ["features", "labels", "predictions", "errors", "regimes"]
        for t in core_tables:
            path = self._symbol_dir(symbol, timeframe) / f"{t}.parquet"
            if path.exists():
                df = pd.read_parquet(path)
                df.to_parquet(path)

    # ------------------------------------------------------------------
    # STATS
    # ------------------------------------------------------------------

    def stats(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """
        Return simple statistics about stored experience.

        For MULTITF, this reports counts from the synthetic MULTITF view.
        """
        data = self.load_all(symbol, timeframe)

        def n(df: Optional[pd.DataFrame]) -> int:
            return 0 if df is None else len(df)

        stats = {
            "features_rows": n(data["features"]),
            "labels_rows": n(data["labels"]),
            "predictions_rows": n(data["predictions"]),
            "errors_rows": n(data["errors"]),
            "regimes_rows": n(data["regimes"]),
            "strategy_counts": {name: n(df) for name, df in data["strategies"].items()},
            "metadata_entries": len(data["metadata"]),
        }

        return stats

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_batch_shapes(
        features: pd.DataFrame,
        labels: Optional[pd.DataFrame],
        predictions: Optional[pd.DataFrame],
        errors: Optional[pd.DataFrame],
        regimes: Optional[pd.DataFrame],
        strategies: Optional[Dict[str, pd.DataFrame]],
    ) -> None:
        n = len(features)
        for name, df in [
            ("labels", labels),
            ("predictions", predictions),
            ("errors", errors),
            ("regimes", regimes),
        ]:
            if df is not None and len(df) != n:
                raise ValueError(
                    f"ExperienceStore.append: length mismatch between features ({n}) and {name} ({len(df)})"
                )

        if strategies:
            for strat_name, df in strategies.items():
                if len(df) != n:
                    raise ValueError(
                        f"ExperienceStore.append: length mismatch between features ({n}) and strategy '{strat_name}' ({len(df)})"
                    )

    @staticmethod
    def _append_parquet(path: Path, batch: pd.DataFrame) -> None:
        if path.exists():
            existing = pd.read_parquet(path)
            combined = pd.concat([existing, batch], axis=0)
        else:
            combined = batch
        combined.to_parquet(path)

    @staticmethod
    def _read_parquet_if_exists(path: Path) -> Optional[pd.DataFrame]:
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        return df if not df.empty else None

    @staticmethod
    def _append_metadata(path: Path, metadata_entry: Dict[str, Any]) -> None:
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, list):
                    data = []
            except json.JSONDecodeError:
                data = []
        else:
            data = []

        data.append(metadata_entry)

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, sort_keys=True)

    @staticmethod
    def _read_metadata_if_exists(path: Path) -> List[Dict[str, Any]]:
        if not path.exists():
            return []
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, list) else []
        except json.JSONDecodeError:
            return []

    @staticmethod
    def _load_all_strategies(symbol_dir: Path) -> Dict[str, pd.DataFrame]:
        strategies: Dict[str, pd.DataFrame] = {}
        if not symbol_dir.exists():
            return strategies

        for path in symbol_dir.glob("strategy_*.parquet"):
            name = path.stem[len("strategy_"):]
            df = pd.read_parquet(path)
            if not df.empty:
                strategies[name] = df

        return strategies
    # --- Global accessor for ExperienceStoreV4 ---

    _experience_store_global = None


    def set_global_experience_store(store) -> None:
        """
        Register the process-wide ExperienceStoreV4 instance.
        """
        global _experience_store_global
        _experience_store_global = store


    def get_global_experience_store():
        """
        Return the process-wide ExperienceStoreV4 instance, or None if not set.
        """
        return _experience_store_global