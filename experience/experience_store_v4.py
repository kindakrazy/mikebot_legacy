# mikebot/experience/experience_store_v4.py

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class ExperienceV4:
    """
    A single experience tuple for v4.
    """
    features: Dict[str, Any]
    target: Optional[float] = None
    reward: Optional[float] = None
    metadata: Dict[str, Any] = None


class ExperienceStoreV4:
    """
    Streaming-friendly experience buffer for v4.
    """

    def __init__(
        self,
        *,
        capacity: Optional[int] = None,
        store_targets: bool = True,
        store_rewards: bool = True,
    ) -> None:
        self.capacity = capacity
        self.store_targets = store_targets
        self.store_rewards = store_rewards

        self._buffer: List[ExperienceV4] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, exp: ExperienceV4) -> None:
        if self.capacity is not None and len(self._buffer) >= self.capacity:
            self._buffer.pop(0)
        self._buffer.append(exp)

    def add_many(self, experiences: Sequence[ExperienceV4]) -> None:
        for exp in experiences:
            self.add(exp)

    def __len__(self) -> int:
        return len(self._buffer)

    def clear(self) -> None:
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def sample(self, n: int) -> List[ExperienceV4]:
        if n <= 0:
            return []
        if n >= len(self._buffer):
            return list(self._buffer)
        idx = np.random.choice(len(self._buffer), size=n, replace=False)
        return [self._buffer[i] for i in idx]

    def tail(self, n: int) -> List[ExperienceV4]:
        if n <= 0:
            return []
        return self._buffer[-n:]

    # ------------------------------------------------------------------
    # Export utilities
    # ------------------------------------------------------------------

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for exp in self._buffer:
            row = {}
            for k, v in exp.features.items():
                row[f"feature__{k}"] = v

            if self.store_targets:
                row["target"] = exp.target

            if self.store_rewards:
                row["reward"] = exp.reward

            if exp.metadata:
                for mk, mv in exp.metadata.items():
                    row[f"meta__{mk}"] = mv

            rows.append(row)

        if not rows:
            return pd.DataFrame()

        return pd.DataFrame(rows)

    def export_features_and_targets(
        self,
    ) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
        df = self.to_dataframe()
        if df.empty:
            return df, None

        feature_cols = [c for c in df.columns if c.startswith("feature__")]
        X = df[feature_cols]

        y = None
        if self.store_targets and "target" in df.columns:
            y = df["target"]

        return X, y


# ---------------------------------------------------------------------------
# Global accessor for ExperienceStoreV4 (required by cockpit)
# ---------------------------------------------------------------------------

_experience_store_global: Optional[ExperienceStoreV4] = None


def set_global_experience_store(store: ExperienceStoreV4) -> None:
    global _experience_store_global
    _experience_store_global = store


def get_global_experience_store() -> Optional[ExperienceStoreV4]:
    return _experience_store_global
