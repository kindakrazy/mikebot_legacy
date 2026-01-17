#mikebot/features/feature_builder_v4.py

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Any, Iterable, Iterator, List, Optional, Sequence, Tuple

from mikebot.data.candle_schema import Candle
from mikebot.data.stream_interface import CandleStream
from mikebot.features.feature_interface import Feature, FeatureState


class FeatureBuilderV4:
    """
    Streaming feature builder for v4.

    - Consumes a CandleStream
    - Maintains a rolling history buffer
    - Manages per-feature state
    - Respects per-feature lookback requirements
    - Produces a dict[str, Any] of features per candle
    """

    def __init__(
        self,
        features: Iterable[Feature],
        max_history: int = 2048,
    ) -> None:
        self._features: List[Feature] = list(features)
        self._max_history = max_history

        # Per-feature state
        self._states: Dict[str, Optional[FeatureState]] = {}
        for f in self._features:
            self._states[f.name] = f.init_state()

        # Rolling history buffer (oldest first)
        self._history: Deque[Candle] = deque(maxlen=max_history)

    def reset(self) -> None:
        """
        Reset history and feature states.
        """
        self._history.clear()
        for f in self._features:
            self._states[f.name] = f.init_state()

    def process_stream(
        self,
        stream: CandleStream,
    ) -> Iterator[Tuple[Candle, Dict[str, Any]]]:
        """
        Iterate over a CandleStream and yield (candle, feature_dict).
        """
        for candle in stream:
            features = self.process_candle(candle)
            yield candle, features

    def process_candle(self, candle: Candle) -> Dict[str, Any]:
        """
        Process a single candle and return a feature dict.
        """
        history: Sequence[Candle] = tuple(self._history)
        feature_values: Dict[str, Any] = {}

        for f in self._features:
            state = self._states.get(f.name)
            lookback = f.requires_lookback

            if lookback > 0 and len(history) < lookback:
                # Not enough history yet; skip this feature
                continue

            values = f.compute(
                candle=candle,
                history=history,
                state=state,
            )
            if values:
                feature_values.update(values)

        # Update history after computing features
        self._history.append(candle)
        return feature_values