#"C:\mikebot\candle_stream.py"
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Iterable, Iterator, Optional, Sequence, List

from .candle_schema import Candle


class CandleStream(ABC):
    """
    Unified interface for live and historical candle streams.
    Implementations must be deterministic and ordered by timestamp.
    """

    @abstractmethod
    def __iter__(self) -> Iterator[Candle]:
        ...

    @abstractmethod
    def next(self) -> Optional[Candle]:
        """
        Return the next candle in the stream, or None if exhausted.
        """
        ...

    @abstractmethod
    def seek(self, timestamp: datetime) -> None:
        """
        Move the internal cursor to the first candle >= timestamp.
        """
        ...

    @abstractmethod
    def lookback(self, n: int) -> Sequence[Candle]:
        """
        Return up to n most recent candles BEFORE the current cursor position.
        Most recent last.
        """
        ...


class BufferedCandleStream(CandleStream):
    """
    Optional helper base for implementations that want built-in lookback buffering.
    """

    def __init__(self, source: Iterable[Candle], max_buffer: int = 1024) -> None:
        self._source_iter = iter(source)
        self._buffer: List[Candle] = []
        self._max_buffer = max_buffer
        self._exhausted = False

    def __iter__(self) -> Iterator[Candle]:
        while True:
            c = self.next()
            if c is None:
                break
            yield c

    def next(self) -> Optional[Candle]:
        if self._exhausted:
            return None
        try:
            candle = next(self._source_iter)
        except StopIteration:
            self._exhausted = True
            return None

        self._buffer.append(candle)
        if len(self._buffer) > self._max_buffer:
            self._buffer.pop(0)
        return candle

    def seek(self, timestamp: datetime) -> None:
        self._buffer.clear()
        self._exhausted = False
        # Default implementation: consume until >= timestamp
        for candle in self._source_iter:
            if candle.timestamp >= timestamp:
                self._buffer.append(candle)
                break

    def lookback(self, n: int) -> Sequence[Candle]:
        if n <= 0:
            return ()
        return self._buffer[-n:]