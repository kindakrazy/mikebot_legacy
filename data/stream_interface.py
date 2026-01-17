from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Iterator
from mikebot.data.candle_schema import Candle

@dataclass
class CandleStream:
    candles: Iterable[Candle]

    def __iter__(self) -> Iterator[Candle]:
        return iter(self.candles)