# mikebot/data/candle_schema.py

from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

@dataclass
class Candle:
    # Adjust fields to match your CANDLE_SCHEMA.md exactly.
    # This is a typical structure; tweak names/types if your docs differ.
    timestamp: datetime
    symbol: str
    open: float
    high: float
    low: float
    close: float
    volume: float
    timeframe: Optional[str] = None