# C:\mikebot\live\services\mt4_status_bridge.py

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class MT4StatusConfig:
    heartbeat_timeout_seconds: int = 30


class MT4StatusBridge:
    """
    Tracks MT4 bridge heartbeat and connectivity.

    Distillation of HighStrike's mt4_status_bridge + heartbeat_check.
    """

    def __init__(self, config: MT4StatusConfig) -> None:
        self.config = config
        self._last_heartbeat: Optional[datetime] = None

    def record_heartbeat(self, ts: datetime) -> None:
        self._last_heartbeat = ts

    def is_alive(self, now: datetime) -> bool:
        if self._last_heartbeat is None:
            return False
        return (now - self._last_heartbeat) <= timedelta(seconds=self.config.heartbeat_timeout_seconds)

    def snapshot(self, now: datetime) -> dict:
        return {
            "alive": self.is_alive(now),
            "last_heartbeat": self._last_heartbeat.isoformat() if self._last_heartbeat else None,
        }
