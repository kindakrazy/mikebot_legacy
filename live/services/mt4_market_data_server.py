from __future__ import annotations

import json
import logging
import socket
import threading
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Tuple

from mikebot.core.candle_engine import CandleEngine

log = logging.getLogger(__name__)


class Mt4MarketDataServer:
    """
    Line-based TCP server that ingests MT4 MarketDataBridgeEA JSON
    into CandleEngine buffers, with basic health tracking per (symbol, timeframe).
    """

    def __init__(
        self,
        candle_engine: CandleEngine,
        host: str = "0.0.0.0",
        port: int = 50010,
    ) -> None:
        self.candle_engine = candle_engine
        self.host = host
        self.port = port
        self._sock: Optional[socket.socket] = None
        self._stop = threading.Event()

        # Health state: (symbol, timeframe) -> dict(hello, last_tick, last_candle, last_update)
        self._alive: Dict[Tuple[str, str], Dict[str, float]] = {}

    # ------------------------------------------------------------------ #
    # Public API                                                         #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        """Start the server in a background thread."""
        t = threading.Thread(target=self._run, name="Mt4MarketDataServer", daemon=True)
        t.start()

    def stop(self) -> None:
        self._stop.set()
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass

    # ------------------------------------------------------------------ #
    # Internal                                                           #
    # ------------------------------------------------------------------ #

    def _run(self) -> None:
        log.info("Mt4MarketDataServer: listening on %s:%d", self.host, self.port)
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((self.host, self.port))
        self._sock.listen(5)

        while not self._stop.is_set():
            try:
                conn, addr = self._sock.accept()
            except OSError:
                break
            log.info("Mt4MarketDataServer: connection from %s", addr)
            threading.Thread(
                target=self._handle_client,
                args=(conn, addr),
                daemon=True,
            ).start()

        log.info("Mt4MarketDataServer: stopped")

    def _handle_client(self, conn: socket.socket, addr) -> None:
        with conn:
            buf = b""
            while not self._stop.is_set():
                try:
                    chunk = conn.recv(4096)
                except OSError:
                    break
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        msg = json.loads(line.decode("utf-8"))
                    except Exception as exc:
                        log.warning("Mt4MarketDataServer: bad JSON from %s: %r", addr, exc)
                        continue
                    self._handle_message(msg)

    # ------------------------------------------------------------------ #
    # Message handling                                                   #
    # ------------------------------------------------------------------ #

    def _handle_message(self, msg: dict) -> None:
        msg_type = msg.get("type")
        if msg_type == "hello":
            self._handle_hello(msg)
        elif msg_type == "history":
            if "candles" in msg:
                self._handle_history_batch(msg)
            else:
                self._handle_single_candle(msg)
        elif msg_type == "candle":
            self._handle_single_candle(msg)
        elif msg_type == "tick":
            self._handle_tick(msg)
        else:
            pass

    def _handle_hello(self, msg: dict) -> None:
        symbol = msg.get("symbol")
        tf = msg.get("timeframe")
        log.info("Mt4MarketDataServer: hello from %s %s", symbol, tf)

        if symbol and tf:
            key = (symbol, tf)
            self._alive[key] = {
                "hello": True,
                "last_tick": 0.0,
                "last_candle": 0.0,
                "last_update": time.time(),
            }

    def _handle_history_batch(self, msg: dict) -> None:
        symbol = msg.get("symbol")
        tf = msg.get("timeframe")
        for c in msg.get("candles", []):
            self._ingest_candle(symbol, tf, c)

    def _handle_single_candle(self, msg: dict) -> None:
        symbol = msg.get("symbol")
        tf = msg.get("timeframe")
        self._ingest_candle(symbol, tf, msg)

    def _handle_tick(self, msg: dict) -> None:
        symbol = msg.get("symbol")
        tf = msg.get("timeframe", "M1")

        if symbol and tf:
            key = (symbol, tf)
            info = self._alive.setdefault(
                key,
                {"hello": False, "last_tick": 0.0, "last_candle": 0.0, "last_update": 0.0},
            )
            info["hello"] = False
            now = time.time()
            info["last_tick"] = now
            info["last_update"] = now

        ts = datetime.fromtimestamp(int(msg["timestamp"]), tz=timezone.utc)
        bid = float(msg["bid"])
        ask = float(msg["ask"])
        mid = (bid + ask) / 2.0
        tick = {
            "open": mid,
            "high": mid,
            "low": mid,
            "close": mid,
            "volume": 0.0,
        }
        self.candle_engine.ingest_live_tick(symbol, tf, tick, ts)

    def _ingest_candle(self, symbol: str, tf: str, c: dict) -> None:
        try:
            if symbol and tf:
                key = (symbol, tf)
                info = self._alive.setdefault(
                    key,
                    {"hello": False, "last_tick": 0.0, "last_candle": 0.0, "last_update": 0.0},
                )
                info["hello"] = False
                now = time.time()
                info["last_candle"] = now
                info["last_update"] = now

            ts = datetime.fromtimestamp(int(c["timestamp"]), tz=timezone.utc)
            tick = {
                "open": float(c["open"]),
                "high": float(c["high"]),
                "low": float(c["low"]),
                "close": float(c["close"]),
                "volume": float(c.get("volume", 0.0)),
            }
            self.candle_engine.ingest_live_tick(symbol, tf, tick, ts)
        except Exception as exc:
            log.exception(
                "Mt4MarketDataServer: failed to ingest candle for %s %s: %r",
                symbol,
                tf,
                exc,
            )

    # ------------------------------------------------------------------ #
    # Health API                                                         #
    # ------------------------------------------------------------------ #

    def market_data_alive(self, symbol: str, timeframe: str) -> bool:
        """
        Returns True if we consider market data for (symbol, timeframe) alive.

        Rules:
          - HELLO counts as alive immediately
          - Any tick/candle in the last N seconds also counts as alive
        """
        key = (symbol, timeframe)
        info = self._alive.get(key)
        if not info:
            return False

        if info.get("hello"):
            return True

        last_update = info.get("last_update") or 0.0

        if last_update and (time.time() - last_update) < 10.0:
            return True

        return False