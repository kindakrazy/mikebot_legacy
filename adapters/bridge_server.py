# C:\mikebot\mikebot\adapters\bridge_server.py

import asyncio
import json
import traceback
from datetime import datetime, timezone
from typing import Dict, Tuple, Callable, Any, Optional, List

try:
    from mikebot.core.candle_engine import CandleEngine, SymbolRegistry
except ImportError:
    CandleEngine = None
    SymbolRegistry = None


class MikebotBridgeServer:
    """
    Central TCP server that all MT4 EAs connect to.

    Handshake v2:
      - EAs send HELLO repeatedly until hello_ack
      - Server replies hello_ack + server_ready (every time it sees HELLO)
      - Heartbeats + ping/pong for liveness

    Health model:
      - execution_alive / market_data_alive: connection presence
      - heartbeat_fresh: last heartbeat age per (role, symbol, timeframe)
      - execution_healthy / market_data_healthy: connection + fresh heartbeat
      - health_snapshot(): aggregate for /bridge/health and guardrails

    Sync model (execution):
      - On execution HELLO, server:
          * registers EA
          * sends hello_ack + server_ready
          * immediately requests account + positions snapshots (per magic)
      - Tracks pending snapshot correlation_ids
      - When both snapshots arrive, marks (symbol, timeframe) as SYNCED
      - Emits execution_sync_complete events to subscribers
    """

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 50010,
        exec_port: Optional[int] = None,
    ):
        self.host = host
        self.port = port
        self.exec_port = exec_port or port

        # execution_eas: symbol -> writer
        self.execution_eas: Dict[str, asyncio.StreamWriter] = {}
        # data_eas: (symbol, timeframe) -> writer
        self.data_eas: Dict[Tuple[str, str], asyncio.StreamWriter] = {}

        self.candle_subs: Dict[Tuple[str, str], List[Callable[[dict], None]]] = {}
        self.tick_subs: Dict[str, List[Callable[[dict], None]]] = {}
        self.exec_event_subs: List[Callable[[dict], None]] = []
        self.history_subs: List[Callable[[dict], None]] = []

        self.pending_history: Dict[str, asyncio.Future] = {}
        self.candle_engine: Optional[CandleEngine] = None

        # Heartbeat tracking: key = "role:symbol:timeframe" -> datetime
        self.last_heartbeat: Dict[str, datetime] = {}

        # Execution sync state:
        #   exec_state: (symbol, timeframe, magic) -> dict with pending CIDs, flags, timestamps
        #   exec_synced: (symbol, timeframe) -> bool
        self.exec_state: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
        self.exec_synced: Dict[Tuple[str, str], bool] = {}

        # Sync event subscribers (orchestrator, etc.)
        self.sync_event_subs: List[Callable[[dict], None]] = []

    # ------------------------------------------------------------
    # Health checks (connection-level)
    # ------------------------------------------------------------

    def execution_alive(self, symbol: str) -> bool:
        return symbol in self.execution_eas

    def market_data_alive(self, symbol: str, timeframe: str) -> bool:
        return (symbol, timeframe) in self.data_eas

    # ------------------------------------------------------------
    # Heartbeat-based health
    # ------------------------------------------------------------

    def _hb_key(self, role: str, symbol: str, timeframe: str) -> str:
        return f"{role}:{symbol}:{timeframe}"

    def heartbeat_fresh(
        self,
        role: str,
        symbol: str,
        timeframe: str,
        max_age: int = 10,
    ) -> bool:
        """
        Returns True if we have a recent heartbeat for (role, symbol, timeframe).
        max_age is in seconds.
        """
        key = self._hb_key(role, symbol, timeframe)
        ts = self.last_heartbeat.get(key)
        if not ts:
            return False
        age = (datetime.now(timezone.utc) - ts).total_seconds()
        return age <= max_age

    def execution_healthy(
        self,
        symbol: str,
        timeframe: str,
        max_age: int = 10,
    ) -> bool:
        """
        Execution is healthy if:
          - execution EA is connected for symbol
          - heartbeat from role=execution is fresh for (symbol, timeframe)
        """
        if not self.execution_alive(symbol):
            return False
        return self.heartbeat_fresh("execution", symbol, timeframe, max_age=max_age)

    def market_data_healthy(
        self,
        symbol: str,
        timeframe: str,
        max_age: int = 10,
    ) -> bool:
        """
        Market data is healthy if:
          - data EA is connected for (symbol, timeframe)
          - heartbeat from role=market_data is fresh for (symbol, timeframe)
        """
        if not self.market_data_alive(symbol, timeframe):
            return False
        return self.heartbeat_fresh("market_data", symbol, timeframe, max_age=max_age)

    def health_snapshot(self, max_age: int = 10) -> Dict[str, Any]:
        """
        Aggregate health snapshot for /bridge/health and guardrails.

        Returns structure like:
        {
          "execution": {
            "BTCUSD": {
              "connected": true,
              "fresh_heartbeat": true,
              "healthy": true,
              "timeframes": ["M5"]
            },
            ...
          },
          "market_data": {
            "BTCUSD_M5": {
              "connected": true,
              "fresh_heartbeat": true,
              "healthy": true,
              "symbol": "BTCUSD",
              "timeframe": "M5"
            },
            ...
          }
        }
        """
        exec_info: Dict[str, Dict[str, Any]] = {}
        for symbol in list(self.execution_eas.keys()):
            tf_set = set()
            for key, ts in self.last_heartbeat.items():
                role, sym, tf = key.split(":", 2)
                if role == "execution" and sym == symbol:
                    tf_set.add(tf)

            if not tf_set:
                exec_info[symbol] = {
                    "connected": True,
                    "fresh_heartbeat": False,
                    "healthy": False,
                    "timeframes": [],
                }
            else:
                fresh_any = any(
                    self.heartbeat_fresh("execution", symbol, tf, max_age=max_age)
                    for tf in tf_set
                )
                exec_info[symbol] = {
                    "connected": True,
                    "fresh_heartbeat": fresh_any,
                    "healthy": fresh_any,
                    "timeframes": sorted(tf_set),
                }

        md_info: Dict[str, Dict[str, Any]] = {}
        for (symbol, timeframe) in list(self.data_eas.keys()):
            key = f"{symbol}_{timeframe}"
            fresh = self.heartbeat_fresh("market_data", symbol, timeframe, max_age=max_age)
            md_info[key] = {
                "connected": True,
                "fresh_heartbeat": fresh,
                "healthy": fresh,
                "symbol": symbol,
                "timeframe": timeframe,
            }

        return {
            "execution": exec_info,
            "market_data": md_info,
        }

    # ------------------------------------------------------------
    # Sync state (execution)
    # ------------------------------------------------------------

    def is_execution_synced(self, symbol: str, timeframe: str) -> bool:
        """
        Returns True if execution for (symbol, timeframe) has completed
        the startup sync (account + positions snapshots).
        """
        return self.exec_synced.get((symbol, timeframe), False)

    def on_sync_event(self, callback: Callable[[dict], None]) -> None:
        """
        Subscribe to sync events, e.g.:

          {"type": "execution_sync_complete",
           "symbol": "...",
           "timeframe": "..."}

        Orchestrator can use this to mark OrderRouter / StrategyEngine as ready.
        """
        self.sync_event_subs.append(callback)
        print("[Bridge] Sync event subscriber registered")

    async def _request_execution_sync(
        self,
        symbol: str,
        timeframe: Optional[str],
        magic: Optional[int],
    ) -> None:
        """
        Request account + positions snapshots from the execution EA
        for the given (symbol, timeframe, magic).

        This is called on initial HELLO and on re-HELLO for execution role.
        """
        if timeframe is None:
            timeframe = "UNKNOWN"

        if magic is None:
            magic = 0

        key3 = (symbol, timeframe, int(magic))
        key2 = (symbol, timeframe)

        writer = self.execution_eas.get(symbol)
        if not writer:
            print(f"[Bridge] Cannot request sync: no execution EA for {symbol}")
            return

        # Mark as not synced yet
        self.exec_synced[key2] = False

        cid_acc = f"sync_{symbol}_{timeframe}_{magic}_acc"
        cid_pos = f"sync_{symbol}_{timeframe}_{magic}_pos"

        state = self.exec_state.setdefault(key3, {})
        state["state"] = "SYNCING"
        state["last_sync_request_ts"] = datetime.now(timezone.utc)
        state["pending"] = {
            "acc_cid": cid_acc,
            "pos_cid": cid_pos,
            "acc_seen": False,
            "pos_seen": False,
        }

        cmds = [
            {
                "type": "cmd",
                "cmd": "account_snapshot",
                "symbol": symbol,
                "magic": magic,
                "correlation_id": cid_acc,
            },
            {
                "type": "cmd",
                "cmd": "positions_snapshot",
                "symbol": symbol,
                "magic": magic,
                "correlation_id": cid_pos,
            },
        ]

        for cmd in cmds:
            writer.write((json.dumps(cmd) + "\n").encode("utf-8"))

        await writer.drain()
        print(f"[Bridge] Requested execution sync for {symbol} {timeframe} magic={magic}")

    def _on_exec_snapshot(
        self,
        msg: dict,
        role: Optional[str],
        symbol: Optional[str],
        timeframe: Optional[str],
    ) -> None:
        """
        Called when an 'account' or 'positions' message arrives from an execution EA.
        Updates sync state and emits execution_sync_complete when both snapshots are seen.
        """
        if role != "execution":
            return

        if symbol is None or timeframe is None:
            return

        magic = msg.get("magic")
        cid = msg.get("correlation_id")

        if magic is None or cid is None:
            return

        key3 = (symbol, timeframe, int(magic))
        key2 = (symbol, timeframe)

        state = self.exec_state.get(key3)
        if not state:
            # No pending sync for this triple; ignore for sync purposes
            return

        pending = state.get("pending")
        if not pending:
            # Already completed or no pending CIDs
            return

        acc_cid = pending.get("acc_cid")
        pos_cid = pending.get("pos_cid")
        acc_seen = pending.get("acc_seen", False)
        pos_seen = pending.get("pos_seen", False)

        if cid == acc_cid:
            acc_seen = True
        if cid == pos_cid:
            pos_seen = True

        pending["acc_seen"] = acc_seen
        pending["pos_seen"] = pos_seen

        # If both snapshots have been seen, mark synced and emit event
        if acc_seen and pos_seen:
            state["state"] = "SYNCED"
            state["last_sync_success_ts"] = datetime.now(timezone.utc)
            state["pending"] = None

            self.exec_synced[key2] = True

            event = {
                "type": "execution_sync_complete",
                "symbol": symbol,
                "timeframe": timeframe,
            }
            print(f"[Bridge] Execution sync complete for {symbol} {timeframe}")

            for cb in self.sync_event_subs:
                try:
                    cb(event)
                except Exception:
                    traceback.print_exc()

    # ------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------

    def attach_candle_engine(self, engine: Any) -> None:
        self.candle_engine = engine
        print("[Bridge] CandleEngine attached")

    def subscribe_candles(self, symbol: str, timeframe: str, callback: Callable[[dict], None]) -> None:
        key = (symbol, timeframe)
        self.candle_subs.setdefault(key, []).append(callback)
        print(f"[Bridge] Subscribed to candles: {symbol} {timeframe}")

    def subscribe_ticks(self, symbol: str, callback: Callable[[dict], None]) -> None:
        self.tick_subs.setdefault(symbol, []).append(callback)
        print(f"[Bridge] Subscribed to ticks: {symbol}")

    def on_execution_event(self, callback: Callable[[dict], None]) -> None:
        self.exec_event_subs.append(callback)
        print("[Bridge] Execution event subscriber registered")

    def on_history(self, callback: Callable[[dict], None]) -> None:
        self.history_subs.append(callback)
        print("[Bridge] History subscriber registered")

    async def send_command(self, symbol: str, cmd: dict) -> None:
        writer = self.execution_eas.get(symbol)
        if not writer:
            raise RuntimeError(f"No execution EA connected for symbol {symbol}")

        writer.write((json.dumps(cmd) + "\n").encode("utf-8"))
        await writer.drain()
        print(f"[Bridge] Sent command to {symbol}: {cmd}")

    async def request_history(self, symbol: str, timeframe: str, bars: int = 500) -> list:
        key = (symbol, timeframe)
        writer = self.data_eas.get(key)
        if not writer:
            raise RuntimeError(f"No data EA connected for {symbol} {timeframe}")

        correlation_id = f"hist_{symbol}_{timeframe}_{id(self)}_{datetime.now(timezone.utc).timestamp()}"
        fut = asyncio.get_event_loop().create_future()
        self.pending_history[correlation_id] = fut

        cmd = {
            "type": "cmd",
            "cmd": "get_history",
            "symbol": symbol,
            "timeframe": timeframe,
            "bars": bars,
            "correlation_id": correlation_id,
        }

        writer.write((json.dumps(cmd) + "\n").encode("utf-8"))
        await writer.drain()
        print(f"[Bridge] Requested history: {symbol} {timeframe} bars={bars} cid={correlation_id}")

        return await fut

    # ------------------------------------------------------------
    # Server loop
    # ------------------------------------------------------------

    async def start(self):
        server_main = await asyncio.start_server(self.handle_client, self.host, self.port)
        servers = [server_main]

        if self.exec_port != self.port:
            server_exec = await asyncio.start_server(self.handle_client, self.host, self.exec_port)
            servers.append(server_exec)

        print(f"[Bridge] Listening on {self.host}:{self.port}")
        if len(servers) > 1:
            print(f"[Bridge] Listening on {self.host}:{self.exec_port} (execution)")

        for s in servers:
            await s.start_serving()

        asyncio.create_task(self._ping_loop())

        await asyncio.Event().wait()

    async def _ping_loop(self) -> None:
        while True:
            await asyncio.sleep(10)
            writers = list(self.execution_eas.values()) + list(self.data_eas.values())
            for w in writers:
                try:
                    w.write(b'{"type":"ping"}\n')
                    await w.drain()
                except Exception:
                    continue

    # ------------------------------------------------------------
    # Client handler
    # ------------------------------------------------------------

    async def _send_hello_ack_and_ready(
        self,
        writer: asyncio.StreamWriter,
        role: str,
        symbol: str,
        timeframe: Optional[str],
    ) -> None:
        ack = {
            "type": "hello_ack",
            "role": role,
            "symbol": symbol,
            "timeframe": timeframe,
            "status": "ok",
        }
        writer.write((json.dumps(ack) + "\n").encode("utf-8"))

        server_ready = {
            "type": "server_ready",
            "role": role,
            "symbol": symbol,
            "timeframe": timeframe,
        }
        writer.write((json.dumps(server_ready) + "\n").encode("utf-8"))
        await writer.drain()

    async def handle_client(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        addr = writer.get_extra_info("peername")
        print(f"[Bridge] Connection from {addr}")

        role: Optional[str] = None
        symbol: Optional[str] = None
        timeframe: Optional[str] = None
        magic: Optional[int] = None

        try:
            # Initial HELLO
            hello_line = await reader.readline()
            if not hello_line:
                print("[Bridge] Empty hello, closing")
                return

            try:
                hello = json.loads(hello_line.decode("utf-8").strip())
            except json.JSONDecodeError as e:
                print("[Bridge] HELLO JSON decode error:", e, "raw:", hello_line)
                return

            print("[Bridge] HELLO:", hello)

            if hello.get("type") != "hello":
                print("[Bridge] Invalid hello message")
                return

            role = hello.get("role")
            symbol = hello.get("symbol")
            timeframe = hello.get("timeframe")
            magic_val = hello.get("magic")
            if isinstance(magic_val, int):
                magic = magic_val
            elif isinstance(magic_val, str):
                try:
                    magic = int(magic_val)
                except ValueError:
                    magic = None

            if role == "execution":
                if symbol:
                    self.execution_eas[symbol] = writer
                    print(f"[Bridge] Execution EA registered for {symbol}")
            elif role == "market_data":
                if symbol and timeframe:
                    key = (symbol, timeframe)
                    self.data_eas[key] = writer
                    print(f"[Bridge] Data EA registered for {symbol} {timeframe}")
            else:
                print("[Bridge] Unknown role:", role)
                return

            await self._send_hello_ack_and_ready(writer, role, symbol, timeframe)

            # For execution role, immediately request sync
            if role == "execution" and symbol:
                asyncio.create_task(self._request_execution_sync(symbol, timeframe, magic))

            while True:
                line = await reader.readline()
                if not line:
                    print(f"[Bridge] EOF from {addr}")
                    break

                try:
                    msg = json.loads(line.decode("utf-8").strip())
                except json.JSONDecodeError as e:
                    print("[Bridge] JSON decode error:", e, "raw:", line)
                    continue

                await self.handle_message(
                    msg,
                    role=role,
                    symbol=symbol,
                    timeframe=timeframe,
                    writer=writer,
                )

        except Exception:
            traceback.print_exc()

        finally:
            print(f"[Bridge] Disconnect {addr}")
            if role == "execution" and symbol in self.execution_eas:
                del self.execution_eas[symbol]
            if role == "market_data" and symbol and timeframe and (symbol, timeframe) in self.data_eas:
                del self.data_eas[(symbol, timeframe)]
            writer.close()
            await writer.wait_closed()

    # ------------------------------------------------------------
    # Message routing
    # ------------------------------------------------------------

    async def handle_message(
        self,
        msg: dict,
        role: Optional[str] = None,
        symbol: Optional[str] = None,
        timeframe: Optional[str] = None,
        writer: Optional[asyncio.StreamWriter] = None,
    ) -> None:
        mtype = msg.get("type")
        print("[Bridge] RAW:", msg)

        # HELLO (late / repeated) — always ACK + READY
        if mtype == "hello":
            r = msg.get("role", role)
            s = msg.get("symbol", symbol)
            tf = msg.get("timeframe", timeframe)
            magic_val = msg.get("magic")
            magic: Optional[int] = None
            if isinstance(magic_val, int):
                magic = magic_val
            elif isinstance(magic_val, str):
                try:
                    magic = int(magic_val)
                except ValueError:
                    magic = None

            if r == "execution" and s:
                self.execution_eas[s] = writer
                print(f"[Bridge] Execution EA re-registered for {s}")
                # Reset sync state and request sync again
                asyncio.create_task(self._request_execution_sync(s, tf, magic))
            elif r == "market_data" and s and tf:
                self.data_eas[(s, tf)] = writer
                print(f"[Bridge] Data EA re-registered for {s} {tf}")

            if writer and r and s:
                await self._send_hello_ack_and_ready(writer, r, s, tf)
            return

        # Heartbeat
        if mtype == "heartbeat":
            r = msg.get("role", role)
            s = msg.get("symbol", symbol)
            tf = msg.get("timeframe", timeframe)
            if r and s and tf:
                key = self._hb_key(r, s, tf)
                self.last_heartbeat[key] = datetime.now(timezone.utc)
            return

        # Pong (treat as heartbeat)
        if mtype == "pong":
            r = msg.get("role", role)
            s = msg.get("symbol", symbol)
            tf = msg.get("timeframe", timeframe)
            if r and s and tf:
                key = self._hb_key(r, s, tf)
                self.last_heartbeat[key] = datetime.now(timezone.utc)
            return

        if mtype == "candle":
            key = (msg["symbol"], msg["timeframe"])
            for cb in self.candle_subs.get(key, []):
                try:
                    cb(msg)
                except Exception:
                    traceback.print_exc()

            if self.candle_engine is not None:
                try:
                    self._ingest_candle_into_engine(msg)
                except Exception:
                    traceback.print_exc()
            return

        if mtype == "tick":
            for cb in self.tick_subs.get(msg["symbol"], []):
                try:
                    cb(msg)
                except Exception:
                    traceback.print_exc()
            return

        if mtype == "event":
            for cb in self.exec_event_subs:
                try:
                    cb(msg)
                except Exception:
                    traceback.print_exc()
            return

        if mtype == "history":
            cid = msg.get("correlation_id")

            if "candles" in msg:
                candles = msg.get("candles", [])

                fut = None
                if cid is not None:
                    fut = self.pending_history.pop(cid, None)
                    if fut and not fut.done():
                        fut.set_result(candles)

                for cb in self.history_subs:
                    try:
                        cb(msg)
                    except Exception:
                        traceback.print_exc()

                if self.candle_engine is not None:
                    for c in candles:
                        try:
                            self._ingest_history_candle_into_engine(
                                symbol=msg["symbol"],
                                timeframe=msg["timeframe"],
                                candle=c,
                            )
                        except Exception:
                            traceback.print_exc()

                print(f"[Bridge] History response cid={cid} count={len(candles)}")
                return

            for cb in self.history_subs:
                try:
                    cb(msg)
                except Exception:
                    traceback.print_exc()

            if self.candle_engine is not None:
                try:
                    self._ingest_history_candle_into_engine(
                        symbol=msg["symbol"],
                        timeframe=msg["timeframe"],
                        candle=msg,
                    )
                except Exception:
                    traceback.print_exc()

            print(
                "[Bridge] History single bar:",
                msg.get("symbol"),
                msg.get("timeframe"),
                msg.get("timestamp"),
            )
            return

        if mtype == "error":
            print("[Bridge] EA error:", msg)
            cid = msg.get("correlation_id")
            fut = self.pending_history.pop(cid, None)
            if fut and not fut.done():
                fut.set_exception(RuntimeError(msg))

            for cb in self.exec_event_subs:
                try:
                    cb(msg)
                except Exception:
                    traceback.print_exc()
            return

        if mtype in ("account", "positions"):
            print("[Bridge] Snapshot:", msg)

            # Update sync state for execution role
            self._on_exec_snapshot(msg, role, symbol, timeframe)

            for cb in self.exec_event_subs:
                try:
                    cb(msg)
                except Exception:
                    traceback.print_exc()
            return

        print("[Bridge] Unknown message type:", mtype, "full:", msg)

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------

    def _ingest_candle_into_engine(self, msg: dict) -> None:
        if self.candle_engine is None:
            return

        symbol = msg["symbol"]
        timeframe = msg["timeframe"]

        ts_raw = msg.get("timestamp")
        if isinstance(ts_raw, (int, float)):
            ts = datetime.fromtimestamp(ts_raw, tz=timezone.utc)
        elif isinstance(ts_raw, str):
            try:
                ts = datetime.fromisoformat(ts_raw)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                else:
                    ts = ts.astimezone(timezone.utc)
            except Exception:
                ts = datetime.now(timezone.utc)
        else:
            ts = datetime.now(timezone.utc)

        tick = {
            "open": float(msg["open"]),
            "high": float(msg["high"]),
            "low": float(msg["low"]),
            "close": float(msg["close"]),
            "volume": float(msg.get("volume", 0.0)),
        }

        self.candle_engine.ingest_live_tick(symbol, timeframe, tick, ts)

    def _ingest_history_candle_into_engine(self, symbol: str, timeframe: str, candle: dict) -> None:
        if self.candle_engine is None:
            return

        ts_raw = candle.get("timestamp")
        if isinstance(ts_raw, (int, float)):
            ts = datetime.fromtimestamp(ts_raw, tz=timezone.utc)
        elif isinstance(ts_raw, str):
            try:
                ts = datetime.fromisoformat(ts_raw)
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                else:
                    ts = ts.astimezone(timezone.utc)
            except Exception:
                ts = datetime.now(timezone.utc)
        else:
            ts = datetime.now(timezone.utc)

        tick = {
            "open": float(candle["open"]),
            "high": float(candle["high"]),
            "low": float(candle["low"]),
            "close": float(candle["close"]),
            "volume": float(candle.get("volume", 0.0)),
        }

        if hasattr(self.candle_engine, "ingest_history_candle"):
            self.candle_engine.ingest_history_candle(symbol, timeframe, tick, ts)
        else:
            self.candle_engine.ingest_live_tick(symbol, timeframe, tick, ts)
