import asyncio
import json
from mikebot.adapters.bridge_server import MikebotBridgeServer

async def main():
    TEST_PORT = 50010

    print("=== Execution Bridge Test ===")

    # Create the bridge
    bridge = MikebotBridgeServer(host="0.0.0.0", port=TEST_PORT)

    # Start the server in the background (start() blocks forever)
    asyncio.create_task(bridge.start())

    print(f"[Bridge] Starting server on {bridge.host}:{bridge.port}")
    print("Waiting for Execution EA to connect...")

    open_tickets = []

    # ------------------------------------------------------------
    # Execution event handler
    # ------------------------------------------------------------
    def on_exec_event(event):
        print("\n[EXEC EVENT RECEIVED]")
        print(json.dumps(event, indent=2))

        etype = event.get("event")

        if etype == "order_opened":
            ticket = event.get("ticket")
            if ticket:
                open_tickets.append(ticket)
                print(f"[INFO] Order opened: ticket={ticket}")

        # When two orders are open, close them
        if len(open_tickets) == 2:
            asyncio.create_task(close_orders())

        if etype == "order_closed":
            print(f"[INFO] Order closed: ticket={event.get('ticket')}")

    bridge.on_execution_event(on_exec_event)

    # ------------------------------------------------------------
    # Close orders once both are opened
    # ------------------------------------------------------------
    async def close_orders():
        await asyncio.sleep(1)
        print("\n=== Closing Orders ===")
        for ticket in open_tickets:
            print(f"[SEND] Close ticket {ticket}")
            await bridge.send_command("BTCUSD", {
                "type": "cmd",
                "cmd": "close",
                "ticket": ticket,
                "correlation_id": f"close_{ticket}"
            })

    # ------------------------------------------------------------
    # Wait for EA connection
    # ------------------------------------------------------------
    while "BTCUSD" not in bridge.execution_eas:
        await asyncio.sleep(0.25)

    print("Execution EA connected.")
    print("Sending two BUY orders...")

    # ------------------------------------------------------------
    # Send two BUY orders
    # ------------------------------------------------------------
    for i in range(2):
        cmd = {
            "type": "cmd",
            "cmd": "open",
            "symbol": "BTCUSD",
            "side": "buy",
            "lots": 0.01,
            "sl": 0,
            "tp": 0,
            "correlation_id": f"open_{i+1}"
        }
        print(f"\n[SEND] Order #{i+1}")
        print(json.dumps(cmd, indent=2))
        await bridge.send_command("BTCUSD", cmd)

    print("\n=== Waiting for execution events ===")

    # Keep alive forever
    while True:
        await asyncio.sleep(1)

asyncio.run(main())
