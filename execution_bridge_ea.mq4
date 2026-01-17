//+------------------------------------------------------------------+
//|                                                   ExecutionBridgeEA.mq4
//|   Pure execution bridge: no decision logic.
//|   - Connects to Mikebot TCP server as client
//|   - Handshake v2 with hello_ack / heartbeat / ping-pong
//|   - Receives JSON commands, executes trades
//|   - Sends JSON events, account + position snapshots (per magic)
//+------------------------------------------------------------------+
#property strict

#import "mt4_socket_bridge.dll"
   int  Socket_Connect(string host, int port);
   int  Socket_IsConnected();
   int  Socket_SendLine(string line);
   int  Socket_RecvLine(string &buffer, int bufferSize);
   void Socket_Close();
#import

//--- input parameters
input string InpHost        = "127.0.0.1";
input int    InpPort        = 50020;
input int    InpMagic       = 12345;
input int    InpReconnectMs = 5000;
input int    InpPollMs      = 100;

//--- internal state
bool   g_connected      = false;
string g_symbol         = "";
int    g_digits         = 0;
double g_point          = 0.0;

// v2 handshake state
bool   g_helloAcked      = false;
uint   g_lastHelloMs     = 0;
uint   g_lastHeartbeatMs = 0;

// trailing stop state
int    g_tsTickets[128];
double g_tsDistance[128];
double g_tsStep[128];
int    g_tsCount = 0;

//+------------------------------------------------------------------+
int OnInit()
{
   g_symbol = Symbol();
   g_digits = (int)MarketInfo(g_symbol, MODE_DIGITS);
   g_point  = MarketInfo(g_symbol, MODE_POINT);

   int timerSec = InpPollMs / 1000;
   if(timerSec < 1) timerSec = 1;
   EventSetTimer(timerSec);

   g_helloAcked      = false;
   g_lastHelloMs     = 0;
   g_lastHeartbeatMs = 0;

   TryConnect();

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   Socket_Close();
}

//+------------------------------------------------------------------+
void OnTimer()
{
   if(!g_connected)
   {
      static int lastTryMs = 0;
      uint nowMs = GetTickCount();

      if(nowMs - lastTryMs >= InpReconnectMs)
      {
         lastTryMs = nowMs;
         TryConnect();

         if(g_connected)
         {
            g_helloAcked      = false;
            g_lastHelloMs     = 0;
            g_lastHeartbeatMs = 0;
         }
      }
      return;
   }

   string line;

   // SINGLE SAFE READ LOOP (line-based, DLL splits by '\n')
   while(Socket_RecvLine(line, 4096) > 0)
   {
      StringTrimLeft(line);
      StringTrimRight(line);

      if(line != "")
         HandleMessage(line);
   }

   uint nowMs = GetTickCount();

   // v2: send HELLO repeatedly until ack
   if(!g_helloAcked)
   {
      if(nowMs - g_lastHelloMs >= 1000)
      {
         SendHello();
         g_lastHelloMs = nowMs;
      }
   }
   else
   {
      // v2: heartbeat every 5s
      if(nowMs - g_lastHeartbeatMs >= 5000)
      {
         string hb = "{"
            + "\"type\":\"heartbeat\","
            + "\"role\":\"execution\","
            + "\"symbol\":\"" + g_symbol + "\","
            + "\"timeframe\":\"" + TimeframeToString(Period()) + "\","
            + "\"magic\":" + IntegerToString(InpMagic)
            + "}";
         Socket_SendLine(hb);
         g_lastHeartbeatMs = nowMs;
      }
   }

   UpdateTrailingStops();
}
//+------------------------------------------------------------------+
void OnTick() {}

//+------------------------------------------------------------------+
void TryConnect()
{
   int ok = Socket_Connect(InpHost, InpPort);
   g_connected = (ok == 1);
   Print("ExecutionBridgeEA TryConnect: ok=", ok, " connected=", g_connected);
}

//+------------------------------------------------------------------+
void SendHello()
{
   string tf = TimeframeToString(Period());
   string json = "{"
      + "\"type\":\"hello\","
      + "\"role\":\"execution\","
      + "\"symbol\":\"" + g_symbol + "\","
      + "\"timeframe\":\"" + tf + "\","
      + "\"magic\":" + IntegerToString(InpMagic)
      + "}";
   Socket_SendLine(json);
   Print("ExecutionBridgeEA: Sent HELLO (v2)");
}

//+------------------------------------------------------------------+
void HandleMessage(string json)
{
    Print("HandleMessage RAW: ", json);

    string type = JsonGetString(json, "type");

    // v2: hello_ack stops HELLO loop
    if(type == "hello_ack")
    {
        string role = JsonGetString(json, "role");
        if(role == "execution")
        {
            g_helloAcked = true;
            Print("ExecutionBridgeEA: Received hello_ack");
        }
        return;
    }

    // v2: server_ready (optional)
    if(type == "server_ready")
    {
        // Python will typically send account/positions_snapshot cmds after this
        return;
    }

    // v2: respond to ping
    if(type == "ping")
    {
        string pong = "{"
           + "\"type\":\"pong\","
           + "\"role\":\"execution\","
           + "\"symbol\":\"" + g_symbol + "\","
           + "\"timeframe\":\"" + TimeframeToString(Period()) + "\","
           + "\"magic\":" + IntegerToString(InpMagic)
           + "}";
        Socket_SendLine(pong);
        return;
    }

    // Only commands beyond this point
    if(type != "cmd")
        return;

    string cmd = JsonGetString(json, "cmd");
    string sym = JsonGetString(json, "symbol");

    if(sym != "" && sym != g_symbol)
        return;

    if(cmd == "open")                    CmdOpen(json);
    else if(cmd == "close")              CmdClose(json);
    else if(cmd == "modify")             CmdModify(json);
    else if(cmd == "trailing_stop")      CmdTrailingStop(json);
    else if(cmd == "account_snapshot")   CmdAccountSnapshot(json);
    else if(cmd == "positions_snapshot") CmdPositionsSnapshot(json);
}

//+------------------------------------------------------------------+
void CmdOpen(string json)
{
   string side = JsonGetString(json, "side");
   double lots = JsonGetDouble(json, "lots");
   double sl   = JsonGetDouble(json, "sl");
   double tp   = JsonGetDouble(json, "tp");
   string cid  = JsonGetString(json, "correlation_id");

   int type = (side == "buy") ? OP_BUY : OP_SELL;

   RefreshRates();
   double price = (type == OP_BUY) ? Ask : Bid;

   int ticket = OrderSend(g_symbol, type, lots, price, slippage(), sl, tp, "", InpMagic, 0, clrNONE);
   if(ticket < 0)
   {
      SendError("open", GetLastError(), cid);
      return;
   }

   SendEventOrderOpened(ticket, price, cid);
}

//+------------------------------------------------------------------+
void CmdClose(string json)
{
   int    ticket = StrToInteger(JsonGetString(json, "ticket"));
   string cid    = JsonGetString(json, "correlation_id");

   if(!OrderSelect(ticket, SELECT_BY_TICKET))
   {
      SendError("close", GetLastError(), cid);
      return;
   }

   RefreshRates();
   int type  = OrderType();
   double lots  = OrderLots();
   double price = (type == OP_BUY) ? Bid : Ask;

   bool ok = OrderClose(ticket, lots, price, slippage(), clrNONE);
   if(!ok)
   {
      SendError("close", GetLastError(), cid);
      return;
   }

   SendEventOrderClosed(ticket, price, cid);
}

//+------------------------------------------------------------------+
void CmdModify(string json)
{
   int    ticket = StrToInteger(JsonGetString(json, "ticket"));
   double sl     = JsonGetDouble(json, "sl");
   double tp     = JsonGetDouble(json, "tp");
   string cid    = JsonGetString(json, "correlation_id");

   if(!OrderSelect(ticket, SELECT_BY_TICKET))
   {
      SendError("modify", GetLastError(), cid);
      return;
   }

   double op = OrderOpenPrice();
   datetime ot = OrderOpenTime();

   bool ok = OrderModify(ticket, op, sl, tp, ot, clrNONE);
   if(!ok)
   {
      SendError("modify", GetLastError(), cid);
      return;
   }

   SendEventOrderModified(ticket, sl, tp, cid);
}

//+------------------------------------------------------------------+
void CmdTrailingStop(string json)
{
   int    ticket   = StrToInteger(JsonGetString(json, "ticket"));
   double distPts  = JsonGetDouble(json, "distance_points");
   double stepPts  = JsonGetDouble(json, "step_points");
   string cid      = JsonGetString(json, "correlation_id");

   int idx = FindTrailingIndex(ticket);
   if(idx < 0 && g_tsCount < 128)
   {
      idx = g_tsCount++;
      g_tsTickets[idx]  = ticket;
   }
   if(idx >= 0)
   {
      g_tsDistance[idx] = distPts;
      g_tsStep[idx]     = stepPts;
   }

   string jsonOut = "{"
      + "\"type\":\"event\","
      + "\"event\":\"trailing_started\","
      + "\"symbol\":\"" + g_symbol + "\","
      + "\"ticket\":" + IntegerToString(ticket) + ","
      + "\"correlation_id\":\"" + cid + "\""
      + "}";
   Socket_SendLine(jsonOut);
}

//+------------------------------------------------------------------+
// Startup sync helpers: account + positions snapshots (per magic)
//+------------------------------------------------------------------+
void CmdAccountSnapshot(string json)
{
   int magic = (int)JsonGetDouble(json, "magic");
   if(magic <= 0) magic = InpMagic;
   string cid = JsonGetString(json, "correlation_id");

   SendAccountSnapshot(magic, cid);
}

//+------------------------------------------------------------------+
void CmdPositionsSnapshot(string json)
{
   string sym = JsonGetString(json, "symbol");
   if(sym == "") sym = g_symbol;

   int magic = (int)JsonGetDouble(json, "magic");
   if(magic <= 0) magic = InpMagic;

   string cid = JsonGetString(json, "correlation_id");

   SendPositionsSnapshot(sym, magic, cid);
}

//+------------------------------------------------------------------+
void UpdateTrailingStops()
{
   if(g_tsCount <= 0) return;

   RefreshRates();

   for(int i=0; i<g_tsCount; i++)
   {
      int ticket = g_tsTickets[i];
      if(ticket <= 0) continue;
      if(!OrderSelect(ticket, SELECT_BY_TICKET)) continue;
      if(OrderSymbol() != g_symbol) continue;
      if(OrderMagicNumber() != InpMagic) continue;

      int type = OrderType();
      if(type != OP_BUY && type != OP_SELL) continue;

      double distPts = g_tsDistance[i];
      double stepPts = g_tsStep[i];

      double price = (type == OP_BUY) ? Bid : Ask;
      double sl    = OrderStopLoss();
      double op    = OrderOpenPrice();

      double distPrice = distPts * g_point;
      double stepPrice = stepPts * g_point;

      if(type == OP_BUY)
      {
         double newSL = price - distPrice;

         if(newSL > sl + stepPrice && newSL > op)
         {
            double sl_norm = NormalizeDouble(newSL, g_digits);

            bool mod_ok = OrderModify(
               ticket,
               op,
               sl_norm,
               OrderTakeProfit(),
               OrderOpenTime(),
               clrNONE
            );

            if(!mod_ok)
               Print("OrderModify BUY failed: ", GetLastError());
         }
      }
      else if(type == OP_SELL)
      {
         double newSL = price + distPrice;

         if(newSL < sl - stepPrice && newSL < op)
         {
            double sl_norm = NormalizeDouble(newSL, g_digits);

            bool mod_ok = OrderModify(
               ticket,
               op,
               sl_norm,
               OrderTakeProfit(),
               OrderOpenTime(),
               clrNONE
            );

            if(!mod_ok)
               Print("OrderModify SELL failed: ", GetLastError());
         }
      }
   }
}

//+------------------------------------------------------------------+
void SendAccountSnapshot(int magic, string cid)
{
   double bal   = AccountBalance();
   double eq    = AccountEquity();
   double mar   = AccountMargin();
   double free  = AccountFreeMargin();

   string json = "{"
      + "\"type\":\"account\","
      + "\"balance\":" + DoubleToString(bal, 2) + ","
      + "\"equity\":" + DoubleToString(eq, 2) + ","
      + "\"margin\":" + DoubleToString(mar, 2) + ","
      + "\"free_margin\":" + DoubleToString(free, 2) + ","
      + "\"magic\":" + IntegerToString(magic) + ","
      + "\"correlation_id\":\"" + cid + "\""
      + "}";
   Socket_SendLine(json);
}

//+------------------------------------------------------------------+
void SendPositionsSnapshot(string sym, int magic, string cid)
{
   if(sym == "") sym = g_symbol;

   string json = "{"
      + "\"type\":\"positions\","
      + "\"symbol\":\"" + sym + "\","
      + "\"magic\":" + IntegerToString(magic) + ","
      + "\"positions\":[";
   bool first = true;

   for(int i=0; i<OrdersTotal(); i++)
   {
      if(!OrderSelect(i, SELECT_BY_POS, MODE_TRADES)) continue;
      if(OrderSymbol() != sym) continue;
      if(OrderMagicNumber() != magic) continue;

      if(!first) json += ",";
      first = false;

      string side = (OrderType() == OP_BUY) ? "buy" : "sell";

      json += "{"
         + "\"ticket\":" + IntegerToString(OrderTicket()) + ","
         + "\"type\":\"" + side + "\","
         + "\"lots\":" + DoubleToString(OrderLots(), 2) + ","
         + "\"open_price\":" + DoubleToString(OrderOpenPrice(), g_digits) + ","
         + "\"sl\":" + DoubleToString(OrderStopLoss(), g_digits) + ","
         + "\"tp\":" + DoubleToString(OrderTakeProfit(), g_digits) + ","
         + "\"profit\":" + DoubleToString(OrderProfit(), 2)
         + "}";
   }

   json += "],"
      + "\"correlation_id\":\"" + cid + "\""
      + "}";
   Socket_SendLine(json);
}

//+------------------------------------------------------------------+
void SendError(string where, int code, string cid)
{
   string json = "{"
      + "\"type\":\"error\","
      + "\"where\":\"" + where + "\","
      + "\"code\":" + IntegerToString(code) + ","
      + "\"message\":\"" + ErrorDescription(code) + "\","
      + "\"correlation_id\":\"" + cid + "\""
      + "}";
   Socket_SendLine(json);
}

//+------------------------------------------------------------------+
void SendEventOrderOpened(int ticket, double price, string cid)
{
   string json = "{"
      + "\"type\":\"event\","
      + "\"event\":\"order_opened\","
      + "\"symbol\":\"" + g_symbol + "\","
      + "\"ticket\":" + IntegerToString(ticket) + ","
      + "\"price\":" + DoubleToString(price, g_digits) + ","
      + "\"correlation_id\":\"" + cid + "\""
      + "}";
   Socket_SendLine(json);
}

//+------------------------------------------------------------------+
void SendEventOrderClosed(int ticket, double price, string cid)
{
   string json = "{"
      + "\"type\":\"event\","
      + "\"event\":\"order_closed\","
      + "\"symbol\":\"" + g_symbol + "\","
      + "\"ticket\":" + IntegerToString(ticket) + ","
      + "\"price\":" + DoubleToString(price, g_digits) + ","
      + "\"correlation_id\":\"" + cid + "\""
      + "}";
   Socket_SendLine(json);
}

//+------------------------------------------------------------------+
void SendEventOrderModified(int ticket, double sl, double tp, string cid)
{
   string json = "{"
      + "\"type\":\"event\","
      + "\"event\":\"order_modified\","
      + "\"symbol\":\"" + g_symbol + "\","
      + "\"ticket\":" + IntegerToString(ticket) + ","
      + "\"sl\":" + DoubleToString(sl, g_digits) + ","
      + "\"tp\":" + DoubleToString(tp, g_digits) + ","
      + "\"correlation_id\":\"" + cid + "\""
      + "}";
   Socket_SendLine(json);
}

//+------------------------------------------------------------------+
int FindTrailingIndex(int ticket)
{
   for(int i=0; i<g_tsCount; i++)
      if(g_tsTickets[i] == ticket)
         return i;
   return -1;
}

//+------------------------------------------------------------------+
string JsonGetString(string json, string key)
{
   string pat = "\"" + key + "\":";
   int pos = StringFind(json, pat);
   if(pos < 0) return "";
   pos += StringLen(pat);

   while(pos < StringLen(json) && (StringGetChar(json, pos) == ' ')) pos++;

   bool quoted = (StringGetChar(json, pos) == '\"');
   if(quoted) pos++;

   string out = "";
   for(int i=pos; i<StringLen(json); i++)
   {
      int ch = StringGetChar(json, i);
      if(quoted)
      {
         if(ch == '\"') break;
      }
      else
      {
         if(ch == ',' || ch == '}' || ch == ' ') break;
      }
      out += StringSubstr(json, i, 1);
   }
   return out;
}

//+------------------------------------------------------------------+
double JsonGetDouble(string json, string key)
{
   string s = JsonGetString(json, key);
   if(s == "") return 0.0;
   return StrToDouble(s);
}

//+------------------------------------------------------------------+
string TimeframeToString(int tf)
{
   if(tf == PERIOD_M1)   return "M1";
   if(tf == PERIOD_M5)   return "M5";
   if(tf == PERIOD_M15)  return "M15";
   if(tf == PERIOD_M30)  return "M30";
   if(tf == PERIOD_H1)   return "H1";
   if(tf == PERIOD_H4)   return "H4";
   if(tf == PERIOD_D1)   return "D1";
   if(tf == PERIOD_W1)   return "W1";
   if(tf == PERIOD_MN1)  return "MN1";
   return IntegerToString(tf);
}

//+------------------------------------------------------------------+
int slippage() { return 3; }

//+------------------------------------------------------------------+
string ErrorDescription(int code)
{
   return(ErrorDescriptionInternal(code));
}

string ErrorDescriptionInternal(int code)
{
   switch(code)
   {
      case 1:   return "No error returned";
      case 2:   return "Common error";
      case 3:   return "Invalid trade parameters";
      case 4:   return "Trade server is busy";
      case 5:   return "Old version of the client terminal";
      case 6:   return "No connection with trade server";
      case 7:   return "Not enough rights";
      case 8:   return "Too frequent requests";
      case 64:  return "Account disabled";
      case 65:  return "Invalid account";
      case 128: return "Trade timeout";
      case 129: return "Invalid price";
      case 130: return "Invalid stops";
      case 131: return "Invalid trade volume";
      case 132: return "Market closed";
      case 133: return "Trade disabled";
      case 134: return "Not enough money";
      case 135: return "Price changed";
      case 136: return "Off quotes";
      case 137: return "Broker is busy";
      case 138: return "Requote";
      case 139: return "Order is locked";
      case 140: return "Long positions only";
      case 141: return "Too many requests";
      default:  return "Unknown error";
   }
}
//+------------------------------------------------------------------+
