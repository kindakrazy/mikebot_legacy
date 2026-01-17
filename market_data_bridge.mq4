//+------------------------------------------------------------------+
//|                                               MarketDataBridgeEA |
//|   Streams market data to Mikebot via TCP bridge                  |
//|   - Handshake v2 with hello_ack / server_ready / heartbeat       |
//|   - Sends historical candles                                     |
//|   - Streams live ticks                                           |
//|   - Streams live candle updates                                  |
//+------------------------------------------------------------------+
#property strict

#import "mt4_socket_bridgev2.dll"
   int  Socket_Connect(string host, int port);
   int  Socket_IsConnected();
   int  Socket_SendLine(string line);
   int  Socket_RecvLine(string &buffer, int bufferSize);
   void Socket_Close();
#import

#import "mt4_socket_bridgev2.dll"
   int TestPing();
#import

//-------------------------
// Inputs
//-------------------------
input string InpHost           = "127.0.0.1";
input int    InpPort           = 50010;
input bool   InpSendTicks      = true;
input int    InpReconnectMs    = 5000;
input int    InpPollMs         = 200;
input int    InpHistoryBars    = 2000;
input bool   InpEnableLogging  = true;

//-------------------------
// Globals
//-------------------------
bool     g_connected      = false;
string   g_symbol         = "";
int      g_timeframe      = 0;
string   g_tfStr          = "";
datetime g_lastCandleTime = 0;
int      g_digits         = 0;
double   g_point          = 0.0;

// v2 handshake state
bool     g_helloAcked      = false;
bool     g_serverReady     = false;
uint     g_lastHelloMs     = 0;
uint     g_lastHeartbeatMs = 0;

//-------------------------
// Logging helper
//-------------------------
void Log(string msg)
{
   if(InpEnableLogging)
      Print("[MDBridge] ", msg);
}

//+------------------------------------------------------------------+
//| Initialization                                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("PING RESULT: ", TestPing());

   g_symbol    = Symbol();
   g_timeframe = Period();
   g_tfStr     = TimeframeToString(g_timeframe);
   g_digits    = (int)MarketInfo(g_symbol, MODE_DIGITS);
   g_point     = MarketInfo(g_symbol, MODE_POINT);

   Log("Initializing for " + g_symbol + " " + g_tfStr);

   int timerSec = InpPollMs / 1000;
   if(timerSec < 1) timerSec = 1;
   EventSetTimer(timerSec);

   g_helloAcked      = false;
   g_serverReady     = false;
   g_lastHelloMs     = 0;
   g_lastHeartbeatMs = 0;

   TryConnect();
   if(g_connected)
   {
      Log("Connected, will send HELLO from OnTimer (v2)");
   }

   g_lastCandleTime = iTime(g_symbol, g_timeframe, 0);

   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   EventKillTimer();
   Socket_Close();
   Log("Deinitialized");
}

//+------------------------------------------------------------------+
void OnTick()
{
   if(!g_connected || !InpSendTicks || !g_serverReady)
      return;

   double bid = Bid;
   double ask = Ask;
   datetime ts = TimeCurrent();

   string json = "{"
      + "\"type\":\"tick\","
      + "\"symbol\":\"" + g_symbol + "\","
      + "\"timestamp\":" + IntegerToString((int)ts) + ","
      + "\"bid\":" + DoubleToString(bid, g_digits) + ","
      + "\"ask\":" + DoubleToString(ask, g_digits)
      + "}";

   Socket_SendLine(json);
}

//+------------------------------------------------------------------+
void OnTimer()
{
   if(!g_connected)
   {
      static uint lastTry = 0;
      uint now = GetTickCount();
      if(now - lastTry >= (uint)InpReconnectMs)
      {
         lastTry = now;

         g_helloAcked      = false;
         g_serverReady     = false;
         g_lastHelloMs     = 0;
         g_lastHeartbeatMs = 0;

         TryConnect();
         if(g_connected)
         {
            Log("Connected to " + InpHost + ":" + IntegerToString(InpPort));
         }
      }
      return;
   }

   // Read any server messages
   string line;
   while(Socket_RecvLine(line, 8192) > 0)
   {
      StringTrimLeft(line);
      StringTrimRight(line);
      if(StringLen(line) > 0)
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
      // v2: send heartbeat every 5s after server_ready
      if(g_serverReady && nowMs - g_lastHeartbeatMs >= 5000)
      {
         string hb = "{"
            + "\"type\":\"heartbeat\","
            + "\"role\":\"market_data\","
            + "\"symbol\":\"" + g_symbol + "\","
            + "\"timeframe\":\"" + g_tfStr + "\""
            + "}";
         Socket_SendLine(hb);
         g_lastHeartbeatMs = nowMs;
      }
   }

   // Candle updates only after server_ready
   if(g_serverReady)
      SendLatestCandleIfNew();
}

//+------------------------------------------------------------------+
void TryConnect()
{
   Print("TRYCONNECT CALLED");

   g_connected = (Socket_Connect(InpHost, InpPort) == 1);

   if(g_connected)
      Log("Connected to " + InpHost + ":" + IntegerToString(InpPort));
   else
      Log("Connection failed");
}

//+------------------------------------------------------------------+
void SendHello()
{
   string json = "{"
      + "\"type\":\"hello\","
      + "\"role\":\"market_data\","
      + "\"symbol\":\"" + g_symbol + "\","
      + "\"timeframe\":\"" + g_tfStr + "\""
      + "}";

   Socket_SendLine(json);
   Log("Sent HELLO (v2)");
}

//+------------------------------------------------------------------+
void HandleMessage(string json)
{
   string type = JsonGetString(json, "type");

   if(type == "hello_ack")
   {
      string role = JsonGetString(json, "role");
      if(role == "market_data")
      {
         g_helloAcked = true;
         Log("Received hello_ack");
      }
      return;
   }

   if(type == "server_ready")
   {
      g_serverReady = true;
      Log("Received server_ready, sending history");
      SendHistory(InpHistoryBars);
      return;
   }

   if(type == "ping")
   {
      string pong = "{"
         + "\"type\":\"pong\","
         + "\"role\":\"market_data\","
         + "\"symbol\":\"" + g_symbol + "\","
         + "\"timeframe\":\"" + g_tfStr + "\""
         + "}";
      Socket_SendLine(pong);
      return;
   }

   if(type == "cmd")
   {
      string cmd = JsonGetString(json, "cmd");
      string cid = JsonGetString(json, "correlation_id");

      if(cmd == "get_history")
         CmdGetHistory(json, cid);
   }
}

//+------------------------------------------------------------------+
void SendHistory(int bars)
{
   int total = Bars;
   if(bars > total)
      bars = total;

   for(int i = bars - 1; i >= 0; i--)
   {
      datetime t = Time[i];
      double o = Open[i];
      double h = High[i];
      double l = Low[i];
      double c = Close[i];
      long   v = Volume[i];

      string json = "{"
         + "\"type\":\"history\","
         + "\"symbol\":\"" + g_symbol + "\","
         + "\"timeframe\":\"" + g_tfStr + "\","
         + "\"timestamp\":" + IntegerToString((int)t) + ","
         + "\"open\":" + DoubleToString(o, g_digits) + ","
         + "\"high\":" + DoubleToString(h, g_digits) + ","
         + "\"low\":" + DoubleToString(l, g_digits) + ","
         + "\"close\":" + DoubleToString(c, g_digits) + ","
         + "\"volume\":" + IntegerToString(v)
         + "}";

      Socket_SendLine(json);
   }

   g_lastCandleTime = Time[0];
}

//+------------------------------------------------------------------+
void SendLatestCandleIfNew()
{
   datetime t0 = iTime(g_symbol, g_timeframe, 0);
   if(t0 == 0 || t0 == g_lastCandleTime)
      return;

   g_lastCandleTime = t0;

   double o = iOpen(g_symbol, g_timeframe, 0);
   double h = iHigh(g_symbol, g_timeframe, 0);
   double l = iLow(g_symbol, g_timeframe, 0);
   double c = iClose(g_symbol, g_timeframe, 0);
   long   v = iVolume(g_symbol, g_timeframe, 0);

   string json = "{"
      + "\"type\":\"candle\","
      + "\"symbol\":\"" + g_symbol + "\","
      + "\"timeframe\":\"" + g_tfStr + "\","
      + "\"timestamp\":" + IntegerToString((int)t0) + ","
      + "\"open\":" + DoubleToString(o, g_digits) + ","
      + "\"high\":" + DoubleToString(h, g_digits) + ","
      + "\"low\":" + DoubleToString(l, g_digits) + ","
      + "\"close\":" + DoubleToString(c, g_digits) + ","
      + "\"volume\":" + IntegerToString(v)
      + "}";

   Socket_SendLine(json);
}

//+------------------------------------------------------------------+
void CmdGetHistory(string json, string cid)
{
   string tfStr = JsonGetString(json, "timeframe");
   int bars = (int)JsonGetDouble(json, "bars");
   if(bars <= 0) bars = 500;

   int tf = StringToTimeframe(tfStr);
   if(tf <= 0) tf = g_timeframe;

   if(bars > 2000) bars = 2000;

   RefreshRates();
   int total = iBars(g_symbol, tf);
   if(total <= 0)
      return;

   if(bars > total) bars = total;

   string out = "{"
      + "\"type\":\"history\","
      + "\"symbol\":\"" + g_symbol + "\","
      + "\"timeframe\":\"" + TimeframeToString(tf) + "\","
      + "\"correlation_id\":\"" + cid + "\","
      + "\"candles\":[";

   bool first = true;

   for(int i = bars - 1; i >= 0; i--)
   {
      datetime t = iTime(g_symbol, tf, i);
      double o = iOpen(g_symbol, tf, i);
      double h = iHigh(g_symbol, tf, i);
      double l = iLow(g_symbol, tf, i);
      double c = iClose(g_symbol, tf, i);
      long   v = iVolume(g_symbol, tf, i);

      if(!first) out += ",";
      first = false;

      out += "{"
         + "\"timestamp\":" + IntegerToString((int)t) + ","
         + "\"open\":" + DoubleToString(o, g_digits) + ","
         + "\"high\":" + DoubleToString(h, g_digits) + ","
         + "\"low\":" + DoubleToString(l, g_digits) + ","
         + "\"close\":" + DoubleToString(c, g_digits) + ","
         + "\"volume\":" + IntegerToString(v)
         + "}";
   }

   out += "]}";
   Socket_SendLine(out);
}

//+------------------------------------------------------------------+
string JsonGetString(string json, string key)
{
   string pat = "\"" + key + "\":";
   int pos = StringFind(json, pat);
   if(pos < 0) return "";
   pos += StringLen(pat);

   while(pos < StringLen(json) && StringGetChar(json, pos) == ' ')
      pos++;

   bool quoted = (StringGetChar(json, pos) == '\"');
   if(quoted) pos++;

   string out = "";
   for(int i = pos; i < StringLen(json); i++)
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
int StringToTimeframe(string s)
{
   if(s == "M1")  return PERIOD_M1;
   if(s == "M5")  return PERIOD_M5;
   if(s == "M15") return PERIOD_M15;
   if(s == "M30") return PERIOD_M30;
   if(s == "H1")  return PERIOD_H1;
   if(s == "H4")  return PERIOD_H4;
   if(s == "D1")  return PERIOD_D1;
   if(s == "W1")  return PERIOD_W1;
   if(s == "MN1") return PERIOD_MN1;
   return 0;
}
//+------------------------------------------------------------------+
