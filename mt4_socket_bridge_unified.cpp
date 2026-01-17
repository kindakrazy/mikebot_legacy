// mt4_socket_bridge_unified.cpp
// Unified MT4 socket bridge for both Execution and Market Data EAs.
// Build with -DBRIDGE_V2 to produce mt4_socket_bridgev2.dll

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <winsock2.h>
#include <ws2tcpip.h>
#include <string>
#include <cstring>

#pragma comment(lib, "Ws2_32.lib")

// ------------------------------------------------------------
// Globals
// ------------------------------------------------------------
static SOCKET      g_sock      = INVALID_SOCKET;
static bool        g_connected = false;
static bool        g_wsaInited = false;
static std::string g_recvBuf;

// ------------------------------------------------------------
// Debug logging (optional)
// ------------------------------------------------------------
static void dbg(const char* msg)
{
    OutputDebugStringA(msg);
    OutputDebugStringA("\r\n");
}

static void dbg_fmt(const char* fmt, ...)
{
    char buf[512];
    va_list args;
    va_start(args, fmt);
    _vsnprintf(buf, sizeof(buf)-1, fmt, args);
    buf[sizeof(buf)-1] = '\0';
    va_end(args);
    dbg(buf);
}

// ------------------------------------------------------------
// WSA init
// ------------------------------------------------------------
static bool EnsureWsa()
{
    if (g_wsaInited)
        return true;

    WSADATA wsa;
    int r = WSAStartup(MAKEWORD(2,2), &wsa);
    if (r != 0)
    {
        dbg_fmt("WSAStartup failed: %d", r);
        return false;
    }

    g_wsaInited = true;
    dbg("WSAStartup OK");
    return true;
}

static void CloseSocketInternal()
{
    if (g_sock != INVALID_SOCKET)
    {
        closesocket(g_sock);
        g_sock = INVALID_SOCKET;
    }
    g_connected = false;
    g_recvBuf.clear();
}

// ------------------------------------------------------------
// Wide → UTF‑8
// ------------------------------------------------------------
static std::string WideToUtf8(const wchar_t* wstr)
{
    if (!wstr)
        return std::string();

    int len = WideCharToMultiByte(CP_UTF8, 0, wstr, -1, nullptr, 0, nullptr, nullptr);
    if (len <= 0)
        return std::string();

    std::string out;
    out.resize(len - 1);
    WideCharToMultiByte(CP_UTF8, 0, wstr, -1, &out[0], len, nullptr, nullptr);
    return out;
}

// ------------------------------------------------------------
// Recv helper
// ------------------------------------------------------------
static int RecvSome()
{
    if (!g_connected || g_sock == INVALID_SOCKET)
        return 0;

    char buf[1024];
    int r = recv(g_sock, buf, sizeof(buf), 0);

    if (r == 0)
    {
        dbg("RecvSome: peer closed");
        CloseSocketInternal();
        return 0;
    }

    if (r == SOCKET_ERROR)
    {
        dbg_fmt("RecvSome: recv error %d", WSAGetLastError());
        CloseSocketInternal();
        return 0;
    }

    g_recvBuf.append(buf, r);
    return r;
}

// ------------------------------------------------------------
// Exported API
// ------------------------------------------------------------
extern "C" {

// Optional TestPing (only for v2)
#ifdef BRIDGE_V2
__declspec(dllexport) int __stdcall TestPing()
{
    dbg("TestPing called");
    return 123456;
}
#endif

// Connect
__declspec(dllexport) int __stdcall Socket_Connect(const wchar_t* host, int port)
{
    if (!host)
        return 0;

    std::string hostStr = WideToUtf8(host);

    // Trim
    while (!hostStr.empty() && isspace(hostStr.back()))
        hostStr.pop_back();
    while (!hostStr.empty() && isspace(hostStr.front()))
        hostStr.erase(hostStr.begin());

    CloseSocketInternal();

    if (!EnsureWsa())
        return 0;

    SOCKET s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (s == INVALID_SOCKET)
        return 0;

    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port   = htons((u_short)port);

    // Try dotted quad
    if (inet_pton(AF_INET, hostStr.c_str(), &addr.sin_addr) != 1)
    {
        // DNS fallback
        addrinfo hints{};
        hints.ai_family   = AF_INET;
        hints.ai_socktype = SOCK_STREAM;
        hints.ai_protocol = IPPROTO_TCP;

        addrinfo* res = nullptr;
        char portBuf[16];
        _snprintf(portBuf, sizeof(portBuf), "%d", port);

        if (getaddrinfo(hostStr.c_str(), portBuf, &hints, &res) != 0 || !res)
        {
            closesocket(s);
            return 0;
        }

        sockaddr_in* resolved = (sockaddr_in*)res->ai_addr;
        addr.sin_addr = resolved->sin_addr;
        freeaddrinfo(res);
    }

    if (connect(s, (sockaddr*)&addr, sizeof(addr)) == SOCKET_ERROR)
    {
        closesocket(s);
        return 0;
    }

    g_sock      = s;
    g_connected = true;
    g_recvBuf.clear();
    return 1;
}

// IsConnected
__declspec(dllexport) int __stdcall Socket_IsConnected()
{
    return g_connected ? 1 : 0;
}

// SendLine
__declspec(dllexport) int __stdcall Socket_SendLine(const wchar_t* line)
{
    if (!g_connected || g_sock == INVALID_SOCKET || !line)
        return 0;

    std::string data = WideToUtf8(line);
    data.push_back('\n');

    const char* buf = data.c_str();
    int total = (int)data.size();
    int sent  = 0;

    while (sent < total)
    {
        int r = send(g_sock, buf + sent, total - sent, 0);
        if (r == SOCKET_ERROR)
        {
            CloseSocketInternal();
            return 0;
        }
        sent += r;
    }

    return 1;
}

// RecvLine
__declspec(dllexport) int __stdcall Socket_RecvLine(char* buffer, int bufferSize)
{
    if (!g_connected || g_sock == INVALID_SOCKET || !buffer || bufferSize <= 1)
        return 0;

    for (;;)
    {
        size_t pos = g_recvBuf.find('\n');
        if (pos != std::string::npos)
        {
            std::string line = g_recvBuf.substr(0, pos);
            g_recvBuf.erase(0, pos + 1);

            int copyLen = (int)line.size();
            if (copyLen >= bufferSize)
                copyLen = bufferSize - 1;

            memcpy(buffer, line.data(), copyLen);
            buffer[copyLen] = '\0';
            return copyLen;
        }

        int r = RecvSome();
        if (r <= 0)
            return 0;
    }
}

// Close
__declspec(dllexport) void __stdcall Socket_Close()
{
    CloseSocketInternal();
}

} // extern "C"