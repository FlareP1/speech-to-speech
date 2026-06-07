"""Synchronous MCP client using raw JSON-RPC 2.0 over HTTP.

Communicates with mcp-proxy (https://github.com/chrishayuk/mcp-proxy) via
plain HTTP POST requests. Each MCP server is exposed at
``/servers/<name>/mcp`` and requires:

1. An ``initialize`` call to obtain a session ID (returned in the
   ``Mcp-Session-Id`` response header).
2. All subsequent calls carry that session ID in the ``Mcp-Session-Id``
   request header.

No ``mcp`` package dependency — only ``httpx`` and ``logging``.
"""

import logging

import httpx

logger = logging.getLogger(__name__)

_REQUIRED_HEADERS = {
    "Content-Type": "application/json",
    "Accept": "application/json",
}


class MCPClient:
    """Client for mcp-proxy. Manages per-server HTTP sessions and tool routing."""

    def __init__(self, base_url: str, servers: list[str]):
        """
        Args:
            base_url: Base URL of mcp-proxy (e.g., "http://127.0.0.1:8008")
            servers: List of server names to connect to (e.g., ["time", "ddg-search"])
        """
        self.base_url = base_url.rstrip("/")
        self.servers = servers
        self.sessions: dict[str, httpx.Client] = {}
        self.tool_to_server: dict[str, str] = {}

    # ---- public API --------------------------------------------------------

    def connect(self) -> list[dict]:
        """Initialize all servers, discover tools, return OpenAI-format tool list.

        For each server:
          1. Create httpx.Client with base_url and required headers
          2. POST /servers/<name>/mcp with method "initialize"
          3. Extract Mcp-Session-Id from response headers, store in session
          4. POST /servers/<name>/mcp with method "tools/list"
          5. Convert each tool to OpenAI format
          6. Build self.tool_to_server mapping
          7. Store httpx.Client session per server in self.sessions

        Returns:
            list of OpenAI-format tool dicts.
        """
        all_tools: list[dict] = []
        rpc_id = 0

        for server_name in self.servers:
            client: httpx.Client | None = None
            try:
                client = httpx.Client(
                    base_url=self.base_url,
                    headers=_REQUIRED_HEADERS.copy(),
                )
                rpc_id += 1
                init_resp = client.post(
                    f"/servers/{server_name}/mcp",
                    json={
                        "jsonrpc": "2.0",
                        "id": rpc_id,
                        "method": "initialize",
                        "params": {
                            "protocolVersion": "2024-11-05",
                            "capabilities": {},
                            "clientInfo": {
                                "name": "speech-to-speech",
                                "version": "1.0",
                            },
                        },
                    },
                )
                init_resp.raise_for_status()

                session_id = init_resp.headers.get("Mcp-Session-Id", "")
                if session_id:
                    client.headers["Mcp-Session-Id"] = session_id

                # Discover tools
                rpc_id += 1
                list_resp = client.post(
                    f"/servers/{server_name}/mcp",
                    json={
                        "jsonrpc": "2.0",
                        "id": rpc_id,
                        "method": "tools/list",
                        "params": {},
                    },
                )
                list_resp.raise_for_status()
                body = list_resp.json()

                mcp_tools = body.get("result", {}).get("tools", [])
                for tool in mcp_tools:
                    openai_tool = {
                        "type": "function",
                        "function": {
                            "name": tool["name"],
                            "description": tool.get("description", ""),
                            "parameters": tool.get("inputSchema", {}),
                        },
                    }
                    all_tools.append(openai_tool)
                    self.tool_to_server[tool["name"]] = server_name

                self.sessions[server_name] = client
                client = None  # transferred ownership to self.sessions
                logger.info(
                    "Connected to MCP server '%s' — discovered %d tool(s)",
                    server_name,
                    len(mcp_tools),
                )

            except Exception:
                logger.warning(
                    "Failed to connect to MCP server '%s': %s",
                    server_name,
                    _format_exc(),
                )
                if client is not None:
                    client.close()

        return all_tools

    def execute_tool(self, name: str, arguments: dict) -> str:
        """Execute a tool via the correct MCP server.

        Returns:
            Joined text content from all ``type=="text"`` blocks, or an
            error string on failure.
        """
        try:
            server_name = self.tool_to_server[name]
            client = self.sessions[server_name]

            resp = client.post(
                f"/servers/{server_name}/mcp",
                json={
                    "jsonrpc": "2.0",
                    "id": 0,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": arguments},
                },
            )
            resp.raise_for_status()
            body = resp.json()

            content_blocks = body.get("result", {}).get("content", [])
            texts = [block["text"] for block in content_blocks if block.get("type") == "text"]
            return "\n".join(texts)

        except Exception:
            return f"Error executing {name}: {_format_exc()}"

    def disconnect(self) -> None:
        """Close all HTTP sessions."""
        for name, client in self.sessions.items():
            client.close()
        self.sessions.clear()


# ---- helpers ---------------------------------------------------------------


def _format_exc() -> str:
    """Return the current exception as a string."""
    import sys

    exc = sys.exc_info()[1]
    if exc is None:
        return "unknown error"
    return str(exc)
