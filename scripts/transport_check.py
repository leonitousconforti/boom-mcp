"""Launch the server over stdio and over HTTP, list tools, and call one that needs no network."""

import json
import subprocess
import sys
import time

import anyio
from mcp.client import Client
from mcp.client.stdio import StdioServerParameters


async def check(target, label):
    async with Client(target) as client:
        tools = [t.name for t in (await client.list_tools()).tools]
        now = await client.call_tool("convert_time", {})
        missing = await client.call_tool("get_object", {"object_id": "ZTF26aaaaaaa"})
    print(f"{label}: {len(tools)} tools, now={json.loads(now.content[0].text)['iso']}")
    print(f"{label}: without a token -> error={missing.is_error}: {missing.content[0].text[:70]}")


async def main():
    await check(StdioServerParameters(command=".venv/bin/boom-mcp", env={"BABAMUL_API_TOKEN": ""}), "stdio")
    server = subprocess.Popen([".venv/bin/boom-mcp", "--http", "--port", "8765"], stderr=subprocess.DEVNULL)
    try:
        time.sleep(2)
        await check("http://127.0.0.1:8765/mcp", "http ")
    finally:
        server.terminate()


anyio.run(main)
