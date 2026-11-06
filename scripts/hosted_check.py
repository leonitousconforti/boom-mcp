"""Hosted mode: the server holds no token; each caller sends their own as a Bearer header."""

import json
import os
import subprocess
import time

import anyio
import httpx2
from mcp.client import Client
from mcp.client.streamable_http import streamable_http_client

URL = "http://127.0.0.1:8766/mcp"


async def call_as(token):
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    async with httpx2.AsyncClient(headers=headers, timeout=60) as http:
        async with Client(streamable_http_client(URL, http_client=http)) as client:
            r = await client.call_tool("get_object", {"object_id": "ZTF26abypkdp"})
    return r.is_error, r.content[0].text


async def main():
    env = {k: v for k, v in os.environ.items() if k != "BABAMUL_API_TOKEN"}
    server = subprocess.Popen([".venv/bin/boom-mcp", "--http", "--port", "8766"], env=env, stderr=subprocess.DEVNULL)
    try:
        time.sleep(2)
        err, text = await call_as(os.environ["BABAMUL_API_TOKEN"])
        print("with token   -> error" if err else "with token   -> ok:", json.loads(text)["ml"]["applecider"]["decision"] if not err else text[:120])
        err, text = await call_as(None)
        print("no token     -> error:", err, text[:80])
        err, text = await call_as("bbml_wrong")
        print("wrong token  -> error:", err, text[:80])
    finally:
        server.terminate()


anyio.run(main)
