"""Run the BOOM MCP server: `boom-mcp` (stdio) or `boom-mcp --http` (Streamable HTTP)."""

import argparse
from pathlib import Path

from dotenv import load_dotenv
from mcp.server.transport_security import TransportSecuritySettings
from starlette.requests import Request
from starlette.responses import JSONResponse

from . import plugins, tools  # noqa: F401  (registers the tools)
from .app import mcp, settings

# Where the school setup puts the Babamul credentials, relative to a notebook folder.
ENV_FILES = [".env", "setup/.env", "../setup/.env", "../../setup/.env"]


@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "tools": len(await mcp.list_tools())})


def transport_security(hosts: list[str]) -> TransportSecuritySettings | None:
    """Host and Origin checks for the names this server is reached by. Without them
    a public bind accepts any Host header."""
    if not hosts:
        return None
    allowed = ["127.0.0.1:*", "localhost:*"]
    origins = ["http://127.0.0.1:*", "http://localhost:*"]
    for host in hosts:
        allowed += [host, f"{host}:*"] if ":" not in host else [host]
        if ":" not in host:
            origins.append(f"https://{host}")
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True, allowed_hosts=allowed, allowed_origins=origins
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--http", action="store_true", help="Serve Streamable HTTP instead of stdio.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        help="A Host header to accept, e.g. boom-mcp.example.org or 10.0.0.5:*. Repeatable.",
    )
    parser.add_argument(
        "--shared-token",
        action="store_true",
        help="Over HTTP, fall back to this server's BABAMUL_API_TOKEN for callers that send none.",
    )
    parser.add_argument("--plugin", action="append", default=[], help="A Python file of extra @tool() functions.")
    parser.add_argument("--no-examples", action="store_true", help="Skip the bundled example plugins.")
    args = parser.parse_args()

    for candidate in ENV_FILES:
        if Path(candidate).exists():
            load_dotenv(candidate)
            break

    if not args.no_examples:
        plugins.load_builtin()
    for path in args.plugin:
        plugins.load_file(path)

    if args.http:
        settings["require_client_token"] = not args.shared_token
        mcp.run(
            "streamable-http",
            host=args.host,
            port=args.port,
            stateless_http=True,
            json_response=True,
            transport_security=transport_security(args.allowed_host),
        )
    else:
        mcp.run()


if __name__ == "__main__":
    main()
