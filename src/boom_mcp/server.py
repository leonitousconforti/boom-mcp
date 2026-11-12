"""Run the BOOM MCP server: `boom-mcp` (stdio) or `boom-mcp --http` (Streamable HTTP)."""

import argparse
import logging
import os
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from mcp.server.transport_security import TransportSecuritySettings
from starlette.middleware.cors import CORSMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from . import assistant, plugins, tools  # noqa: F401  (registers the tools)
from .app import mcp, settings

# Where the school setup puts the Babamul credentials, relative to a notebook folder.
ENV_FILES = [".env", "setup/.env", "../setup/.env", "../../setup/.env"]

# Browser clients on this machine, such as a local llama.cpp web UI, on any port.
LOCAL_ORIGINS = r"http://(localhost|127\.0\.0\.1|\[::1\]):\d+"

# The binds for which the MCP SDK accepts only local Host headers on /mcp.
LOCAL_BINDS = ("127.0.0.1", "localhost", "::1")

# Seconds a restart waits for open answers before it cuts them off. Without a
# limit uvicorn waits for every one, up to the whole answer timeout.
SHUTDOWN_GRACE = 10

log = logging.getLogger("boom_mcp.server")


@mcp.custom_route("/healthz", methods=["GET"])
async def healthz(request: Request) -> JSONResponse:
    return JSONResponse({"status": "ok", "tools": len(await mcp.list_tools())})


def transport_security(hosts: list[str], cors_origins: list[str]) -> TransportSecuritySettings | None:
    """Host and Origin checks for the names this server is reached by. Without them
    a public bind accepts any Host header."""
    if not hosts and not cors_origins:
        return None
    allowed = ["127.0.0.1:*", "localhost:*", "[::1]:*"]
    origins = ["http://127.0.0.1:*", "http://localhost:*", "http://[::1]:*", *cors_origins]
    for host in hosts:
        allowed += [host, f"{host}:*"] if ":" not in host else [host]
        if ":" not in host:
            origins.append(f"https://{host}")
    return TransportSecuritySettings(
        enable_dns_rebinding_protection=True, allowed_hosts=allowed, allowed_origins=origins
    )


def host_warning(args) -> str | None:
    """What to say at startup when /mcp will refuse, or accept, more Host headers than
    is likely meant. The Host check lives in the MCP transport, so only /mcp has it."""
    if args.allowed_host:
        return None
    if args.host in LOCAL_BINDS or args.cors_origin:
        return (
            f"/mcp accepts only the Host headers localhost:{args.port} and 127.0.0.1:{args.port}. "
            "Behind a proxy or tunnel that passes on a public name, such as babamul.umn.edu, "
            "/mcp answers 421: start with --allowed-host NAME for each name it is reached by. "
            "The assistant's routes do not check Host."
        )
    return (
        f"--host {args.host} without --allowed-host: /mcp accepts any Host header. "
        "Name the hosts it is reached by with --allowed-host."
    )


def http_app(args):
    """The Streamable HTTP app, with CORS for browser-based MCP clients, and the
    assistant's routes when it is enabled."""
    app = mcp.streamable_http_app(
        stateless_http=True,
        json_response=True,
        transport_security=transport_security(args.allowed_host, args.cors_origin),
        host=args.host,
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=args.cors_origin,
        allow_origin_regex=LOCAL_ORIGINS,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["*"],
        expose_headers=["Mcp-Session-Id"],
    )
    if config := getattr(args, "assistant_config", None):
        assistant.mount(app, assistant.Assistant(config), args.assistant_prefix)
    return app


def quiet_http_logs():
    for name in ("httpx", "httpcore"):
        logging.getLogger(name).setLevel(logging.WARNING)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--http", action="store_true", help="Serve Streamable HTTP instead of stdio.")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="The address to listen on. The default, 127.0.0.1, is for a proxy or tunnel on this machine.",
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--allowed-host",
        action="append",
        default=[],
        help="A Host header /mcp accepts, e.g. boom-mcp.example.org or 10.0.0.5:*. Repeatable. "
        "Needed for each public name a proxy passes on.",
    )
    parser.add_argument(
        "--cors-origin",
        action="append",
        default=[],
        help="A web page origin allowed to call this server from a browser, e.g. "
        "https://chat.example.org for a hosted llama.cpp web UI. Repeatable. "
        "Pages on localhost are always allowed.",
    )
    parser.add_argument(
        "--assistant",
        action="store_true",
        help="Also serve the Babamul chat assistant (needs --http and ASSISTANT_LLM_URL). "
        "ASSISTANT_ENABLED=true does the same.",
    )
    parser.add_argument(
        "--assistant-prefix",
        default=None,
        help="Where the assistant's routes live (default /assistant, or ASSISTANT_PREFIX).",
    )
    parser.add_argument(
        "--shared-token",
        action="store_true",
        help="Over HTTP, fall back to this server's BABAMUL_API_TOKEN for callers that send none.",
    )
    parser.add_argument("--plugin", action="append", default=[], help="A Python file of extra @tool() functions.")
    parser.add_argument("--no-examples", action="store_true", help="Skip the bundled example plugins.")
    args = parser.parse_args()

    # httpx logs every request at INFO, and httpcore each step of it at DEBUG with
    # any error quoted whole: for a header h11 refuses, that is the key in it. The
    # assistant's own lines say what matters without them.
    quiet_http_logs()

    for candidate in ENV_FILES:
        if Path(candidate).exists():
            load_dotenv(candidate)
            break

    if not args.no_examples:
        plugins.load_builtin()
    for path in args.plugin:
        plugins.load_file(path)

    if args.assistant and not args.http:
        parser.error("the assistant needs --http")
    if args.http and (args.assistant or assistant.env_flag(os.environ.get("ASSISTANT_ENABLED"))):
        try:
            args.assistant_config = assistant.Config.from_env()
        except ValueError as e:
            parser.error(str(e))
        if args.assistant_config is None:
            parser.error("the assistant needs ASSISTANT_LLM_URL, the model's OpenAI-compatible /v1 URL")
        args.assistant_prefix = args.assistant_prefix or os.environ.get("ASSISTANT_PREFIX") or "/assistant"
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        assistant.log.addHandler(handler)
        assistant.log.setLevel(logging.INFO)
        assistant.log.propagate = False

    if args.http:
        if warning := host_warning(args):
            log.warning(warning)
        settings["require_client_token"] = not args.shared_token
        uvicorn.run(
            http_app(args),
            host=args.host,
            port=args.port,
            log_level=mcp.settings.log_level.lower(),
            timeout_graceful_shutdown=SHUTDOWN_GRACE,
        )
    else:
        mcp.run()


if __name__ == "__main__":
    main()
