# boom-mcp

An MCP server for the [BOOM](https://github.com/boom-astro/boom) alert broker, through the public
Babamul API. It lets a language-model agent search ZTF and Rubin alerts, read light curves and
image cutouts, cross-match catalogs, and use the machine-learning outputs BOOM computes for every
public ZTF alert (AppleCiDEr, ACAI, BTSbot, embedding similarity, Villar fits).

It authenticates as you: over stdio with `BABAMUL_API_TOKEN` (the same `.env` the
[Rubin ML Training School](https://github.com/asasli/Rubin-ML-Training-School) setup uses), over
HTTP with each caller's own `Authorization: Bearer bbml_...` header. Get a token from your profile
at [babamul.caltech.edu](https://babamul.caltech.edu).

## Tools

| Tool | What it does |
| --- | --- |
| `convert_time` | ISO / JD / MJD conversion, and the current time |
| `search_alerts` | Alerts by time window (up to 1 day), cone or object, with BOOM-side cuts and ML cuts (AppleCiDEr class, out-of-distribution flag, BTSbot) |
| `get_object` | One object summarized: light-curve shape per band, flags, host, ML outputs and their history, nearest catalog matches |
| `get_light_curve` | Photometry as CSV, optionally plotted as a PNG |
| `get_cutouts` | Science / template / difference cutouts as one PNG |
| `get_cross_matches` | Catalog matches (TNS, NED, Gaia, VSX, Milliquas, DESI, ...) |
| `cone_search` | Objects near a position |
| `find_similar_objects` | Nearest neighbors in AppleCiDEr's embedding space |
| `get_light_curve_fit` | BOOM's Villar fit |
| `estimate_explosion_time` | Example plugin: bounds when a transient appeared |

Plus a `boom://guide` resource and a `triage` prompt.

## Install and run

```bash
pip install git+https://github.com/leonitousconforti/boom-mcp
echo 'BABAMUL_API_TOKEN=bbml_...' > .env
boom-mcp                        # stdio, for a local client
boom-mcp --http --port 8000     # Streamable HTTP at /mcp; each caller sends their own token
```

`--host` is the address to listen on. The default, `127.0.0.1`, takes connections from this
machine only: a reverse proxy or SSH tunnel on it. `GET /healthz` is there for the proxy.

`/mcp` checks the `Host` header of each request against the names in `--allowed-host`
(repeatable), plus `localhost:PORT` and `127.0.0.1:PORT`; any other gets a 421. Name every
public name a proxy or tunnel passes on to `/mcp`:

```bash
boom-mcp --http --port 8765 --allowed-host babamul.umn.edu --allowed-host boom-mcp.example.org
```

Without `--allowed-host` the server warns at startup. With `--host 127.0.0.1` it warns that a
proxy passing on a public name gets a 421 on `/mcp`; that is the usual cause of a 421 behind a
proxy. With a public `--host` it warns that `/mcp` then accepts any `Host` header: do not fix a 421
that way. The check is only on `/mcp`: the assistant's routes and `/healthz` answer whatever the
`Host`.

A web page on another origin can call the server from a browser only if you name that origin
with `--cors-origin https://chat.example.org` (repeatable). Pages on localhost are always allowed,
and so are pages served from an `--allowed-host` name itself.

A restart waits at most 10 seconds for open answers, then cuts them off; the web app tells those
people their answer was cut off. Nothing restarts the server after a crash, so on a shared node
run it under a loop in tmux, for example
`while true; do boom-mcp --http ... >> boom-mcp.log 2>&1; sleep 2; done`, and keep to one process:
the limits and caches live in it.

To work on it, `nix develop` gives a shell with `uv` and Node (for the MCP Inspector) and syncs
the environment.

## Connect a client

Claude Code:

```bash
claude mcp add boom -e BABAMUL_API_TOKEN=bbml_... -- boom-mcp
claude mcp add --transport http boom https://HOST/mcp --header "Authorization: Bearer bbml_..."
```

Claude Desktop, Cursor and Gemini CLI take the same `mcpServers` block (in
`claude_desktop_config.json`, `~/.cursor/mcp.json`, `~/.gemini/settings.json`):

```json
{ "mcpServers": { "boom": { "command": "boom-mcp", "env": { "BABAMUL_API_TOKEN": "bbml_..." } } } }
```

VS Code (`.vscode/mcp.json`):

```json
{ "servers": { "boom": { "type": "http", "url": "https://HOST/mcp", "headers": { "Authorization": "Bearer bbml_..." } } } }
```

## Babamul assistant

A chat panel for the Babamul web app. The browser sends the conversation and the person's
Babamul token to this server. The server asks a language model, runs the tools the model picks
with that person's token, and streams the answer back. The model's API key stays on the server.

Turn it on with `--assistant` and point it at an OpenAI-compatible model. When the model runs
on the same machine, use its own address (such as `http://127.0.0.1:8081/v1`) rather than a
public URL, so no proxy sits between the two:

```bash
export ASSISTANT_LLM_URL=http://127.0.0.1:8081/v1
export ASSISTANT_LLM_MODEL=ggml-org/Qwen3.8-27B-GGUF:Q8_0
export ASSISTANT_LLM_API_KEY=   # empty for a local llama.cpp without --api-key
boom-mcp --http --assistant
```

This adds two routes next to `/mcp`, under `/assistant` (change it with `--assistant-prefix`):

- `GET /assistant/health` returns `{"status": "ok", "model": "...", "vision": false}`. When the
  assistant is off the route does not exist, and the web app hides its button.
- `POST /assistant/chat` takes `{"messages": [...], "context": {...}}` and an
  `Authorization: Bearer <Babamul token>` header. It answers with server-sent events, one
  `data: {...}` line each: `status`, `text`, `tool_call`, `tool_result` (with any images),
  `done` and `error`. A body over 1 MB gets a 413.

The token is checked with Babamul's `GET /profile` before the model is asked, and the result is
kept for 5 minutes. A missing or rejected token gets a 401, and so does one not shaped like a
Bearer token, without asking Babamul. Each person has one question at a time: a new one stops
the one running, which ends with an `error` event, or a 409 if its stream has not started. Each
person can ask 20 questions per 10 minutes, replaced ones included; past that they get a 429.

`ASSISTANT_CONCURRENCY` questions are answered at once, 2 by default. Set it to the number of
slots the model server has (llama.cpp's `-np`), for example 8 on a dedicated instance. More would
wait inside the model server, where nobody can see how long. A question keeps its place on the
model until it is answered, tool calls included, so a slot sits idle while a tool runs; in
exchange every question let in can finish, and the model keeps its prompt cached. The others wait
in line, first come first served, and see their place in it ("Waiting for the model, 3 questions
ahead of you"). A question's answer timeout starts when it leaves the line. Waiting gives up after
`ASSISTANT_QUEUE_TIMEOUT`; when `ASSISTANT_MAX_QUEUE` questions are already waiting, a new one
gets a 429 at once.

| Variable | Default | What it does |
| --- | --- | --- |
| `ASSISTANT_ENABLED` | off | Same as `--assistant` (with `--http`) |
| `ASSISTANT_PREFIX` | `/assistant` | Same as `--assistant-prefix` |
| `ASSISTANT_LLM_URL` | required | The model's base URL, ending in `/v1` |
| `ASSISTANT_LLM_MODEL` | empty | The model name sent with each request |
| `ASSISTANT_LLM_API_KEY` | empty | Sent to the model as a Bearer token, never to the browser. Printable ASCII with no spaces: anything else stops startup with an error that names the variable, not the key |
| `ASSISTANT_LLM_VISION` | `false` | Also show the model the images the tools return |
| `ASSISTANT_THINKING` | `false` | Let a reasoning model think before it answers |
| `ASSISTANT_MAX_TOOL_CALLS` | `10` | Tool calls per question before it must answer |
| `ASSISTANT_ANSWER_TIMEOUT` | `240` | Seconds per question, from when it leaves the line. Tools stop at three quarters of it, the rest is for the answer. An answer already being written may use the rest too |
| `ASSISTANT_STALL_TIMEOUT` | `60` | Seconds the model may stay silent once its reply has started. The wait for its first word only has the answer timeout |
| `ASSISTANT_MAX_TOKENS` | `1024` | Longest reply, in tokens, so a reply that repeats itself ends. Thinking counts too, so raise it with `ASSISTANT_THINKING`. `0` for no limit |
| `ASSISTANT_RESULT_BUDGET` | `20000` | Characters of one tool result the model reads |
| `ASSISTANT_CONTEXT_BUDGET` | `60000` | Characters of conversation before old tool results are cut down |
| `ASSISTANT_MAX_MESSAGES` | `30` | Past messages sent to the model |
| `ASSISTANT_CONCURRENCY` | `2` | Questions answered at once, for everyone together. Match the model server's slots |
| `ASSISTANT_QUEUE_TIMEOUT` | `300` | Seconds a question may wait in line before it gives up |
| `ASSISTANT_MAX_QUEUE` | `20` | Questions that may wait in line. Past it a new question gets a 429 |
| `ASSISTANT_RATE_LIMIT` | `20` | Questions per person per window |
| `ASSISTANT_RATE_WINDOW` | `600` | The window, in seconds |
| `ASSISTANT_FALLBACK_URL` | none | A second model, tried when the first is down, busy or failing. It is then used first for 5 minutes |
| `ASSISTANT_FALLBACK_MODEL` | empty | Its model name |
| `ASSISTANT_FALLBACK_API_KEY` | empty | Its key, held to the same rule |
| `BABAMUL_API_URL` | production | The Babamul API for the tools and the token check |
| `BOOM_MCP_SHARED_ALERT_CACHE` | unset | `1` shares search_alerts' hour cache between users (Babamul's alert search does not depend on the user today); unset keys it by user |

The logs never hold a key, a request body or a model server's error text, since a server can
quote back what it was sent. A refused request is logged with its status and the error's type and
code (`401 type authentication_error`), a broken connection with the error's type only
(`LocalProtocolError`), and the model's URL without any user name, password or query.
`boom-mcp` turns the `httpx` and `httpcore` loggers down to warnings: httpcore's debug lines
quote a failed request's error whole, Authorization header included.

### With Docker

Put the settings in a file, say `assistant.env`:

```bash
ASSISTANT_ENABLED=true
ASSISTANT_LLM_URL=https://babamul.umn.edu/llm/qwen3-27b/v1
ASSISTANT_LLM_MODEL=qwen3-27b
ASSISTANT_LLM_API_KEY=...
BABAMUL_API_URL=https://babamul.caltech.edu/api/babamul
```

Then build and run:

```bash
docker build -t boom-mcp .
docker run -d --name boom-mcp -p 8000:8000 --env-file assistant.env boom-mcp
curl http://localhost:8000/assistant/health
```

Without `ASSISTANT_ENABLED` the image is a plain MCP server at `/mcp`.

### With Apptainer

Convert the Docker image, then run it. Apptainer shares the host's network, so the server
listens on port 8000 of the host.

```bash
apptainer build boom-mcp.sif docker-daemon://boom-mcp:latest
apptainer run --cleanenv --env-file assistant.env boom-mcp.sif
```

If the image is in a registry, build from it instead:
`apptainer build boom-mcp.sif docker://ghcr.io/OWNER/boom-mcp:latest`.

### Behind a reverse proxy

The web app calls `/api/assistant/` on its own host. Send that to the service.

nginx:

```nginx
location /api/assistant/ {
    proxy_pass http://127.0.0.1:8000/assistant/;
    proxy_http_version 1.1;
    proxy_set_header Connection "";
    proxy_buffering off;      # pass each event on as it comes
    proxy_read_timeout 300s;  # longer than the 15 s between events
}
```

Traefik, as Docker Compose labels. Here the prefix is set on the service, so no path rewrite is
needed:

```yaml
services:
  boom-mcp:
    image: boom-mcp
    env_file: assistant.env
    environment:
      ASSISTANT_PREFIX: /api/assistant
    labels:
      - traefik.enable=true
      - traefik.http.routers.babamul-assistant.rule=Host(`babamul.example.org`) && PathPrefix(`/api/assistant/`)
      - traefik.http.routers.babamul-assistant.entrypoints=websecure
      - traefik.http.routers.babamul-assistant.tls=true
      - traefik.http.services.babamul-assistant.loadbalancer.server.port=8000
```

The web app reaches the assistant on its own host, so it needs no `--cors-origin`.

## Add your own model

A plugin is a Python file of `@tool()` functions. See
[`src/boom_mcp/plugins/explosion_time.py`](src/boom_mcp/plugins/explosion_time.py), then
`boom-mcp --plugin my_model.py`.

## Notes on the Babamul API

- `GET /babamul/surveys/{survey}/alerts` times out (504 after 60 s) on time windows of a few hours,
  while one hour returns in about a second
  ([boom-astro/boom#900](https://github.com/boom-astro/boom/issues/900)). `search_alerts` fetches
  one UTC hour at a time and caches finished hours.
- One hour of public ZTF is about 20,000 alerts (115 MB through the API). With
  `drb ≥ 0.8`, positive, stationary, not star, not asteroid it is about 400 (3 MB).
- The `jd` bounds are inclusive at both ends, so adjacent windows share boundary alerts.
- ZTF forced photometry comes back as fluxes only (no `magpsf`, no SNR).

## Tests

```bash
pytest                              # offline, against a fake Babamul
python scripts/live_smoke.py OUT/   # every tool against the real API (needs a token)
python scripts/hosted_check.py      # HTTP mode with per-caller tokens
```
