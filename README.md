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

Behind a reverse proxy, bind publicly and name the hosts it is reached by (other `Host`
headers get a 421); `GET /healthz` is there for the proxy:

```bash
boom-mcp --http --host 0.0.0.0 --port 8765 --allowed-host boom-mcp.example.org
```

A web page on another origin can call the server from a browser only if you name that origin
with `--cors-origin https://chat.example.org` (repeatable). Pages on localhost are always allowed.

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

Turn it on with `--assistant` and point it at an OpenAI-compatible model:

```bash
export ASSISTANT_LLM_URL=https://babamul.umn.edu/llm/qwen3-27b/v1
export ASSISTANT_LLM_MODEL=qwen3-27b
export ASSISTANT_LLM_API_KEY=...
boom-mcp --http --assistant
```

This adds two routes next to `/mcp`, under `/assistant` (change it with `--assistant-prefix`):

- `GET /assistant/health` returns `{"status": "ok", "model": "...", "vision": false}`. When the
  assistant is off the route does not exist, and the web app hides its button.
- `POST /assistant/chat` takes `{"messages": [...], "context": {...}}` and an
  `Authorization: Bearer <Babamul token>` header. It answers with server-sent events, one
  `data: {...}` line each: `status`, `text`, `tool_call`, `tool_result` (with any images),
  `done` and `error`.

The token is checked with Babamul's `GET /profile` before the model is asked, and the result is
kept for 5 minutes. A missing or rejected token gets a 401. Each person can have one question
running, and ask 20 questions per 10 minutes; past that they get a 429. At most 4 questions use
the model at once. The others wait their turn and see "Waiting for the model".

| Variable | Default | What it does |
| --- | --- | --- |
| `ASSISTANT_ENABLED` | off | Same as `--assistant` (with `--http`) |
| `ASSISTANT_PREFIX` | `/assistant` | Same as `--assistant-prefix` |
| `ASSISTANT_LLM_URL` | required | The model's base URL, ending in `/v1` |
| `ASSISTANT_LLM_MODEL` | empty | The model name sent with each request |
| `ASSISTANT_LLM_API_KEY` | empty | Sent to the model as a Bearer token, never to the browser |
| `ASSISTANT_LLM_VISION` | `false` | Also show the model the images the tools return |
| `ASSISTANT_THINKING` | `false` | Let a reasoning model think before it answers |
| `ASSISTANT_MAX_TOOL_CALLS` | `10` | Tool calls per question before it must answer |
| `ASSISTANT_ANSWER_TIMEOUT` | `240` | Seconds per question. Tools stop at three quarters of it, the rest is for the answer |
| `ASSISTANT_STALL_TIMEOUT` | `60` | Seconds the model may stay silent |
| `ASSISTANT_RESULT_BUDGET` | `20000` | Characters of one tool result the model reads |
| `ASSISTANT_CONTEXT_BUDGET` | `60000` | Characters of conversation before old tool results are cut down |
| `ASSISTANT_MAX_MESSAGES` | `30` | Past messages sent to the model |
| `ASSISTANT_CONCURRENCY` | `4` | Model calls at once, for everyone together |
| `ASSISTANT_RATE_LIMIT` | `20` | Questions per person per window |
| `ASSISTANT_RATE_WINDOW` | `600` | The window, in seconds |
| `ASSISTANT_FALLBACK_URL` | none | A second model, tried when the first is down, busy or failing. It is then used first for 5 minutes |
| `ASSISTANT_FALLBACK_MODEL` | empty | Its model name |
| `ASSISTANT_FALLBACK_API_KEY` | empty | Its key |
| `BABAMUL_API_URL` | production | The Babamul API for the tools and the token check |

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
    proxy_read_timeout 300s;  # longer than ASSISTANT_ANSWER_TIMEOUT
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
