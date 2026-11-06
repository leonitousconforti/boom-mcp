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
