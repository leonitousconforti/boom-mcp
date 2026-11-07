# boom-mcp over HTTP: the MCP server at /mcp, plus the Babamul assistant when
# ASSISTANT_ENABLED=true (see "Babamul assistant" in the README).
FROM python:3.11-slim

COPY --from=ghcr.io/astral-sh/uv:0.13.5 /uv /bin/uv

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

# Dependencies first, so a code change does not reinstall them.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
RUN uv sync --frozen --no-dev --no-editable

# Matplotlib and Astropy want a writable home; Apptainer images are read-only.
ENV PATH="/app/.venv/bin:$PATH" \
    HOME=/tmp \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONUNBUFFERED=1

USER nobody
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4)"]
CMD ["boom-mcp", "--http", "--host", "0.0.0.0", "--port", "8000"]
