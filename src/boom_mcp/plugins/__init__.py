"""Load tool modules: the bundled examples, and any files given with --plugin."""

import importlib
import importlib.util
import pkgutil
from pathlib import Path


def load_builtin():
    for info in pkgutil.iter_modules(__path__):
        importlib.import_module(f"{__name__}.{info.name}")


def load_file(path: str):
    path = Path(path).resolve()
    spec = importlib.util.spec_from_file_location(f"boom_mcp_plugin_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"Cannot load plugin {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
