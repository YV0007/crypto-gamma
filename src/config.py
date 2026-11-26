"""Single entry point to config.yaml.

Nothing in this project may hardcode a parameter that belongs in config
(statistical standard #6: every figure regenerable, every choice inspectable).
"""

from __future__ import annotations

import functools
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


@functools.lru_cache(maxsize=1)
def load(path: str | Path | None = None) -> dict[str, Any]:
    """Load and cache config.yaml."""
    p = Path(path) if path else ROOT / "config.yaml"
    with p.open() as fh:
        return yaml.safe_load(fh)


def get(dotted: str, default: Any = ...) -> Any:
    """Fetch a nested key, e.g. get("fetch.workers")."""
    node: Any = load()
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            if default is ...:
                raise KeyError(f"missing config key: {dotted}")
            return default
        node = node[part]
    return node


def path(dotted: str) -> Path:
    """Resolve a config path relative to the repository root."""
    p = Path(get(dotted))
    return p if p.is_absolute() else ROOT / p


def sample_bounds_ms() -> tuple[int, int]:
    """(start, end) of the sample period as epoch milliseconds, end-exclusive."""
    start = to_ms(get("sample.start"))
    end = to_ms(get("sample.end")) + 86_400_000  # end date inclusive
    return start, end


def to_ms(day: str) -> int:
    return int(datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def from_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
