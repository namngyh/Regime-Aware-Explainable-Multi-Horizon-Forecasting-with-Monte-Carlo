"""Configuration loading utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def write_config_snapshot(config: dict[str, Any], path: str | Path) -> None:
    """Persist a configuration snapshot."""
    with Path(path).open("w", encoding="utf-8") as fh:
        yaml.safe_dump(config, fh, sort_keys=False, allow_unicode=True)
