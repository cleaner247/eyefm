from __future__ import annotations

from pathlib import Path
from typing import Any

from .config import deep_merge, load_config


def load_downstream_config(path: str | Path) -> dict[str, Any]:
    cfg_path = Path(path)
    cfg = load_config(cfg_path)
    base_path = cfg.pop("base_config", None)
    if not base_path:
        return cfg
    base = Path(base_path)
    if not base.is_absolute():
        base = cfg_path.parent / base
    return deep_merge(load_downstream_config(base), cfg)
