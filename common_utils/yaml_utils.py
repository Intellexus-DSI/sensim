from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Union
import yaml
from .path_utils import resolve_path


def load_yaml(path: Optional[str | Path], must_exist: bool = False) -> Dict[str, Any]:
    p = resolve_path(path, must_exist=must_exist, allowed_suffixes=[".yaml"])

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("YAML content must be a mapping at the root level.")

    return data


def save_yaml(data: Dict[str, Any], path: Union[str, Path]) -> None:
    """Save a dictionary to a YAML file."""
    p = resolve_path(path, must_exist=False, allowed_suffixes=[".yaml"])
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
