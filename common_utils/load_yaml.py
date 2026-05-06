from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_yaml(path: Optional[str], must_exist: bool = False) -> Dict[str, Any]:
    """Load a YAML file into a dict.

    - If `path` is None/empty:
      - returns {} unless `must_exist=True`, in which case raises.
    - If the file exists but is empty, returns {}.
    - The YAML root must be a mapping (dict).

    Args:
        path: YAML path (string) or None.
        must_exist: Whether an empty/None path is considered an error.

    Returns:
        Parsed YAML mapping.
    """
    if not path:
        if must_exist:
            raise FileNotFoundError("Yaml file path must be provided.")
        return {}

    p = Path(path).expanduser()
    if not p.exists():
        raise FileNotFoundError(f"Yaml file not found: {p}")

    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError("YAML content must be a mapping at the root level.")

    return data
