from pathlib import Path
from typing import Optional


def resolve_path(path: Optional[str | Path], must_exist: bool = False,
                 allowed_suffixes: Optional[list[str]] = None) -> Path:
    """Resolve a path, ensuring it exists and has an allowed suffix."""
    if not path:
        if must_exist:
            raise FileNotFoundError("Path must be provided.")
        return Path()

    if isinstance(path, Path):
        resolved_path = path.expanduser().resolve()
    elif isinstance(path, str):
        resolved_path = Path(path).expanduser().resolve()
    else:
        raise TypeError("Path must be a string or Path object.")

    if must_exist and not resolved_path.exists():
        raise FileNotFoundError(f"Path does not exist: {resolved_path}")

    if allowed_suffixes and not any(resolved_path.suffix.lower() == suffix for suffix in allowed_suffixes):
        raise ValueError(f"Unsupported file suffix: {resolved_path.suffix} for {resolved_path}. "
                         f"Allowed suffixes are: {allowed_suffixes}")

    return resolved_path
