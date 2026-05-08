"""Load reviewer-editable paths from config/paths.yml.

All scripts that need a local-filesystem location (raw data, classifier
outputs, ROI CSV, etc.) read it through ``load_paths()`` instead of
hardcoding. This keeps the reproducibility contract explicit: the reviewer
edits ``config/paths.yml`` once, and every script picks up the same values.

Author: Sudan Duwadi <sudan@bu.edu>
Notes: Module scaffold and docstrings were AI-assisted; path keys and the
       layout decision rest with the author.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import yaml

_PLACEHOLDER_PREFIX = "EDIT_ME"


def _find_repo_root(start: Path) -> Path:
    """Walk upward from ``start`` until a directory containing ``config/`` is found."""
    cur = start.resolve()
    for parent in [cur, *cur.parents]:
        if (parent / "config" / "paths.yml").is_file():
            return parent
    raise FileNotFoundError(
        "Could not locate config/paths.yml by walking up from "
        f"{start}. Set the WHCP_PATHS_YML environment variable to the "
        "absolute path of paths.yml, or run from inside the repository."
    )


def _resolve_yml() -> Path:
    env = os.environ.get("WHCP_PATHS_YML")
    if env:
        p = Path(env)
        if not p.is_file():
            raise FileNotFoundError(f"WHCP_PATHS_YML points to a missing file: {p}")
        return p
    return _find_repo_root(Path(__file__).parent) / "config" / "paths.yml"


def load_paths(yml: Path | str | None = None) -> SimpleNamespace:
    """Return a namespace of ``pathlib.Path`` values from ``config/paths.yml``.

    Parameters
    ----------
    yml : optional
        Path to a YAML file. If omitted, ``WHCP_PATHS_YML`` is checked, then
        the repo's ``config/paths.yml`` is used.

    Returns
    -------
    SimpleNamespace
        Attribute access for each top-level key in the YAML. String values
        become ``Path`` objects; non-string values are returned as-is.
    """
    src = Path(yml) if yml is not None else _resolve_yml()
    with open(src, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{src} must contain a top-level mapping, got {type(raw).__name__}")
    out: dict[str, object] = {}
    for k, v in raw.items():
        out[k] = Path(v) if isinstance(v, str) else v
    return SimpleNamespace(**out)


def require(paths: SimpleNamespace, *keys: str) -> None:
    """Raise if any of ``keys`` is missing or still set to the EDIT_ME placeholder.

    Scripts call this near the top to fail fast with a clear message instead
    of producing a confusing FileNotFoundError deep inside an analysis loop.
    """
    missing: list[str] = []
    placeholder: list[str] = []
    for k in keys:
        if not hasattr(paths, k):
            missing.append(k)
            continue
        v = getattr(paths, k)
        if isinstance(v, Path) and str(v).startswith(_PLACEHOLDER_PREFIX):
            placeholder.append(k)
    msgs: list[str] = []
    if missing:
        msgs.append(f"missing keys in paths.yml: {missing}")
    if placeholder:
        msgs.append(
            f"unedited placeholder values in paths.yml for keys: {placeholder}. "
            "Open config/paths.yml and replace the EDIT_ME prefix with your local path."
        )
    if msgs:
        raise RuntimeError("; ".join(msgs))


def whichscript_archive_dir(paths: SimpleNamespace) -> Path:
    """Resolve the whichscript provenance archive directory.

    If ``whichscript_archive`` is set in paths.yml (non-null, not EDIT_ME),
    that location is used. Otherwise it falls back to
    ``<derivatives_root>/whichscript_archive``, so reviewers do not need to
    configure a separate location.
    """
    val = getattr(paths, "whichscript_archive", None)
    if isinstance(val, Path) and not str(val).startswith(_PLACEHOLDER_PREFIX):
        return val
    return Path(paths.derivatives_root) / "whichscript_archive"


__all__ = ["load_paths", "require", "whichscript_archive_dir"]
