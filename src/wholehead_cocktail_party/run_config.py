"""Load reviewer-editable run knobs from config/run.yml.

The reviewer edits ``config/run.yml`` to choose the condition and the pipeline
mode. Every script reads it at startup via ``load_run_config()`` and validates
its accepted values via ``require_run()``. Reviewers do not need to open
any .py file or use the command line.

Author: Sudan Duwadi <sudan@bu.edu>
Notes: Module scaffold and docstrings were AI-assisted; the choice of which
       knobs are exposed rests with the author.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import yaml

from .paths import _find_repo_root  # reuse the same upward-walk logic


def _resolve_yml() -> Path:
    env = os.environ.get("WHCP_RUN_YML")
    if env:
        p = Path(env)
        if not p.is_file():
            raise FileNotFoundError(f"WHCP_RUN_YML points to a missing file: {p}")
        return p
    return _find_repo_root(Path(__file__).parent) / "config" / "run.yml"


def load_run_config(yml: Path | str | None = None) -> SimpleNamespace:
    """Return a namespace with ``condition``, ``mode`` (and any extra keys).

    Values are returned as plain strings; scripts compare them against the
    allowlists they pass to :func:`require_run`.
    """
    src = Path(yml) if yml is not None else _resolve_yml()
    with open(src, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{src} must contain a top-level mapping, got {type(raw).__name__}")
    return SimpleNamespace(**raw)


def require_run(
    cfg: SimpleNamespace,
    *,
    supported_conditions: set[str] | None = None,
    supported_modes: set[str] | None = None,
) -> None:
    """Fail fast if the run config asks for something this script does not implement.

    Each script declares which conditions and which pipeline modes it supports.
    If the reviewer set ``run.yml`` to something else, they get a clear,
    actionable message instead of a silent half-run or a deep traceback.

    Pass ``supported_conditions=None`` for single-purpose scripts whose
    condition is fixed by their filename (e.g. overt-control / overt-orient
    variants). The condition field in run.yml is then ignored by that script.
    """
    cond = getattr(cfg, "condition", None)
    mode = getattr(cfg, "mode", None)
    msgs: list[str] = []
    if supported_conditions is not None and cond not in supported_conditions:
        msgs.append(
            f"condition={cond!r} is not yet supported by this script. "
            f"Edit config/run.yml; supported: {sorted(supported_conditions)}."
        )
    if supported_modes is not None and mode not in supported_modes:
        msgs.append(
            f"mode={mode!r} is not yet supported by this script. "
            f"Edit config/run.yml; supported: {sorted(supported_modes)}."
        )
    if msgs:
        raise RuntimeError(" ".join(msgs))


_TEST_SUBJECT = "10"


def resolve_subjects(cfg: SimpleNamespace, default_cohort: list[str]) -> list[str]:
    """Return the subject list to process based on ``cfg.subjects``.

    Accepted values for ``cfg.subjects`` (set in config/run.yml):

    * ``'all'`` or ``None`` -- use ``default_cohort`` (the script's paper cohort).
    * ``'test'`` -- use ``[_TEST_SUBJECT]`` (currently sub-10) for smoke testing.
    * a list of strings -- explicit override.

    Any other value raises a clear ``ValueError`` so the reviewer notices.
    """
    val = getattr(cfg, "subjects", None)
    if val is None or val == "all":
        return list(default_cohort)
    if val == "test":
        return [_TEST_SUBJECT]
    if isinstance(val, list):
        if not all(isinstance(s, str) for s in val):
            raise ValueError(
                f"subjects in config/run.yml must be a list of strings, got {val!r}"
            )
        return list(val)
    raise ValueError(
        f"subjects in config/run.yml must be 'all', 'test', or a list of strings; "
        f"got {val!r}"
    )


__all__ = ["load_run_config", "require_run", "resolve_subjects"]
