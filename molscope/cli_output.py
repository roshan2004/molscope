"""Shared, pipeline-friendly shapes for CLI command output.

Every command that emits machine-readable JSON wraps its payload in one common
envelope, so a downstream tool can rely on a stable set of keys no matter which
command produced it::

    {
      "tool": "molscope",
      "version": "0.16.0",
      "command": "qc",
      "input": "examples/data/3ptb.pdb",   # path/id, or a list for batch commands
      "parser": "pdb",                       # the reader chosen from the extension
      "backends": ["gemmi"],                 # optional backends this run engaged
      "warnings": ["..."],                   # human-readable command warnings
      "result": {...}                        # the command-specific payload
    }

Batch commands (``analyze``, ``export``) add ``feature_names`` and ``skipped``
(one entry per input that could not be processed, with the reason). Keeping the
shape in one place lets the commands stay short and stay consistent.

``backends`` is computed by snapshotting :data:`sys.modules` before the work and
reporting which optional packages were imported during it — an honest "engaged in
this run" signal that, in a one-shot CLI process, reflects exactly what was used.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

#: Optional dependencies worth reporting when a run pulls them in.
_OPTIONAL_BACKENDS = (
    "rdkit", "gemmi", "scipy", "torch", "torch_geometric", "dgl",
    "networkx", "cupy", "propka", "openpyxl",
)

#: Map a data extension (no dot) to the parser MolScope selects for it.
_PARSER_BY_EXT = {
    "pdb": "pdb", "ent": "pdb", "cif": "cif", "mmcif": "cif",
    "xyz": "xyz", "sdf": "sdf", "mol": "sdf",
}


def parser_name(path: str) -> Optional[str]:
    """The parser MolScope picks for ``path`` (``"pdb"``/``"cif"``/...), or ``None``."""
    from .io import _data_extension

    return _PARSER_BY_EXT.get(_data_extension(str(path)).lstrip("."))


def parser_for_inputs(paths) -> Optional[str]:
    """A single parser name when every input shares one, else ``"mixed"``/``None``."""
    names = {parser_name(p) for p in paths}
    names.discard(None)
    if not names:
        return None
    return next(iter(names)) if len(names) == 1 else "mixed"


def backend_snapshot() -> frozenset:
    """Snapshot the imported modules so :func:`backends_since` can diff against it."""
    return frozenset(sys.modules)


def backends_since(before: frozenset) -> list:
    """Optional backends imported since ``before`` (sorted, honest "used this run")."""
    return sorted(
        name for name in _OPTIONAL_BACKENDS
        if name in sys.modules and name not in before
    )


def envelope(
    command: str,
    *,
    source=None,
    parser: Optional[str] = None,
    backends=None,
    warnings=None,
    result=None,
    **extra,
) -> dict:
    """Build the standard output envelope (see the module docstring).

    ``source`` becomes the ``input`` key (a path/id or a list for batch commands).
    ``result`` is the command payload; ``extra`` keys (e.g. ``feature_names``,
    ``skipped``) are merged in at the top level.
    """
    from . import __version__

    env = {
        "tool": "molscope",
        "version": __version__,
        "command": command,
        "input": source,
        "parser": parser,
        "backends": list(backends or []),
        "warnings": list(warnings or []),
    }
    if result is not None:
        env["result"] = result
    env.update(extra)
    return env


def emit_json(obj, *, file=None) -> None:
    """Print ``obj`` as indented JSON to ``file`` (default stdout)."""
    print(json.dumps(obj, indent=2), file=file or sys.stdout)


def write_json(path: str, obj) -> None:
    """Write ``obj`` as indented JSON to ``path`` (creating parent dirs)."""
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2)
