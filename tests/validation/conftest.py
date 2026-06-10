"""Pytest plugin that turns a validation run into a generated summary.

When pytest is invoked with ``--validation-summary-dir=DIR`` over
``tests/validation``, this records the pass / skip / fail outcome of every
validation check (including whole modules skipped because a reference tool is
absent) and writes ``validation-summary.json`` and ``validation-summary.md`` to
``DIR`` at the end of the session. CI publishes those as a run artifact and to
the job summary, so the scientific cross-checks that actually ran are visible
without trawling the logs. Without the option the plugin is inert, so normal
test runs are unaffected.
"""

from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import _summary  # noqa: E402  (sibling module, made importable above)

#: Collected results for the validation modules, filled by the report hooks.
_RECORDS: list = []


def pytest_addoption(parser):
    parser.addoption(
        "--validation-summary-dir", action="store", default=None, metavar="DIR",
        help="write validation-summary.{json,md} to DIR after the run",
    )


def _is_validation_module(module: str) -> bool:
    """Whether a module stem is a validation-claim module worth summarising."""
    return module.endswith("_ref") or module in ("test_invariants", "test_graph_invariants")


def _module_and_name(nodeid: str):
    nodeid = nodeid.replace(os.sep, "/")
    if "validation/" not in nodeid:
        return None, None
    path, _, name = nodeid.partition("::")
    stem = path.rsplit("/", 1)[-1]
    if stem.endswith(".py"):
        stem = stem[:-3]
    return stem, (name or "(module)")


def _skip_reason(report) -> str:
    longrepr = getattr(report, "longrepr", None)
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2]).removeprefix("Skipped: ")
    return ""


def _failure_reason(report) -> str:
    text = str(getattr(report, "longrepr", "") or "")
    return text.strip().splitlines()[-1][:200] if text else ""


def _record(stem, name, outcome, reason=""):
    if stem and _is_validation_module(stem):
        _RECORDS.append(_summary.Outcome(stem, name, outcome, reason))


def pytest_runtest_logreport(report):
    """Record per-test pass/fail (call phase) and per-test skips (setup phase)."""
    stem, name = _module_and_name(report.nodeid)
    if stem is None:
        return
    if report.when == "call":
        if report.passed:
            _record(stem, name, "passed")
        elif report.failed:
            _record(stem, name, "failed", _failure_reason(report))
        else:
            _record(stem, name, "skipped", _skip_reason(report))
    elif report.when == "setup" and report.skipped:
        _record(stem, name, "skipped", _skip_reason(report))


def pytest_collectreport(report):
    """Record whole modules skipped at collection (e.g. a missing reference tool)."""
    if not report.skipped:
        return
    nodeid = str(report.nodeid)
    if nodeid.endswith(".py"):
        stem, _ = _module_and_name(nodeid)
        _record(stem, "(module skipped)", "skipped", _skip_reason(report))


def pytest_sessionfinish(session, exitstatus):
    out_dir = session.config.getoption("--validation-summary-dir")
    if not out_dir:
        return
    os.makedirs(out_dir, exist_ok=True)
    version = _molscope_version()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    import json

    md = _summary.to_markdown(_RECORDS, version=version, generated_at=stamp)
    payload = _summary.to_json(_RECORDS, version=version, generated_at=stamp)
    Path(out_dir, "validation-summary.md").write_text(md, encoding="utf-8")
    Path(out_dir, "validation-summary.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(f"wrote validation summary to {out_dir}/validation-summary.md")


def _molscope_version() -> str:
    try:
        import molscope

        return molscope.__version__
    except Exception:  # pragma: no cover - molscope is always importable here
        return "unknown"
