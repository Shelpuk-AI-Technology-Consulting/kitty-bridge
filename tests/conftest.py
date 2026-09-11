"""Shared fixtures for kitty tests, and the layer-marker wiring."""

from __future__ import annotations

import json
import socket
from collections.abc import Generator
from pathlib import Path

import pytest
from layers import (
    ENFORCEMENT_PREFIX,
    LAYER_MARKERS,
    default_layer_for,
    layer_markers_in,
    missing_required_categories,
    unknown_categories,
)

# Shared fixture modules. `pytest_plugins` is honoured in an *initial*
# conftest -- one on the path from rootdir to a collection argument, which
# this file is for every invocation the suite uses -- and is an error in any
# other conftest, so a deeper `tests/<dir>/conftest.py` could not hold it.
# A test asks for `connect_proxy` or `tls_target` by name; nothing imports
# `harness.connect_proxy` to get them.
pytest_plugins = ("harness.connect_proxy",)

# The collected items and their layers, as `(node id, [layer names])`, published
# for `tests/test_layer_markers.py`. A stash key rather than a module global so
# it is scoped to the session and typed.
collected_layer_records: pytest.StashKey[list[tuple[str, list[str]]]] = pytest.StashKey()


def _marker_names(item: pytest.Item) -> list[str]:
    """Return every marker name on an item, layer and otherwise.

    Args:
        item: A collected test item.

    Returns:
        The marker names, including inherited class- and module-level ones.
    """
    return [mark.name for mark in item.iter_markers()]


def _relative_path(item: pytest.Item, rootpath: Path) -> str:
    """Return an item's file path relative to the repository root.

    Args:
        item: A collected test item.
        rootpath: The pytest root directory.

    Returns:
        The relative path, or the absolute one when the item somehow sits
        outside the root -- in which case the path default falls back to the
        gating layer rather than raising, because refusing to collect is a
        worse answer than over-including one test.
    """
    try:
        return str(item.path.relative_to(rootpath))
    except ValueError:
        return str(item.path)


@pytest.fixture(autouse=True)
def _reset_backend_context() -> None:
    """Reset the per-request backend selection context before every test.

    The module-level ``_backend_context`` ContextVar in ``server.py``
    persists across tests that run in the same thread.  This fixture
    ensures every test starts with a clean slate so properties like
    ``_active_provider`` and ``_current_backend_idx`` read from the
    instance fields, not a stale context-var value from a prior test.
    """
    from kitty.bridge.server import _backend_context

    _backend_context.set({})


@pytest.fixture(autouse=True)
def _reset_egress() -> None:
    """Clear the process-wide egress configuration before every test.

    ``kitty.egress`` holds the resolved proxy in a module global, because
    provider adapters have no other channel for bridge-level settings. Without
    this reset, a test that enables egress would silently proxy every later
    test in the same process.
    """
    from kitty.egress import set_egress

    set_egress(None)


@pytest.fixture(autouse=True)
def _session_settings_in_tmp(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep per-session agent settings files inside the test's temp directory.

    ``ClaudeAdapter.prepare_launch`` creates its file with ``mkstemp`` and no
    explicit ``dir``, which resolves to the real OS temp directory. Redirecting
    ``tempfile.tempdir`` keeps the suite from scattering session files (each
    holding a test credential) across the developer's machine.
    """
    import tempfile

    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))


# ── Layer markers and job selection ────────────────────────────────────────
#
# `.system_design/TEST_SUITE.md` §8 defines each CI job as a marker expression
# over the eight layers in `tests/layers.py`, which only divides the suite
# cleanly if every test carries exactly one. The decisions live in that module,
# pure and separately testable; what is left here is the wiring.


def pytest_addoption(parser: pytest.Parser) -> None:
    """Add the per-category collection check flag.

    Args:
        parser: The pytest option parser.
    """
    parser.addoption(
        "--layer-report",
        default=None,
        dest="layer_report",
        metavar="PATH",
        help=(
            "Write the collected items and their layer markers to PATH as JSON. "
            "Lets a check outside this process reason about the whole suite's "
            "labelling without re-deriving it."
        ),
    )
    parser.addoption(
        "--require-category",
        action="append",
        default=[],
        dest="require_category",
        metavar="LAYER",
        help=(
            "Fail the run unless this layer collected at least one test. "
            "Repeatable. A job names every category its -m expression claims "
            "to run, because an 'or' expression is satisfied by either side."
        ),
    )


@pytest.hookimpl(wrapper=True)
def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> Generator[None, object, object]:
    """Apply layer defaults, then enforce the run's required categories.

    Args:
        config: The active pytest configuration.
        items: The collected items, mutated in place by this and other hooks.

    Yields:
        Control to the remaining ``pytest_collection_modifyitems`` implementations.

    Returns:
        The wrapped hook result, unchanged.

    Raises:
        UsageError: When ``--require-category`` names an unknown layer, or names
            a layer that collected nothing.

    The wrapper form is load-bearing, not stylistic. Defaults must be applied
    **before** pytest's own ``-m`` deselection -- otherwise ``-m l1`` deselects a
    suite whose files carry no markers yet -- and the category check must run
    **after** it, or it counts tests the job will not run. A wrapper puts both
    halves in one function with the ordering guaranteed by the hook protocol
    rather than by plugin registration order, which is an implementation detail.
    """
    # Default by path. An item that already names a layer -- by decorator or by
    # module-level `pytestmark` -- is left alone: the default exists to spare
    # 3,266 pre-existing tests an edit each, not to overrule a deliberate choice.
    rootpath = config.rootpath
    for item in items:
        if layer_markers_in(_marker_names(item)):
            continue

        item.add_marker(getattr(pytest.mark, default_layer_for(_relative_path(item, rootpath))))

    result = yield

    # `items` is now the surviving selection: pytest's mark plugin deselects in
    # place during the wrapped call.
    #
    # Stash the judged list FIRST, unconditionally. It is what the meta-test
    # reads, and populating it only on the branch that also runs the category
    # check would leave the meta-test judging an absent list on every ordinary
    # run -- which is the "stopped looking" failure, introduced by the very code
    # meant to prevent it.
    records = [(item.nodeid, layer_markers_in(_marker_names(item))) for item in items]
    config.stash[collected_layer_records] = records

    # Publish to a file when asked. The whole-suite checks in
    # `tests/test_layer_selection.py` read this rather than re-deriving the
    # labelling, so what they judge is what this hook actually assigned.
    report_path: str | None = config.getoption("layer_report")
    if report_path is not None:
        # An unwritable path is the caller's mistake, so it is reported as one.
        # Left unhandled it surfaces as pytest's INTERNALERROR traceback, which
        # reads as a bug in the suite rather than as a bad flag -- exactly the
        # confusion `ENFORCEMENT_PREFIX` exists to prevent.
        try:
            Path(report_path).write_text(
                json.dumps(
                    [{"nodeid": node_id, "layers": layers} for node_id, layers in records]
                ),
                encoding="utf-8",
            )
        except OSError as exc:
            raise pytest.UsageError(
                f"{ENFORCEMENT_PREFIX} could not write --layer-report to {report_path}: {exc}"
            ) from exc

    required: list[str] = config.getoption("require_category")
    if not required:
        return result

    unknown = unknown_categories(required)
    if unknown:
        raise pytest.UsageError(
            f"{ENFORCEMENT_PREFIX} --require-category names unknown layers: "
            + ", ".join(unknown)
            + f". Known layers: {', '.join(LAYER_MARKERS)}."
        )

    counts: dict[str, int] = {}
    for item in items:
        for name in layer_markers_in(_marker_names(item)):
            counts[name] = counts.get(name, 0) + 1

    missing = missing_required_categories(counts, required)
    if missing:
        raise pytest.UsageError(
            f"{ENFORCEMENT_PREFIX} these required categories collected no "
            "tests: "
            + ", ".join(missing)
            + ". A job that reports success without running a category it "
            "claims is worse than one that fails."
        )

    return result


@pytest.fixture()
def tmp_dir(tmp_path: Path) -> Path:
    """Temporary directory for profile/credential stores."""
    return tmp_path


@pytest.fixture()
def sample_profile_dict() -> dict:
    """Valid profile data dict for reuse across tests."""
    return {
        "name": "test-profile",
        "provider": "zai_regular",
        "model": "gpt-4o",
        "auth_ref": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
        "base_url": None,
        "provider_config": {},
        "is_default": False,
    }


@pytest.fixture()
def mock_provider_response() -> dict:
    """Sample Chat Completions response dict."""
    return {
        "id": "chatcmpl-test123",
        "object": "chat.completion",
        "created": 1700000000,
        "model": "gpt-4o",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "Hello from the provider."},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
    }


@pytest.fixture()
def unused_tcp_port() -> int:
    """Find a free TCP port for bridge tests."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture()
def collected_layer_markers(
    pytestconfig: pytest.Config,
) -> list[tuple[str, list[str]]]:
    """Return ``(node id, [layer names])`` for every item this run selected.

    Args:
        pytestconfig: The active pytest configuration.

    Returns:
        One record per collected, non-deselected item, exactly as the collection
        hook saw them.

    Raises:
        RuntimeError: When the collection hook did not run, which would
            otherwise let the meta-test certify an empty list.
    """
    records = pytestconfig.stash.get(collected_layer_records, None)

    if records is None:
        raise RuntimeError(
            "The layer-marker collection hook did not publish its records. "
            "The meta-test cannot certify a run it never saw."
        )

    return records
