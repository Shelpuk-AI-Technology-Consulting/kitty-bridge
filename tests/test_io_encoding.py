"""Unit tests for :func:`kitty.io_encoding.relinquish_output_streams`.

KBR-96 / plan task T-I4's product half: the background bridge child
(``kitty.bridge_runner``) points pipe-shaped fds 1 and 2 at ``os.devnull``
once it has reported ready, so a write after ``kitty bridge start``'s parent
has exited can never raise ``BrokenPipeError`` (KBR-219). The helper is the
one place that dup2 happens; these tests pin its fd-level contract.

**Portable oracles only.** The fast gate runs this module on the Windows and
macOS legs (plan §1.3), so nothing here may depend on ``/proc``: "points at
devnull" is answered by ``stat.S_ISCHR`` (every platform's ``os.devnull`` is
a character device), and "still the file" by the written bytes landing in
that file. The end-to-end, /proc-based oracle lives in the l3 ownership
module, where the platform skip is part of the scenario.

**fd hygiene.** The helper's contract is about fds 1 and 2, which the tests
temporarily replace (dup2 a probe fd onto them, restore in ``finally``).
pytest's own capture holds the originals, so restoring them is what keeps
the session's output intact.

**Layer.** ``l1`` by path default: pure fd mechanics, no sockets, no child
processes — and deliberately so, so the six fast-gate legs exercise it from
day one (TEST_SUITE.md §8.2 keeps the process-spawning modules out of the
gate until T-K6).
"""

from __future__ import annotations

import os
import stat
import tempfile
from pathlib import Path

from kitty.io_encoding import relinquish_output_streams


def _points_at_devnull(fd: int) -> bool:
    """Report whether ``fd`` names a character device — ``os.devnull`` on every platform.

    Args:
        fd: The file descriptor to inspect.

    Returns:
        True when the fd's file is a character device.
    """
    return stat.S_ISCHR(os.fstat(fd).st_mode)


def test_relinquish_output_streams_dup2s_devnull_onto_pipe_streams():
    """A pipe-shaped fd 1 and 2 end up pointing at os.devnull.

    The bridge child is spawned with ``stdout=PIPE``; after ready, those fds
    are pipes. After the relinquish, a write cannot raise BrokenPipeError —
    the whole point of the KBR-219 fix — because the fd names os.devnull.
    """
    reader, writer = os.pipe()
    try:
        os.dup2(writer, 1)
        os.dup2(writer, 2)
        # The reader's death is what makes a later write raise
        # BrokenPipeError; the relinquish must make the write safe anyway.
        os.close(reader)
        os.close(writer)

        relinquish_output_streams()

        assert _points_at_devnull(1), "fd 1 was not pointed at os.devnull"
        assert _points_at_devnull(2), "fd 2 was not pointed at os.devnull"
        os.write(1, b"safe")  # must not raise BrokenPipeError
    finally:
        # Restore the fds pytest's capture installed. A devnull fd needs no
        # closing (os.devnull is opened internally by the helper), but the
        # originals must come back before the test ends.
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        os.close(devnull)


def test_relinquish_output_streams_leaves_regular_files_alone():
    """A file-shaped fd 1 and 2 are never redirected.

    Service managers hand the bridge files, sockets or /dev/null — never a
    pipe. Widening the redirect past pipes would swallow a deployment's log
    stream.
    """
    with tempfile.TemporaryDirectory() as tmp:
        probe = Path(tmp) / "probe.log"
        fd = os.open(probe, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        saved1, saved2 = os.dup(1), os.dup(2)
        try:
            os.dup2(fd, 1)
            os.dup2(fd, 2)
            os.close(fd)

            relinquish_output_streams()

            assert not _points_at_devnull(1), "fd 1 was redirected despite being a regular file"
            assert not _points_at_devnull(2), "fd 2 was redirected despite being a regular file"
            os.write(1, b"still-here")
        finally:
            os.dup2(saved1, 1)
            os.dup2(saved2, 2)
            os.close(saved1)
            os.close(saved2)
        assert probe.read_bytes() == b"still-here", "the file lost its write"


def test_relinquish_output_streams_leaves_closed_streams_alone():
    """A closed fd is skipped, not a reason to fail.

    ``harden_output_streams``' rule applies here too: hardening output must
    never itself become the thing that fails.
    """
    saved2 = os.dup(2)
    try:
        os.close(2)
        relinquish_output_streams()  # must not raise
    finally:
        os.dup2(saved2, 2)
        os.close(saved2)


def test_relinquish_output_streams_is_idempotent():
    """A second call changes nothing and raises nothing.

    The runner calls the helper once at ready; the warning's except-branch
    may call it again. Both must be safe.
    """
    reader, writer = os.pipe()
    try:
        os.dup2(writer, 1)
        os.dup2(writer, 2)
        os.close(reader)
        os.close(writer)

        relinquish_output_streams()
        relinquish_output_streams()

        assert _points_at_devnull(1) and _points_at_devnull(2)
        os.write(1, b"safe")
    finally:
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        os.close(devnull)
