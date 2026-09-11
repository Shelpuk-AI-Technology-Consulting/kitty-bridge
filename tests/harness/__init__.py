"""Shared test infrastructure for the fidelity, containment and corpus suites.

`.system_design/TEST_SUITE.md` §7 · plan **Milestone 0**.

This package is the home of the infrastructure several streams share, rather
than each stream's own tests:

- :mod:`harness.contract` — the input contract (T-W2): the types every wire
  projection reads, every recorder produces, and the oracle compares.
- T-W4's recording upstreams, T-W5's CONNECT proxy fixture, T-W6's corpus
  loader and T-W8's bridge fixture land here alongside it.

**Why a package and not loose modules.**  ``tests/`` has no ``__init__.py``, so
pytest inserts ``tests/`` onto ``sys.path`` and these modules import as
``harness.<name>`` — the same mechanism ``tests/bridge/`` relies on.  That works
under **bare** ``pytest``, which is what CI runs; ``python -m pytest`` would also
put the working directory on the path and hide a mistake here.  Adding a
``tests/__init__.py`` would break it, exactly as
``tests/internal_key_scan.py``'s docstring warns.

The house rule that test modules are self-contained still holds: it governs
*test* modules importing each other, not shared support modules like these.
"""
