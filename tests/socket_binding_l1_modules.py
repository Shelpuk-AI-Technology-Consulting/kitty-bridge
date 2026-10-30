"""The §8.2 socket-binding L1 module registry -- shared source of truth.

``.system_design/TEST_SUITE.md`` §8.2 enumerates sixteen bullet modules that
bind real sockets or spawn real processes and so are ``l1`` by path default
today; the KBR-10 paragraph entry below the bullets makes it seventeen in
all. Two guards read this tuple, on different schedules:

* **KBR-272** (DRAFT PR #235): a per-module ``pytest.mark.timeout(120)`` mark
  so the §8.2 modules' hang-prone fixtures and teardowns stay bounded on the
  Fast job. The mark registry lives at
  ``tests/test_socket_binding_l1_timeout_marks.py``.

* **KBR-290** (this ticket): a ``--ignore <path>`` row in
  ``pyproject.toml``'s ``[tool.mutmut] pytest_add_cli_args_test_selection`` so
  the nightly mutation run never depends on loopback socket timing. The
  deselection guard lives at ``tests/test_socket_binding_l1_mutation_exclusion.py``.

Both guards import :data:`SOCKET_BINDING_L1_MODULES` from this module. There
must be exactly one enumeration, edited here, read by both. Two copies of one
list is how a module loses its mark or its ignore row and the guard that was
supposed to catch that exact drift stops catching it.

**Order matters.** The guard asserts the deselection list matches the
registry's order -- pyproject rows are emitted in this order, the §8.2 bullets
follow this order, and the KBR-272 timeout-mark guard inherits the same order.
Reorder this tuple only with a corresponding edit to ``pyproject.toml`` and
the design doc.

**Cross-ticket coordination (owned handoff).** KBR-272's branch
(``tests/test_socket_binding_l1_timeout_marks.py``) defines its tuple inline
today. When KBR-272's PR lands on main, **the KBR-272 PR author** must
replace that file's inline

    SOCKET_BINDING_L1_MODULES: tuple[str, ...] = (...)

with

    from socket_binding_l1_modules import SOCKET_BINDING_L1_MODULES

and drop the redundant §8.2 reconciliation paragraph from their branch on
rebase (KBR-290 lands that paragraph). The owner statement is in the Jira
comment and PR description; the swap itself is a one-line diff in KBR-272's
branch at merge time.
"""

from __future__ import annotations

# The §8.2 set: the bullet list plus the KBR-10 paragraph's module, in the
# order the design doc introduces them. One entry per module; both the
# KBR-272 timeout-mark guard and the KBR-290 mutmut-exclusion guard hold
# each against the live source.
SOCKET_BINDING_L1_MODULES: tuple[str, ...] = (
    # KBR-132 bullet.
    "tests/bridge/test_tls_certs.py",
    # T-W4 bullets.
    "tests/harness/test_recorder.py",
    "tests/harness/test_recorder_falsification.py",
    # T-W8 bullets.
    "tests/harness/test_bridge.py",
    "tests/harness/test_bridge_falsification.py",
    # T-W9 bullet.
    "tests/harness/test_vertical_slice.py",
    # T-B1 bullet.
    "tests/harness/test_provider_aiohttp.py",
    # T-E1 bullet.
    "tests/harness/test_containment.py",
    # KBR-144 bullet.
    "tests/bridge/test_responses_string_input.py",
    # KBR-176 bullet.
    "tests/bridge/test_bridge_management.py",
    # KBR-220 bullet.
    "tests/cli/test_bridge_state_location.py",
    # Prose-named since T-W5, bulleted by the KBR-272 reconciliation.
    "tests/test_egress_https_proxy.py",
    "tests/harness/test_connect_proxy.py",
    # Binding real BridgeServers since before KBR-144 named it as the
    # convention; bulleted by the KBR-272 reconciliation.
    "tests/bridge/test_crash_resilience.py",
    # KBR-56 (T-D6) bullet: the botocore-oracle slice starts a real
    # ``BridgeServer`` and the ``BedrockRecordingUpstream`` (two sockets).
    "tests/harness/test_oracle_botocore_slice.py",
    # KBR-57 (T-D7) bullet: the provider-aiohttp-oracle slice starts a real
    # ``BridgeServer`` and the ``ProviderRecordingUpstream`` (two sockets).
    "tests/harness/test_oracle_provider_aiohttp_slice.py",
    # KBR-10 paragraph (38 child interpreters; §8.2 describes it after the
    # bullets, which is why it is a separate entry here).
    "tests/cli/test_stream_encoding.py",
)
