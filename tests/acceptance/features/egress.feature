Feature: Configured egress cannot be bypassed

  # The four scenarios below are EG-0..EG-3 from TEST_SUITE.md §6.4.1, written
  # at L4 on top of the L3 sealed-network harness (T-E2 / T-E8) and the L3
  # start-path drive (T-E8, in tests/test_egress_start_path.py). The L4 layer
  # owns the *user-visible journey* and binds to the L3 helpers rather than
  # re-implementing them — the §2.2 allocation rule. The agent in the drive is
  # a Claude-Code-shaped Anthropic-Messages client; a real pinned Claude Code
  # binary is agent_smoke's concern (T-I5/T-I6), not this layer's.
  #
  # One deliberate deviation from §6.4.1's wording: EG-0's "on each supported
  # transport" stays at L3 (T-E3 curl_cffi, T-E4 botocore, T-E5
  # provider-aiohttp already prove the matrix there). These four scenarios
  # exercise the bridge's own aiohttp serving path — the one every
  # `kitty claude` session rides — so the L4 journey and the L3 matrix are
  # never re-proving each other.
  #
  # EG-1 and EG-2 carry @needs_python_311 because the proxied drive shape
  # (TLS-in-TLS over stdlib asyncio) only works on Python 3.11+ (bpo-44011).
  # The same guard exists on the L3 phase-2/2b/3 tests; this tag mirrors it
  # at L4 so the 3.10 CI leg skips cleanly rather than failing on a known
  # dependency shape. EG-0 (single TLS hop) and EG-3 (process-level) do not
  # need the guard.

  Scenario: EG-0  The destination is reachable directly when egress is off
    Given no egress gateway configured
    When a turn is sent through the bridge serving path
    Then the recording upstream records the connection
    And the egress proxy saw no connection attempt
    # Control. Without it, EG-2 can pass because nothing could ever arrive
    # (TEST_SUITE.md §5.3 trap 2). The L3 phase-1 proof lives in
    # tests/harness/test_aiohttp_containment_slice.py::TestPhase1PositiveControl.

  @needs_python_311
  Scenario: EG-1  Every request arrives through the gateway
    Given a configured egress gateway
    When Claude Code runs a session through kitty
    Then every connection the recording upstream accepted arrived through the gateway
    # The L3 proof (proxy source-port join on every recorder peer port) lives
    # in tests/harness/test_aiohttp_containment_slice.py::TestPhase2bContainmentHealthy.

  @needs_python_311
  Scenario: EG-2  Traffic stops rather than leaks
    Given a configured egress gateway that has become unreachable
    When Claude Code sends a turn through kitty
    Then the turn fails with a clear error
    And the recording upstream receives nothing
    # Reads sealed_network.recorder.connections and .requests directly so the
    # assertion cannot silently pass on a non-empty leak. The L3 proof with
    # falsification lives in tests/harness/test_aiohttp_containment_slice.py
    # ::TestPhase2ContainmentProxyDown and ::TestPhase3Falsification.

  Scenario: EG-3  An unproxyable profile stops the launch
    Given a configured egress gateway
    And a profile whose transport cannot honour it
    When the user runs kitty
    Then kitty refuses to start
    And the refusal names the profile
    # Mirrors tests/test_egress_start_path.py::TestRefusingProfile, which
    # carries the §6.2.3 falsification (discarding the guard's return value
    # reaches BridgeServer.__init__). Falsification sensitivity inherited.