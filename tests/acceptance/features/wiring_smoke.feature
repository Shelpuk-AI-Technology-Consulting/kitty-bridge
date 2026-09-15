Feature: pytest-bdd wiring drives the L3 bridge harness

  T-J1 (KBR-107) proves the Gherkin ↔ L3 binding that TR-1 (T-J2, KBR-108) and
  EG-0..EG-3 (T-J3, KBR-109) reuse. The `When` step drives a real
  `BridgeFixture.post`; the `Then` step asserts through the L3 helper
  `assert_fixture_reached_its_recorder`. If either L3 symbol disappears or the
  step-lookup chain breaks, this scenario goes red before the real TR/EG
  scenarios land.

  Scenario: A turn driven through pytest-bdd reaches the recording transport
    Given a started bridge session for the ANTHROPIC_MESSAGES protocol
    When Claude Code sends a minimal turn through the bridge
    Then the recording transport reports the turn
