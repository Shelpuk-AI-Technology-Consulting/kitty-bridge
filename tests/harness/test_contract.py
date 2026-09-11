"""L1 tests for the input contract — the vocabulary the fidelity oracle is written in.

`.system_design/TEST_SUITE.md` §3.3.1 · plan task **T-W2** (KBR-25).

These tests prove the *contract*, not any reader.  Nothing here exercises
kitty: the module under test imports nothing from ``src/kitty`` by design
(§7.4), because an oracle written in terms of the code under test proves
self-consistency rather than fidelity.

**The falsification set is the point of this module.**  Plan §1.4 requires the
first working version of every harness to ship with a deliberate defect it must
detect.  :class:`TestTotalityFalsification` holds four stub readers over one
body — one that drops an unknown key, one that residualises it, one that drops a
key *nested* under a key it consumed, and one that accounts for everything.  The
last is the control: without it, a :func:`verify_total` that raised
unconditionally would satisfy the other three.

**Why ``isinstance`` and never ``issubclass``.**  ``Projection`` carries a data
member (``wire_format``), and ``issubclass`` raises ``TypeError`` on such a
protocol.  ``isinstance`` checks member *presence* only — never signatures — so
it cannot prove :attr:`read_request`'s parameter type; the annotation test does
that work separately.
"""

from __future__ import annotations

import dataclasses
import hashlib
import re
from pathlib import Path
from typing import get_type_hints

import pytest

from harness import contract as c


class TestCapturedRequestShape:
    """The seven ticket-named fields, in order, carried losslessly."""

    def test_the_seven_named_fields_come_first_and_in_the_tickets_order(self) -> None:
        """Downstream tasks destructure this type; the order is part of the contract."""
        names = [f.name for f in dataclasses.fields(c.CapturedRequest)]

        assert names[:7] == ["method", "scheme", "host", "path", "query", "headers", "body"]

    def test_the_recorder_slots_default_to_none_so_t_w4_can_fill_them(self) -> None:
        """§7.2's arrival time and peer port are T-W4's to populate, not fidelity's to read."""
        captured = c.CapturedRequest(
            method="POST", scheme="https", host="api.example.com", path="/v1/messages", query="", headers=(), body=b""
        )

        assert captured.arrival is None
        assert captured.peer_port is None

    def test_duplicate_header_names_and_original_casing_both_survive(self) -> None:
        """§4.3 C1 asserts the exact header set, so a mapping would destroy the evidence."""
        captured = _capture(headers=(("X-Api-Key", "k"), ("x-api-key", "j")))

        assert captured.headers == (("X-Api-Key", "k"), ("x-api-key", "j"))

    def test_the_query_string_is_stored_verbatim_not_reordered(self) -> None:
        """T-W9's falsification is a recorder that drops the query; parsing is the caller's job."""
        assert _capture(query="b=2&a=1").query == "b=2&a=1"

    def test_a_body_that_is_not_valid_utf8_survives_unchanged(self) -> None:
        """The capture is bytes: T-C6 contributes a malformed body and it must reach the reader."""
        raw = b"\xff\xfe not json at all"

        assert _capture(body=raw).body == raw

    def test_headers_given_as_a_mapping_are_rejected_rather_than_shredded(self) -> None:
        """Iterating a mapping yields its keys, so a dict would become character tuples.

        R1.2 makes headers a sequence of pairs *because* someone will reach for
        a mapping — most likely T-W4's recorder author. Silently accepting one
        destroys the exact header evidence §4.3 C1 asserts on, and the loss
        would surface much later as a mysterious `repr` crash.
        """
        with pytest.raises(TypeError, match="mapping"):
            _capture(headers={"Authorization": "Bearer sk-LEAK"})  # type: ignore[arg-type]

    def test_a_header_entry_that_is_not_a_pair_is_rejected(self) -> None:
        """The control for the guard above, covering a malformed sequence."""
        with pytest.raises(TypeError, match="pair"):
            _capture(headers=(("Authorization",),))  # type: ignore[arg-type]

    def test_a_two_character_string_does_not_slip_through_as_a_pair(self) -> None:
        """A string is a sequence, so `tuple("ab")` is a valid-looking 2-tuple.

        A length check cannot catch this — `("a", "b")` has length 2 and would
        be stored as a header named `a` with the value `b`. Only rejecting the
        type does, and this is the one shape that walked through a guard whose
        own docstring says failing loudly is the point.
        """
        with pytest.raises(TypeError, match="not a string"):
            _capture(headers=("ab",))  # type: ignore[arg-type]

    def test_a_longer_string_is_rejected_for_the_same_reason(self) -> None:
        """The control: a length check alone would reject this one and miss the pair above."""
        with pytest.raises(TypeError, match="not a string"):
            _capture(headers=("Authorization",))  # type: ignore[arg-type]

    def test_a_bytes_header_name_is_rejected_because_it_would_bypass_the_mask(self) -> None:
        """The pair-length check let a credential through the redaction entirely.

        `b"authorization".lower()` is `b"authorization"`, which is not in a set
        of `str` — so a bytes-named `Authorization` header missed the mask and
        its value was rendered in full into every log and assertion diff. A
        length check that ignores element types is how a leak walks through a
        guard that looks like it covers this.
        """
        with pytest.raises(TypeError, match="both be str"):
            _capture(headers=((b"Authorization", "Bearer sk-LEAKED"),))  # type: ignore[arg-type]

    def test_a_bytes_header_value_is_rejected_too(self) -> None:
        """The other element, so the check cannot be half-applied."""
        with pytest.raises(TypeError, match="both be str"):
            _capture(headers=(("Authorization", b"Bearer sk-LEAKED"),))  # type: ignore[arg-type]

    def test_headers_given_as_a_generator_are_not_consumed_by_the_guard(self) -> None:
        """A one-shot iterable must survive validation.

        The guard has to look at the entries *and* store them. Reading the
        input twice consumes a generator on the first pass, so the second sees
        nothing and the capture is stored with **no headers at all** — no
        error, from a guard whose whole purpose is to fail loudly. A recorder
        streaming its headers is the realistic source.
        """
        captured = _capture(headers=(pair for pair in (("A", "1"), ("B", "2"))))  # type: ignore[arg-type]

        assert captured.headers == (("A", "1"), ("B", "2"))


class TestCapturedRequestRedaction:
    """Credentials must not reach a pytest diff or a CI log (R1.7)."""

    def test_the_authorization_header_value_is_absent_from_the_repr(self) -> None:
        """A dataclass repr lands in every assertion failure; §7.1 flags this trap for the corpus."""
        captured = _capture(headers=(("Authorization", "Bearer sk-secret-value"),))

        assert "sk-secret-value" not in repr(captured)

    def test_the_gemini_api_key_query_parameter_is_absent_from_the_repr(self) -> None:
        """Gemini carries its credential in the URL, which R1.3 preserves verbatim."""
        captured = _capture(query="key=AIzaSecretValue&alt=sse")

        assert "AIzaSecretValue" not in repr(captured)

    def test_the_query_mask_actually_matched_and_left_the_rest_alone(self) -> None:
        """A masking rule that silently stopped matching would pass the test above.

        The absence of a secret proves nothing on its own — a `repr` that
        dropped the query entirely would also pass. This pins that the mask
        fired *and* that the non-credential parameter survived.
        """
        rendered = repr(_capture(query="key=AIzaSecretValue&alt=sse"))

        assert f"key={c.REDACTION_MASK}" in rendered
        assert "alt=sse" in rendered

    def test_every_masked_query_key_is_lowercase(self) -> None:
        """Matching is case-insensitive, so the constant must not carry mixed case."""
        for name in c.REDACTED_QUERY_KEYS:
            assert name == name.lower()

    def test_the_masked_repr_still_shows_that_a_credential_was_present(self) -> None:
        """Masking to nothing would hide a missing-auth bug as effectively as a leak."""
        captured = _capture(headers=(("Authorization", "Bearer sk-secret-value"),))

        assert c.REDACTION_MASK in repr(captured)

    def test_a_non_credential_header_is_shown_in_full(self) -> None:
        """This is the control: a repr that masked everything would pass the two tests above."""
        captured = _capture(headers=(("Content-Type", "application/json"),))

        assert "application/json" in repr(captured)

    def test_the_raw_credential_is_still_reachable_by_field_access(self) -> None:
        """C1 asserts on the real header values; redaction is a display concern only."""
        captured = _capture(headers=(("Authorization", "Bearer sk-secret-value"),))

        assert captured.headers[0][1] == "Bearer sk-secret-value"

    def test_every_masked_name_is_lowercase_so_matching_is_case_insensitive(self) -> None:
        """A mask set holding `Authorization` would silently miss the wire's `authorization`."""
        for name in c.REDACTED_HEADERS:
            assert name == name.lower()

    def test_the_mask_set_is_exactly_five_carriers_plus_two_precautionary(self) -> None:
        """Asserts its own subject set exactly, so it cannot shrink *or* grow unexplained.

        **Five with a carrier** the recorders will meet (§7.2): `x-api-key`
        (Anthropic), `api-key` (Azure and P9b's MiMo), `x-goog-api-key`
        (Gemini), `authorization` (Vertex's OAuth leg) and
        `proxy-authorization` (the CONNECT legs, §5.2.1).

        **Two precautionary**: `cookie` and `set-cookie`. Nothing authenticates
        by cookie today, so nothing exercises them — kept because listing a
        credential nobody sends costs nothing, while omitting one somebody
        later starts sending leaks it into every CI log.

        Exact, not a subset: the subset form could not notice an entry nobody
        had explained, which is how the constant and its prose drifted apart
        in the first place.
        """
        assert set(c.REDACTED_HEADERS) == {
            "authorization",
            "proxy-authorization",
            "x-api-key",
            "api-key",
            "x-goog-api-key",
            "cookie",
            "set-cookie",
        }

    def test_every_masked_header_is_actually_redacted(self) -> None:
        """Membership in the set is not proof the mask reaches the entry.

        Swept over the constant itself rather than a hand-picked few, so an
        eighth entry cannot ship asserted-but-unexercised — which is how four
        of the seven sat untested through four review rounds.

        Each is sent in **upper case** to prove the match is case-insensitive
        per entry, not merely that the constant happens to be lowercase.
        """
        for name in sorted(c.REDACTED_HEADERS):
            rendered = repr(_capture(headers=((name.upper(), "SECRET-VALUE"),)))

            assert "SECRET-VALUE" not in rendered, f"{name} is in the set but not masked"
            assert c.REDACTION_MASK in rendered, f"{name} masked to nothing rather than to the mask"

    def test_every_masked_query_key_is_actually_redacted(self) -> None:
        """The query-side counterpart, swept the same way and for the same reason.

        Asserts the **pair survives with the mask as its value**, not merely
        that the secret is gone. A `repr` that dropped the pair outright, or
        emptied it to `key=`, would satisfy an absence check while hiding a
        missing-credential bug exactly as effectively as a leak hides a present
        one. The header sweep already pins this; the two must not disagree.
        """
        for key in sorted(c.REDACTED_QUERY_KEYS):
            sent = key.upper()
            rendered = repr(_capture(query=f"{sent}=SECRET-VALUE&alt=sse"))

            assert "SECRET-VALUE" not in rendered, f"{key} is in the set but not masked"
            assert f"{sent}={c.REDACTION_MASK}" in rendered, f"{key} was dropped or emptied, not masked"
            assert "alt=sse" in rendered, f"masking {key} destroyed an innocent parameter"


class TestCapturedReply:
    """The response direction's capture, for T-A7."""

    def test_it_carries_status_headers_and_body(self) -> None:
        """`ReplyProjection` reads this; `Reply` is what it produces."""
        reply = c.CapturedReply(status=200, headers=(("Content-Type", "application/json"),), body=b"{}")

        assert (reply.status, reply.body) == (200, b"{}")

    def test_it_redacts_credentials_in_its_repr_too(self) -> None:
        """A reply carries `set-cookie`; the same log-leak applies."""
        reply = c.CapturedReply(status=200, headers=(("Set-Cookie", "session=secret-token"),), body=b"{}")

        assert "secret-token" not in repr(reply)

    @pytest.mark.parametrize(
        ("headers", "expected"),
        [
            ({"Set-Cookie": "session=secret"}, "mapping"),
            ("ab", "not a string"),
            ((("Set-Cookie",),), "pair"),
        ],
        ids=["mapping", "two-character-string", "wrong-length-pair"],
    )
    def test_it_rejects_the_same_malformed_headers_a_request_does(self, headers: object, expected: str) -> None:
        """The reply's guard must not drift from the request's.

        Both types validate headers identically, and both were previously
        hand-copied — so a defect found in one was silently a defect in the
        other. They now share one helper, and this exercises the reply side of
        all three rejection paths so the shared behaviour is pinned from both
        ends rather than assumed from one.
        """
        with pytest.raises(TypeError, match=expected):
            c.CapturedReply(status=200, headers=headers)  # type: ignore[arg-type]

    def test_its_headers_survive_a_generator_too(self) -> None:
        """The one-shot case, on the second type that shares the helper."""
        reply = c.CapturedReply(status=200, headers=(p for p in (("A", "1"),)))  # type: ignore[arg-type]

        assert reply.headers == (("A", "1"),)


class TestImmutability:
    """The oracle diffs projections and must not be able to alter what it compares (R3.8)."""

    def test_rebinding_a_field_is_rejected(self) -> None:
        """`frozen=True` blocks the obvious mutation."""
        part = c.Text("hello")

        with pytest.raises(dataclasses.FrozenInstanceError):
            part.text = "goodbye"  # type: ignore[misc]

    def test_mutating_through_a_mapping_field_is_rejected(self) -> None:
        """`frozen=True` alone does NOT give this — a plain dict field stays mutable."""
        envelope = c.Envelope(model="m", extra={"thinking": {"type": "enabled"}})

        with pytest.raises(TypeError):
            envelope.extra["thinking"] = "tampered"  # type: ignore[index]

    def test_no_projection_type_is_hashable(self) -> None:
        """Forced off so hashability is deterministic, not data-dependent (R3.11).

        A frozen dataclass with no mapping field would otherwise hash happily,
        and §3.3.3's counterpart matching would pass on text-only fixtures then
        raise on the first corpus entry carrying a ``ToolUse``.

        The subject set is **derived from the module**, not hand-listed, so a
        fourteenth projection type shipping hashable is caught — which is
        exactly the accident this guards.
        """
        captures = {c.CapturedRequest, c.CapturedReply}
        projections = [
            obj
            for obj in vars(c).values()
            if dataclasses.is_dataclass(obj) and isinstance(obj, type) and obj not in captures
        ]

        assert len(projections) >= 13, f"expected the projection types, found {projections}"

        for projection in projections:
            assert projection.__hash__ is None, f"{projection.__name__} is hashable"

    def test_a_capture_stays_hashable_because_recorders_key_connections_by_it(self) -> None:
        """The asymmetry is deliberate: only *projections* are unhashable (§7.4)."""
        assert hash(_capture()) == hash(_capture())


class TestPartGrammar:
    """The closed union every reader maps its format onto (R3.4)."""

    def test_all_seven_variants_inhabit_the_union(self) -> None:
        """Asserts its own subject set so a new variant cannot be added unnoticed."""
        variants = [
            c.Text("t"),
            c.ToolUse(id="1", name="get_weather", arguments={"city": "Kyiv"}),
            c.ToolResult(tool_use_id="1", content=(c.Text("sunny"),), is_error=False),
            c.Thinking(text="hmm", signature=None),
            c.Image(digest="d", media_type="image/png", ref=None),
            c.Json(value={"k": "v"}),
            c.Opaque(kind="document", digest="d"),
        ]

        for variant in variants:
            assert isinstance(variant, c.PART_TYPES)
        assert len(variants) == len(c.PART_TYPES)

    def test_tool_arguments_are_a_parsed_mapping_never_a_json_string(self) -> None:
        """Chat Completions encodes arguments as a string and Messages as an object (R3.5)."""
        use = c.ToolUse(id="1", name="f", arguments={"a": 1})

        assert use.arguments["a"] == 1

    def test_two_tool_uses_differing_only_in_an_argument_value_are_unequal(self) -> None:
        """Equality is by value: the diff is what the whole oracle rests on."""
        assert c.ToolUse(id="1", name="f", arguments={"a": 1}) != c.ToolUse(id="1", name="f", arguments={"a": 2})

    def test_a_tool_result_carries_ordered_mixed_content(self) -> None:
        """Converse carries `json`, Anthropic carries `document`; `Text | Image` dropped them (R3.6)."""
        result = c.ToolResult(
            tool_use_id="1",
            content=(
                c.Text("see"),
                c.Image(digest="d"),
                c.Json(value={"rows": 2}),
                c.Opaque(kind="document", digest="d"),
            ),
            is_error=False,
        )

        assert [type(p) for p in result.content] == [c.Text, c.Image, c.Json, c.Opaque]

    def test_tool_call_ids_may_be_absent_because_gemini_carries_none(self) -> None:
        """Gemini's functionCall/functionResponse have no id; a required one forces a fake (R3.9)."""
        use = c.ToolUse(id=None, name="f", arguments={})
        result = c.ToolResult(tool_use_id=None, content=(c.Text("x"),), is_error=False)

        assert use.id is None and result.tool_use_id is None

    def test_an_image_holds_a_digest_and_media_type_but_never_bytes(self) -> None:
        """Carrying bytes would put megabytes into every failure message (R3.7)."""
        names = {f.name for f in dataclasses.fields(c.Image)}

        assert names == {"digest", "media_type", "ref"}

    def test_the_documented_digest_is_sha256_over_the_decoded_bytes(self) -> None:
        """Unpinned, two readers produce different digests for one image and the corpus fails."""
        raw = b"\x89PNG\r\n\x1a\n fake image bytes"

        assert c.image_digest(raw) == hashlib.sha256(raw).hexdigest()

    def test_a_referenced_image_has_no_digest(self) -> None:
        """Gemini's `fileData.fileUri` carries a URI and no bytes to digest."""
        image = c.Image(digest=None, media_type=None, ref="gs://bucket/cat.png")

        assert image.digest is None and image.ref == "gs://bucket/cat.png"

    def test_an_empty_thinking_block_is_distinguishable_from_no_block(self) -> None:
        """P5e and P8 inject empty blocks; P8's trigger is conditional, so absence must be visible."""
        assert (c.Thinking(text="", signature=None),) != ()

    def test_thinking_carries_the_signature_m8_manipulates(self) -> None:
        """Anthropic's `signature` and Gemini's `thoughtSignature` are M8's subject."""
        assert c.Thinking(text="t", signature="sig").signature == "sig"


class TestEnvelopeAndConversation:
    """The projection's two halves, and the register rows that address them."""

    def test_the_three_named_control_fields_the_register_addresses(self) -> None:
        """M1 is `envelope.model`; P17 is `envelope.stream` and `envelope.store` (§3.3.1)."""
        envelope = c.Envelope(model="claude-opus-5", stream=True, store=False)

        assert (envelope.model, envelope.stream, envelope.store) == ("claude-opus-5", True, False)

    def test_a_format_specific_control_field_is_reachable_by_its_wire_key(self) -> None:
        """P2a is `thinking`, P3 `reasoning`, P4 `reasoning_effort`, P10 `reasoning_split`."""
        envelope = c.Envelope(extra={"thinking": {"type": "enabled"}})

        assert envelope.extra["thinking"] == {"type": "enabled"}

    def test_absent_is_distinguishable_from_false(self) -> None:
        """P15 strips `strict`; if absent and False were one value the row would be untestable."""
        assert c.ToolDecl(name="f", strict=None) != c.ToolDecl(name="f", strict=False)

    def test_ordering_is_preserved_across_system_turns_and_tools(self) -> None:
        """Order is meaning: R7's paths are index-based, so drift reports false deltas."""
        conversation = c.Conversation(
            system=(c.Text("a"), c.Text("b")),
            turns=(c.Turn(role="user", parts=(c.Text("1"),)), c.Turn(role="assistant", parts=(c.Text("2"),))),
            tools=(c.ToolDecl(name="x"), c.ToolDecl(name="y")),
        )

        assert [t.text for t in conversation.system] == ["a", "b"]
        assert [t.role for t in conversation.turns] == ["user", "assistant"]
        assert [t.name for t in conversation.tools] == ["x", "y"]

    @pytest.mark.parametrize(
        ("build", "expected"),
        [
            (lambda: c.Turn(role="user", parts=("not a part",)), "Turn.parts"),
            (lambda: c.Reply(parts=(42,)), "Reply.parts"),
            (lambda: c.ToolResult(content=(c.ToolUse(name="f"),)), "ToolResult.content"),
            (lambda: c.Conversation(system=("plain string",)), "Conversation.system"),
            (lambda: c.Conversation(turns=("not a turn",)), "Conversation.turns"),
            (lambda: c.Conversation(tools=("not a tool",)), "Conversation.tools"),
        ],
        ids=["turn-parts", "reply-parts", "tool-result-content", "system", "turns", "tools"],
    )
    def test_a_closed_union_rejects_a_member_outside_it(self, build: object, expected: str) -> None:
        """Closed means enforced — the posture the vocabularies already take.

        Swept across **every** field that declares a closed set rather than the
        one that was reported, because fixing these one at a time is what
        produced four separate rounds of the same finding. `ToolResult.content`
        is deliberately narrower than `Part`: a tool call cannot nest inside a
        tool result, and that claim is now checked rather than merely written.
        """
        with pytest.raises(TypeError, match=expected):
            build()  # type: ignore[operator]

    def test_a_role_outside_the_closed_vocabulary_is_rejected_at_construction(self) -> None:
        """Construction is not parsing, so this is a ValueError, not UnreadableBodyError (R8.1)."""
        with pytest.raises(ValueError):
            c.Turn(role="system", parts=())

    def test_system_is_not_a_role_because_it_lifts_into_the_conversation(self) -> None:
        """R8.2: `system`, `developer` and Responses' `instructions` all lift, never become turns."""
        assert set(c.ROLES) == {"user", "assistant"}

    def test_the_canonical_sampling_set_holds_every_parameter_p13_drops(self) -> None:
        """P13 drops fourteen named parameters; readers must agree all fourteen are sampling."""
        p13 = {
            "temperature",
            "top_p",
            "max_tokens",
            "max_completion_tokens",
            "frequency_penalty",
            "presence_penalty",
            "logprobs",
            "top_logprobs",
            "response_format",
            "stop",
            "n",
            "stream_options",
            "seed",
            "logit_bias",
        }

        assert p13 <= c.SAMPLING_KEYS

    def test_max_completion_tokens_is_not_collapsed_onto_max_tokens(self) -> None:
        """R8.5 maps Responses' `max_output_tokens` onto `max_tokens`; this is a different field."""
        assert {"max_tokens", "max_completion_tokens"} <= c.SAMPLING_KEYS
        assert "max_output_tokens" not in c.SAMPLING_KEYS

    def test_top_k_is_canonical_even_though_chat_completions_lacks_it(self) -> None:
        """Gemini and Converse both carry it; a CC-only set would residualise a legitimate field."""
        assert "top_k" in c.SAMPLING_KEYS

    def test_a_non_canonical_sampling_key_is_rejected_at_construction(self) -> None:
        """The set is closed (R9.1), and closed means enforced.

        Gemini's `generationConfig` members are the likely accident: a reader
        that tips them straight into `sampling` would otherwise be caught by
        nothing here and nothing downstream.
        """
        with pytest.raises(ValueError, match="generationConfig"):
            c.Conversation(sampling={"generationConfig": {"topK": 40}})

    def test_every_canonical_sampling_key_is_accepted(self) -> None:
        """The control: the guard above would pass if construction rejected everything."""
        conversation = c.Conversation(sampling=dict.fromkeys(c.SAMPLING_KEYS, 1))

        assert set(conversation.sampling) == set(c.SAMPLING_KEYS)

    def test_sampling_given_as_something_other_than_a_mapping_is_rejected_clearly(self) -> None:
        """A list of pairs would otherwise be reported as "these keys are not canonical"."""
        with pytest.raises(TypeError, match="mapping"):
            c.Conversation(sampling=[("temperature", 1)])  # type: ignore[arg-type]

    def test_the_stop_reason_vocabulary_is_closed_but_has_an_escape(self) -> None:
        """Gemini adds SAFETY and RECITATION; Ollama reports `load` (R8.7).

        Without `other`, a legitimate safety-blocked reply would fail the run —
        the mistake R9.3 avoids on the request side.
        """
        assert set(c.STOP_REASONS) == {"end_turn", "max_tokens", "stop_sequence", "tool_use", "error", "other"}

    def test_the_stop_reason_escape_does_not_itself_fail_the_run(self) -> None:
        """The escape must not be built out of the thing that fails the run.

        Keeping the wire's value in the residual would have made every
        safety-blocked Gemini reply fail `verify_total`, defeating the escape
        entirely. A value mapped to `other` has been seen and classified, so it
        is accounted for — `stop_reason_raw` is its home.
        """
        blocked = c.Reply(stop_reason="other", stop_reason_raw="SAFETY", consumed=frozenset({"finishReason"}),
                          source={"finishReason": "SAFETY"})

        c.verify_total(blocked)
        assert blocked.stop_reason_raw == "SAFETY"

    def test_the_tool_choice_vocabulary_is_a_constant_not_prose(self) -> None:
        """Four formats spell one concept four ways; six readers must not each guess (R8.6)."""
        assert set(c.TOOL_CHOICE_VALUES) == {"auto", "any", "none"}
        assert c.TOOL_CHOICE_KEY == "tool_choice"

    def test_tool_choice_lands_under_its_single_agreed_key(self) -> None:
        """The one deliberate exception to keying `extra` by the wire key."""
        envelope = c.Envelope(extra={c.TOOL_CHOICE_KEY: "auto"})

        assert c.extra_path(c.TOOL_CHOICE_KEY) == "envelope.extra[tool_choice]"
        assert envelope.extra["tool_choice"] == "auto"

    def test_a_wire_spelling_of_tool_choice_is_rejected(self) -> None:
        """Closed means enforced. Gemini's `AUTO` must be normalised, not passed through."""
        with pytest.raises(ValueError, match="tool_choice"):
            c.Envelope(extra={c.TOOL_CHOICE_KEY: "AUTO"})

    def test_a_named_tool_selection_is_accepted(self) -> None:
        """The control: `tool:<name>` is the fourth legal form (R8.6)."""
        envelope = c.Envelope(extra={c.TOOL_CHOICE_KEY: "tool:get_weather"})

        assert envelope.extra[c.TOOL_CHOICE_KEY] == "tool:get_weather"

    def test_other_entries_in_extra_are_not_validated(self) -> None:
        """`extra` is open by design; `tool_choice` is the one entry with a canonical value."""
        envelope = c.Envelope(extra={"thinking": {"type": "enabled"}, "reasoning_effort": "high"})

        assert envelope.extra["reasoning_effort"] == "high"

    def test_a_wire_stop_reason_is_rejected_rather_than_carried_through(self) -> None:
        """Gemini's `MAX_TOKENS` must map onto the canonical set, or the diff sees two spellings."""
        with pytest.raises(ValueError, match="stop_reason"):
            c.Reply(stop_reason="MAX_TOKENS")

    def test_every_canonical_stop_reason_is_accepted(self) -> None:
        """The control: the guard above would pass if construction rejected everything.

        `other` is passed with its paired raw value, because the two are only
        meaningful together — see the pairing tests below.
        """
        for reason in c.STOP_REASONS:
            raw = "SAFETY" if reason == "other" else None
            assert c.Reply(stop_reason=reason, stop_reason_raw=raw).stop_reason == reason

    def test_other_without_the_wire_value_is_rejected(self) -> None:
        """`other` alone discards the very thing the escape exists to keep.

        T-D10 must be able to tell a Gemini `SAFETY` block from a `RECITATION`;
        a reader that maps both to bare `other` has thrown that away.
        """
        with pytest.raises(ValueError, match="stop_reason_raw"):
            c.Reply(stop_reason="other")

    def test_a_raw_value_beside_a_canonical_reason_is_rejected(self) -> None:
        """The other direction: a stale original left over from a mapped value."""
        with pytest.raises(ValueError, match="only for 'other'"):
            c.Reply(stop_reason="end_turn", stop_reason_raw="STOP")


class TestPathVocabulary:
    """How a delta and a register row name the same location (R7)."""

    def test_the_named_envelope_roots_are_constants_not_hand_assembled_strings(self) -> None:
        """M1's anchor is spelled once, here, so T-W3 and T-D1 cannot drift apart."""
        assert (c.ENVELOPE_MODEL, c.ENVELOPE_STREAM, c.ENVELOPE_STORE) == (
            "envelope.model",
            "envelope.stream",
            "envelope.store",
        )

    def test_a_tool_is_addressed_by_name_because_translators_reorder_them(self) -> None:
        """A positional path would report a delta whenever the list order changed (R7.3)."""
        assert c.tool_path("get_weather", "strict") == "conversation.tools[get_weather].strict"

    def test_a_part_is_addressed_by_turn_and_part_index(self) -> None:
        """§3.3.4: a failure must name the exact turn and part."""
        assert c.part_path(2, 1) == "conversation.turns[2].parts[1]"

    def test_the_remaining_forms_build(self) -> None:
        """Each form exists because a register row or a design section needs it."""
        assert c.extra_path("thinking") == "envelope.extra[thinking]"
        assert c.sampling_path("temperature") == "conversation.sampling[temperature]"
        assert c.system_path(0) == "conversation.system[0]"
        assert c.residual_path("generationConfig.topK") == "residual[generationConfig.topK]"

    def test_header_rows_have_a_form_because_three_register_rows_change_headers(self) -> None:
        """P9a sets User-Agent, P9b swaps the auth scheme, P9c impersonates Codex (§4.3 C1)."""
        assert c.header_path("User-Agent") == "headers[user-agent]"

    def test_bare_collections_are_legal_paths(self) -> None:
        """M5 and M13 change `turns` wholesale; §3.3.1 pins P13 to `conversation.sampling`.

        Asserts the exact spellings, not merely a prefix: a check for
        ``startswith("conversation.")`` would pass on ``conversation.oops``.
        """
        assert (c.CONVERSATION_TURNS, c.CONVERSATION_SYSTEM, c.CONVERSATION_TOOLS, c.CONVERSATION_SAMPLING) == (
            "conversation.turns",
            "conversation.system",
            "conversation.tools",
            "conversation.sampling",
        )

    def test_a_turn_role_has_a_path_of_its_own(self) -> None:
        """R8.1 closes the role vocabulary, so a changed role is a nameable delta."""
        assert c.turn_path(1, "role") == "conversation.turns[1].role"
        assert c.turn_path(1) == "conversation.turns[1]"

    def test_the_route_components_have_paths_because_the_body_cannot_show_them(self) -> None:
        """M14 replaces the destination, P20 encodes Azure's deployment, P21 Vertex's project."""
        assert (c.ROUTE_METHOD, c.ROUTE_SCHEME, c.ROUTE_HOST, c.ROUTE_PATH, c.ROUTE_QUERY) == (
            "route.method",
            "route.scheme",
            "route.host",
            "route.path",
            "route.query",
        )
        assert c.route_path("host") == c.ROUTE_HOST

    def test_a_route_path_rejects_a_component_that_is_not_one(self) -> None:
        """The builder exists to stop hand-assembled strings drifting (R7.4)."""
        with pytest.raises(ValueError):
            c.route_path("deployment")

    def test_the_reply_direction_has_its_own_root(self) -> None:
        """M12 is a response-path row and T-D10 diffs Reply projections."""
        assert c.reply_part_path(0) == "reply.parts[0]"
        assert c.REPLY_STOP_REASON == "reply.stop_reason"
        assert c.reply_usage_path("input_tokens") == "reply.usage[input_tokens]"
        assert c.REPLY_USAGE == "reply.usage"

    def test_a_tool_name_containing_a_dot_survives_the_brackets(self) -> None:
        """Bracket contents are literal, so a dotted name is not shattered (R7.2d)."""
        assert c.tool_path("a.b", "strict") == "conversation.tools[a.b].strict"
        assert c.path_matches("conversation.tools[*].strict", c.tool_path("a.b", "strict"))

    def test_an_unbalanced_bracket_is_rejected_rather_than_silently_mis_split(self) -> None:
        """R7.2d calls such a path "not addressable"; a wrong answer would be worse."""
        with pytest.raises(ValueError):
            c.path_matches("residual[a]].b", "residual[a]")

        with pytest.raises(ValueError):
            c.path_matches("conversation.tools[abc", "conversation.tools[abc]")

    def test_a_row_the_projection_does_not_model_carries_an_explicit_escape(self) -> None:
        """M2, M9, P11, P12 and P16 change everything or nothing nameable (R7.2c)."""
        assert c.NOT_PROJECTABLE == "not projectable"


class TestPathMatching:
    """A register row writes a pattern; a delta is concrete. Assertion 1 is that match (R7.5)."""

    def test_a_concrete_tool_path_matches_the_wildcard_pattern_a_register_row_carries(self) -> None:
        """§3.3.1 writes P15 as `conversation.tools[].strict` — a wildcard over every tool."""
        assert c.path_matches("conversation.tools[*].strict", "conversation.tools[get_weather].strict")

    def test_it_does_not_match_a_sibling_field(self) -> None:
        """The control: a matcher that returned True everywhere would pass the test above."""
        assert not c.path_matches("conversation.tools[*].strict", "conversation.tools[get_weather].description")

    def test_a_wildcard_matches_any_index(self) -> None:
        """Turn and part indices vary per corpus entry; the row is written once."""
        assert c.path_matches("conversation.turns[*].parts[*]", "conversation.turns[3].parts[7]")

    def test_a_bare_collection_pattern_matches_everything_beneath_it(self) -> None:
        """P13 is anchored to `conversation.sampling`, and drops fourteen keys under it."""
        assert c.path_matches("conversation.sampling", "conversation.sampling[temperature]")

    def test_an_exact_path_matches_only_itself(self) -> None:
        """M1's anchor must not accidentally claim a delta on `envelope.stream`."""
        assert c.path_matches("envelope.model", "envelope.model")
        assert not c.path_matches("envelope.model", "envelope.stream")

    def test_bracket_contents_are_literal_so_a_dotted_key_survives(self) -> None:
        """`residual[generationConfig.topK]` must not be re-parsed as two segments (R7.2d)."""
        assert c.path_matches("residual[*]", "residual[generationConfig.topK]")

    def test_a_literal_star_on_the_concrete_side_is_not_special(self) -> None:
        """Wildcards live in patterns only.

        A tool genuinely named `*` appearing in a *delta* must not turn that
        delta into something that matches every pattern.
        """
        assert not c.path_matches("conversation.tools[x].strict", "conversation.tools[*].strict")

    def test_a_pattern_claims_paths_beneath_it_even_when_it_carries_a_wildcard(self) -> None:
        """The case that would have manufactured a false I1 breach.

        T-D1 legitimately reports `conversation.turns[2].parts[0].signature` —
        M8's carrier repair changes exactly that. A row anchored at
        `conversation.turns[*].parts[*]` must claim it, or §3.3.2 assertion 1
        fails the run over a mutation that *is* registered.
        """
        assert c.path_matches("conversation.turns[*].parts[*]", "conversation.turns[2].parts[0].signature")

    def test_a_pattern_never_matches_a_shallower_path(self) -> None:
        """The control: prefix matching runs one way only."""
        assert not c.path_matches("conversation.turns[*].parts[*]", "conversation.turns[2]")

    def test_the_old_empty_bracket_spelling_still_matches(self) -> None:
        """§3.3.1 wrote P15 as `conversation.tools[].strict` before this vocabulary existed.

        A row carried over in the old notation must not silently match nothing.
        """
        assert c.path_matches("conversation.tools[].strict", "conversation.tools[get_weather].strict")

    def test_a_sibling_collection_is_not_claimed(self) -> None:
        """`conversation.tools` must not claim a delta under `conversation.turns`."""
        assert not c.path_matches(c.CONVERSATION_TOOLS, "conversation.turns[0].parts[0]")

    def test_the_prefix_stops_at_a_bracket(self) -> None:
        """The second rule sitting next to the first, and the one that surprises.

        Bracket contents are literal, so a pattern naming a *parent key* does
        not claim paths nested under it. Only the wildcard or the bare
        collection reaches inside. A T-W3 author who has internalised "a
        pattern is a prefix" gets this wrong once, so it is pinned.
        """
        assert not c.path_matches("residual[generationConfig]", "residual[generationConfig.topK]")
        assert c.path_matches("residual[*]", "residual[generationConfig.topK]")
        assert c.path_matches("residual", "residual[generationConfig.topK]")


def _capture(
    *,
    method: str = "POST",
    scheme: str = "https",
    host: str = "api.example.com",
    path: str = "/v1/messages",
    query: str = "",
    headers: tuple[tuple[str, str], ...] = (),
    body: bytes = b"{}",
) -> c.CapturedRequest:
    """Build a capture with everything but the field under test defaulted.

    Args:
        method: HTTP method.
        scheme: URL scheme.
        host: URL host.
        path: URL path.
        query: Raw query string.
        headers: Ordered header pairs, casing preserved.
        body: Raw body bytes.

    Returns:
        A :class:`~harness.contract.CapturedRequest` for one assertion.
    """
    return c.CapturedRequest(
        method=method, scheme=scheme, host=host, path=path, query=query, headers=headers, body=body
    )


#: One body, read four different ways below. ``mystery`` is the key none of the
#: readers were written to expect; ``generationConfig`` is the nesting key that
#: makes the nested-drop case possible.
_BODY = {
    "model": "claude-opus-5",
    "mystery": "an unregistered field",
    "generationConfig": {"topK": 40},
}


def _project(consumed: set[str], residual: dict[str, object], source: dict[str, object] | None = None) -> c.Request:
    """Build a projection claiming a given account of the body.

    Args:
        consumed: Top-level body keys the stub reader claims to have mapped.
        residual: Paths the stub reader could not classify, with their values.
        source: The body the stub reader read; defaults to :data:`_BODY`.

    Returns:
        A :class:`~harness.contract.Request` standing in for a reader's output.
    """
    return c.Request(
        envelope=c.Envelope(model="claude-opus-5"),
        conversation=c.Conversation(),
        residual=residual,
        consumed=frozenset(consumed),
        source=source if source is not None else _BODY,
    )


class TestTotalityFalsification:
    """The harness rule (plan §1.4): a deliberate defect this contract must detect.

    Four stub readers over one body, differing only in how they account for the
    keys they were not written to expect.  This is why :attr:`Request.consumed`
    exists — see :meth:`test_a_reader_that_drops_an_unknown_key_is_caught`.
    """

    def test_a_reader_that_drops_an_unknown_key_is_caught(self) -> None:
        """**The ticket's named falsification case.**

        This is the one a residual-only rule cannot catch.  The reader neither
        maps ``mystery`` nor residualises it, so the residual is *empty* and a
        "residual must be empty" check would pass.  Totality is decidable only
        against the source body, which is the whole reason the projection
        declares what it consumed.
        """
        dropped = _project(consumed={"model", "generationConfig"}, residual={})

        with pytest.raises(c.DroppedFieldsError, match="mystery"):
            c.verify_total(dropped)

    def test_a_reader_that_residualises_an_unknown_key_is_caught(self) -> None:
        """§3.3.1: a non-empty residual fails the run — it is neither a diff nor ignored."""
        residualised = _project(consumed={"model", "generationConfig"}, residual={"mystery": "an unregistered field"})

        with pytest.raises(c.ResidualFieldsError, match="mystery"):
            c.verify_total(residualised)

    def test_a_nested_key_the_reader_could_not_classify_fails_closed(self) -> None:
        """A residual keyed by a *path* fails the run, not just a top-level one.

        Gemini puts every sampling parameter under ``generationConfig``, so a
        reader must be able to say "I could not classify ``topK`` inside a key
        I did handle" and have that fail. Path-keyed residuals are what allow
        it.

        Note what this does **not** prove — see
        :meth:`test_a_nested_key_the_reader_silently_drops_is_a_known_limit`.
        """
        nested = _project(
            consumed={"model", "mystery", "generationConfig"},
            residual={"generationConfig.topK": 40},
        )

        with pytest.raises(c.ResidualFieldsError, match="generationConfig.topK"):
            c.verify_total(nested)

    def test_a_nested_key_the_reader_silently_drops_is_a_known_limit(self) -> None:
        """The boundary of the totality rule, pinned so it is deliberate.

        ``consumed`` holds **top-level** keys, so a reader that claims
        ``generationConfig`` and silently ignores ``topK`` inside it passes.
        Catching that would need the contract to walk the body itself, which
        would make it a second reader — and the oracle must not be written in
        terms of anything that reads bodies (§3.3.1).

        What closes it instead: each reader's own L1 tests against its format's
        published examples (§7.4), and T-D8's "residual empty across the whole
        corpus" over all seven readers.

        This test fails the day someone extends totality to nested keys — which
        is the point. It is not an endorsement, it is a boundary marker.
        """
        nested_drop = _project(consumed={"model", "mystery", "generationConfig"}, residual={})

        c.verify_total(nested_drop)

    def test_a_reader_that_accounts_for_every_key_passes(self) -> None:
        """**The control.**

        Without it, a :func:`verify_total` that raised unconditionally would
        satisfy all three cases above — a harness that cannot pass is as useless
        as one that cannot fail.
        """
        total = _project(consumed={"model", "mystery", "generationConfig"}, residual={})

        c.verify_total(total)

    def test_a_dropped_key_is_reported_ahead_of_a_residual(self) -> None:
        """A silent drop is the more dangerous defect, so it names the failure (R2.6).

        The message must name **every** dropped key, not just the first: a
        checker truncating the list would leave a maintainer fixing one field
        and re-running to discover the next.
        """
        both = _project(consumed=set(), residual={"mystery": "x"})

        with pytest.raises(c.DroppedFieldsError, match=r"generationConfig.*model"):
            c.verify_total(both)

    def test_a_reply_that_drops_a_key_is_caught_too(self) -> None:
        """The rule is bidirectional (R2.4), and only its *passing* case was covered.

        T-A7 and T-D10 read this direction. A checker that silently returned
        for anything that is not a ``Request`` would satisfy every other case
        in this class — a harness blind on one side is the shape §1.4 exists to
        prevent, and it would go unnoticed until T-D10.
        """
        dropped = c.Reply(consumed=frozenset({"content"}), source={"content": [], "surprise": 1})

        with pytest.raises(c.DroppedFieldsError, match="surprise"):
            c.verify_total(dropped)

    def test_a_nested_residual_does_not_account_for_its_top_level_parent(self) -> None:
        """The tempting wrong reading of this contract's own split (R2.2/R2.3).

        ``consumed`` holds top-level keys while ``residual`` holds full paths,
        which invites a checker that credits ``generationConfig.topK`` to
        ``generationConfig``. It must not: the reader residualised something
        *inside* the key without ever claiming the key itself, so the key is
        still dropped.
        """
        nested_only = _project(
            consumed=set(),
            residual={"generationConfig.topK": 40},
            source={"generationConfig": {"topK": 40}},
        )

        with pytest.raises(c.DroppedFieldsError, match="generationConfig"):
            c.verify_total(nested_only)

    def test_a_key_claimed_but_absent_from_the_body_is_named_in_the_message(self) -> None:
        """A typo'd claim already fails as a drop; this stops it being mis-diagnosed (R2.7).

        The realistic accident: a reader claims ``max_token`` while the body
        carries ``max_tokens``.  The real key is then in neither account, so it
        already fails as a drop — but reported alone that reads as "the reader
        ignored max_tokens", when in fact it handled it and misspelled the
        claim.  Naming the absent claim alongside points at the actual defect.

        Claimed-but-absent is deliberately **not** its own error: on its own it
        harms nothing, and a second error type would fire on a body that is
        fully accounted for.
        """
        typo = _project(
            consumed={"max_token"},
            residual={},
            source={"max_tokens": 4096},
        )

        with pytest.raises(c.DroppedFieldsError, match="claimed but absent.*max_token"):
            c.verify_total(typo)

    def test_a_claim_absent_from_the_body_does_not_fail_a_run_on_its_own(self) -> None:
        """The control for the test above: it is a diagnostic, not a failure (R2.7)."""
        over_claimed = _project(
            consumed={"max_tokens", "not_in_this_body"},
            residual={},
            source={"max_tokens": 4096},
        )

        c.verify_total(over_claimed)

    def test_a_key_in_both_accounts_is_treated_as_a_residual(self) -> None:
        """R2.8 states the edge rather than leaving each reader to discover it."""
        both = _project(consumed={"model", "mystery", "generationConfig"}, residual={"mystery": "x"})

        with pytest.raises(c.ResidualFieldsError):
            c.verify_total(both)

    def test_it_accepts_a_reply_as_well_as_a_request(self) -> None:
        """M12 is a response-path row and T-D10 diffs replies (R2.4)."""
        reply = c.Reply(parts=(), stop_reason="end_turn", usage={}, residual={}, consumed=frozenset({"content"}),
                        source={"content": []})

        c.verify_total(reply)


class TestProtocols:
    """What the six readers implement (R4)."""

    def test_a_conforming_reader_satisfies_the_request_protocol(self) -> None:
        """`isinstance`, never `issubclass` — the latter raises on a data protocol."""
        assert isinstance(_StubReader(), c.Projection)

    def test_a_class_missing_the_method_does_not_satisfy_it(self) -> None:
        """The control for the test above."""

        class NotAReader:
            """A class that declares the format but cannot read one."""

            wire_format = c.WireFormat.GEMINI

        assert not isinstance(NotAReader(), c.Projection)

    def test_read_request_takes_a_whole_capture_not_a_body(self) -> None:
        """Gemini carries the model and the operation in the URL (§3.3.5, R4.2).

        `isinstance` checks member presence only, never signatures, so this
        annotation check is what actually pins the parameter type.

        Resolved with ``get_type_hints`` rather than read from
        ``__annotations__``: the module uses ``from __future__ import
        annotations``, so the raw attribute holds the *string*
        ``"CapturedRequest"`` and would compare equal to a same-named class
        from anywhere at all.
        """
        hints = get_type_hints(c.Projection.read_request)

        assert hints["captured"] is c.CapturedRequest

    def test_the_reply_protocol_is_separate_and_a_request_reader_does_not_satisfy_it(self) -> None:
        """§3.3.1: response translation "is a different claim and it gets a different test"."""
        assert not isinstance(_StubReader(), c.ReplyProjection)

    def test_both_protocols_declare_the_wire_format_the_oracle_selects_on(self) -> None:
        """§7.4 selects a projection by the shape observed on the wire, in both directions."""
        assert "wire_format" in c.Projection.__annotations__
        assert "wire_format" in c.ReplyProjection.__annotations__

    def test_the_wire_format_enum_names_exactly_the_six_formats_the_design_lists(self) -> None:
        """A bare `str` would let six authors spell one format three ways (R4.4)."""
        assert {f.value for f in c.WireFormat} == {
            "anthropic_messages",
            "chat_completions",
            "openai_responses",
            "gemini",
            "bedrock_converse",
            "ollama_chat",
        }


class _StubReader:
    """A minimal conforming request projection, for the protocol tests."""

    wire_format = c.WireFormat.ANTHROPIC_MESSAGES

    def read_request(self, captured: c.CapturedRequest) -> c.Request:
        """Return an empty projection.

        Args:
            captured: The request as observed on the wire.

        Returns:
            A projection accounting for nothing, which is enough for a
            protocol-conformance check.
        """
        return _project(consumed=set(), residual={}, source={})


def test_unreadable_body_error_is_named_by_the_contract() -> None:
    """Six readers would otherwise raise six types and T-D1 would catch `Exception` (R2.10)."""
    assert issubclass(c.UnreadableBodyError, Exception)


#: Any form of reaching into the product package. Deliberately not a
#: ``startswith`` check: that missed ``from src.kitty import ...``, a double
#: space after the keyword, and both dynamic forms.
#: The static half is anchored to the start of a line (with leading whitespace
#: allowed, so an indented import still matches) rather than floating, so prose
#: mentioning an import in a comment or docstring cannot trip it. The two
#: dynamic forms stay unanchored, because they appear mid-expression.
_KITTY_IMPORT = re.compile(
    r"^\s*(?:from|import)\s+(?:src\.)?kitty\b"
    r"|import_module\(\s*[\"'](?:src\.)?kitty"
    r"|__import__\(\s*[\"'](?:src\.)?kitty"
)


def test_the_contract_imports_nothing_from_kitty() -> None:
    """The oracle must not be written in terms of the code under test (§3.3.1, §7.4).

    This is the single structural guarantee the whole I1 claim rests on: a
    projection that asked kitty how to read a body would inherit kitty's bugs,
    and the oracle would prove self-consistency rather than fidelity.

    Read from source rather than by inspecting imports, so an import inside a
    function body is caught as well as a module-level one.
    """
    source = Path(c.__file__).read_text(encoding="utf-8")

    # Assert the subject was actually read: a guard that passes on an empty
    # string is indistinguishable from one that cannot fail (house rule,
    # `tests/test_egress_coverage.py`).
    assert len(source) > 1000, "read no meaningful source; the guard would pass vacuously"

    offending = [line.strip() for line in source.splitlines() if _KITTY_IMPORT.search(line)]

    assert offending == [], f"contract.py must not import kitty: {offending}"


def test_the_import_guard_actually_fires_on_every_form_it_claims_to_catch() -> None:
    """The positive control for the guard above.

    Without it, a pattern that had stopped matching anything would read as a
    clean bill of health forever.
    """
    forms = [
        "from kitty.bridge import server",
        "import kitty",
        "from  kitty import server",
        "import  kitty.bridge",
        "from src.kitty import server",
        'mod = importlib.import_module("kitty.bridge.server")',
        '__import__("kitty")',
        'mod = importlib.import_module("src.kitty.bridge.server")',
        '__import__("src.kitty")',
    ]

    undetected = [form for form in forms if not _KITTY_IMPORT.search(form)]

    assert undetected == [], f"the guard would miss these: {undetected}"


def test_the_import_guard_does_not_fire_on_innocent_text() -> None:
    """The negative control: a pattern matching everything would also pass above."""
    innocent = [
        "# kitty-bridge is the product under test",
        "from harness import contract",
        "kitty = 1",
        "# never write `import kitty` in this module",
    ]

    assert [line for line in innocent if _KITTY_IMPORT.search(line)] == []
