"""The corpus format, the scrubber and the loader — T-W6 (KBR-29).

``.system_design/TEST_SUITE.md`` §7.1 · requirements
``.requirements/20260912T113118Z_corpus_format_scrubber_loader/REQUIREMENTS.md``.

Everything here is L1: the functions under test are pure, or take a ``tmp_path``
root.  The guards that read the **committed** corpus and its README live in
:mod:`tests.harness.test_corpus_lint` at L2, following the split
:mod:`tests.harness.test_register` and
:mod:`tests.harness.test_register_agreement` set.

Per plan §1.4 the scrubber ships with the deliberate defects it must detect:
:class:`TestTheScrubberIsFalsifiable` breaks the pattern table and the scan in
turn and asserts each break is noticed.  The false-positive control matters as
much as the positive one — §3.3.3 requires the text ``Please explain how
kitty-bridge works`` to survive byte-identically, and a scrubber that mangles
legitimate content breaks I1 in the act of defending the repository.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import pytest

from harness import corpus as k
from harness.contract import REDACTED_HEADERS, REDACTED_QUERY_KEYS, CapturedRequest, WireFormat
from harness.register import Trigger

#: A body shaped like a real Claude Code request and carrying **no** secret:
#: a thinking-block signature, a ``toolu_`` identifier, a base64 image, a large
#: ``max_tokens``, the §3.3.3 regression text, and prose using the very words
#: the ``assigned_secret`` rule keys on.  Every one of these is a long opaque
#: run that a high-entropy rule would flag; none of them is a credential.
CLEAN_BODY = json.dumps(
    {
        "model": "claude-opus-4-20250514",
        "max_tokens": 32000,
        "stream": True,
        "messages": [
            {"role": "user", "content": "Please explain how kitty-bridge works"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "thinking",
                        "thinking": "The user wants an explanation.",
                        "signature": "ErcBCkYIBBgCKkDq8mQ3Xv2LpR7sT1uWbNcZaYxE9fGhJkLmNoPqRsTuVwXyZ0123456789AbCdEf",
                    },
                    {"type": "tool_use", "id": "toolu_01A9FKtPqRs7uVwXyZ012345", "name": "Read", "input": {}},
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNkYPhfDwAChwGA60e6kgAA",
                        },
                    },
                    {
                        "type": "text",
                        "text": "Explain where the api_key is validated and how the secret token is read.",
                    },
                ],
            },
        ],
    }
).encode()

#: The same body with five secrets planted, each of a different class.
DIRTY_BODY = json.dumps(
    {
        "messages": [
            {
                "role": "user",
                "content": (
                    "ANTHROPIC_API_KEY=sk-ant-api03-" + "x" * 93 + "AA\n"
                    "AWS_ACCESS_KEY_ID=" + "AKIA" + "234567ABCDEFGH34" + "\n"
                    "GOOGLE_API_KEY=AIza" + "b" * 35 + "\n"
                    'file: /home/someuser/projects/notes.md\n'
                    "contact: someone@example.com"
                ),
            }
        ]
    }
).encode()


#: A well-formed Anthropic key, assembled at import rather than spelled.
#:
#: GitHub push protection blocks a pushed file containing a partner-pattern key
#: literal, so a test that spelled one out would block the pull request that
#: carries it. The repository's existing fixtures follow the same convention.
PLANTED_KEY = "sk-ant-" + "api03-" + "x" * 93 + "AA"

#: One example of every shape the table knows, for the adjacency sweep.
#:
#: Assembled rather than spelled for :data:`PLANTED_KEY`'s reason — GitHub push
#: protection blocks a pushed file containing a partner-pattern key literal.
SECRET_SHAPES: tuple[str, ...] = (
    "sk-ant-" + "api03-" + "x" * 93 + "AA",
    "sk-proj-" + "y" * 74 + "T3BlbkFJ" + "z" * 74,
    "AIza" + "b" * 35,
    "AKIA" + "234567ABCDEFGH34",
    "ghp_" + "a" * 36,
    "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uh",
    "-----BEGIN RSA PRIVATE KEY-----",
    "Bearer abcdefghij0123456789xyz",
    'SECRET="9f8a7b6c5d4e3f2a1b0c9d8e"',
    "someone@example.com",
    "/home/someuser/x",
)

#: Every ordered pair of :data:`SECRET_SHAPES` at three separations.
#:
#: The empty separator is the one that matters and the one no hand-written
#: fixture contains: it is where a word-boundary anchor on the second secret has
#: nothing to anchor against until the first is redacted.
ADJACENT_PAIRS: tuple[bytes, ...] = tuple(
    f"{first}{separator}{second}".encode()
    for first in SECRET_SHAPES
    for second in SECRET_SHAPES
    for separator in ("", " ", ",")
)

#: A body carrying :data:`PLANTED_KEY`, for the lint's falsification cases.
PLANTED_BODY = json.dumps({"messages": [{"role": "user", "content": PLANTED_KEY}]}).encode()


def capture(body: bytes = b"{}", **overrides: object) -> CapturedRequest:
    """Return a Messages-shaped capture, overridable field by field.

    Args:
        body: The raw body bytes.
        **overrides: Any :class:`~harness.contract.CapturedRequest` field.

    Returns:
        The capture.
    """
    fields: dict[str, object] = {
        "method": "POST",
        "scheme": "https",
        "host": "api.anthropic.com",
        "path": "/v1/messages",
        "query": "",
        "headers": [("Host", "api.anthropic.com"), ("anthropic-version", "2023-06-01")],
        "body": body,
    }
    fields.update(overrides)
    return CapturedRequest(**fields)  # type: ignore[arg-type]


def entry(**overrides: object) -> k.CorpusEntry:
    """Return a valid synthetic entry, overridable field by field.

    Args:
        **overrides: Any :class:`~harness.corpus.CorpusEntry` field.

    Returns:
        The entry.
    """
    fields: dict[str, object] = {
        "id": "sample",
        "description": "A sample entry.",
        "origin": k.SYNTHETIC,
        "origin_note": "Hand-written for the tests.",
        "captured_from": "",
        "captured_at": "",
        "request": capture(CLEAN_BODY),
        "wire_format": WireFormat.ANTHROPIC_MESSAGES,
        "known_non_secrets": (),
        "triggers_met": frozenset(),
        "triggers_absent": frozenset(),
    }
    fields.update(overrides)
    return k.CorpusEntry(**fields)  # type: ignore[arg-type]


#: Sentinel asking :func:`manifest_for` to remove a key rather than set it.
#:
#: ``None`` cannot serve: the format accepts JSON ``null`` for ``wire_format``,
#: so overloading it would make "set this to null" and "delete this" the same
#: request — and the test for the first would silently exercise the second.
DELETE = object()


def manifest_for(root: Path, body: bytes = CLEAN_BODY, **overrides: object) -> Path:
    """Write a valid manifest and body to ``root`` and return the manifest path.

    Args:
        root: The corpus directory.
        body: The sidecar's bytes. The digest follows it, so a test can commit
            an entry the way a careless author would — manifest and body
            consistent with each other, and the secret still in the file.
        **overrides: Manifest keys to replace. Pass :data:`DELETE` to remove a
            key; ``None`` sets it to JSON ``null``, which is a value the format
            accepts for ``wire_format``.

    Returns:
        The manifest's path.
    """
    data: dict[str, object] = {
        "id": "sample",
        "description": "A sample entry.",
        "origin": k.SYNTHETIC,
        "origin_note": "Hand-written for the tests.",
        "captured_from": "",
        "captured_at": "",
        "method": "POST",
        "scheme": "https",
        "host": "api.anthropic.com",
        "path": "/v1/messages",
        "query": "",
        "headers": [["Host", "api.anthropic.com"]],
        "body_file": "sample.body",
        "body_sha256": hashlib.sha256(body).hexdigest(),
        "wire_format": WireFormat.ANTHROPIC_MESSAGES.value,
        "known_non_secrets": [],
        "triggers_met": [],
        "triggers_absent": [],
    }
    for key, value in overrides.items():
        if value is DELETE:
            data.pop(key, None)
        else:
            data[key] = value

    root.mkdir(parents=True, exist_ok=True)
    (root / "sample.body").write_bytes(body)
    path = root / f"{data.get('id', 'sample')}.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


class TestTheFormatIsClosed:
    """A manifest key the loader ignores is a declaration nobody made."""

    def test_a_valid_manifest_loads(self, tmp_path: Path) -> None:
        """The positive control: without it every rejection test below could pass vacuously."""
        loaded = k.load_entry(manifest_for(tmp_path))

        assert loaded.id == "sample"

    def test_a_misspelled_key_is_rejected(self, tmp_path: Path) -> None:
        """`trigers_met` must fail, not silently mean "this entry declares nothing".

        This is the realistic mistake, and the one §3.3.2 assertion 2 cannot
        survive: an entry that declares no complement is an entry the assertion
        quantifies over without anyone having vetted it.
        """
        path = manifest_for(tmp_path, trigers_met=[])

        with pytest.raises(k.CorpusEntryError, match="unknown manifest key"):
            k.load_entry(path)

    def test_a_missing_key_is_rejected(self, tmp_path: Path) -> None:
        """Every key is required to be present, so an entry cannot be half-written."""
        path = manifest_for(tmp_path, description=DELETE)

        with pytest.raises(k.CorpusEntryError, match="missing manifest key"):
            k.load_entry(path)

    def test_the_id_must_equal_the_filename(self, tmp_path: Path) -> None:
        """Two names for one entry is two things a failure message could mean."""
        path = manifest_for(tmp_path, id="other")
        path.rename(tmp_path / "sample.json")

        with pytest.raises(k.CorpusEntryError, match="must equal the filename stem"):
            k.load_entry(tmp_path / "sample.json")

    def test_headers_as_an_object_are_rejected(self, tmp_path: Path) -> None:
        """A mapping loses duplicates and casing, which §4.3 C1 asserts on."""
        path = manifest_for(tmp_path, headers={"Host": "api.anthropic.com"})

        with pytest.raises(k.CorpusEntryError, match="headers must be a list"):
            k.load_entry(path)

    def test_a_header_pair_of_the_wrong_arity_is_rejected(self, tmp_path: Path) -> None:
        """`CapturedRequest` raises `TypeError`; the format promises `CorpusEntryError`."""
        path = manifest_for(tmp_path, headers=[["Host", "h", "extra"]])

        with pytest.raises(k.CorpusEntryError):
            k.load_entry(path)

    def test_a_rejection_names_the_entry(self, tmp_path: Path) -> None:
        """A corpus failure that does not say which entry is a corpus-wide search."""
        path = manifest_for(tmp_path, id="distinctive")

        with pytest.raises(k.CorpusEntryError, match="distinctive"):
            k.load_entry(path.rename(tmp_path / "renamed.json"))

    def test_a_missing_body_file_is_rejected(self, tmp_path: Path) -> None:
        """An entry whose body is gone would otherwise load as an empty request."""
        path = manifest_for(tmp_path)
        (tmp_path / "sample.body").unlink()

        with pytest.raises(k.CorpusEntryError, match="does not exist"):
            k.load_entry(path)

    @pytest.mark.parametrize("name", ["other.body", "../escape.body", "sample.txt"])
    def test_a_body_file_that_is_not_the_entrys_own_is_rejected(self, tmp_path: Path, name: str) -> None:
        """The name is fixed by the format, not followed.

        `write_entry` always writes `<id>.body`, so a manifest naming anything
        else describes a file a rewrite would orphan — and `../escape.body`
        would leave the corpus directory entirely, raising `ValueError` from
        `Path.with_name` rather than the `CorpusEntryError` every rejection
        promises.
        """
        path = manifest_for(tmp_path, body_file=name)

        with pytest.raises(k.CorpusEntryError, match="it must be"):
            k.load_entry(path)

    def test_a_hand_written_dotfile_entry_is_rejected(self, tmp_path: Path) -> None:
        """The load side validates the stem too, and that is not belt-and-braces.

        `Path.glob("*.json")` **does** match a leading dot — verified, because
        shell globbing does not and the difference is the whole point. So a
        hand-written `.hidden.json` is loaded by `load_corpus`, and an entry the
        writer would refuse to create must not be one the reader accepts.
        """
        path = manifest_for(tmp_path, id=".hidden")
        path.rename(tmp_path / ".hidden.json")
        (tmp_path / "sample.body").rename(tmp_path / ".hidden.body")

        with pytest.raises(k.CorpusEntryError, match="legal entry id"):
            k.load_entry(tmp_path / ".hidden.json")

    @pytest.mark.parametrize("field_name", ["method", "scheme", "host", "path", "query"])
    def test_a_non_string_request_field_is_rejected(self, tmp_path: Path, field_name: str) -> None:
        """A frozen dataclass enforces no annotation at runtime.

        `CapturedRequest.__post_init__` validates the headers and nothing else,
        so `"query": 5` loaded cleanly and died much later as `AttributeError:
        'int' object has no attribute 'split'` inside `findings` — and a
        non-string `path` was quieter still, round-tripping back into the
        manifest untouched.
        """
        path = manifest_for(tmp_path, **{field_name: 5})

        with pytest.raises(k.CorpusEntryError, match=f"{field_name} must be a string"):
            k.load_entry(path)

    def test_a_manifest_that_is_not_json_is_rejected(self, tmp_path: Path) -> None:
        """Named explicitly so the failure says so, rather than escaping as a `JSONDecodeError`."""
        path = manifest_for(tmp_path)
        path.write_text("{not json", encoding="utf-8")

        with pytest.raises(k.CorpusEntryError, match="not JSON"):
            k.load_entry(path)


class TestProvenanceIsRequired:
    """The refresh cadence is unactionable if entries do not say what they are."""

    def test_a_synthetic_entry_needs_a_reason(self, tmp_path: Path) -> None:
        """Plan §6 requires the reason recorded beside the fixture for T-C4 and T-C6.

        §7.1's rationale is that captures are evidence and hand-written bodies
        are belief; an unexplained synthetic entry is belief filed as evidence.
        """
        path = manifest_for(tmp_path, origin_note="")

        with pytest.raises(k.CorpusEntryError, match="origin_note"):
            k.load_entry(path)

    def test_a_captured_entry_needs_its_claude_code_version(self, tmp_path: Path) -> None:
        """The product owner's cadence is "re-capture on a version bump".

        An entry that does not record which version it came from cannot be told
        stale from current, so the cadence would have nothing to act on.
        """
        path = manifest_for(tmp_path, origin=k.CAPTURED, origin_note="", captured_from="", captured_at="2026-09-12")

        with pytest.raises(k.CorpusEntryError, match="captured_from"):
            k.load_entry(path)

    def test_a_captured_entry_needs_a_date(self, tmp_path: Path) -> None:
        """The version alone does not say when, and a version can be re-released."""
        path = manifest_for(
            tmp_path, origin=k.CAPTURED, origin_note="", captured_from="claude-code/1.2.3", captured_at=""
        )

        with pytest.raises(k.CorpusEntryError, match="captured_at"):
            k.load_entry(path)

    def test_an_unknown_origin_is_rejected(self, tmp_path: Path) -> None:
        """Two values, closed, so no third spelling escapes both sets of requirements."""
        path = manifest_for(tmp_path, origin="recorded")

        with pytest.raises(k.CorpusEntryError, match="origin must be"):
            k.load_entry(path)


class TestTriggersHaveThreeStates:
    """Met, explicitly absent, and silent — and silence is not absence."""

    def test_an_unknown_trigger_name_is_rejected(self, tmp_path: Path) -> None:
        """`Trigger` is closed so two authors cannot spell one condition two ways.

        A free-form name here would defeat that at the point of use.
        """
        path = manifest_for(tmp_path, triggers_met=["over_compation_budget"])

        with pytest.raises(k.CorpusEntryError, match="unknown trigger"):
            k.load_entry(path)

    def test_a_trigger_cannot_be_both_met_and_absent(self, tmp_path: Path) -> None:
        """A contradiction would make the entry serve as its own complement."""
        path = manifest_for(
            tmp_path, triggers_met=["over_compaction_budget"], triggers_absent=["over_compaction_budget"]
        )

        with pytest.raises(k.CorpusEntryError, match="both met and absent"):
            k.load_entry(path)

    def test_a_declared_trigger_is_indexed(self) -> None:
        """The index is what T-D8 reads to find a row's trigger case."""
        met = entry(triggers_met=frozenset({Trigger.ORPHAN_TOOL_RESULT}))

        assert k.entries_meeting([met], Trigger.ORPHAN_TOOL_RESULT) == (met,)

    def test_an_explicitly_absent_trigger_is_indexed_as_a_complement(self) -> None:
        """§3.3.2 assertion 2 needs the complement, and only an explicit one will do."""
        absent = entry(triggers_absent=frozenset({Trigger.OVER_COMPACTION_BUDGET}))

        assert k.entries_without([absent], Trigger.OVER_COMPACTION_BUDGET) == (absent,)

    def test_silence_is_offered_as_neither(self) -> None:
        """The load-bearing one.

        If absence meant "not met", every entry whose author never considered a
        trigger would be silently offered as its complement, and §3.3.2's second
        assertion would run over entries nobody vetted.
        """
        silent = entry()

        assert k.entries_meeting([silent], Trigger.GEMINI_PROTOCOL) == ()
        assert k.entries_without([silent], Trigger.GEMINI_PROTOCOL) == ()

    def test_a_met_trigger_is_not_also_a_complement(self) -> None:
        """The two indexes must partition, or a trigger case would prove its own absence."""
        met = entry(triggers_met=frozenset({Trigger.ORPHAN_TOOL_RESULT}))

        assert k.entries_without([met], Trigger.ORPHAN_TOOL_RESULT) == ()


class TestTheScrubberRemovesSecrets:
    """Every class in the table, positively."""

    @pytest.mark.parametrize(
        ("name", "secret"),
        [
            ("anthropic_key", "sk-ant-api03-" + "x" * 93 + "AA"),
            ("openai_key", "sk-proj-" + "y" * 74 + "T3BlbkFJ" + "z" * 74),
            ("gcp_key", "AIza" + "b" * 35),
            ("aws_key_id", "AKIA" + "234567ABCDEFGH34"),
            ("github_token", "ghp_" + "a" * 36),
            ("jwt", "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dBjftJeZ4CVPmB92K27uhbUJU1p1r"),
            ("private_key", "-----BEGIN RSA PRIVATE KEY-----"),
            ("bearer_token", "Bearer abcdefghij0123456789xyz"),
            ("assigned_secret", 'SERVICE_SECRET="9f8a7b6c5d4e3f2a1b0c9d8e"'),
            ("email", "someone@example.com"),
            ("home_path", "/home/someuser/notes.md"),
        ],
    )
    def test_each_class_is_found_and_replaced(self, name: str, secret: str) -> None:
        """One case per class, so a class that stops working fails alone."""
        scrubbed = k.scrub(capture(secret.encode()))

        assert k.REDACTION.format(name=name).encode() in scrubbed.body

    def test_a_credential_header_value_is_replaced_whole(self) -> None:
        """A credential header's value IS the secret, whatever shape it has.

        Pattern-matching it would leave an API key in a format the table has
        never seen sitting in the file.
        """
        scrubbed = k.scrub(capture(headers=[("X-Api-Key", "a-key-of-no-known-shape")]))

        assert dict(scrubbed.headers)["X-Api-Key"] == k.REDACTION.format(name="credential_header")

    def test_a_credential_in_the_path_is_scrubbed(self) -> None:
        """The routing fields were exempt, and the gap was real.

        A path of `/v1/key/<token>/messages` committed the token and linted
        clean, while this module's docstring, the README's table and §7.1.1 all
        claimed hostnames and credentials were removed.
        """
        capture_with_key = capture(path=f"/v1/key/{PLANTED_KEY}/messages")

        assert PLANTED_KEY not in k.scrub(capture_with_key).path
        assert [f.where for f in k.findings(capture_with_key)] == ["path"]

    def test_an_operator_literal_reaches_the_host(self) -> None:
        """An internal hostname has no shape, so `extra` is the only mechanism.

        It reached the body, the headers and the query but not the host — so an
        operator could not redact their own build box even by naming it.
        """
        internal = "internal-build-box.corp.example"

        assert internal not in k.scrub(capture(host=internal), extra=(internal,)).host

    def test_a_credential_query_value_is_replaced_whole(self) -> None:
        """Gemini carries its credential in the URL, which `query` preserves verbatim."""
        scrubbed = k.scrub(capture(query="key=AnyValueAtAll&alt=sse"))

        assert scrubbed.query == f"key={k.REDACTION.format(name='credential_query')}&alt=sse"

    def test_an_operator_literal_is_replaced(self) -> None:
        """A hostname or an organisation name has no shape; only the operator knows it."""
        scrubbed = k.scrub(capture(b"host is grace-workstation"), extra=("grace-workstation",))

        assert b"grace-workstation" not in scrubbed.body

    @pytest.mark.parametrize(
        "path",
        [
            "/home/someuser/notes.md",
            "/Users/someuser/notes.md",
            "C:\\Users\\someuser\\notes.md",
            '{"f": "C:\\\\Users\\\\someuser\\\\notes.md"}',
        ],
    )
    def test_a_home_path_is_scrubbed_on_every_platform(self, path: str) -> None:
        """Found by falsifying the README against the code.

        The table promised `C:\\Users\\<user>\\` and a lookbehind-based pattern
        silently did not deliver it, so a Windows capture leaked its username
        through a scrubber that looked complete. The last case is the form a
        body actually carries: JSON escapes each backslash.
        """
        assert "someuser" not in k.scrub(capture(path.encode())).body.decode()

    def test_a_json_escaped_assignment_is_found(self) -> None:
        """Measured, not assumed.

        A corpus body is JSON, so a settings file the agent read arrives with its
        quotes escaped. Without allowing for that, this rule finds shell
        assignments and misses every quoted one — which is the shape a leaked
        credentials file actually has.
        """
        body = json.dumps({"content": '{"api_key": "zyxw9876vuts5432rqpo"}'}).encode()

        assert b"zyxw9876vuts5432rqpo" not in k.scrub(capture(body)).body


class TestTheScrubberLeavesEverythingElseAlone:
    """§3.3.3: a scrubber that mangles legitimate content breaks I1."""

    def test_a_clean_claude_code_body_is_untouched(self) -> None:
        """The false-positive control.

        Signatures, `toolu_` ids, base64 images and prose about api keys are all
        long opaque runs that a high-entropy rule would flag. None is a secret.
        """
        assert k.scrub(capture(CLEAN_BODY)).body == CLEAN_BODY

    @pytest.mark.parametrize(
        "text",
        [
            "disk-usage-monitoring-service.py",
            "task-oriented-planning-module",
            "risk-assessment-controller-v2",
            "logo@2x.png",
            "vendor/sdk-client-generated-types.ts",
            # Reaches `anthropic_key`'s boundary specifically: `fla|sk-ant-design…`.
            # Without it a mutation removing that one `\\b` survives the suite.
            "flask-ant-design-components-v2",
        ],
    )
    def test_an_ordinary_filename_is_not_redacted(self, text: str) -> None:
        """The control that was missing, and the bug it was missing.

        `CLEAN_BODY` contains no `sk-` or `@` substring at all, so it could not
        see the `sk-` rules' false-positive surface: unanchored, they turned
        `disk-usage-monitoring-service.py` into `di<redacted:openai_key>.py`.
        File paths are the commonest payload in a Claude Code body, so this was
        mangling legitimate content at scale.
        """
        body = json.dumps({"file": text}).encode()

        assert k.scrub(capture(body)).body == body

    @pytest.mark.parametrize("address", ["admin@empresa.py", "contact@firma.md", "dev@example.js"])
    def test_a_real_address_on_a_file_like_cctld_is_still_scrubbed(self, address: str) -> None:
        """`.py` is Paraguay and `.md` is Moldova.

        The first attempt at keeping `logo@2x.png` out excluded a list of file
        extensions and silently stopped scrubbing these — trading a cosmetic
        false positive for a real miss, which is a bad trade for a scrubber.
        The rule targets the retina *shape* instead.
        """
        body = json.dumps({"contact": address}).encode()

        assert address.encode() not in k.scrub(capture(body)).body

    def test_the_vendor_string_regression_text_survives_byte_identically(self) -> None:
        """§3.3.3 names this exact string; T-C5 commits it as an entry."""
        text = b'{"content": "Please explain how kitty-bridge works"}'

        assert k.scrub(capture(text)).body == text

    def test_max_tokens_is_not_read_as_a_token_assignment(self) -> None:
        """`max_tokens` ends in `token` + `s`, and its value can be long.

        The rule requires the separator immediately after the name, which is
        what keeps this out — worth pinning, because widening the name list is
        the obvious "improvement" that would break it.
        """
        body = b'{"max_tokens": 32000000000000000000000000}'

        assert k.scrub(capture(body)).body == body

    @pytest.mark.parametrize(
        "path",
        [
            "/v1/messages",
            "/openai/deployments/gpt-4o/chat/completions",
            "/v1/projects/my-project-1234/locations/us-central1/publishers/google/models/gemini-2.0:generateContent",
            "/v1beta/models/gemini-2.0-flash:streamGenerateContent",
            "/model/anthropic.claude-3-5-sonnet-20241022-v2:0/converse",
        ],
    )
    def test_a_real_route_survives_the_path_scan(self, path: str) -> None:
        """Scanning the path must not cost the evidence the path *is*.

        §3.3.5 asserts on routing because on Azure the deployment id is the only
        thing separating two byte-identical requests, and on Vertex the project
        and location are "the account being billed". A scrubber that rewrote any
        of these would destroy the very difference the oracle exists to see.
        """
        assert k.scrub(capture(path=path)).path == path

    def test_an_empty_query_stays_empty(self) -> None:
        """Splitting an empty string on `&` yields one empty part, not none."""
        assert k.scrub(capture(query="")).query == ""


class TestScrubbingIsConsistentWithDetection:
    """The property that makes the lint trustworthy."""

    @pytest.mark.parametrize(
        "body", [CLEAN_BODY, DIRTY_BODY, b"{}", b"sk-ant-" + b"q" * 40, *ADJACENT_PAIRS[:8]]
    )
    def test_scrubbed_output_has_no_findings(self, body: bytes) -> None:
        """`findings(scrub(x)) == ()`.

        True by construction — both read one pass over one table — and asserted
        anyway, because "by construction" is a claim about code that changes.
        """
        assert k.findings(k.scrub(capture(body))) == ()

    @pytest.mark.parametrize("body", [CLEAN_BODY, DIRTY_BODY, b"{}", *ADJACENT_PAIRS[:8]])
    def test_scrubbing_is_idempotent(self, body: bytes) -> None:
        """`scrub(scrub(x)) == scrub(x)`, so re-running the procedure is safe."""
        once = k.scrub(capture(body))

        assert k.scrub(once) == once

    def test_no_pattern_matches_a_placeholder(self) -> None:
        """The structural guard behind both properties above.

        Found by measurement: `home_path` matched its own
        `/home/<redacted:home_path>/` output, so `scrub` looked idempotent while
        `findings` went on reporting a secret that was no longer there. This
        fails the moment a new pattern does the same.
        """
        placeholders = " ".join(k.REDACTION.format(name=name) for name in k.REDACTION_CLASSES)
        # Every prefix `home_path` recognises, since that is the pattern whose
        # own output it matched.
        surrounded = " ".join(
            f"{prefix}{placeholders}" for prefix in ("/home/", "/Users/", "\\Users\\", "")
        ).encode()

        assert k.findings(capture(surrounded)) == ()

    def test_no_secret_survives_any_adjacency(self) -> None:
        """The sweep four fixed bodies could never have run.

        Every credential pattern is anchored on a word boundary, and two secrets
        can sit flush together with no separator: an `AIza…` key ending in a
        letter, immediately followed by `sk-proj-…`. On the first pass the
        second has no boundary in front of it; redacting the first *creates*
        that boundary, and a single-pass scan has already moved on. That
        shipped, briefly, as a live OpenAI key left in a scrubbed body — an
        invariant three documents call load-bearing, false in practice.

        Every ordered pair at three separations, which is the shape of the bug.
        """
        unclean = [
            body
            for body in ADJACENT_PAIRS
            if k.findings(k.scrub(capture(body))) or k.scrub(k.scrub(capture(body))) != k.scrub(capture(body))
        ]

        assert unclean == []

    def test_both_secrets_of_a_flush_pair_are_reported(self) -> None:
        """`_find` must iterate, not merely `_rewrite`.

        The lint's message is how a maintainer learns what to fix. A detector
        that stops after one pass reports the first secret of a flush pair and
        stays silent about the second, so the entry looks one edit away from
        clean when it is two.
        """
        body = ("AIza" + "b" * 35 + "sk-proj-" + "y" * 74 + "T3BlbkFJ" + "z" * 74).encode()

        assert sorted(f.name for f in k.findings(capture(body))) == ["gcp_key", "openai_key"]

    def test_the_detector_claims_only_what_the_rewriter_replaces(self) -> None:
        """The rule that keeps the two halves from diverging.

        `assigned_secret` rewrites its value and leaves `SECRET="` standing. If
        the detector claimed the whole match instead, it would manufacture a
        word boundary the rewriter never creates — and report an AWS key that
        the scrubbed file still contains, flush against it.
        """
        body = b'AKIA' + b'234567ABCDEFGH34' + b'SECRET="9f8a7b6c5d4e3f2a1b0c9d8e"'

        reported = [f.name for f in k.findings(capture(body))]
        replaced = k.scrub(capture(body)).body.decode().count("<redacted:")

        assert reported == ["assigned_secret"]
        assert replaced == 1

    def test_a_body_already_containing_the_claim_character_still_scans(self) -> None:
        """The mask character may collide with the body, and it does not matter.

        `_find`'s `claimed` bytearray is the source of truth for what has been
        claimed; the character only breaks word boundaries in the text the
        patterns see, and a pre-existing `\x01` already breaks one. Pinned
        because the obvious defensive reading is the opposite — an earlier
        version searched for a free control character and *raised* when none
        was, which would have refused to scan a perfectly scannable body.
        Reachable input: T-C6 commits a malformed body, which is not
        constrained to being JSON.
        """
        body = b"\x01\x02 padding " + PLANTED_KEY.encode() + b" \x01"

        assert [f.name for f in k.findings(capture(body))] == ["anthropic_key"]

    def test_the_two_halves_agree_on_a_realistically_separated_body(self) -> None:
        """`_find` and `_rewrite` share a table but run different algorithms.

        Agreement is therefore a property to keep, not one to assume. Under-
        reporting is the direction that matters — a lint whose message names
        fewer secrets than it redacted lets a reviewer fix the one finding and
        believe they are done.
        """
        body = json.dumps(
            {
                "m": " ".join(
                    [PLANTED_KEY, "AIza" + "b" * 35, "AKIA" + "234567ABCDEFGH34",
                     "someone@example.com", "/home/someuser/x"]
                )
            }
        ).encode()

        reported = k.findings(capture(body))
        replaced = k.scrub(capture(body)).body.decode().count("<redacted:")

        assert len(reported) == replaced == 5

    @pytest.mark.parametrize(
        "body",
        [
            b"<redacted:anthropic_key>" * 200,
            b"/home/" + b"<redacted:home_path>/" * 200,
            b"api_key=<redacted:openai_key>sk-proj-" + b"y" * 40,
            b"\x01" * 5000,
            b"sk-ant-" * 3000,
        ],
    )
    def test_the_scan_terminates_on_output_shaped_input(self, body: bytes) -> None:
        """Both halves loop until nothing changes, so termination is a property.

        It holds because a placeholder matches no pattern — which is what
        `test_no_pattern_matches_a_placeholder` asserts — but a body built out
        of placeholders is the input that would expose a rule that broke it, and
        no realistic fixture contains one. A failure here hangs the gate rather
        than failing it, which is the worst way to find out.
        """
        scrubbed = k.scrub(capture(body))

        assert k.findings(scrubbed) == ()
        assert k.scrub(scrubbed) == scrubbed

    def test_an_offset_points_at_the_secrets_real_byte(self) -> None:
        """The offset is the only actionable datum a finding carries.

        R12 forbids reporting the matched value, so a wrong offset is worse than
        none. An earlier version measured positions against text that previous
        patterns had already shortened — six bytes out on a two-secret body, and
        further on every additional one. The non-ASCII case is the second half:
        a character index equals a byte offset only while the body is ASCII, and
        a Claude Code transcript rarely is.
        """
        # `ensure_ascii=False` is the whole point: the default escapes `é` to
        # `\\u00e9`, leaving a pure-ASCII body in which a character index and a
        # byte offset are the same number — so the test could not fail.
        body = json.dumps(
            {"m": "é" * 50 + " /home/someuser/x someone@example.com"}, ensure_ascii=False
        ).encode()

        located = {f.name: f.offset for f in k.findings(capture(body))}

        assert located["home_path"] == body.index(b"someuser")
        assert located["email"] == body.index(b"someone@")

    def test_a_dirty_body_reports_one_finding_per_secret(self) -> None:
        """Five planted secrets, five findings — not four, and not fifteen.

        Over-reporting is not harmless: a table where several rules claim one
        secret hides the rule that has stopped working.
        """
        names = [f.name for f in k.findings(capture(DIRTY_BODY))]

        assert sorted(names) == ["anthropic_key", "aws_key_id", "email", "gcp_key", "home_path"]

    def test_the_header_and_query_vocabulary_is_not_respelled(self) -> None:
        """One owner for the credential names.

        `contract` already defines them for its `repr` mask; a second spelling
        here is a second thing to forget when a provider is added. Asserted as
        **identity**, not equality: a copied literal set would compare equal on
        the day it was copied and diverge silently afterwards, which is the
        failure this is for.
        """
        assert k.REDACTED_HEADERS is REDACTED_HEADERS
        assert k.REDACTED_QUERY_KEYS is REDACTED_QUERY_KEYS


class TestContentLengthFollowsTheBody:
    """A capture whose declared length disagrees with its body is not a capture."""

    def test_the_value_is_recomputed(self) -> None:
        """Scrubbing changes the body's length; the header must follow it.

        The first two assertions are the ones that bite. Comparing the header
        against `len(scrubbed.body)` alone restates the implementation and
        passes whether or not scrubbing changed anything — so the test would
        survive a body that never shrank, which is the case it exists for.
        """
        scrubbed = k.scrub(capture(DIRTY_BODY, headers=[("Content-Length", str(len(DIRTY_BODY)))]))

        assert len(scrubbed.body) < len(DIRTY_BODY)
        assert dict(scrubbed.headers)["Content-Length"] != str(len(DIRTY_BODY))
        assert dict(scrubbed.headers)["Content-Length"] == str(len(scrubbed.body))

    def test_the_casing_and_position_survive(self) -> None:
        """§4.3 C1 asserts on the exact header set, casing included."""
        headers = [("Host", "h"), ("content-length", "9"), ("anthropic-beta", "b")]
        scrubbed = k.scrub(capture(b"{}", headers=headers))

        assert [name for name, _ in scrubbed.headers] == ["Host", "content-length", "anthropic-beta"]

    def test_duplicate_headers_survive(self) -> None:
        """`anthropic-beta` is sent more than once, and the set is the evidence."""
        headers = [("anthropic-beta", "one"), ("anthropic-beta", "two")]
        scrubbed = k.scrub(capture(b"{}", headers=headers))

        assert scrubbed.headers == (("anthropic-beta", "one"), ("anthropic-beta", "two"))


class TestTheRoundTrip:
    """What `write_entry` wrote, `load_corpus` must read back unchanged."""

    def test_an_entry_survives_write_and_load(self, tmp_path: Path) -> None:
        """The capture procedure's last step and the suite's first must agree."""
        written = entry(triggers_met=frozenset({Trigger.ORPHAN_TOOL_RESULT}))
        k.write_entry(tmp_path, written)

        (loaded,) = k.load_corpus(tmp_path)

        assert loaded == written

    def test_the_body_is_byte_exact(self, tmp_path: Path) -> None:
        """§3.3.2 asserts byte-level key order on the native passthrough path.

        A body re-serialised on the way through would lose it, which is why the
        body is a sidecar file and not a value inside the manifest.
        """
        body = b'{"b": 1, "a": 2,   "c": [ ]}'
        k.write_entry(tmp_path, entry(request=capture(body)))

        assert k.load_corpus(tmp_path)[0].request.body == body

    def test_the_manifest_holds_no_body(self, tmp_path: Path) -> None:
        """The sidecar is the format; a body inside the manifest is a second copy."""
        path = k.write_entry(tmp_path, entry(request=capture(CLEAN_BODY)))

        assert "claude-opus" not in path.read_text(encoding="utf-8")

    def test_writing_scrubs(self, tmp_path: Path) -> None:
        """The one step a tired operator skips must not be the one that matters."""
        k.write_entry(tmp_path, entry(request=capture(DIRTY_BODY)))

        assert b"AKIA" + b"234567ABCDEFGH34" not in (tmp_path / "sample.body").read_bytes()

    @pytest.mark.parametrize("bad", ["../escaped", "a/b", "a\\b", ".hidden", "-flag", "", "wrong.dot"])
    def test_writing_refuses_an_id_that_is_not_a_bare_name(self, tmp_path: Path, bad: str) -> None:
        """An id is interpolated into two paths, so it must be a file name and nothing else.

        The load side already refused these — a manifest's id must equal its
        filename stem, and a filename cannot contain a separator — and that is
        what made the gap easy to miss: the asymmetry looked like a check that
        existed. It did not. `write_entry` with `id="../escaped"` wrote both
        files into the corpus directory's **parent**.

        The leading dot and dash are refused for their own reasons: a dotfile is
        invisible to the corpus glob and therefore to the lint, and a leading
        dash is an option to every command a maintainer runs over these files.
        """
        with pytest.raises(k.CorpusEntryError, match="legal entry id"):
            k.write_entry(tmp_path, entry(id=bad))

    def test_writing_refuses_a_short_operator_literal(self, tmp_path: Path) -> None:
        """The floor must hold at the procedure's last step, not only inside the scan."""
        with pytest.raises(ValueError, match="extra literals"):
            k.write_entry(tmp_path, entry(), extra=("tas",))

    def test_writing_refuses_a_short_cleared_literal(self, tmp_path: Path) -> None:
        """`write_entry` reads `known_non_secrets` straight into `allow`.

        A short one there disables the lint on the entry being written, which is
        the worst moment for it to happen.
        """
        short = entry(known_non_secrets=(("key", "looks fine, disables the lint"),))

        with pytest.raises(ValueError, match="allow literals"):
            k.write_entry(tmp_path, short)

    def test_nothing_is_written_when_the_id_is_refused(self, tmp_path: Path) -> None:
        """A refusal that half-wrote would leave an orphan body the lint never opens."""
        with pytest.raises(k.CorpusEntryError):
            k.write_entry(tmp_path, entry(id="../escaped"))

        assert list(tmp_path.parent.glob("escaped.*")) == []
        assert list(tmp_path.glob("*")) == []

    @pytest.mark.parametrize(
        ("overrides", "expected"),
        [
            ({"origin": k.CAPTURED, "origin_note": "", "captured_from": "", "captured_at": "d"}, "captured_from"),
            ({"origin": k.CAPTURED, "origin_note": "", "captured_from": "v", "captured_at": ""}, "captured_at"),
            ({"origin_note": ""}, "origin_note"),
            ({"origin": "recorded"}, "origin must be"),
            (
                {
                    "triggers_met": frozenset({Trigger.ORPHAN_TOOL_RESULT}),
                    "triggers_absent": frozenset({Trigger.ORPHAN_TOOL_RESULT}),
                },
                "both met and absent",
            ),
            ({"triggers_met": frozenset({Trigger.UPSTREAM_EMPTY_RESPONSE})}, "cannot be arranged"),
            ({"known_non_secrets": (("a-literal", ""),)}, "needs a reason"),
        ],
    )
    def test_the_writer_refuses_what_the_reader_refuses(
        self, tmp_path: Path, overrides: dict[str, object], expected: str
    ) -> None:
        """Sharing `_checked_id` closed this asymmetry for the id; these are the rest.

        Without them `write_entry` succeeds on an entry `load_corpus` then
        rejects: the capture procedure's last step reports success and the
        failure surfaces in CI, against a file already committed.
        """
        with pytest.raises(k.CorpusEntryError, match=expected):
            k.write_entry(tmp_path, entry(**overrides))

    def test_anything_the_writer_accepts_the_reader_accepts(self, tmp_path: Path) -> None:
        """The invariant behind the case list above, asserted directly.

        A rule added to one side and not the other fails here rather than
        drifting until someone writes an entry nothing can read.
        """
        accepted = [
            entry(id="synthetic-one"),
            entry(
                id="captured-one",
                origin=k.CAPTURED,
                origin_note="",
                captured_from="claude-code/1.2.3",
                captured_at="2026-09-12",
            ),
            entry(id="with-triggers", triggers_met=frozenset({Trigger.TOOL_RESULT_OVER_LIMIT})),
        ]
        for one in accepted:
            k.write_entry(tmp_path, one)

        assert [e.id for e in k.load_corpus(tmp_path)] == ["captured-one", "synthetic-one", "with-triggers"]

    def test_a_short_literal_refusal_names_the_entry(self, tmp_path: Path) -> None:
        """Every rejection in `load_entry` names the entry; this one did not.

        A corpus failure that does not say which entry is a corpus-wide search.
        """
        with pytest.raises(ValueError, match="sample: extra literals"):
            k.write_entry(tmp_path, entry(), extra=("tas",))

    def test_entries_come_back_in_id_order(self, tmp_path: Path) -> None:
        """A stable order keeps a failure message the same across runs."""
        k.write_entry(tmp_path, entry(id="zebra"))
        k.write_entry(tmp_path, entry(id="alpha"))

        assert [e.id for e in k.load_corpus(tmp_path)] == ["alpha", "zebra"]


class TestTheLint:
    """`assert_corpus_clean` — and the ways it could pass while blind."""

    def test_a_clean_corpus_passes(self, tmp_path: Path) -> None:
        """The positive control for every rejection below."""
        k.write_entry(tmp_path, entry())

        k.assert_corpus_clean(k.load_corpus(tmp_path))

    def test_an_empty_corpus_fails(self, tmp_path: Path) -> None:
        """"No secrets found" is satisfied perfectly by having looked at nothing.

        `layers.assert_no_layer_violations` sets the precedent; a lint that
        passes over an empty directory is indistinguishable from a healthy one.
        """
        with pytest.raises(k.UnscrubbedCorpusError, match="handed no entries"):
            k.assert_corpus_clean(k.load_corpus(tmp_path))

    def test_a_planted_key_fails_and_is_named(self, tmp_path: Path) -> None:
        """Plan §1.4's named falsification case for this harness.

        The entry is internally consistent — the author simply never scrubbed —
        so the digest check passes and the lint is the only thing standing
        between the key and the repository.
        """
        manifest_for(tmp_path, body=PLANTED_BODY)

        with pytest.raises(k.UnscrubbedCorpusError, match="anthropic_key"):
            k.assert_corpus_clean(k.load_corpus(tmp_path))

    @pytest.mark.parametrize("field_name", ["description", "origin_note"])
    def test_a_secret_in_the_manifests_prose_fails_the_lint(self, tmp_path: Path, field_name: str) -> None:
        """Operator prose is the one part of an entry no pattern had seen.

        It is where a maintainer is most likely to write the thing the policy
        exists to keep out — "captured on <internal host>", "the customer's key
        was in this one". Reported rather than rewritten: silently mangling a
        description makes the entry harder to review, not safer.
        """
        manifest_for(tmp_path, **{field_name: f"note {PLANTED_KEY}"})

        with pytest.raises(k.UnscrubbedCorpusError, match=f"anthropic_key in {field_name}"):
            k.assert_corpus_clean(k.load_corpus(tmp_path))

    def test_the_failure_message_does_not_print_the_secret(self, tmp_path: Path) -> None:
        """The message reaches CI logs, which are as public as the repository.

        A finding that quoted the run it matched would turn a contained
        authoring mistake into a published one, and the remedy for a published
        credential is rotation, not a better diff.
        """
        manifest_for(tmp_path, body=PLANTED_BODY)

        with pytest.raises(k.UnscrubbedCorpusError) as raised:
            k.assert_corpus_clean(k.load_corpus(tmp_path))

        assert PLANTED_KEY not in str(raised.value)
        assert "offset" in str(raised.value)


class TestTheScrubberIsFalsifiable:
    """Plan §1.4 — deliberate defects the harness must detect, in the suite."""

    def test_a_header_only_scrubber_is_caught(self) -> None:
        """The defect: a scrubber that cleans headers and ignores the body.

        It looks right — the credential header is the obvious secret — and it
        leaves every key the agent ever read in a tool result untouched.  Built
        by hand rather than by blinding the table, because the table is shared:
        blinding it would blind the detector too, and the test would prove
        nothing about the detector's independence from the scrubber.
        """
        half_done = capture(DIRTY_BODY, headers=[("X-Api-Key", k.REDACTION.format(name="credential_header"))])

        assert "anthropic_key" in [f.name for f in k.findings(half_done)]
        with pytest.raises(k.UnscrubbedCorpusError):
            k.assert_corpus_clean([entry(request=half_done)])

    def test_an_empty_pattern_table_reports_every_corpus_clean(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """The defect that makes the lint useless while staying green.

        A detector matching nothing reports every fixture clean, so this asserts
        the table is what does the work — not the loop around it.
        """
        assert k.findings(capture(DIRTY_BODY))

        monkeypatch.setattr(k, "PATTERNS", ())

        assert k.findings(capture(DIRTY_BODY)) == ()

    def test_a_loosened_pattern_is_caught_by_the_false_positive_control(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The defect in the other direction: a rule that flags legitimate content.

        A high-entropy rule is the tempting addition, and this is what it costs —
        the thinking signature and the base64 image in a real body both match.
        """
        monkeypatch.setattr(k, "PATTERNS", (("entropy", re.compile(r"[A-Za-z0-9+/=]{40,}")),))

        assert k.scrub(capture(CLEAN_BODY)).body != CLEAN_BODY


class TestTheBodyCannotBeModifiedUnnoticed:
    """`.gitattributes` prevents line-ending translation; this catches it."""

    def test_a_modified_body_is_rejected(self, tmp_path: Path) -> None:
        """The realistic corruption is `core.autocrlf`, not malice.

        CI runs on `ubuntu-latest` only, so a CRLF-rewritten body would be
        consumed by the gate rather than noticed by it — and it lands on the
        byte-level key-order assertion and on every size-derived trigger.
        """
        path = manifest_for(tmp_path)
        (tmp_path / "sample.body").write_bytes(CLEAN_BODY.replace(b"\n", b"\r\n") + b" ")

        with pytest.raises(k.CorpusEntryError, match="body_sha256"):
            k.load_entry(path)

    def test_the_digest_written_is_the_digest_of_what_was_written(self, tmp_path: Path) -> None:
        """The digest covers the **scrubbed** body, which is what lands in the file."""
        k.write_entry(tmp_path, entry(request=capture(DIRTY_BODY)))
        manifest = json.loads((tmp_path / "sample.json").read_text(encoding="utf-8"))

        assert manifest["body_sha256"] == hashlib.sha256((tmp_path / "sample.body").read_bytes()).hexdigest()


class TestEncodedCapturesAreRefused:
    """A compressed body is one the scrubber reads as noise and reports clean."""

    @pytest.mark.parametrize("header", ["Content-Encoding", "transfer-encoding"])
    def test_an_encoded_capture_cannot_be_written(self, tmp_path: Path, header: str) -> None:
        """The false clean no plaintext falsification case can detect.

        `transfer-encoding` is refused for a second reason: `request.read()`
        de-chunks, so the stored body would not correspond to its own headers.
        """
        encoded = entry(request=capture(DIRTY_BODY, headers=[(header, "gzip")]))

        with pytest.raises(k.EncodedCaptureError, match="entity bodies"):
            k.write_entry(tmp_path, encoded)


class TestTheOperatorLiteralsAreBounded:
    """Both literal channels are unbounded substring operations, in opposite directions."""

    def test_a_short_redaction_literal_is_refused(self) -> None:
        """A three-character username shreds a body everywhere it occurs.

        A fixed false-positive control cannot see this, because the literal is
        operator-supplied — so the guard has to be at the seam.
        """
        with pytest.raises(ValueError, match="extra literals must be at least 4"):
            k.scrub(capture(b"{}"), extra=("tas",))

    def test_a_short_cleared_literal_is_refused(self) -> None:
        """The dangerous direction, and the one that is easy to miss.

        Clearing `"key"` parks that substring inside `api_key`, so the
        `assigned_secret` rule stops matching the name and **the lint goes
        silently blind on the entry a reviewer was vouching for**. The
        stale-exemption check cannot catch it either: `"key"` is still in the
        body, so the exemption is not stale.
        """
        with pytest.raises(ValueError, match="allow literals must be at least 4"):
            k.findings(capture(b'{"a":"api_key = SECRETVALUE1234567890abcd"}'), ("key",))

    def test_a_long_enough_literal_is_accepted(self) -> None:
        """The positive control: the floor must not refuse a usable literal."""
        assert b"grace-host" not in k.scrub(capture(b"host grace-host"), extra=("grace-host",)).body


class TestKnownNonSecrets:
    """The captures are taken while working on this repository."""

    def test_a_declared_non_secret_survives_scrubbing(self) -> None:
        """`tests/test_integration.py` really does contain `api_key = "sk-test-…"`.

        A tool result quoting this tree therefore trips an honest pattern table,
        and without an escape hatch the entry could not be committed at all.
        """
        literal = "sk-test-integration-key-12345"
        body = f'{{"content": "api_key = \\"{literal}\\""}}'.encode()

        scrubbed = k.scrub(capture(body), allow=(literal,))

        assert literal.encode() in scrubbed.body
        assert k.findings(scrubbed, (literal,)) == ()

    def test_an_undeclared_one_is_still_caught(self) -> None:
        """The hatch must be opt-in per entry, or it is not a hatch."""
        literal = "sk-test-integration-key-12345"
        body = f'{{"content": "api_key = {literal}"}}'.encode()

        assert k.findings(capture(body))

    def test_a_body_carrying_a_nul_byte_is_not_corrupted(self) -> None:
        """Found by falsifying the parking sentinel against a malformed body.

        Cleared literals are parked under a sentinel while the table runs. With
        a fixed `\\x00`-delimited one, a body containing a literal NUL had the
        literal **injected** at a position it never occupied — unparking rewrote
        content that merely looked like a sentinel. T-C6 commits a malformed
        body, which is not constrained to being JSON, so this is reachable.
        """
        literal = "sk-test-integration-key-12345"
        raw = b'{"a":"' + literal.encode() + b'"} \x000\x00 \x00corpus0\x00 tail'

        assert k.scrub(capture(raw), allow=(literal,)).body == raw

    def test_a_stale_exemption_fails_the_lint(self, tmp_path: Path) -> None:
        """§6.2.3's rule for the `web.Request` exclusion, applied here.

        An exemption that no longer matches anything has stopped being an
        exemption and become a blanket one, and nothing else would say so.
        """
        manifest_for(tmp_path, known_non_secrets=[["a-literal-not-present", "was removed upstream"]])

        with pytest.raises(k.UnscrubbedCorpusError, match="no longer contains"):
            k.assert_corpus_clean(k.load_corpus(tmp_path))

    def test_an_exemption_without_a_reason_is_rejected(self, tmp_path: Path) -> None:
        """A reason is what a reviewer judges; a bare literal is a request to trust."""
        path = manifest_for(tmp_path, known_non_secrets=[["something-long", ""]])

        with pytest.raises(k.CorpusEntryError, match="needs a reason"):
            k.load_entry(path)

    def test_a_short_exemption_is_rejected_at_load_and_names_the_entry(self, tmp_path: Path) -> None:
        """The floor again, at the boundary where the entry is authored.

        `_scan` raises a bare `ValueError` with no idea which entry it came
        from; a corpus failure that does not name the entry is a corpus-wide
        search.
        """
        path = manifest_for(tmp_path, known_non_secrets=[["key", "looks fine, disables the lint"]])

        with pytest.raises(k.CorpusEntryError, match="sample: known_non_secrets"):
            k.load_entry(path)

    @pytest.mark.parametrize("value", [{"a": "b"}, [["only-one"]], [["lit", 7]], "not-a-list"])
    def test_a_malformed_exemption_list_is_rejected(self, tmp_path: Path, value: object) -> None:
        """Every arity and type the shape check covers, so none of them can rot."""
        path = manifest_for(tmp_path, known_non_secrets=value)

        with pytest.raises(k.CorpusEntryError, match="list of \\[literal, reason\\] pairs"):
            k.load_entry(path)


class TestTriggersAnEntryCannotArrange:
    """G21's over-declaration hazard, closed for the cases the repository proves."""

    @pytest.mark.parametrize(
        "name",
        [
            "upstream_empty_response",
            "thinking_roundtrip_rejected",
            "native_tool_use_format_error",
            "upstream_rejected_oversized_on_balancing",
        ],
    )
    def test_a_response_trigger_cannot_be_declared(self, tmp_path: Path, name: str) -> None:
        """These four are decided by the upstream, not by the request.

        `register.py`'s own docstring names each one; an entry claiming it would
        be claiming something it is not the thing that decides, which is exactly
        the over-declaration that makes the oracle pass over a broken bridge.
        """
        path = manifest_for(tmp_path, triggers_met=[name])

        with pytest.raises(k.CorpusEntryError, match="cannot arrange"):
            k.load_entry(path)

    def test_always_cannot_be_declared(self, tmp_path: Path) -> None:
        """The register calls it "the absence of a condition, not a condition".

        Declaring it met is noise; declaring it absent is false.
        """
        path = manifest_for(tmp_path, triggers_absent=["always"])

        with pytest.raises(k.CorpusEntryError, match="cannot arrange"):
            k.load_entry(path)

    def test_a_request_trigger_is_still_accepted(self, tmp_path: Path) -> None:
        """The positive control: the guard must not reject what the corpus is for."""
        path = manifest_for(tmp_path, triggers_met=["tool_result_over_limit"])

        assert Trigger.TOOL_RESULT_OVER_LIMIT in k.load_entry(path).triggers_met


class TestTheInboundFormatIsDeclared:
    """§7.4's oracle takes an `inbound_format`; the corpus supplies the inbound half."""

    def test_a_format_round_trips(self, tmp_path: Path) -> None:
        """Inbound is not always Messages — P14–P16 exist because it is not."""
        k.write_entry(tmp_path, entry(wire_format=WireFormat.OPENAI_RESPONSES))

        assert k.load_corpus(tmp_path)[0].wire_format is WireFormat.OPENAI_RESPONSES

    def test_null_is_allowed_for_a_body_no_reader_can_classify(self, tmp_path: Path) -> None:
        """§7.4 requires T-C6's malformed entry to be distinguishable from an I1 breach."""
        path = manifest_for(tmp_path, wire_format=None)

        assert k.load_entry(path).wire_format is None

    def test_an_unknown_format_is_rejected(self, tmp_path: Path) -> None:
        """`WireFormat` is closed so six reader authors cannot spell one format three ways."""
        path = manifest_for(tmp_path, wire_format="anthropic")

        with pytest.raises(k.CorpusEntryError, match="not a WireFormat"):
            k.load_entry(path)


class TestEvidenceIsSeparableFromConstruction:
    """§7.1: a captured body is evidence; a hand-written one is our belief."""

    def test_synthetic_entries_are_excluded(self) -> None:
        """Plan §6 synthesises two Epic C entries and this task ships a third.

        Every query meaning "what has the agent been observed to send" has to
        exclude them, and a helper does that by construction where a convention
        does it until somebody forgets.
        """
        real = entry(
            id="real",
            origin=k.CAPTURED,
            origin_note="",
            captured_from="claude-code/1.2.3",
            captured_at="2026-09-12",
        )

        assert k.captured_only([entry(), real]) == (real,)
