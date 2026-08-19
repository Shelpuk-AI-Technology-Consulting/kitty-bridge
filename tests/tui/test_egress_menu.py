"""Tests for the interactive egress gateway menu.

Real stores, mocked prompts — the convention used by test_profile_menu.py.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from kitty.credentials.file_backend import FileBackend
from kitty.credentials.store import CredentialStore
from kitty.egress_store import EgressRecord, EgressStore

_MOD = "kitty.cli.egress_cmd"


@pytest.fixture()
def store(tmp_path) -> EgressStore:
    return EgressStore(path=tmp_path / "egress.json")


@pytest.fixture()
def cred_store(tmp_path) -> CredentialStore:
    return CredentialStore(backends=[FileBackend(path=tmp_path / "credentials.json")])


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch) -> None:
    from kitty.egress import ENV_PROXY

    monkeypatch.delenv(ENV_PROXY, raising=False)


def _import_menu():
    from kitty.cli.egress_cmd import run_egress_menu

    return run_egress_menu


class TestConfigureFlow:
    """R12: the guided setup persists a usable gateway."""

    def test_configures_an_authenticated_gateway(self, store: EgressStore, cred_store: CredentialStore):
        menu = _import_menu()

        with (
            patch("sys.stdin.isatty", return_value=True),
            patch(f"{_MOD}.SelectionMenu.show", side_effect=["Configure gateway", "Back"]),
            patch(f"{_MOD}.prompt_text", side_effect=["proxy.iproyal.com:12323", "myuser"]),
            patch(f"{_MOD}.prompt_secret", return_value="s3cr3t"),
            # needs-auth = yes, test-now = no
            patch(f"{_MOD}.prompt_confirm", side_effect=[True, False]),
        ):
            menu(cred_store, store)

        record = store.load()
        assert record is not None
        assert record.proxy_url == "http://proxy.iproyal.com:12323"
        assert record.username == "myuser"
        assert record.auth_ref is not None
        assert cred_store.get(record.auth_ref) == "s3cr3t"

    def test_bare_host_port_gets_a_scheme(self, store: EgressStore, cred_store: CredentialStore):
        """Proxy vendors quote host:port, so accept that spelling."""
        menu = _import_menu()

        with (
            patch("sys.stdin.isatty", return_value=True),
            patch(f"{_MOD}.SelectionMenu.show", side_effect=["Configure gateway", "Back"]),
            patch(f"{_MOD}.prompt_text", side_effect=["10.20.0.5:3128"]),
            patch(f"{_MOD}.prompt_confirm", side_effect=[False, False]),
        ):
            menu(cred_store, store)

        record = store.load()
        assert record is not None
        assert record.proxy_url == "http://10.20.0.5:3128"

    def test_unauthenticated_gateway_stores_no_credential(self, store: EgressStore, cred_store: CredentialStore):
        menu = _import_menu()

        with (
            patch("sys.stdin.isatty", return_value=True),
            patch(f"{_MOD}.SelectionMenu.show", side_effect=["Configure gateway", "Back"]),
            patch(f"{_MOD}.prompt_text", side_effect=["http://proxy.example.com:3128"]),
            patch(f"{_MOD}.prompt_secret") as mock_secret,
            patch(f"{_MOD}.prompt_confirm", side_effect=[False, False]),
        ):
            menu(cred_store, store)

        record = store.load()
        assert record is not None
        assert record.username is None
        assert record.auth_ref is None
        mock_secret.assert_not_called()

    def test_invalid_address_reprompts_instead_of_crashing(self, store: EgressStore, cred_store: CredentialStore):
        menu = _import_menu()

        with (
            patch("sys.stdin.isatty", return_value=True),
            patch(f"{_MOD}.SelectionMenu.show", side_effect=["Configure gateway", "Back"]),
            # first answer is a scheme kitty cannot use, second is valid
            patch(f"{_MOD}.prompt_text", side_effect=["socks5://proxy.example.com:1080", "proxy.example.com:3128"]),
            patch(f"{_MOD}.prompt_confirm", side_effect=[False, False]),
            patch(f"{_MOD}.print_error") as mock_error,
        ):
            menu(cred_store, store)

        assert mock_error.called, "the user should have been told why the address was rejected"
        record = store.load()
        assert record is not None
        assert record.proxy_url == "http://proxy.example.com:3128"

    def test_reconfiguring_discards_the_previous_password(self, store: EgressStore, cred_store: CredentialStore):
        """A superseded credential must not linger in the store."""
        menu = _import_menu()
        cred_store.set("old-ref", "old-pass")
        store.save(EgressRecord(proxy_url="http://old.example.com:3128", username="old", auth_ref="old-ref"))

        with (
            patch("sys.stdin.isatty", return_value=True),
            patch(f"{_MOD}.SelectionMenu.show", side_effect=["Configure gateway", "Back"]),
            patch(f"{_MOD}.prompt_text", side_effect=["new.example.com:9999", "newuser"]),
            patch(f"{_MOD}.prompt_secret", return_value="new-pass"),
            patch(f"{_MOD}.prompt_confirm", side_effect=[True, False]),
        ):
            menu(cred_store, store)

        assert cred_store.get("old-ref") is None
        record = store.load()
        assert record is not None and record.auth_ref is not None
        assert cred_store.get(record.auth_ref) == "new-pass"


class TestRemoveFlow:
    def test_removes_gateway_and_credential(self, store: EgressStore, cred_store: CredentialStore):
        menu = _import_menu()
        cred_store.set("ref", "pass")
        store.save(EgressRecord(proxy_url="http://proxy.example.com:3128", username="u", auth_ref="ref"))

        with (
            patch("sys.stdin.isatty", return_value=True),
            patch(f"{_MOD}.SelectionMenu.show", side_effect=["Remove gateway", "Back"]),
            patch(f"{_MOD}.prompt_confirm", return_value=True),
        ):
            menu(cred_store, store)

        assert store.load() is None
        assert cred_store.get("ref") is None

    def test_declining_keeps_the_gateway(self, store: EgressStore, cred_store: CredentialStore):
        menu = _import_menu()
        store.save(EgressRecord(proxy_url="http://proxy.example.com:3128"))

        with (
            patch("sys.stdin.isatty", return_value=True),
            patch(f"{_MOD}.SelectionMenu.show", side_effect=["Remove gateway", "Back"]),
            patch(f"{_MOD}.prompt_confirm", return_value=False),
        ):
            menu(cred_store, store)

        assert store.load() is not None


class TestMenuShape:
    def test_actions_are_limited_when_nothing_is_configured(self, store: EgressStore, cred_store: CredentialStore):
        menu = _import_menu()

        with (
            patch("sys.stdin.isatty", return_value=True),
            patch(f"{_MOD}.SelectionMenu") as mock_menu_cls,
        ):
            mock_menu_cls.return_value.show.return_value = None
            menu(cred_store, store)

        _title, actions = mock_menu_cls.call_args.args
        assert actions == ["Configure gateway", "Back"]

    def test_actions_expand_once_configured(self, store: EgressStore, cred_store: CredentialStore):
        menu = _import_menu()
        store.save(EgressRecord(proxy_url="http://proxy.example.com:3128"))

        with (
            patch("sys.stdin.isatty", return_value=True),
            patch(f"{_MOD}.SelectionMenu") as mock_menu_cls,
        ):
            mock_menu_cls.return_value.show.return_value = None
            menu(cred_store, store)

        _title, actions = mock_menu_cls.call_args.args
        assert actions == ["Configure gateway", "Test connection", "Remove gateway", "Back"]

    def test_non_tty_is_rejected(self, store: EgressStore, cred_store: CredentialStore):
        menu = _import_menu()

        with patch("sys.stdin.isatty", return_value=False), pytest.raises(Exception, match="interactive"):
            menu(cred_store, store)


class TestShowAndTest:
    """R12a: the non-interactive subcommands."""

    def test_show_reports_nothing_configured(self, store: EgressStore, cred_store: CredentialStore):
        from kitty.cli.egress_cmd import run_egress_show

        assert run_egress_show(cred_store, store) == 1

    def test_show_masks_the_password(self, store: EgressStore, cred_store: CredentialStore, capsys):
        from kitty.cli.egress_cmd import run_egress_show

        cred_store.set("ref", "sup3rs3cret")
        store.save(EgressRecord(proxy_url="http://proxy.example.com:3128", username="u", auth_ref="ref"))

        assert run_egress_show(cred_store, store) == 0

        out = capsys.readouterr().out
        assert "proxy.example.com" in out
        assert "sup3rs3cret" not in out

    def test_test_reports_the_observed_public_ip(self, store: EgressStore, cred_store: CredentialStore, capsys):
        from kitty.cli.egress_cmd import run_egress_test

        store.save(EgressRecord(proxy_url="http://proxy.example.com:3128"))

        with patch(f"{_MOD}._probe", return_value=("185.1.2.3", 41, None)):
            assert run_egress_test(cred_store, store) == 0

        assert "185.1.2.3" in capsys.readouterr().out

    def test_test_reports_failure(self, store: EgressStore, cred_store: CredentialStore):
        from kitty.cli.egress_cmd import run_egress_test

        store.save(EgressRecord(proxy_url="http://proxy.example.com:3128"))

        with patch(f"{_MOD}._probe", return_value=(None, 12, "ClientProxyConnectionError: refused")):
            assert run_egress_test(cred_store, store) == 1
