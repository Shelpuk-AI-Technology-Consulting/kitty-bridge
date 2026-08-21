"""Tests for the egress gateway store and configuration resolution."""

from __future__ import annotations

import json
import uuid
from concurrent.futures import ThreadPoolExecutor

import pytest

from kitty.credentials.file_backend import FileBackend
from kitty.credentials.store import CredentialStore
from kitty.egress import ENV_PROXY
from kitty.egress_store import STORE_VERSION, EgressRecord, EgressStore, resolve_egress


@pytest.fixture()
def store(tmp_path) -> EgressStore:
    return EgressStore(path=tmp_path / "egress.json")


@pytest.fixture()
def cred_store(tmp_path) -> CredentialStore:
    return CredentialStore(backends=[FileBackend(path=tmp_path / "credentials.json")])


@pytest.fixture(autouse=True)
def _clear_env(monkeypatch) -> None:
    """Keep a developer's own KITTY_EGRESS_PROXY out of these tests."""
    monkeypatch.delenv(ENV_PROXY, raising=False)


class TestEgressStore:
    def test_empty_store_loads_none(self, store: EgressStore):
        assert store.load() is None

    def test_round_trip(self, store: EgressStore):
        record = EgressRecord(proxy_url="http://proxy.example.com:12323", username="u", auth_ref="ref-1")
        store.save(record)

        assert store.load() == record

    def test_round_trip_without_credentials(self, store: EgressStore):
        store.save(EgressRecord(proxy_url="http://proxy.example.com:3128"))

        loaded = store.load()
        assert loaded is not None
        assert loaded.username is None
        assert loaded.auth_ref is None

    def test_save_replaces_previous_entry(self, store: EgressStore):
        store.save(EgressRecord(proxy_url="http://old.example.com:3128"))
        store.save(EgressRecord(proxy_url="http://new.example.com:3128"))

        loaded = store.load()
        assert loaded is not None
        assert loaded.proxy_url == "http://new.example.com:3128"

    def test_delete_clears_the_entry(self, store: EgressStore):
        store.save(EgressRecord(proxy_url="http://proxy.example.com:3128"))
        store.delete()

        assert store.load() is None

    def test_password_is_never_written_to_disk(self, tmp_path, store: EgressStore):
        """The whole point of the auth_ref indirection."""
        store.save(EgressRecord(proxy_url="http://proxy.example.com:3128", username="u", auth_ref="ref-1"))

        raw = (tmp_path / "egress.json").read_text()
        assert "ref-1" in raw
        assert "password" not in raw.lower()

    def test_corrupt_file_disables_egress_rather_than_raising(self, tmp_path, caplog):
        path = tmp_path / "egress.json"
        path.write_text("{not json")

        assert EgressStore(path=path).load() is None
        assert any("corrupt" in r.message.lower() for r in caplog.records)

    def test_unknown_version_is_ignored(self, tmp_path):
        path = tmp_path / "egress.json"
        path.write_text(json.dumps({"version": STORE_VERSION + 1, "egress": {"proxy_url": "http://h:1"}}))

        assert EgressStore(path=path).load() is None

    def test_entry_missing_proxy_url_is_ignored(self, tmp_path):
        path = tmp_path / "egress.json"
        path.write_text(json.dumps({"version": STORE_VERSION, "egress": {"username": "u"}}))

        assert EgressStore(path=path).load() is None


class TestResolveEgress:
    """R4: precedence is CLI > env > stored."""

    def test_returns_none_when_nothing_configured(self, store: EgressStore, cred_store: CredentialStore):
        assert resolve_egress(store=store, cred_store=cred_store) is None

    def test_stored_value_is_used(self, store: EgressStore, cred_store: CredentialStore):
        ref = str(uuid.uuid4())
        cred_store.set(ref, "stored-pass")
        store.save(EgressRecord(proxy_url="http://stored.example.com:3128", username="stored-user", auth_ref=ref))

        cfg = resolve_egress(store=store, cred_store=cred_store)

        assert cfg is not None
        assert cfg.proxy_url == "http://stored.example.com:3128"
        assert cfg.username == "stored-user"
        assert cfg.password == "stored-pass"

    def test_env_overrides_stored(self, monkeypatch, store: EgressStore, cred_store: CredentialStore):
        store.save(EgressRecord(proxy_url="http://stored.example.com:3128"))
        monkeypatch.setenv(ENV_PROXY, "http://env.example.com:9999")

        cfg = resolve_egress(store=store, cred_store=cred_store)

        assert cfg is not None
        assert cfg.proxy_url == "http://env.example.com:9999"

    def test_cli_overrides_env_and_stored(self, monkeypatch, store: EgressStore, cred_store: CredentialStore):
        store.save(EgressRecord(proxy_url="http://stored.example.com:3128"))
        monkeypatch.setenv(ENV_PROXY, "http://env.example.com:9999")

        cfg = resolve_egress(cli_proxy="http://cli.example.com:1111", store=store, cred_store=cred_store)

        assert cfg is not None
        assert cfg.proxy_url == "http://cli.example.com:1111"

    def test_env_credentials_are_split_out(self, monkeypatch, store: EgressStore, cred_store: CredentialStore):
        monkeypatch.setenv(ENV_PROXY, "http://envuser:envpass@env.example.com:9999")

        cfg = resolve_egress(store=store, cred_store=cred_store)

        assert cfg is not None
        assert (cfg.username, cfg.password) == ("envuser", "envpass")

    def test_blank_env_is_treated_as_unset(self, monkeypatch, store: EgressStore, cred_store: CredentialStore):
        store.save(EgressRecord(proxy_url="http://stored.example.com:3128"))
        monkeypatch.setenv(ENV_PROXY, "   ")

        cfg = resolve_egress(store=store, cred_store=cred_store)

        assert cfg is not None
        assert cfg.proxy_url == "http://stored.example.com:3128"

    def test_malformed_cli_value_raises(self, store: EgressStore, cred_store: CredentialStore):
        with pytest.raises(ValueError, match="scheme"):
            resolve_egress(cli_proxy="socks5://proxy.example.com:1080", store=store, cred_store=cred_store)

    def test_missing_stored_credential_raises_rather_than_silently_disabling_auth(
        self, store: EgressStore, cred_store: CredentialStore
    ):
        """Dropping the password would authenticate as empty and fail obscurely."""
        store.save(EgressRecord(proxy_url="http://proxy.example.com:3128", username="u", auth_ref="gone"))

        with pytest.raises(ValueError, match="reconfigure"):
            resolve_egress(store=store, cred_store=cred_store)


class TestConcurrentResolution:
    """Several kitty processes resolving the stored gateway at once.

    Every launch resolves egress from the same shared files (``egress.json`` +
    the credential store, both guarded by file locks). This models concurrent
    ``kitty`` processes the way production runs them: each builds its own store
    instances over the same on-disk paths.
    """

    def test_concurrent_resolutions_all_succeed_identically(
        self, tmp_path, store: EgressStore, cred_store: CredentialStore
    ):
        ref = str(uuid.uuid4())
        cred_store.set(ref, "shared-pass")
        store.save(EgressRecord(proxy_url="http://proxy.example.com:3128", username="u", auth_ref=ref))

        egress_path = tmp_path / "egress.json"
        cred_path = tmp_path / "credentials.json"

        def resolve_once(_):
            return resolve_egress(
                store=EgressStore(path=egress_path),
                cred_store=CredentialStore(backends=[FileBackend(path=cred_path)]),
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            configs = list(pool.map(resolve_once, range(16)))

        assert len(configs) == 16
        assert all(config == configs[0] for config in configs)
        assert configs[0].password == "shared-pass"
