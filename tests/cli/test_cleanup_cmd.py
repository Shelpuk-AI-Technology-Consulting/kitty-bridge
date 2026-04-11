"""Tests for the kitty cleanup command."""

from __future__ import annotations

import json

from kitty.cli.cleanup_cmd import _detect_stale_env, _display_value, run_cleanup


def test_detect_stale_env_with_localhost_url():
    env = {
        "ANTHROPIC_BASE_URL": "http://127.0.0.1:12345",
        "ANTHROPIC_API_KEY": "test-key",
        "ANTHROPIC_MODEL": "glm-5.1",
        "API_TIMEOUT_MS": "3000000",
    }
    stale = _detect_stale_env(env)
    assert "ANTHROPIC_BASE_URL" in stale
    assert "ANTHROPIC_API_KEY" in stale
    assert "ANTHROPIC_MODEL" in stale
    assert "API_TIMEOUT_MS" not in stale


def test_detect_stale_env_with_localhost_hostname():
    env = {
        "ANTHROPIC_BASE_URL": "http://localhost:8080",
    }
    stale = _detect_stale_env(env)
    assert "ANTHROPIC_BASE_URL" in stale


def test_detect_stale_env_clean():
    env = {
        "ANTHROPIC_BASE_URL": "https://api.anthropic.com",
        "API_TIMEOUT_MS": "3000000",
    }
    stale = _detect_stale_env(env)
    assert stale == []


def test_detect_stale_env_no_base_url():
    env = {
        "API_TIMEOUT_MS": "3000000",
    }
    stale = _detect_stale_env(env)
    assert stale == []


def test_run_cleanup_removes_stale_values(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_data = {
        "env": {
            "ANTHROPIC_BASE_URL": "http://127.0.0.1:32987",
            "ANTHROPIC_API_KEY": "test-key",
            "ANTHROPIC_MODEL": "glm-5.1",
            "ANTHROPIC_DEFAULT_OPUS_MODEL": "glm-5.1",
            "API_TIMEOUT_MS": "3000000",
            "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1",
        },
        "model": "opus",
    }
    settings_path.write_text(json.dumps(settings_data))

    exit_code = run_cleanup(settings_path=settings_path)
    assert exit_code == 0

    result = json.loads(settings_path.read_text())
    env = result["env"]
    assert "ANTHROPIC_BASE_URL" not in env
    assert "ANTHROPIC_API_KEY" not in env
    assert "ANTHROPIC_MODEL" not in env
    assert "ANTHROPIC_DEFAULT_OPUS_MODEL" not in env
    # User values preserved
    assert env["API_TIMEOUT_MS"] == "3000000"
    assert env["CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC"] == "1"


def test_run_cleanup_already_clean(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_data = {
        "env": {
            "API_TIMEOUT_MS": "3000000",
        },
    }
    settings_path.write_text(json.dumps(settings_data))

    exit_code = run_cleanup(settings_path=settings_path)
    assert exit_code == 0


def test_display_value_short_string():
    assert _display_value("hello") == "hello"


def test_display_value_long_string():
    assert _display_value("x" * 41) == "x" * 37 + "..."


def test_display_value_none():
    assert _display_value(None) == "None"


def test_display_value_int():
    assert _display_value(42) == "42"


def test_run_cleanup_no_settings_file(tmp_path):
    settings_path = tmp_path / "nonexistent.json"
    exit_code = run_cleanup(settings_path=settings_path)
    assert exit_code == 0


def test_run_cleanup_non_localhost_url_preserved(tmp_path):
    settings_path = tmp_path / "settings.json"
    settings_data = {
        "env": {
            "ANTHROPIC_BASE_URL": "https://my-proxy.example.com",
            "API_TIMEOUT_MS": "3000000",
        },
    }
    settings_path.write_text(json.dumps(settings_data))

    exit_code = run_cleanup(settings_path=settings_path)
    assert exit_code == 0

    result = json.loads(settings_path.read_text())
    env = result["env"]
    # Non-localhost URL should be preserved
    assert env["ANTHROPIC_BASE_URL"] == "https://my-proxy.example.com"
