"""Tests for optional Sentry integration in src/sentry.py."""

import importlib.metadata
import logging
import sys
import types
from typing import Any

import pytest

from src.sentry import (
    _parse_bool,
    capture_exception,
    init_sentry,
    set_tag,
    set_user_context,
)

SENTRY_ENV_VARS = (
    "SENTRY_DSN",
    "SENTRY_TRACES_SAMPLE_RATE",
    "SENTRY_SEND_DEFAULT_PII",
    "SENTRY_ENVIRONMENT",
    "SENTRY_RELEASE",
    "SENTRY_PROFILE_SESSION_SAMPLE_RATE",
    "SENTRY_PROFILE_LIFECYCLE",
    "SENTRY_ENABLE_LOGS",
)


class _LiteLLMIntegration:
    """Stub for LiteLLMIntegration."""


class _LoggingIntegration:
    """Stub for LoggingIntegration capturing constructor kwargs."""

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs


@pytest.fixture
def fake_sentry(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Install fake sentry_sdk modules and clear SENTRY_* env vars."""
    state: dict[str, Any] = {
        "init_calls": [],
        "captured": [],
        "users": [],
        "tags": [],
        "send_default_pii": False,
    }

    class _FakeClient:
        def should_send_default_pii(self) -> bool:
            return bool(state["send_default_pii"])

    sentry_sdk_mod = types.SimpleNamespace(
        init=lambda **kw: state["init_calls"].append(kw),
        capture_exception=lambda err: state["captured"].append(err),
        set_user=lambda user: state["users"].append(user),
        set_tag=lambda key, value: state["tags"].append((key, value)),
        get_client=lambda: _FakeClient(),
    )
    modules = {
        "sentry_sdk": sentry_sdk_mod,
        "sentry_sdk.integrations": types.SimpleNamespace(),
        "sentry_sdk.integrations.litellm": types.SimpleNamespace(
            LiteLLMIntegration=_LiteLLMIntegration
        ),
        "sentry_sdk.integrations.logging": types.SimpleNamespace(
            LoggingIntegration=_LoggingIntegration
        ),
    }
    for name, mod in modules.items():
        monkeypatch.setitem(sys.modules, name, mod)
    for var in SENTRY_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("SENTRY_DSN", "https://key@sentry.invalid/1")
    return state


@pytest.mark.parametrize(
    ("value", "default", "expected"),
    [
        (None, False, False),
        (None, True, True),
        ("", True, True),
        ("   ", False, False),
        ("true", False, True),
        ("TRUE", False, True),
        ("1", False, True),
        ("yes", False, True),
        ("on", False, True),
        ("false", True, False),
        ("0", True, False),
        ("no", True, False),
    ],
)
def test_parse_bool(value: str | None, default: bool, expected: bool) -> None:
    assert _parse_bool(value, default=default) is expected


@pytest.mark.parametrize("dsn", [None, ""])
def test_init_sentry_disabled_without_dsn(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, dsn: str | None
) -> None:
    if dsn is None:
        monkeypatch.delenv("SENTRY_DSN", raising=False)
    else:
        monkeypatch.setenv("SENTRY_DSN", dsn)
    with caplog.at_level(logging.WARNING):
        assert init_sentry() is False
    assert caplog.records == []


def test_init_sentry_warns_when_sdk_missing(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("SENTRY_DSN", "https://key@sentry.invalid/1")
    monkeypatch.setitem(sys.modules, "sentry_sdk", None)
    with caplog.at_level(logging.WARNING):
        assert init_sentry() is False
    assert "Sentry SDK not installed" in caplog.text


def test_init_sentry_defaults_and_release_lookup(
    fake_sentry: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    asked: list[str] = []

    def fake_version(name: str) -> str:
        asked.append(name)
        return "0.3.2"

    monkeypatch.setattr(importlib.metadata, "version", fake_version)

    with caplog.at_level(logging.INFO):
        assert init_sentry() is True

    assert asked == ["webex-bot-ai"]
    [kwargs] = fake_sentry["init_calls"]
    integrations = kwargs.pop("integrations")
    assert len(integrations) == 2
    assert isinstance(integrations[0], _LiteLLMIntegration)
    assert isinstance(integrations[1], _LoggingIntegration)
    assert integrations[1].kwargs == {
        "level": logging.WARNING,
        "event_level": logging.ERROR,
        "sentry_logs_level": logging.WARNING,
    }
    assert kwargs == {
        "dsn": "https://key@sentry.invalid/1",
        "traces_sample_rate": 0.1,
        "send_default_pii": False,
        "environment": None,
        "release": "0.3.2",
        "profile_session_sample_rate": 0.1,
        "profile_lifecycle": "trace",
        "enable_logs": True,
        "include_local_variables": False,
    }
    assert "Sentry monitoring enabled" in caplog.text


def test_init_sentry_custom_env_and_pii_enabled(
    fake_sentry: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SENTRY_TRACES_SAMPLE_RATE", "0.5")
    monkeypatch.setenv("SENTRY_SEND_DEFAULT_PII", "yes")
    monkeypatch.setenv("SENTRY_ENVIRONMENT", "staging")
    monkeypatch.setenv("SENTRY_RELEASE", "2.0.0")
    monkeypatch.setenv("SENTRY_PROFILE_SESSION_SAMPLE_RATE", "0.25")
    monkeypatch.setenv("SENTRY_PROFILE_LIFECYCLE", "manual")
    monkeypatch.setenv("SENTRY_ENABLE_LOGS", "true")

    assert init_sentry() is True
    [kwargs] = fake_sentry["init_calls"]
    integrations = kwargs.pop("integrations")
    assert isinstance(integrations[1], _LoggingIntegration)
    assert integrations[1].kwargs["level"] == logging.INFO
    assert integrations[1].kwargs["sentry_logs_level"] == logging.INFO
    assert kwargs == {
        "dsn": "https://key@sentry.invalid/1",
        "traces_sample_rate": 0.5,
        "send_default_pii": True,
        "environment": "staging",
        "release": "2.0.0",
        "profile_session_sample_rate": 0.25,
        "profile_lifecycle": "manual",
        "enable_logs": True,
        "include_local_variables": True,
    }


def test_init_sentry_logs_disabled_and_package_not_found(
    fake_sentry: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    def missing(_name: str) -> str:
        raise importlib.metadata.PackageNotFoundError

    monkeypatch.setattr(importlib.metadata, "version", missing)
    monkeypatch.setenv("SENTRY_ENABLE_LOGS", "false")
    monkeypatch.setenv("SENTRY_PROFILE_LIFECYCLE", "invalid-mode")

    assert init_sentry() is True
    [kwargs] = fake_sentry["init_calls"]
    assert len(kwargs["integrations"]) == 1
    assert kwargs["release"] is None
    assert kwargs["profile_lifecycle"] == "trace"
    assert kwargs["enable_logs"] is False


def test_init_sentry_error_handling(
    fake_sentry: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    monkeypatch.setenv("SENTRY_TRACES_SAMPLE_RATE", "not-a-float")
    with caplog.at_level(logging.ERROR):
        assert init_sentry() is False
    assert "Failed to initialize Sentry" in caplog.text
    assert fake_sentry["init_calls"] == []

    caplog.clear()
    monkeypatch.setenv("SENTRY_TRACES_SAMPLE_RATE", "0.1")

    def boom(**_kw: Any) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(sys.modules["sentry_sdk"], "init", boom)
    with caplog.at_level(logging.ERROR):
        assert init_sentry() is False
    assert "Failed to initialize Sentry: boom" in caplog.text


def test_sentry_helper_functions_with_and_without_sdk(
    fake_sentry: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    err = ValueError("test error")
    capture_exception(err)
    assert fake_sentry["captured"] == [err]

    set_tag("llm_model", "gpt-4o")
    assert fake_sentry["tags"] == [("llm_model", "gpt-4o")]

    fake_sentry["send_default_pii"] = False
    set_user_context("user-1", email="u@example.com", username="User")
    assert fake_sentry["users"] == [{"id": "user-1"}]

    fake_sentry["send_default_pii"] = True
    set_user_context("user-2", email="u2@example.com", username="User 2")
    assert fake_sentry["users"][-1] == {
        "id": "user-2",
        "email": "u2@example.com",
        "username": "User 2",
    }

    # Missing SDK should silently no-op
    monkeypatch.setitem(sys.modules, "sentry_sdk", None)
    capture_exception(err)
    set_user_context("user-3", email="u3@example.com")
    set_tag("k", "v")
