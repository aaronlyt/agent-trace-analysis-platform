"""OpenAI-compatible client unit tests (transport stubbed -- no network;
parameter contract only)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

openai = pytest.importorskip("openai")

from atap.llm.openai_client import OpenAICompatibleLLMClient  # noqa: E402


def _bad_request(message: str) -> Exception:
    """A BadRequestError carrying only what the fallback detection reads
    (isinstance + message); openai 3.x builds its response objects from an
    HTTP backend not installed here, so __new__ bypasses that constructor."""
    e = openai.BadRequestError.__new__(openai.BadRequestError)
    Exception.__init__(e, message)
    return e


def _client(monkeypatch, **kw) -> OpenAICompatibleLLMClient:
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-key")
    return OpenAICompatibleLLMClient(model="m", request_interval=0.0, **kw)


def _fake_transport(calls: list[dict], fail_on_max_completion: bool):
    """A chat.completions.create stub that records the kwargs it was sent;
    optionally rejects the modern parameter the way an old backend does."""

    def create(**kw):
        calls.append(kw)
        if fail_on_max_completion and "max_completion_tokens" in kw:
            raise _bad_request(
                "Unrecognized request argument supplied: max_completion_tokens"
            )
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=None,
        )

    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )


def test_sends_max_completion_tokens(monkeypatch):
    """The wire parameter must be max_completion_tokens (the constructor
    argument's own name): newer reasoning models -- o-series / gpt-5 --
    reject max_tokens outright (review 2026-08-27)."""
    c = _client(monkeypatch)
    calls: list[dict] = []
    c._client = _fake_transport(calls, fail_on_max_completion=False)
    r = c.complete([{"role": "user", "content": "hi"}], tag="t")
    assert r.text == "ok"
    assert len(calls) == 1
    assert calls[0]["max_completion_tokens"] == 4096
    assert "max_tokens" not in calls[0]


def test_falls_back_to_max_tokens_once_when_rejected(monkeypatch):
    """An old OpenAI-compatible backend that 400s on the parameter itself
    gets one transparent max_tokens fallback, which then sticks for the
    rest of the run (recorded in the retry audit)."""
    c = _client(monkeypatch)
    calls: list[dict] = []
    c._client = _fake_transport(calls, fail_on_max_completion=True)
    r = c.complete([{"role": "user", "content": "hi"}], tag="t1")
    assert r.text == "ok"
    assert ["max_completion_tokens" in kw for kw in calls] == [True, False]
    assert c._legacy_max_tokens is True
    assert any("max_completion_tokens" in e["error"] for e in c.retries)
    # the fallback decision sticks: the next call goes straight to max_tokens
    c.complete([{"role": "user", "content": "hi"}], tag="t2")
    assert len(calls) == 3
    assert "max_tokens" in calls[2] and "max_completion_tokens" not in calls[2]


def test_non_param_400_is_not_a_fallback_trigger(monkeypatch):
    """Only a 400 that names max_completion_tokens triggers the fallback --
    an ordinary bad request (bad model name etc.) must surface, not
    silently retry with the legacy parameter."""
    c = _client(monkeypatch)
    calls: list[dict] = []

    def create(**kw):
        calls.append(kw)
        raise _bad_request("The model `m` does not exist")

    c._client = SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=create))
    )
    with pytest.raises(Exception, match="does not exist"):
        c.complete([{"role": "user", "content": "hi"}], tag="t")
    assert c._legacy_max_tokens is False
    # every attempt still used the modern parameter
    assert all("max_completion_tokens" in kw for kw in calls)


def test_extra_body_forwarded_on_every_request(monkeypatch):
    """Provider-specific fields (e.g. DeepSeek ``thinking: {type: disabled}``)
    ride along on every create call -- including the legacy max_tokens
    fallback branch -- and stay absent when not configured."""
    eb = {"thinking": {"type": "disabled"}}
    c = _client(monkeypatch, extra_body=eb)
    calls: list[dict] = []
    c._client = _fake_transport(calls, fail_on_max_completion=False)
    c.complete([{"role": "user", "content": "hi"}], tag="t")
    assert calls[0]["extra_body"] == eb

    # legacy fallback branch keeps forwarding it too
    c2 = _client(monkeypatch, extra_body=eb)
    calls2: list[dict] = []
    c2._client = _fake_transport(calls2, fail_on_max_completion=True)
    c2.complete([{"role": "user", "content": "hi"}], tag="t")
    assert all(kw["extra_body"] == eb for kw in calls2)

    # default: the kwarg is not sent at all
    c3 = _client(monkeypatch)
    calls3: list[dict] = []
    c3._client = _fake_transport(calls3, fail_on_max_completion=False)
    c3.complete([{"role": "user", "content": "hi"}], tag="t")
    assert "extra_body" not in calls3[0]
