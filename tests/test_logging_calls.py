"""Unified logging and LLM call-audit tests (the atap.log framework +
llm_calls.jsonl persistence)."""

from __future__ import annotations

import json
import logging

from atap.llm import FakeLLMClient
from atap.log import attach_run_log, get_logger, setup_logging


# ------------------------------------------------------------- call audit --


def test_fake_client_call_log_records_prompt_and_response(tmp_path):
    client = FakeLLMClient(responses=["hello", "world"])
    client.attach_call_log(tmp_path / "llm_calls.jsonl")
    r1 = client.complete([{"role": "user", "content": "question A"}], tag="t1")
    r2 = client.complete([{"role": "user", "content": "question B"}], model="m", tag="t2")
    assert r1.text == "hello" and r2.text == "world"
    recs = [
        json.loads(line)
        for line in (tmp_path / "llm_calls.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(recs) == 2
    assert recs[0]["client"] == "fake" and recs[0]["tag"] == "t1"
    assert recs[0]["ok"] is True and recs[0]["response"] == "hello"
    assert recs[0]["messages"] == [{"role": "user", "content": "question A"}]
    assert recs[0]["latency_ms"] >= 0 and "ts" in recs[0]
    assert recs[1]["model"] == "m" and recs[1]["schema"] is None
    # without an attached log, silently write nothing (the library-mode default is unchanged)
    FakeLLMClient(responses=["x"]).complete(
        [{"role": "user", "content": "q"}], tag="t3"
    )


def test_fake_client_call_log_records_error(tmp_path):
    client = FakeLLMClient(responses=[])   # no responses and the tag has no handler branch
    client.attach_call_log(tmp_path / "calls.jsonl")
    try:
        client.complete([{"role": "user", "content": "q"}], tag="__nope__")
        raise AssertionError("should have raised LLMError")
    except Exception:
        pass
    rec = json.loads((tmp_path / "calls.jsonl").read_text(encoding="utf-8"))
    assert rec["ok"] is False and "LLMError" in rec["error"]


def test_openai_client_wrapper_logs_usage_without_network(tmp_path):
    """__new__ bypasses __init__ (no openai import, no network); after
    monkeypatching _create, verify the auditing wrapper: usage tokens land
    in the record, LLMResult.usage is backfilled."""
    import types

    from atap.llm.openai_client import OpenAICompatibleLLMClient

    client = OpenAICompatibleLLMClient.__new__(OpenAICompatibleLLMClient)
    client.model = "test-model"
    client.temperature = 0.0
    client.max_completion_tokens = 128
    client.max_retries = 0
    client.retry_base_delay = 0.0
    client.request_interval = 0.0
    client._last_call_ts = 0.0
    client.calls = []
    client.retries = []
    client.http_requests = 0
    client._call_log_path = None

    resp = types.SimpleNamespace(
        usage=types.SimpleNamespace(
            prompt_tokens=11, completion_tokens=7, total_tokens=18
        ),
        choices=[types.SimpleNamespace(
            message=types.SimpleNamespace(content="plain text reply")
        )],
    )

    def fake_create(payload_messages, use_model, tag):
        client.http_requests += 1   # simulate the HTTP counting of the real _create
        return resp

    client._create = fake_create
    client.attach_call_log(tmp_path / "calls.jsonl")
    result = client.complete([{"role": "user", "content": "hello"}], tag="real")
    assert result.text == "plain text reply"
    assert result.usage == {
        "prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18,
    }
    rec = json.loads((tmp_path / "calls.jsonl").read_text(encoding="utf-8"))
    assert rec["client"] == "openai" and rec["tag"] == "real" and rec["ok"] is True
    assert rec["usage"]["total_tokens"] == 18 and rec["model"] == "test-model"
    assert rec["http_requests"] == 1 and rec["latency_ms"] >= 0


def test_openai_client_empty_choices_retry_and_crash_audit(tmp_path):
    """OpenRouter occasionally returns HTTP 200 with choices=null: _create
    treats it as a retryable error with backoff, raises LLMError when
    retries are exhausted; an unexpected non-LLMError exception is likewise
    fully audited and then re-raised unchanged (real incident: the 13th call
    of the nemotron-ultra smoke test returned 200/choices=null, raised a
    bare TypeError, and the audit had no record)."""
    import types

    from atap.llm import LLMError
    from atap.llm.openai_client import OpenAICompatibleLLMClient

    good = types.SimpleNamespace(
        usage=types.SimpleNamespace(
            prompt_tokens=1, completion_tokens=1, total_tokens=2
        ),
        choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))],
    )
    empty = types.SimpleNamespace(choices=None, error=None)

    def make_client(create):
        c = OpenAICompatibleLLMClient.__new__(OpenAICompatibleLLMClient)
        c.model = "m"
        c.temperature = 0.0
        c.max_completion_tokens = 64
        c.max_retries = 2
        c.retry_base_delay = 0.0
        c.request_interval = 0.0
        c._last_call_ts = 0.0
        c.calls = []
        c.retries = []
        c.http_requests = 0
        c._call_log_path = None
        c._client = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(create=create)
            )
        )
        return c

    # 1) first empty choices -> one backoff retry then success
    seen = []

    def flaky_create(**kw):
        seen.append(kw.get("model"))
        return empty if len(seen) == 1 else good

    c = make_client(flaky_create)
    c.attach_call_log(tmp_path / "c1.jsonl")
    assert c.complete([{"role": "user", "content": "q"}], tag="flaky").text == "ok"
    assert len(c.retries) == 1 and "empty choices" in c.retries[0]["error"]
    rec = json.loads((tmp_path / "c1.jsonl").read_text(encoding="utf-8"))
    assert rec["ok"] is True and rec["http_requests"] == 2

    # 2) persistent empty choices -> retries exhausted, LLMError raised, error audited
    c = make_client(lambda **kw: empty)
    c.attach_call_log(tmp_path / "c2.jsonl")
    try:
        c.complete([{"role": "user", "content": "q"}], tag="dead")
        raise AssertionError("should have raised LLMError")
    except LLMError as e:
        assert "empty choices" in str(e)
    rec = json.loads((tmp_path / "c2.jsonl").read_text(encoding="utf-8"))
    assert rec["ok"] is False and rec["http_requests"] == 3

    # 3) an unexpected exception (not LLMError) is likewise audited, then re-raised unchanged
    c = make_client(lambda **kw: good)
    c.attach_call_log(tmp_path / "c3.jsonl")

    def boom(messages, *, schema, model, tag):
        raise TypeError("boom")

    c._complete = boom
    try:
        c.complete([{"role": "user", "content": "q"}], tag="crash")
        raise AssertionError("should have raised TypeError")
    except TypeError:
        pass
    rec = json.loads((tmp_path / "c3.jsonl").read_text(encoding="utf-8"))
    assert rec["ok"] is False and "TypeError" in rec["error"]


# ------------------------------------------------------------ log framework --


def test_setup_logging_idempotent_and_run_log_replaces(tmp_path):
    logger = setup_logging()
    n_handlers = len(logger.handlers)
    setup_logging(verbose=True)
    assert len(logger.handlers) == n_handlers  # no stacking
    assert logger.level == logging.DEBUG

    attach_run_log(tmp_path / "a.log")
    attach_run_log(tmp_path / "b.log")   # replace semantics: no stacking of file handlers
    files = [h for h in logger.handlers if isinstance(h, logging.FileHandler)]
    assert len(files) == 1
    get_logger("t").info("hello-runlog")
    files[0].flush()
    assert "hello-runlog" in (tmp_path / "b.log").read_text(encoding="utf-8")
    assert not (tmp_path / "a.log").exists() or "hello-runlog" not in (
        tmp_path / "a.log").read_text(encoding="utf-8")

    setup_logging()   # cleanup: avoid leaking file handlers into later tests


def test_cli_demo_writes_run_log_and_llm_call_audit(tmp_path):
    """End to end: atap demo writes run.log (process) + llm_calls.jsonl (the
    full prompt/response of every call); acceptance numbers unchanged."""
    from atap.cli import main

    out = tmp_path / "demo"
    assert main(["demo", "--out", str(out)]) == 0

    run_log = (out / "run.log").read_text(encoding="utf-8")
    assert "run start" in run_log
    assert "represent/canonical_events" in run_log   # stage execution record
    assert "run finished" in run_log

    lines = (out / "llm_calls.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) >= 6
    recs = [json.loads(x) for x in lines]
    tags = {r["tag"] for r in recs}
    assert "mast_judge" in tags and "all_at_once" in tags
    assert all(r["ok"] for r in recs)
    # audit records = the judges' actual inputs; same-origin check of the
    # anti-leak iron rule (prompts carry no ground-truth fields)
    blob = "".join(json.dumps(r["messages"], ensure_ascii=False) for r in recs)
    assert "injected_fault" not in blob and "ground_truth" not in blob
    setup_logging()   # cleanup: drop the file handlers attached during the demo call
