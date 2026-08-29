"""Live-Langfuse bridge tests -- fully offline and deterministic.

Unit paths run against ``httpx.MockTransport`` (no sockets); the CLI
end-to-end path runs against a threaded stdlib stub HTTP server so the whole
command (env credentials -> pull -> pipeline -> score write-back -> skip on
re-run) is exercised exactly as a user would invoke it.
"""

from __future__ import annotations

import http.server
import json
import threading
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import httpx
import pytest

import atap  # noqa: F401  registration bootstrap
from atap.cli import main as cli_main
from atap.core.bundle import TrajectoryBundle
from atap.core.config import ConfigError
from atap.core.context import RunContext
from atap.core.registry import create
from atap.core.schema import Hypothesis, Outcome, TraceEvent, Trajectory
from atap.io import LangfuseAPISource, LangfuseClient, ScoreWriter, push_langfuse
from atap.io.jsonl_store import build_source
from atap.io.langfuse import LangfuseTraceSource
from atap.io.langfuse_live import _parse_since, observation_id_by_event_index


# ---------------------------------------------------------------------------
# stub API state (shapes follow the v3 public REST envelopes)
# ---------------------------------------------------------------------------

def _obs(oid, otype, name, *, parent=None, inp=None, out=None, metadata=None,
         level=None, start="2026-08-29T10:00:00Z"):
    o = {
        "id": oid, "type": otype, "name": name, "startTime": start,
        "parentObservationId": parent, "input": inp, "output": out,
        "metadata": metadata,
    }
    if level:
        o["level"] = level
    return o


TRACES = [
    {"id": "t-ok", "name": "research-qa", "timestamp": "2026-08-29T10:00:00Z",
     "input": "question: ...", "tags": ["prod", "exp3"]},
    {"id": "t-fail", "name": "research-qa", "timestamp": "2026-08-29T11:00:00Z",
     "input": "question: ...", "tags": ["prod"]},
    {"id": "t-old", "name": "research-qa", "timestamp": "2026-08-20T00:00:00Z",
     "input": "question: old", "tags": ["prod"]},
]

OBSERVATIONS = {
    "t-ok": [
        _obs("o1", "GENERATION", "plan", metadata={"agent": "planner"},
             inp={"messages": "plan the task"}, out="plan: search then report",
             start="2026-08-29T10:00:00Z"),
        _obs("o2", "SPAN", "web_search", metadata={"agent": "searcher"},
             inp={"query": "agent traces"}, out="search results: 2 docs [d1, d2]",
             start="2026-08-29T10:00:05Z"),
        _obs("o3", "GENERATION", "answer", metadata={"agent": "reporter"},
             out="based on d1: answer ...", start="2026-08-29T10:00:10Z"),
    ],
    "t-fail": [
        _obs("f1", "GENERATION", "plan", metadata={"agent": "planner"},
             out="plan: search then report", start="2026-08-29T11:00:00Z"),
        _obs("f2", "SPAN", "web_search", metadata={"agent": "searcher"},
             inp={"query": "x"}, out="tool call failed: invalid arguments",
             level="ERROR", start="2026-08-29T11:00:05Z"),
        _obs("f3", "GENERATION", "answer", metadata={"agent": "reporter"},
             out="based on d3: answer ...", start="2026-08-29T11:00:10Z"),
    ],
    "t-old": [
        _obs("g1", "GENERATION", "note", metadata={"agent": "planner"},
             out="standalone decision", start="2026-08-20T00:00:00Z"),
    ],
}

SCORES = {
    "t-ok": [{"id": "s1", "name": "user_feedback", "value": 1, "dataType": "NUMERIC"}],
    "t-fail": [],
    "t-old": [],
}


def _make_api(traces=TRACES, observations=OBSERVATIONS, scores=SCORES):
    """MockTransport-backed fake API; returns (transport, state dict).

    POSTed scores accumulate into ``scores`` so the skip-on-rerun path can be
    exercised against the same fake instance."""
    state = {"posts": [], "ingestion": None, "requests": [], "auth": None}

    def envelope(items):
        return httpx.Response(200, json={
            "data": items,
            "meta": {"page": 1, "limit": 50, "totalItems": len(items), "totalPages": 1},
        })

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        params = dict(request.url.params)
        state["requests"].append((request.method, path, dict(params)))
        state["auth"] = request.headers.get("authorization")
        if request.method == "GET":
            if path == "/api/public/traces":
                return envelope(traces)
            if path == "/api/public/observations":
                tid = params.get("traceId", "")
                return envelope([dict(o, traceId=tid)
                                 for o in observations.get(tid, [])])
            if path == "/api/public/scores":
                tid = params.get("traceId", "")
                return envelope([dict(s, traceId=tid)
                                 for s in scores.get(tid, [])])
        elif request.method == "POST":
            body = json.loads(request.content)
            if path == "/api/public/scores":
                state["posts"].append(body)
                scores.setdefault(body["traceId"], []).append(body)
                return httpx.Response(200, json={"id": f"sc-{len(state['posts'])}", **body})
            if path == "/api/public/ingestion":
                state["ingestion"] = body
                return httpx.Response(200, json={"code": 200})
        return httpx.Response(404, json={"error": f"no route {request.method} {path}"})

    return httpx.MockTransport(handler), state


def _client(transport):
    return LangfuseClient("http://langfuse.test", "pk-test", "sk-test", transport=transport)


def _flatten(traces):
    ctx = RunContext()
    bundles = [TrajectoryBundle(t) for t in traces]
    for b in bundles:
        create("represent", "canonical_events").run_one(b, ctx)
    return bundles


# ---------------------------------------------------------------------------
# source: generic mapping / filters / outcome
# ---------------------------------------------------------------------------

def test_source_maps_live_observations_to_r0():
    transport, state = _make_api()
    src = LangfuseAPISource(
        client=_client(transport),
        outcome_from={"score": "user_feedback", "op": ">=", "value": 1},
    )
    traces = src.load()
    by_id = {t.trace_id: t for t in traces}
    assert set(by_id) == {"t-ok", "t-fail", "t-old"}

    ok = by_id["t-ok"]
    assert ok.outcome.success is True and ok.outcome.score == 1.0
    assert ok.meta["task_id"] == "research-qa"
    assert ok.meta["langfuse_tags"] == ["prod", "exp3"]
    # conservative default: no matching score -> failure, not a silent skip
    assert by_id["t-fail"].outcome.success is False
    assert by_id["t-old"].outcome.success is False

    # basic auth reached every request (public key as user)
    assert state["auth"] is not None and state["auth"].startswith("Basic ")

    _flatten(traces)
    evs = by_id["t-fail"].events
    assert [e.kind for e in evs] == ["LLM_CALL", "TOOL_CALL", "LLM_CALL"]
    assert [e.agent for e in evs] == ["planner", "searcher", "reporter"]
    assert [e.action for e in evs] == ["plan", "web_search", "answer"]
    # input merges into payload; output lands in content; level=ERROR is
    # visible through the error-prefix convention
    assert evs[1].payload["query"] == "x"
    assert evs[1].payload["content"].startswith("error: tool call failed")
    # sibling order follows startTime
    assert [e.id for e in evs] == ["e000", "e001", "e002"]
    # DFS pre-order replay recovers the originating observation ids
    assert observation_id_by_event_index(by_id["t-fail"]) == {0: "f1", 1: "f2", 2: "f3"}


def test_source_container_span_and_name_overrides():
    obs = [
        _obs("c1", "SPAN", "crew", metadata={"agent": "crew"},
             start="2026-08-29T10:00:00Z"),
        _obs("c1a", "GENERATION", "think", parent="c1", metadata={"agent": "planner"},
             start="2026-08-29T10:00:01Z"),
        _obs("c2", "EVENT", "handoff-to-reporter", start="2026-08-29T10:00:02Z"),
        _obs("c3", "SPAN", "verify_answer", start="2026-08-29T10:00:03Z"),
    ]
    transport, _ = _make_api(
        traces=[{"id": "t1", "name": "crew-run", "timestamp": "2026-08-29T10:00:00Z",
                 "input": "task", "tags": []}],
        observations={"t1": obs},
    )
    src = LangfuseAPISource(client=_client(transport))
    traces = src.load()
    _flatten(traces)
    evs = traces[0].events
    # container SPAN -> AGENT_MESSAGE (orchestration, not a tool call);
    # name keywords override the type defaults for HANDOFF / VERIFIER
    assert [e.kind for e in evs] == ["AGENT_MESSAGE", "LLM_CALL", "HANDOFF", "VERIFIER"]
    assert evs[2].agent == "unknown"  # no metadata agent -> explicit unknown


def test_source_tags_since_limit():
    transport, state = _make_api()
    # tags: AND semantics, applied client-side
    src = LangfuseAPISource(client=_client(transport), tags=["exp3"])
    assert [t.trace_id for t in src.load()] == ["t-ok"]

    # since: enforced client-side even though the server ignores fromTimestamp
    src = LangfuseAPISource(client=_client(transport), since="2026-08-29T00:00:00Z")
    assert {t.trace_id for t in src.load()} == {"t-ok", "t-fail"}
    sent = [r for r in state["requests"] if r[1] == "/api/public/traces"]
    assert sent and sent[-1][2].get("fromTimestamp") == "2026-08-29T00:00:00Z"

    # limit caps accepted traces
    src = LangfuseAPISource(client=_client(transport), limit=1)
    assert len(src.load()) == 1


def test_parse_since_forms():
    import time

    iso, epoch = _parse_since("24h")
    now = time.time()
    assert now - 24 * 3600 - 60 <= epoch <= now - 24 * 3600 + 60
    assert iso.endswith("Z")
    iso2, epoch2 = _parse_since("2026-08-01T00:00:00Z")
    assert epoch2 == pytest.approx(1785542400.0, abs=5)   # 2026-08-01T00:00:00Z
    assert iso2.startswith("2026-08-01T")
    with pytest.raises(ValueError):
        _parse_since("yesterday")


def test_build_source_langfuse_api(monkeypatch):
    monkeypatch.setenv("LANGFUSE_BASE_URL", "http://langfuse.test")
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk")
    src = build_source({"type": "langfuse_api", "tags": ["prod"], "since": "24h"})
    assert isinstance(src, LangfuseAPISource)
    assert src.tags == ["prod"] and src.since == "24h"
    # file-backed types still require a path
    with pytest.raises(ConfigError):
        build_source({"type": "jsonl"})
    # credentials come from the environment only
    monkeypatch.delenv("LANGFUSE_PUBLIC_KEY")
    with pytest.raises(Exception):
        build_source({"type": "langfuse_api"})


# ---------------------------------------------------------------------------
# ScoreWriter
# ---------------------------------------------------------------------------

def _bundle_with_hypothesis():
    t = Trajectory(
        trace_id="t-9",
        task="do something",
        events=[TraceEvent(id="e000", ts=0.0, kind="LLM_CALL", agent="planner", index=0)],
        outcome=Outcome(success=False),
        meta={"injected_fault": "step_repetition", "qrels": {"E": ["d1"]}},  # GT must not leak
        raw={"task_id": "x", "spans": [{
            "id": "obs-9", "logical": "plan", "kind": "LLM_CALL", "agent": "planner",
            "payload": {}, "refs": [], "children": [],
        }]},
    )
    b = TrajectoryBundle(t)
    b.put("attribute", "all_at_once", {"hypotheses": [Hypothesis(
        agent="planner", step=0, root_cause="repeats the same tool call",
        root_cause_code="FM-1.3", responsible_side="model",
        evidence=["e000 repeated web_search", "e001 repeated web_search"],
        fix_suggestion="Use the existing results to move forward.",
        confidence=0.7, source="all_at_once",
    )]})
    return b


def test_scorewriter_format_and_leak_freedom():
    payloads = ScoreWriter(None).scores_for_bundle(_bundle_with_hypothesis())
    by_name = {p["name"]: p for p in payloads}
    assert set(by_name) == {"atap:root-cause", "atap:confidence", "atap:blamed-step"}
    assert by_name["atap:root-cause"]["value"] == "FM-1.3"
    assert by_name["atap:root-cause"]["dataType"] == "CATEGORICAL"
    assert by_name["atap:confidence"]["dataType"] == "NUMERIC"
    assert by_name["atap:confidence"]["value"] == 0.7
    step = by_name["atap:blamed-step"]
    assert step["observationId"] == "obs-9"   # event index 0 -> originating observation
    comment = by_name["atap:root-cause"]["comment"]
    assert "agent: planner" in comment
    assert "Use the existing results" in comment            # fix suggestion present
    # metadata carries the full hypothesis verbatim: machine-readable without
    # parsing the comment
    for p in payloads:
        meta = p["metadata"]
        assert meta["agent"] == "planner"
        assert meta["step"] == 0
        assert meta["root_cause_code"] == "FM-1.3"
        assert meta["responsible_side"] == "model"
        assert meta["source"] == "all_at_once"
        assert meta["evidence"] == ["e000 repeated web_search", "e001 repeated web_search"]
        assert meta["confidence"] == 0.7
    # leak discipline: GT/meta keys never reach the external system
    for p in payloads:
        blob = json.dumps(p)
        assert "injected_fault" not in blob and "qrels" not in blob


def test_scorewriter_run_meta_distinguishes_batches():
    """Langfuse scores are append-only, so two evaluation batches on one trace
    are indistinguishable unless every payload tags its run identity."""
    b = _bundle_with_hypothesis()
    run_a = {"run_id": "eval_a", "run_name": "langfuse-eval",
             "llm": "openai:deepseek-v4-flash", "seed": 7}
    run_b = {"run_id": "eval_b", "run_name": "langfuse-eval", "llm": "fake", "seed": 7}
    pa = ScoreWriter(None, run_meta=run_a).scores_for_bundle(b)
    pb = ScoreWriter(None, run_meta=run_b).scores_for_bundle(b)
    for p in pa + pb:
        meta = p["metadata"]
        assert meta["agent"] == "planner"    # hypothesis fields stay flat
        assert set(("run_id", "run_name", "llm", "seed")) <= set(meta)
    assert {p["metadata"]["run_id"] for p in pa} == {"eval_a"}
    assert {p["metadata"]["run_id"] for p in pb} == {"eval_b"}
    assert pa[0]["comment"].startswith("atap attribution (by all_at_once, run eval_a):")
    assert pa[1]["comment"].endswith("(by all_at_once, run eval_a)")
    # without run_meta nothing breaks, comments stay in the legacy shape
    legacy = ScoreWriter(None).scores_for_bundle(b)
    assert legacy[0]["comment"].startswith("atap attribution (by all_at_once):")
    assert "run_id" not in legacy[0]["metadata"]


def test_scorewriter_skip_force_dryrun():
    transport, state = _make_api()
    writer = ScoreWriter(_client(transport))
    b = _bundle_with_hypothesis()
    prior = [{"name": "atap:root-cause", "value": "FM-1.3"}]

    assert writer.write_bundle(b, prior_scores=prior) == ("skipped", 0)
    assert state["posts"] == []

    assert writer.write_bundle(b, prior_scores=prior, force=True) == ("written", 3)
    assert len(state["posts"]) == 3

    dry = ScoreWriter(_client(transport), dry_run=True)
    lines = []
    assert dry.write_bundle(b, emit=lines.append) == ("dry-run", 3)
    assert len(state["posts"]) == 3          # nothing new was sent
    assert any("atap:root-cause" in ln for ln in lines)

    # non-dry-run without a client is a hard error, not a silent no-op
    with pytest.raises(Exception):
        ScoreWriter(None).write_bundle(b)


# ---------------------------------------------------------------------------
# push (demo seeding): live push payload stays consumable by the file adapter
# ---------------------------------------------------------------------------

def test_push_langfuse_feeds_the_offline_adapter(tmp_path):
    from atap.sandbox import ToySandbox

    traces = [ToySandbox().generate("q-trajaudit", None)]
    _flatten(traces)  # push consumes the flattened R0 stream
    traces[0].meta["injected_fault"] = "info_withholding"  # GT present pre-push
    traces[0].meta["qrels"] = {"E": ["d1"], "G": ["d1"]}    # gold sets present pre-push
    traces[0].outcome.note = "failed: info withheld by the searcher"

    transport, state = _make_api()
    n_events = push_langfuse(traces, _client(transport))
    batch = state["ingestion"]["batch"]
    assert n_events == len(batch)
    assert any(e["type"] == "trace-create" for e in batch)
    # GT never leaves through the push path: the fault key, the gold sets,
    # and the fault-mechanics note are all external-mode ground truth
    blob = json.dumps(batch)
    assert "injected_fault" not in blob and "qrels" not in blob
    assert "info withheld" not in blob
    # offline file export keeps qrels by contract (rg_ug data dependency)
    from atap.io.langfuse import export_langfuse
    offline = export_langfuse(traces)
    assert "qrels" in json.dumps(offline)

    # the pushed batch is a valid v3 ingestion file for the offline importer
    f = tmp_path / "pushed.json"
    f.write_text(json.dumps({"batch": batch}), encoding="utf-8")
    imported = LangfuseTraceSource(str(f)).load()
    _flatten(imported)
    assert len(imported) == 1
    o, i = traces[0], imported[0]
    assert i.trace_id == o.trace_id and i.task == o.task
    assert [(e.kind, e.agent, e.action) for e in i.events] == \
           [(e.kind, e.agent, e.action) for e in o.events]


def test_push_rejects_raw_span_only_trajectories():
    """Library callers who skip flattening get a loud error, not silent loss
    (a bare trace-create would drop every event)."""
    from atap.sandbox import ToySandbox

    transport, state = _make_api()
    raw_only = ToySandbox().generate("q-trajaudit", None)   # events=[], raw spans
    assert not raw_only.events
    with pytest.raises(ValueError, match="flatten"):
        push_langfuse([raw_only], _client(transport))
    assert state["ingestion"] is None                       # nothing was sent


def test_push_langfuse_tags_scope_a_corpus_batch():
    """--tags is the corpus scoping handle, and live pushes carry real
    timestamps: the exporter's epoch-0 pin is offline determinism only -- on
    a live instance it would hide the corpus outside the UI's default
    time window (last N days)."""
    from atap.sandbox import ToySandbox
    from datetime import datetime, timezone

    traces = [ToySandbox().generate("q-trajaudit", None)]
    _flatten(traces)
    transport, state = _make_api()
    push_langfuse(traces, _client(transport), tags=["corpus-x", "drift"])
    batch = state["ingestion"]["batch"]
    creates = [e for e in batch if e["type"] == "trace-create"]
    assert len(creates) == 1
    assert creates[0]["body"]["tags"] == ["corpus-x", "drift"]
    # observation events stay untouched (tags are a trace-level concept)
    assert all("tags" not in e["body"] for e in batch if e["type"] != "trace-create")
    # every event restamped at push time: within the last minute, not epoch 0
    recent = datetime.now(timezone.utc).timestamp() - 60
    for e in batch:
        ts = datetime.fromisoformat(e["timestamp"].replace("Z", "+00:00")).timestamp()
        assert ts > recent, f"{e['id']} still carries a pinned/old timestamp"


def test_live_pull_restores_atap_pushed_traces():
    """push -> live-pull roundtrip: the atap namespace under observation
    metadata restores kind/agent/action/refs and the trace-level outcome --
    full fidelity, not the degraded generic mapping (agent=unknown)."""
    from atap.io.langfuse import export_langfuse
    from atap.io.langfuse_live import DEFAULT_AGENT_KEYS, trace_to_trajectory
    from atap.sandbox import ToySandbox

    orig = [
        ToySandbox().generate("q-trajaudit", None),
        ToySandbox().generate("q-who-when", "info_withholding"),
    ]
    _flatten(orig)
    batch = export_langfuse(orig)["batch"]
    trace_bodies = {e["body"]["id"]: e["body"] for e in batch if e["type"] == "trace-create"}
    obs_by_trace: dict[str, list[dict]] = {}
    for e in batch:
        if e["type"] in ("span-create", "generation-create"):
            obs_by_trace.setdefault(e["body"]["traceId"], []).append(e["body"])

    pulled = [
        trace_to_trajectory(trace_bodies[t.trace_id], obs_by_trace[t.trace_id], [],
                            agent_keys=DEFAULT_AGENT_KEYS)
        for t in orig
    ]
    _flatten(pulled)
    for o, p in zip(orig, pulled):
        assert p.trace_id == o.trace_id
        assert p.outcome.success == o.outcome.success
        assert [(e.kind, e.agent, e.action, e.refs, e.phase) for e in p.events] == \
               [(e.kind, e.agent, e.action, e.refs, e.phase) for e in o.events]


def test_client_enforces_trace_filter_on_leaky_servers():
    """Regression (found against a live self-hosted v3 build): some servers
    ignore the traceId query param on /scores and /observations and return
    project-wide lists -- the client must enforce the filter itself, or
    skip-scored decisions and outcome_from derivation read OTHER traces'
    scores/observations."""
    ALL_SCORES = [
        {"id": "s1", "name": "user_feedback", "value": 1, "traceId": "t-a",
         "timestamp": "2026-08-29T10:00:00Z"},
        {"id": "s2", "name": "atap:root-cause", "value": 0, "stringValue": "FM-1.3",
         "traceId": "t-b", "timestamp": "2026-08-29T10:00:00Z"},
    ]
    ALL_OBS = [
        _obs("oa", "GENERATION", "plan", metadata={"agent": "planner"}),
        _obs("ob", "GENERATION", "plan", metadata={"agent": "planner"}),
    ]
    ALL_OBS[0]["traceId"] = "t-a"
    ALL_OBS[1]["traceId"] = "t-b"

    def leaky_handler(request: httpx.Request) -> httpx.Response:
        # ignores traceId on purpose: every per-trace query returns everything
        path = request.url.path
        if request.method == "GET":
            items = ALL_SCORES if path.endswith("/scores") else ALL_OBS
            return httpx.Response(200, json={
                "data": items,
                "meta": {"page": 1, "limit": 50, "totalItems": len(items), "totalPages": 1},
            })
        return httpx.Response(404, json={"error": "?"})

    client = LangfuseClient("http://leaky.test", "pk", "sk",
                            transport=httpx.MockTransport(leaky_handler))
    # t-a's own list only: the atap score of t-b must not leak in (it would
    # make t-a look already-scored), and vice versa for observations
    assert [s["id"] for s in client.scores("t-a")] == ["s1"]
    assert [s["id"] for s in client.scores("t-b")] == ["s2"]
    assert [o["id"] for o in client.observations("t-a")] == ["oa"]
    assert [o["id"] for o in client.observations("t-b")] == ["ob"]


# ---------------------------------------------------------------------------
# CLI end-to-end against a threaded stub server (real HTTP, still offline)
# ---------------------------------------------------------------------------

_EVAL_CFG = """\
run_name: lf-eval
seed: 7
source:
  type: langfuse_api
  outcome_from: {score: user_feedback, op: ">=", value: 1}
llm:
  type: fake
stages:
  represent:
    - canonical_events
  analyze:
    - judge_eval
  classify:
    - mast_judge
  attribute:
    - all_at_once
"""


class _StubLangfuse(http.server.BaseHTTPRequestHandler):
    """Serves the same fake API as _make_api over real HTTP; posted scores
    accumulate in ``self.server.store`` and become visible to later pulls."""

    def log_message(self, *args):  # silence the default stderr chatter
        pass

    def _send(self, obj):
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _envelope(self, items):
        self._send({"data": items,
                    "meta": {"page": 1, "limit": 50, "totalItems": len(items), "totalPages": 1}})

    def do_GET(self):
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        st = self.server.store
        if u.path == "/api/public/traces":
            self._envelope(st["traces"])
        elif u.path == "/api/public/observations":
            tid = q.get("traceId", "")
            self._envelope([dict(o, traceId=tid)
                            for o in st["observations"].get(tid, [])])
        elif u.path == "/api/public/scores":
            tid = q.get("traceId", "")
            self._envelope([dict(s, traceId=tid)
                            for s in st["scores"].get(tid, [])])
        else:
            self._send({"error": f"no route {u.path}"})

    def do_POST(self):
        u = urlparse(self.path)
        n = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(n)) if n else {}
        st = self.server.store
        if u.path == "/api/public/scores":
            st["posts"].append(body)
            st["scores"].setdefault(body["traceId"], []).append(body)
            self._send({"id": f"sc-{len(st['posts'])}", **body})
        elif u.path == "/api/public/ingestion":
            st["ingestion"] = body
            self._send({"code": 200})
        else:
            self._send({"error": f"no route {u.path}"})


@pytest.fixture()
def stub_langfuse():
    import copy

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), _StubLangfuse)
    srv.store = {
        "traces": copy.deepcopy(TRACES),
        "observations": copy.deepcopy(OBSERVATIONS),
        "scores": copy.deepcopy(SCORES),
        "posts": [],
        "ingestion": None,
    }
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield f"http://127.0.0.1:{srv.server_address[1]}", srv.store
    finally:
        srv.shutdown()
        srv.server_close()


def test_cli_langfuse_eval_write_skip_force_cycle(tmp_path, stub_langfuse, monkeypatch, capsys):
    url, store = stub_langfuse
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf")
    cfg = tmp_path / "eval.yaml"
    cfg.write_text(_EVAL_CFG, encoding="utf-8")

    def run(out, *extra):
        return cli_main([
            "langfuse-eval", "--config", str(cfg), "--out", str(tmp_path / out),
            "--base-url", url, *extra,
        ])

    # round 1: t-ok is a success via outcome_from; t-fail and t-old are
    # failures whose FakeLLM fallback attribution yields 3 scores each
    assert run("r1") == 0
    posted = store["posts"]
    assert len(posted) == 6
    by_trace: dict[str, list[dict]] = {}
    for p in posted:
        by_trace.setdefault(p["traceId"], []).append(p["name"])
    assert set(by_trace) == {"t-fail", "t-old"}
    assert set(by_trace["t-fail"]) == {"atap:root-cause", "atap:confidence", "atap:blamed-step"}
    blamed = [p for p in posted if p["name"] == "atap:blamed-step"]
    assert all(p.get("observationId") in {"f1", "f2", "f3", "g1"} for p in blamed)
    assert "scores: 6 written across 2 trace(s)" in capsys.readouterr().out
    # batch identity: every posted score tags the run it came from
    # (run_id = --out dir name; llm label = fake stack of this config)
    for p in posted:
        assert p["metadata"]["run_id"] == "r1"
        assert p["metadata"]["run_name"] == "lf-eval"
        assert p["metadata"]["llm"] == "fake"
        assert isinstance(p["metadata"]["seed"], int)
        assert p["metadata"]["agent"]    # hypothesis fields flat in metadata

    # round 2 (fresh out dir): every scored trace is skipped -- idempotent
    assert run("r2") == 0
    assert len(store["posts"]) == 6
    assert "skipped(already scored)=2" in capsys.readouterr().out

    # round 3: --force re-evaluates
    assert run("r3", "--force") == 0
    assert len(store["posts"]) == 12

    # dry-run sends nothing
    n_before = len(store["posts"])
    assert run("r4", "--force", "--dry-run") == 0
    assert len(store["posts"]) == n_before
    assert "(dry-run, nothing written)" in capsys.readouterr().out


def test_cli_langfuse_eval_requires_canonical_events(tmp_path, stub_langfuse, monkeypatch):
    url, _ = stub_langfuse
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf")
    cfg = tmp_path / "no-rep.yaml"
    cfg.write_text(
        "run_name: bad\nllm: {type: fake}\nstages:\n  analyze:\n    - judge_eval\n",
        encoding="utf-8",
    )
    rc = cli_main([
        "langfuse-eval", "--config", str(cfg), "--out", str(tmp_path / "r"),
        "--base-url", url,
    ])
    assert rc == 1


def test_cli_langfuse_push(tmp_path, stub_langfuse, monkeypatch, capsys):
    url, store = stub_langfuse
    monkeypatch.setenv("LANGFUSE_PUBLIC_KEY", "pk-lf")
    monkeypatch.setenv("LANGFUSE_SECRET_KEY", "sk-lf")
    from atap.sandbox import ToySandbox

    traces = [ToySandbox().generate("q-trajaudit", None)]
    f = tmp_path / "traces.jsonl"
    f.write_text("\n".join(json.dumps(t.to_dict()) for t in traces), encoding="utf-8")

    assert cli_main(["langfuse-push", "--traces", str(f), "--base-url", url]) == 0
    assert store["ingestion"] is not None
    assert any(e["type"] == "trace-create" for e in store["ingestion"]["batch"])
    assert "pushed" in capsys.readouterr().out
