"""Langfuse live-instance bridge -- external evaluation over a running deployment.

Unlike io/langfuse.py (offline file adapter: ingestion-batch JSON in, batch
JSON out), this module talks to a **live** Langfuse REST API and turns atap
into the "external evaluation pipeline" role Langfuse reserves for third-party
judges: pull traces -> run the atap pipeline -> write the attribution results
back as Scores on the original traces.

REST surface used (Basic auth: public key as user, secret key as password):
* ``GET  /api/public/traces``      -- paginated trace list (``page``/``limit``,
  envelope ``{data, meta}``; ``fromTimestamp`` is sent server-side when
  ``since`` is given and ALSO enforced client-side, so a backend that ignores
  the parameter cannot silently widen the window [adaptation]);
* ``GET  /api/public/observations?traceId=`` / ``GET /api/public/scores?traceId=``
  -- same envelope, fetched per trace;
* ``POST /api/public/scores``      -- score write-back (v3/v4 stable);
* ``POST /api/public/ingestion``   -- demo seeding only (v3 batch endpoint;
  deprecated on v4 cloud with 2026-11 shutdown -- self-hosted v3 remains the
  supported push target, see docker-compose.langfuse.yml).

Generic mapping (live observations carry NO ``metadata["atap"]`` namespace --
this is the real gap versus the offline adapter, where kind/agent/refs were
exported explicitly):
* observation type -> R0 kind: GENERATION->LLM_CALL, EVENT->AGENT_MESSAGE,
  SPAN->TOOL_CALL for leaves / AGENT_MESSAGE for containers (a span with
  children is orchestration, not a tool call) [adaptation, heuristic];
* observation ``name`` keyword overrides any type: handoff/transfer/delegate
  -> HANDOFF, verif/evaluat/judge/guardrail/critic/assert -> VERIFIER;
* ``agent`` is taken from the first hit of a configurable metadata key chain
  (default agent/agent_name/gen_ai.agent.name/llm.app), else "unknown" --
  Langfuse has no first-class agent field, and NOT falling back to the trace
  name keeps multi-agent attribution from collapsing onto one agent
  [adaptation];
* ``input`` -> payload (dict merge), ``output`` -> ``payload["content"]``
  (stringified when structured); ``level=ERROR`` prefixes ``error: `` onto the
  content so the error-observation convention of core/render keeps working
  [adaptation];
* refs stay empty: reference edges have no Langfuse counterpart (same
  declaration as the offline adapter);
* **atap-namespace fast path**: observations pushed by ``push_langfuse``
  carry the original R0 fields under ``metadata["atap"]`` -- when present,
  kind/agent/action/phase/ts are restored from it and refs are remapped via
  an event-id -> observation-id alias table, so the push -> live-pull
  roundtrip is lossless instead of degrading to the generic heuristics;
* ``startTime`` (ISO) -> node ``ts``; unparsable values fall back to the
  flattened ordinal inside canonical_events;
* outcome: derived from a Langfuse score via the ``outcome_from`` spec
  (``{score, op, value}``, numeric compare when both sides are numeric);
  with no spec or no matching score the conservative default is
  ``success=False`` -- every trace then enters analyze and the judge decides,
  rather than silently skipping possibly-broken runs [fix].

Idempotency: write-back skips traces that already carry any ``atap:*`` score
(the per-trace score list fetched during the pull is reused -- no extra
request); ``force=True`` re-scores. Combined with ``--since`` this replaces a
cursor file: re-running the same window writes nothing twice.

Leak discipline: score comments are assembled from ``Hypothesis`` fields only
(never ``trajectory.meta``), and the push path goes through
export_langfuse/export_safe_meta, so ground-truth keys (injected_fault/
origin_fault/fault_removed) cannot reach the external system.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from typing import Any, Callable, Iterable

from atap.core.schema import Outcome, Trajectory

from atap.io.langfuse import export_langfuse

#: score names atap writes (all under the ``atap:`` namespace)
SCORE_ROOT_CAUSE = "atap:root-cause"
SCORE_CONFIDENCE = "atap:confidence"
SCORE_STEP = "atap:blamed-step"
ATAP_SCORE_NAMES = frozenset({SCORE_ROOT_CAUSE, SCORE_CONFIDENCE, SCORE_STEP})

#: metadata key chain probed (in order) for the acting agent of an observation
DEFAULT_AGENT_KEYS = ("agent", "agent_name", "gen_ai.agent.name", "llm.app")

_HANDOFF_HINTS = ("handoff", "hand_over", "handover", "transfer", "delegate", "escalate")
_VERIFIER_HINTS = ("verif", "evaluat", "judge", "guardrail", "critic", "assert")

_PAGE_SIZE = 50


class LangfuseError(RuntimeError):
    """Live Langfuse API failure (connection, auth, non-2xx)."""


def _api_base(base_url: str) -> str:
    b = base_url.rstrip("/")
    if b.endswith("/api/public"):
        return b
    return b + "/api/public"


def _iso_to_epoch(s: Any) -> float | None:
    if not isinstance(s, str) or not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


_SINCE_RE = re.compile(r"^(\d+)\s*([smhdw])$", re.I)
_SINCE_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def _parse_since(spec: str) -> tuple[str, float]:
    """``'24h'``/``'30m'``/``'7d'`` or an ISO 8601 timestamp -> (iso, epoch).

    The ISO form is what goes into the ``fromTimestamp`` query parameter; the
    epoch form drives the mandatory client-side re-check."""
    m = _SINCE_RE.match(spec.strip())
    if m:
        epoch = datetime.now(timezone.utc).timestamp() - int(m.group(1)) * _SINCE_UNITS[m.group(2).lower()]
    else:
        epoch = _iso_to_epoch(spec)
        if epoch is None:
            raise ValueError(
                f"cannot parse --since {spec!r}: use a relative form like '24h'/'7d' "
                f"or an ISO 8601 timestamp"
            )
    iso = datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat().replace("+00:00", "Z")
    return iso, epoch


class LangfuseClient:
    """Minimal REST client (httpx, lazy import -- only the ``langfuse`` extra carries it)."""

    def __init__(
        self,
        base_url: str,
        public_key: str,
        secret_key: str,
        *,
        timeout: float = 30.0,
        transport: Any = None,
    ) -> None:
        import httpx  # lazy: keeps atap importable without the langfuse extra

        self._http = httpx.Client(
            base_url=_api_base(base_url),
            auth=httpx.BasicAuth(public_key, secret_key),
            timeout=timeout,
            transport=transport,
        )

    @classmethod
    def from_env(
        cls,
        base_url: str | None = None,
        public_key: str | None = None,
        secret_key: str | None = None,
        *,
        transport: Any = None,
    ) -> "LangfuseClient":
        import os

        base_url = base_url or os.environ.get("LANGFUSE_BASE_URL") or os.environ.get("LANGFUSE_HOST")
        public_key = public_key or os.environ.get("LANGFUSE_PUBLIC_KEY")
        secret_key = secret_key or os.environ.get("LANGFUSE_SECRET_KEY")
        missing = [
            name for name, v in (
                ("LANGFUSE_BASE_URL", base_url),
                ("LANGFUSE_PUBLIC_KEY", public_key),
                ("LANGFUSE_SECRET_KEY", secret_key),
            ) if not v
        ]
        if missing:
            raise LangfuseError(
                f"missing Langfuse credentials: {missing} (set them as environment "
                f"variables; keys are never read from or written to disk)"
            )
        return cls(base_url, public_key, secret_key, transport=transport)

    # -- transport ----------------------------------------------------------

    def _request(self, method: str, path: str, *, params: dict | None = None, json_body: Any = None) -> Any:
        try:
            r = self._http.request(method, path, params=params, json=json_body)
        except Exception as e:  # httpx transport errors (connection/refused/dns)
            raise LangfuseError(f"Langfuse request failed ({method} {path}): {e}") from e
        if r.status_code < 200 or r.status_code >= 300:
            raise LangfuseError(
                f"Langfuse {method} {path} returned {r.status_code}: {r.text[:300]}"
            )
        if not r.content:
            return None
        try:
            return r.json()
        except ValueError as e:
            raise LangfuseError(f"Langfuse {method} {path} returned non-JSON body: {r.text[:300]}") from e

    def _paged(self, path: str, params: dict) -> list[dict]:
        """Collect every item of a ``{data, meta}``-enveloped list endpoint."""
        items: list[dict] = []
        page = 1
        while True:
            body = self._request("GET", path, params={**params, "page": page, "limit": _PAGE_SIZE})
            if isinstance(body, list):  # defensive: a version without the envelope
                return [x for x in body if isinstance(x, dict)]
            batch = [x for x in (body or {}).get("data") or [] if isinstance(x, dict)]
            items.extend(batch)
            meta = (body or {}).get("meta") or {}
            try:
                total_pages = int(meta.get("totalPages", 0))
            except (TypeError, ValueError):
                total_pages = 0
            if not batch or len(batch) < _PAGE_SIZE or (total_pages and page >= total_pages):
                return items
            page += 1

    # -- endpoints ----------------------------------------------------------

    def iter_traces(self, *, from_iso: str | None = None) -> Iterable[dict]:
        params: dict[str, Any] = {}
        if from_iso:
            params["fromTimestamp"] = from_iso
        return self._paged("/traces", params)

    def observations(self, trace_id: str) -> list[dict]:
        # traceId enforced client-side as well: some self-hosted v3 builds
        # ignore the query param and return project-wide lists, which would
        # cross-contaminate per-trace state (found against a live instance,
        # verification round 2)
        return [
            o for o in self._paged("/observations", {"traceId": trace_id})
            if o.get("traceId") == trace_id
        ]

    def scores(self, trace_id: str) -> list[dict]:
        return [
            s for s in self._paged("/scores", {"traceId": trace_id})
            if s.get("traceId") == trace_id
        ]

    def post_score(self, payload: dict) -> dict:
        out = self._request("POST", "/scores", json_body=payload)
        return out if isinstance(out, dict) else {}

    def post_ingestion(self, batch: dict) -> None:
        self._request("POST", "/ingestion", json_body=batch)


# ---------------------------------------------------------------------------
# generic live-observation -> span-tree mapping
# ---------------------------------------------------------------------------

def _kind_for(obs: dict, *, has_children: bool) -> str:
    name = str(obs.get("name") or "").lower()
    if any(h in name for h in _HANDOFF_HINTS):
        return "HANDOFF"
    if any(h in name for h in _VERIFIER_HINTS):
        return "VERIFIER"
    otype = str(obs.get("type") or "").upper()
    if otype == "GENERATION":
        return "LLM_CALL"
    if otype == "EVENT":
        return "AGENT_MESSAGE"
    # SPAN: a leaf is a tool call; a container is orchestration
    return "AGENT_MESSAGE" if has_children else "TOOL_CALL"


def _atap_ns(obs: dict) -> dict:
    """The atap namespace under observation metadata (present when the trace
    was pushed by atap itself; empty dict for third-party traces)."""
    md = obs.get("metadata")
    ns = md.get("atap") if isinstance(md, dict) else None
    return ns if isinstance(ns, dict) else {}


def _agent_for(obs: dict, agent_keys: tuple[str, ...]) -> str:
    md = obs.get("metadata")
    for k in agent_keys:
        if isinstance(md, dict) and md.get(k) is not None:
            return str(md[k])
    for k in agent_keys:
        if obs.get(k) is not None:
            return str(obs[k])
    return "unknown"


def _payload_for(obs: dict) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    inp = obs.get("input")
    if isinstance(inp, dict):
        payload.update(inp)
    elif inp is not None:
        payload["input"] = inp
    out = obs.get("output")
    if out is not None:
        payload["content"] = out if isinstance(out, str) else json.dumps(out, ensure_ascii=False)
    if str(obs.get("level") or "").upper() == "ERROR":
        # keep the error-observation convention visible downstream
        # (core/render.is_error_observation matches the "error:" prefix)
        c = payload.get("content", "")
        payload["content"] = f"error: {c if isinstance(c, str) else repr(c)}"
    return payload


def _outcome_from_scores(scores: list[dict], spec: dict | None) -> Outcome | None:
    """Derive the trajectory outcome from a Langfuse score (``{score, op, value}``).

    Returns None when no spec is given or no matching score exists -- the
    caller then applies the conservative failure default."""
    if not spec:
        return None
    name = spec.get("score")
    op = spec.get("op", "==")
    want = spec.get("value")
    if not name or op not in ("==", "!=", ">=", "<=", ">", "<"):
        raise ValueError(
            f"outcome_from requires {{score, op in ==/!=/>=/<=/>/<, value}}, got {spec!r}"
        )
    values = [s for s in scores if s.get("name") == name and s.get("value") is not None]
    if not values:
        return None
    # latest write wins: the score's own timestamp decides when present (list
    # order is not guaranteed chronological across server versions)
    best = max(values, key=lambda s: _iso_to_epoch(s.get("timestamp")) or float("-inf"))
    v = best.get("value")
    numeric = (
        isinstance(v, (int, float)) and not isinstance(v, bool)
        and isinstance(want, (int, float)) and not isinstance(want, bool)
    )
    if numeric:
        ok = {"==": v == want, "!=": v != want, ">=": v >= want,
              "<=": v <= want, ">": v > want, "<": v < want}[op]
        score = float(v)
    elif op in ("==", "!="):
        ok = (str(v) == str(want)) == (op == "==")
        score = None
    else:
        # ordered comparison on non-numeric values: unusable signal
        return Outcome(success=False, note=f"outcome_from: non-numeric value for op {op!r}")
    return Outcome(success=bool(ok), score=score, note=f"from langfuse score {name!r} ({v!r} {op} {want!r})")


def trace_to_trajectory(
    trace: dict,
    observations: list[dict],
    scores: list[dict],
    *,
    agent_keys: tuple[str, ...] = DEFAULT_AGENT_KEYS,
    outcome_from: dict | None = None,
) -> Trajectory:
    """Live trace + observations + scores -> Trajectory (nested span tree in ``raw``).

    The span tree is the same inter-layer contract the offline adapters
    produce; represent/canonical_events flattens it. Sibling order is fixed by
    (startTime, id) so repeated pulls of the same trace are byte-stable.

    atap-namespace fast path: observations pushed by ``push_langfuse`` carry
    the original R0 fields under ``metadata["atap"]`` (kind/agent/action/
    phase/refs/ts) -- when present they take priority over the generic
    heuristics, and refs are remapped from original event ids to observation
    ids through an alias table (the same trick as the offline adapter), so the
    push -> live-pull roundtrip restores the trajectory in full instead of
    degrading every agent to "unknown"."""
    obs_by_id = {o["id"]: o for o in observations if o.get("id")}
    missing_id = [o for o in observations if not o.get("id")]
    if missing_id:
        # explicit instead of a bare KeyError downstream: ids anchor the tree
        raise ValueError(
            f"trace {trace.get('id')}: {len(missing_id)} observation(s) without "
            f"an id (first: name={missing_id[0].get('name')!r}) -- refusing to "
            f"build the span tree"
        )
    alias: dict[str, str] = {}   # original atap event id -> observation id
    for o in observations:
        ns = _atap_ns(o)
        if ns.get("id"):
            alias[str(ns["id"])] = o["id"]
    children: dict[str | None, list[dict]] = defaultdict(list)
    for o in observations:
        pid = o.get("parentObservationId")
        children[pid if pid in obs_by_id else None].append(o)
    for ids in children.values():
        # chronological key, not lexicographic ISO (mixed offsets/fractions
        # sort deterministically but out of order as strings); id breaks ties
        ids.sort(key=lambda o: (_iso_to_epoch(o.get("startTime")) or 0.0, str(o.get("id"))))

    def build(o: dict) -> dict:
        ns = _atap_ns(o)
        kids = children.get(o["id"], [])
        ts = ns.get("ts") if isinstance(ns.get("ts"), (int, float)) and not isinstance(ns.get("ts"), bool) else None
        return {
            "id": o["id"],
            "logical": o.get("name") or ns.get("kind") or "step",
            "kind": str(ns.get("kind") or "") or _kind_for(o, has_children=bool(kids)),
            "agent": str(ns["agent"]) if ns.get("agent") else _agent_for(o, agent_keys),
            # ns carries the original action verbatim (possibly None) -- the
            # observation-name fallback applies only to third-party traces
            "action": ns.get("action") if ns else o.get("name"),
            "payload": _payload_for(o),
            "refs": [alias.get(r, r) for r in (ns.get("refs") or [])],
            "phase": ns.get("phase"),
            "ts": ts if ts is not None else _iso_to_epoch(o.get("startTime")),
            "children": [build(k) for k in kids],
        }

    meta: dict[str, Any] = {"task_id": trace.get("name") or trace["id"]}
    if isinstance(trace.get("metadata"), dict):
        meta.update(trace["metadata"])
    # consume the export-side outcome payload (same contract as the offline
    # adapter: the outcome lives in Trajectory.outcome, no residue in meta)
    raw_outcome = meta.pop("outcome", None)
    meta.pop("langfuse_tags", None)   # rebuilt below from the trace's own tags
    if trace.get("tags"):
        meta["langfuse_tags"] = list(trace.get("tags") or [])
    if trace.get("timestamp"):
        meta["langfuse_timestamp"] = trace["timestamp"]
    if trace.get("sessionId"):
        meta["langfuse_session_id"] = trace["sessionId"]
    if isinstance(raw_outcome, dict):
        outcome = Outcome.from_dict(raw_outcome)
    else:
        outcome = _outcome_from_scores(scores, outcome_from) or Outcome(success=False)
    return Trajectory(
        trace_id=trace["id"],
        task=str(trace.get("input") or meta["task_id"]),
        events=[],
        outcome=outcome,
        meta=meta,
        raw={"task_id": meta["task_id"], "spans": [build(o) for o in children.get(None, [])]},
    )


class LangfuseAPISource:
    """TraceSource over a live Langfuse instance (``source: {type: langfuse_api}``).

    ``load()`` pulls traces (``tags`` = AND-filter applied client-side, ``since``
    = server-side fromTimestamp + client-side re-check, ``limit`` caps the
    number of ACCEPTED traces), maps each to a Trajectory and remembers the
    per-trace score lists in ``scores_by_trace`` so the eval command can apply
    the already-scored skip without a second fetch.
    """

    def __init__(
        self,
        *,
        client: LangfuseClient | None = None,
        base_url: str | None = None,
        tags: list[str] | None = None,
        since: str | None = None,
        limit: int | None = None,
        outcome_from: dict | None = None,
        agent_keys: list[str] | None = None,
    ) -> None:
        self.client = client or LangfuseClient.from_env(base_url=base_url)
        self.tags = list(tags or [])
        self.since = since
        self.limit = limit
        self.outcome_from = outcome_from
        self.agent_keys = tuple(agent_keys) if agent_keys else DEFAULT_AGENT_KEYS
        self.scores_by_trace: dict[str, list[dict]] = {}

    def load(self) -> list[Trajectory]:
        from_iso, epoch_from = (_parse_since(self.since) if self.since else (None, None))
        wanted = set(self.tags)
        out: list[Trajectory] = []
        for t in self.client.iter_traces(from_iso=from_iso):
            if epoch_from is not None:
                ts = _iso_to_epoch(t.get("timestamp"))
                if ts is None or ts < epoch_from:
                    continue
            if wanted and not wanted <= set(t.get("tags") or []):
                continue
            tid = t.get("id")
            if not tid:
                continue
            obs = self.client.observations(tid)
            scores = self.client.scores(tid)
            self.scores_by_trace[tid] = scores
            out.append(trace_to_trajectory(
                t, obs, scores,
                agent_keys=self.agent_keys,
                outcome_from=self.outcome_from,
            ))
            if self.limit is not None and len(out) >= self.limit:
                break
        return out


# ---------------------------------------------------------------------------
# score write-back
# ---------------------------------------------------------------------------

def observation_id_by_event_index(trajectory: Trajectory) -> dict[int, str]:
    """Flattened event index -> originating Langfuse observation id.

    Replays canonical_events' DFS pre-order walk over ``raw["spans"]`` (parent
    first, then children in list order) -- the walk that assigns ``index``, so
    position i in the replay is exactly event i. Lets observation-level scores
    pin the blamed step without touching the represent layer."""
    order: list[str | None] = []

    def walk(nodes: list[dict] | None) -> None:
        for n in nodes or []:
            order.append(n.get("id"))
            walk(n.get("children"))

    raw = trajectory.raw if isinstance(trajectory.raw, dict) else {}
    walk(raw.get("spans"))
    return {i: oid for i, oid in enumerate(order) if oid}


def _top_evidence(h: Any, limit: int = 3) -> str:
    lines = [str(e) for e in (h.evidence or [])[:limit]]
    if len(h.evidence or []) > limit:
        lines.append(f"(+{len(h.evidence) - limit} more)")
    return "\n".join(f"- {ln}" for ln in lines) if lines else "- (none)"


def _comment(h: Any, comment_max: int, run_id: str = "") -> str:
    by = h.source or "atap"
    run = f", run {run_id}" if run_id else ""
    text = (
        f"atap attribution (by {by}{run}):\n"
        f"agent: {h.agent} | step: {h.step} | side: {h.responsible_side}\n"
        f"root cause: {h.root_cause}\n"
        f"fix: {h.fix_suggestion or '(none)'}\n"
        f"evidence:\n{_top_evidence(h)}"
    )
    if len(text) > comment_max:
        text = text[: comment_max - 3] + "..."
    return text


def _hyp_metadata(h: Any) -> dict:
    """The full Hypothesis verbatim, flat -- downstream reads fields directly
    instead of parsing the comment text. Same leak discipline as the comment:
    Hypothesis fields only (never trajectory.meta / GT keys)."""
    return h.to_dict()


class ScoreWriter:
    """Hypothesis -> Langfuse score payloads + write-back (with dry-run and skip).

    Every payload carries ``metadata``:

    * the full Hypothesis dict, flat (agent / step / root_cause /
      root_cause_code / responsible_side / evidence / fix_suggestion /
      confidence / source) -- machine-readable without comment parsing;
    * the caller-supplied ``run_meta`` (run_id / run_name / llm / seed):
      Langfuse scores are append-only, so repeated evaluations of the same
      trace pile up indistinguishably -- the run block is what lets a reader
      tell which evaluation batch a score came from (metadata + the run tag
      in the comment header).
    """

    def __init__(
        self,
        client: LangfuseClient | None,
        *,
        dry_run: bool = False,
        comment_max: int = 4000,
        run_meta: dict | None = None,
    ) -> None:
        self.client = client
        self.dry_run = dry_run
        self.comment_max = comment_max
        self.run_meta = dict(run_meta) if run_meta else {}
        self.run_id = str(self.run_meta.get("run_id") or "")

    def scores_for_bundle(self, bundle: Any) -> list[dict]:
        """Pure formatting (no network): top hypothesis -> trace-level scores,
        every hypothesis -> one observation-level blamed-step score."""
        hyps = bundle.hypotheses()
        if not hyps:
            return []
        tid = bundle.trace_id
        top = max(hyps, key=lambda h: h.confidence)
        payloads: list[dict] = [
            {
                "traceId": tid,
                "name": SCORE_ROOT_CAUSE,
                "value": top.root_cause_code or "unlabeled",
                "dataType": "CATEGORICAL",
                "comment": _comment(top, self.comment_max, self.run_id),
                "metadata": {**_hyp_metadata(top), **self.run_meta},
            },
            {
                "traceId": tid,
                "name": SCORE_CONFIDENCE,
                "value": round(float(top.confidence), 4),
                "dataType": "NUMERIC",
                "comment": (
                    f"confidence of the top atap hypothesis (by {top.source or 'atap'}"
                    + (f", run {self.run_id}" if self.run_id else "")
                    + ")"
                ),
                "metadata": {**_hyp_metadata(top), **self.run_meta},
            },
        ]
        obs_ids = observation_id_by_event_index(bundle.trajectory)
        seen: set[tuple[str, str]] = set()
        # highest-confidence first, so the per-observation dedup below keeps
        # the strongest hypothesis when several algorithms blame the same step
        for h in sorted(hyps, key=lambda x: -x.confidence):
            oid = obs_ids.get(h.step)
            if oid is None or (oid, SCORE_STEP) in seen:
                continue
            seen.add((oid, SCORE_STEP))
            payloads.append({
                "traceId": tid,
                "observationId": oid,
                "name": SCORE_STEP,
                "value": h.root_cause_code or "blamed",
                "dataType": "CATEGORICAL",
                "comment": f"{h.agent} @ step {h.step}: {h.root_cause}"[: self.comment_max],
                "metadata": {**_hyp_metadata(h), **self.run_meta},
            })
        return payloads

    def write_bundle(
        self,
        bundle: Any,
        *,
        prior_scores: list[dict] | None = None,
        force: bool = False,
        emit: Callable[[str], None] | None = None,
    ) -> tuple[str, int]:
        """Write scores for one bundle; returns ``(decision, n_payloads)`` with
        decision in {no-hypotheses, skipped, dry-run, written}."""
        say = emit or (lambda s: None)
        if prior_scores and not force and any(s.get("name") in ATAP_SCORE_NAMES for s in prior_scores):
            say(f"skip {bundle.trace_id}: already carries an atap:* score (use --force to re-evaluate)")
            return "skipped", 0
        payloads = self.scores_for_bundle(bundle)
        if not payloads:
            return "no-hypotheses", 0
        if self.dry_run:
            for p in payloads:
                where = p.get("observationId") or "trace"
                say(f"[dry-run] {p['traceId']} {where} {p['name']}={p['value']!r} ({p['dataType']})")
            return "dry-run", len(payloads)
        if self.client is None:
            raise LangfuseError("ScoreWriter needs a client for non-dry-run writes")
        for p in payloads:
            self.client.post_score(p)
        say(f"scored {bundle.trace_id}: {len(payloads)} score(s) written")
        return "written", len(payloads)


def push_langfuse(
    traces: list[Trajectory], client: LangfuseClient, *, tags: list[str] | None = None
) -> int:
    """Seed a Langfuse instance with atap traces (demo round-trip).

    Reuses the offline v3 ingestion-batch exporter (GT-stripped); returns the
    number of ingestion events (1 trace-create + n observation events).
    ``tags`` lands on every trace-create body -- pushed corpora are best
    scoped by tag: push with ``--tags corpus-X``, evaluate with
    ``atap langfuse-eval --tags corpus-X`` to run exactly that batch.
    Event timestamps are restamped with the real push time (the exporter's
    epoch-0 pin is for offline determinism; on a live instance it would hide
    the corpus outside the UI's default time window).

    Requires FLATTENED trajectories: the exporter consumes
    ``trajectory.events``, so a raw-span-only trajectory (empty events + raw
    spans) would go out as a bare trace-create and silently lose every event.
    The CLI path flattens first (``cli._ensure_flattened``); direct library
    callers must flatten via represent/canonical_events -- this guard raises
    instead of losing data [fix, verification round]."""
    raw_only = [
        t.trace_id for t in traces
        if not t.events and isinstance(t.raw, dict) and t.raw.get("spans")
    ]
    if raw_only:
        raise ValueError(
            f"push_langfuse: {len(raw_only)} raw-span-only trajectory(es) "
            f"(first: {raw_only[0]}): empty events would export as a bare "
            f"trace-create and lose every event -- flatten first via "
            f"represent/canonical_events (the atap CLI does this automatically)"
        )
    batch = export_langfuse(traces, external=True)   # strict GT mode: qrels/fault-text stay local
    if tags:
        for ev in batch["batch"]:
            if ev.get("type") == "trace-create":
                ev["body"]["tags"] = list(tags)
    # The deterministic exporter pins epoch 0, which makes a live-seeded
    # corpus invisible under the UI's default time window (last N days).
    # Live pushes get real wall-clock timestamps instead -- one shared "now"
    # per call keeps event order stable (sibling sort falls back to id on
    # equal timestamps, mirroring the pull-side mapping rule).
    now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"
    for ev in batch["batch"]:
        ev["timestamp"] = now
    client.post_ingestion(batch)
    return len(batch["batch"])
