"""R1 saliency folding (SSF, Semantic Saliency Folding) -- TrajAudit, arXiv:2605.26563.

Mechanism (Algorithm 1 of the original paper, adapted to this framework's domain):
* **Patch preservation**: content matching code diff structural patterns
  (``--- a`` / ``+++ b`` / ``@@ -N,M +P,Q @@``) -> kept in full (explicitly
  shows what the agent changed);
* **Failure observation preservation**: structured error messages (``error:``
  / ``exception`` and similar prefixes, :func:`is_error_observation`) ->
  kept. The original paper uses a literal dictionary K (literal substrings,
  anywhere within the observation), but this framework's toy corpus is itself
  prose "about error analysis", so literal matching would misclassify body
  text as errors -- hence the default ``keyword_mode=strict`` (structural
  prefixes), which is a **narrowing** relative to the original K
  (``invalid/denied/missing/exhausted/timeout`` do not participate in the
  keep decision by default; ``fatal`` is a new addition). ``keyword_mode=loose``
  does not restore the original behavior; it is strict prefixes union a
  **word-boundary** dictionary (the original is pure substring matching, so
  forms like "errors"/"failover" match in the original but not at word
  boundaries) -- suited to code/tool-log domains, i.e. the original paper's
  setting [inferred];
* **Short observation exemption**: observations of length <= ``min_fold_len``
  (default 120 characters) are not folded. Algorithm 1 of the original has no
  length condition (signal-free observations are always folded); this is an
  engineering adaptation (folding short observations saves few tokens, and
  placeholder + digest may even cost more) [inferred];
* All remaining long observations -> replaced by a **reversible placeholder**
  ``⟦folded:F3 | digest...⟧``; the original text goes into a side table and
  can be unfolded on demand (a prototype of the investigator's inspection
  tool; the phase-two single-pass judge reads only the folded view, on-demand
  unfolding is left for phase-three L2 drill-down).

Differences from the original paper [inferred]: the placeholder carries a
first-line digest (<=100 characters) so that a single-pass judge can still
obtain minimal evidence from the folded view (e.g. a list of search hits);
this is an engineering adaptation to this framework's "single-pass judge
without drill-down tools" setting; the original placeholder is pure prompt
text. The original investigator consumes the folded view over multiple
drill-down rounds (20.2% of folded steps are unfolded on demand); this slice
has no drill-down, and the digest serves as a fallback for the information
loss. Two further edge behaviors: empty observations are skipped outright
(the paper's Alg.1 would literally fold them, but folding empty text is
pointless; they still count toward the ``fold_ratio`` denominator); the
``extra_keywords`` parameter allows configuring extra keep-keywords (substring
matching, empty by default, does not change default behavior).

Artifacts: ``{"fold", "table", "stats"}``; original events are not modified
(views are kept separate from data). Reference numbers from the original
paper (different measurement basis, for reference only): on average 94.6% of
steps are foldable, and only 20.2% of folded steps need on-demand unfolding --
the original denominator is all trajectory steps, while this implementation's
``fold_ratio`` denominator is the number of TOOL_RESULT events, so the two
are not directly comparable.
"""

from __future__ import annotations

import re

from atap.core.registry import register
from atap.core.render import (
    FOLD_PLACEHOLDER_RE,
    is_error_observation,
    matches_failure_keyword,
)
from atap.represent.base import Representer

# The three structural markers of a unified diff (the original paper's patch pattern P)
PATCH_PATTERNS: tuple[re.Pattern, ...] = (
    re.compile(r"^--- a/", re.MULTILINE),
    re.compile(r"^\+\+\+ b/", re.MULTILINE),
    re.compile(r"^@@ -\d+(,\d+)? \+\d+(,\d+)? @@", re.MULTILINE),
)

DIGEST_CHARS = 100


def matches_patch(text: str) -> bool:
    return any(p.search(text) for p in PATCH_PATTERNS)


@register
class SSFRepresenter(Representer):
    stage = "represent"
    name = "ssf"

    def run_one(self, bundle, ctx) -> None:
        extra = [str(k).lower() for k in (self.param("extra_keywords") or [])]
        loose = self.param("keyword_mode", "strict") == "loose"
        min_len = int(self.param("min_fold_len", 120))
        fold: dict[str, str] = {}   # event_id -> placeholder
        table: dict[str, str] = {}  # fid -> original text
        stats = {
            "n_tool_results": 0,
            "n_folded": 0,
            "n_kept_error": 0,
            "n_kept_patch": 0,
            "n_kept_short": 0,
        }
        for ev in bundle.trajectory.events:
            if ev.kind != "TOOL_RESULT":
                continue
            stats["n_tool_results"] += 1
            content = str(ev.payload.get("content", ""))
            if not content:
                continue
            if is_error_observation(content) or any(k in content.lower() for k in extra):
                stats["n_kept_error"] += 1
                continue
            if loose and matches_failure_keyword(content):
                stats["n_kept_error"] += 1
                continue
            if matches_patch(content):
                stats["n_kept_patch"] += 1
                continue
            if len(content) <= min_len:
                stats["n_kept_short"] += 1
                continue
            fid = f"F{len(table) + 1}"
            digest = " ".join(content.split())[:DIGEST_CHARS]
            fold[ev.id] = f"⟦folded:{fid} | {digest}⟧"
            table[fid] = content
            stats["n_folded"] += 1

        stats["fold_ratio"] = (
            round(stats["n_folded"] / stats["n_tool_results"], 4)
            if stats["n_tool_results"]
            else 0.0
        )
        bundle.put("represent", self.name, {"fold": fold, "table": table, "stats": stats})


def unfold(artifact: dict, fid: str) -> str:
    """Unfold on demand (a prototype of the investigator's inspection tool)."""
    content = artifact.get("table", {}).get(fid)
    if content is None:
        raise KeyError(f"{fid} not found in the folding table; available: {sorted(artifact.get('table', {}))}")
    return content


def unfold_line(line: str, artifact: dict) -> str:
    """Unfold placeholders in a rendered line back to the original text (for debugging/auditing).

    A rendered line looks like ``[8] TOOL_RESULT env read_doc :: ⟦folded:F3 | ...⟧`` --
    the placeholder sits inside the line rather than at its start, so a
    non-anchored ``search`` must locate it and replace it in place.
    """
    m = FOLD_PLACEHOLDER_RE.search(line)
    if m:
        return line[: m.start()] + unfold(artifact, m.group("fid")) + line[m.end():]
    return line
