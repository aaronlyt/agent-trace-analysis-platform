"""Internal export helpers shared by the collection adapters (io-private)."""

from __future__ import annotations

from typing import Any

#: ground-truth trace-meta keys that must never leave through an export
#: (leak prevention: the injected fault is evaluation-side information).
#: ``origin_fault`` is the full GT copy sandbox reruns carry (chain
#: re-attribution), ``fault_removed`` the environment's construction-side
#: removal signal -- both are evaluation machinery, not trace content.
_GT_META_KEYS = ("injected_fault", "origin_fault", "fault_removed")

#: keys that additionally count as ground truth once the export leaves the
#: local machine: ``qrels`` (gold sufficient sets) is rg_ug's data
#: dependency and legitimately survives an offline roundtrip file, but it
#: must never reach an external server [fix: live pushes shipped qrels to
#: Langfuse trace metadata, caught from a demo screenshot].
_EXTERNAL_GT_META_KEYS = _GT_META_KEYS + ("qrels",)


def export_safe_meta(meta: dict[str, Any], *, external: bool = False) -> dict[str, Any]:
    """Trace meta without ground-truth keys (both otel and langfuse exports
    strip through here so the deny-list stays in one place). ``external``
    additionally drops keys whose GT status only matters off-box (qrels)."""
    drop = _EXTERNAL_GT_META_KEYS if external else _GT_META_KEYS
    return {k: v for k, v in meta.items() if k not in drop}
