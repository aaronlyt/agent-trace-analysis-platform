"""Internal export helpers shared by the collection adapters (io-private)."""

from __future__ import annotations

from typing import Any

#: ground-truth trace-meta keys that must never leave through an export
#: (leak prevention: the injected fault is evaluation-side information).
#: ``origin_fault`` is the full GT copy sandbox reruns carry (chain
#: re-attribution), ``fault_removed`` the environment's construction-side
#: removal signal -- both are evaluation machinery, not trace content.
_GT_META_KEYS = ("injected_fault", "origin_fault", "fault_removed")


def export_safe_meta(meta: dict[str, Any]) -> dict[str, Any]:
    """Trace meta without ground-truth keys (both otel and langfuse exports
    strip through here so the deny-list stays in one place)."""
    return {k: v for k, v in meta.items() if k not in _GT_META_KEYS}
