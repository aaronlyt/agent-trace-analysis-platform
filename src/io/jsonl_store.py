"""Local JSONL storage -- the default implementation with zero service dependencies.

* One trajectory JSON per line (the native form of Trajectory.to_dict);
* source supports glob (e.g. ``runs/traces/*.jsonl``);
* artifact store writes to ``<dir>/<trace_id>/<stage>__<name>.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from atap.core.schema import Trajectory

if TYPE_CHECKING:
    from atap.core.bundle import TrajectoryBundle


class JSONLTraceSource:
    """Read trajectories from JSONL (glob, directory, or single-file path)."""

    def __init__(self, path: str) -> None:
        self.path = path

    def _files(self) -> list[Path]:
        import glob as _glob

        if Path(self.path).is_file():
            return [Path(self.path)]
        hits = sorted(Path(p) for p in _glob.glob(self.path))
        if not hits and Path(self.path).is_dir():
            hits = sorted(Path(self.path).glob("*.jsonl"))
        if not hits:
            raise FileNotFoundError(f"trajectory files not found: {self.path}")
        return hits

    def load(self) -> list[Trajectory]:
        out: list[Trajectory] = []
        for f in self._files():
            for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(Trajectory.from_dict(json.loads(line)))
                except Exception as e:
                    raise ValueError(f"{f}:{i} failed to parse trajectory: {e}") from e
        return out


class JSONLTraceStore:
    """Append trajectories back to JSONL (closed loop: rerun trajectories
    return to the source of truth)."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, trajectory: Trajectory) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(trajectory.to_dict(), ensure_ascii=False) + "\n")


class JSONLArtifactStore:
    """Write artifacts into per-trace directories + run-level report files."""

    def __init__(self, dir: str | Path) -> None:
        self.dir = Path(dir)
        self.dir.mkdir(parents=True, exist_ok=True)

    def save_artifact(self, trace_id: str, stage: str, name: str, artifact: Any) -> None:
        d = self.dir / trace_id
        d.mkdir(parents=True, exist_ok=True)
        payload = artifact
        if hasattr(artifact, "to_dict"):
            payload = artifact.to_dict()
        path = d / f"{stage}__{name}.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )

    def save_report(self, filename: str, payload: Any) -> None:
        path = self.dir / filename
        if hasattr(payload, "to_dict"):
            payload = payload.to_dict()
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )


def build_source(spec: dict):
    kind = spec.get("type", "jsonl")
    if "path" not in spec:
        raise ValueError("source requires a path field")
    if kind == "jsonl":
        return JSONLTraceSource(spec["path"])
    if kind == "langfuse":
        from atap.io.langfuse import LangfuseTraceSource

        return LangfuseTraceSource(spec["path"])
    if kind == "otel":
        from atap.io.otel import OTelTraceSource

        return OTelTraceSource(spec["path"])
    raise ValueError(
        f"unknown source type: {kind!r} (supported: jsonl / langfuse / otel)"
    )


def build_store(spec: dict | None, run_dir: str):
    """Assemble the ArtifactStore from config (defaults to run_dir/artifacts)."""
    from atap.io.base import ArtifactStore

    if spec is None:
        return JSONLArtifactStore(Path(run_dir) / "artifacts")
    kind = spec.get("type", "jsonl")
    if kind != "jsonl":
        raise ValueError(f"unknown store type: {kind!r} (supported: jsonl)")
    return JSONLArtifactStore(spec.get("dir") or (Path(run_dir) / "artifacts"))
