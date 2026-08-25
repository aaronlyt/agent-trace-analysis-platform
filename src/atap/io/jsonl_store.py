"""JSONL 本地存储 —— 零服务依赖的默认实现。

* 每行一个轨迹 JSON（Trajectory.to_dict 的原生形态）；
* source 支持 glob（如 ``runs/traces/*.jsonl``）；
* artifact store 按 ``<dir>/<trace_id>/<stage>__<name>.json`` 落盘。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from atap.core.schema import Trajectory

if TYPE_CHECKING:
    from atap.core.bundle import TrajectoryBundle


class JSONLTraceSource:
    """从 JSONL（glob 或目录或单文件路径）读轨迹。"""

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
            raise FileNotFoundError(f"轨迹文件未找到：{self.path}")
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
                    raise ValueError(f"{f}:{i} 轨迹解析失败：{e}") from e
        return out


class JSONLTraceStore:
    """把轨迹追加写回 JSONL（闭环：重跑轨迹回到事实源）。"""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def save(self, trajectory: Trajectory) -> None:
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(trajectory.to_dict(), ensure_ascii=False) + "\n")


class JSONLArtifactStore:
    """产物按 trace 分目录落盘 + 运行级报告文件。"""

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


def build_source(spec: dict) -> JSONLTraceSource:
    kind = spec.get("type", "jsonl")
    if kind != "jsonl":
        raise ValueError(f"未知 source type：{kind!r}（当前支持：jsonl）")
    if "path" not in spec:
        raise ValueError("source 需要path 字段")
    return JSONLTraceSource(spec["path"])


def build_store(spec: dict | None, run_dir: str):
    """按配置装配 ArtifactStore（默认在 run_dir/artifacts 下）。"""
    from atap.io.base import ArtifactStore

    if spec is None:
        return JSONLArtifactStore(Path(run_dir) / "artifacts")
    kind = spec.get("type", "jsonl")
    if kind != "jsonl":
        raise ValueError(f"未知 store type：{kind!r}（当前支持：jsonl）")
    return JSONLArtifactStore(spec.get("dir") or (Path(run_dir) / "artifacts"))
