"""atap 命令行入口。

用法::

    atap run --config configs/pipeline_offline.yaml [--out runs/demo]
    atap list                       # 列出已注册算法
    atap demo [--seed 42] [--out runs/demo]   # 阶段二：离线全链路演示
"""

from __future__ import annotations

import argparse
import sys


def _cmd_list(_args: argparse.Namespace) -> int:
    import atap  # 触发注册引导

    grouped = atap.list_algorithms()
    for stage, names in grouped.items():
        print(f"{stage}: {', '.join(names)}")
    return 0


def _cmd_run(args: argparse.Namespace) -> int:
    from atap.core.config import load_config
    from atap.runtime import run_config

    cfg = load_config(args.config)
    _, reports = run_config(cfg, args.out)
    print(f"run={cfg.run_name} rounds={len(reports)}")
    for i, r in enumerate(reports):
        print(
            f"  round{i}: traces={r.n_traces} failures={r.n_failures} "
            f"attributed={r.n_attributed} reruns={r.n_reruns}(ok={r.n_rerun_success})"
        )
    print(f"artifacts -> {args.out}/artifacts")
    return 0


def _cmd_demo(args: argparse.Namespace) -> int:
    from atap.demo import run_demo

    run_demo(seed=args.seed, out=args.out)
    return 0


def _cmd_compare(args: argparse.Namespace) -> int:
    from atap.compare import run_compare

    run_compare(args.config, args.out, traces=args.traces)
    return 0


def _cmd_corpus(args: argparse.Namespace) -> int:
    import json
    from pathlib import Path

    from atap.sandbox import ToySandbox

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    traces = ToySandbox().generate_corpus(
        successes_per_task=args.successes_per_task
    )
    with out.open("w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
    print(f"corpus: {len(traces)} traces -> {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="atap", description="Agent 轨迹分析与错误归因平台")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="按配置运行 pipeline")
    p_run.add_argument("--config", required=True)
    p_run.add_argument("--out", default="runs/run")

    p_list = sub.add_parser("list", help="列出已注册算法")

    p_demo = sub.add_parser("demo", help="离线全链路演示（FakeLLM 确定性判官）")
    p_demo.add_argument("--seed", type=int, default=7)
    p_demo.add_argument("--out", default="runs/demo")

    p_cmp = sub.add_parser("compare", help="同一轨迹集上对比多组算法配置")
    p_cmp.add_argument("--config", action="append", required=True,
                       help="配置文件路径，可多次给出")
    p_cmp.add_argument("--traces", default=None,
                       help="覆盖所有配置的轨迹源（保证同一轨迹集）")
    p_cmp.add_argument("--out", default="runs/compare")

    p_corpus = sub.add_parser("corpus", help="生成 SBFL 频谱语料（每任务成败对照）")
    p_corpus.add_argument("--out", default="runs/corpus/traces.jsonl")
    p_corpus.add_argument("--successes-per-task", type=int, default=2)

    args = parser.parse_args(argv)
    handlers = {
        "run": _cmd_run,
        "list": _cmd_list,
        "demo": _cmd_demo,
        "compare": _cmd_compare,
        "corpus": _cmd_corpus,
    }
    try:
        return handlers[args.command](args)
    except Exception as e:  # CLI 边界：显式报错退出
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
