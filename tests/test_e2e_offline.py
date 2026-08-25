"""阶段二验收：离线全链路 e2e（FakeLLM 确定性判官）。

对六种注入故障逐一断言：归因 step/agent/MAST 与 ground truth 对齐、
定向重跑恢复成功、闭环验证改善——完整复现文献流程的可复现缩影。
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from atap.core.config import config_from_dict
from atap.runtime import run_config
from atap.sandbox import ToySandbox
from atap.sandbox.faults import FAULTS


def offline_cfg(traces_path: str, closed_loop: bool = True) -> dict:
    return {
        "run_name": "e2e-offline",
        "seed": 7,
        "source": {"type": "jsonl", "path": traces_path},
        "llm": {"type": "fake"},
        "sandbox": {"type": "toy"},
        "closed_loop": closed_loop,
        "stages": {
            "represent": ["canonical_events", "ssf"],
            "analyze": ["judge_eval"],
            "classify": ["mast_judge"],
            "attribute": ["all_at_once"],
            "recover": [{"name": "targeted_rerun", "params": {"max_rounds": 5}}],
        },
    }


@pytest.fixture(scope="module")
def e2e(tmp_path_factory):
    out = tmp_path_factory.mktemp("e2e")
    traces = ToySandbox().generate_population(seed=7)
    path = out / "traces.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for t in traces:
            f.write(json.dumps(t.to_dict(), ensure_ascii=False) + "\n")
    bundles, reports = run_config(config_from_dict(offline_cfg(str(path))), out)
    return bundles, reports, out


def test_population_shape(e2e):
    bundles, _, _ = e2e
    faults = [b for b in bundles if not b.succeeded]
    assert len(bundles) == 7 and len(faults) == 6
    assert {b.trajectory.meta["injected_fault"]["kind"] for b in faults} == set(FAULTS)


@pytest.mark.parametrize("kind", list(FAULTS))
def test_attribution_matches_ground_truth(e2e, kind):
    bundles, _, _ = e2e
    b = next(x for x in bundles if x.trajectory.meta.get("injected_fault", {}).get("kind") == kind)
    gt = b.trajectory.meta["injected_fault"]
    hyps = b.hypotheses()
    assert hyps, f"{kind}: 无归因输出"
    top = max(hyps, key=lambda h: h.confidence)
    assert top.step == gt["step"], f"{kind}: 归因 step {top.step} != gt {gt['step']}"
    assert top.agent == gt["agent"]
    assert top.root_cause_code == gt["mast_code"]


@pytest.mark.parametrize("kind", list(FAULTS))
def test_classify_and_recover(e2e, kind):
    bundles, _, _ = e2e
    b = next(x for x in bundles if x.trajectory.meta.get("injected_fault", {}).get("kind") == kind)
    gt = b.trajectory.meta["injected_fault"]
    labels = b.get("classify", "mast_judge")["labels"]
    assert labels and labels[0]["code"] == gt["mast_code"]

    rec = b.get("recover", "targeted_rerun")
    assert rec["recovered"] is True and rec["rounds"] == 1
    assert b.reruns[-1].outcome.success

    loop = b.get("recover", "closed_loop")
    assert loop["verified_improved"] is True


def test_success_traces_scored_high_not_attributed(e2e):
    bundles, _, _ = e2e
    for b in bundles:
        if b.succeeded:
            assert b.get("analyze", "judge_eval")["score"] >= 8
            assert not b.hypotheses()
            assert b.get("classify", "mast_judge")["labels"] == []


def test_closed_loop_second_round_all_pass(e2e):
    _, reports, _ = e2e
    assert len(reports) == 2  # 初始轮 + 闭环验证轮
    assert reports[0].n_failures == 6
    assert reports[1].n_failures == 0  # 重跑轨迹全部通过全流程验证
    assert reports[1].n_attributed == 0


def test_artifacts_persisted(e2e):
    _, _, out = e2e
    arts = out / "artifacts"
    assert (arts / "report.json").exists()
    files = list(arts.rglob("*.json"))
    # 每条轨迹至少有 5 个阶段产物 × 7 条 + report
    assert len(files) >= 7 * 5 + 1
    report = json.loads((arts / "report.json").read_text(encoding="utf-8"))
    assert report["n_reruns"] == 6 and report["n_rerun_success"] == 6
