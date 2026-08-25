"""StageAlgorithm 抽象基类 —— 所有流程算法的共同祖先（transformers 式插拔单元）。

每个算法 = 一个模块内的一个类：
* 声明 ``stage``（所属流程）与 ``name``（算法注册名）两个 ClassVar；
* 用 ``@register`` 装饰后即可在 YAML 配置中按名组合；
* 只依赖 core/llm/io 的接口与产物（bundle），不 import 其它算法模块。

双作用域（文献原则"聚合先于单例"：agent 级 53.5% 可用 vs step 级 14.2%，
Who&When 2505.00212）：
* :meth:`run_one` —— 单轨迹作用域，必须实现；
* :meth:`run_corpus` —— 跨轨迹聚合作用域，默认逐条调用 run_one；
  跨轨迹算法（阶段三 SBFL / 失败聚类）覆写它并忽略 run_one。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:  # 避免运行时循环依赖，仅类型引用
    from atap.core.bundle import TrajectoryBundle
    from atap.core.context import RunContext

# 六环节中的五个可插拔流程（"采集"由 io 层承担，不在此列）。
STAGE_ORDER: tuple[str, ...] = (
    "represent",  # 表征：R0/R1/R5...
    "analyze",    # 分析与评测
    "classify",   # 错误分类打标
    "attribute",  # 失败归因
    "recover",    # 恢复与增强
)


class StageAlgorithm(ABC):
    """流程算法基类。子类必须设置 ``stage`` 与 ``name``。"""

    stage: ClassVar[str]
    name: ClassVar[str]

    def __init__(self, **params: Any) -> None:
        self.params: dict[str, Any] = dict(params)

    # -- 供子类读取配置参数的便捷方法 ---------------------------------------

    def param(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    # -- 双作用域 -------------------------------------------------------------

    @abstractmethod
    def run_one(self, bundle: "TrajectoryBundle", ctx: "RunContext") -> None:
        """处理单条轨迹，把产物写入 ``bundle.artifacts``。

        契约：算法不得修改 ``bundle.trajectory`` 的 outcome（检测/归因不改
        历史事实）；表征类算法负责填充 ``trajectory.events``。
        """

    def run_corpus(self, bundles: list["TrajectoryBundle"], ctx: "RunContext") -> None:
        """跨轨迹聚合作用域。默认实现 = 逐条 run_one。"""
        for bundle in bundles:
            self.run_one(bundle, ctx)

    # -- 描述 ---------------------------------------------------------------

    def describe(self) -> str:
        return f"[{self.stage}/{self.name}] {type(self).__name__}(params={self.params})"
