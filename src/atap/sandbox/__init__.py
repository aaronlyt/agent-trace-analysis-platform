"""sandbox —— 玩具研究问答沙盒（env + 脚本策略 + 故障注入 + 定向重放）。

只依赖 atap.core（分层不变量）；被 runtime/cli/demo 装配，算法模块
通过 RunContext.env（ReplayEnvironment 协议）访问重放能力。
"""

from atap.sandbox.policy import ToySandbox

__all__ = ["ToySandbox"]
