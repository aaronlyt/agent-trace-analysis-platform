"""classify —— 错误分类打标层（文献 §5）。

分类体系服务：融合标签结构（交互=MAST × 模块=AgentError × 系统级=SysTax
× 责任侧=Model or Harness）。taxonomy.py 是共享词表（非算法模块），
打标算法（judge / 规则包）输出统一挂在 artifacts["classify"]。
"""

from __future__ import annotations

from atap.core.base import StageAlgorithm


class Classifier(StageAlgorithm):
    """分类算法基类。产物契约：写标签列表到 artifacts["classify"]。"""

    stage = "classify"
