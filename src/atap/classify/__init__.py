"""classify 包：错误分类打标算法 + 共享词表。导入即注册。"""

from atap.classify.base import Classifier
from atap.classify.mast_judge import MastJudgeClassifier, MastLabel, MastLabels
from atap.classify.rule_pack import RulePackClassifier

__all__ = ["Classifier", "MastJudgeClassifier", "MastLabel", "MastLabels", "RulePackClassifier"]
