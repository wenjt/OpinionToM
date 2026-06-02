import numpy as np
from typing import List, Optional, Dict

class HypothesisV3:
    """单个假设类：封装假设文本、权重、父假设"""
    def __init__(self, text: str, weight: float, parent: Optional['HypothesisV3'] = None):
        self.hypothesis = text  # 假设
        self.weight = weight  # 假设权重
        self.parent = parent  # 父假设

    def __repr__(self) -> str:
        return f"Hypothesis: {self.hypothesis}... (Weight: {self.weight:.2f})"

class HypothesesSet:
    def __init__(
            self,
            target: str,
            contexts: Dict, # 仅保留当前上下文（不含历史解析）
            hypotheses: List[str],
            weights: np.ndarray,
            thought_state: str,
            parse: str,
            parent_hypotheses: Optional[List[HypothesisV3]] = None,
            weight_details: Optional[Dict] = None
    ):
        self.target = target  # 目标议题
        self.contexts = contexts  # 当前上下文（仅含tweet和当前Agent解析）
        self.hypotheses = hypotheses  # 假设文本列表
        self.weights = weights  # 假设权重数组
        self.thought_state = thought_state  # 当前思考状态S（仅当前Agent）
        self.parse = parse  # 当前Agent的解析结果（仅当前Agent）
        self.weight_details = weight_details  # 权重更新细节
        self.parent_hypotheses = parent_hypotheses or []  # 仅保留上一时刻假设（父假设）

        # 创建HypothesisV3实例列表（仅关联上一时刻父假设）
        self.hypotheses = [
            HypothesisV3(text=text, weight=weight, parent=parent)
            for text, weight, parent in
            zip(hypotheses, weights, self.parent_hypotheses + [None] * (len(hypotheses) - len(self.parent_hypotheses)))
        ]

    def update_weights(self, new_weights: np.ndarray):
        """更新假设权重"""
        self.weights = new_weights
        for hyp, w in zip(self.hypotheses, new_weights):
            hyp.weight = w

