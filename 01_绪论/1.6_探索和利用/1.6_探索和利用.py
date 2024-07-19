# 1.6_探索和利用

"""
Lecture: 01_绪论/1.6_探索和利用
Content: 1.6_探索和利用
"""

import numpy as np
from typing import List, Tuple

class ExplorationStrategy:
    """基础探索策略类"""

    def select_action(self, q_values: np.ndarray) -> int:
        """
        根据Q值选择动作

        :param q_values: 动作的Q值
        :return: 选择的动作
        """
        raise NotImplementedError("需要在子类中实现")

class RandomExploration(ExplorationStrategy):
    """随机探索策略类"""

    def select_action(self, q_values: np.ndarray) -> int:
        """随机选择动作"""
        return np.random.choice(len(q_values))

class EpsilonGreedy(ExplorationStrategy):
    """epsilon-贪心策略类"""

    def __init__(self, epsilon: float):
        """
        初始化epsilon-贪心策略

        :param epsilon: 随机选择动作的概率
        """
        self.epsilon = epsilon

    def select_action(self, q_values: np.ndarray) -> int:
        """根据epsilon-贪心策略选择动作"""
        if np.random.rand() < self.epsilon:
            return np.random.choice(len(q_values))
        return np.argmax(q_values)

class SoftPolicy(ExplorationStrategy):
    """软策略类"""

    def select_action(self, q_values: np.ndarray) -> int:
        """根据软策略选择动作"""
        probabilities = np.exp(q_values) / np.sum(np.exp(q_values))
        return np.random.choice(len(q_values), p=probabilities)

class UpperConfidenceBound(ExplorationStrategy):
    """上置信界策略类"""

    def __init__(self, c: float, action_counts: np.ndarray):
        """
        初始化上置信界策略

        :param c: 调节参数，控制探索和利用的平衡
        :param action_counts: 每个动作被选择的次数
        """
        self.c = c
        self.action_counts = action_counts
        self.total_counts = np.sum(action_counts)

    def select_action(self, q_values: np.ndarray) -> int:
        """根据上置信界策略选择动作"""
        ucb_values = q_values + self.c * np.sqrt(np.log(self.total_counts + 1) / (self.action_counts + 1e-5))
        return np.argmax(ucb_values)

# 测试探索策略类
if __name__ == "__main__":
    q_values = np.array([1.0, 2.0, 1.5, 1.8])
    action_counts = np.array([10, 5, 15, 20])
    
    # 随机探索
    random_exploration = RandomExploration()
    print(f"随机探索选择的动作: {random_exploration.select_action(q_values)}")
    
    # epsilon-贪心策略
    epsilon_greedy = EpsilonGreedy(epsilon=0.1)
    print(f"epsilon-贪心策略选择的动作: {epsilon_greedy.select_action(q_values)}")
    
    # 软策略
    soft_policy = SoftPolicy()
    print(f"软策略选择的动作: {soft_policy.select_action(q_values)}")
    
    # 上置信界策略
    ucb = UpperConfidenceBound(c=2, action_counts=action_counts)
    print(f"上置信界策略选择的动作: {ucb.select_action(q_values)}")
