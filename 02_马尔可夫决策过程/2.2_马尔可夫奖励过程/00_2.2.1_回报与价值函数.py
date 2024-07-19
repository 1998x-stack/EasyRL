# 00_2.2.1_回报与价值函数

"""
Lecture: 02_马尔可夫决策过程/2.2_马尔可夫奖励过程
Content: 00_2.2.1_回报与价值函数
"""

from typing import List, Dict, Tuple
import numpy as np

class MarkovRewardProcess:
    """
    Markov Reward Process (MRP) class for handling states, rewards, and value functions.
    
    Attributes:
        states (List[str]): List of state names.
        transition_matrix (np.ndarray): State transition probability matrix.
        rewards (Dict[str, float]): Rewards for each state.
        gamma (float): Discount factor.
    """
    
    def __init__(self, states: List[str], transition_matrix: np.ndarray, rewards: Dict[str, float], gamma: float):
        """
        初始化 MRP 实例。
        
        Args:
            states (List[str]): 状态名称列表。
            transition_matrix (np.ndarray): 状态转移概率矩阵。
            rewards (Dict[str, float]): 每个状态的奖励。
            gamma (float): 折扣因子。
        """
        assert len(states) == transition_matrix.shape[0] == transition_matrix.shape[1], "状态和转移矩阵大小不匹配"
        assert 0 <= gamma <= 1, "折扣因子必须在 0 到 1 之间"

        self.states = states
        self.state_index = {state: i for i, state in enumerate(states)}
        self.transition_matrix = transition_matrix
        self.rewards = rewards
        self.gamma = gamma
    
    def get_return(self, state_sequence: List[str]) -> float:
        """
        计算给定状态序列的回报（累积奖励）。
        
        Args:
            state_sequence (List[str]): 状态序列。
        
        Returns:
            float: 累积回报值。
        """
        total_return = 0.0
        for t, state in enumerate(state_sequence):
            if state in self.rewards:
                total_return += (self.gamma ** t) * self.rewards[state]
        return total_return

    def compute_value_function(self, epsilon: float = 1e-6) -> Dict[str, float]:
        """
        使用贝尔曼方程计算状态的价值函数。
        
        Args:
            epsilon (float): 收敛条件。
        
        Returns:
            Dict[str, float]: 各状态的价值函数。
        """
        num_states = len(self.states)
        V = np.zeros(num_states)
        delta = float('inf')

        while delta > epsilon:
            delta = 0
            for i, state in enumerate(self.states):
                v = V[i]
                V[i] = self.rewards[state] + self.gamma * np.sum(self.transition_matrix[i, :] * V)
                delta = max(delta, abs(v - V[i]))

        return {state: V[i] for i, state in enumerate(self.states)}
    
    def print_value_function(self, value_function: Dict[str, float]) -> None:
        """
        打印状态价值函数。
        
        Args:
            value_function (Dict[str, float]): 各状态的价值函数。
        """
        for state, value in value_function.items():
            print(f"State: {state}, Value: {value:.4f}")

# 测试示例
if __name__ == "__main__":
    states = ["A", "B", "C"]
    transition_matrix = np.array([[0.5, 0.3, 0.2],
                                  [0.2, 0.6, 0.2],
                                  [0.1, 0.1, 0.8]])
    rewards = {"A": 1.0, "B": 0.5, "C": 0.0}
    gamma = 0.9

    mrp = MarkovRewardProcess(states, transition_matrix, rewards, gamma)
    
    # 计算状态价值函数
    value_function = mrp.compute_value_function()
    
    # 打印状态价值函数
    mrp.print_value_function(value_function)
    
    # 计算状态序列的回报
    state_sequence = ["A", "B", "C", "A", "C"]
    total_return = mrp.get_return(state_sequence)
    print(f"Total Return for the sequence {state_sequence}: {total_return:.4f}")