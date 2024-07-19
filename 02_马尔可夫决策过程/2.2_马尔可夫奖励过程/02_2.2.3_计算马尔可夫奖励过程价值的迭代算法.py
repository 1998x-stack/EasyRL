# 02_2.2.3_计算马尔可夫奖励过程价值的迭代算法

"""
Lecture: 02_马尔可夫决策过程/2.2_马尔可夫奖励过程
Content: 02_2.2.3_计算马尔可夫奖励过程价值的迭代算法
"""

from typing import List, Dict, Tuple
import numpy as np

class MarkovRewardProcess:
    """
    Markov Reward Process (MRP) class for handling states, rewards, transition probabilities, and value functions.
    
    Attributes:
        states (List[str]): List of state names.
        transition_matrix (np.ndarray): State transition probability matrix.
        rewards (Dict[str, float]): Rewards for each state.
        gamma (float): Discount factor.
    """
    
    def __init__(self, states: List[str], transition_matrix: np.ndarray, rewards: Dict[str, float], gamma: float):
        """
        初始化 MarkovRewardProcess 实例。
        
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

    def dynamic_programming(self, epsilon: float = 1e-6) -> Dict[str, float]:
        """
        使用动态规划方法计算状态的价值函数。
        
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
    
    def monte_carlo(self, episodes: int = 1000) -> Dict[str, float]:
        """
        使用蒙特卡洛方法计算状态的价值函数。
        
        Args:
            episodes (int): 模拟的轨迹数量。
        
        Returns:
            Dict[str, float]: 各状态的价值函数。
        """
        returns = {state: [] for state in self.states}
        V = {state: 0 for state in self.states}

        for _ in range(episodes):
            state = np.random.choice(self.states)
            episode = []
            while state in self.rewards:
                next_state = np.random.choice(self.states, p=self.transition_matrix[self.state_index[state]])
                reward = self.rewards[state]
                episode.append((state, reward))
                state = next_state

            G = 0
            for state, reward in reversed(episode):
                G = reward + self.gamma * G
                returns[state].append(G)

        for state in self.states:
            if returns[state]:
                V[state] = np.mean(returns[state])

        return V

    def temporal_difference(self, alpha: float = 0.1, episodes: int = 1000) -> Dict[str, float]:
        """
        使用时序差分方法计算状态的价值函数。
        
        Args:
            alpha (float): 学习率。
            episodes (int): 模拟的轨迹数量。
        
        Returns:
            Dict[str, float]: 各状态的价值函数。
        """
        V = {state: 0 for state in self.states}

        for _ in range(episodes):
            state = np.random.choice(self.states)
            while state in self.rewards:
                next_state = np.random.choice(self.states, p=self.transition_matrix[self.state_index[state]])
                reward = self.rewards[state]
                V[state] += alpha * (reward + self.gamma * V[next_state] - V[state])
                state = next_state

        return V
    
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
    
    # 动态规划方法计算状态价值函数
    dp_value_function = mrp.dynamic_programming()
    print("Dynamic Programming Value Function:")
    mrp.print_value_function(dp_value_function)
    
    # 蒙特卡洛方法计算状态价值函数
    mc_value_function = mrp.monte_carlo()
    print("\nMonte Carlo Value Function:")
    mrp.print_value_function(mc_value_function)
    
    # 时序差分方法计算状态价值函数
    td_value_function = mrp.temporal_difference()
    print("\nTemporal Difference Value Function:")
    mrp.print_value_function(td_value_function)
