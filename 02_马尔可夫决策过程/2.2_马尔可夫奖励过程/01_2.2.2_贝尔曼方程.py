# 01_2.2.2_贝尔曼方程

"""
Lecture: 02_马尔可夫决策过程/2.2_马尔可夫奖励过程
Content: 01_2.2.2_贝尔曼方程
"""

from typing import List, Dict
import numpy as np

class BellmanEquationSolver:
    """
    Bellman Equation Solver class for handling states, rewards, transition probabilities, and value functions.
    
    Attributes:
        states (List[str]): List of state names.
        actions (List[str]): List of action names.
        transition_matrix (Dict[str, np.ndarray]): State transition probability matrices for each action.
        rewards (Dict[str, Dict[str, float]]): Rewards for each state-action pair.
        gamma (float): Discount factor.
    """
    
    def __init__(self, states: List[str], actions: List[str], transition_matrix: Dict[str, np.ndarray], rewards: Dict[str, Dict[str, float]], gamma: float):
        """
        初始化 BellmanEquationSolver 实例。
        
        Args:
            states (List[str]): 状态名称列表。
            actions (List[str]): 动作名称列表。
            transition_matrix (Dict[str, np.ndarray]): 各动作对应的状态转移概率矩阵。
            rewards (Dict[str, Dict[str, float]]): 每个状态-动作对的奖励。
            gamma (float): 折扣因子。
        """
        assert all(len(states) == tm.shape[0] == tm.shape[1] for tm in transition_matrix.values()), "状态和转移矩阵大小不匹配"
        assert 0 <= gamma <= 1, "折扣因子必须在 0 到 1 之间"

        self.states = states
        self.actions = actions
        self.state_index = {state: i for i, state in enumerate(states)}
        self.transition_matrix = transition_matrix
        self.rewards = rewards
        self.gamma = gamma
    
    def compute_state_value_function(self, epsilon: float = 1e-6) -> Dict[str, float]:
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
                V[i] = max(sum(self.transition_matrix[action][i, j] * (self.rewards[state][action] + self.gamma * V[j]) for j, next_state in enumerate(self.states)) for action in self.actions)
                delta = max(delta, abs(v - V[i]))

        return {state: V[i] for i, state in enumerate(self.states)}
    
    def compute_action_value_function(self) -> Dict[str, Dict[str, float]]:
        """
        使用贝尔曼方程计算状态-动作对的价值函数。
        
        Returns:
            Dict[str, Dict[str, float]]: 各状态-动作对的价值函数。
        """
        num_states = len(self.states)
        Q = {state: {action: 0 for action in self.actions} for state in self.states}
        
        while True:
            delta = 0
            for i, state in enumerate(self.states):
                for action in self.actions:
                    q = Q[state][action]
                    Q[state][action] = sum(self.transition_matrix[action][i, j] * (self.rewards[state][action] + self.gamma * max(Q[next_state].values())) for j, next_state in enumerate(self.states))
                    delta = max(delta, abs(q - Q[state][action]))
            if delta < epsilon:
                break
        
        return Q
    
    def print_value_function(self, value_function: Dict[str, float]) -> None:
        """
        打印状态价值函数。
        
        Args:
            value_function (Dict[str, float]): 各状态的价值函数。
        """
        for state, value in value_function.items():
            print(f"State: {state}, Value: {value:.4f}")

    def print_action_value_function(self, action_value_function: Dict[str, Dict[str, float]]) -> None:
        """
        打印状态-动作对的价值函数。
        
        Args:
            action_value_function (Dict[str, Dict[str, float]]): 各状态-动作对的价值函数。
        """
        for state, actions in action_value_function.items():
            for action, value in actions.items():
                print(f"State: {state}, Action: {action}, Value: {value:.4f}")

# 测试示例
if __name__ == "__main__":
    states = ["A", "B", "C"]
    actions = ["X", "Y"]
    transition_matrix = {
        "X": np.array([[0.7, 0.2, 0.1],
                       [0.1, 0.8, 0.1],
                       [0.2, 0.3, 0.5]]),
        "Y": np.array([[0.6, 0.3, 0.1],
                       [0.4, 0.4, 0.2],
                       [0.3, 0.2, 0.5]])
    }
    rewards = {
        "A": {"X": 5, "Y": 10},
        "B": {"X": 2, "Y": 4},
        "C": {"X": 1, "Y": 2}
    }
    gamma = 0.9

    solver = BellmanEquationSolver(states, actions, transition_matrix, rewards, gamma)
    
    # 计算状态价值函数
    state_value_function = solver.compute_state_value_function()
    
    # 打印状态价值函数
    solver.print_value_function(state_value_function)
    
    # 计算状态-动作对的价值函数
    action_value_function = solver.compute_action_value_function()
    
    # 打印状态-动作对的价值函数
    solver.print_action_value_function(action_value_function)