# 00_1.4.1_策略

"""
Lecture: 01_绪论/1.4_强化学习智能体的组成成分和类型
Content: 00_1.4.1_策略
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
from typing import List, Tuple

class PolicyNetwork(nn.Module):
    """
    策略网络类，用于输出给定状态下的动作概率分布
    """
    def __init__(self, state_dim: int, action_dim: int):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x

class REINFORCE:
    """
    REINFORCE算法类，用于训练策略网络以最大化累积回报
    """
    def __init__(self, policy_network: PolicyNetwork, lr: float = 1e-2):
        self.policy_network = policy_network
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        
    def select_action(self, state: np.ndarray) -> Tuple[int, torch.Tensor]:
        """
        选择动作
        :param state: 当前状态
        :return: 动作和动作的概率
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        probs = self.policy_network(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item(), action_dist.log_prob(action)

    def update_policy(self, rewards: List[float], log_probs: List[torch.Tensor], gamma: float = 0.99):
        """
        更新策略网络
        :param rewards: 回报列表
        :param log_probs: 动作对数概率列表
        :param gamma: 折扣因子
        """
        discounted_rewards = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            discounted_rewards.insert(0, R)
        
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)
        
        policy_loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        policy_loss = torch.cat(policy_loss).sum()
        
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

def train_reinforce(env_name: str, num_episodes: int, lr: float = 1e-2, gamma: float = 0.99):
    """
    使用REINFORCE算法训练策略网络
    :param env_name: Gym环境名称
    :param num_episodes: 训练的回合数
    :param lr: 学习率
    :param gamma: 折扣因子
    """
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy_network = PolicyNetwork(state_dim, action_dim)
    reinforce = REINFORCE(policy_network, lr)
    
    for episode in range(num_episodes):
        state = env.reset()
        rewards = []
        log_probs = []
        
        while True:
            action, log_prob = reinforce.select_action(state)
            next_state, reward, done, _ = env.step(action)
            
            rewards.append(reward)
            log_probs.append(log_prob)
            
            if done:
                break
            state = next_state
        
        reinforce.update_policy(rewards, log_probs, gamma)
        print(f"Episode {episode + 1}/{num_episodes} finished with reward: {sum(rewards)}")

if __name__ == "__main__":
    train_reinforce(env_name="CartPole-v1", num_episodes=1000, lr=1e-2, gamma=0.99)