# 00_1.7.1_Gym

"""
Lecture: 01_绪论/1.7_强化学习实验
Content: 00_1.7.1_Gym
"""

import gym
import numpy as np
from typing import Tuple

class GymEnvWrapper:
    """
    Gym 环境包装类，用于强化学习实验

    该类封装了 OpenAI Gym 环境，提供了环境重置、步进、渲染等功能，并记录和输出重要信息。
    """

    def __init__(self, env_name: str):
        """
        初始化 Gym 环境包装类

        :param env_name: Gym 环境名称
        """
        self.env = gym.make(env_name)
        self.state = None
        self.done = False
        self.info = None

    def reset(self) -> np.ndarray:
        """
        重置环境

        :return: 重置后的初始状态
        """
        self.state = self.env.reset()
        self.done = False
        self.info = None
        return self.state

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, dict]:
        """
        执行动作并更新环境状态

        :param action: 智能体选择的动作
        :return: 新状态、奖励、是否结束标志和额外信息
        """
        self.state, reward, self.done, self.info = self.env.step(action)
        return self.state, reward, self.done, self.info

    def render(self) -> None:
        """
        渲染环境
        """
        self.env.render()

    def close(self) -> None:
        """
        关闭环境
        """
        self.env.close()

# 测试 GymEnvWrapper 类的功能
def test_gym_env_wrapper():
    env_wrapper = GymEnvWrapper(env_name="CartPole-v1")
    state = env_wrapper.reset()
    print("初始状态:", state)

    for _ in range(1000):
        action = env_wrapper.env.action_space.sample()  # 随机选择动作
        state, reward, done, info = env_wrapper.step(action)
        env_wrapper.render()
        print(f"状态: {state}, 奖励: {reward}, 结束标志: {done}, 额外信息: {info}")

        if done:
            print("环境已终止，重新开始")
            state = env_wrapper.reset()

    env_wrapper.close()

import gym
import numpy as np
from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor

class VectorGymEnvWrapper:
    """
    Vector Gym 环境包装类，用于并行运行多个 Gym 环境

    该类封装了多个 OpenAI Gym 环境，提供了环境重置、步进、渲染等功能，并记录和输出重要信息。
    """

    def __init__(self, env_name: str, num_envs: int):
        """
        初始化 Vector Gym 环境包装类

        :param env_name: Gym 环境名称
        :param num_envs: 并行环境的数量
        """
        self.env_name = env_name
        self.num_envs = num_envs
        self.envs = [gym.make(env_name) for _ in range(num_envs)]
        self.states = None
        self.dones = None
        self.infos = None
        self.executor = ThreadPoolExecutor(max_workers=num_envs)

    def reset(self) -> List[np.ndarray]:
        """
        重置所有环境

        :return: 所有环境重置后的初始状态列表
        """
        self.states = list(self.executor.map(lambda env: env.reset(), self.envs))
        self.dones = [False] * self.num_envs
        self.infos = [None] * self.num_envs
        return self.states

    def step(self, actions: List[int]) -> Tuple[List[np.ndarray], List[float], List[bool], List[dict]]:
        """
        并行执行多个环境的动作并更新状态

        :param actions: 智能体选择的动作列表
        :return: 新状态列表、奖励列表、是否结束标志列表和额外信息列表
        """
        results = list(self.executor.map(lambda p: p[0].step(p[1]), zip(self.envs, actions)))
        self.states, rewards, self.dones, self.infos = zip(*results)
        return list(self.states), list(rewards), list(self.dones), list(self.infos)

    def render(self) -> None:
        """
        渲染所有环境
        """
        for env in self.envs:
            env.render()

    def close(self) -> None:
        """
        关闭所有环境
        """
        for env in self.envs:
            env.close()
        self.executor.shutdown()

# 测试 VectorGymEnvWrapper 类的功能
if __name__ == "__main__":
    num_envs = 4
    env_wrapper = VectorGymEnvWrapper(env_name="CartPole-v1", num_envs=num_envs)
    states = env_wrapper.reset()
    print("初始状态:", states)

    for _ in range(100):
        actions = [env_wrapper.envs[i].action_space.sample() for i in range(num_envs)]  # 随机选择动作
        states, rewards, dones, infos = env_wrapper.step(actions)
        env_wrapper.render()
        print(f"状态: {states}, 奖励: {rewards}, 结束标志: {dones}, 额外信息: {infos}")

        if any(dones):
            print("有环境已终止，重新开始")
            states = env_wrapper.reset()

    env_wrapper.close()
