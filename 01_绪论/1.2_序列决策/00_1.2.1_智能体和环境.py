# 00_1.2.1_智能体和环境

"""
Lecture: 01_绪论/1.2_序列决策
Content: 00_1.2.1_智能体和环境
"""

import numpy as np

class CartPole:
    """
    CartPole 环境模拟类

    该类模拟了经典的 CartPole 问题，智能体需要通过控制小车移动来保持杆子的平衡。
    """
    
    def __init__(self):
        """
        初始化 CartPole 环境的参数和状态
        """
        self.gravity = 9.8  # 重力加速度
        self.cart_mass = 1.0  # 小车质量
        self.pole_mass = 0.1  # 杆子质量
        self.total_mass = self.cart_mass + self.pole_mass  # 总质量
        self.pole_length = 0.5  # 杆子长度
        self.pole_mass_length = self.pole_mass * self.pole_length  # 杆子的质量长度
        self.force_mag = 10.0  # 施加力的大小
        self.tau = 0.02  # 时间步长
        self.theta_threshold = 12 * 2 * np.pi / 360  # 杆子角度阈值（弧度）
        self.x_threshold = 2.4  # 小车位置阈值
        self.state = None  # 环境的状态
        self.reset()

    def reset(self):
        """
        重置环境状态
        
        :return: 返回重置后的状态
        """
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        return self.state

    def step(self, action: int):
        """
        执行动作并更新环境状态
        
        :param action: 智能体选择的动作（0 或 1）
        :return: 返回新的状态、奖励、是否结束标志和额外信息
        """
        assert action in [0, 1], "Action must be 0 or 1"
        
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        
        # 动力学方程
        costheta = np.cos(theta)
        sintheta = np.sin(theta)
        
        temp = (force + self.pole_mass_length * theta_dot ** 2 * sintheta) / self.total_mass
        theta_acc = (self.gravity * sintheta - costheta * temp) / \
                    (self.pole_length * (4.0 / 3.0 - self.pole_mass * costheta ** 2 / self.total_mass))
        x_acc = temp - self.pole_mass_length * theta_acc * costheta / self.total_mass
        
        # 更新状态
        x = x + self.tau * x_dot
        x_dot = x_dot + self.tau * x_acc
        theta = theta + self.tau * theta_dot
        theta_dot = theta_dot + self.tau * theta_acc
        
        self.state = (x, x_dot, theta, theta_dot)
        
        done = x < -self.x_threshold or x > self.x_threshold or theta < -self.theta_threshold or theta > self.theta_threshold
        done = bool(done)
        
        reward = 1.0 if not done else 0.0
        
        return np.array(self.state), reward, done, {}

    def render(self):
        """
        渲染环境（可以在此处添加具体的渲染代码）
        """
        pass

    def close(self):
        """
        关闭环境（可以在此处添加具体的清理代码）
        """
        pass

# 测试 CartPole 类的功能
if __name__ == "__main__":
    env = CartPole()
    state = env.reset()
    print("初始状态:", state)
    
    for _ in range(1000):
        action = np.random.choice([0, 1])
        state, reward, done, _ = env.step(action)
        print(f"状态: {state}, 奖励: {reward}, 结束标志: {done}")
        
        if done:
            print("环境已终止，重新开始")
            state = env.reset()