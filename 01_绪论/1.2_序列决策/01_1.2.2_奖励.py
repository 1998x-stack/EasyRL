# 01_1.2.2_奖励

"""
Lecture: 01_绪论/1.2_序列决策
Content: 01_1.2.2_奖励
"""
import numpy as np

class StockTradingReward:
    """
    股票交易奖励类，用于计算交易智能体的奖励
    """
    
    def __init__(self, risk_free_rate=0.01, transaction_cost=0.001):
        """
        初始化奖励类参数
        :param risk_free_rate: 无风险收益率
        :param transaction_cost: 交易成本
        """
        self.risk_free_rate = risk_free_rate
        self.transaction_cost = transaction_cost
    
    def calculate_reward(self, price: np.ndarray, action: np.ndarray, volatility: np.ndarray):
        """
        计算奖励函数
        :param price: 股票价格序列
        :param action: 动作序列
        :param volatility: 波动性序列
        :return: 奖励序列
        """
        returns = np.diff(price) / price[:-1]
        risk_adjusted_returns = (returns - self.risk_free_rate) / volatility[:-1]
        rewards = returns * action[:-1] - self.transaction_cost * np.abs(np.diff(action))
        adjusted_rewards = rewards / volatility[:-1]
        
        return adjusted_rewards

# 测试奖励类的功能
def test():
    prices = np.array([100, 102, 101, 105, 107])
    actions = np.array([1, 0, -1, 1, 0])
    volatilities = np.array([0.1, 0.2, 0.15, 0.25, 0.3])
    
    reward_calculator = StockTradingReward()
    rewards = reward_calculator.calculate_reward(prices, actions, volatilities)
    print("奖励序列:", rewards)


import numpy as np

class StockTradingEnv:
    """
    股票交易环境类，用于模拟股票交易过程

    该类包含了股票交易的主要功能，包括重置环境、执行交易动作、计算奖励等。
    """

    def __init__(self, prices: np.ndarray, initial_balance: float = 100000, volatility_window: int = 30):
        """
        初始化股票交易环境的参数和状态
        :param prices: 股票价格序列
        :param initial_balance: 初始账户余额
        :param volatility_window: 计算波动性的窗口大小
        """
        self.prices = prices
        self.initial_balance = initial_balance
        self.volatility_window = volatility_window
        self.n_prices = len(prices)
        self.reset()

    def reset(self):
        """
        重置环境状态
        :return: 返回重置后的状态
        """
        self.balance = self.initial_balance
        self.shares_held = 0
        self.current_step = self.volatility_window  # 从波动性窗口后开始
        self.done = False
        return self._get_observation()

    def step(self, action: int):
        """
        执行动作并更新环境状态
        :param action: 智能体选择的动作（0: 持有，1: 买入，2: 卖出）
        :return: 返回新的状态、奖励、是否结束标志和额外信息
        """
        assert action in [0, 1, 2], "Action must be 0, 1, or 2"

        prev_value = self._calculate_value()
        self._take_action(action)
        self.current_step += 1
        current_value = self._calculate_value()

        reward = self._calculate_reward(prev_value, current_value)
        self.done = self.current_step >= self.n_prices - 1

        return self._get_observation(), reward, self.done, {}

    def _get_observation(self):
        """
        获取当前的环境状态
        :return: 当前状态，包括当前价格、账户余额和持有股票数量
        """
        return np.array([self.prices[self.current_step], self.balance, self.shares_held])

    def _take_action(self, action: int):
        """
        根据动作执行交易
        :param action: 智能体选择的动作（0: 持有，1: 买入，2: 卖出）
        """
        current_price = self.prices[self.current_step]

        if action == 1:  # 买入
            shares_bought = self.balance // current_price
            self.balance -= shares_bought * current_price
            self.shares_held += shares_bought
        elif action == 2:  # 卖出
            self.balance += self.shares_held * current_price
            self.shares_held = 0

    def _calculate_value(self):
        """
        计算当前的总资产价值
        :return: 总资产价值
        """
        current_price = self.prices[self.current_step]
        return self.balance + self.shares_held * current_price

    def _calculate_reward(self, prev_value, current_value):
        """
        计算波动性调整后的奖励
        :param prev_value: 前一步的总资产价值
        :param current_value: 当前的总资产价值
        :return: 波动性调整后的奖励
        """
        returns = np.diff(self.prices[max(0, self.current_step - self.volatility_window):self.current_step + 1])
        volatility = np.std(returns)
        if volatility == 0:
            return 0
        reward = (current_value - prev_value) / volatility
        return reward

    def render(self):
        """
        渲染环境（可以在此处添加具体的渲染代码）
        """
        print(f"Step: {self.current_step}")
        print(f"Price: {self.prices[self.current_step]}")
        print(f"Balance: {self.balance}")
        print(f"Shares held: {self.shares_held}")
        print(f"Total value: {self._calculate_value()}")

    def close(self):
        """
        关闭环境（可以在此处添加具体的清理代码）
        """
        pass

# 测试 StockTradingEnv 类的功能
if __name__ == "__main__":
    prices = np.array([100, 102, 101, 105, 107, 110, 108, 106, 109, 111])
    env = StockTradingEnv(prices)
    
    state = env.reset()
    print("初始状态:", state)
    
    for _ in range(len(prices) - env.volatility_window - 1):
        action = np.random.choice([0, 1, 2])
        state, reward, done, _ = env.step(action)
        env.render()
        print(f"奖励: {reward}, 结束标志: {done}")
        
        if done:
            print("环境已终止，重新开始")
            state = env.reset()