import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
class Portfolio:
    def __init__(self, initial_cash, universe, benchmark=None):
        self.initial_cash = initial_cash
        self.cash = initial_cash
        self.positions = {}  # {ticker: shares}
        self.position_history = pd.DataFrame(columns=["Date", "Ticker", "Percentage"])
        self.orders = {}  # 当日订单
        self.universe = universe
        self.bench_mark = benchmark
        if benchmark is not None:
            self.history = pd.DataFrame(columns=["Date", "Portfolio", "Benchmark"])
        else:
            self.history = pd.DataFrame(columns=["Date", "Portfolio"])

    def reset(self):
        self.cash = self.initial_cash
        self.positions = {}  # {ticker: shares}
        self.position_history = pd.DataFrame(columns=["Date", "Ticker", "Percentage"])
        self.orders = {}  # 当日订单
        if self.bench_mark is not None:
            self.history = pd.DataFrame(columns=["Date", "Portfolio", "Benchmark"])
        else:
            self.history = pd.DataFrame(columns=["Date", "Portfolio"])

    def update_portfolio(self, time, prices, benchmark_price: Optional[float] = None):
        total_value = self.cash
        hold_percentage = dict()
        for ticker, shares in self.positions.items():
            total_value += shares * prices.get(ticker, 0)
        for ticker, shares in self.positions.items():
            hold_percentage[ticker] = (shares * prices.get(ticker, 0)) / total_value

        # self.nav_history.append(total_value)
        # if benchmark_price is not None:
        #    self.bench_history.append(benchmark_price)
        if benchmark_price is not None:
            self.history.loc[len(self.history)] = {"Date": time, "Portfolio": total_value, "Benchmark": benchmark_price}
        else:
            self.history.loc[len(self.history)] = {"Date": time, "Portfolio": total_value}

        for ticker, percent in hold_percentage.items():
            self.position_history.loc[len(self.position_history)] = {"Date": time, "Ticker": ticker,
                                                                     "Percentage": percent}

    def order_to_percentage(self, target_percentages, prices):
        """
        指定标的占组合的比例。
        target_percentages: dict {ticker: target_percentage}
        prices: dict {ticker: price}
        """
        self.orders = target_percentages
        self.execute_orders(prices)

    def execute_orders(self, prices):
        # total_portfolio_value = self.cash + sum(
        #     [shares * (prices.get(ticker, 0) if not np.isnan(prices.get(ticker, 0)) else 0) for ticker, shares in self.positions.items()]
        # )
        # if np.isnan(total_portfolio_value):
        #     raise ValueError("Total portfolio value cannot be NaN")
        # for simplicity, we sell all the securities first
        if self.orders != {}:
            for ticker in self.positions.keys():
                price = prices.get(ticker, 0)
                price = price if not np.isnan(price) else 0 # if nan consider it delisted
                if price is None:
                    Warning(f"Price of {ticker} is {price}! Which should not be correct, skip")
                    continue  # 跳过没有价格的数据
                current_shares = self.positions.get(ticker, 0)
                current_value = current_shares * price
                self.cash += current_value

            self.positions = dict()
            total_portfolio_value = self.cash
            for ticker, target_percentage in self.orders.items():
                price = prices.get(ticker, None)
                if price is None or price == 0 or np.isnan(price):
                    Warning(f"Price of {ticker} is {price}! Which should not be correct, skip")
                    continue  # 跳过没有价格的数据

                target_value = total_portfolio_value * target_percentage
                current_shares = self.positions.get(ticker, 0)
                current_value = current_shares * price
                trade_value = target_value - current_value

                # 计算需要交易的股数，确保为整数
                trade_shares = trade_value // price

                # 更新持仓和现金
                self.positions[ticker] = current_shares + trade_shares
                self.cash -= trade_shares * price

                # 检查现金是否足够，避免出现负现金余额
                if self.cash < 0:
                    # 回退交易
                    self.cash += trade_shares * price
                    self.positions[ticker] = current_shares
                    print(f"现金不足，无法完成对{ticker}的交易。")
                    continue

            # 清空当日订单
            self.orders = {}

    def get_current_holdings(self):
        return self.positions.copy()

    def get_total_position_value(self, prices):
        return sum([shares * prices.get(ticker, 0) for ticker, shares in self.positions.items()])

    def plot_nav(self):
        plt.figure(figsize=(10, 6))
        # df = pd.DataFrame()
        # df["portfolio"] = np.array(self.nav_history) / self.nav_history[0]
        # if self.bench_mark is not None:
        #     df["benchmark"] = np.array(self.bench_history) / self.bench_history[0]
        df = self.history.set_index("Date").dropna()
        for col in df.columns:
            plt.plot(df.index, (df[col] / df[col].iloc[0]).apply(np.log), label=col)
        plt.title('Portfolio Net Asset Value')
        plt.xlabel('Time')
        plt.ylabel('Value')
        plt.legend()
        plt.show()
