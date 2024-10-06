import datetime
from typing import Union, Optional

import numpy as np
import pandas as pd
from enum import Enum
#import deprecation

from fmp import Income_statement_entry, Balance_statement_entry, Cashflow_statement_entry, Financial_ratio_entry, \
    Price_entry, API_KEY, fmp_get_etf_holdings
from main import stock_database
import matplotlib.pyplot as plt


class DataHandler:
    def __init__(self, universe: list[str], data_start_date: str, data_end_date: str
                 , data_requirements: [Union[Income_statement_entry
            , Balance_statement_entry, Cashflow_statement_entry
            , Financial_ratio_entry, Price_entry]]):
        self.universe = universe
        self.start_date = data_start_date
        self.end_date = data_end_date
        self.data_requirements = data_requirements
        # self.data = self.loaddata(date_requirements)
        self.current_time_index = 0
        # 使用价格数据的日期索引作为时间
        self.data = None

    def loaddata(self, dynamic_universe: Optional[pd.DataFrame] = None):

        # 这里调用外部API获取数据，返回一个dict=None
        db = stock_database(network_apikey=API_KEY)
        price_entries = list(filter(lambda x: isinstance(x, Price_entry), self.data_requirements))
        fund_entries = list(filter(lambda x: isinstance(x, (Balance_statement_entry
                                                            , Cashflow_statement_entry
                                                            , Income_statement_entry
                                                            , Financial_ratio_entry)), self.data_requirements))
        if price_entries != []:
            df_price = db.get_stock_price(self.universe
                                          , fields=price_entries
                                          , end_date=self.end_date
                                          , start_date=self.start_date)
        else:
            df_price = pd.DataFrame()
        if fund_entries != []:
            dict_fund = db.get_fundamentals(self.universe
                                            , fields=fund_entries
                                            , end_date=self.end_date
                                            , start_date=self.start_date)
        else:
            dict_fund = dict()
        # 为了示例，假设data_dict已经被加载并符合要求
        data_dict = {
            'price': df_price,  # 价格数据
            'fundamental': dict_fund,  # 财务数据
            'macro': pd.DataFrame(),  # 宏观数据
            'dynamic_universe': pd.DataFrame() if dynamic_universe is None else dynamic_universe
        }
        # 数据预处理，处理NaN值，只进行前向填充
        # for key in data_dict:
        #     data_dict[key] = data_dict[key].ffill()

        self.data = data_dict
        self.time_index = self.data['price'].index
        self.current_date = self.time_index[self.current_time_index]

    def reset(self):
        self.current_time_index = 0
        self.current_date = self.time_index[self.current_time_index]

    def get_dynamic_universe(self):
        df = self.data["dynamic_universe"]
        df = df[df.index < self.current_date]
        return df["holdings"].values.tolist()[0]

    def get_price(self,metric:Price_entry, prev=1, allow_today_data=False):
        df = self.data["price"]

        # 确保不获取未来数据，只获取当前日期之前的数据
        if allow_today_data == False:
            df = df[df.index < self.current_date]  # latest would be yesterday's data
        else:
            df = df[df.index <= self.current_date]  # the last one would be today's data, for backtester ONLY

        # 检查数据长度是否足够
        if len(df) < prev:
            return None  # 数据不足，返回None

        # 获取前N个数据
        result = df.iloc[-prev:]

        # 选择指标列，假设列是MultiIndex (ticker, metric)
        if metric.name in result.columns.get_level_values(1):
            data = result.xs(metric.name, level=1, axis=1)
            return data
        else:
            return None  # 未找到对应的指标

    def get_fundamentals(self, metric: Union[Income_statement_entry, Balance_statement_entry, Cashflow_statement_entry
    , Financial_ratio_entry], prev=1):
        dict_fund = self.data["fundamental"]
        dfs = []
        for ticker, df in dict_fund.items():
            if len(df) == 0:
                continue
            df = df[df.index < self.current_date]
            if len(df) < prev:
                continue
            dfs.append(df.iloc[-prev:].reset_index(drop=True))

        if dfs == []:
            return None

        result = pd.concat(dfs,axis=1)

        if metric.name in result.columns.get_level_values(1):
            data = result.xs(metric.name, level=1, axis=1)
            return data
        else:
            return None  # 未找到对应的指标

    def update_time(self):
        self.current_time_index += 1
        if self.current_time_index < len(self.time_index):
            self.current_date = self.time_index[self.current_time_index]
        else:
            self.current_date = None  # 数据结束

    def is_data_end(self):
        return self.current_time_index >= len(self.time_index)

    def get_current_time(self):
        return self.current_date


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
        self.universe = universe
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
        total_portfolio_value = self.cash + sum(
            [shares * prices.get(ticker, 0) for ticker, shares in self.positions.items()]
        )
        # for simplicity, we sell all the securities first
        if self.orders != {}:
            for ticker in self.positions.keys():
                price = prices.get(ticker, None)
                if price is None or price == 0:
                    continue  # 跳过没有价格的数据
                current_shares = self.positions.get(ticker, 0)
                current_value = current_shares * price
                self.cash += current_value

            self.positions = dict()

            for ticker, target_percentage in self.orders.items():
                price = prices.get(ticker, None)
                if price is None or price == 0 or price == np.nan:
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


class Backtester:
    def __init__(self, universe_i: Union[list[str], pd.DataFrame], start_date: str, end_date: str, initial_cash: int,
                 data_requirements: list[Union[Income_statement_entry
                 , Balance_statement_entry, Cashflow_statement_entry
                 , Financial_ratio_entry, Price_entry]], strategy, benchmark: Optional[str] = None):
        if type(universe_i) == pd.DataFrame:
            universe = list(set([j for i in universe_i["holdings"].values.tolist() for j in i]))
            using_dynamic_universe = True
        else:
            universe = universe_i
            using_dynamic_universe = False

        if benchmark is not None:
            self.data_handler = DataHandler(list(set(universe + [benchmark]))
                                            , start_date, end_date, data_requirements)
            self.universe = universe
        else:
            self.data_handler = DataHandler(universe
                                            , start_date, end_date, data_requirements)
            self.universe = universe
        self.data_handler.loaddata(universe_i if using_dynamic_universe else None)
        self.portfolio = Portfolio(initial_cash, universe, benchmark)
        self.strategy = strategy(self.data_handler, universe)  # Strategy(self.data_handler, universe)
        self.benchmark = benchmark

    def run_backtest(self):
        while not self.data_handler.is_data_end():
            current_time = self.data_handler.get_current_time()
            print(f"回测日期: {current_time}")

            # 获取策略信号
            signals = self.strategy.generate_signals()

            open_prices_df = self.data_handler.get_price(Price_entry.open, prev=1, allow_today_data=True)
            close_prices_df = self.data_handler.get_price(Price_entry.close, prev=1, allow_today_data=True)
            adjclose_prices_df = self.data_handler.get_price(Price_entry.adjClose, prev=1, allow_today_data=True)
            adjopen_prices_df = (open_prices_df / close_prices_df) * adjclose_prices_df
            adjopen_prices = adjopen_prices_df[self.universe].iloc[-1].to_dict()
            if self.benchmark is not None:
                bench_adjopen_prices = adjopen_prices_df[self.benchmark].iloc[-1]
            else:
                bench_adjopen_prices = None

            if adjopen_prices_df is None:
                # 无法获取开盘价，跳过当前日期
                self.data_handler.update_time()
                continue

            if signals is not None:
                # 获取当日开盘价
                # 下订单
                self.portfolio.order_to_percentage(signals, adjopen_prices)
                # 更新投资组合
                self.portfolio.update_portfolio(self.data_handler.get_current_time(), adjopen_prices,
                                                bench_adjopen_prices)

            # 时间推进
            self.data_handler.update_time()

        # 绘制净值曲线
        self.portfolio.plot_nav()


def rebalance(rebalance_period):
    def decorator(func):
        temp = rebalance_period
        i = 0
        last_signal = None

        def wrapper(*args, **kwargs):
            nonlocal temp, i, last_signal
            if i % temp == 0:
                signal = func(*args, **kwargs)
                last_signal = signal
            else:
                signal = last_signal
            i += 1
            return signal

        return wrapper

    return decorator


class Strategy:

    def __init__(self, data_handler, universe):
        self.data_handler = data_handler
        self.universe = universe

    def generate_signals(self):
        pass


class MOM_STRATEGY(Strategy):
    def __init__(self, data_handler, universe):

        super().__init__(data_handler, universe)

    @rebalance(20)
    def generate_signals(self):
        def z_score(df):
            return (df - df.mean()) / df.std()

        def calc_mom_score(series_A):
            # Step 1: Create Series B based on the rules given for Series A
            series_B = series_A  # series_A.clip(lower=-3, upper=3)

            # Step 2: Apply the transformation rules for Series B
            def update_value(Z):
                if Z > 0:
                    return 1 + Z
                elif Z < 0:
                    return 1 / (1 - Z)
                else:
                    return Z  # If Z == 0, it remains 0

            series_B_transformed = series_B.apply(update_value)

            return series_B_transformed
        def sharpe_mom(close_prices, d_ret, abs_filter= False):
            std_d_ret_6 = np.sqrt(252) * (d_ret.iloc[-120:-1].std())
            std_d_ret_12 = np.sqrt(252) * (d_ret.std())
            raw_mom_12 = (close_prices.iloc[-1] / close_prices.iloc[0]) / std_d_ret_12
            raw_mom_6 = (close_prices.iloc[-1] / close_prices.iloc[-120]) / std_d_ret_6
            mom4 = z_score(raw_mom_12)
            mom3 = z_score(raw_mom_6)
            #mom4 = raw_mom_12.rank(ascending=True)
            #mom3 = raw_mom_6.rank(ascending=True)
            comp_mom = (0.5 * mom3 + 0.5 * mom4)
            comp_mom_z = z_score(comp_mom)

            if abs_filter:
                abs_mom_idx_gt_0 = ((0.5 * raw_mom_12 + 0.5 * np.sqrt(2) * raw_mom_6) > 0).index
                return comp_mom_z.loc[abs_mom_idx_gt_0]
            else:
                return comp_mom_z

        close_prices: pd.DataFrame = self.data_handler.get_price(Price_entry.close, prev=250)
        if close_prices is None:
            # 数据不足，无法生成信号
            return None
        close_prices.dropna(axis=1, inplace=True)
        close_prices = close_prices.rolling(5).mean()
        close_prices.dropna(inplace=True)
        d_ret = close_prices.diff().dropna()
        comp_mom_z = sharpe_mom(close_prices,d_ret,abs_filter=False)
        # comp_mom = mom12
        uni = self.universe
        mom_score = calc_mom_score(comp_mom_z[uni])
        ranked_securities = list(mom_score.rank(ascending=True).sort_values(ascending=False).index)

        def geometric_weights(N, M):
            """
            Calculate geometric weights for M selected stocks from a total of N stocks.
            Parameters:
            N (int): Total number of stocks.
            M (int): Number of stocks to select for weighting.
            Returns:
            list: Geometric weights for the selected M stocks.
            """
            # Ensure M <= N
            if M > N:
                # raise ValueError("M cannot be greater than N.")
                M = N
            # Define the common ratio for geometric progression
            # Usually, a ratio less than 1 is used, e.g., 0.5 for halving weights
            ratio = 0.5
            # Generate the geometric progression weights before normalization
            weights = np.array([ratio ** i for i in range(M)])
            # Normalize the weights so they sum to 1
            normalized_weights = weights / np.sum(weights)
            return normalized_weights

        def equal_weights(N, M):
            weights = np.array([1 for i in range(N)])
            # Normalize the weights so they sum to 1
            normalized_weights = weights / np.sum(weights)
            return normalized_weights

        M = 3
        # w = equal_weights(M, M)
        # w = geometric_weights(len(ranked_securities), M)
        w0 = mom_score.rank(ascending=True).sort_values(ascending=False)[0:M]
        w = w0/w0.sum()
        signals = {}
        for s, w1 in zip(ranked_securities[0:M], w):
            signals[s] = w1

        return signals

class TOP_N_REPLICATE(Strategy):
    def __init__(self, data_handler, universe):

        super().__init__(data_handler, universe)

    @rebalance(60)
    def generate_signals(self):
        def equal_weights(N, M):
            weights = np.array([1 for i in range(N)])
            # Normalize the weights so they sum to 1
            normalized_weights = weights / np.sum(weights)
            return normalized_weights

        #close_prices: pd.DataFrame = self.data_handler.get_price(Price_entry.close, prev=250)
        uni = self.data_handler.get_dynamic_universe()
        w = equal_weights(len(uni),None)
        signals = {}
        for s, w1 in zip(uni, w):
            signals[s] = w1

        return signals

class MOM_STRATEGY_DYN(Strategy):
    def __init__(self, data_handler, universe):

        super().__init__(data_handler, universe)

    @rebalance(20)
    def generate_signals(self):
        def z_score(df):
            return (df - df.mean()) / df.std()

        def calc_mom_score(series_A):
            # Step 1: Create Series B based on the rules given for Series A
            series_B = series_A.clip(lower=-3, upper=3)

            # Step 2: Apply the transformation rules for Series B
            def update_value(Z):
                if Z > 0:
                    return 1 + Z
                elif Z < 0:
                    return 1 / (1 - Z)
                else:
                    return Z  # If Z == 0, it remains 0

            series_B_transformed = series_B.apply(update_value)

            return series_B_transformed

        def sharpe_mom(close_prices,d_ret):
            std_d_ret_6 = np.sqrt(252) * (d_ret.iloc[-120:-1].std())
            std_d_ret_12 = np.sqrt(252) * (d_ret.std())
            raw_mom_12 = (close_prices.iloc[-1] / close_prices.iloc[0]) / std_d_ret_12
            raw_mom_6 = (close_prices.iloc[-1] / close_prices.iloc[-120]) / std_d_ret_6
            mom4 = z_score(raw_mom_12)
            mom3 = z_score(raw_mom_6)
            comp_mom = (0.5 * mom3 + 0.5 * mom4)
            comp_mom_z = z_score(comp_mom)
            return comp_mom_z

        def ID_mom(close_prices,d_ret):
            raw_mom_12 = (close_prices.iloc[-1] / close_prices.iloc[0]) * ((d_ret <0).sum() - (d_ret > 0).sum())
            raw_mom_6 = (close_prices.iloc[-1] / close_prices.iloc[-120]) * ((d_ret.iloc[-120:-1] <0).sum() - (d_ret.iloc[-120:-1] > 0).sum())
            mom4 = z_score(raw_mom_12)
            mom3 = z_score(raw_mom_6)
            comp_mom = (0.5 * mom3 + 0.5 * mom4)
            comp_mom_z = z_score(comp_mom)
            return comp_mom_z

        close_prices: pd.DataFrame = self.data_handler.get_price(Price_entry.close, prev=250)
        if close_prices is None:
            # 数据不足，无法生成信号
            return None
        close_prices.dropna(axis=1, inplace=True)
        close_prices = close_prices.rolling(5).mean()
        close_prices.dropna(inplace=True)
        # close_prices = close_prices.apply(np.log)
        d_ret = close_prices.diff().dropna()
        comp_mom_z = sharpe_mom(close_prices, d_ret)
        # comp_mom = mom12
        # uni = self.universe
        uni = self.data_handler.get_dynamic_universe()
        intercect_uni = list(set(uni).intersection(set(comp_mom_z.index)))
        mom_score = calc_mom_score(comp_mom_z[intercect_uni])
        ranked_securities = list(mom_score.rank(ascending=True).sort_values(ascending=False).index)

        def geometric_weights(N, M):
            """
            Calculate geometric weights for M selected stocks from a total of N stocks.
            Parameters:
            N (int): Total number of stocks.
            M (int): Number of stocks to select for weighting.
            Returns:
            list: Geometric weights for the selected M stocks.
            """
            # Ensure M <= N
            if M > N:
                # raise ValueError("M cannot be greater than N.")
                M = N
            # Define the common ratio for geometric progression
            # Usually, a ratio less than 1 is used, e.g., 0.5 for halving weights
            ratio = 0.5
            # Generate the geometric progression weights before normalization
            weights = np.array([ratio ** i for i in range(M)])
            # Normalize the weights so they sum to 1
            normalized_weights = weights / np.sum(weights)
            return normalized_weights

        def equal_weights(N, M):
            weights = np.array([1 for i in range(N)])
            # Normalize the weights so they sum to 1
            normalized_weights = weights / np.sum(weights)
            return normalized_weights

        M = len(ranked_securities) // 5
        #w = equal_weights(M, M)
        w = geometric_weights(len(ranked_securities), M)
        signals = {}
        for s, w1 in zip(ranked_securities[0:M], w):
            signals[s] = w1

        return signals

class Simple_Fundamental_Stratery(Strategy):
    def __init__(self, data_handler:DataHandler, universe):
        super().__init__(data_handler, universe)

    @rebalance(60)
    def generate_signals(self):
        def z_score(df:pd.Series):
            return (df - df.mean()) / df.std()
        def equal_weights(N, M):
            weights = np.array([1 for i in range(N)])
            # Normalize the weights so they sum to 1
            normalized_weights = weights / np.sum(weights)
            return normalized_weights
        ROE = self.data_handler.get_fundamentals(metric=Financial_ratio_entry.ROE,prev=1)
        GP = self.data_handler.get_fundamentals(metric=Income_statement_entry.gross_profit,prev=1)
        Asset = self.data_handler.get_fundamentals(metric=Balance_statement_entry.total_assets,prev=1)

        if GP is None or Asset is None or ROE is None:
            return None
        GP.dropna(axis=1,how="all", inplace=True)
        Asset.dropna(axis=1,how="all", inplace=True)
        ROE.dropna(axis=0,how="all",inplace=True)
        uni = self.data_handler.get_dynamic_universe()
        uni = list(set(ROE.iloc[0].index).intersection(set(uni)))
        ROE_z = z_score(ROE[uni].iloc[0]).clip(lower=-3, upper=3)
        ranked_securities = list(ROE_z.rank(ascending=True).sort_values(ascending=False).index)
        N = len(ranked_securities) // 5
        w = equal_weights(N,N)
        signals = {}
        for s, w1 in zip(ranked_securities[0:N], w):
            signals[s] = w1

        return signals


class PM_ratio(Strategy):
    def __init__(self, data_handler:DataHandler, universe):
        super().__init__(data_handler, universe)

    @rebalance(60)
    def generate_signals(self):
        def z_score(df:pd.Series):
            return (df - df.mean()) / df.std()
        def equal_weights(N, M):
            weights = np.array([1 for i in range(N)])
            # Normalize the weights so they sum to 1
            normalized_weights = weights / np.sum(weights)
            return normalized_weights
        def calc_score(series_A):
            # Step 1: Create Series B based on the rules given for Series A
            series_B = series_A.clip(lower=-3, upper=3)

            # Step 2: Apply the transformation rules for Series B
            def update_value(Z):
                if Z > 0:
                    return 1 + Z
                elif Z < 0:
                    return 1 / (1 - Z)
                else:
                    return Z  # If Z == 0, it remains 0

            series_B_transformed = series_B.apply(update_value)

            return series_B_transformed
        def z_score(df):
            return (df - df.mean()) / df.std()
        LEV = self.data_handler.get_fundamentals(metric=Financial_ratio_entry.equity_multiplier,prev=1)
        D = self.data_handler.get_fundamentals(metric=Balance_statement_entry.total_debt,prev=1)
        I = self.data_handler.get_fundamentals(metric=Income_statement_entry.interest_expense,prev=1)
        GP = self.data_handler.get_fundamentals(metric=Income_statement_entry.gross_profit,prev=1)
        Asset = self.data_handler.get_fundamentals(metric=Balance_statement_entry.total_assets,prev=1)
        if LEV is None or D is None or I is None:
            return None
        I.dropna(axis=1,how="all", inplace=True)
        D.dropna(axis=1,how="all", inplace=True)
        LEV.dropna(axis=1, how="all", inplace=True)
        GP.dropna(axis=1,how="all", inplace=True)
        Asset.dropna(axis=1,how="all", inplace=True)
        uni = self.data_handler.get_dynamic_universe()
        uni = list(set((LEV).iloc[0].index).intersection(set((I).iloc[0].index)).intersection(set((D).iloc[0].index)).intersection(set(uni)))
        LEV = LEV[uni].iloc[0]
        Interest_rate_inv = (D/I)[uni].iloc[0]
        GPOA = GP[uni].iloc[0] / Asset[uni].iloc[0]
        comp_factor = calc_score(GPOA.rank(ascending=True) + (LEV * Interest_rate_inv).rank(ascending=True))
        ranked_securities = list(comp_factor.rank(ascending=True).sort_values(ascending=False).index)
        N = len(ranked_securities) // 5
        #w = equal_weights(N,N)
        w = comp_factor[ranked_securities[0:N]] / comp_factor[ranked_securities[0:N]].sum()
        signals = {}
        for s, w1 in zip(ranked_securities[0:N], w):
            signals[s] = w1

        return signals
if __name__ == "__main__":
    # 使用示例
    universe = ['XLK', 'XLV', 'XLE']

    start_date = '2013-11-01'
    end_date = '2024-09-13'
    initial_cash = 10000
    # backtester = Backtester(universe
    #                         , start_date, end_date, initial_cash
    #                         , [Price_entry.adjClose, Price_entry.open, Price_entry.close]
    #                         , strategy= MOM_STRATEGY
    #                         , benchmark="QQQ")
    backtester = Backtester(fmp_get_etf_holdings("DGRW", apikey=API_KEY
                                                 , start_date=start_date, end_date=end_date, highest_N=30)
                            , start_date, end_date, initial_cash
                            , [Price_entry.adjClose, Price_entry.open, Price_entry.close
                               ,Income_statement_entry.gross_profit
                               ,Balance_statement_entry.total_assets
                               ,Financial_ratio_entry.equity_multiplier
                               ,Income_statement_entry.interest_expense,Balance_statement_entry.total_debt]
                            , strategy=PM_ratio
                            , benchmark="DGRW")
    backtester.run_backtest()
    backtester.portfolio.position_history.set_index("Date").to_csv("DGRW.GPOE_mult.holdings.csv")
    backtester.portfolio.history.set_index("Date").to_csv("DGRW.GPOE_mult.hold_all.csv")
