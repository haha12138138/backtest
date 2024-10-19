from FMP.fmp import Price_entry, Income_statement_entry, Balance_statement_entry, Financial_ratio_entry
from Universe import Stock_Universe
import pandas as pd
import numpy as np
from Strategy import Strategy, rebalance
from Backtester import Backtester


class Simple_Fundamental_Stratery(Strategy):
    def __init__(self, data_handler, universe):
        super().__init__(data_handler, universe)

    @rebalance(60)
    def generate_signals(self):
        def z_score(df: pd.Series):
            return (df - df.mean()) / df.std()

        def equal_weights(N, M):
            weights = np.array([1 for i in range(N)])
            # Normalize the weights so they sum to 1
            normalized_weights = weights / np.sum(weights)
            return normalized_weights

        # ROE = self.data_handler.get_fundamentals(metric=Financial_ratio_entry.ROE,prev=1)
        GP = self.data_handler.get_fundamentals(metric=Income_statement_entry.gross_profit, prev=1)
        equity = self.data_handler.get_fundamentals(metric=Balance_statement_entry.total_equity, prev=1)

        if GP is None or equity is None:
            return None
        GP.dropna(axis=1, how="all", inplace=True)
        equity.dropna(axis=1, how="all", inplace=True)
        # ROE.dropna(axis=0,how="all",inplace=True)
        uni = self.data_handler.get_dynamic_universe()
        uni = list(set(GP.iloc[0].index).intersection(set(equity.iloc[0].index)).intersection(set(uni)))
        GPOE_z = z_score((GP / equity)[uni].iloc[0]).clip(lower=-3, upper=3)
        ranked_securities = list(GPOE_z.rank(ascending=True).sort_values(ascending=False).index)
        N = len(ranked_securities) // 5
        w = equal_weights(N, N)
        signals = {}
        for s, w1 in zip(ranked_securities[0:N], w):
            signals[s] = w1

        return signals


class PM_ratio(Strategy):
    def __init__(self, data_handler, universe):
        super().__init__(data_handler, universe)

    @rebalance(60)
    def generate_signals(self):
        def z_score(df: pd.Series):
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

        LEV = self.data_handler.get_fundamentals(metric=Financial_ratio_entry.equity_multiplier, prev=1)
        D = self.data_handler.get_fundamentals(metric=Balance_statement_entry.total_debt, prev=1)
        I = self.data_handler.get_fundamentals(metric=Income_statement_entry.interest_expense, prev=1)
        GP = self.data_handler.get_fundamentals(metric=Income_statement_entry.gross_profit, prev=1)
        Asset = self.data_handler.get_fundamentals(metric=Balance_statement_entry.total_assets, prev=1)
        if LEV is None or D is None or I is None:
            return None
        I.dropna(axis=1, how="all", inplace=True)
        D.dropna(axis=1, how="all", inplace=True)
        LEV.dropna(axis=1, how="all", inplace=True)
        GP.dropna(axis=1, how="all", inplace=True)
        Asset.dropna(axis=1, how="all", inplace=True)
        uni = self.data_handler.get_dynamic_universe()
        uni = list(set((LEV).iloc[0].index).intersection(set((I).iloc[0].index)).intersection(
            set((D).iloc[0].index)).intersection(set(uni)))
        LEV = LEV[uni].iloc[0]
        Interest_rate_inv = (D / I)[uni].iloc[0]
        GPOA = GP[uni].iloc[0] / Asset[uni].iloc[0]
        comp_factor = calc_score(GPOA.rank(ascending=True) + (LEV * Interest_rate_inv).rank(ascending=True))
        ranked_securities = list(comp_factor.rank(ascending=True).sort_values(ascending=False).index)
        N = int(len(ranked_securities) / 5 + 1)
        # w = equal_weights(N,N)
        w = comp_factor[ranked_securities[0:N]] / comp_factor[ranked_securities[0:N]].sum()
        signals = {}
        for s, w1 in zip(ranked_securities[0:N], w):
            signals[s] = w1

        return signals


if __name__ == "__main__":
    # 使用示例
    # universe = ['XLK', 'XLV', 'XLE']

    start_date = '2013-11-01'
    end_date = '2024-09-13'
    initial_cash = 10000
    # backtester = Backtester(universe
    #                         , start_date, end_date, initial_cash
    #                         , [Price_entry.adjClose, Price_entry.open, Price_entry.close]
    #                         , strategy= MOM_STRATEGY
    #                         , benchmark="QQQ")
    uni = Stock_Universe()
    uni.add_holdings_to_group("DGRW", "index", topN=30)
    # uni.add_holdings_to_group("XMHQ", "index", topN= 10)
    uni.add_benchmark("SPY")
    backtester = Backtester(uni
                            , start_date, end_date, initial_cash
                            , [Price_entry.adjClose, Price_entry.open, Price_entry.close
                                , Income_statement_entry.gross_profit, Balance_statement_entry.total_equity]
                            , strategy=Simple_Fundamental_Stratery)
    backtester.run_backtest()
    backtester.portfolio.position_history.set_index("Date").to_csv("TEST.holdings.csv")
    backtester.portfolio.history.set_index("Date").to_csv("TEST.hold_all.csv")
