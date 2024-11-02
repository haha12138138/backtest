import numpy as np
import pandas as pd

from Backtester import Backtester
from FMP.fmp import Income_statement_entry, Balance_statement_entry, Datapoint_period, Financial_ratio_entry
from Strategy import Strategy, rebalance, Strategy_dependence
from Universe import Stock_Universe


class GPOE_Value(Strategy):
    inherent_dependencies = Strategy.inherent_dependencies + [
        Strategy_dependence(Income_statement_entry.gross_profit, Datapoint_period.quarter, 4),
        Strategy_dependence(Balance_statement_entry.total_equity, Datapoint_period.quarter, 4),
        Strategy_dependence(Financial_ratio_entry.PB, Datapoint_period.quarter, num_of_data_needed=1),
        Strategy_dependence(Financial_ratio_entry.PE, Datapoint_period.quarter, num_of_data_needed=1),
        Strategy_dependence(Financial_ratio_entry.PEG, Datapoint_period.quarter, num_of_data_needed=1)
    ]

    def __init__(self):
        super().__init__()

    @rebalance(60)
    def generate_signals(self):
        def z_score(df: pd.Series):
            return (df - df.mean()) / df.std()

        def msci_mom_normalize(series_A):
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
        def equal_weights(N, M):
            weights = np.array([1 for i in range(N)])
            # Normalize the weights so they sum to 1
            normalized_weights = weights / np.sum(weights)
            return normalized_weights

        # ROE = self.data_handler.get_fundamentals(metric=Financial_ratio_entry.ROE,prev=1)
        GP = self.data_handler.get_fundamentals(metric=Income_statement_entry.gross_profit, prev=4)
        equity = self.data_handler.get_fundamentals(metric=Balance_statement_entry.total_equity, prev=4)
        PB = self.data_handler.get_fundamentals(metric=Financial_ratio_entry.PB, prev=1)
        PE = self.data_handler.get_fundamentals(metric=Financial_ratio_entry.PE, prev=1)
        PEG = self.data_handler.get_fundamentals(metric=Financial_ratio_entry.PEG, prev=1)
        for e in (GP, equity, PB, PE, PEG):
            if e is None:
                return None
        GP.dropna(axis=1, how="any", inplace=True)
        equity.dropna(axis=1, how="any", inplace=True)
        PB.dropna(axis=1, how="any", inplace=True)
        PE.dropna(axis=1, how="any", inplace=True)
        PEG.dropna(axis=1, how="any", inplace=True)
        GP_TTM = GP.sum().to_frame().T
        equity_TTM = equity.mean().to_frame().T

        uni = self.data_handler.get_dynamic_universe()
        uni = list(set(uni)
                   .intersection(set(equity_TTM.iloc[0].index))
                   .intersection(set(GP_TTM.iloc[0].index))
                   .intersection(set((PB.iloc[0] > 1).index))
                   # .intersection(set((PE.iloc[0]).index))
                   .intersection(set((PEG.iloc[0]).index))
                   )

        if len(uni) == 0:
            return None
        GPOE_z = z_score((GP_TTM / equity_TTM)[uni].iloc[0]).clip(lower=-3, upper=3)

        ranked_securities = list(GPOE_z.rank(ascending=True).sort_values(ascending=False).index)
        N = int(len(ranked_securities) / 5 + 1)
        selected = ranked_securities[0:N]
        # Value_z = msci_mom_normalize(z_score(1/z_score(PB[selected.index].iloc[0]) + 1/z_score(PE[selected.index].iloc[0])))
        value_z = msci_mom_normalize(z_score(
            -1 * z_score(PEG[selected].iloc[0]) +
            -1 * z_score(PB[selected].iloc[0])
            # -1 * z_score(PE[selected].iloc[0])
        ))

        signals = {}
        for s, w1 in (value_z / value_z.sum()).items():
            signals[s] = w1

        return signals


class GPOE_LYR(Strategy):
    inherent_dependencies = Strategy.inherent_dependencies + [
        Strategy_dependence(Income_statement_entry.gross_profit, Datapoint_period.annual, 1),
        Strategy_dependence(Balance_statement_entry.total_equity, Datapoint_period.annual, 1),
        # Strategy_dependence(Financial_ratio_entry.PB,Datapoint_period.quarter,num_of_data_needed=1),
        # Strategy_dependence(Financial_ratio_entry.PEG,Datapoint_period.quarter,num_of_data_needed=1)
    ]

    def __init__(self):
        super().__init__()

    @rebalance(60)
    def generate_signals(self):
        def z_score(df: pd.Series):
            return (df - df.mean()) / df.std()

        def msci_mom_normalize(series_A):
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

        def equal_weights(N, M):
            weights = np.array([1 for i in range(N)])
            # Normalize the weights so they sum to 1
            normalized_weights = weights / np.sum(weights)
            return normalized_weights

        # ROE = self.data_handler.get_fundamentals(metric=Financial_ratio_entry.ROE,prev=1)
        GP = self.data_handler.get_fundamentals(metric=Income_statement_entry.gross_profit, prev=1)
        equity = self.data_handler.get_fundamentals(metric=Balance_statement_entry.total_equity, prev=1)
        # PB = self.data_handler.get_fundamentals(metric=Financial_ratio_entry.PB, prev=1)
        # PEG = self.data_handler.get_fundamentals(metric=Financial_ratio_entry.PEG,prev=1)
        for e in (GP, equity):
            if e is None:
                return None
        GP.dropna(axis=1, how="any", inplace=True)
        equity.dropna(axis=1, how="any", inplace=True)
        # PB.dropna(axis=1, how="all", inplace=True)
        # PEG.dropna(axis=1, how="all", inplace=True)
        GP_TTM = GP
        equity_TTM = equity

        uni = self.data_handler.get_dynamic_universe()
        uni = list(set(uni)
                   .intersection(set(equity_TTM.iloc[0].index))
                   .intersection(set(GP_TTM.iloc[0].index))
                   # .intersection(set((PB.iloc[0] >1).index))
                   # .intersection(set((PEG.iloc[0] >0).index))
                   )

        if len(uni) == 0:
            return None
        GPOE_z = z_score((GP_TTM / equity_TTM)[uni].iloc[0]).clip(lower=-3, upper=3)

        ranked_securities = list(GPOE_z.rank(ascending=True).sort_values(ascending=False).index)
        N = int(len(ranked_securities) / 5 + 1)
        selected = ranked_securities[0:N]
        # Value_z = msci_mom_normalize(z_score(1/z_score(PB[selected.index].iloc[0]) + 1/z_score(PE[selected.index].iloc[0])))
        # value_z = msci_mom_normalize(z_score(1/(PEG[selected.index].iloc[0])))
        signals = {}
        for s, w1 in zip(ranked_securities[0:N], equal_weights(N, N)):
            signals[s] = w1

        return signals

if __name__ == "__main__":
    # 使用示例
    # universe = ['XLK', 'XLV', 'XLE']

    start_date = '2019-01-01'
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
                            , strategy=GPOE_Value)
    backtester.run_backtest()
    backtester.portfolio.position_history.set_index("Date").to_csv(
        r"../results/Fundamental_DGRW/GPOE_TTM.PB_PEG.holdings.csv")
    backtester.portfolio.history.set_index("Date").to_csv("../results/Fundamental_DGRW/GPOE_TTM.PB_PEG.csv")
